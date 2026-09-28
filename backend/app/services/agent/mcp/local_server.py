import asyncio
import json
import logging
import re
import sys
import threading
import time
from datetime import datetime
from typing import Any

from app.services.agent.input_merge_buffer import (
    SOURCE_JOB_RESULT,
    MergeBufferEntry,
    input_merge_buffer,
)

logger = logging.getLogger(__name__)


def debug_log(msg: str):
    print(msg, file=sys.stderr, flush=True)


TERMINAL_NOT_CONNECTED_MESSAGE = "终端未连接或未打开，命令没有发送。请先启动或连接终端后再试。"
BACKGROUND_JOB_STARTED_MARKER = "background_job_started"
JOB_RESULT_POLL_INITIAL_DELAY_SECONDS = 1.0
JOB_RESULT_POLL_INTERVAL_SECONDS = 2.0
JOB_RESULT_POLL_MAX_ERRORS = 30
COMMAND_DISPATCH_FAILED_MARKER = "command_dispatch_failed"
TERMINAL_UNAVAILABLE_RESULT_MARKER = "terminal_unavailable"

# Background job result coalescing: results arriving while a turn is running
# are buffered in the system-level input merge buffer and merged into ONE
# follow-up turn when the turn ends.


def buffer_background_job_result(item_id: str, entry: dict[str, Any]) -> None:
    _pending_size, evicted = input_merge_buffer.add(
        MergeBufferEntry(
            source_type=SOURCE_JOB_RESULT,
            item_id=str(item_id),
            scope_key=str(entry.get("conversation_key") or ""),
            sender_label="background_job",
            content=str(entry.get("command") or ""),
            reply_ticket_id=str(entry.get("reply_ticket_id") or ""),
            payload=entry,
        )
    )
    if evicted:
        evicted_entries = [e.payload for e in evicted if e.payload is not None]
        debug_log(
            f"[LocalMCPServer] job result buffer overflow: flushing {len(evicted_entries)} "
            f"evicted result(s) immediately, item={item_id}"
        )
        if evicted_entries:
            flush_background_job_results_for_entries(item_id, evicted_entries)


def _pop_buffered_job_results(
    item_id: str, conversation_key: str = ""
) -> list[dict[str, Any]]:
    return [
        entry.payload
        for entry in input_merge_buffer.pop(item_id, conversation_key)
        if entry.payload is not None
    ]


def _format_background_job_results_batch(entries: list[dict[str, Any]]) -> str:
    lines = [f"[Background job results batch: {len(entries)} jobs finished]"]
    for index, entry in enumerate(entries, start=1):
        result = entry["result"]
        if result.get("cancelled"):
            status = "cancelled (user-initiated cancel, NOT a failure; do not retry)"
        else:
            status = "succeeded" if result.get("success") else "failed"
        exit_code = result.get("exit_code")
        duration = result.get("duration_seconds")
        ticket = str(entry.get("reply_ticket_id") or "")[:8]
        lines.append(
            f"{index}. {status}: {entry['command']} "
            f"(exit {exit_code}, {duration}s, ticket={ticket})"
        )
        tail = str(result.get("output_tail") or "").strip()
        if tail:
            lines.append(f"   output: {tail[-300:]}")
        elif result.get("error"):
            lines.append(f"   error: {str(result.get('error'))[:300]}")
    lines.append(
        "If a plan is active, sync it via mcp_local_update_plan (mark completed "
        "steps, start next, or clear when all done). Then verify and report."
    )
    return "\n".join(lines)


def flush_background_job_results_for_entries(
    item_id: str, entries: list[dict[str, Any]]
) -> bool:
    if not entries:
        return False
    server = local_mcp_server
    flushed_any = False
    robot_entries = [entry for entry in entries if entry.get("robot_job_context")]
    other_entries = [entry for entry in entries if not entry.get("robot_job_context")]

    if robot_entries:
        first = robot_entries[0]
        robot_ticket = first.get("reply_ticket_id") or ""
        if len(robot_entries) == 1:
            message = server._format_background_job_robot_message(
                first["command"],
                first["result"],
                reply_ticket_id=robot_ticket,
            )
        else:
            message = _format_background_job_results_batch(robot_entries)
        message += server._plan_reminder(item_id, robot_ticket)
        delivered = server._deliver_background_job_to_robot(
            item_id=item_id,
            command=first["command"],
            result=first["result"],
            robot_job_context=first["robot_job_context"],
            pending_reply_id=first.get("pending_robot_reply_id") or "",
            reply_ticket_id=robot_ticket,
            message_override=message,
        )
        flushed_any = delivered or flushed_any
        if delivered and len(robot_entries) > 1:
            # The batch message covers every entry; clear the per-job
            # pending replies of the non-first entries so they do not
            # linger as orphaned queue items.
            for extra in robot_entries[1:]:
                if extra.get("pending_robot_reply_id"):
                    server._clear_background_job_robot_reply(
                        robot_job_context=extra.get("robot_job_context"),
                        pending_reply_id=extra.get("pending_robot_reply_id") or "",
                    )

    if other_entries:
        message = (
            other_entries[0]["feedback"]
            if len(other_entries) == 1
            else _format_background_job_results_batch(other_entries)
        )
        message += server._plan_reminder(
            item_id,
            str(other_entries[0].get("reply_ticket_id") or ""),
        )
        for entry in other_entries:
            session = entry.get("agent_session")
            if session is None:
                continue
            entry_ticket = entry.get("reply_ticket_id") or ""
            try:
                from app.services.agent.session import InputMessage, InputType

                session.process_input(
                    InputMessage(
                        input_type=InputType.TERMINAL,
                        content=message,
                        raw_content=message,
                        query="background job completed",
                        reply_ticket_id=entry_ticket,
                    )
                )
                flushed_any = True
            except Exception as exc:
                debug_log(
                    f"[LocalMCPServer] failed to deliver batched job feedback: item={item_id}, error={exc}"
                )
            break

    if not flushed_any:
        for entry in entries:
            if server._deliver_background_job_to_reply_ticket(
                reply_ticket_id=entry.get("reply_ticket_id") or "",
                command=entry["command"],
                result=entry["result"],
            ):
                flushed_any = True
                if entry.get("pending_robot_reply_id") and entry.get("robot_job_context"):
                    server._clear_background_job_robot_reply(
                        robot_job_context=entry["robot_job_context"],
                        pending_reply_id=entry["pending_robot_reply_id"],
                    )
    return flushed_any


def flush_job_results_for_turn_end(item_id: str, conversation_key: str = "") -> bool:
    entries = _pop_buffered_job_results(item_id, conversation_key)
    return flush_background_job_results_for_entries(item_id, entries)


def flush_all_job_results_for_item(item_id: str) -> int:
    """Drain every buffered job result of an item at turn end.

    Results are grouped per conversation so each conversation gets exactly
    one merged follow-up turn, but no conversation's results can be stranded
    by a turn that belongs to a different conversation.
    """
    drained = [
        entry.payload
        for entry in input_merge_buffer.pop_all_for_item(str(item_id))
        if entry.payload is not None
    ]
    if not drained:
        return 0
    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for entry in drained:
        key = str(entry.get("conversation_key") or "")
        by_conversation.setdefault(key, []).append(entry)
    flushed = 0
    for entries in by_conversation.values():
        if flush_background_job_results_for_entries(str(item_id), entries):
            flushed += len(entries)
    if flushed:
        debug_log(
            f"[LocalMCPServer] turn-end item flush: item={item_id} "
            f"drained={len(drained)} flushed={flushed} "
            f"conversations={len(by_conversation)}"
        )
    return flushed


def _conversation_turn_busy(agent_session, robot_job_context: dict | None) -> bool:
    if _session_turn_busy(agent_session):
        return True
    if not robot_job_context:
        return False
    try:
        from app.services.agent.integrations.hooks import (
            is_integration_conversation_processing,
        )

        return bool(
            is_integration_conversation_processing(
                robot_job_context.get("robot_id", ""),
                robot_job_context.get("conversation_key", ""),
            )
        )
    except Exception:
        return False


def _session_turn_busy(agent_session) -> bool:
    if agent_session is None:
        return False
    try:
        from app.services.agent.session import SessionState

        with agent_session.lock:
            if agent_session.state in {
                SessionState.RUNNING,
                SessionState.INTERRUPTING,
                SessionState.COLLECTING,
            }:
                return True
            return agent_session.input_queue.qsize() > 0
    except Exception:
        return False


def cancel_background_jobs_for_item(
    item_id: str,
    *,
    commands: set[str] | None = None,
) -> int:
    """Cancel running daemon jobs for an item (user-initiated task cancel).

    When ``commands`` is given, only jobs whose command matches are
    cancelled; otherwise every running job of the item is cancelled.
    Returns the number of jobs actually cancelled.
    """
    try:
        server = local_mcp_server
        _item, connection = server._get_item_daemon_context(str(item_id))
        result = connection.list_jobs_http(item_uuid=str(item_id))
    except Exception as exc:
        debug_log(
            f"[LocalMCPServer] cancel_background_jobs_for_item: list failed for item={item_id}: {exc}"
        )
        return 0
    if not isinstance(result, dict) or not result.get("success"):
        return 0

    cancelled = 0
    for job in result.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        command = str(job.get("command") or "").strip()
        if commands is not None and command not in commands:
            continue
        job_id = str(job.get("job_id") or job.get("id") or "").strip()
        if not job_id:
            continue
        try:
            cancel_result = connection.cancel_job_http(
                item_uuid=str(item_id),
                job_id=job_id,
            )
        except Exception as exc:
            debug_log(
                f"[LocalMCPServer] cancel_background_jobs_for_item: cancel failed job={job_id}: {exc}"
            )
            continue
        if cancel_result.get("success") or cancel_result.get("cancelled"):
            cancelled += 1
            debug_log(
                f"[LocalMCPServer] cancel_background_jobs_for_item: cancelled job={job_id} command={command!r}"
            )
    if cancelled:
        try:
            from app.services.agent.session import agent_session_manager

            session = agent_session_manager.get_session(str(item_id))
            if session:
                session.clear_terminal_job()
        except Exception:
            pass
    return cancelled


class LocalMCPServer:
    # The Codex-style plan scratchpad lives on the reply ticket (ticket.plan);
    # its identity, ownership and cleanup all reuse the ticket lifecycle, so
    # there is no separate plan store here anymore.

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        self.register_tool(
            name="update_plan",
            description=(
                "Track multi-step work. Any task with more than one step must be "
                "tracked with a plan so the user can see progress; only skip for "
                "single-step questions or chat. "
                "plan: list of {step, status} with status pending/in_progress/completed/cancelled; "
                "exactly one in_progress. Update it after finishing each step. "
                "任务结束（全部完成或中途放弃）时调用 update_plan(plan=[]) 直接清除计划，"
                "不需要把每步都标 completed。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "Current terminal item id."},
                    "explanation": {"type": "string", "description": "Optional note about what changed."},
                    "supersede_ticket_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "计划 id 列表：这些旧票据上的计划与本次是同一任务，"
                            "关闭它们（步骤标记取消并清除），把任务接管到当前票据。"
                        ),
                    },
                    "plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string"},
                                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                            },
                            "required": ["step", "status"],
                        },
                    },
                },
                "required": ["item_id", "plan"],
            },
            handler=self._update_plan,
            skip_memory=True,
        )
        self.register_tool(
            name="get_terminal_status",
            description="Read live terminal state (open/connected/usable).",
            input_schema={"type": "object", "properties": {}, "required": []},
            handler=self._get_terminal_status,
            skip_memory=True,
        )
        self.register_tool(
            name="execute_command",
            description=(
                "在主终端前台执行命令：仅用于需要持续交互的进程（服务器、REPL、控制台）"
                "或几秒内出结果的短命令。一次只发一条。"
                "安装/下载/构建/编译等耗时任务禁止用本工具，必须用 mcp_local_run_job；"
                "确实要在前台跑慢命令时，必须把 timeout_seconds 显式设够（如 600-1800），"
                "默认 20 秒必超时。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command."},
                    "timeout_seconds": {"type": "integer", "default": 20},
                    "auto_interrupt_on_timeout": {"type": "boolean", "default": False}
                },
                "required": ["command"]
            },
            handler=self._execute_command,
            skip_memory=True
        )
        self.register_tool(
            name="run_job",
            description="后台一次性命令，只需最终结果。用于下载、安装、构建、查询等不需要交互的操作。可并行多个不同任务。用户明确要求的安装/运行直接用本工具执行，无需先向用户确认。大型安装（如 JDK 等几百 MB 的包）必须显式设大 timeout_seconds（如 1800）。",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command."},
                    "timeout_seconds": {"type": "integer", "default": 1200},
                    "tail_lines": {"type": "integer", "default": 80}
                },
                "required": ["command"]
            },
            handler=self._run_job,
            skip_memory=True
        )
        self.register_tool(
            name="list_jobs",
            description="列出运行中的后台任务及输出尾部。查进度/状态时先调这个。",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._list_jobs,
            skip_memory=True
        )
        self.register_tool(
            name="cancel_job",
            description="按 job_id 取消后台任务。",
            input_schema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job id returned by list_jobs."}
                },
                "required": ["job_id"]
            },
            handler=self._cancel_job,
            skip_memory=True
        )

        self.register_tool(
            name="interrupt_command",
            description="发送 Ctrl+C 中断当前终端正在运行的命令",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            },
            handler=self._interrupt_command,
            skip_memory=True
        )

        self.register_tool(
            name="read_terminal_log",
            description="读取终端原始日志，查看完整错误或命令结果。",
            input_schema={
                "type": "object",
                "properties": {
                    "lines": {"type": "integer", "description": "读取最后 N 行，默认 64", "default": 64}
                },
                "required": []
            },
            handler=self._read_terminal_log,
            skip_memory=True
        )
        self.register_tool(
            name="read_chat_history",
            description="Read recent chat/agent/terminal history for context recovery.",
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "Item id."},
                    "limit": {"type": "integer", "description": "Messages to return (default 30).", "default": 30},
                    "offset": {"type": "integer", "description": "Skip newest N messages.", "default": 0},
                    "query": {"type": "string", "description": "Substring filter."},
                    "include_summary": {"type": "boolean", "description": "Include session summary.", "default": True},
                },
                "required": ["item_id"],
            },
            handler=self._read_chat_history,
            skip_memory=True,
        )
        self.register_tool(
            name="prepare_capabilities",
            description=(
                "Load on-demand capabilities listed in the capability catalog. "
                "Pass the exact tool names and/or guide ids you need; returns "
                "their full schemas/guidance and keeps them available for this "
                "item. Call this before using any catalog-listed tool."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Current terminal item id.",
                    },
                    "names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tool names and/or guide ids to load.",
                    },
                },
                "required": ["item_id", "names"],
            },
            handler=self._prepare_capabilities,
            skip_memory=True,
        )
        self.register_tool(
            name="add_terminal_input_filter_rule",
            description="添加终端输出过滤规则，屏蔽重复无害日志。",
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "name": {"type": "string", "description": "Rule name."},
                    "regex_patterns": {"type": "array", "items": {"type": "string"}, "description": "Regex patterns to filter."},
                    "action_type": {"type": "string", "enum": ["block", "ignore", "log"], "default": "block"},
                    "reason": {"type": "string"},
                },
                "required": ["item_id", "name", "regex_patterns"]
            },
            handler=self._add_terminal_input_filter_rule,
            skip_memory=True
        )
        self.register_tool(
            name="list_terminal_input_filter_rules",
            description="List terminal input filter rules.",
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "Current terminal item id."},
                },
                "required": ["item_id"]
            },
            handler=self._list_terminal_input_filter_rules,
            skip_memory=True
        )
        self.register_tool(
            name="list_terminal_filter_rules",
            description="List all terminal filter rules.",
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "Current terminal item id."},
                },
                "required": ["item_id"]
            },
            handler=self._list_terminal_filter_rules,
            skip_memory=True
        )
        self.register_tool(
            name="delete_terminal_input_filter_rule",
            description="Delete a terminal input filter rule by name.",
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "name": {"type": "string", "description": "Rule name."},
                    "disable_when_empty": {"type": "boolean", "default": True},
                },
                "required": ["item_id", "name"]
            },
            handler=self._delete_terminal_input_filter_rule,
            skip_memory=True
        )
        self.register_tool(
            name="clear_terminal_input_filter_rules",
            description="Delete all terminal input filter rules.",
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string"},
                    "disable": {"type": "boolean", "default": True},
                },
                "required": ["item_id"]
            },
            handler=self._clear_terminal_input_filter_rules,
            skip_memory=True
        )

        self.register_tool(
            name="list_scheduled_tasks",
            description=(
                "Read all scheduled tasks for the current terminal item, including ids, "
                "instructions, enabled state, next run time, and last result."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "Current terminal item id; injected automatically.",
                    }
                },
                "required": [],
            },
            handler=self._list_scheduled_tasks,
            skip_memory=True,
        )
        self.register_tool(
            name="write_scheduled_task",
            description=(
                "Create or update one scheduled task for the current terminal item. "
                "Pass task_id to update an existing task. Supported schedules are once, "
                "interval, and daily."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "Item id (auto-injected)."},
                    "task_id": {"type": "string", "description": "Task id if updating."},
                    "name": {"type": "string", "description": "Task name."},
                    "instruction": {"type": "string", "description": "Agent instruction."},
                    "schedule_type": {"type": "string", "enum": ["once", "interval", "daily"]},
                    "run_at": {"type": "string", "description": "ISO datetime for once."},
                    "interval_seconds": {
                        "type": "integer",
                        "description": "Interval in seconds for an interval task; minimum 10.",
                    },
                    "time_of_day": {
                        "type": "string",
                        "description": "Local HH:MM time for a daily task.",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone, default Asia/Shanghai.",
                        "default": "Asia/Shanghai",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "Whether the task should run.",
                        "default": True,
                    },
                },
                "required": ["name", "instruction", "schedule_type"],
            },
            handler=self._write_scheduled_task,
            skip_memory=True,
        )
        self.register_tool(
            name="delete_scheduled_task",
            description="Delete a scheduled task by id.",
            input_schema={
                "type": "object",
                "properties": {
                    "item_id": {"type": "string", "description": "Item id (auto-injected)."},
                    "task_id": {"type": "string", "description": "Task id."},
                    "reason": {"type": "string", "description": "Deletion reason."},
                },
                "required": ["task_id"],
            },
            handler=self._delete_scheduled_task,
            skip_memory=True,
        )
        self.register_tool(
            name="save_memory",
            description="保存稳定、可复用、已验证的重要信息到长期记忆中。用户随口提到的个人信息（家庭、喜好、计划、所有物、常去地点、正在做的事）也值得保存，不必等用户说『记住』。执行中的任务状态只放任务队列。不要保存原生日志、命令回显、等待态消息或敏感信息。",
            input_schema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Memory content."},
                    "memory_type": {"type": "string", "enum": ["fact", "preference", "error", "context"]},
                    "ttl_days": {"type": "integer", "description": "Expiry days (optional)."}
                },
                "required": ["content"]
            },
            handler=self._save_memory
        )

        self.register_tool(
            name="recall_memory",
            description="从长期记忆中检索相关信息。使用语义搜索，返回与查询最相关的记忆。",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词或问题"},
                    "n_results": {"type": "integer", "description": "返回结果数量；0 或不填表示不限条数（按相关性阈值过滤）", "default": 0},
                    "memory_type": {"type": "string", "enum": ["fact", "preference", "error", "context"], "description": "可选：限定记忆类型"}
                },
                "required": ["query"]
            },
            handler=self._recall_memory
        )

        self.register_tool(
            name="list_memories",
            description="列出所有记忆，可按类型过滤。",
            input_schema={
                "type": "object",
                "properties": {
                    "memory_type": {"type": "string", "enum": ["fact", "preference", "error", "context"]}
                },
                "required": []
            },
            handler=self._list_memories
        )

        self.register_tool(
            name="delete_memory",
            description="删除指定的记忆。",
            input_schema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory ID."}
                },
                "required": ["memory_id"]
            },
            handler=self._delete_memory
        )

        self.register_tool(
            name="compress_memories",
            description="把多条重复、冗余或过时的长期记忆压缩合并成一条精炼记忆：先写入新记忆，再删除列出的旧记忆。当 recall_memory 或 list_memories 的结果里有重复内容、同一事实的旧版本或废话时使用。",
            input_schema={
                "type": "object",
                "properties": {
                    "memory_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "description": "IDs to merge."},
                    "content": {"type": "string", "description": "Merged content."},
                    "memory_type": {"type": "string", "enum": ["fact", "preference", "error", "context"]},
                    "ttl_days": {"type": "integer"}
                },
                "required": ["memory_ids", "content"]
            },
            handler=self._compress_memories
        )

    def register_tool(self, name: str, description: str, input_schema: dict, handler: callable, skip_memory: bool = False):
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler,
            "skip_memory": skip_memory
        }

    def _get_terminal_status(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]
        try:
            from app.services.terminal_runtime_state import get_terminal_runtime_state

            state = get_terminal_runtime_state(item_id)
            return [
                {"type": "text", "text": state.user_message()},
                {
                    "type": "metadata",
                    "terminal_active": state.active,
                    "daemon_connected": state.daemon_connected,
                    "process_status": state.process_status,
                    "backend_room_connected": state.backend_room_connected,
                    "permanent_subscriber_count": state.permanent_subscriber_count,
                    "reason": state.reason,
                },
            ]
        except Exception as exc:
            debug_log(f"[LocalMCPServer] get_terminal_status error: {exc}")
            return [{"type": "text", "text": f"Error: {exc}"}]

    def _restore_existing_terminal_input(self, item_id: str) -> bool:
        try:
            import uuid

            from sqlmodel import Session

            from app.core.db import engine
            from app.models import Item
            from app.services import (
                DaemonConfig,
                connection_manager,
                socket_pool_facade,
            )
            from app.services.terminal_service import TerminalService

            try:
                item_uuid = uuid.UUID(str(item_id))
            except ValueError:
                debug_log(f"[LocalMCPServer] invalid item_id for restore: {item_id}")
                return False

            with Session(engine) as db:
                item = db.get(Item, item_uuid)
                if not item:
                    debug_log(f"[LocalMCPServer] item not found for restore: {item_id}")
                    return False
                if not item.socket_host or not item.socket_port or not item.api_key:
                    debug_log(f"[LocalMCPServer] item missing daemon config for restore: {item_id}")
                    return False

                daemon_config = DaemonConfig(item.socket_host, item.socket_port, item.api_key)
                owner_uuid = str(item.owner_id)

            terminal_service = TerminalService(connection_manager, socket_pool_facade)
            restored = terminal_service.restore_running_terminal(
                item_uuid=str(item_id),
                owner_uuid=owner_uuid,
                daemon_config=daemon_config,
            )
            debug_log(f"[LocalMCPServer] terminal input restore result: item={item_id}, restored={restored}")
            return bool(restored)
        except Exception as exc:
            debug_log(f"[LocalMCPServer] terminal input restore error for item={item_id}: {exc}")
            return False

    def _ensure_terminal_input_handler(self, item_id: str) -> bool:
        from app.services.socket_pool.input_center import input_center
        from app.services.terminal_runtime_state import get_terminal_runtime_state

        state = get_terminal_runtime_state(item_id)
        if state.active and input_center.has_handler(item_id):
            return True

        # A live PTY can survive a Backend restart. Rejoin its Item Room before
        # accepting input, then verify the Room state again instead of trusting
        # the newly registered local handler.
        if state.daemon_connected and state.terminal_process_active:
            self._restore_existing_terminal_input(item_id)
            state = get_terminal_runtime_state(item_id)
            if state.active and input_center.has_handler(item_id):
                return True

        # Cached handlers are not connection evidence. Remove them so a later
        # command cannot be reported as sent through a dead socket callback.
        if input_center.has_handler(item_id):
            input_center.unregister_all_by_item(item_id)
        return False

    def _terminal_unavailable_message(self, item_id: str) -> str:
        try:
            from app.services.terminal_runtime_state import get_terminal_runtime_state

            state = get_terminal_runtime_state(item_id)
            if state.active:
                return (
                    "终端已启动并已连接，但 Backend 输入处理器不可用，"
                    "自动恢复失败。命令没有发送。"
                )
            return f"{state.user_message()}命令没有发送。"
        except Exception:
            return TERMINAL_NOT_CONNECTED_MESSAGE

    def _terminal_unavailable_result(self, item_id: str) -> list[dict[str, Any]]:
        return [
            {"type": "text", "text": self._terminal_unavailable_message(item_id)},
            {
                "type": "metadata",
                COMMAND_DISPATCH_FAILED_MARKER: True,
                TERMINAL_UNAVAILABLE_RESULT_MARKER: True,
                "reason": TERMINAL_UNAVAILABLE_RESULT_MARKER,
                "terminal_active": False,
                "command_sent": False,
            },
        ]

    def _get_item_daemon_context(self, item_id: str):
        import uuid

        from sqlmodel import Session

        from app.core.db import engine
        from app.models import Item
        from app.services import DaemonConfig, connection_manager

        try:
            item_uuid = uuid.UUID(str(item_id))
        except ValueError:
            raise ValueError(f"invalid item_id: {item_id}")

        with Session(engine) as db:
            item = db.get(Item, item_uuid)
            if not item:
                raise ValueError(f"item not found: {item_id}")
            if not item.socket_host or not item.socket_port or not item.api_key:
                raise ValueError(f"daemon is not configured for item: {item_id}")
            daemon_config = DaemonConfig(item.socket_host, item.socket_port, item.api_key)
            connection = connection_manager.get_or_create_connection(daemon_config)
            return item, connection

    def _coerce_job_int(self, value, default: int, minimum: int, maximum: int) -> int:
        try:
            coerced = int(float(value)) if value is not None and value != "" else default
        except (TypeError, ValueError):
            coerced = default
        return max(minimum, min(coerced, maximum))

    def _format_job_result(self, result: dict) -> str:
        exit_code = result.get("exit_code")
        timed_out = bool(result.get("timed_out"))
        status = "timed out" if timed_out else ("succeeded" if exit_code == 0 else "failed")
        output_tail = (result.get("output_tail") or "").strip() or "(no output)"
        return (
            f"Job {status}\n"
            f"job_id: {result.get('job_id', '')}\n"
            f"command: {result.get('command', '')}\n"
            f"cwd: {result.get('cwd', '')}\n"
            f"exit_code: {exit_code}\n"
            f"timed_out: {timed_out}\n"
            f"duration_seconds: {result.get('duration_seconds', '')}\n"
            f"output_tail:\n{output_tail}"
        )

    def _format_jobs_result(self, result: dict) -> str:
        jobs = result.get("jobs") or []
        if not jobs:
            return "没有运行中的后台任务。"
        lines = [f"运行中的后台任务：{len(jobs)} 个"]
        for job in jobs:
            command = str(job.get("command") or "")
            if len(command) > 120:
                command = f"{command[:117]}..."
            lines.append(
                " - "
                f"job_id={job.get('job_id', '')} "
                f"elapsed={int(float(job.get('elapsed_seconds') or 0))}s "
                f"cancel_requested={bool(job.get('cancel_requested'))} "
                f"command={command}"
            )
            output_tail = str(job.get("output_tail") or "").strip()
            if output_tail:
                if len(output_tail) > 1200:
                    output_tail = f"{output_tail[-1200:]}"
                lines.append(f"   output_tail:\n{output_tail}")
        return "\n".join(lines)

    def _clip_history_text(self, value: Any, limit: int = 900) -> str:
        text = str(value or "").replace("\r\n", "\n").strip()
        if len(text) <= limit:
            return text
        head = max(limit // 2 - 20, 120)
        tail = max(limit - head - 40, 120)
        return f"{text[:head].rstrip()}\n...<truncated>...\n{text[-tail:].lstrip()}"

    def _format_chat_history_entry(self, index: int, message: dict[str, Any]) -> str:
        timestamp = str(message.get("timestamp") or "").strip() or "unknown-time"
        message_type = str(message.get("type") or "message").strip()
        role = str(message.get("role") or "").strip()
        detail_parts: list[str] = []
        if role:
            detail_parts.append(f"role={role}")
        for key in ("tool_name", "source_type", "source_label", "status"):
            value = str(message.get(key) or "").strip()
            if value:
                detail_parts.append(f"{key}={self._clip_history_text(value, 120)}")
        details = f" ({', '.join(detail_parts)})" if detail_parts else ""
        content = self._clip_history_text(message.get("content"), 1000) or "(empty)"
        return f"[{index}] {timestamp} {message_type}{details}\n{content}"

    def _read_chat_history(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]

        limit = self._coerce_job_int(args.get("limit"), 30, 1, 100)
        offset = self._coerce_job_int(args.get("offset"), 0, 0, 10000)
        query = str(args.get("query") or "").strip().lower()
        include_summary = bool(args.get("include_summary", True))

        try:
            from app.services.agent.history.chat import (
                SESSION_SUMMARY_TYPE,
                get_chat_messages,
                get_latest_session_summary,
            )

            messages = get_chat_messages(item_id)
            summary = get_latest_session_summary(item_id) if include_summary else None
            history_messages = [
                message
                for message in messages
                if isinstance(message, dict)
                and message.get("type") != SESSION_SUMMARY_TYPE
            ]
            if query:
                history_messages = [
                    message
                    for message in history_messages
                    if query in str(message.get("content") or "").lower()
                    or query in str(message.get("type") or "").lower()
                    or query in str(message.get("tool_name") or "").lower()
                ]

            total = len(history_messages)
            effective_offset = min(offset, total)
            end = max(total - effective_offset, 0)
            start = max(end - limit, 0)
            page = history_messages[start:end]

            lines = [
                "TermPaws item chat history:",
                f"item_id={item_id}",
                f"total_matching_messages={total}",
                f"limit={limit}",
                f"offset={effective_offset}",
                "order=oldest_to_newest",
            ]
            if query:
                lines.append(f"query={query}")
            if summary and summary.get("content"):
                lines.append("")
                lines.append("Latest session summary:")
                lines.append(self._clip_history_text(summary.get("content"), 1200))
            if page:
                lines.append("")
                lines.append("Messages:")
                for display_index, message in enumerate(page, start=start + 1):
                    lines.append(self._format_chat_history_entry(display_index, message))
            else:
                lines.append("")
                lines.append("No matching chat history messages.")
            lines.append("")
            lines.append(
                "Use this as short-term evidence for previous work; do not quote raw history unless the user asks."
            )
            return [{"type": "text", "text": "\n".join(lines)}]
        except Exception as e:
            debug_log(f"[LocalMCPServer] read_chat_history error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _update_plan(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        reply_ticket_id = str(args.get("_reply_ticket_id") or "").strip()
        plan = args.get("plan")
        supersede_ticket_ids = [
            str(value or "").strip()
            for value in (args.get("supersede_ticket_ids") or [])
            if str(value or "").strip()
        ]
        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]

        from app.services.agent.reply_ticket import reply_ticket_manager

        # The plan is a field of the owning reply ticket. A turn without a
        # resolvable ticket gets an error instead of writing the plan anywhere
        # it would be misattributed.
        ticket = reply_ticket_manager.get(reply_ticket_id) if reply_ticket_id else None
        if ticket is None:
            return [
                {
                    "type": "text",
                    "text": (
                        "Error: 无法关联 ticket（reply ticket 不存在或已失效），"
                        "plan 未保存。请先继续当前任务，必要时重新发起。"
                    ),
                }
            ]
        if ticket.plan_cancelled:
            return [
                {
                    "type": "text",
                    "text": (
                        "Error: 该任务已被用户取消（计划已终止）。本次写入被拒绝。"
                        "不要重建计划，不要继续执行该任务的任何步骤。"
                    ),
                }
            ]

        # GLM 类模型在并行工具调用时偶尔会补一个占位调用
        # （plan=[{step:"placeholder"}]），它会覆盖掉真实计划、逼模型下轮
        # 再恢复——纯浪费。占位写入直接忽略，保持当前计划不变。
        if (
            isinstance(plan, list)
            and len(plan) == 1
            and isinstance(plan[0], dict)
            and str(plan[0].get("step") or "").strip().lower() in {"placeholder", "占位"}
        ):
            existing = reply_ticket_manager.get(ticket.ticket_id)
            current = existing.plan if existing else []
            return [
                {
                    "type": "text",
                    "text": (
                        f"已忽略占位 plan 写入，当前计划保持不变（{len(current)} 步）。"
                        "不要重复发起占位调用。"
                    ),
                }
            ]

        # 接管同任务旧计划：关闭旧票据上的计划，条带立即去重。
        superseded_notes: list[str] = []
        for ref in supersede_ticket_ids:
            if ticket.ticket_id == ref or ticket.ticket_id.startswith(ref):
                continue
            finalized = reply_ticket_manager.supersede_plan(ref, item_id)
            if finalized is not None:
                superseded_notes.append(
                    f"已关闭旧计划 {ref[:8]}（{len(finalized)} 个步骤标记取消）"
                )

        # plan=[] means "task done" — mark all remaining steps as completed
        # and keep the plan on the ticket for UI history.
        if isinstance(plan, list) and not plan:
            existing_ticket = reply_ticket_manager.get(ticket.ticket_id)
            finalized: list[dict] = []
            if existing_ticket and existing_ticket.plan:
                finalized = [
                    {**step, "status": "completed"}
                    if step.get("status") not in {"completed", "cancelled"}
                    else step
                    for step in existing_ticket.plan
                ]
            # The agent explicitly declared the task done: broadcast the final
            # all-completed snapshot once, then clear the plan from the ticket
            # so finished work leaves the plan table immediately.
            reply_ticket_manager.update_ticket_plan(ticket.ticket_id, [])
            try:
                from app.services.agent.stream_manager import stream_manager

                stream_manager.broadcast_chat_event(
                    item_id,
                    {
                        "type": "plan_updated",
                        "item_id": item_id,
                        "plan": finalized,
                        "explanation": "completed",
                    },
                )
            except Exception:
                pass
            return [{"type": "text", "text": "Plan completed."}]

        if not isinstance(plan, list):
            return [{"type": "text", "text": "Error: plan must be a list"}]
        normalized: list[dict] = []
        for entry in plan:
            if not isinstance(entry, dict):
                continue
            step = str(entry.get("step") or "").strip()
            status = str(entry.get("status") or "pending").strip()
            if not step:
                continue
            if status not in {"pending", "in_progress", "completed", "cancelled"}:
                status = "pending"
            normalized.append({"step": step, "status": status})
        if not normalized:
            return [{"type": "text", "text": "Error: plan has no valid steps"}]

        all_terminal = all(
            item["status"] in {"completed", "cancelled"} for item in normalized
        )
        # Once every step reached a terminal state, broadcast the final
        # snapshot once, then clear the plan from the ticket so finished work
        # leaves the plan table immediately (no extra plan=[] call needed).
        reply_ticket_manager.update_ticket_plan(
            ticket.ticket_id, [] if all_terminal else normalized
        )
        try:
            from app.services.agent.stream_manager import stream_manager

            stream_manager.broadcast_chat_event(
                item_id,
                {
                    "type": "plan_updated",
                    "item_id": item_id,
                    "plan": normalized,
                    "explanation": "completed"
                    if all_terminal
                    else str(args.get("explanation") or ""),
                },
            )
        except Exception:
            pass
        explanation = str(args.get("explanation") or "").strip()
        lines = ["Plan updated:"]
        lines.extend(
            f"  {index}. [{item['status']}] {item['step']}"
            for index, item in enumerate(normalized, start=1)
        )
        lines.extend(superseded_notes)
        if explanation:
            lines.append(f"Note: {explanation}")
        if all(item["status"] == "completed" for item in normalized):
            lines.append(
                "All steps complete. Report the final result to the user now "
                "(QQ: mcp_robot_send_message; web: direct reply)."
            )
        elif all(item["status"] == "cancelled" for item in normalized):
            lines.append(
                "All steps cancelled. Briefly confirm the task was stopped."
            )
        else:
            current = next(
                (item for item in normalized if item["status"] == "in_progress"),
                None,
            )
            if current is not None:
                remaining = [
                    item
                    for item in normalized
                    if item["status"] not in {"completed", "cancelled"}
                ]
                if len(remaining) == 1:
                    lines.append(
                        f"Next action: finish step '{current['step']}' and report to the user. "
                        "汇报发出后，调用 update_plan(plan=[]) 直接清除计划，不用逐步标 completed。"
                    )
                else:
                    lines.append(
                        f"Next action: execute step '{current['step']}' immediately. "
                        "Do NOT reply to the user yet — report only when all steps are "
                        "done, or one is genuinely blocked."
                    )
        return [{"type": "text", "text": "\n".join(lines)}]

    def _provision_turn_reply_ticket(self, item_id: str, args: dict) -> str:
        """Create and attach a reply ticket for a turn that has none.

        Some turns (e.g. queued/merged follow-ups) start without a linked
        reply ticket, which used to make action=create fail outright. The
        ticket source (web/QQ) is inferred from the agent's live context by
        reply_ticket_manager, and the new ticket is attached to the agent so
        subsequent tool calls in this turn are linked automatically.
        """
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager
            from app.services.agent.session import agent_session_manager

            session = agent_session_manager.get_session(str(item_id))
            if session is None:
                return ""
            agent = session.get_agent()
            if agent is None:
                return ""
            handler_id = str(args.get("_handler_id") or "") or str(
                getattr(session, "handler_id", "") or ""
            )
            ticket = reply_ticket_manager.create_for_agent(
                agent,
                item_id=str(item_id),
                handler_id=handler_id,
                message=str(args.get("note") or args.get("title") or ""),
            )
            debug_log(
                f"[LocalMCPServer] provisioned reply ticket {ticket.ticket_id} "
                f"for item={item_id}"
            )
            return str(ticket.ticket_id or "")
        except Exception as exc:
            debug_log(f"[LocalMCPServer] provision reply ticket error: {exc}")
            return ""

    def _list_jobs(self, args: dict) -> list:
        item_id = args.get("item_id", "")
        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]
        try:
            _item, connection = self._get_item_daemon_context(str(item_id))
            result = connection.list_jobs_http(item_uuid=str(item_id))
            if not result.get("success"):
                return [{"type": "text", "text": f"Error: {result.get('error', 'failed to list jobs')}"}]
            return [{"type": "text", "text": self._format_jobs_result(result)}]
        except Exception as e:
            debug_log(f"[LocalMCPServer] list_jobs error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _cancel_job(self, args: dict) -> list:
        item_id = args.get("item_id", "")
        job_id = str(args.get("job_id") or "").strip()
        if not item_id or not job_id:
            return [{"type": "text", "text": "Error: item_id and job_id required"}]
        try:
            _item, connection = self._get_item_daemon_context(str(item_id))
            result = connection.cancel_job_http(item_uuid=str(item_id), job_id=job_id)
            if result.get("success") or result.get("cancelled"):
                from app.services.agent.session import agent_session_manager

                agent_session = agent_session_manager.get_session(str(item_id))
                if agent_session:
                    agent_session.clear_terminal_job()
                # bot 侧取消 = 任务终态：给票据立碑。否则 job 杀掉了但计划
                # 还在，sweep 续跑 → agent 重建计划 → 任务复活循环。
                ticket_id = str(args.get("_reply_ticket_id") or "").strip()
                if ticket_id:
                    from app.services.agent.reply_ticket import reply_ticket_manager

                    reply_ticket_manager.cancel_ticket_plan(ticket_id)
                return [{"type": "text", "text": f"后台任务已取消：{result.get('job_id', job_id)}"}]
            return [{"type": "text", "text": f"Error: {result.get('error', 'failed to cancel job')}"}]
        except Exception as e:
            debug_log(f"[LocalMCPServer] cancel_job error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _run_job(self, args: dict) -> list:
        command = (args.get("command") or "").strip()
        item_id = args.get("item_id", "")
        if not command or not item_id:
            return [{"type": "text", "text": "Error: command and item_id required"}]

        if not self._ensure_terminal_input_handler(str(item_id)):
            debug_log(
                f"[LocalMCPServer] run_job blocked because main terminal is inactive: "
                f"item={item_id}, command={command}"
            )
            return self._terminal_unavailable_result(str(item_id))

        timeout_seconds = self._coerce_job_int(args.get("timeout_seconds"), 1200, 1, 3600)
        tail_lines = self._coerce_job_int(args.get("tail_lines"), 80, 1, 300)
        requested_wait_for_completion = bool(args.get("wait_for_completion"))
        robot_job_context = self._robot_job_context_from_args(args)
        reply_ticket_id = str(args.get("_reply_ticket_id") or "").strip()
        if reply_ticket_id:
            try:
                from app.services.agent.reply_ticket import reply_ticket_manager

                if reply_ticket_manager.get(reply_ticket_id) is None:
                    return [
                        {
                            "type": "text",
                            "text": (
                                "Task request was superseded by a newer message; "
                                "the background job was not started."
                            ),
                        }
                    ]
                reply_ticket_manager.mark_command(reply_ticket_id, command)
            except Exception:
                pass
        debug_log(
            f"[LocalMCPServer] _run_job: item={item_id}, timeout={timeout_seconds}, "
            f"tail_lines={tail_lines}, requested_wait={requested_wait_for_completion}, "
            f"command={command}"
        )
        if requested_wait_for_completion:
            debug_log(
                "[LocalMCPServer] run_job ignored wait_for_completion=True; "
                "all daemon jobs are asynchronous"
            )
        agent_session = None
        try:
            from app.services.agent.session import (
                RUN_JOB_TOOL_NAME,
                agent_session_manager,
            )

            agent_session = agent_session_manager.get_session(str(item_id))
            if agent_session:
                guard_error = agent_session.validate_terminal_tool_input(
                    RUN_JOB_TOOL_NAME,
                    {
                        "item_id": str(item_id),
                        "command": command,
                        "timeout_seconds": timeout_seconds,
                        "_reply_ticket_id": reply_ticket_id,
                    },
                )
                if guard_error:
                    debug_log(
                        f"[LocalMCPServer] run_job blocked before start: item={item_id}, command={command}"
                    )
                    return [{"type": "text", "text": guard_error}]
                agent_session.mark_terminal_job_started(
                    RUN_JOB_TOOL_NAME,
                    {
                        "item_id": str(item_id),
                        "command": command,
                        "timeout_seconds": timeout_seconds,
                        "_reply_ticket_id": reply_ticket_id,
                    },
                )
        except Exception as guard_error:
            debug_log(f"[LocalMCPServer] run_job guard error: {guard_error}")
            agent_session = None

        try:
            item, connection = self._get_item_daemon_context(str(item_id))
            request_kwargs = {
                "item_uuid": str(item_id),
                "user_uuid": str(item.owner_id),
                "command": command,
                "working_directory": item.working_directory,
                "timeout_seconds": timeout_seconds,
                "tail_lines": tail_lines,
                "item_title": item.title,
            }

            # Check for duplicate running command
            _dup_hint = ""
            try:
                _existing = connection.call("list_jobs", {"item_uuid": str(item_id)})
                for _j in (_existing or {}).get("jobs", []):
                    if str(_j.get("command", "")).strip() == command:
                        _dup_hint = (
                            f" WARNING: identical command already running as "
                            f"job_id={_j.get('job_id','')} "
                            f"elapsed={int(float(_j.get('elapsed_seconds',0)))}s. "
                            f"Consider waiting for it instead of starting a duplicate."
                        )
                        break
            except Exception:
                pass

            self._start_background_job_thread(
                item_id=str(item_id),
                command=command,
                connection=connection,
                request_kwargs=request_kwargs,
                agent_session=agent_session,
                robot_job_context=robot_job_context,
                reply_ticket_id=reply_ticket_id,
                pending_robot_reply_id=self._register_background_job_robot_reply(
                    item_id=str(item_id),
                    command=command,
                    robot_job_context=robot_job_context,
                ),
            )
            return [
                {
                    "type": "text",
                    "text": (
                        "后台任务已启动，完成后自动送回结果。"
                        + _dup_hint
                    ),
                },
                {
                    "type": "metadata",
                    BACKGROUND_JOB_STARTED_MARKER: True,
                    "effective_tool_name": "mcp_local_run_job",
                },
            ]
        except Exception as e:
            debug_log(f"[LocalMCPServer] run_job error: {e}")
            if agent_session:
                agent_session.clear_terminal_job(command)
            return [{"type": "text", "text": f"Error: {e}"}]

    def _poll_background_job_result(
        self,
        connection,
        *,
        item_id: str,
        command: str,
        daemon_job_id: str,
        timeout_seconds,
    ) -> dict:
        timeout_value = float(self._coerce_job_int(timeout_seconds, 600, 1, 3600))
        deadline = time.monotonic() + timeout_value + 180.0
        consecutive_errors = 0
        time.sleep(JOB_RESULT_POLL_INITIAL_DELAY_SECONDS)
        while time.monotonic() < deadline:
            try:
                poll = connection.get_job_result_http(
                    item_uuid=str(item_id),
                    job_id=daemon_job_id,
                )
            except Exception as exc:
                poll = {"success": False, "error": str(exc)}
            status = str(poll.get("status") or "")
            if poll.get("success") and status == "finished":
                result = poll.get("result")
                if isinstance(result, dict):
                    return result
                break
            if poll.get("success") and status == "running":
                consecutive_errors = 0
                time.sleep(JOB_RESULT_POLL_INTERVAL_SECONDS)
                continue
            if status == "unknown":
                break
            consecutive_errors += 1
            if consecutive_errors >= JOB_RESULT_POLL_MAX_ERRORS:
                break
            time.sleep(JOB_RESULT_POLL_INTERVAL_SECONDS)
        return {
            "success": False,
            "error": "daemon job result unavailable (job lost or daemon restarted)",
            "command": command,
            "job_id": daemon_job_id,
            "exit_code": None,
            "timed_out": False,
            "duration_seconds": "",
            "output_tail": "",
        }

    def _start_background_job_thread(
        self,
        *,
        item_id: str,
        command: str,
        connection,
        request_kwargs: dict,
        agent_session,
        robot_job_context: dict | None = None,
        reply_ticket_id: str = "",
        pending_robot_reply_id: str | None = None,
    ) -> None:
        def worker() -> None:
            daemon_job_id = ""
            pending_job_recorded = False
            try:
                start_result = connection.run_job_http(**request_kwargs)
            except Exception as exc:
                start_result = {"success": False, "error": str(exc)}
            if not start_result.get("success"):
                result = {
                    "success": False,
                    "error": str(start_result.get("error") or "daemon job failed to start"),
                    "command": command,
                    "job_id": str(start_result.get("job_id") or ""),
                    "exit_code": None,
                    "timed_out": False,
                    "duration_seconds": "",
                    "output_tail": "",
                }
            else:
                daemon_job_id = str(start_result.get("job_id") or "")
                if "exit_code" in start_result:
                    # Old daemon without async start: the blocking call already
                    # returned the finished result.
                    result = start_result
                else:
                    debug_log(
                        f"[LocalMCPServer] background job scheduled: item={item_id}, job_id={daemon_job_id}, command={command}"
                    )
                    if reply_ticket_id:
                        try:
                            from app.services.agent.reply_ticket import (
                                reply_ticket_manager,
                            )

                            reply_ticket_manager.add_pending_job(
                                reply_ticket_id, daemon_job_id
                            )
                            pending_job_recorded = True
                        except Exception:
                            pass
                    result = self._poll_background_job_result(
                        connection,
                        item_id=item_id,
                        command=command,
                        daemon_job_id=daemon_job_id,
                        timeout_seconds=request_kwargs.get("timeout_seconds"),
                    )
            debug_log(
                f"[LocalMCPServer] background run_job result: item={item_id}, success={result.get('success')}, exit_code={result.get('exit_code')}, timed_out={result.get('timed_out')}"
            )

            if pending_job_recorded:
                # exactly-once 闸门：resume sweep 若已抢先拉取并投递了这个
                # 结果（进程重启后轮询线程已死的情况），这里直接收尾退出。
                try:
                    from app.services.agent.reply_ticket import reply_ticket_manager

                    if not reply_ticket_manager.take_pending_job(
                        reply_ticket_id, daemon_job_id
                    ):
                        if agent_session:
                            agent_session.clear_terminal_job(command)
                        return
                except Exception:
                    pass

            if result.get("cancelled") and reply_ticket_id:
                # User-initiated cancel: broadcast the cancelled snapshot for
                # UI, then clear the plan from the ticket — same lifecycle as
                # update_plan with all-terminal steps. The plan is gone from
                # every subsequent turn.
                try:
                    from app.services.agent.reply_ticket import reply_ticket_manager

                    ticket = reply_ticket_manager.get(reply_ticket_id)
                    if ticket and ticket.plan:
                        finalized = [
                            {**step, "status": "cancelled"}
                            if step.get("status") not in {"completed", "cancelled"}
                            else step
                            for step in ticket.plan
                        ]
                        reply_ticket_manager.update_ticket_plan(
                            ticket.ticket_id, []
                        )
                        try:
                            from app.services.agent.stream_manager import (
                                stream_manager,
                            )

                            stream_manager.broadcast_chat_event(
                                item_id,
                                {
                                    "type": "plan_updated",
                                    "item_id": item_id,
                                    "plan": finalized,
                                    "explanation": "cancelled",
                                },
                            )
                        except Exception:
                            pass
                except Exception as exc:
                    debug_log(
                        f"[LocalMCPServer] failed to finalize plan on cancel: "
                        f"ticket={reply_ticket_id}, error={exc}"
                    )

            feedback = self._format_background_job_feedback(result, reply_ticket_id=reply_ticket_id)
            if reply_ticket_id:
                try:
                    from app.services.agent.scheduled_tasks import (
                        record_scheduled_ticket_result,
                    )

                    record_scheduled_ticket_result(
                        reply_ticket_id,
                        success=(
                            bool(result.get("success"))
                            and result.get("exit_code") == 0
                            and not result.get("timed_out")
                            and not result.get("cancelled")
                        ),
                        error="" if result.get("success") else self._format_job_result(result),
                    )
                except Exception as exc:
                    debug_log(
                        f"[LocalMCPServer] failed to update scheduled task result: "
                        f"ticket={reply_ticket_id}, error={exc}"
                    )
            if agent_session:
                agent_session.clear_terminal_job(command)
                agent_session.record_finished_job(
                    command,
                    job_id=str(result.get("job_id") or ""),
                    success=bool(result.get("success")) and not result.get("cancelled"),
                    exit_code=result.get("exit_code"),
                )

            busy = _conversation_turn_busy(agent_session, robot_job_context)
            entry = {
                "command": command,
                "result": result,
                "robot_job_context": robot_job_context,
                "pending_robot_reply_id": pending_robot_reply_id or "",
                "reply_ticket_id": reply_ticket_id,
                "feedback": feedback,
                "agent_session": agent_session,
                "conversation_key": (robot_job_context or {}).get("conversation_key", ""),
            }
            if busy:
                # A turn is running: park the result in the merge buffer so
                # results finishing during this turn are merged into ONE
                # follow-up turn at turn end. Idle conversations still get an
                # immediate turn (else branch), so nothing waits artificially.
                buffer_background_job_result(item_id, entry)
                debug_log(
                    f"[LocalMCPServer] job result buffered while turn active: "
                    f"item={item_id}, command={command}"
                )
                if not _conversation_turn_busy(agent_session, robot_job_context):
                    flush_all_job_results_for_item(item_id)
            else:
                flush_background_job_results_for_entries(item_id, [entry])

        thread = threading.Thread(
            target=worker,
            name=f"TermPaws-job-{item_id[:8]}",
            daemon=True,
        )
        thread.start()

    def _robot_job_context_from_args(self, args: dict) -> dict | None:
        context_token = str(args.get("_robot_context_token") or "").strip()
        if not context_token:
            return None
        try:
            from app.services.agent.integrations.hooks import (
                get_integration_context_by_token,
            )

            context = get_integration_context_by_token(context_token)
            if context is None or getattr(context, "reply_target", None) is None:
                return None
            return {
                "robot_id": str(getattr(context, "robot_id", "") or ""),
                "sender_key": str(getattr(context, "sender_key", "") or ""),
                "reply_target": context.reply_target.model_dump(mode="json"),
                "conversation_key": str(getattr(context, "conversation_key", "") or ""),
                "conversation_generation": int(
                    getattr(context, "conversation_generation", 0) or 0
                ),
            }
        except Exception as exc:
            debug_log(f"[LocalMCPServer] failed to capture robot job context: {exc}")
            return None

    def _register_background_job_robot_reply(
        self,
        *,
        item_id: str,
        command: str,
        robot_job_context: dict | None,
    ) -> str | None:
        if not robot_job_context:
            return None
        try:
            from app.services.agent.integrations.hooks import (
                register_integration_background_job_reply,
            )

            return register_integration_background_job_reply(
                integration_id=robot_job_context.get("robot_id", ""),
                item_id=item_id,
                sender_key=robot_job_context.get("sender_key", ""),
                reply_target=robot_job_context.get("reply_target") or {},
                conversation_key=robot_job_context.get("conversation_key", ""),
                conversation_generation=int(
                    robot_job_context.get("conversation_generation") or 0
                ),
                command=command,
            )
        except Exception as exc:
            debug_log(
                f"[LocalMCPServer] failed to register robot background job reply: item={item_id}, error={exc}"
            )
            return None

    def _deliver_background_job_to_robot(
        self,
        *,
        item_id: str,
        command: str,
        result: dict,
        robot_job_context: dict | None,
        pending_reply_id: str = "",
        reply_ticket_id: str = "",
        message_override: str = "",
    ) -> bool:
        if not robot_job_context:
            return False
        try:
            from app.services.agent.integrations.hooks import (
                enqueue_integration_background_job_result,
            )

            message = message_override or self._format_background_job_robot_message(
                command, result, reply_ticket_id=reply_ticket_id,
            )
            queued = enqueue_integration_background_job_result(
                integration_id=robot_job_context.get("robot_id", ""),
                item_id=item_id,
                sender_key=robot_job_context.get("sender_key", ""),
                reply_target=robot_job_context.get("reply_target") or {},
                conversation_key=robot_job_context.get("conversation_key", ""),
                conversation_generation=int(
                    robot_job_context.get("conversation_generation") or 0
                ),
                reply_requires_awake=False,
                reply_ticket_id=reply_ticket_id,
                message=message,
            )
            if not queued:
                debug_log(
                    f"[LocalMCPServer] robot background job result not queued: item={item_id}, command={command}"
                )
            return bool(queued)
        except Exception as exc:
            debug_log(
                f"[LocalMCPServer] failed to queue robot background job result: item={item_id}, error={exc}"
            )
            if pending_reply_id:
                self._clear_background_job_robot_reply(
                    robot_job_context=robot_job_context,
                    pending_reply_id=pending_reply_id,
                )
            return False

    def _clear_background_job_robot_reply(
        self,
        *,
        robot_job_context: dict | None,
        pending_reply_id: str,
    ) -> None:
        if not robot_job_context or not pending_reply_id:
            return
        try:
            from app.services.agent.integrations.hooks import (
                clear_integration_background_job_reply,
            )

            clear_integration_background_job_reply(
                integration_id=robot_job_context.get("robot_id", ""),
                conversation_key=robot_job_context.get("conversation_key", ""),
                pending_reply_id=pending_reply_id,
            )
        except Exception:
            pass

    def _deliver_background_job_to_reply_ticket(
        self,
        *,
        reply_ticket_id: str,
        command: str,
        result: dict,
    ) -> bool:
        if not reply_ticket_id:
            return False
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            message = self._format_background_job_reply_ticket_message(command, result)
            return reply_ticket_manager.deliver(reply_ticket_id, message)
        except Exception as exc:
            debug_log(
                f"[LocalMCPServer] failed to deliver background job via reply ticket: ticket={reply_ticket_id}, error={exc}"
            )
            return False

    def _format_background_job_reply_ticket_message(self, command: str, result: dict) -> str:
        output_tail = str(result.get("output_tail") or "").strip()
        if len(output_tail) > 1200:
            output_tail = output_tail[-1200:]
        if result.get("success"):
            duration = result.get("duration_seconds")
            duration_text = f"，耗时 {duration}s" if duration not in (None, "") else ""
            message = f"后台任务已完成，退出码 {result.get('exit_code', 0)}{duration_text}。"
            if output_tail:
                return f"{message}\n结果：\n{output_tail}"
            return f"{message}\n没有输出内容。"
        error = str(result.get("error") or "daemon job failed").strip()
        if len(error) > 160:
            error = f"{error[:157]}..."
        message = f"后台任务失败。原因：{error}"
        if output_tail:
            return f"{message}\n输出：\n{output_tail}"
        return message

    def _plan_reminder(self, item_id: str, reply_ticket_id: str = "") -> str:
        """Lightweight plan<->job link: remind the agent to sync its plan
        when a background job result arrives. The plan is read from the
        ticket that owns this job result (not from the item's latest
        ticket), so multi-ticket items never see a foreign plan.
        Plans whose steps are all in terminal states (completed/cancelled)
        are finished work and never reminded again."""
        if not reply_ticket_id:
            return ""
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            ticket = reply_ticket_manager.get(reply_ticket_id)
        except Exception:
            return ""
        plan = list(ticket.plan) if ticket is not None else []
        if not plan:
            return ""
        if all(
            str(entry.get("status")) in {"completed", "cancelled"}
            for entry in plan
        ):
            return ""
        lines = ["", "当前计划（请根据本结果用 update_plan 同步进度）："]
        lines.extend(
            f"  {index}. [{entry['status']}] {entry['step']}"
            for index, entry in enumerate(plan, start=1)
        )
        return "\n".join(lines)

    def _format_background_job_robot_message(self, command: str, result: dict, *, reply_ticket_id: str = "") -> str:
        if result.get("cancelled"):
            return (
                "[后台终端任务结果 - 来自 QQ 会话]\n"
                "后台任务已被主动取消（用户发起的取消，不是失败）。\n"
                "对应 plan 已由系统标记为 cancelled，不要重新执行这个任务。\n"
                "简短确认任务已停止即可。\n"
                f"命令: {command}\n"
            )
        status = "完成" if result.get("success") else "失败"
        has_plan = bool(self._plan_reminder("", reply_ticket_id))
        request_boundary = self._job_callback_task_boundary(reply_ticket_id)
        if has_plan:
            instruction = (
                "你的第一个动作必须是调用 mcp_local_update_plan，"
                "把当前步骤标为 completed 并开始下一步（或全部完成时清空 plan）。"
                "然后再根据结果回复用户。不要为中间结果发送 QQ 消息。"
                "如果不需要回复用户，最终只输出 NRN 即可。"
            )
        else:
            instruction = ""
        instruction_line = f"{instruction}\n" if instruction else ""
        return (
            "[后台终端任务结果 - 来自 QQ 会话]\n"
            f"后台任务已{status}。\n"
            f"{instruction_line}"
            f"{request_boundary}"
            f"命令: {command}\n"
            f"{self._format_job_result(result)}"
        )


    def _job_callback_task_boundary(self, reply_ticket_id: str) -> str:
        """Task background for job callbacks: surface the ticket's original
        user request as pure context so the agent itself can judge scope and
        whether a reply is wanted (e.g. an explicit no-reply). No behavioral
        instructions -- the agent decides."""
        if not reply_ticket_id:
            return ""
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            ticket = reply_ticket_manager.get(reply_ticket_id)
            request = str(getattr(ticket, "request_message", "") or "").strip() if ticket else ""
            if not request:
                return ""
            if len(request) > 200:
                request = request[:200] + "…"
            return f"任务背景：本次后台任务源自用户请求：「{request}」\n"
        except Exception:
            return ""

    def _format_background_job_feedback(self, result: dict, *, reply_ticket_id: str = "") -> str:
        if result.get("cancelled"):
            return (
                "[Background terminal job cancelled]\n"
                "后台任务已被取消（这是主动取消，不是失败）。\n"
                "对应 plan 已由系统标记为 cancelled，不要重试这个任务。\n"
                "简短确认任务已停止即可。\n"
                f"command: {result.get('command', '')}"
            )
        has_plan = bool(self._plan_reminder("", reply_ticket_id))
        request_boundary = self._job_callback_task_boundary(reply_ticket_id)
        if result.get("success"):
            if has_plan:
                header = (
                    "[Background terminal job completed]\n"
                    "后台任务已完成。你的第一个动作必须是调用 mcp_local_update_plan，"
                    "把当前步骤标为 completed 并开始下一步（或全部完成时清空 plan）。"
                    "然后再根据结果回复用户。"
                    "如果此命令的结果已在之前的回复中处理过，静默结束即可，不要重复回复。"
                )
            else:
                header = (
                    "[Background terminal job completed]\n"
                    "后台任务已完成。"
                    "对照任务背景判断是否还有未完成的后续步骤：有就立即继续执行，"
                    "全部完成才回复用户；已在之前回复中处理过则静默结束。"
                )
            return f"{header}\n{request_boundary}{self._format_job_result(result)}"
        if has_plan:
            return (
                "[Background terminal job failed]\n"
                "后台任务失败。你的第一个动作必须是调用 mcp_local_update_plan，"
                "把当前步骤标为 completed 并换方法继续下一步，"
                "或确认无更多方法时用 update_plan(plan=[]) 清空计划并汇报失败。\n"
                f"Error: {result.get('error', 'daemon job failed')}\n"
                f"command: {result.get('command', '')}"
            )
        return (
            "[Background terminal job failed]\n"
            "后台任务失败。对照任务背景换方法继续，或确认无更多方法时汇报失败。\n"
            f"{request_boundary}"
            f"Error: {result.get('error', 'daemon job failed')}\n"
            f"command: {result.get('command', '')}"
        )

    def _resolve_agent_for_item(self, item_id: str):
        try:
            import uuid as _uuid

            from sqlmodel import Session

            from app.core.db import engine
            from app.models import ItemHandler
            from app.services.agent.agent import agent_manager, item_handler_context

            handler_id = item_handler_context.get_handler(str(item_id))
            if not handler_id:
                return None
            with Session(engine) as session:
                handler = session.get(ItemHandler, _uuid.UUID(str(handler_id)))
            if handler is None:
                return None
            return agent_manager.get_or_create(handler)
        except Exception as exc:
            debug_log(f"[LocalMCPServer] _resolve_agent_for_item error: {exc}")
            return None

    def _memory_scope(self, item_id: str) -> str:
        """Memory/token operations are keyed by handler id: one handler's
        items share its memories. Resolve via the live agent when possible."""
        try:
            agent = self._resolve_agent_for_item(item_id)
            handler_id = str(getattr(agent, "handler_id", "") or "").strip()
            if handler_id:
                return handler_id
            from app.services.agent.memory.scope import resolve_handler_id

            return resolve_handler_id(item_id)
        except Exception:
            return str(item_id or "")

    def _prepare_capabilities(self, args: dict) -> list:
        import json as _json

        from app.services.agent.capability_state import ensure_loaded

        item_id = str(args.get("item_id") or "").strip()
        names = [str(name).strip() for name in (args.get("names") or []) if str(name).strip()]
        if not item_id or not names:
            return [{"type": "text", "text": "Error: item_id and names are required."}]

        agent = self._resolve_agent_for_item(item_id)
        if agent is None:
            return [{"type": "text", "text": "Error: no agent context available for this item."}]

        try:
            tool_index = {}
            for tool in agent.get_tools_for_litellm():
                function = tool.get("function") if isinstance(tool, dict) else None
                if isinstance(function, dict):
                    tool_name = str(function.get("name") or "").strip()
                    if tool_name:
                        tool_index[tool_name] = tool
        except Exception:
            tool_index = {}

        guide_index = {}
        try:
            for skill in agent.get_skills():
                category = str(getattr(skill, "category", "") or "")
                if category in {"system", "persona"}:
                    continue
                action = getattr(skill, "action", None)
                prompt = str(getattr(action, "prompt", "") or "").strip() if action else ""
                if prompt:
                    guide_index[str(getattr(skill, "skill_id", "") or "")] = prompt
        except Exception:
            guide_index = {}

        loaded_tools: list[str] = []
        loaded_guides: list[str] = []
        unknown: list[str] = []
        for name in names:
            if name in tool_index:
                loaded_tools.append(name)
            elif name in guide_index:
                loaded_guides.append(name)
            else:
                unknown.append(name)

        ensure_loaded(item_id, tools=loaded_tools, guides=loaded_guides)

        parts: list[str] = []
        for name in loaded_tools:
            parts.append(
                f"Tool `{name}` is now loaded. Schema:\n{_json.dumps(tool_index[name], ensure_ascii=False)}"
            )
        for guide_id in loaded_guides:
            parts.append(f"Guide `{guide_id}` is now loaded:\n{guide_index[guide_id]}")
        if unknown:
            parts.append(
                "Unknown names (not present in the capability catalog): "
                + ", ".join(unknown)
            )
        parts.append(
            "Loaded capabilities stay available for this item; call them normally now."
        )
        return [{"type": "text", "text": "\n\n".join(parts)}]

    def _execute_command(self, args: dict) -> list:
        command = args.get("command", "")
        item_id = args.get("item_id", "")

        debug_log(f"[LocalMCPServer] _execute_command: item={item_id}, command={command}")

        if not command or not item_id:
            return [{"type": "text", "text": "Error: command and item_id required"}]

        try:
            has_handler = self._ensure_terminal_input_handler(str(item_id))
            debug_log(f"[LocalMCPServer] has_handler={has_handler}")
            if not has_handler:
                return self._terminal_unavailable_result(str(item_id))


            try:
                from app.services.agent.session import (
                    EXECUTE_COMMAND_TOOL_NAME,
                    agent_session_manager,
                )

                existing_session = agent_session_manager.get_session(str(item_id))
                if existing_session:
                    terminal_input_error = existing_session.validate_terminal_tool_input(
                        EXECUTE_COMMAND_TOOL_NAME,
                        {"item_id": str(item_id), "command": command},
                    )
                    if terminal_input_error:
                        debug_log(
                            f"[LocalMCPServer] command blocked before send: item={item_id}, command={command}"
                        )
                        return [{"type": "text", "text": terminal_input_error}]
            except Exception as guard_error:
                debug_log(f"[LocalMCPServer] terminal input guard error: {guard_error}")

            from app.services.socket_pool import InputSDK

            if not command.endswith("\n"):
                command = command + "\n"

            success = InputSDK().send(item_id, command)
            debug_log(f"[LocalMCPServer] send result: success={success}")

            return [
                {
                    "type": "text",
                    "text": (
                        f"命令已发送: {command.strip()}。终端输出会在下一轮反馈中到达，你可以继续执行其他操作或等待结果。"
                        if success
                        else self._terminal_unavailable_message(str(item_id))
                    ),
                }
            ]
        except Exception as e:
            debug_log(f"[LocalMCPServer] execute_command error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _cancel_running_job_for_item(self, item_id: str) -> dict | None:
        try:
            from app.services.agent.session import agent_session_manager

            agent_session = agent_session_manager.get_session(str(item_id))
            if not agent_session or not agent_session.has_running_terminal_job():
                return None

            _item, connection = self._get_item_daemon_context(str(item_id))
            result = connection.cancel_job_http(item_uuid=str(item_id))
            if result.get("success") or result.get("cancelled"):
                agent_session.clear_terminal_job()
            elif result.get("error") == "No running job for item":
                agent_session.clear_terminal_job()
                result["local_lock_cleared"] = True
            return result
        except Exception as exc:
            debug_log(f"[LocalMCPServer] cancel running job error for item={item_id}: {exc}")
            return {"success": False, "error": str(exc)}

    def _interrupt_command(self, args: dict) -> list:
        item_id = args.get("item_id", "")

        debug_log(f"[LocalMCPServer] _interrupt_command: item={item_id}")

        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]

        cancel_result = self._cancel_running_job_for_item(str(item_id))
        if cancel_result is not None:
            if cancel_result.get("success") or cancel_result.get("cancelled"):
                return [{"type": "text", "text": "\u540e\u53f0\u4efb\u52a1\u5df2\u4e2d\u65ad\uff0c\u7ec8\u7aef\u9501\u5df2\u91ca\u653e\u3002"}]
            if cancel_result.get("local_lock_cleared"):
                return [{"type": "text", "text": "daemon \u91cc\u6ca1\u6709\u627e\u5230\u6b63\u5728\u8fd0\u884c\u7684\u540e\u53f0\u4efb\u52a1\uff0c\u5df2\u6e05\u7406\u672c\u5730\u7ec8\u7aef\u9501\u3002"}]
            return [{"type": "text", "text": f"\u540e\u53f0\u4efb\u52a1\u4e2d\u65ad\u5931\u8d25: {cancel_result.get('error', 'unknown error')}"}]

        try:
            from app.services.socket_pool import InputSDK
            has_handler = self._ensure_terminal_input_handler(item_id)
            debug_log(f"[LocalMCPServer] interrupt has_handler={has_handler}")
            if not has_handler:
                return self._terminal_unavailable_result(str(item_id))

            success = InputSDK().send(item_id, "\x03")
            debug_log(f"[LocalMCPServer] interrupt result: success={success}")

            return [
                {
                    "type": "text",
                    "text": (
                        "Ctrl+C 已发送到终端，是否已中断需等待后续输出确认"
                        if success
                        else "Ctrl+C 发送失败"
                    ),
                }
            ]
        except Exception as e:
            debug_log(f"[LocalMCPServer] interrupt error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _read_terminal_log(self, args: dict) -> list:
        item_id = args.get("item_id", "")
        lines = args.get("lines", 64)

        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]

        try:
            from app.services.log_manager import LogManager
            # The PTY log only carries interactive-terminal content — job
            # stream output is split into {item}.jobs.log at write time and
            # reaches the agent through structured job callbacks instead.
            content = LogManager().get_last_lines(item_id, lines)

            if not content:
                return [{"type": "text", "text": f"No log found for {item_id}"}]

            if len(content) > 4000:
                content = "…[截断，仅保留最新部分]…\n" + content[-4000:]
            return [{"type": "text", "text": f"=== 终端日志 (最后 {lines} 行) ===\n{content}"}]
        except Exception as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    def _sanitize_filter_rule_name(self, name: str, existing: set[str]) -> str:
        candidate = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "").strip().lower()).strip("_")
        if not candidate:
            candidate = "agent_noise_filter"
        if candidate not in existing:
            return candidate
        suffix = 2
        while f"{candidate}_{suffix}" in existing:
            suffix += 1
        return f"{candidate}_{suffix}"

    def _coerce_filter_patterns(self, raw_patterns) -> list[str]:
        if not isinstance(raw_patterns, list):
            raise ValueError("regex_patterns must be a non-empty list of strings")
        patterns: list[str] = []
        for raw_pattern in raw_patterns:
            if not isinstance(raw_pattern, str):
                continue
            pattern = raw_pattern.strip()
            if not pattern:
                continue
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"invalid regex pattern {pattern!r}: {exc}") from exc
            if pattern not in patterns:
                patterns.append(pattern)
        if not patterns:
            raise ValueError("regex_patterns must contain at least one valid regex")
        return patterns

    def _add_terminal_input_filter_rule(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        name = str(args.get("name") or "agent_noise_filter").strip()
        action_type = str(args.get("action_type") or "block").strip().lower()
        reason = str(args.get("reason") or "").strip()
        if action_type not in {"block", "ignore", "log"}:
            return [{"type": "text", "text": "Error: action_type must be block, ignore, or log"}]
        try:
            import uuid

            from sqlmodel import Session

            from app.core.db import engine
            from app.models import Item

            try:
                item_uuid = uuid.UUID(item_id)
            except ValueError as exc:
                raise ValueError(f"invalid item_id: {item_id}") from exc

            patterns = self._coerce_filter_patterns(args.get("regex_patterns"))
            with Session(engine) as db:
                item = db.get(Item, item_uuid)
                if not item:
                    raise ValueError(f"item not found: {item_id}")

                rules = dict(item.input_filter_rules or {})
                for rule_name, rule in rules.items():
                    if not isinstance(rule, dict):
                        continue
                    existing_patterns = [
                        p for p in rule.get("regex_patterns", []) if isinstance(p, str)
                    ]
                    if (
                        rule.get("action_type", "ignore") == action_type
                        and all(pattern in existing_patterns for pattern in patterns)
                    ):
                        item.input_filter_enabled = True
                        db.add(item)
                        db.commit()
                        return [
                            {
                                "type": "text",
                                "text": (
                                    "Terminal input filter already contains this rule: "
                                    f"{rule_name}. input_filter_enabled=true"
                                ),
                            }
                        ]

                normalized_name = re.sub(
                    r"[^a-zA-Z0-9_]+",
                    "_",
                    name.lower(),
                ).strip("_")
                if normalized_name in rules and isinstance(rules[normalized_name], dict):
                    rule = dict(rules[normalized_name])
                    existing_patterns = [
                        p for p in rule.get("regex_patterns", []) if isinstance(p, str)
                    ]
                    merged_patterns = existing_patterns + [
                        pattern for pattern in patterns if pattern not in existing_patterns
                    ]
                    rule["regex_patterns"] = merged_patterns
                    rule["action_type"] = action_type
                    if reason:
                        rule["reason"] = reason
                    rules[normalized_name] = rule
                    rule_name = normalized_name
                else:
                    rule_name = self._sanitize_filter_rule_name(name, set(rules))
                    rule_payload = {
                        "regex_patterns": patterns,
                        "action_type": action_type,
                    }
                    if reason:
                        rule_payload["reason"] = reason
                    rules[rule_name] = rule_payload

                item.input_filter_enabled = True
                item.input_filter_rules = rules
                db.add(item)
                db.commit()

            return [
                {
                    "type": "text",
                    "text": (
                        "Added terminal output -> Agent input filter rule "
                        f"`{rule_name}` action={action_type}; patterns={len(patterns)}. "
                        "input_filter_enabled=true"
                    ),
                }
            ]
        except Exception as e:
            debug_log(f"[LocalMCPServer] add terminal input filter rule error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _list_terminal_input_filter_rules(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        try:
            import uuid

            from sqlmodel import Session

            from app.core.db import engine
            from app.models import Item

            try:
                item_uuid = uuid.UUID(item_id)
            except ValueError as exc:
                raise ValueError(f"invalid item_id: {item_id}") from exc

            with Session(engine) as db:
                item = db.get(Item, item_uuid)
                if not item:
                    raise ValueError(f"item not found: {item_id}")
                enabled = bool(item.input_filter_enabled)
                rules = item.input_filter_rules or {}

            if not rules:
                return [
                    {
                        "type": "text",
                        "text": f"Terminal input filter enabled={enabled}; no rules.",
                    }
                ]

            lines = [f"Terminal input filter enabled={enabled}; rules={len(rules)}"]
            for rule_name, rule in rules.items():
                if not isinstance(rule, dict):
                    continue
                patterns = rule.get("regex_patterns", [])
                action_type = rule.get("action_type", "ignore")
                lines.append(
                    f"- {rule_name}: action={action_type}, patterns={len(patterns)}"
                )
                for pattern in patterns[:5]:
                    lines.append(f"  - {pattern}")
                if len(patterns) > 5:
                    lines.append(f"  - ... {len(patterns) - 5} more")
            return [{"type": "text", "text": "\n".join(lines)}]
        except Exception as e:
            debug_log(f"[LocalMCPServer] list terminal input filter rules error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _format_terminal_filter_section(
        self,
        title: str,
        *,
        enabled: bool,
        rules: dict,
    ) -> list[str]:
        if not rules:
            return [f"{title} enabled={enabled}; no rules."]

        lines = [f"{title} enabled={enabled}; rules={len(rules)}"]
        for rule_name, rule in rules.items():
            if not isinstance(rule, dict):
                lines.append(f"- {rule_name}: invalid rule payload")
                continue
            patterns = [
                pattern
                for pattern in rule.get("regex_patterns", [])
                if isinstance(pattern, str)
            ]
            action_type = rule.get("action_type", "ignore")
            reason = str(rule.get("reason") or "").strip()
            action = rule.get("action") if isinstance(rule.get("action"), dict) else {}
            replace_rules = action.get("replace_rules") if isinstance(action, dict) else None
            details = [f"action={action_type}", f"patterns={len(patterns)}"]
            if isinstance(replace_rules, dict):
                details.append(f"replace_rules={len(replace_rules)}")
            if reason:
                details.append(f"reason={reason}")
            lines.append(f"- {rule_name}: {', '.join(details)}")
            for pattern in patterns[:5]:
                lines.append(f"  - {pattern}")
            if len(patterns) > 5:
                lines.append(f"  - ... {len(patterns) - 5} more")
        return lines

    def _list_terminal_filter_rules(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        try:
            import uuid

            from sqlmodel import Session

            from app.core.db import engine
            from app.models import Item

            try:
                item_uuid = uuid.UUID(item_id)
            except ValueError as exc:
                raise ValueError(f"invalid item_id: {item_id}") from exc

            with Session(engine) as db:
                item = db.get(Item, item_uuid)
                if not item:
                    raise ValueError(f"item not found: {item_id}")
                input_enabled = bool(item.input_filter_enabled)
                input_rules = dict(item.input_filter_rules or {})
                output_enabled = bool(item.output_filter_enabled)
                output_rules = dict(item.output_filter_rules or {})

            lines = ["Terminal filter rules for current item:"]
            lines.extend(
                self._format_terminal_filter_section(
                    "Input filter (terminal output -> Agent)",
                    enabled=input_enabled,
                    rules=input_rules,
                )
            )
            lines.append("")
            lines.extend(
                self._format_terminal_filter_section(
                    "Output filter (Agent command -> terminal)",
                    enabled=output_enabled,
                    rules=output_rules,
                )
            )
            return [{"type": "text", "text": "\n".join(lines)}]
        except Exception as e:
            debug_log(f"[LocalMCPServer] list terminal filter rules error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _delete_terminal_input_filter_rule(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        name = str(args.get("name") or "").strip()
        disable_when_empty = bool(args.get("disable_when_empty", True))
        if not name:
            return [{"type": "text", "text": "Error: name is required"}]
        try:
            import uuid

            from sqlmodel import Session

            from app.core.db import engine
            from app.models import Item

            try:
                item_uuid = uuid.UUID(item_id)
            except ValueError as exc:
                raise ValueError(f"invalid item_id: {item_id}") from exc

            with Session(engine) as db:
                item = db.get(Item, item_uuid)
                if not item:
                    raise ValueError(f"item not found: {item_id}")

                rules = dict(item.input_filter_rules or {})
                if name not in rules:
                    existing = ", ".join(sorted(rules)) or "none"
                    return [
                        {
                            "type": "text",
                            "text": (
                                f"Terminal input filter rule not found: {name}. "
                                f"Existing rules: {existing}"
                            ),
                        }
                    ]

                rules.pop(name, None)
                item.input_filter_rules = rules
                if disable_when_empty and not rules:
                    item.input_filter_enabled = False
                enabled = bool(item.input_filter_enabled)
                db.add(item)
                db.commit()

            return [
                {
                    "type": "text",
                    "text": (
                        f"Deleted terminal output -> Agent input filter rule `{name}`. "
                        f"remaining={len(rules)} input_filter_enabled={enabled}"
                    ),
                }
            ]
        except Exception as e:
            debug_log(f"[LocalMCPServer] delete terminal input filter rule error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _clear_terminal_input_filter_rules(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        disable = bool(args.get("disable", True))
        try:
            import uuid

            from sqlmodel import Session

            from app.core.db import engine
            from app.models import Item

            try:
                item_uuid = uuid.UUID(item_id)
            except ValueError as exc:
                raise ValueError(f"invalid item_id: {item_id}") from exc

            with Session(engine) as db:
                item = db.get(Item, item_uuid)
                if not item:
                    raise ValueError(f"item not found: {item_id}")

                removed = len(item.input_filter_rules or {})
                item.input_filter_rules = {}
                if disable:
                    item.input_filter_enabled = False
                db.add(item)
                db.commit()

            return [
                {
                    "type": "text",
                    "text": (
                        "Cleared terminal output -> Agent input filter rules. "
                        f"removed={removed} input_filter_enabled={not disable}"
                    ),
                }
            ]
        except Exception as e:
            debug_log(f"[LocalMCPServer] clear terminal input filter rules error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    def _list_scheduled_tasks(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]
        try:
            from app.services.agent.scheduled_tasks import (
                format_scheduled_tasks,
                list_scheduled_tasks,
            )

            tasks = list_scheduled_tasks(item_id)
            return [
                {
                    "type": "text",
                    "text": "Scheduled tasks:\n" + format_scheduled_tasks(tasks),
                }
            ]
        except Exception as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    def _write_scheduled_task(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]
        try:
            from app.services.agent.scheduled_tasks import write_scheduled_task

            task = write_scheduled_task(
                item_id,
                task_id=str(args.get("task_id") or ""),
                name=str(args.get("name") or ""),
                instruction=str(args.get("instruction") or ""),
                schedule_type=str(args.get("schedule_type") or ""),
                run_at=str(args.get("run_at") or ""),
                interval_seconds=int(args.get("interval_seconds") or 0),
                time_of_day=str(args.get("time_of_day") or ""),
                timezone_name=str(args.get("timezone") or "Asia/Shanghai"),
                enabled=bool(args.get("enabled", True)),
            )
            return [
                {
                    "type": "text",
                    "text": (
                        f"Scheduled task saved: id={task['id']} name={task['name']!r} "
                        f"enabled={task['enabled']} next_run_at={task.get('next_run_at') or 'none'}"
                    ),
                }
            ]
        except KeyError as e:
            return [{"type": "text", "text": f"Error: {e.args[0]}"}]
        except Exception as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    def _delete_scheduled_task(self, args: dict) -> list:
        item_id = str(args.get("item_id") or "").strip()
        task_id = str(args.get("task_id") or "").strip()
        if not item_id or not task_id:
            return [{"type": "text", "text": "Error: item_id and task_id required"}]
        try:
            from app.services.agent.scheduled_tasks import delete_scheduled_task

            result = delete_scheduled_task(
                item_id,
                task_id=task_id,
                reason=str(args.get("reason") or "Agent decision"),
            )
            if not result.get("count"):
                return [{"type": "text", "text": "Error: scheduled task not found"}]
            return [
                {
                    "type": "text",
                    "text": f"Scheduled task deleted: id={task_id}",
                }
            ]
        except Exception as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    def _save_memory(self, args: dict) -> list:
        content = args.get("content", "")
        memory_type = str(args.get("memory_type", "fact") or "fact")
        item_id = args.get("item_id", "")

        if not content or not item_id:
            return [{"type": "text", "text": "Error: content and item_id required"}]
        if memory_type not in {"fact", "preference", "error", "context"}:
            return [{"type": "text", "text": f"Error: invalid memory_type: {memory_type}"}]

        try:
            from app.services.agent.memory.vector_store import vector_store
            from app.services.agent.prompts import policy as memory_policy

            default_ttl_days = memory_policy.resolve_memory_ttl_days(str(memory_type))
            raw_ttl_days = args.get("ttl_days")
            if default_ttl_days is None:
                ttl_days = None
            else:
                ttl_days = default_ttl_days if raw_ttl_days in (None, "") else int(raw_ttl_days)
                ttl_days = max(1, min(3650, ttl_days))

            metadata: dict[str, Any] = {
                "type": "agent_saved",
                "source": "local_agent_saved",
                "verified": True,
                "content_hash": memory_policy._build_content_hash(str(content)),
                "updated_at": datetime.now().isoformat(),
                "source_item_id": str(item_id),
            }
            memory_key = memory_policy.infer_memory_key(str(content), str(memory_type))
            if memory_key:
                metadata["memory_key"] = memory_key
            if memory_type == "error":
                metadata["status"] = "active"

            memory_id = vector_store.add_memory(
                handler_id=self._memory_scope(item_id),
                content=content,
                memory_type=memory_type,
                metadata=metadata,
                ttl_days=ttl_days,
                allow_duplicate=True,
            )
            if not memory_id:
                return [{"type": "text", "text": "Memory was not saved."}]
            lifetime = "永久" if ttl_days is None else f"{ttl_days}天"
            return [{"type": "text", "text": f"✓ 记忆已保存 (ID: {memory_id[:8]}..., 类型: {memory_type}, 有效期: {lifetime})"}]
        except Exception as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    def _compress_memories(self, args: dict) -> list:
        raw_ids = args.get("memory_ids")
        if not isinstance(raw_ids, list):
            return [{"type": "text", "text": "Error: memory_ids must be a list"}]
        requested_ids = [
            value for value in (str(item or "").strip() for item in raw_ids) if value
        ]
        if len(set(requested_ids)) < 2:
            return [{"type": "text", "text": "Error: 至少提供 2 个不同的记忆 ID"}]
        content = str(args.get("content") or "").strip()
        if not content:
            return [{"type": "text", "text": "Error: content required"}]
        item_id = str(args.get("item_id") or "").strip()
        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]
        memory_type = str(args.get("memory_type") or "").strip()
        if memory_type and memory_type not in {"fact", "preference", "error", "context"}:
            return [{"type": "text", "text": f"Error: invalid memory_type: {memory_type}"}]

        try:
            from app.services.agent.memory.vector_store import vector_store
            from app.services.agent.prompts import policy as memory_policy

            all_memories = vector_store.get_all_memories(self._memory_scope(item_id))
            sources: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            unmatched: list[str] = []
            for requested in requested_ids:
                match: dict[str, Any] | None = None
                for memory in all_memories:
                    memory_id = str(memory.get("id") or "")
                    if not memory_id or memory_id in seen_ids:
                        continue
                    if memory_id == requested or memory_id.startswith(requested):
                        match = memory
                        break
                if match is None:
                    unmatched.append(requested)
                    continue
                seen_ids.add(str(match.get("id") or ""))
                sources.append(match)
            if unmatched:
                return [{"type": "text", "text": f"Error: 未找到记忆 ID: {', '.join(unmatched)}"}]
            if len(sources) < 2:
                return [{"type": "text", "text": "Error: 至少提供 2 个不同的记忆 ID"}]

            if not memory_type:
                type_counts: dict[str, int] = {}
                for memory in sources:
                    source_type = str((memory.get("metadata") or {}).get("memory_type") or "fact")
                    type_counts[source_type] = type_counts.get(source_type, 0) + 1
                memory_type = max(
                    ("preference", "fact", "context", "error"),
                    key=lambda candidate: type_counts.get(candidate, 0),
                )

            default_ttl_days = memory_policy.resolve_memory_ttl_days(str(memory_type))
            raw_ttl_days = args.get("ttl_days")
            if default_ttl_days is None:
                ttl_days = None
            else:
                ttl_days = default_ttl_days if raw_ttl_days in (None, "") else int(raw_ttl_days)
                ttl_days = max(1, min(3650, ttl_days))

            metadata: dict[str, Any] = {
                "type": "agent_saved",
                "source": "local_agent_compress",
                "verified": True,
                "content_hash": memory_policy._build_content_hash(content),
                "updated_at": datetime.now().isoformat(),
                "source_item_id": item_id,
            }
            memory_key = memory_policy.infer_memory_key(content, str(memory_type))
            if memory_key:
                metadata["memory_key"] = memory_key
            if memory_type == "error":
                metadata["status"] = "active"

            new_memory_id = vector_store.add_memory(
                handler_id=self._memory_scope(item_id),
                content=content,
                memory_type=memory_type,
                metadata=metadata,
                ttl_days=ttl_days,
                allow_duplicate=True,
                run_maintenance=False,
            )
            if not new_memory_id:
                return [{"type": "text", "text": "压缩记忆写入失败，旧记忆未删除。"}]
            deleted = 0
            for memory in sources:
                try:
                    if vector_store.delete_memory(str(memory.get("id") or "")):
                        deleted += 1
                except Exception:
                    continue
            return [{"type": "text", "text": f"✓ 已将 {len(sources)} 条记忆压缩为 1 条 (新 ID: {str(new_memory_id)[:8]}...)，删除旧记忆 {deleted} 条"}]
        except Exception as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    def _recall_memory(self, args: dict) -> list:
        query = args.get("query", "")
        try:
            n_results = int(args.get("n_results", 0) or 0)
        except (TypeError, ValueError):
            n_results = 0
        memory_type = args.get("memory_type")
        item_id = args.get("item_id", "")

        if not query or not item_id:
            return [{"type": "text", "text": "Error: query and item_id required"}]

        try:
            from app.services.agent.memory.vector_store import vector_store
            scope = self._memory_scope(item_id)
            if n_results <= 0:
                n_results = max(1, len(vector_store.get_all_memories(scope)))
            results = vector_store.search_memories(
                handler_id=scope,
                query=query,
                n_results=n_results,
                memory_type=memory_type
            )
            results = [
                memory
                for memory in results
                if str((memory.get("metadata") or {}).get("memory_type") or "fact")
                in {"fact", "preference", "error", "context"}
            ]

            if not results:
                return [{"type": "text", "text": "未找到相关记忆"}]

            from app.services.agent.prompts.builder import LONG_TERM_MEMORY_FRAMING

            lines = [f"相关长期记忆{LONG_TERM_MEMORY_FRAMING}："]
            for memory in results:
                metadata = memory.get("metadata") or {}
                sender = str(
                    metadata.get("speaker")
                    or metadata.get("speaker_label")
                    or metadata.get("sender")
                    or ""
                ).strip()
                content = str(memory.get("content") or "").strip()
                if not content:
                    continue
                prefix = f"{sender}: " if sender and not content.startswith(sender) else ""
                if len(content) > 400:
                    content = content[:397] + "..."
                lines.append(f"- {prefix}{content}")

            return [{"type": "text", "text": "\n".join(lines)}]
        except Exception as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    def _list_memories(self, args: dict) -> list:
        memory_type = args.get("memory_type")
        item_id = args.get("item_id", "")

        if not item_id:
            return [{"type": "text", "text": "Error: item_id required"}]

        try:
            from app.services.agent.memory.vector_store import vector_store
            memories = vector_store.get_all_memories(
                self._memory_scope(item_id), memory_type=memory_type
            )
            memories = [
                memory
                for memory in memories
                if str((memory.get("metadata") or {}).get("memory_type") or "fact")
                in {"fact", "preference", "error", "context"}
            ]

            if not memories:
                return [{"type": "text", "text": "暂无记忆"}]

            lines = [f"=== 共 {len(memories)} 条记忆 ==="]
            for m in memories:
                m_type = m.get("metadata", {}).get("memory_type", "unknown")
                m_id = m.get("id", "")[:8]
                lines.append(f"[{m_id}] ({m_type}) {m['content'][:50]}...")

            return [{"type": "text", "text": "\n".join(lines)}]
        except Exception as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    def _delete_memory(self, args: dict) -> list:
        memory_id = args.get("memory_id", "")
        item_id = args.get("item_id", "")

        if not memory_id or not item_id:
            return [{"type": "text", "text": "Error: memory_id and item_id required"}]

        try:
            from app.services.agent.memory.vector_store import vector_store
            success = vector_store.delete_memory(memory_id)
            if success:
                return [{"type": "text", "text": f"✓ 记忆已删除 (ID: {memory_id[:8]}...)"}]
            else:
                return [{"type": "text", "text": "记忆不存在或删除失败"}]
        except Exception as e:
            return [{"type": "text", "text": f"Error: {e}"}]

    def list_tools(self) -> list:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
                "skip_memory": t.get("skip_memory", False)
            }
            for t in self._tools.values()
        ]

    def call_tool(self, name: str, args: dict) -> list:
        debug_log(f"[LocalMCPServer] call_tool: name={name}, args={args}")
        if name not in self._tools:
            debug_log(f"[LocalMCPServer] Tool '{name}' not found, available: {list(self._tools.keys())}")
            return [{"type": "text", "text": f"Tool '{name}' not found"}]
        try:
            result = self._tools[name]["handler"](args)
            debug_log(f"[LocalMCPServer] Tool '{name}' result: {result}")
            return result
        except Exception as e:
            debug_log(f"[LocalMCPServer] Tool '{name}' error: {e}")
            return [{"type": "text", "text": f"Error: {e}"}]

    async def run(self):
        logger.info("[LocalMCPServer] Starting stdio server")
        loop = asyncio.get_running_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                request = json.loads(line.strip())
                response = await self._handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                print(json.dumps({"error": f"Invalid JSON: {e}"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)

    async def _handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        if method == "initialize":
            return {"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "TermPaws-local", "version": "1.0.0"}
            }}

        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": self.list_tools()}}

        if method == "tools/call":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"content": self.call_tool(params.get("name", ""), params.get("arguments", {}))}}

        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}


local_mcp_server = LocalMCPServer()


def main():
    asyncio.run(local_mcp_server.run())


if __name__ == "__main__":
    main()
