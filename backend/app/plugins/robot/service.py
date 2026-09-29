from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any

from fastapi import HTTPException
from sqlmodel import Session, select

from app.core.config import settings
from app.core.db import engine
from app.models import Item, Robot, RobotItem, User
from app.plugins.robot.conversation_memory import robot_conversation_memory
from app.services.agent.chat_runtime import ChatResponseResult, collect_chat_response
from app.services.agent.input_merge_buffer import (
    SOURCE_QQ_MESSAGE,
    MergeBufferEntry,
    input_merge_buffer,
)

from .contracts import RobotDispatchResponse, RobotInboundMessage, RobotReplyTarget
from .debug_log import preview_text, record_robot_event
from .memory_scope import (
    memory_content_is_question_like,
    memory_scope_rank,
    speaker_global_key_from_context,
)
from .message_chunks import is_group_reply_target, split_robot_message_for_target
from .platforms import (
    DEFAULT_REPLY_CONTEXT_WINDOW_SECONDS,
    get_robot_platform,
    get_robot_runtime_config,
    normalize_robot_platform_id,
)

logger = logging.getLogger(__name__)

CONVERSATION_TTL = timedelta(hours=6)
CONVERSATION_PROCESSING_MIN_TIMEOUT_SECONDS = 120
CONVERSATION_PROCESSING_MAX_TIMEOUT_SECONDS = 600
DISPATCH_WORKER_MAX_COUNT = 8
PENDING_CHAT_QUEUE_LIMIT = 5
BACKGROUND_JOB_ROUTE_KEY = "background_job"
PENDING_DIRECT_WAKE_TRIGGER_REASONS = frozenset({
    "mention_bot",
    "reply_to_bot",
    "private_chat",
})
USER_INSTRUCTION_TRIGGER_REASONS = frozenset({
    "mention_bot",
    "reply_to_bot",
    "private_chat",
    "active_chat_window",
})
RECENT_LIVE_CONTEXT_ACTIVE_LINES = 6
RECENT_LIVE_CONTEXT_BASE_LINES = 4
RECENT_LIVE_CONTEXT_EXPANDED_LINES = 12
RECENT_LIVE_CONTEXT_LINES = RECENT_LIVE_CONTEXT_EXPANDED_LINES
# Action frame placed immediately before [Current QQ message] — the highest
# attention position. It makes first-turn planning the default opening move
# for multi-step work so task chains do not stall at turn one.
_TURN_ACTION_FRAME = (
    "[本轮行动框架]\n"
    "- 如果 [Current QQ message] 是需要多步操作的任务（安装/搭建/启动/停止/修改/连续操作）："
    "第一个动作必须是 mcp_local_update_plan 列出步骤，然后立刻开始执行第一步。"
    "禁止先回复、禁止先调查、禁止反问确认。\n"
    "- 如果它只是闲聊或单步问答：直接回答，不要 plan。"
)
DEFAULT_MAX_MESSAGE_LENGTH = 1200
REPLY_MESSAGE_TYPE_PRIVATE = "private"
REPLY_MESSAGE_TYPE_GROUP = "group"
REPLY_MESSAGE_TYPE_CHANNEL = "channel"
REPLY_MESSAGE_TYPE_COMMAND = "command"
REPLY_MESSAGE_TYPE_MENTION = "mention"
DEFAULT_REPLY_MESSAGE_TYPES = frozenset(
    {
        REPLY_MESSAGE_TYPE_PRIVATE,
        REPLY_MESSAGE_TYPE_COMMAND,
        REPLY_MESSAGE_TYPE_MENTION,
    }
)
ALLOWED_REPLY_MESSAGE_TYPES = frozenset(
    {
        REPLY_MESSAGE_TYPE_PRIVATE,
        REPLY_MESSAGE_TYPE_GROUP,
        REPLY_MESSAGE_TYPE_CHANNEL,
        REPLY_MESSAGE_TYPE_COMMAND,
        REPLY_MESSAGE_TYPE_MENTION,
    }
)
REPLY_MESSAGE_TYPE_DISABLED_REASON = "reply_message_type_disabled"
MENTION_MATCH_MODE_BOT = "bot"
DEFAULT_MENTION_MATCH_MODE = MENTION_MATCH_MODE_BOT
QQ_AGENT_IGNORED_CQ_TYPES = frozenset({"image"})
CQ_CODE_PATTERN = re.compile(r"\[CQ:([A-Za-z0-9_]+)(?:,[^\]]*)?\]")
ALLOWED_MENTION_MATCH_MODES = frozenset(
    {
        MENTION_MATCH_MODE_BOT,
    }
)


@dataclass
class ConversationState:
    item_id: uuid.UUID
    updated_at: datetime


@dataclass
class RobotConversationController:
    generation: int
    updated_at: datetime
    expires_at: datetime | None = None
    processing_expires_at: datetime | None = None
    sleeping: bool = False
    processing: bool = False


@dataclass(frozen=True)
class RobotConversationGate:
    active: bool
    generation: int
    sleeping: bool = False
    processing: bool = False


@dataclass
class ResolvedRobotBinding:
    binding: RobotItem
    item: Item
    route_key: str


@dataclass
class RobotCommand:
    mode: str
    target: str | None
    text: str


@dataclass(frozen=True)
class PendingRobotChatInput:
    item_id: uuid.UUID
    route_key: str
    message_text: str
    sender_key: str
    sender_label: str
    trigger_reason: str
    reply_target: RobotReplyTarget
    enqueued_at: datetime
    # 入站时已落盘缓存的图片（公开路径），跟进轮次要还原给 agent
    image_urls: tuple[str, ...] = ()
    # True once the entry has been written to long-term memory by a first
    # drain; requeued copies must not be persisted again.


@dataclass(frozen=True)
class PendingRobotTaskReply:
    pending_id: str
    item_id: uuid.UUID
    route_key: str
    command: str
    sender_key: str
    sender_label: str
    trigger_reason: str
    reply_target: RobotReplyTarget
    enqueued_at: datetime


@dataclass(frozen=True)
class QueuedRobotChatJob:
    robot_id: uuid.UUID
    robot_owner_id: uuid.UUID
    item_id: uuid.UUID
    route_key: str
    message: str
    sender_key: str
    reply_target: RobotReplyTarget
    conversation_key: str
    enqueued_at: datetime
    direct_reply_trigger: bool = False
    reply_context_active: bool = False
    conversation_generation: int = 0
    reply_requires_awake: bool = False
    message_text: str = ""
    trigger_reason: str = ""
    inbound_message: RobotInboundMessage | None = None
    reply_ticket_id: str = ""
    pending_reply_id: str = ""
    job_id: str = ""
    image_urls: list[str] = field(default_factory=list)
    # 仅图识开关开启时非空：喂给多模态模型；image_urls 是展示用（始终缓存）
    agent_image_urls: list[str] = field(default_factory=list)
    # Pending inputs drained into this job's batch message. Kept so a dispatch
    # skipped by the generation guard can return them to the merge buffer
    # instead of dropping them silently.
    pending_entries: tuple[PendingRobotChatInput, ...] = ()


class RobotServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RobotService:
    def __init__(self) -> None:
        self._conversation_routes: dict[tuple[str, str], ConversationState] = {}
        self._conversation_controllers: dict[tuple[str, str], RobotConversationController] = {}
        self._pending_task_replies: dict[tuple[str, str], list[PendingRobotTaskReply]] = {}
        self._lock = threading.RLock()
        self._dispatch_queue: queue.Queue[QueuedRobotChatJob] = queue.Queue(
            maxsize=max(1, settings.ROBOT_BACKEND_DISPATCH_QUEUE_SIZE)
        )
        self._active_dispatch_jobs: dict[str, tuple[float, QueuedRobotChatJob]] = {}
        self._dispatch_workers_started = False
        self._dispatch_worker_count = 0
        self._dispatch_worker_lock = threading.Lock()

    def dispatch_queue_snapshot(self) -> dict[str, int]:
        return {
            "size": self._dispatch_queue.qsize(),
            "max_size": self._dispatch_queue.maxsize,
            "workers": self._dispatch_worker_count,
        }

    def _ensure_dispatch_workers(self) -> None:
        if self._dispatch_workers_started:
            return
        with self._dispatch_worker_lock:
            if self._dispatch_workers_started:
                return
            worker_count = max(2, min(settings.ROBOT_BACKEND_DISPATCH_WORKERS, 8))
            for index in range(worker_count):
                worker = threading.Thread(
                    target=self._dispatch_worker_loop,
                    name=f"TermPaws-backend-robot-dispatch-{index}",
                    daemon=True,
                )
                worker.start()
            self._dispatch_worker_count = worker_count
            self._dispatch_workers_started = True

    def _enqueue_chat_job(self, job: QueuedRobotChatJob) -> bool:
        self._ensure_dispatch_workers()
        if not job.job_id:
            job = replace(job, job_id=uuid.uuid4().hex)
        try:
            self._dispatch_queue.put_nowait(job)
        except queue.Full:
            return False
        return True

    def enqueue_background_job_result(
        self,
        *,
        robot_id: uuid.UUID | str,
        item_id: uuid.UUID | str,
        sender_key: str,
        reply_target: RobotReplyTarget | dict,
        conversation_key: str,
        conversation_generation: int = 0,
        reply_requires_awake: bool = False,
        message: str,
        pending_reply_id: str = "",
        reply_ticket_id: str = "",
    ) -> bool:
        if not message.strip():
            return False
        try:
            parsed_robot_id = (
                robot_id if isinstance(robot_id, uuid.UUID) else uuid.UUID(str(robot_id))
            )
            parsed_item_id = (
                item_id if isinstance(item_id, uuid.UUID) else uuid.UUID(str(item_id))
            )
        except (TypeError, ValueError):
            return False

        try:
            target = (
                reply_target
                if isinstance(reply_target, RobotReplyTarget)
                else RobotReplyTarget.model_validate(reply_target)
            )
        except Exception:
            return False
        resolved_conversation_key = conversation_key or self._conversation_key_from_target(
            target,
            sender_key,
        )
        if not resolved_conversation_key:
            return False

        with Session(engine) as session:
            robot = session.get(Robot, parsed_robot_id)
            item = session.get(Item, parsed_item_id)
            if robot is None or item is None:
                return False
            if not robot.is_enabled:
                return False
            job = QueuedRobotChatJob(
                robot_id=robot.id,
                robot_owner_id=robot.owner_id,
                item_id=item.id,
                route_key=BACKGROUND_JOB_ROUTE_KEY,
                message=message,
                sender_key=sender_key,
                reply_target=target.model_copy(deep=True),
                conversation_key=resolved_conversation_key,
                direct_reply_trigger=True,
                reply_context_active=False,
                conversation_generation=conversation_generation,
                reply_requires_awake=reply_requires_awake,
                enqueued_at=self._now(),
                reply_ticket_id=str(reply_ticket_id or ""),
                pending_reply_id=str(pending_reply_id or ""),
            )

        queued = self._enqueue_chat_job(job)
        record_robot_event(
            str(parsed_robot_id),
            direction="backend_queue",
            event="background_job_result_queued" if queued else "background_job_result_dropped",
            status="ok" if queued else "ignored",
            message=preview_text(message),
            payload={
                "item_id": str(parsed_item_id),
                "conversation": resolved_conversation_key,
                "queue": self.dispatch_queue_snapshot(),
            },
        )
        return queued

    def register_background_job_reply(
        self,
        *,
        robot_id: uuid.UUID | str,
        item_id: uuid.UUID | str,
        sender_key: str,
        reply_target: RobotReplyTarget | dict,
        conversation_key: str,
        conversation_generation: int = 0,
        command: str,
    ) -> str | None:
        if not command.strip():
            return None
        try:
            parsed_robot_id = (
                robot_id if isinstance(robot_id, uuid.UUID) else uuid.UUID(str(robot_id))
            )
            parsed_item_id = (
                item_id if isinstance(item_id, uuid.UUID) else uuid.UUID(str(item_id))
            )
        except (TypeError, ValueError):
            return None

        try:
            target = (
                reply_target
                if isinstance(reply_target, RobotReplyTarget)
                else RobotReplyTarget.model_validate(reply_target)
            )
        except Exception:
            return None

        resolved_conversation_key = conversation_key or self._conversation_key_from_target(
            target,
            sender_key,
        )
        if not resolved_conversation_key:
            return None

        with Session(engine) as session:
            robot = session.get(Robot, parsed_robot_id)
            item = session.get(Item, parsed_item_id)
            if robot is None or item is None or not robot.is_enabled:
                return None
            route_key = "background_job"

        now = self._now()
        pending_id = uuid.uuid4().hex
        pending = PendingRobotTaskReply(
            pending_id=pending_id,
            item_id=parsed_item_id,
            route_key=route_key,
            command=command,
            sender_key=sender_key,
            sender_label=self._sender_label_from_reply_target(target, sender_key),
            trigger_reason="background_job",
            reply_target=target.model_copy(deep=True),
            enqueued_at=now,
        )
        key = self._pending_chat_key(parsed_robot_id, resolved_conversation_key)
        with self._lock:
            self._prune_conversation_controllers_locked(now)
            controller = self._get_or_create_controller_locked(
                parsed_robot_id,
                resolved_conversation_key,
                now,
            )
            controller.updated_at = now
            queue_items = list(self._pending_task_replies.get(key) or [])
            queue_items.append(pending)
            self._pending_task_replies[key] = queue_items[-PENDING_CHAT_QUEUE_LIMIT:]

        record_robot_event(
            str(parsed_robot_id),
            direction="backend_queue",
            event="background_job_reply_registered",
            status="ok",
            message=preview_text(command),
            payload={
                "item_id": str(parsed_item_id),
                "conversation": resolved_conversation_key,
                "pending_reply_id": pending_id,
                "generation": conversation_generation,
            },
        )
        return pending_id

    def _pop_pending_task_replies(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> list[PendingRobotTaskReply]:
        key = self._pending_chat_key(robot_id, conversation_key)
        with self._lock:
            return list(self._pending_task_replies.pop(key, None) or [])

    def clear_background_job_reply(
        self,
        *,
        robot_id: uuid.UUID | str,
        conversation_key: str,
        pending_reply_id: str,
    ) -> None:
        if not pending_reply_id or not conversation_key:
            return
        key = self._pending_chat_key(robot_id, conversation_key)
        with self._lock:
            queue_items = list(self._pending_task_replies.get(key) or [])
            if not queue_items:
                return
            kept = [entry for entry in queue_items if entry.pending_id != pending_reply_id]
            if kept:
                self._pending_task_replies[key] = kept
            else:
                self._pending_task_replies.pop(key, None)

    def _dispatch_worker_loop(self) -> None:
        while True:
            job = self._dispatch_queue.get()
            tracking_key = job.job_id or uuid.uuid4().hex
            with self._lock:
                self._active_dispatch_jobs[tracking_key] = (time.monotonic(), job)
            try:
                self._process_chat_job(job)
            except Exception:
                logger.exception(
                    "[RobotService] Robot dispatch worker failed for robot=%s item=%s",
                    job.robot_id,
                    job.item_id,
                )
                try:
                    self._record_and_send_job_error(
                        job,
                        "任务处理异常，已放弃重试。请重新发一次。",
                    )
                except Exception:
                    logger.debug(
                        "[RobotService] Failed to report worker failure for robot %s",
                        job.robot_id,
                    )
            finally:
                with self._lock:
                    self._active_dispatch_jobs.pop(tracking_key, None)
                try:
                    from app.services.agent.mcp.local_server import (
                        flush_all_job_results_for_item,
                    )

                    flush_all_job_results_for_item(str(job.item_id))
                except Exception:
                    logger.debug(
                        "[RobotService] Failed to flush buffered job results for item=%s",
                        job.item_id,
                    )
                self._dispatch_queue.task_done()

    def _reattach_pending_jobs(self, ticket, agent_session) -> bool:
        """重启后接管票据上的后台 job：向 daemon 重新拉取结果。

        已完成 → 复用正常投递链路注入结果输入（驱动计划前进）；
        仍在跑 → 等下一轮 sweep；查不到 → 注入失败让计划换路继续。
        返回 True 表示本轮已处理（无需再发通用 resume）。
        """
        from app.services.agent.mcp.local_server import (
            flush_background_job_results_for_entries,
            local_mcp_server,
        )
        from app.services.agent.reply_ticket import reply_ticket_manager

        item_id = str(ticket.item_id or "")
        try:
            _item, connection = local_mcp_server._get_item_daemon_context(item_id)
        except Exception:
            logger.exception(
                "[RobotService] Failed to get daemon context for job reattach: item=%s",
                item_id,
            )
            return False
        if connection is None:
            return False

        robot_job_context = None
        if ticket.robot_id and ticket.reply_target:
            robot_job_context = {
                "robot_id": str(ticket.robot_id),
                "sender_key": str(ticket.sender_key or ""),
                "reply_target": ticket.reply_target,
                "conversation_key": str(ticket.conversation_key or ""),
                "conversation_generation": int(ticket.conversation_generation or 0),
            }

        entries = []
        for job_id in list(ticket.pending_job_ids):
            try:
                poll = connection.get_job_result_http(item_uuid=item_id, job_id=job_id)
            except Exception:
                poll = {"success": False, "error": "fetch failed"}
            status = str(poll.get("status") or "")
            if poll.get("success") and status == "running":
                # job 还在正常跑：touch 票据保持新鲜。否则纯等待期间没有
                # 轮次迭代刷新 updated_at，30 分钟 FRESH 窗口会把还在正常
                # 执行的计划从条带上隐藏，看起来像任务消失了。
                reply_ticket_manager.touch(ticket.ticket_id)
                continue
            if (
                poll.get("success")
                and status == "finished"
                and isinstance(poll.get("result"), dict)
            ):
                result = poll["result"]
            else:
                result = {
                    "success": False,
                    "error": "daemon job result unavailable (job lost or daemon restarted)",
                    "command": str(ticket.command or ""),
                    "job_id": job_id,
                    "exit_code": None,
                    "timed_out": False,
                    "duration_seconds": "",
                    "output_tail": "",
                }
            if not reply_ticket_manager.take_pending_job(ticket.ticket_id, job_id):
                continue
            entries.append(
                {
                    "command": str(ticket.command or ""),
                    "result": result,
                    "robot_job_context": robot_job_context,
                    "pending_robot_reply_id": "",
                    "reply_ticket_id": ticket.ticket_id,
                    "feedback": local_mcp_server._format_background_job_feedback(
                        result, reply_ticket_id=ticket.ticket_id
                    ),
                    "agent_session": agent_session,
                    "conversation_key": str(ticket.conversation_key or ""),
                }
            )
        if entries:
            flush_background_job_results_for_entries(item_id, entries)
            logger.info(
                "[RobotService] Reattached %s finished job result(s) for ticket=%s item=%s",
                len(entries),
                ticket.ticket_id,
                item_id,
            )
        return True

    def _resume_open_plan_tickets(self) -> int:
        """计划驱动自动续跑：找出"计划未完成 + 当前没有轮次"的票据，
        自动起一个新轮次接着执行——这才是 plan 机制该有的生命周期，
        不依赖用户说"继续"。空闲阈值防止在正常执行间隙误触发。
        唯一判断条件是 plan 还有未完成步骤；票据 status 不参与
        （delivered 只代表发过回复，不代表任务结束）。"""
        from app.plugins.robot.contracts import RobotReplyTarget
        from app.services.agent.integrations import (
            clear_integration_chat_contexts,
            setup_integration_chat_contexts,
        )
        from app.services.agent.reply_ticket import (
            plan_has_open_steps,
            reply_ticket_manager,
        )
        from app.services.agent.session import (
            InputMessage,
            InputType,
            agent_session_manager,
        )

        resumed = 0
        now = datetime.now()  # 与 ticket.updated_at 的 naive 本地时间对齐

        def _idle_seconds(ticket) -> float:
            updated = ticket.updated_at
            if updated.tzinfo is not None:
                updated = updated.replace(tzinfo=None)
            return (now - updated).total_seconds()

        with reply_ticket_manager._lock:
            all_tickets = list(reply_ticket_manager._tickets.values())
            tickets = [
                t
                for t in all_tickets
                if (plan_has_open_steps(t.plan) or t.pending_job_ids)
                and _idle_seconds(t) > 90
            ]
        if all_tickets:
            logger.debug(
                "[RobotService] plan sweep pass: total=%s candidates=%s",
                len(all_tickets),
                len(tickets),
            )
        for ticket in tickets:
            conversation_key = str(ticket.conversation_key or "")
            robot_id = str(ticket.robot_id or "")
            item_id = str(ticket.item_id or "")
            if not item_id:
                continue
            is_robot_ticket = bool(conversation_key and robot_id)
            try:
                robot = None
                if is_robot_ticket:
                    if self.conversation_has_active_dispatch(robot_id, conversation_key):
                        continue
                    with Session(engine) as session:
                        robot = session.get(Robot, uuid.UUID(robot_id))
                    if robot is None or not robot.is_enabled:
                        continue

                open_steps = [
                    str(step.get("step") or "")
                    for step in ticket.plan
                    if str(step.get("status")) not in {"completed", "cancelled"}
                ]
                from app.services.agent.session import SessionState

                handler_id = str(ticket.handler_id or "")
                agent_session = agent_session_manager.get_or_create_session(
                    item_id, handler_id
                )
                # 会话正忙（轮次运行中或队列里有待处理输入）就不要再塞
                # resume——否则长 LLM 思考间隙的 90s 空闲会反复触发，
                # 多个 resume 排队串行执行导致任务被重做。
                if (
                    agent_session.state == SessionState.RUNNING
                    or agent_session.input_queue.qsize() > 0
                ):
                    continue
                agent = agent_session.get_agent()
                if agent is None:
                    continue

                # 进程重启会杀死后台 job 的轮询线程，结果永远送不回来。
                # 按票据上持久化的 job_id 重新向 daemon 拉取并注入结果，
                # 让任务链断线重连；仍在跑的 job 就等下一轮 sweep。
                if ticket.pending_job_ids and self._reattach_pending_jobs(
                    ticket, agent_session
                ):
                    continue
                # 纯 job 票据（无计划）结果注入后即可，不做计划续跑。
                if not plan_has_open_steps(ticket.plan):
                    continue

                reply_target = RobotReplyTarget.model_validate(ticket.reply_target) if ticket.reply_target else None
                if reply_target is not None:
                    contexts = {
                        "robot": {
                            "robot_id": robot_id,
                            "sender_key": str(ticket.sender_key or ""),
                            "reply_target": reply_target,
                            "conversation_key": conversation_key,
                            "conversation_generation": int(ticket.conversation_generation or 0),
                            "reply_requires_awake": self._reply_context_window_seconds(robot) > 0,
                        }
                    }
                    setup_integration_chat_contexts(agent, contexts)
                try:
                    agent_session.process_input(
                        InputMessage(
                            input_type=InputType.CHAT,
                            content=(
                                "[Internal plan resume]\n"
                                f"当前任务计划还有未完成步骤：{'; '.join(open_steps) or '见计划'}。"
                                "不要总结、不要表态，立即调用工具继续执行当前步骤；"
                                "每完成一步用 mcp_local_update_plan 更新状态，全部完成后才总结。"
                            ),
                            raw_content="",
                            query="resume plan",
                            reply_ticket_id=ticket.ticket_id,
                            source_label=f"plan-resume {conversation_key or item_id}",
                        )
                    )
                    resumed += 1
                    if robot_id:
                        record_robot_event(
                            robot_id,
                            direction="backend_queue",
                            event="plan_resume_kicked",
                            payload={
                                "ticket_id": ticket.ticket_id,
                                "conversation": conversation_key,
                                "open_steps": len(open_steps),
                            },
                        )
                    else:
                        logger.info(
                            "[RobotService] Plan resume kicked for web ticket=%s item=%s open_steps=%s",
                            ticket.ticket_id,
                            item_id,
                            len(open_steps),
                        )
                finally:
                    if reply_target is not None:
                        clear_integration_chat_contexts(agent, contexts)
            except Exception:
                logger.exception(
                    "[RobotService] Failed to resume plan ticket=%s conversation=%s",
                    ticket.ticket_id,
                    conversation_key,
                )
        return resumed

    def sweep_pending_conversations(self) -> dict[str, int]:
        """Global invariant: no active LLM turn + pending input -> start a turn.

        Safety net for races/crashes where the turn-end drain never fired
        (e.g. a job result buffered right after the last flush, or a worker
        that died before its finally hook). Runs from the task watchdog every
        pass. Busy conversations are skipped; buffered job results re-buffer
        themselves if their conversation turns out to be busy, so this can
        never force concurrent turns on one conversation.
        """
        stats = {"pending_turns_kicked": 0, "buffered_results_flushed": 0}
        try:
            stats["plan_resumed"] = self._resume_open_plan_tickets()
        except Exception:
            logger.exception("[RobotService] plan resume sweep failed")
        with self._lock:
            pending_keys = [
                (robot_id, conversation_key)
                for (robot_id, conversation_key), entries
                in self._pending_task_replies.items()
                if entries
            ]
        for robot_id, conversation_key in pending_keys:
            try:
                if self.conversation_has_active_dispatch(robot_id, conversation_key):
                    continue
                with Session(engine) as session:
                    robot = session.get(Robot, uuid.UUID(robot_id))
                if robot is None or not robot.is_enabled:
                    continue
                # 停放的后台 job 结果标记：此前 sweep 只排聊天输入那个 store，
                # 这些标记永远没人处理（远端实测两个群积压 9 条几天不清）。
                # 转成一条"请查证后汇报"的追问输入，走正常 followup 一次救出。
                for reply in self._pop_pending_task_replies(robot_id, conversation_key):
                    self._record_pending_chat_input(
                        robot=robot,
                        conversation_key=conversation_key,
                        item_id=reply.item_id,
                        route_key=reply.route_key,
                        message_text=(
                            f"[系统] 后台任务（{reply.command}）的结果此前未能送达。"
                            "请用 mcp_local_list_jobs 查询其最新状态和输出并如实汇报；"
                            "若任务已完成或已查无此任务，简要说明即可。"
                        ),
                        sender_key=reply.sender_key,
                        sender_label=reply.sender_label,
                        trigger_reason=reply.trigger_reason or "background_job",
                        reply_target=reply.reply_target,
                    )
                if self._enqueue_pending_chat_followup(
                    robot=robot, conversation_key=conversation_key
                ):
                    stats["pending_turns_kicked"] += 1
                    logger.info(
                        "[RobotService] Sweep kicked pending turn: robot=%s conversation=%s",
                        robot_id,
                        conversation_key,
                    )
            except Exception:
                logger.exception(
                    "[RobotService] Sweep failed for pending conversation robot=%s conversation=%s",
                    robot_id,
                    conversation_key,
                )
        try:
            from app.services.agent.input_merge_buffer import input_merge_buffer
            from app.services.agent.mcp.local_server import (
                flush_all_job_results_for_item,
            )

            for item_id in input_merge_buffer.distinct_item_ids():
                if item_id == "qq":
                    continue
                try:
                    uuid.UUID(item_id)
                except ValueError:
                    continue
                flushed = flush_all_job_results_for_item(item_id)
                if flushed:
                    stats["buffered_results_flushed"] += flushed
        except Exception:
            logger.exception("[RobotService] Sweep buffer flush failed")
        return stats

    def reap_stuck_dispatch_jobs(self) -> int:
        """Force-fail dispatch jobs frozen past the hard timeout.

        A worker thread frozen inside a job (LLM stall, store hang, bridge
        stall) blocks the whole queue. The reaper force-fails the job (clears
        the conversation controller, reports the timeout) and starts a
        replacement worker so the queue keeps draining. The frozen thread is
        left to die on its own; its late writes are idempotent.
        """
        hard_timeout = float(settings.ROBOT_BACKEND_DISPATCH_HARD_TIMEOUT_SECONDS)
        now = time.monotonic()
        with self._lock:
            stuck = [
                (tracking_key, started, job)
                for tracking_key, (started, job) in self._active_dispatch_jobs.items()
                if now - started > hard_timeout
            ]
        reaped = 0
        for tracking_key, started, job in stuck:
            with self._lock:
                self._active_dispatch_jobs.pop(tracking_key, None)
            logger.warning(
                "[RobotService] Reaping stuck dispatch job robot=%s item=%s conversation=%s after %.0fs",
                job.robot_id,
                job.item_id,
                job.conversation_key,
                now - started,
            )
            record_robot_event(
                str(job.robot_id),
                direction="backend_worker",
                event="dispatch_hard_timeout",
                status="error",
                payload={
                    "item_id": str(job.item_id),
                    "route_key": job.route_key,
                    "conversation": job.conversation_key,
                    "stuck_seconds": round(now - started, 1),
                    "hard_timeout_seconds": hard_timeout,
                },
            )
            try:
                self._record_job_timeout_silent(
                    job,
                    f"hard timeout after {round(now - started, 1)}s",
                )
            except Exception:
                logger.exception(
                    "[RobotService] Failed to report hard-timeout for robot %s",
                    job.robot_id,
                )
            reaped += 1

        if reaped:
            self._ensure_replacement_workers(reaped)
        return reaped

    def _ensure_replacement_workers(self, needed: int = 1) -> None:
        """Start fresh workers to replace frozen ones, one per reaped job.

        Every reaped job means its worker thread is frozen inside that job,
        so the queue has effectively lost a consumer. Start one replacement
        per reaped job, capped at DISPATCH_WORKER_MAX_COUNT total workers.
        (The old check ran after the reaper had already removed the stuck
        jobs, so active_count could never reach worker_count and no
        replacement ever started -- once every worker froze, the queue
        stalled until a backend restart.)
        """
        if needed <= 0 or not self._dispatch_workers_started:
            return
        started = 0
        with self._dispatch_worker_lock:
            while (
                started < needed
                and self._dispatch_worker_count < DISPATCH_WORKER_MAX_COUNT
            ):
                index = self._dispatch_worker_count
                worker = threading.Thread(
                    target=self._dispatch_worker_loop,
                    name=f"TermPaws-backend-robot-dispatch-replacement-{index}",
                    daemon=True,
                )
                worker.start()
                self._dispatch_worker_count += 1
                started += 1
        if started:
            logger.warning(
                "[RobotService] Started %s replacement dispatch worker(s) (now %s) after stuck workers",
                started,
                self._dispatch_worker_count,
            )
        else:
            logger.error(
                "[RobotService] %s dispatch worker(s) stuck but worker cap %s reached; queue may stall",
                needed,
                DISPATCH_WORKER_MAX_COUNT,
            )

    def _process_chat_job(self, job: QueuedRobotChatJob) -> None:
        queue_wait_seconds = (self._now() - job.enqueued_at).total_seconds()
        record_robot_event(
            str(job.robot_id),
            direction="backend_worker",
            event="dispatch_started",
            payload={
                "item_id": str(job.item_id),
                "route_key": job.route_key,
                "queue_wait_seconds": round(queue_wait_seconds, 3),
                "queue": self.dispatch_queue_snapshot(),
            },
        )

        with Session(engine) as session:
            robot = session.get(Robot, job.robot_id)
            item = session.get(Item, job.item_id)
            if robot is None or item is None:
                self._record_and_send_job_error(
                    job,
                    "Robot or item no longer exists.",
                )
                return

            try:
                # 进队即处理：能进队列的消息在入站时已经过了"值不值得回"的
                # 判断，派发/回复路径上不再加"是否还醒着/generation 是否一致"
                # 的硬闸——那套闸在忙群里会让每条消息都因新消息不断 bump
                # generation 而永远无法通过，是排队卡死的根因。
                chat_message = self._prepare_queued_chat_message(
                    robot=robot,
                    item=item,
                    job=job,
                )
                record_robot_event(
                    str(job.robot_id),
                    direction="backend_worker",
                    event="dispatch_prepared",
                    payload={"item_id": str(job.item_id), "route_key": job.route_key},
                )
                job_started_at = time.monotonic()
                try:
                    response = asyncio.run(
                        asyncio.wait_for(
                            self._chat_with_item(
                                session=session,
                                robot=robot,
                                item=item,
                                message=chat_message,
                                sender_key=job.sender_key,
                                reply_target=job.reply_target,
                                conversation_key=job.conversation_key,
                                conversation_generation=job.conversation_generation,
                                reply_requires_awake=job.reply_requires_awake,
                                reply_ticket_id=job.reply_ticket_id,
                                request_message=self._job_request_display(job),
                                image_urls=job.image_urls or None,
                                agent_image_urls=job.agent_image_urls or None,
                                force_image_vision=bool(job.agent_image_urls),
                            ),
                            timeout=settings.ROBOT_BACKEND_JOB_TIMEOUT_SECONDS,
                        )
                    )
                    record_robot_event(
                        str(job.robot_id),
                        direction="backend_worker",
                        event="dispatch_chat_done",
                        payload={
                            "item_id": str(job.item_id),
                            "route_key": job.route_key,
                            "chat_seconds": round(time.monotonic() - job_started_at, 1),
                        },
                    )
                except TimeoutError:
                    record_robot_event(
                        str(job.robot_id),
                        direction="backend_worker",
                        event="dispatch_job_timeout",
                        status="error",
                        payload={
                            "item_id": str(job.item_id),
                            "route_key": job.route_key,
                            "timeout_seconds": settings.ROBOT_BACKEND_JOB_TIMEOUT_SECONDS,
                        },
                    )
                    self._record_job_timeout_silent(
                        job,
                        f"turn exceeded {settings.ROBOT_BACKEND_JOB_TIMEOUT_SECONDS}s",
                    )
                    # 超时必须释放 processing hold：job 已死，控制器继续挂着
                    # processing 会让后续消息全部堆进队列且永远没有轮次来消费。
                    if (
                        self._session_has_running_terminal_jobs(str(job.item_id))
                        or self._item_has_running_jobs(str(job.item_id))
                    ):
                        self._extend_processing_for_active_jobs(
                            robot, job.conversation_key,
                        )
                    else:
                        self._release_processing_hold(
                            robot.id, job.conversation_key,
                        )
                    self._enqueue_pending_chat_followup(
                        robot=robot,
                        conversation_key=job.conversation_key,
                    )
                    return
                response_text = self._visible_agent_response_text(response)
                robot_message_sent = response.robot_message_sent
                if (
                    not response_text
                    and not robot_message_sent
                    and job.direct_reply_trigger
                    and job.route_key != BACKGROUND_JOB_ROUTE_KEY
                ):
                    # 直接触发但空回复：只记录，不再自动补跑第二轮。
                    # prompt 层的"必须回复"规则是防沉默的正解；整轮纠正重试
                    # 是双发放大器（送达检测一旦有洞，单发变双发）且白费一倍 LLM 调用。
                    record_robot_event(
                        str(job.robot_id),
                        direction="backend_worker",
                        event="dispatch_empty_reply_direct_trigger",
                        status="ignored",
                        payload={
                            "item_id": str(job.item_id),
                            "conversation": job.conversation_key,
                            "raw_response": preview_text(response.content),
                        },
                    )
                    logger.info(
                        "[RobotService] Empty reply for direct trigger robot=%s raw=%r (no corrective retry)",
                        job.robot_id,
                        preview_text(response.content, limit=200),
                    )
                if response_text and not robot_message_sent:
                    # 最终文本不再由系统代发到 QQ：回复的唯一出口是
                    # mcp_robot_send_message，契约干净、结构上不可能双发。
                    # 模型没调工具的回复只进 web 聊天历史。
                    logger.info(
                        "[RobotService] Final text without tool send is NOT auto-delivered: robot=%s text=%r",
                        job.robot_id,
                        preview_text(response_text, limit=200),
                    )
                self._apply_reply_context_result(
                    robot,
                    job.conversation_key,
                    robot_message_sent=robot_message_sent,
                    reply_target=job.reply_target,
                    conversation_generation=job.conversation_generation or None,
                    sleep_when_no_reply=job.reply_context_active
                    and not job.direct_reply_trigger,
                )
                if (
                    self._session_has_running_terminal_jobs(str(job.item_id))
                    or self._item_has_running_jobs(str(job.item_id))
                ):
                    self._extend_processing_for_active_jobs(
                        robot, job.conversation_key,
                    )
                else:
                    self._release_processing_hold(
                        robot.id, job.conversation_key,
                    )
                try:
                    from app.services.agent.stream_manager import stream_manager
                    stream_manager.broadcast_chat_event(
                        str(job.item_id),
                        {
                            "type": "qq_message_processed",
                            "item_id": str(job.item_id),
                            "conversation_key": job.conversation_key,
                            "robot_message_sent": robot_message_sent,
                        },
                    )
                except Exception:
                    pass
                if job.pending_reply_id:
                    # One delivery attempt per job result: whether the send
                    # landed, was superseded by a newer turn, or failed, the
                    # turn has run its course. Keeping the entry would leak it
                    # as an orphaned queue item (stale by definition then).
                    self.clear_background_job_reply(
                        robot_id=job.robot_id,
                        conversation_key=job.conversation_key,
                        pending_reply_id=job.pending_reply_id,
                    )
                if robot_message_sent:
                    self._enqueue_pending_chat_followup(
                        robot=robot,
                        conversation_key=job.conversation_key,
                    )
                else:
                    self._enqueue_pending_chat_followup(
                        robot=robot,
                        conversation_key=job.conversation_key,
                        direct_wakeup_only=not job.direct_reply_trigger,
                    )
            except RobotServiceError as exc:
                self._record_and_send_job_error(job, exc.message)
                self._release_processing_hold(robot.id, job.conversation_key)
                self._enqueue_pending_chat_followup(
                    robot=robot,
                    conversation_key=job.conversation_key,
                )
                return
            except HTTPException as exc:
                detail = (
                    exc.detail if isinstance(exc.detail, str) else "Robot dispatch failed"
                )
                self._record_and_send_job_error(job, detail)
                self._release_processing_hold(robot.id, job.conversation_key)
                self._enqueue_pending_chat_followup(
                    robot=robot,
                    conversation_key=job.conversation_key,
                )
                return
            except Exception as exc:
                logger.exception(
                    "[RobotService] Unexpected queued robot dispatch error for robot %s",
                    job.robot_id,
                )
                self._record_and_send_job_error(
                    job,
                    str(exc) or "Robot dispatch failed.",
                )
                self._release_processing_hold(robot.id, job.conversation_key)
                self._enqueue_pending_chat_followup(
                    robot=robot,
                    conversation_key=job.conversation_key,
                )
                return

        if not response_text:
            record_robot_event(
                str(job.robot_id),
                direction="backend_to_bridge",
                event="dispatch_completed",
                payload={
                    "item_id": str(job.item_id),
                    "route_key": job.route_key,
                    "robot_message_sent": robot_message_sent,
                    "queue": self.dispatch_queue_snapshot(),
                },
            )
            logger.info(
                "[RobotService] Queued agent completed robot=%s item=%s route=%s sent=%s",
                job.robot_id,
                job.item_id,
                job.route_key,
                robot_message_sent,
            )
            return

        record_robot_event(
            str(job.robot_id),
            direction="backend_to_bridge",
            event="dispatch_response",
            message=response_text,
            payload={
                "item_id": str(job.item_id),
                "route_key": job.route_key,
                "reply_delivery": "mcp_tool"
                if response.robot_message_sent
                else "service_fallback",
                "robot_message_sent": robot_message_sent,
                "queue": self.dispatch_queue_snapshot(),
            },
        )
        logger.info(
            "[RobotService] Queued agent response robot=%s item=%s route=%s text=%s",
            job.robot_id,
            job.item_id,
            job.route_key,
            preview_text(response_text),
        )

    def _send_visible_agent_response(
        self,
        job: QueuedRobotChatJob,
        response_text: str,
    ) -> bool:
        if not response_text.strip():
            return False
        # The agent decided this conversation should sleep (via
        # mcp_robot_sleep_conversation or the no-reply path). Whatever final
        # text it produced is internal reasoning, not a reply — never
        # forward it to QQ.
        if self.conversation_is_sleeping(job.robot_id, job.conversation_key):
            record_robot_event(
                str(job.robot_id),
                direction="backend_to_bridge",
                event="dispatch_visible_response_dropped_sleeping_conversation",
                status="ignored",
                message=preview_text(response_text),
                payload={
                    "item_id": str(job.item_id),
                    "route_key": job.route_key,
                    "conversation": job.conversation_key,
                    "generation": job.conversation_generation,
                },
            )
            return False
        try:
            from .bridge_client import robot_bridge_client

            robot_bridge_client.send_message(job.robot_id, job.reply_target, response_text)
        except Exception:
            logger.exception(
                "[RobotService] Failed to send visible agent response robot=%s",
                job.robot_id,
            )
            record_robot_event(
                str(job.robot_id),
                direction="backend_to_bridge",
                event="dispatch_visible_response_send_failed",
                status="error",
                message=preview_text(response_text),
                payload={
                    "item_id": str(job.item_id),
                    "route_key": job.route_key,
                    "conversation": job.conversation_key,
                },
            )
            return False

        self._remember_assistant_conversation_memory(
            job.robot_id,
            job.conversation_key,
            response_text,
        )
        return True

    def _visible_agent_response_text(self, response: ChatResponseResult) -> str:
        text = (response.content or "").strip()
        if not text:
            return ""
        from app.plugins.robot.internal_trace import (
            is_robot_internal_trace_text,
            sanitize_robot_visible_text,
        )

        text = sanitize_robot_visible_text(text)
        if not text or is_robot_internal_trace_text(text):
            return ""
        try:
            from app.plugins.robot.agent.integration import (
                is_robot_read_only_tool_result,
            )
            from app.services.agent.integrations import fallback_is_delivery_result

            if fallback_is_delivery_result(text) or is_robot_read_only_tool_result(text):
                return ""
        except Exception:
            pass
        return text

    def _prepare_queued_chat_message(
        self,
        *,
        robot: Robot,
        item: Item,
        job: QueuedRobotChatJob,
    ) -> str:
        inbound_message = job.inbound_message
        message_text = (job.message_text or "").strip()
        if inbound_message is None or not message_text:
            return job.message

        pending_message_texts = self._pending_chat_message_texts(
            robot.id,
            job.conversation_key,
        )
        return self._agent_message_with_context(
            inbound_message,
            message_text,
            trigger_reason=job.trigger_reason,
            impression_card=self._conversation_impression_card(
                item_id=item.id,
                robot=robot,
                conversation_key=job.conversation_key,
                sender_key=job.sender_key,
                reply_target=job.reply_target,
                query=message_text,
            ),
            live_context_card=self._recent_live_context_card(
                robot=robot,
                conversation_key=job.conversation_key,
                trigger_reason=job.trigger_reason,
                current_message_text=message_text,
                excluded_message_texts=pending_message_texts,
            ),
        )

    def _record_job_timeout_silent(
        self,
        job: QueuedRobotChatJob,
        detail: str,
    ) -> None:
        """Timeout cleanup that stays invisible to the user: the turn dies
        quietly (task state in plan/ticket survives and later turns can pick
        it up), so there is no scary 'task timed out' message to send."""
        logger.warning(
            "[RobotService] Silent timeout cancel robot=%s item=%s conversation=%s: %s",
            job.robot_id,
            job.item_id,
            job.conversation_key,
            detail,
        )
        record_robot_event(
            str(job.robot_id),
            direction="backend_worker",
            event="dispatch_timeout_silent",
            status="ignored",
            message=detail,
            payload={
                "item_id": str(job.item_id),
                "route_key": job.route_key,
                "queue": self.dispatch_queue_snapshot(),
            },
        )
        if job.pending_reply_id:
            try:
                self.clear_background_job_reply(
                    robot_id=job.robot_id,
                    conversation_key=job.conversation_key,
                    pending_reply_id=job.pending_reply_id,
                )
            except Exception:
                pass

    def _record_and_send_job_error(
        self,
        job: QueuedRobotChatJob,
        message: str,
    ) -> None:
        # 错误汇报与正常回复同一条原则：既然轮次已经跑了，结果必须送达，
        # 不再过 generation/醒睡闸。
        allowed = True
        record_robot_event(
            str(job.robot_id),
            direction="backend_worker",
            event="dispatch_error",
            status="error",
            message=message,
            payload={
                "item_id": str(job.item_id),
                "route_key": job.route_key,
                "queue": self.dispatch_queue_snapshot(),
            },
        )
        delivered = False
        if allowed:
            try:
                from .bridge_client import robot_bridge_client

                robot_bridge_client.send_message(job.robot_id, job.reply_target, message)
            except Exception:
                logger.exception(
                    "[RobotService] Failed to send queued dispatch error robot=%s",
                    job.robot_id,
                )
            else:
                delivered = True
        self._clear_reply_context_window_for_key(
            job.robot_id,
            job.conversation_key,
            reason="dispatch_error",
            expected_generation=job.conversation_generation or None,
        )
        if delivered:
            self._remember_assistant_conversation_memory(job.robot_id, job.conversation_key, message)

    def handle_inbound_message(
        self,
        session: Session,
        robot: Robot,
        message: RobotInboundMessage,
    ) -> RobotDispatchResponse:
        self._enrich_message_bot_identity(robot, message)
        logger.info(
            "[RobotService] Inbound robot=%s sender=%s target=%s text=%s",
            robot.id,
            message.sender_key,
            message.reply_target.target_id,
            preview_text(message.text),
        )
        record_robot_event(
            str(robot.id),
            direction="bridge_to_backend",
            event="inbound_message",
            message=message.text,
            payload={
                "sender_key": message.sender_key,
                "target_type": message.reply_target.target_type,
                "target_id": message.reply_target.target_id,
                "reply_target": message.reply_target.model_dump(mode="json"),
            },
        )
        self._assert_robot_supported(robot)
        if not robot.is_enabled:
            return RobotDispatchResponse(success=True, ignored=True, reason="robot_disabled")

        direct_reply_trigger = self._message_directly_addresses_bot(message, robot)
        conversation_key = self._conversation_key(message)
        text = (message.text or "").strip()
        if not text and not direct_reply_trigger:
            return RobotDispatchResponse(success=True, ignored=True, reason="empty_message")
        if not text and direct_reply_trigger:
            text = "[empty robot wakeup]"
        controller_gate = self._record_conversation_controller_inbound(
            robot,
            message,
            conversation_key,
            text,
            direct_reply_trigger=direct_reply_trigger,
        )
        reply_context_active = controller_gate.active
        command_parse_text = self._strip_leading_bot_mentions_for_command(message, text)
        command = self._parse_robot_command(command_parse_text)
        if command.mode == "chat" and command.target is None:
            command = RobotCommand(mode="chat", target=None, text=text)
        if command.mode == "term" and self.normalize_chat_alias(command.target) in {
            "list",
            "ls",
        }:
            return self._binding_listing_response(session, robot, message, conversation_key)
        # 会被并入排队（忙+唤醒窗内）的消息先不写 .log：否则在跑的轮次
        # 会通过实况上下文/读 .log 看到"未回答的问题"并提前回答，
        # 跟进轮次再答一遍 → 重复回复。排队条目在跟进轮次派发时才落 .log。
        will_park_pending = bool(
            command.mode == "chat"
            and reply_context_active
            and self.conversation_has_active_dispatch(robot.id, conversation_key)
        )
        if not will_park_pending:
            self._remember_inbound_conversation_memory(
                robot,
                message,
                conversation_key,
                self._agent_visible_message_text(text),
            )
        if self._message_requests_conversation_sleep(command_parse_text) and (
            direct_reply_trigger or reply_context_active or controller_gate.sleeping
        ):
            self.sleep_conversation_controller(
                robot.id,
                conversation_key,
                reason="user_sleep_command",
            )
            record_robot_event(
                str(robot.id),
                direction="backend",
                event="message_ignored",
                status="ignored",
                message=message.text,
                payload={
                    "reason": "sleep_command",
                    "conversation": conversation_key,
                },
            )
            return RobotDispatchResponse(
                success=True,
                ignored=True,
                reason="sleep_command",
            )
        if command.mode == "chat" and not (direct_reply_trigger or reply_context_active):
            record_robot_event(
                str(robot.id),
                direction="backend",
                event="message_ignored",
                status="ignored",
                message=message.text,
                payload={
                    "reason": "conversation_sleeping",
                    "conversation": conversation_key,
                    "direct_reply_trigger": direct_reply_trigger,
                    "reply_context_active": reply_context_active,
                },
            )
            return RobotDispatchResponse(
                success=True,
                ignored=True,
                reason="conversation_sleeping",
            )
        mention_match_mode = self._mention_match_mode(robot)
        reply_categories = self._reply_message_categories(
            message,
            command,
            direct_reply_trigger=direct_reply_trigger,
            reply_context_active=reply_context_active,
            mention_match_mode=mention_match_mode,
        )
        allowed_reply_types = self._allowed_reply_message_types(robot)
        if not allowed_reply_types.intersection(reply_categories):
            record_robot_event(
                str(robot.id),
                direction="backend",
                event="message_ignored",
                status="ignored",
                message=message.text,
                payload={
                    "reason": REPLY_MESSAGE_TYPE_DISABLED_REASON,
                    "reply_message_types": sorted(reply_categories),
                    "allowed_reply_message_types": sorted(allowed_reply_types),
                    "mention_match_mode": mention_match_mode,
                },
            )
            return RobotDispatchResponse(
                success=True,
                ignored=True,
                reason=REPLY_MESSAGE_TYPE_DISABLED_REASON,
            )
        reply_requires_awake = (
            (direct_reply_trigger or reply_context_active)
            and self._reply_context_window_seconds(robot) > 0
        )
        conversation_generation = 0

        try:
            resolved_binding, message_text = self._resolve_chat_binding(
                session,
                robot,
                command,
                conversation_key,
                fallback_sender_key=message.sender_key,
            )
            if command.mode == "term" and not command.text:
                # `/term <alias>` with no content: switch the conversation's
                # default terminal. Plain chat messages then route to it via
                # the remembered binding below.
                self._remember_conversation(
                    robot.id,
                    conversation_key,
                    resolved_binding.item.id,
                )
                response_text = (
                    f"已切换到终端「{resolved_binding.item.title}」"
                    f"（别名 {resolved_binding.route_key}）。后续消息默认发送到该终端。"
                )
                record_robot_event(
                    str(robot.id),
                    direction="backend",
                    event="terminal_switch",
                    message=message.text,
                    payload={
                        "item_id": str(resolved_binding.item.id),
                        "route_key": resolved_binding.route_key,
                    },
                )
                self._remember_assistant_conversation_memory(
                    robot.id,
                    conversation_key,
                    response_text,
                )
                return RobotDispatchResponse(
                    success=True,
                    ignored=False,
                    item_id=str(resolved_binding.item.id),
                    route_key=resolved_binding.route_key,
                    reply_chunks=self._reply_chunks_for_target(
                        robot,
                        message.reply_target,
                        response_text,
                    ),
                )
            message_text = self._agent_visible_message_text(message_text)
            inbound_image_urls = self._cache_inbound_images(message.text)
            if not message_text and inbound_image_urls:
                message_text = "[图片]"
            if not message_text:
                record_robot_event(
                    str(robot.id),
                    direction="backend",
                    event="message_ignored",
                    status="ignored",
                    message=message.text,
                    payload={
                        "reason": "media_message_ignored",
                        "conversation": conversation_key,
                    },
                )
                return RobotDispatchResponse(
                    success=True,
                    ignored=True,
                    reason="media_message_ignored",
                )
            self._remember_conversation(
                robot.id,
                conversation_key,
                resolved_binding.item.id,
            )

            trigger_reason = self._agent_trigger_reason(
                message,
                direct_reply_trigger=direct_reply_trigger,
                reply_context_active=reply_context_active,
                mention_match_mode=mention_match_mode,
            )
            # Every chat message merges into the pending queue while a turn is
            # running, regardless of how it was triggered. The queue batches
            # rapid-fire messages into one follow-up turn, and the batch
            # claims its generation only when drained at turn end — so the
            # in-flight turn keeps its delivery right and the agent always
            # sees pending messages merged, never one-by-one.
            turn_actively_busy = self.conversation_has_active_dispatch(
                robot.id, conversation_key
            )
            if (
                command.mode == "chat"
                and turn_actively_busy
                and reply_context_active
            ):
                pending_size = self._record_pending_chat_input(
                    robot=robot,
                    conversation_key=conversation_key,
                    item_id=resolved_binding.item.id,
                    route_key=resolved_binding.route_key,
                    message_text=message_text,
                    sender_key=message.sender_key,
                    sender_label=self._sender_memory_label(message),
                    trigger_reason=trigger_reason,
                    reply_target=message.reply_target,
                    image_urls=inbound_image_urls,
                )
                record_robot_event(
                    str(robot.id),
                    direction="backend_queue",
                    event="dispatch_deferred_pending_chat",
                    payload={
                        "item_id": str(resolved_binding.item.id),
                        "route_key": resolved_binding.route_key,
                        "conversation": conversation_key,
                        "pending_size": pending_size,
                    },
                )
                return RobotDispatchResponse(
                    success=True,
                    ignored=False,
                    item_id=str(resolved_binding.item.id),
                    route_key=resolved_binding.route_key,
                    reason="queued_pending",
                    reply_chunks=[],
                )

            if direct_reply_trigger or reply_context_active:
                conversation_generation = self._begin_reply_context_dispatch(
                    robot,
                    conversation_key,
                    metadata=message.reply_target.metadata,
                )

            queued_job = QueuedRobotChatJob(
                robot_id=robot.id,
                robot_owner_id=robot.owner_id,
                item_id=resolved_binding.item.id,
                route_key=resolved_binding.route_key,
                message=self._agent_message_with_context(
                    message,
                    message_text,
                    trigger_reason=trigger_reason,
                ),
                sender_key=message.sender_key,
                reply_target=message.reply_target.model_copy(deep=True),
                conversation_key=conversation_key,
                direct_reply_trigger=direct_reply_trigger,
                reply_context_active=reply_context_active,
                conversation_generation=conversation_generation,
                reply_requires_awake=reply_requires_awake,
                message_text=message_text,
                trigger_reason=trigger_reason,
                inbound_message=message.model_copy(deep=True),
                enqueued_at=self._now(),
                image_urls=inbound_image_urls,
                agent_image_urls=(
                    inbound_image_urls
                    if self._image_recognition_enabled(robot)
                    else []
                ),
            )
            if not self._enqueue_chat_job(queued_job):
                self._clear_reply_context_window_for_key(
                    robot.id,
                    conversation_key,
                    reason="dispatch_queue_full",
                    expected_generation=conversation_generation or None,
                )
                record_robot_event(
                    str(robot.id),
                    direction="backend_queue",
                    event="dispatch_dropped_queue_full",
                    status="ignored",
                    message="Robot backend dispatch queue is full; message dropped.",
                    payload={
                        "item_id": str(resolved_binding.item.id),
                        "route_key": resolved_binding.route_key,
                        "queue": self.dispatch_queue_snapshot(),
                    },
                )
                return RobotDispatchResponse(
                    success=True,
                    ignored=True,
                    item_id=str(resolved_binding.item.id),
                    route_key=resolved_binding.route_key,
                    reason="dispatch_queue_full",
                    reply_chunks=[],
                )

            response = RobotDispatchResponse(
                success=True,
                ignored=False,
                item_id=str(resolved_binding.item.id),
                route_key=resolved_binding.route_key,
                reason="queued",
                reply_chunks=[],
            )
            record_robot_event(
                str(robot.id),
                direction="backend_queue",
                event="dispatch_queued",
                payload={
                    "item_id": str(resolved_binding.item.id),
                    "route_key": resolved_binding.route_key,
                    "queue": self.dispatch_queue_snapshot(),
                    "conversation": conversation_key,
                    "direct_reply_trigger": direct_reply_trigger,
                    "reply_context_active": reply_context_active,
                },
            )
            logger.info(
                "[RobotService] Queued inbound robot=%s item=%s route=%s queue=%s",
                robot.id,
                resolved_binding.item.id,
                resolved_binding.route_key,
                self.dispatch_queue_snapshot(),
            )
            return response
        except RobotServiceError as exc:
            self._clear_reply_context_window_for_key(
                robot.id,
                conversation_key,
                reason="dispatch_error",
                expected_generation=conversation_generation or None,
            )
            record_robot_event(
                str(robot.id),
                direction="backend",
                event="dispatch_error",
                status="error",
                message=exc.message,
            )
            return RobotDispatchResponse(
                success=False,
                ignored=False,
                error=exc.message,
                reply_chunks=self._reply_chunks_for_target(
                    robot,
                    message.reply_target,
                    exc.message,
                ),
            )
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "Robot dispatch failed"
            self._clear_reply_context_window_for_key(
                robot.id,
                conversation_key,
                reason="dispatch_error",
                expected_generation=conversation_generation or None,
            )
            record_robot_event(
                str(robot.id),
                direction="backend",
                event="dispatch_error",
                status="error",
                message=detail,
            )
            return RobotDispatchResponse(
                success=False,
                ignored=False,
                error=detail,
                reply_chunks=self._reply_chunks_for_target(
                    robot,
                    message.reply_target,
                    detail,
                ),
            )
        except Exception as exc:
            logger.exception(
                "[RobotService] Unexpected inbound message error for robot %s",
                robot.id,
            )
            self._clear_reply_context_window_for_key(
                robot.id,
                conversation_key,
                reason="dispatch_error",
                expected_generation=conversation_generation or None,
            )
            record_robot_event(
                str(robot.id),
                direction="backend",
                event="dispatch_error",
                status="error",
                message=str(exc),
            )
            fallback = "机器人处理失败"
            return RobotDispatchResponse(
                success=False,
                ignored=False,
                error=str(exc),
                reply_chunks=self._reply_chunks_for_target(
                    robot,
                    message.reply_target,
                    fallback,
                ),
            )

    def dispatch_filtered_output(
        self,
        item_id: str,
        filtered_output: str,
        *,
        item_title: str | None = None,
    ) -> None:
        logger.debug(
            "[RobotService] Skipping terminal output dispatch to robot for item=%s",
            item_id,
        )

    def _pending_chat_key(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> tuple[str, str]:
        return (str(robot_id), conversation_key)

    def _pending_chat_scope(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> tuple[str, str]:
        """Scope used in the system-level input merge buffer for QQ pending chat."""
        return ("qq", f"{robot_id}:{conversation_key}")

    @staticmethod
    def _chat_input_to_merge_entry(
        entry: PendingRobotChatInput,
        scope_key: str,
    ) -> MergeBufferEntry:
        return MergeBufferEntry(
            source_type=SOURCE_QQ_MESSAGE,
            item_id="qq",
            scope_key=scope_key,
            sender_label=entry.sender_label,
            sender_key=entry.sender_key,
            content=entry.message_text,
            enqueued_at=entry.enqueued_at,
            payload=entry,
        )

    def _record_pending_chat_input(
        self,
        *,
        robot: Robot,
        conversation_key: str,
        item_id: uuid.UUID,
        route_key: str,
        message_text: str,
        sender_key: str,
        sender_label: str,
        trigger_reason: str,
        reply_target: RobotReplyTarget,
        image_urls: list[str] | None = None,
    ) -> int:
        entry = PendingRobotChatInput(
            item_id=item_id,
            route_key=route_key,
            message_text=message_text,
            sender_key=sender_key,
            sender_label=sender_label or sender_key,
            trigger_reason=trigger_reason or "active_chat_window",
            reply_target=reply_target.model_copy(deep=True),
            enqueued_at=self._now(),
            image_urls=tuple(image_urls or ()),
        )
        _scope_item, scope_key = self._pending_chat_scope(robot.id, conversation_key)
        pending_size, evicted = input_merge_buffer.add(
            self._chat_input_to_merge_entry(entry, scope_key)
        )

        # 排队的消息也立即写进聊天历史（带图片标记）：前端能完整看到
        # 用户发了什么，而不是只有被直接派发的那一条。
        try:
            from app.services.agent.history.chat import append_chat_message
            from app.services.agent.stream_manager import stream_manager

            display_parts = [entry.message_text] if entry.message_text.strip() else []
            display_parts.extend(f"![图片]({url})" for url in entry.image_urls)
            display_content = "\n".join(display_parts) or "[空消息]"
            history_event = append_chat_message(
                str(item_id),
                role="user",
                content=display_content,
                message_type="qq_user",
                extra={
                    "sender_key": entry.sender_key,
                    "sender_label": entry.sender_label,
                    "source": "qq",
                    "pending": True,
                },
            )
            stream_manager.broadcast_chat_event(str(item_id), history_event)
        except Exception:
            logger.exception(
                "[RobotService] Failed to persist pending chat input: item=%s",
                item_id,
            )

        if evicted:
            record_robot_event(
                str(robot.id),
                direction="backend_queue",
                event="pending_chat_evicted",
                status="ignored",
                payload={
                    "conversation": conversation_key,
                    "evicted_count": len(evicted),
                    "pending_size": pending_size,
                },
            )
        return pending_size

    def _drain_pending_chat_inputs(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> list[PendingRobotChatInput]:
        scope_item, scope_key = self._pending_chat_scope(robot_id, conversation_key)
        return [
            entry.payload
            for entry in input_merge_buffer.pop(scope_item, scope_key)
            if entry.payload is not None
        ]

    def _pending_chat_message_texts(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> list[str]:
        scope_item, scope_key = self._pending_chat_scope(robot_id, conversation_key)
        return [
            entry.payload.message_text
            for entry in input_merge_buffer.peek(scope_item, scope_key)
            if entry.payload is not None and entry.payload.message_text.strip()
        ]

    def _prepend_pending_chat_inputs(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
        entries: list[PendingRobotChatInput],
    ) -> None:
        if not entries:
            return
        scope_item, scope_key = self._pending_chat_scope(robot_id, conversation_key)
        input_merge_buffer.prepend(
            scope_item,
            scope_key,
            [self._chat_input_to_merge_entry(entry, scope_key) for entry in entries],
        )

    def _pending_chat_snapshot_locked(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> tuple[list[dict[str, object]], uuid.UUID | None]:
        key = self._pending_chat_key(robot_id, conversation_key)
        scope_item, scope_key = self._pending_chat_scope(robot_id, conversation_key)
        entries = [
            entry.payload
            for entry in input_merge_buffer.peek(scope_item, scope_key)
            if entry.payload is not None
        ]
        task_replies = list(self._pending_task_replies.get(key) or [])
        snapshots: list[dict[str, object]] = []
        first_item_id: uuid.UUID | None = None
        index = 1
        for entry in entries[-PENDING_CHAT_QUEUE_LIMIT:]:
            if first_item_id is None:
                first_item_id = entry.item_id
            text = re.sub(
                r"\s+",
                " ",
                self._agent_visible_message_text(entry.message_text),
            ).strip()
            if len(text) > 180:
                text = f"{text[:177]}..."
            snapshots.append(
                {
                    "index": index,
                    "item_id": str(entry.item_id),
                    "route_key": entry.route_key,
                    "sender_key": entry.sender_key,
                    "sender_label": entry.sender_label,
                    "trigger_reason": entry.trigger_reason,
                    "message_preview": text,
                    "enqueued_at": entry.enqueued_at.isoformat(),
                    "direct_wakeup": self._pending_chat_entry_is_direct_wakeup(entry),
                }
            )
            index += 1
        remaining_slots = max(0, PENDING_CHAT_QUEUE_LIMIT - len(snapshots))
        for task_reply in task_replies[:remaining_slots]:
            if first_item_id is None:
                first_item_id = task_reply.item_id
            command = re.sub(r"\s+", " ", task_reply.command).strip()
            if len(command) > 140:
                command = f"{command[:137]}..."
            snapshots.append(
                {
                    "index": index,
                    "item_id": str(task_reply.item_id),
                    "route_key": task_reply.route_key,
                    "sender_key": task_reply.sender_key,
                    "sender_label": task_reply.sender_label,
                    "trigger_reason": task_reply.trigger_reason,
                    "message_preview": f"后台 Job 完成后回复这个 QQ 会话：{command}",
                    "enqueued_at": task_reply.enqueued_at.isoformat(),
                    "direct_wakeup": True,
                    "pending_reply_id": task_reply.pending_id,
                    "command_preview": command,
                }
            )
            index += 1
        return snapshots, first_item_id

    def _agent_visible_message_text(self, message_text: str) -> str:
        def replace_ignored_cq(match: re.Match[str]) -> str:
            cq_type = match.group(1).strip().lower()
            if cq_type in QQ_AGENT_IGNORED_CQ_TYPES:
                return " "
            return match.group(0)

        sanitized = CQ_CODE_PATTERN.sub(replace_ignored_cq, message_text or "")
        sanitized = re.sub(r"[ \t\r\f\v]+", " ", sanitized)
        sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
        return sanitized.strip()

    def _pending_chat_entry_is_direct_wakeup(self, entry: PendingRobotChatInput) -> bool:
        return entry.trigger_reason in PENDING_DIRECT_WAKE_TRIGGER_REASONS

    def _pending_chat_batch_text(
        self,
        entries: list[PendingRobotChatInput],
        *,
        conversation_key: str = "",
    ) -> str:
        lines = [
            "[Pending QQ messages; merge same-sender follow-ups and answer current unresolved intents]",
            "These messages arrived while the bot was already thinking. Treat them as current live QQ messages, not old log history.",
            f"source=QQ; conversation={conversation_key or 'current'}",
        ]
        index = 1
        for entry in entries[-PENDING_CHAT_QUEUE_LIMIT:]:
            text = re.sub(
                r"\s+", " ", self._agent_visible_message_text(entry.message_text)
            ).strip()
            if not text:
                continue
            if len(text) > 220:
                text = f"{text[:217]}..."
            lines.append(
                f"{index}. sender={entry.sender_label}; sender_key={entry.sender_key}; "
                f"trigger={entry.trigger_reason}: {text}"
            )
            index += 1
        lines.append(
            "每条内容不同的消息都要覆盖到（不能漏答），但回复要语义融合成一条简短气泡："
            "多个问题的答案揉成一句自然的话（如\"你在哪\"\"你在干嘛\"→\"我在外面吹风\"），"
            "不要逐条分开发多个气泡。只有不同发送者各自提问且必须区分时，才在一条里点名对应。"
        )
        return "\n".join(lines)

    def _enqueue_pending_chat_followup(
        self,
        *,
        robot: Robot,
        conversation_key: str,
        direct_wakeup_only: bool = False,
    ) -> bool:
        entries = [
            replace(entry, message_text=visible_text)
            for entry in self._drain_pending_chat_inputs(robot.id, conversation_key)
            for visible_text in [self._agent_visible_message_text(entry.message_text)]
            if visible_text
            and (not direct_wakeup_only or self._pending_chat_entry_is_direct_wakeup(entry))
        ]
        if not entries:
            return False

        latest = entries[-1]
        distinct_sender_keys = {
            entry.sender_key for entry in entries if entry.sender_key.strip()
        }
        followup_reply_target = latest.reply_target.model_copy(deep=True)
        followup_reply_target.metadata["allow_multiple_reply_messages"] = (
            len(distinct_sender_keys) > 1
        )
        batch_text = self._pending_chat_batch_text(
            entries,
            conversation_key=conversation_key,
        )
        batch_image_urls: list[str] = []
        for entry in entries:
            for url in entry.image_urls:
                if url not in batch_image_urls:
                    batch_image_urls.append(url)
        live_context_card = self._recent_live_context_card(
            robot=robot,
            conversation_key=conversation_key,
            lines=RECENT_LIVE_CONTEXT_EXPANDED_LINES,
            trigger_reason="pending_queue",
            current_message_text=latest.message_text,
            excluded_message_texts=[entry.message_text for entry in entries],
        )
        synthetic_message = RobotInboundMessage(
            sender_key=latest.sender_key,
            text=latest.message_text,
            reply_target=followup_reply_target.model_copy(deep=True),
        )
        conversation_generation = self._begin_reply_context_dispatch(
            robot,
            conversation_key,
            metadata=latest.reply_target.metadata,
        )
        composed_message = self._agent_message_with_context(
            synthetic_message,
            batch_text,
            trigger_reason="pending_queue",
            impression_card=self._conversation_impression_card(
                item_id=latest.item_id,
                robot=robot,
                conversation_key=conversation_key,
                sender_key=latest.sender_key,
                reply_target=latest.reply_target,
                query=batch_text,
            ),
            live_context_card=live_context_card,
        )
        dispatched = self._dispatch_pending_batch_to_session_queue(
            robot=robot,
            conversation_key=conversation_key,
            latest=latest,
            followup_reply_target=followup_reply_target,
            composed_message=composed_message,
            batch_text=batch_text,
            conversation_generation=conversation_generation,
            image_urls=batch_image_urls,
        )
        if not dispatched:
            self._prepend_pending_chat_inputs(robot.id, conversation_key, entries)
            self._clear_reply_context_window_for_key(
                robot.id,
                conversation_key,
                reason="pending_dispatch_failed",
                expected_generation=conversation_generation or None,
            )
            record_robot_event(
                str(robot.id),
                direction="backend_queue",
                event="pending_dispatch_dropped_enqueue_failed",
                status="ignored",
                payload={
                    "item_id": str(latest.item_id),
                    "route_key": latest.route_key,
                    "conversation": conversation_key,
                    "pending_size": len(entries),
                },
            )
            return False

        record_robot_event(
            str(robot.id),
            direction="backend_queue",
            event="pending_dispatch_queued",
            payload={
                "item_id": str(latest.item_id),
                "route_key": latest.route_key,
                "conversation": conversation_key,
                "pending_size": len(entries),
                "queue": self.dispatch_queue_snapshot(),
            },
        )
        # 排队条目在派发成功后才落 .log（入站时故意没写）：在跑的轮次不会把
        # 它们当成"未回答的问题"提前回答，跟进轮次独占回答权；失败重排不重复写。
        for entry in entries:
            try:
                robot_conversation_memory.append_user_message(
                    robot.id,
                    conversation_key,
                    entry.message_text,
                    sender=entry.sender_label or entry.sender_key,
                )
            except Exception:
                logger.debug(
                    "[RobotService] Failed to write pending entry to conversation memory",
                    exc_info=True,
                )
        return True

    def _dispatch_pending_batch_to_session_queue(
        self,
        *,
        robot: Robot,
        conversation_key: str,
        latest: Any,
        followup_reply_target: RobotReplyTarget,
        composed_message: str,
        batch_text: str,
        conversation_generation: int,
        image_urls: list[str] | None = None,
    ) -> bool:
        """Feed the merged pending batch into the item's system input queue.

        The batch rides the same per-item queue as web/terminal inputs, so
        cross-source messages merge into one turn instead of each source
        opening its own. The QQ reply ticket bound here is the batch's reply
        route; the session turn rebuilds the robot context from it, so the
        agent's send tools target the right conversation.
        """
        from sqlmodel import Session as _DbSession

        from app.core.db import engine as _engine
        from app.models import User as _User
        from app.services.agent.chat_runtime import get_item_handler_llm_config
        from app.services.agent.integrations import (
            clear_integration_chat_contexts,
            setup_integration_chat_contexts,
        )
        from app.services.agent.reply_ticket import reply_ticket_manager
        from app.services.agent.session import (
            InputMessage,
            InputType,
            agent_session_manager,
        )

        try:
            with _DbSession(_engine) as db:
                owner = db.get(_User, robot.owner_id)
                if owner is None:
                    return False
                prepared = get_item_handler_llm_config(
                    db, latest.item_id, owner
                )
            if not prepared:
                return False
            handler, _item = prepared
            agent_session = agent_session_manager.get_or_create_session(
                str(latest.item_id), str(handler.id)
            )
            agent = agent_session.get_agent()
            if agent is None:
                return False

            contexts = {
                "robot": {
                    "robot_id": str(robot.id),
                    "sender_key": latest.sender_key,
                    "reply_target": followup_reply_target,
                    "conversation_key": conversation_key,
                    "conversation_generation": conversation_generation,
                    "reply_requires_awake": self._reply_context_window_seconds(robot) > 0,
                }
            }
            setup_integration_chat_contexts(agent, contexts)
            try:
                ticket = reply_ticket_manager.create_for_agent(
                    agent,
                    item_id=str(latest.item_id),
                    handler_id=str(handler.id),
                    message=latest.message_text or batch_text,
                )
            finally:
                clear_integration_chat_contexts(agent, contexts)

            agent_session.process_input(
                InputMessage(
                    input_type=InputType.CHAT,
                    content=composed_message,
                    raw_content=batch_text,
                    query=latest.message_text,
                    reply_ticket_id=ticket.ticket_id,
                    source_label=f"QQ {conversation_key}",
                    image_urls=list(image_urls or []),
                    force_image_vision=self._image_recognition_enabled(robot),
                )
            )
            return True
        except Exception:
            logger.exception(
                "[RobotService] Failed to dispatch pending batch to session queue: conversation=%s",
                conversation_key,
            )
            return False

    def _recent_live_context_card(
        self,
        *,
        robot: Robot,
        conversation_key: str,
        lines: int | None = None,
        trigger_reason: str = "",
        current_message_text: str = "",
        excluded_message_texts: list[str] | None = None,
    ) -> str:
        if not conversation_key:
            return ""
        line_budget, _strategy, _hint = self._recent_live_context_budget(
            current_message_text,
            trigger_reason=trigger_reason,
            requested_lines=lines,
        )
        if line_budget <= 0:
            return ""
        try:
            from app.plugins.robot.conversation_memory import (
                recent_dialogue_scan_lines,
                select_recent_dialogue_lines,
            )
            from app.plugins.robot.internal_trace import sanitize_robot_visible_text

            recent = self._agent_visible_message_text(
                sanitize_robot_visible_text(
                    robot_conversation_memory.read_recent(
                        robot.id,
                        conversation_key,
                        lines=recent_dialogue_scan_lines(line_budget),
                    )
                )
            ).strip()
        except Exception:
            logger.debug(
                "[RobotService] Failed to read recent live context robot=%s conversation=%s",
                robot.id,
                conversation_key,
                exc_info=True,
            )
            return ""
        if not recent:
            return ""
        recent_lines = self._recent_context_lines_without_current(
            recent,
            current_message_text=current_message_text,
            excluded_message_texts=excluded_message_texts,
        )
        recent_lines = select_recent_dialogue_lines(
            "\n".join(recent_lines),
            lines=line_budget,
        )
        if not recent_lines:
            return ""
        return (
            "[Recent QQ live context; latest "
            f"{line_budget} line(s), background only; answer only current/pending "
            "messages; unclear → call `mcp_robot_read_conversation_memory` once, "
            "still unclear → ask one brief clarification]\n"
            + "\n".join(recent_lines)
        )

    def _recent_live_context_budget(
        self,
        message_text: str,
        *,
        trigger_reason: str = "",
        requested_lines: int | None = None,
    ) -> tuple[int, str, str]:
        if requested_lines is not None:
            line_budget = max(0, min(int(requested_lines), RECENT_LIVE_CONTEXT_LINES))
            if line_budget <= 0:
                return 0, "off", "caller disabled recent context"
            return (
                line_budget,
                "requested",
                "caller requested a fixed recent-context budget",
            )

        if trigger_reason == "active_chat_window":
            return (
                RECENT_LIVE_CONTEXT_ACTIVE_LINES,
                "active_window",
                "active wake window needs a small same-conversation sample to decide reply vs sleep",
            )

        # Static budget for directly-addressed turns: never shrink context
        # by guessing whether the message "needs" it — the LLM decides what
        # is relevant; starving it of history only forces clarification loops.
        return (
            RECENT_LIVE_CONTEXT_EXPANDED_LINES,
            "expanded",
            "full recent same-conversation window",
        )

    def _recent_context_lines_without_current(
        self,
        recent: str,
        *,
        current_message_text: str = "",
        excluded_message_texts: list[str] | None = None,
    ) -> list[str]:
        recent_lines = [line for line in str(recent or "").splitlines() if line.strip()]
        excluded_counts = Counter(
            normalized
            for value in (excluded_message_texts or [])
            for normalized in [self._normalize_live_context_message_text(value)]
            if normalized
        )
        current_text = self._normalize_live_context_message_text(current_message_text)
        if current_text and excluded_counts[current_text] <= 0:
            excluded_counts[current_text] += 1
        if not excluded_counts:
            return recent_lines

        kept_reversed: list[str] = []
        for line in reversed(recent_lines):
            normalized_line = self._normalize_live_context_message_text(
                self._conversation_memory_line_text(line)
            )
            if excluded_counts[normalized_line] > 0:
                excluded_counts[normalized_line] -= 1
                continue
            kept_reversed.append(line)
        return list(reversed(kept_reversed))

    @staticmethod
    def _conversation_memory_line_text(line: str) -> str:
        match = re.match(
            r"^\[[^\]]+\]\s+(?:user|assistant|system)(?:\s+[^:]+)?:\s*(.*)$",
            str(line or "").strip(),
        )
        if match:
            return match.group(1).strip()
        return str(line or "").strip()

    @staticmethod
    def _normalize_live_context_message_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    def _remember_assistant_conversation_memory(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
        message_text: str,
    ) -> None:
        from app.plugins.robot.internal_trace import sanitize_robot_visible_text

        message_text = sanitize_robot_visible_text(message_text)
        if not message_text.strip():
            return
        try:
            robot_conversation_memory.append_assistant_message(
                robot_id,
                conversation_key,
                message_text,
            )
        except Exception:
            logger.exception(
                "[RobotService] Failed to write assistant conversation memory robot=%s conversation=%s",
                robot_id,
                conversation_key,
            )

    def _remember_inbound_conversation_memory(
        self,
        robot: Robot,
        message: RobotInboundMessage,
        conversation_key: str,
        message_text: str,
    ) -> None:
        if not message_text.strip():
            return
        try:
            robot_conversation_memory.append_user_message(
                robot.id,
                conversation_key,
                message_text,
                sender=self._sender_memory_label(message),
            )
        except Exception:
            logger.exception(
                "[RobotService] Failed to write conversation memory robot=%s conversation=%s",
                robot.id,
                conversation_key,
            )

    def _sender_memory_label(self, message: RobotInboundMessage) -> str:
        sender_data = message.reply_target.metadata.get("sender")
        if isinstance(sender_data, dict):
            sender_id = str(sender_data.get("user_id") or "").strip()
            display_name = str(sender_data.get("display_name") or sender_data.get("card") or sender_data.get("nickname") or sender_id or "").strip()
            if display_name and sender_id and display_name != sender_id:
                return f"{display_name} ({sender_id})"
            return display_name or sender_id
        return message.sender_key

    def _sender_label_from_reply_target(
        self,
        reply_target: RobotReplyTarget,
        sender_key: str,
    ) -> str:
        sender_data = reply_target.metadata.get("sender")
        if isinstance(sender_data, dict):
            sender_id = str(sender_data.get("user_id") or "").strip()
            display_name = str(
                sender_data.get("display_name")
                or sender_data.get("card")
                or sender_data.get("nickname")
                or sender_id
                or ""
            ).strip()
            if display_name and sender_id and display_name != sender_id:
                return f"{display_name} ({sender_id})"
            if display_name or sender_id:
                return display_name or sender_id
        return sender_key

    def normalize_chat_alias(self, value: str | None) -> str | None:
        if value is None:
            return None

        normalized = re.sub(r"\s+", "-", value.strip())
        normalized = re.sub(r"[^\w\-]+", "-", normalized, flags=re.UNICODE)
        normalized = re.sub(r"-{2,}", "-", normalized).strip("-_")
        return normalized.lower() or None

    def build_route_key(self, item: Item, binding: RobotItem) -> str:
        return (
            self.normalize_chat_alias(binding.chat_alias)
            or self.normalize_chat_alias(item.title)
            or str(item.id)
        )

    def collect_route_keys(
        self,
        session: Session,
        robot_id: uuid.UUID,
        *,
        exclude_item_id: uuid.UUID | None = None,
    ) -> set[str]:
        keys: set[str] = set()
        for binding, item in self._load_robot_bindings(session, robot_id):
            if exclude_item_id and item.id == exclude_item_id:
                continue
            keys.add(self.build_route_key(item, binding))
        return keys

    def get_binding_route_key(
        self,
        session: Session,
        robot_id: uuid.UUID,
        item_id: uuid.UUID,
    ) -> str | None:
        for binding, item in self._load_robot_bindings(session, robot_id):
            if item.id == item_id:
                return self.build_route_key(item, binding)
        return None

    def _load_robot_bindings(
        self,
        session: Session,
        robot_id: uuid.UUID,
    ) -> list[tuple[RobotItem, Item]]:
        bindings = session.exec(select(RobotItem).where(RobotItem.robot_id == robot_id)).all()

        pairs: list[tuple[RobotItem, Item]] = []
        for binding in bindings:
            item = session.get(Item, binding.item_id)
            if item is not None:
                pairs.append((binding, item))
        return pairs

    def _resolve_chat_binding(
        self,
        session: Session,
        robot: Robot,
        command: RobotCommand,
        conversation_key: str,
        fallback_sender_key: str = "",
    ) -> tuple[ResolvedRobotBinding, str]:
        bindings = [
            ResolvedRobotBinding(
                binding=binding,
                item=item,
                route_key=self.build_route_key(item, binding),
            )
            for binding, item in self._load_robot_bindings(session, robot.id)
            if binding.allow_chat
        ]

        if not bindings:
            raise RobotServiceError("当前机器人没有可聊天的终端绑定")

        explicit_route_key = self.normalize_chat_alias(command.target)
        if explicit_route_key:
            for resolved in bindings:
                if (
                    resolved.route_key == explicit_route_key
                    or str(resolved.item.id) == command.target
                    or self.normalize_chat_alias(resolved.item.title) == explicit_route_key
                ):
                    # Empty text = switch the conversation's default terminal;
                    # the caller turns this into a switch confirmation.
                    return resolved, command.text
            raise RobotServiceError(f"没有找到路由 `{command.target}` 对应的终端")

        remembered_item_id = self._get_remembered_item_id(robot.id, conversation_key)
        if not remembered_item_id and fallback_sender_key:
            remembered_item_id = self._get_remembered_item_id(
                robot.id,
                fallback_sender_key,
            )
        if remembered_item_id:
            for resolved in bindings:
                if resolved.item.id == remembered_item_id:
                    return resolved, command.text

        default_bindings = [binding for binding in bindings if binding.binding.is_default_target]
        if len(default_bindings) == 1:
            return default_bindings[0], command.text

        if len(bindings) == 1:
            return bindings[0], command.text

        aliases = ", ".join(sorted(binding.route_key for binding in bindings))
        raise RobotServiceError(
            "当前机器人绑定了多个终端，请使用 "
            f"`/term <别名> <内容>` 发送。可用别名: {aliases}"
        )

    def _binding_listing_response(
        self,
        session: Session,
        robot: Robot,
        message: RobotInboundMessage,
        conversation_key: str,
    ) -> RobotDispatchResponse:
        """`/term list`: show the terminals this robot can route to and how
        to address them (`/term <alias>` switches the default terminal)."""
        bindings = [
            ResolvedRobotBinding(
                binding=binding,
                item=item,
                route_key=self.build_route_key(item, binding),
            )
            for binding, item in self._load_robot_bindings(session, robot.id)
            if binding.allow_chat
        ]
        remembered_item_id = self._get_remembered_item_id(robot.id, conversation_key)
        if not bindings:
            response_text = "当前机器人没有绑定任何终端。"
            current_item_id = None
            current_route_key = None
        else:
            lines = ["当前机器人绑定的终端："]
            current_item_id = None
            current_route_key = None
            for index, resolved in enumerate(bindings, 1):
                markers: list[str] = []
                if remembered_item_id and resolved.item.id == remembered_item_id:
                    markers.append("当前")
                elif not remembered_item_id and resolved.binding.is_default_target:
                    markers.append("默认")
                if markers:
                    current_item_id = str(resolved.item.id)
                    current_route_key = resolved.route_key
                suffix = f" ← {'/'.join(markers)}" if markers else ""
                lines.append(f"{index}. {resolved.route_key}（{resolved.item.title}）{suffix}")
            lines.append("切换默认终端：/term <别名>；本条消息直接指定：/term <别名> <内容>")
            response_text = "\n".join(lines)
        record_robot_event(
            str(robot.id),
            direction="backend",
            event="terminal_list",
            message=message.text,
            payload={"conversation": conversation_key},
        )
        return RobotDispatchResponse(
            success=True,
            ignored=False,
            item_id=current_item_id,
            route_key=current_route_key,
            reply_chunks=self._reply_chunks_for_target(
                robot,
                message.reply_target,
                response_text,
            ),
        )

    def _extract_route_key(self, text: str) -> tuple[str | None, str]:
        normalized = (text or "").strip()
        if not normalized:
            return None, ""

        command_match = re.match(
            r"^/(?:term|item|terminal)\s+(\S+)(?:\s+(.*))?$",
            normalized,
            flags=re.IGNORECASE,
        )
        if command_match:
            route_key = self.normalize_chat_alias(command_match.group(1))
            return route_key, (command_match.group(2) or "").strip()

        hash_match = re.match(r"^#(\S+)(?:\s+(.*))?$", normalized)
        if hash_match:
            route_key = self.normalize_chat_alias(hash_match.group(1))
            return route_key, (hash_match.group(2) or "").strip()

        return None, normalized

    def _bot_self_ids_from_message(self, message: RobotInboundMessage) -> set[str]:
        metadata = message.reply_target.metadata
        raw_ids = metadata.get("bot_self_ids")
        values = list(raw_ids) if isinstance(raw_ids, list) else []
        bot_identity = metadata.get("bot_identity")
        if isinstance(bot_identity, dict):
            identity_ids = bot_identity.get("self_ids")
            if isinstance(identity_ids, list):
                values.extend(identity_ids)
        return {str(value).strip() for value in values if str(value or "").strip()}

    def _enrich_message_bot_identity(
        self,
        robot: Robot,
        message: RobotInboundMessage,
    ) -> None:
        metadata = message.reply_target.metadata
        identity = metadata.get("bot_identity")
        identity = dict(identity) if isinstance(identity, dict) else {}

        self_ids = set(self._bot_self_ids_from_message(message))
        try:
            credentials = get_robot_runtime_config(robot).get("credentials") or {}
        except Exception:
            credentials = {}
        configured_self_id = str(credentials.get("self_id") or "").strip()
        if configured_self_id:
            self_ids.add(configured_self_id)

        names = [
            str(value).strip()
            for value in (identity.get("aliases") or [])
            if str(value or "").strip()
        ]
        display_name = str(identity.get("display_name") or "").strip()
        configured_name = str(robot.name or "").strip()
        for value in (display_name, configured_name):
            if value and value not in names:
                names.append(value)

        metadata["bot_self_ids"] = sorted(self_ids)
        metadata["bot_identity"] = {
            "self_ids": sorted(self_ids),
            "display_name": display_name or configured_name,
            "aliases": names,
        }

    @staticmethod
    def _bot_display_name_from_message(message: RobotInboundMessage) -> str:
        identity = message.reply_target.metadata.get("bot_identity")
        if not isinstance(identity, dict):
            return ""
        return str(identity.get("display_name") or "").strip()

    def _message_mentions_bot_self_id(self, message: RobotInboundMessage) -> bool:
        metadata = message.reply_target.metadata
        bot_self_ids = self._bot_self_ids_from_message(message)
        if not bot_self_ids:
            return bool(metadata.get("mentioned_bot"))

        mention_ids: set[str] = set()
        mentions = metadata.get("mentions")
        if isinstance(mentions, list):
            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                mention_id = str(
                    mention.get("qq") or mention.get("id") or mention.get("user_id") or ""
                ).strip()
                if mention_id:
                    mention_ids.add(mention_id)

        native_message = metadata.get("message")
        if isinstance(native_message, dict):
            segments = native_message.get("segments")
            if isinstance(segments, list):
                for segment in segments:
                    if not isinstance(segment, dict):
                        continue
                    if str(segment.get("type") or "").strip().lower() != "at":
                        continue
                    data = segment.get("data")
                    if not isinstance(data, dict):
                        continue
                    mention_id = str(
                        data.get("qq") or data.get("id") or data.get("user_id") or ""
                    ).strip()
                    if mention_id:
                        mention_ids.add(mention_id)
            raw_message = str(native_message.get("raw_message") or "")
            for match in re.finditer(r"\[CQ:at,([^\]]+)\]", raw_message):
                params = self._parse_cq_params(match.group(1))
                mention_id = str(
                    params.get("qq") or params.get("id") or params.get("user_id") or ""
                ).strip()
                if mention_id:
                    mention_ids.add(mention_id)

        for match in re.finditer(r"\[CQ:at,([^\]]+)\]", message.text or ""):
            params = self._parse_cq_params(match.group(1))
            mention_id = str(
                params.get("qq") or params.get("id") or params.get("user_id") or ""
            ).strip()
            if mention_id:
                mention_ids.add(mention_id)

        if mention_ids:
            return bool(mention_ids.intersection(bot_self_ids))

        return self._conversation_message_type(message) not in {
            REPLY_MESSAGE_TYPE_GROUP,
            REPLY_MESSAGE_TYPE_CHANNEL,
        } and bool(metadata.get("mentioned_bot"))

    def _strip_leading_bot_mentions_for_command(
        self,
        message: RobotInboundMessage,
        text: str,
    ) -> str:
        normalized = (text or "").strip()
        bot_self_ids = self._bot_self_ids_from_message(message)
        if not normalized or not bot_self_ids:
            return normalized

        while True:
            match = re.match(r"^\[CQ:at,([^\]]+)\]\s*", normalized)
            if not match:
                return normalized
            params = self._parse_cq_params(match.group(1))
            mention_id = str(
                params.get("qq") or params.get("id") or params.get("user_id") or ""
            ).strip()
            if mention_id not in bot_self_ids:
                return normalized
            normalized = normalized[match.end():].lstrip()

    def _parse_cq_params(self, raw_params: str) -> dict[str, str]:
        params: dict[str, str] = {}
        for part in raw_params.split(","):
            key, separator, value = part.partition("=")
            if separator:
                params[key.strip()] = value.strip()
        return params
    def _parse_robot_command(self, text: str) -> RobotCommand:
        normalized = (text or "").strip()
        if not normalized:
            return RobotCommand(mode="chat", target=None, text="")

        command_match = re.match(
            r"^/(term|item|terminal)\s+(\S+)(?:\s+(.*))?$",
            normalized,
            flags=re.IGNORECASE,
        )
        if command_match:
            return RobotCommand(
                mode="term",
                target=command_match.group(2),
                text=(command_match.group(3) or "").strip(),
            )

        hash_match = re.match(r"^#(\S+)(?:\s+(.*))?$", normalized)
        if hash_match:
            return RobotCommand(
                mode="term",
                target=hash_match.group(1),
                text=(hash_match.group(2) or "").strip(),
            )

        return RobotCommand(mode="chat", target=None, text=normalized)

    def _allowed_reply_message_types(self, robot: Robot) -> set[str]:
        config = robot.config if isinstance(robot.config, dict) else {}
        options = config.get("options") if isinstance(config.get("options"), dict) else {}
        if "reply_message_types" not in options:
            return set(DEFAULT_REPLY_MESSAGE_TYPES)

        raw_types = options.get("reply_message_types")
        if not isinstance(raw_types, list):
            return set(DEFAULT_REPLY_MESSAGE_TYPES)

        return {
            normalized_type
            for value in raw_types
            if isinstance(value, str)
            for normalized_type in {value.strip().lower()}
            if normalized_type in ALLOWED_REPLY_MESSAGE_TYPES
        }

    def _mention_match_mode(self, robot: Robot) -> str:
        config = robot.config if isinstance(robot.config, dict) else {}
        options = config.get("options") if isinstance(config.get("options"), dict) else {}
        raw_mode = options.get("mention_match_mode")
        if not isinstance(raw_mode, str):
            return DEFAULT_MENTION_MATCH_MODE

        normalized_mode = raw_mode.strip().lower()
        if normalized_mode in ALLOWED_MENTION_MATCH_MODES:
            return normalized_mode
        return DEFAULT_MENTION_MATCH_MODE

    def _reply_context_window_seconds(self, robot: Robot) -> int:
        config = robot.config if isinstance(robot.config, dict) else {}
        options = config.get("options") if isinstance(config.get("options"), dict) else {}
        raw_value = options.get("reply_context_window_seconds")
        try:
            return max(0, int(raw_value if raw_value is not None else DEFAULT_REPLY_CONTEXT_WINDOW_SECONDS))
        except (TypeError, ValueError):
            return max(0, DEFAULT_REPLY_CONTEXT_WINDOW_SECONDS)

    def _image_recognition_enabled(self, robot: Robot) -> bool:
        """图识开关：开启后，被触发的消息里附带的图片会下载并传给多模态模型。"""
        config = robot.config if isinstance(robot.config, dict) else {}
        options = config.get("options") if isinstance(config.get("options"), dict) else {}
        raw_value = options.get("image_recognition")
        if isinstance(raw_value, bool):
            return raw_value
        return str(raw_value or "").strip().lower() in {"1", "true", "yes", "on", "启用", "开"}

    def _cache_inbound_images(self, text: str) -> list[str]:
        """入站图片立即下载落盘（NapCat 的 url 会过期且对浏览器防盗链），
        返回本地公开路径列表，用于历史展示与（开关开启时）喂给模型。"""
        urls = self._extract_cq_image_urls(text)
        if not urls:
            return []
        from .image_store import public_image_path, store_image_from_url

        cached: list[str] = []
        for url in urls:
            name = store_image_from_url(url)
            if name:
                cached.append(public_image_path(name))
        return cached

    @staticmethod
    def _extract_cq_image_urls(text: str) -> list[str]:
        """从 raw_message 的 [CQ:image,...,url=...] 段提取图片地址。
        CQ 码参数按规范做了 HTML 转义（& -> &amp;），需要反转义还原真实 URL。"""
        import html as _html

        urls: list[str] = []
        for match in re.finditer(r"\[CQ:image,([^\]]*)\]", str(text or "")):
            params = match.group(1)
            url_match = re.search(r"(?:^|,)url=([^,\]]+)", params)
            if url_match:
                url = _html.unescape(url_match.group(1).strip())
                if url.startswith(("http://", "https://")):
                    urls.append(url)
        return urls

    def _wake_words(self, robot: Robot) -> list[str]:
        config = robot.config if isinstance(robot.config, dict) else {}
        options = config.get("options") if isinstance(config.get("options"), dict) else {}
        raw = options.get("wake_words")
        words: list[str] = []
        if isinstance(raw, str):
            words = [part.strip() for part in raw.replace("，", ",").split(",")]
        elif isinstance(raw, list):
            words = [str(part).strip() for part in raw]
        return [word for word in words if word]

    def _message_contains_wake_word(self, message: RobotInboundMessage, robot: Robot) -> bool:
        text = (message.text or "").strip().lower()
        if not text:
            return False
        return any(word.lower() in text for word in self._wake_words(robot))

    def _message_directly_addresses_bot(self, message: RobotInboundMessage, robot: Robot | None = None) -> bool:
        if self._conversation_message_type(message) == REPLY_MESSAGE_TYPE_PRIVATE:
            return True
        if bool(message.reply_target.metadata.get("replied_to_bot")):
            return True
        if self._message_mentions_bot_self_id(message):
            return True
        if robot is not None and self._message_contains_wake_word(message, robot):
            return True
        return False

    def _reply_context_key(
        self,
        robot: Robot,
        message: RobotInboundMessage,
    ) -> tuple[str, str]:
        return (str(robot.id), self._conversation_key(message))

    def _message_conversation_parts(
        self,
        message: RobotInboundMessage,
    ) -> tuple[str, str]:
        metadata = message.reply_target.metadata
        target_data = metadata.get("target")
        if not isinstance(target_data, dict):
            target_data = {}
        sender_data = metadata.get("sender")
        if not isinstance(sender_data, dict):
            sender_data = {}
        conversation_data = metadata.get("conversation")
        if not isinstance(conversation_data, dict):
            conversation_data = {}

        conversation_type = str(
            conversation_data.get("type")
            or conversation_data.get("conversation_type")
            or ""
        ).strip().lower()
        conversation_id = str(
            conversation_data.get("id")
            or conversation_data.get("conversation_id")
            or ""
        ).strip()
        if conversation_type and conversation_id:
            return conversation_type, conversation_id

        message_type = str(
            target_data.get("message_type")
            or sender_data.get("message_type")
            or ""
        ).strip().lower()
        group_id = str(
            target_data.get("group_id")
            or conversation_data.get("group_id")
            or ""
        ).strip()
        user_id = str(
            target_data.get("user_id")
            or sender_data.get("user_id")
            or conversation_data.get("user_id")
            or ""
        ).strip()

        if message_type == "private" or bool(target_data.get("private")):
            return REPLY_MESSAGE_TYPE_PRIVATE, (
                user_id
                or str(target_data.get("id") or message.reply_target.target_id or "").strip()
                or message.sender_key
            )
        if message_type == "group" or group_id:
            return REPLY_MESSAGE_TYPE_GROUP, (
                group_id
                or str(
                    target_data.get("parent_id")
                    or target_data.get("id")
                    or message.reply_target.target_id
                    or ""
                ).strip()
                or message.sender_key
            )
        if bool(target_data.get("channel")):
            return REPLY_MESSAGE_TYPE_CHANNEL, (
                str(
                    target_data.get("parent_id")
                    or target_data.get("id")
                    or message.reply_target.target_id
                    or ""
                ).strip()
                or message.sender_key
            )

        target_type = (message.reply_target.target_type or "").strip().lower()
        if target_type in {"private", "c2c", "direct", "direct_message", "friend"}:
            return REPLY_MESSAGE_TYPE_PRIVATE, (
                str(message.reply_target.target_id or "").strip()
                or user_id
                or message.sender_key
            )
        if target_type in {"channel", "guild", "guild_channel"}:
            return REPLY_MESSAGE_TYPE_CHANNEL, (
                str(message.reply_target.target_id or "").strip() or message.sender_key
            )
        if target_type == "group":
            return REPLY_MESSAGE_TYPE_GROUP, (
                str(message.reply_target.target_id or "").strip() or message.sender_key
            )

        if ":private:" in message.sender_key:
            tail = message.sender_key.rsplit(":private:", 1)[-1].split(":", 1)[0]
            return REPLY_MESSAGE_TYPE_PRIVATE, tail or message.sender_key
        if ":channel:" in message.sender_key:
            tail = message.sender_key.rsplit(":channel:", 1)[-1].split(":", 1)[0]
            return REPLY_MESSAGE_TYPE_CHANNEL, tail or message.sender_key
        if ":group:" in message.sender_key:
            tail = message.sender_key.rsplit(":group:", 1)[-1].split(":", 1)[0]
            return REPLY_MESSAGE_TYPE_GROUP, tail or message.sender_key
        return REPLY_MESSAGE_TYPE_GROUP, message.sender_key

    def _conversation_key(self, message: RobotInboundMessage) -> str:
        conversation_type, conversation_id = self._message_conversation_parts(message)
        return f"{conversation_type}:{conversation_id or message.sender_key}"

    def _conversation_key_from_target(
        self,
        target: RobotReplyTarget,
        sender_key: str,
    ) -> str:
        target_type = str(target.target_type or "").strip().lower()
        target_id = str(target.target_id or "").strip()
        if target_type == "private":
            return f"{REPLY_MESSAGE_TYPE_PRIVATE}:{target_id or sender_key}"
        if target_type == "channel":
            return f"{REPLY_MESSAGE_TYPE_CHANNEL}:{target_id or sender_key}"
        if target_type == "group":
            return f"{REPLY_MESSAGE_TYPE_GROUP}:{target_id or sender_key}"
        if target_type == "command":
            return f"{REPLY_MESSAGE_TYPE_COMMAND}:{target_id or sender_key}"
        return f"{target_type or REPLY_MESSAGE_TYPE_GROUP}:{target_id or sender_key}"

    def _conversation_controller_key(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> tuple[str, str]:
        return (str(robot_id), conversation_key)

    def _controller_is_awake_locked(
        self,
        controller: RobotConversationController,
        now: datetime,
    ) -> bool:
        if controller.sleeping:
            return False
        if controller.processing:
            processing_expires_at = controller.processing_expires_at or (
                controller.updated_at
                + timedelta(seconds=CONVERSATION_PROCESSING_MAX_TIMEOUT_SECONDS)
            )
            if processing_expires_at <= now:
                controller.expires_at = None
                controller.processing_expires_at = None
                controller.processing = False
                controller.sleeping = True
                controller.generation += 1
                controller.updated_at = now
                return False
            return True
        if controller.expires_at is None:
            return False
        if controller.expires_at <= now:
            controller.expires_at = None
            controller.processing_expires_at = None
            controller.sleeping = True
            controller.generation += 1
            controller.updated_at = now
            return False
        return True

    def _get_or_create_controller_locked(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
        now: datetime,
    ) -> RobotConversationController:
        key = self._conversation_controller_key(robot_id, conversation_key)
        controller = self._conversation_controllers.get(key)
        if controller is None:
            controller = RobotConversationController(generation=0, updated_at=now)
            self._conversation_controllers[key] = controller
        return controller

    def _record_conversation_controller_inbound(
        self,
        robot: Robot,
        message: RobotInboundMessage,
        conversation_key: str,
        message_text: str,
        *,
        direct_reply_trigger: bool = False,
    ) -> RobotConversationGate:
        now = self._now()
        with self._lock:
            self._prune_conversation_controllers_locked(now)
            controller = self._get_or_create_controller_locked(
                robot.id,
                conversation_key,
                now,
            )
            if direct_reply_trigger and controller.sleeping:
                controller.sleeping = False
                controller.generation += 1
            active = self._controller_is_awake_locked(controller, now)
            if not active and not direct_reply_trigger:
                controller.sleeping = True
            controller.updated_at = now
            return RobotConversationGate(
                active=active,
                generation=controller.generation,
                sleeping=controller.sleeping,
                processing=bool(controller.processing and not controller.sleeping),
            )

    def _begin_reply_context_dispatch(
        self,
        robot: Robot,
        conversation_key: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> int:
        now = self._now()
        with self._lock:
            self._prune_conversation_controllers_locked(now)
            controller = self._get_or_create_controller_locked(
                robot.id,
                conversation_key,
                now,
            )
            controller.generation += 1
            controller.sleeping = False
            controller.processing = True
            controller.processing_expires_at = now + timedelta(
                seconds=self._controller_processing_timeout_seconds(robot)
            )
            controller.expires_at = None
            controller.updated_at = now
            generation = controller.generation
        metadata = metadata or {}
        record_robot_event(
            str(robot.id),
            direction="backend",
            event="conversation_controller_processing",
            payload={
                "conversation": conversation_key,
                "generation": generation,
                "mentioned_bot": bool(metadata.get("mentioned_bot")),
                "replied_to_bot": bool(metadata.get("replied_to_bot")),
            },
        )
        return generation

    def _is_reply_context_active(
        self,
        robot: Robot,
        message: RobotInboundMessage,
    ) -> bool:
        conversation_key = self._conversation_key(message)
        now = self._now()
        with self._lock:
            self._prune_conversation_controllers_locked(now)
            controller = self._conversation_controllers.get(
                self._conversation_controller_key(robot.id, conversation_key)
            )
            return bool(controller and self._controller_is_awake_locked(controller, now))

    def _remember_reply_context_window(
        self,
        robot: Robot,
        message: RobotInboundMessage,
    ) -> None:
        self._remember_reply_context_window_for_key(
            robot,
            self._conversation_key(message),
            metadata=message.reply_target.metadata,
        )

    def _remember_reply_context_window_for_key(
        self,
        robot: Robot,
        conversation_key: str,
        *,
        metadata: dict[str, object] | None = None,
        expected_generation: int | None = None,
    ) -> None:
        window_seconds = self._reply_context_window_seconds(robot)
        if window_seconds <= 0:
            self._clear_reply_context_window_for_key(
                robot.id,
                conversation_key,
                reason="reply_context_window_disabled",
                expected_generation=expected_generation,
            )
            return

        now = self._now()
        expires_at = now + timedelta(seconds=window_seconds)
        key = self._conversation_controller_key(robot.id, conversation_key)
        with self._lock:
            self._prune_conversation_controllers_locked(now)
            controller = self._get_or_create_controller_locked(
                robot.id,
                conversation_key,
                now,
            )
            if expected_generation is not None and controller.generation != expected_generation:
                return
            controller.expires_at = expires_at
            controller.processing_expires_at = None
            controller.sleeping = False
            controller.processing = False
            controller.updated_at = now
            generation = controller.generation
        metadata = metadata or {}
        record_robot_event(
            str(robot.id),
            direction="backend",
            event="reply_context_window_refreshed",
            payload={
                "conversation": key[1],
                "generation": generation,
                "window_seconds": window_seconds,
                "expires_at": expires_at.isoformat(),
                "mentioned_bot": bool(metadata.get("mentioned_bot")),
                "replied_to_bot": bool(metadata.get("replied_to_bot")),
            },
        )

    def _clear_reply_context_window_for_key(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
        *,
        reason: str,
        expected_generation: int | None = None,
    ) -> bool:
        if not conversation_key:
            return False

        key = self._conversation_controller_key(robot_id, conversation_key)
        with self._lock:
            controller = self._conversation_controllers.get(key)
            if controller is None:
                removed = False
            elif expected_generation is not None and controller.generation != expected_generation:
                removed = False
            else:
                was_awake = controller.expires_at is not None or controller.processing
                controller.expires_at = None
                controller.processing_expires_at = None
                controller.processing = False
                controller.sleeping = True
                controller.generation += 1
                controller.updated_at = self._now()
                removed = was_awake
        if removed:
            record_robot_event(
                str(robot_id),
                direction="backend",
                event="reply_context_window_cleared",
                payload={
                    "conversation": conversation_key,
                    "reason": reason,
                },
            )
        return removed

    def sleep_conversation_controller(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
        *,
        reason: str,
    ) -> bool:
        if not conversation_key:
            return False

        now = self._now()
        with self._lock:
            self._prune_conversation_controllers_locked(now)
            controller = self._get_or_create_controller_locked(
                robot_id,
                conversation_key,
                now,
            )
            changed = bool(
                controller.expires_at is not None
                or controller.processing
                or not controller.sleeping
            )
            controller.expires_at = None
            controller.processing_expires_at = None
            controller.processing = False
            controller.sleeping = True
            controller.generation += 1
            controller.updated_at = now
            generation = controller.generation
        record_robot_event(
            str(robot_id),
            direction="backend",
            event="conversation_controller_sleeping",
            payload={
                "conversation": conversation_key,
                "generation": generation,
                "reason": reason,
                "changed": changed,
            },
        )
        return True

    def reply_context_window_seconds(self, robot: Robot) -> int:
        return self._reply_context_window_seconds(robot)

    def conversation_controller_snapshots(
        self,
        robot_ids: set[uuid.UUID | str] | None = None,
        item_ids: set[uuid.UUID | str] | None = None,
    ) -> list[dict[str, object]]:
        allowed_robot_ids = (
            {str(robot_id) for robot_id in robot_ids} if robot_ids else None
        )
        allowed_item_ids = {str(item_id) for item_id in item_ids} if item_ids else None
        now = self._now()
        snapshots: list[dict[str, object]] = []
        with self._lock:
            self._prune_conversation_controllers_locked(now)
            for (
                robot_id,
                conversation_key,
            ), controller in self._conversation_controllers.items():
                if allowed_robot_ids is not None and robot_id not in allowed_robot_ids:
                    continue
                route_state = self._conversation_routes.get((robot_id, conversation_key))
                route_item_id = route_state.item_id if route_state else None
                pending_messages, pending_item_id = self._pending_chat_snapshot_locked(
                    robot_id,
                    conversation_key,
                )
                effective_item_id = route_item_id or pending_item_id
                if (
                    allowed_item_ids is not None
                    and str(effective_item_id or "") not in allowed_item_ids
                ):
                    continue
                awake = self._controller_is_awake_locked(controller, now)
                processing = bool(controller.processing and not controller.sleeping)
                expires_at = controller.expires_at
                seconds_remaining = (
                    max(0, ceil((expires_at - now).total_seconds()))
                    if expires_at is not None
                    else 0
                )
                processing_expires_at = controller.processing_expires_at
                processing_seconds_remaining = (
                    max(0, ceil((processing_expires_at - now).total_seconds()))
                    if processing and processing_expires_at is not None
                    else 0
                )
                conversation_type, _, conversation_id = conversation_key.partition(":")
                snapshots.append(
                    {
                        "robot_id": robot_id,
                        "conversation_key": conversation_key,
                        "conversation_type": conversation_type,
                        "conversation_id": conversation_id,
                        "item_id": str(effective_item_id) if effective_item_id else None,
                        "status": (
                            "processing"
                            if processing
                            else "awake" if awake else "sleeping"
                        ),
                        "awake": awake,
                        "sleeping": not awake,
                        "processing": processing,
                        "generation": controller.generation,
                        "expires_at": expires_at.isoformat() if expires_at else None,
                        "processing_expires_at": (
                            processing_expires_at.isoformat()
                            if processing_expires_at
                            else None
                        ),
                        "processing_seconds_remaining": processing_seconds_remaining,
                        "updated_at": controller.updated_at.isoformat(),
                        "seconds_remaining": seconds_remaining,
                        "pending_count": len(pending_messages),
                        "pending_messages": pending_messages,
                    }
                )

        snapshots.sort(
            key=lambda item: str(item.get("updated_at") or ""),
            reverse=True,
        )
        return snapshots

    def _apply_reply_context_result(
        self,
        robot: Robot,
        conversation_key: str,
        *,
        robot_message_sent: bool,
        reply_target: RobotReplyTarget,
        conversation_generation: int | None = None,
        sleep_when_no_reply: bool = False,
    ) -> None:
        if robot_message_sent:
            self._remember_reply_context_window_for_key(
                robot,
                conversation_key,
                metadata=reply_target.metadata,
                expected_generation=conversation_generation,
            )
            return

        if sleep_when_no_reply:
            self.sleep_conversation_controller(
                robot.id,
                conversation_key,
                reason="agent_no_reply_active_chat_window",
            )
            return

        self._clear_reply_context_window_for_key(
            robot.id,
            conversation_key,
            reason="agent_did_not_send_qq_message",
            expected_generation=conversation_generation,
        )

    def _session_has_running_terminal_jobs(self, item_id: str) -> bool:
        """In-process check for running terminal jobs (no daemon round-trip).

        The daemon list lookup races with freshly started jobs; the session
        bookkeeping is updated synchronously when run_job starts, so it is
        the reliable signal right after a dispatch turn.
        """
        try:
            from app.services.agent.session import agent_session_manager

            session = agent_session_manager.get_session(item_id)
            return bool(session and session.has_running_terminal_job())
        except Exception:
            return False

    def _item_has_running_jobs(self, item_id: str) -> bool:
        """Check if the daemon has running background jobs for this item."""
        try:
            from app.services.agent.mcp.local_server import local_mcp_server

            _item, connection = local_mcp_server._get_item_daemon_context(item_id)
            result = connection.list_jobs_http(item_uuid=item_id)
            if not isinstance(result, dict) or not result.get("success"):
                return False
            for job in result.get("jobs") or []:
                if isinstance(job, dict) and str(job.get("status") or "") == "running":
                    return True
            return False
        except Exception:
            return False

    def _extend_processing_for_active_jobs(
        self,
        robot: Robot,
        conversation_key: str,
    ) -> None:
        """Keep conversation in processing state while background jobs run."""
        key = self._conversation_controller_key(robot.id, conversation_key)
        now = self._now()
        timeout = timedelta(seconds=CONVERSATION_PROCESSING_MAX_TIMEOUT_SECONDS)
        with self._lock:
            controller = self._conversation_controllers.get(key)
            if controller is None:
                return
            controller.processing = True
            controller.processing_expires_at = now + timeout
            controller.updated_at = now
        logger.info(
            "[RobotService] Extended processing for active jobs: conversation=%s timeout=%ss",
            conversation_key,
            CONVERSATION_PROCESSING_MAX_TIMEOUT_SECONDS,
        )

    def conversation_is_sleeping(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> bool:
        key = self._conversation_controller_key(robot_id, conversation_key)
        with self._lock:
            controller = self._conversation_controllers.get(key)
            return bool(controller and controller.sleeping)

    def conversation_is_processing(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> bool:
        key = self._conversation_controller_key(robot_id, conversation_key)
        with self._lock:
            controller = self._conversation_controllers.get(key)
            return bool(controller and controller.processing)

    def conversation_has_active_dispatch(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> bool:
        """Whether a dispatch job for this conversation is running or queued.

        This is the authoritative "a turn is in flight" signal used to decide
        whether background job results should be buffered and merged into the
        running turn instead of starting a new turn immediately. Unlike the
        raw controller `processing` flag, an active/queued dispatch job
        guarantees the turn-end flush hook will drain the buffer, so buffered
        results can never linger.
        """
        if not conversation_key:
            return False
        robot_id_text = str(robot_id)
        with self._lock:
            for _tracking_key, (_started, queued_job) in self._active_dispatch_jobs.items():
                if (
                    str(queued_job.robot_id) == robot_id_text
                    and queued_job.conversation_key == conversation_key
                ):
                    return True
        try:
            with self._dispatch_queue.mutex:
                queued_snapshot = list(self._dispatch_queue.queue)
        except Exception:
            queued_snapshot = []
        for queued_job in queued_snapshot:
            if (
                str(queued_job.robot_id) == robot_id_text
                and queued_job.conversation_key == conversation_key
            ):
                return True
        return False

    def _release_processing_hold(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
    ) -> None:
        """Clear the job-scoped processing hold once no background jobs remain.

        Callback rounds carry a stale conversation generation, so the normal
        reply-context bookkeeping (guarded by generation) never clears the
        processing flag set by `_extend_processing_for_active_jobs`. Without
        this release the conversation stays "processing" until the hard
        timeout, and new inbound messages pile up in the pending queue with
        no turn to drain them.
        """
        if not conversation_key:
            return
        key = self._conversation_controller_key(robot_id, conversation_key)
        with self._lock:
            controller = self._conversation_controllers.get(key)
            if controller is None or not controller.processing:
                return
            controller.processing = False
            controller.processing_expires_at = None
            controller.updated_at = self._now()

    def conversation_controller_allows_reply(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
        conversation_generation: int,
        *,
        requires_awake: bool,
    ) -> bool:
        if not requires_awake:
            return True
        if not conversation_key or conversation_generation <= 0:
            return False

        now = self._now()
        key = self._conversation_controller_key(robot_id, conversation_key)
        with self._lock:
            self._prune_conversation_controllers_locked(now)
            controller = self._conversation_controllers.get(key)
            if controller is None or controller.generation != conversation_generation:
                return False
            return self._controller_is_awake_locked(controller, now)

    def conversation_controller_allows_completion_reply(
        self,
        robot_id: uuid.UUID | str,
        conversation_key: str,
        conversation_generation: int,
        *,
        requires_awake: bool,
    ) -> bool:
        if not requires_awake:
            return True
        if not conversation_key or conversation_generation <= 0:
            return False

        now = self._now()
        key = self._conversation_controller_key(robot_id, conversation_key)
        with self._lock:
            self._prune_conversation_controllers_locked(now)
            controller = self._conversation_controllers.get(key)
            if controller is None or controller.generation != conversation_generation:
                return False
            if controller.sleeping:
                return False
            if controller.processing:
                return True
            return self._controller_is_awake_locked(controller, now)

    def _controller_processing_timeout_seconds(self, robot: Robot) -> int:
        window_seconds = max(0, self._reply_context_window_seconds(robot))
        return max(
            CONVERSATION_PROCESSING_MIN_TIMEOUT_SECONDS,
            min(CONVERSATION_PROCESSING_MAX_TIMEOUT_SECONDS, window_seconds * 12),
        )

    def _prune_conversation_controllers_locked(self, now: datetime) -> None:
        expired_after = now - CONVERSATION_TTL
        expired_keys = [
            key
            for key, controller in self._conversation_controllers.items()
            if controller.updated_at < expired_after
        ]
        for key in expired_keys:
            self._conversation_controllers.pop(key, None)

    def _reply_message_categories(
        self,
        message: RobotInboundMessage,
        command: RobotCommand,
        *,
        direct_reply_trigger: bool = False,
        reply_context_active: bool = False,
        mention_match_mode: str = DEFAULT_MENTION_MATCH_MODE,
    ) -> set[str]:
        message_type = self._conversation_message_type(message)
        categories: set[str] = set()
        if message_type == REPLY_MESSAGE_TYPE_PRIVATE:
            categories.add(REPLY_MESSAGE_TYPE_PRIVATE)
        elif message_type == REPLY_MESSAGE_TYPE_GROUP:
            categories.add(REPLY_MESSAGE_TYPE_GROUP)
        elif message_type == REPLY_MESSAGE_TYPE_CHANNEL:
            categories.add(REPLY_MESSAGE_TYPE_CHANNEL)
        if command.mode != "chat" or command.target:
            categories.add(REPLY_MESSAGE_TYPE_COMMAND)
        if direct_reply_trigger or reply_context_active:
            categories.add(REPLY_MESSAGE_TYPE_MENTION)
        return categories

    def _conversation_message_type(self, message: RobotInboundMessage) -> str:
        return self._message_conversation_parts(message)[0]

    def _message_has_mentions(self, message: RobotInboundMessage) -> bool:
        mentions = message.reply_target.metadata.get("mentions")
        if not isinstance(mentions, list):
            return False
        return any(
            isinstance(mention, dict)
            and bool(str(mention.get("id") or mention.get("qq") or "").strip())
            for mention in mentions
        )

    @staticmethod
    def _memory_timestamp_label(memory: dict[str, object]) -> str:
        metadata = memory.get("metadata") if isinstance(memory, dict) else {}
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("updated_at") or metadata.get("created_at") or "")

    @staticmethod
    def _memory_expired(memory: dict[str, object]) -> bool:
        metadata = memory.get("metadata") if isinstance(memory, dict) else {}
        if not isinstance(metadata, dict):
            return False
        if str(metadata.get("memory_type") or "") in {"preference", "error"}:
            return False
        expires_at = metadata.get("expires_at")
        if not expires_at:
            return False
        try:
            parsed = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        except ValueError:
            return False
        now = datetime.now(parsed.tzinfo) if parsed.tzinfo else datetime.now()
        return parsed < now

    @staticmethod
    def _memory_inactive(memory: dict[str, object]) -> bool:
        metadata = memory.get("metadata") if isinstance(memory, dict) else {}
        if not isinstance(metadata, dict):
            return False
        memory_type = str(metadata.get("memory_type") or "")
        status = str(metadata.get("status") or "").lower()
        return memory_type == "task" and status == "completed"

    @staticmethod
    def _memory_matches_robot_conversation(
        memory: dict[str, object],
        *,
        robot: Robot,
        conversation_key: str,
        speaker_global_key: str = "",
    ) -> bool:
        if not isinstance(memory, dict) or not conversation_key:
            return False
        return memory_scope_rank(
            memory,
            robot_id=str(robot.id),
            conversation_key=conversation_key,
            speaker_global_key=speaker_global_key,
        ) > 0

    @staticmethod
    def _impression_memory_sort_key(memory: dict[str, object]) -> tuple[int, int, str]:
        metadata = memory.get("metadata") if isinstance(memory, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        memory_type = str(metadata.get("memory_type") or "fact")
        type_rank = {
            "preference": 5,
            "fact": 4,
            "context": 3,
            "error": 2,
        }.get(memory_type, 1)
        verified_rank = 1 if metadata.get("verified") is True else 0
        updated_at = str(metadata.get("updated_at") or metadata.get("created_at") or "")
        return type_rank, verified_rank, updated_at

    def _conversation_impression_card(
        self,
        *,
        item_id: uuid.UUID | str,
        robot: Robot,
        conversation_key: str,
        sender_key: str = "",
        reply_target: RobotReplyTarget | None = None,
        query: str = "",
    ) -> str:
        if not conversation_key:
            return ""
        try:
            from app.plugins.robot.internal_trace import sanitize_robot_visible_text
            from app.plugins.robot.memory_migration import (
                ensure_legacy_robot_memories_upgraded,
            )
            from app.services.agent.memory.vector_store import vector_store
        except Exception:
            logger.debug("[RobotService] Robot impression card dependencies unavailable")
            return ""

        speaker_global_key = speaker_global_key_from_context(sender_key, reply_target)
        from app.services.agent.memory.scope import resolve_handler_id

        memory_scope = resolve_handler_id(str(item_id))
        ensure_legacy_robot_memories_upgraded(memory_scope, store=vector_store)
        vector_store.maintain_memories(memory_scope)
        try:
            all_memories = vector_store.get_all_memories(memory_scope)
        except Exception as exc:
            logger.debug(
                "[RobotService] Failed to load robot impression memories scope=%s: %s",
                memory_scope,
                exc,
            )
            all_memories = []

        scoped_memories = [
            memory
            for memory in all_memories
            if isinstance(memory, dict)
            and not memory_content_is_question_like(str(memory.get("content") or ""))
            and not self._memory_expired(memory)
            and not self._memory_inactive(memory)
            and self._memory_matches_robot_conversation(
                memory,
                robot=robot,
                conversation_key=conversation_key,
                speaker_global_key=speaker_global_key,
            )
        ]
        if not scoped_memories:
            return ""

        query_text = re.sub(r"\s+", " ", str(query or "")).strip()
        relevant_memories: list[dict[str, object]] = []
        if query_text:
            try:
                recalled = vector_store.search_memories(
                    handler_id=memory_scope,
                    query=query_text,
                    n_results=max(1, len(scoped_memories)),
                    include_expired=False,
                    active_only=True,
                    min_similarity=0.18,
                )
            except Exception as exc:
                logger.debug(
                    "[RobotService] Failed to query robot impression memories item=%s: %s",
                    item_id,
                    exc,
                )
                recalled = []
            relevant_memories = [
                memory
                for memory in recalled
                if isinstance(memory, dict)
                and not memory_content_is_question_like(str(memory.get("content") or ""))
                and self._memory_matches_robot_conversation(
                    memory,
                    robot=robot,
                    conversation_key=conversation_key,
                    speaker_global_key=speaker_global_key,
                )
            ]

        scoped_memories.sort(
            key=lambda memory: (
                memory_scope_rank(
                    memory,
                    robot_id=str(robot.id),
                    conversation_key=conversation_key,
                    speaker_global_key=speaker_global_key,
                ),
                *self._impression_memory_sort_key(memory),
            ),
            reverse=True,
        )
        relevant_memories.sort(
            key=lambda memory: (
                max(
                    0.0,
                    1.0
                    - float(
                        memory.get("distance")
                        if memory.get("distance") is not None
                        else 1.0
                    ),
                ),
                memory_scope_rank(
                    memory,
                    robot_id=str(robot.id),
                    conversation_key=conversation_key,
                    speaker_global_key=speaker_global_key,
                ),
                *self._impression_memory_sort_key(memory),
            ),
            reverse=True,
        )

        stable_memories = [
            memory
            for memory in scoped_memories
            if str((memory.get("metadata") or {}).get("memory_type") or "")
            in {"preference", "error"}
            or (memory.get("metadata") or {}).get("verified") is True
        ]
        ordered_memories = [*stable_memories, *relevant_memories, *scoped_memories]
        lines = [
            "[Current QQ conversation impression card; background only, do not answer old items]"
        ]
        seen_content: set[str] = set()
        seen_ids: set[str] = set()
        for memory in ordered_memories:
            memory_id = str(memory.get("id") or "")
            if memory_id and memory_id in seen_ids:
                continue
            metadata = memory.get("metadata") if isinstance(memory, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            content = sanitize_robot_visible_text(str(memory.get("content") or "")).strip()
            content = re.sub(r"\s+", " ", content)
            content_key = content.casefold()
            if (
                not content
                or content_key in seen_content
                or (query_text and content.casefold() == query_text.casefold())
            ):
                continue
            if memory_id:
                seen_ids.add(memory_id)
            seen_content.add(content_key)
            if len(content) > 160:
                content = f"{content[:157]}..."
            sender = str(
                metadata.get("speaker")
                or metadata.get("speaker_label")
                or metadata.get("sender")
                or ""
            ).strip()
            prefix = f"{sender}: " if sender and not content.startswith(sender) else ""
            lines.append(f"- {prefix}{content}")
        if len(lines) == 1:
            return ""
        return "\n".join(lines)
    def _message_requests_conversation_sleep(self, text: str) -> bool:
        normalized = (text or "").strip().casefold()
        if not normalized:
            return False
        compact = re.sub(
            r"[\s\uFF0C\u3002\uFF01\uFF1F!,.\u3001~\uFF5E\u2026]+",
            "",
            normalized,
        )
        return compact in {
            "/sleep",
            "sleep",
            "gosleep",
            "stopreplying",
            "shutup",
            "\u4f11\u7720",
            "\u4f11\u7720\u53bb",
            "\u53bb\u4f11\u7720",
            "\u7761\u89c9",
            "\u7761\u89c9\u53bb",
            "\u95ed\u5634",
            "\u522b\u8bf4\u8bdd",
            "\u522b\u56de\u4e86",
            "\u522b\u56de\u590d\u4e86",
            "\u4e0d\u7528\u56de\u4e86",
            "\u4e0d\u8981\u56de\u4e86",
            "\u505c\u6b62\u56de\u590d",
            "\u505c\u4e00\u4e0b",
            "\u5148\u522b\u56de",
            "\u5148\u522b\u56de\u590d",
            "\u5b89\u9759",
        }

    def _agent_message_with_context(
        self,
        inbound_message: RobotInboundMessage,
        message_text: str,
        *,
        trigger_reason: str = "",
        impression_card: str = "",
        live_context_card: str = "",
    ) -> str:
        prefix = self._agent_message_context_prefix(
            inbound_message,
            trigger_reason=trigger_reason,
        )
        if not prefix:
            return message_text
        parts = [prefix]
        sender_identity_card = self._agent_sender_identity_context_card(inbound_message)
        if sender_identity_card:
            parts.append(sender_identity_card)
        identity_card = self._agent_identity_context_card(
            inbound_message,
            trigger_reason=trigger_reason,
        )
        if identity_card:
            parts.append(identity_card)
        if impression_card.strip():
            parts.append(impression_card.strip())
        if live_context_card.strip():
            parts.append(live_context_card.strip())
        reply_reference_card = self._agent_reply_reference_context_card(inbound_message)
        if reply_reference_card:
            parts.append(reply_reference_card)
        parts.append(_TURN_ACTION_FRAME)
        parts.append(f"[Current QQ message]\n{message_text}")
        return "\n".join(parts)

    def _agent_sender_identity_context_card(
        self,
        message: RobotInboundMessage,
    ) -> str:
        sender_data = message.reply_target.metadata.get("sender")
        if not isinstance(sender_data, dict):
            return ""

        sender_id = str(
            sender_data.get("user_id")
            or sender_data.get("qq")
            or sender_data.get("id")
            or ""
        ).strip()
        display_name = str(
            sender_data.get("display_name")
            or sender_data.get("card")
            or sender_data.get("nickname")
            or sender_id
            or "unknown"
        ).strip()
        sender_label = display_name
        if sender_id and sender_id != display_name:
            sender_label = f"{display_name} ({sender_id})"

        return (
            f"[Current QQ sender: {sender_label}; 本条消息里的"
            "“我/我的/我是谁”都指这个人，不指你或上下文其他人]"
        )

    def _agent_reply_reference_context_card(
        self,
        message: RobotInboundMessage,
    ) -> str:
        reply_data = message.reply_target.metadata.get("reply")
        if not isinstance(reply_data, dict):
            return ""

        lines = [
            "[Replied QQ message; background only, use it to resolve references in the current message]",
        ]
        message_id = str(
            reply_data.get("message_id") or reply_data.get("id") or ""
        ).strip()
        if message_id:
            lines.append(f"- message_id: {message_id}")

        sender_data = reply_data.get("sender")
        if isinstance(sender_data, dict):
            sender_label = self._reply_sender_label_for_context(sender_data)
            if sender_label:
                lines.append(f"- sender: {sender_label}")

        text = str(
            reply_data.get("text")
            or reply_data.get("plain_text")
            or reply_data.get("raw_message")
            or ""
        ).strip()
        if text:
            lines.append(f"- text: {text[:600].rstrip()}")
        elif message_id:
            lines.append("- text: unavailable; only the replied message id was provided")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _reply_sender_label_for_context(self, sender_data: dict[str, object]) -> str:
        sender_id = str(
            sender_data.get("user_id")
            or sender_data.get("qq")
            or sender_data.get("id")
            or ""
        ).strip()
        display_name = str(
            sender_data.get("display_name")
            or sender_data.get("card")
            or sender_data.get("nickname")
            or sender_id
            or ""
        ).strip()
        if sender_id and display_name and sender_id != display_name:
            return f"{display_name} ({sender_id})"
        return display_name or sender_id

    def _agent_identity_context_card(
        self,
        message: RobotInboundMessage,
        *,
        trigger_reason: str = "",
    ) -> str:
        bot_self_ids = sorted(self._bot_self_ids_from_message(message))
        if not bot_self_ids:
            return ""

        self_id_label = ", ".join(bot_self_ids)
        bot_identity = message.reply_target.metadata.get("bot_identity")
        bot_identity = bot_identity if isinstance(bot_identity, dict) else {}
        display_name = str(bot_identity.get("display_name") or "").strip()
        aliases = [
            str(value).strip()
            for value in (bot_identity.get("aliases") or [])
            if str(value or "").strip()
        ]
        identity_lines = [
            "[Robot identity; background only]",
            f"- self_id: {self_id_label} (this QQ id is you, the bot)",
        ]
        if display_name:
            identity_lines.append(f"- QQ display name: {display_name} (this name is you)")
        if aliases:
            identity_lines.append(f"- known QQ names: {', '.join(dict.fromkeys(aliases))}")
        identity_lines.append(
            "- identity_rule: @/回复指向上述 self_id 或名字就是在叫你；"
            "是否直接叫你由消息头的 trigger 字段给出，不要再猜。"
        )
        return "\n".join(identity_lines)

    def _agent_trigger_reason(
        self,
        message: RobotInboundMessage,
        *,
        direct_reply_trigger: bool,
        reply_context_active: bool,
        mention_match_mode: str = DEFAULT_MENTION_MATCH_MODE,
    ) -> str:
        if bool(message.reply_target.metadata.get("replied_to_bot")):
            return "reply_to_bot"
        if bool(message.reply_target.metadata.get("mentioned_bot")):
            return "mention_bot"
        if self._conversation_message_type(message) == REPLY_MESSAGE_TYPE_PRIVATE:
            return "private_chat"
        if reply_context_active and not direct_reply_trigger:
            return "active_chat_window"
        return "plain"

    def _agent_message_context_prefix(
        self,
        message: RobotInboundMessage,
        *,
        trigger_reason: str = "",
    ) -> str:
        sender_data = message.reply_target.metadata.get("sender")
        if not isinstance(sender_data, dict):
            return ""

        target_data = message.reply_target.metadata.get("target")
        if not isinstance(target_data, dict):
            target_data = {}

        conversation_type, conversation_id = self._message_conversation_parts(message)
        sender_id = str(sender_data.get("user_id") or "").strip()
        display_name = str(
            sender_data.get("display_name")
            or sender_data.get("card")
            or sender_data.get("nickname")
            or sender_id
            or "unknown"
        ).strip()

        sender_label = display_name
        if sender_id and sender_id != display_name:
            sender_label = f"{display_name} ({sender_id})"

        conversation_label = conversation_type
        if conversation_id:
            conversation_label = f"{conversation_type}:{conversation_id}"

        parts = [
            "Robot message",
            f"conversation={conversation_label}",
        ]
        if trigger_reason:
            parts.append(f"trigger={trigger_reason}")
        parts.append(f"sender={sender_label}")
        mentions_label = self._format_mentions_for_context(
            message.reply_target.metadata.get("mentions"),
            bot_self_ids=self._bot_self_ids_from_message(message),
            bot_display_name=self._bot_display_name_from_message(message),
        )
        if mentions_label:
            parts.append(f"mentions={mentions_label}")
        return f"[{'; '.join(parts)}]"

    def _format_mentions_for_context(
        self,
        raw_mentions: object,
        *,
        bot_self_ids: set[str] | None = None,
        bot_display_name: str = "",
    ) -> str:
        if not isinstance(raw_mentions, list):
            return ""

        bot_self_ids = bot_self_ids or set()
        labels: list[str] = []
        for raw_mention in raw_mentions[:8]:
            if not isinstance(raw_mention, dict):
                continue
            mention_id = str(
                raw_mention.get("id") or raw_mention.get("qq") or ""
            ).strip()
            name = str(
                raw_mention.get("name")
                or raw_mention.get("display_name")
                or raw_mention.get("nickname")
                or ""
            ).strip()
            is_self = bool(mention_id and mention_id in bot_self_ids)
            if is_self and not name and bot_display_name:
                name = bot_display_name
            if name and mention_id and name != mention_id:
                label = f"{name} ({mention_id})"
            elif name:
                label = name
            elif mention_id:
                label = mention_id
            else:
                continue
            if is_self:
                label = f"{label} (you)"
            labels.append(label)
        return ", ".join(labels)

    @staticmethod
    def _job_request_display(job: QueuedRobotChatJob) -> str:
        """Visible user request behind a dispatch job.

        Direct dispatches carry the sanitized message text; pending-batch
        dispatches carry the latest drained message plus a count of earlier
        batched messages. Internal jobs (background reports) have none.
        """
        text = (job.message_text or "").strip()
        if not text:
            return ""
        earlier = max(len(job.pending_entries) - 1, 0)
        if earlier:
            return f"{text} (+{earlier} earlier)"
        return text

    async def _chat_with_item(
        self,
        *,
        session: Session,
        robot: Robot,
        item: Item,
        message: str,
        sender_key: str,
        reply_target: RobotReplyTarget,
        conversation_key: str = "",
        conversation_generation: int = 0,
        reply_requires_awake: bool = False,
        reply_ticket_id: str = "",
        request_message: str = "",
        image_urls: list[str] | None = None,
        force_image_vision: bool = False,
        agent_image_urls: list[str] | None = None,
    ) -> ChatResponseResult:
        owner = session.get(User, robot.owner_id)
        if owner is None:
            raise RobotServiceError("机器人所属用户不存在", status_code=404)

        try:
            result = await collect_chat_response(
                session=session,
                item_id=str(item.id),
                current_user=owner,
                message=message,
                robot_id=str(robot.id),
                robot_sender_key=sender_key,
                robot_reply_target=reply_target,
                robot_conversation_key=conversation_key,
                robot_conversation_generation=conversation_generation,
                robot_reply_requires_awake=reply_requires_awake,
                reply_ticket_id=reply_ticket_id,
                robot_request_message=request_message,
                return_result=True,
                image_urls=image_urls,
                force_image_vision=force_image_vision,
                agent_image_urls=agent_image_urls,
            )
            if isinstance(result, ChatResponseResult):
                return result
            return ChatResponseResult(content=str(result or ""))
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "Item agent is unavailable"
            raise RobotServiceError(detail, status_code=exc.status_code) from exc

    def _assert_robot_supported(self, robot: Robot) -> None:
        provider = (robot.provider or "").lower()
        if provider != "nonebot2":
            raise RobotServiceError(
                f"Unsupported robot provider `{robot.provider}`",
                status_code=400,
            )

        platform_id = normalize_robot_platform_id(robot.platform or robot.protocol)
        try:
            get_robot_platform(platform_id)
            get_robot_runtime_config(robot)
        except ValueError as exc:
            raise RobotServiceError(str(exc), status_code=400) from exc

    def _remember_conversation(
        self,
        robot_id: uuid.UUID,
        sender_key: str,
        item_id: uuid.UUID,
    ) -> None:
        with self._lock:
            self._prune_locked()
            self._conversation_routes[(str(robot_id), sender_key)] = ConversationState(
                item_id=item_id,
                updated_at=self._now(),
            )

    def _get_remembered_item_id(
        self,
        robot_id: uuid.UUID,
        sender_key: str,
    ) -> uuid.UUID | None:
        with self._lock:
            self._prune_locked()
            state = self._conversation_routes.get((str(robot_id), sender_key))
            return state.item_id if state else None

    def _prune_locked(self) -> None:
        now = self._now()
        expired_after = now - CONVERSATION_TTL

        expired_routes = [
            key
            for key, state in self._conversation_routes.items()
            if state.updated_at < expired_after
        ]
        for key in expired_routes:
            self._conversation_routes.pop(key, None)

    def _chunk_text(self, robot: Robot, text: str) -> list[str]:
        """只在内容超长时按自然边界（段落/换行）兜底切分，不做机械分段——
        口语化短消息由模型自己写（见 ROBOT_HUMAN_STYLE_INSTRUCTION）。"""
        normalized = (text or "").strip()
        if not normalized:
            return []

        limit = self._max_message_length(robot, default=DEFAULT_MAX_MESSAGE_LENGTH)
        if len(normalized) <= limit:
            return [normalized]

        chunks: list[str] = []
        remaining = normalized
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            split_at = remaining.rfind("\n", 0, limit)
            if split_at <= 0:
                split_at = limit
            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()
        return [chunk for chunk in chunks if chunk]

    def _reply_chunks_for_target(
        self,
        robot: Robot,
        target: RobotReplyTarget,
        text: str,
    ) -> list[str]:
        if is_group_reply_target(target):
            return split_robot_message_for_target(target, text)
        return self._chunk_text(robot, text)

    def _max_message_length(self, robot: Robot, *, default: int) -> int:
        config = robot.config if isinstance(robot.config, dict) else {}
        options = config.get("options") if isinstance(config.get("options"), dict) else {}
        try:
            return max(200, int(options.get("max_message_length") or default))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)


robot_service = RobotService()
