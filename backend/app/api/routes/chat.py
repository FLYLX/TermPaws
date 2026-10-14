import asyncio
import json
import logging
import queue
import re
import threading
import time
from collections.abc import Generator
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from litellm import completion
from pydantic import BaseModel

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.core.tool_markup import extract_dsml_tool_calls
from app.models import Item, ItemHandler
from app.services.agent.agent import agent_manager
from app.services.agent.chat_runtime import (
    _get_item_handler_cached,
    get_item_handler_llm_config,
    prepare_chat_agent,
)
from app.services.agent.history.chat import (
    append_chat_message,
    get_previous_assistant_response_before_latest_user_message,
)
from app.services.agent.integrations import (
    extract_integration_context_targets,
    fallback_is_delivery_result,
    record_integration_context_targets,
)
from app.services.agent.memory.vector_store import vector_store
from app.services.agent.pending_context import (
    build_pending_terminal_continuation_prompt,
    clear_pending_terminal_continuation,
    record_pending_terminal_continuation,
)
from app.services.agent.persona_guard import enforce_persona_identity_robot_tool_args
from app.services.agent.prompts.builder import build_chat_turn_messages
from app.services.agent.prompts.policy import (
    build_confirmation_memory_candidate,
    build_conversation_memory_candidate,
    build_status_update_memory_candidate,
    persist_memory_candidate,
)
from app.services.agent.prompts.system import get_system_prompt
from app.services.agent.reply_ticket import (
    SOURCE_QQ,
    SOURCE_WEB,
    reply_ticket_manager,
    robot_send_has_explicit_destination,
)
from app.services.agent.robot_delivery import (
    ROBOT_QQ_REPLY_EVENT_TYPE,
    ROBOT_SEND_TOOL_NAME,
    robot_reply_event_content,
)
from app.services.agent.session import (
    COMMAND_DISPATCH_FAILURE_MESSAGE,
    COMMAND_TOOL_NAMES,
    RUN_JOB_TOOL_NAME,
    agent_session_manager,
    is_background_job_started_result,
    is_command_dispatch_failure_result,
    is_command_dispatch_pending_result,
    is_terminal_unavailable_error,
)
from app.services.agent.stream_manager import stream_manager
from app.services.agent.tool_arguments import (
    ToolArgumentParseError,
    parse_tool_arguments,
)
from app.services.agent.tool_grounding import (
    append_tool_call_footer,
    guard_fabricated_tool_trace,
    strip_think_tags,
)
from app.services.agent.tool_selection import select_tools_for_turn
from app.services.agent.turn_coordinator import agent_turn_coordinator, agent_turn_key
from app.services.llm_completion import build_litellm_completion_kwargs
from app.services.terminal_runtime_state import get_terminal_runtime_state

if TYPE_CHECKING:
    from app.services.agent.agent import Agent

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 1
REQUEST_TIMEOUT = settings.CHAT_LLM_TIMEOUT_SECONDS
MAX_ITERATIONS = 6
LOOP_DETECTION_WINDOW = 6
LOOP_THRESHOLD = 3
TOOL_LOOP_STOP_REASON = (
    "Task failed because repeated tool calls produced no new executable progress."
)
TOOL_BUDGET_STOP_REASON = (
    "Task failed after automatic continuation attempts without a verified result."
)
SILENT_TOOL_NAMES = {
    "mcp_local_get_terminal_status",
    "mcp_local_read_terminal_log",
    RUN_JOB_TOOL_NAME,
    "mcp_robot_send_message",
    "mcp_robot_save_memory",
}
HIDDEN_TOOL_RESULT_NAMES = {
    "mcp_robot_send_message",
}
MAX_AUTO_TASKS = 5
INTERNAL_QQ_BACKGROUND_JOB_PREFIX = (
    "[后台终端任务结果 - 本 QQ 会话]"
)
INTERNAL_QQ_BACKGROUND_JOB_BATCH_PREFIX = "[Background job results batch:"
INTERNAL_QQ_BACKGROUND_JOB_ROBOT_PREFIX = "[后台终端任务结果 - 来自 QQ 会话]"
INTERNAL_AGENT_RETRY_PREFIX = "[Internal corrective turn]"
CURRENT_QQ_MESSAGE_MARKER = "[Current QQ message]"
CQ_CODE_RE = re.compile(r"\[CQ:[^\]]+\]", re.IGNORECASE)

router = APIRouter(prefix="/chat", tags=["chat"])


def _is_internal_agent_callback(message: str, source_type: str) -> bool:
    if str(source_type or "").strip().lower() != SOURCE_QQ:
        return False
    text = str(message or "").lstrip()
    return (
        text.startswith(INTERNAL_QQ_BACKGROUND_JOB_PREFIX)
        or text.startswith(INTERNAL_QQ_BACKGROUND_JOB_ROBOT_PREFIX)
        or text.startswith(INTERNAL_QQ_BACKGROUND_JOB_BATCH_PREFIX)
        or text.startswith(INTERNAL_AGENT_RETRY_PREFIX)
    )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    images: list[str] = []


class ChatStreamRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    images: list[str] = []


def get_relevant_memories(
    handler_id: str,
    query: str,
    agent: "Agent" = None,
    n_results: int = 3,
) -> str:
    if agent:
        skip_tools = agent.get_skip_memory_tools()
        query_lower = query.lower()
        for tool_name in skip_tools:
            tool_base = tool_name.replace("mcp_local_", "").replace("_", " ")
            if tool_base in query_lower:
                logger.info(f"[Chat] Skipping memory search for tool operation: {tool_name}")
                return ""

    try:
        memories = vector_store.search_memories(
            handler_id=handler_id,
            query=query,
            n_results=n_results,
        )
        if not memories:
            return ""

        return "\n".join([f"- {memory['content']}" for memory in memories])
    except Exception as exc:
        logger.warning(f"[Chat] Failed to retrieve memories: {exc}")
        return ""


def extract_important_info(
    user_msg: str,
    assistant_msg: str,
    *,
    item_id: str | None = None,
    handler_id: str = "",
    matched_skills: list | None = None,
):
    explicit_candidate = build_conversation_memory_candidate(
        user_msg,
        assistant_msg,
        matched_skills=matched_skills,
    )
    if explicit_candidate is not None:
        return explicit_candidate

    if not handler_id:
        return None

    previous_assistant_message = ""
    if item_id:
        previous_assistant_message = get_previous_assistant_response_before_latest_user_message(
            item_id,
            user_msg,
        )
    confirmation_candidate = build_confirmation_memory_candidate(
        user_msg,
        previous_assistant_message,
        matched_skills=matched_skills,
    )
    if confirmation_candidate is not None:
        return confirmation_candidate

    return build_status_update_memory_candidate(
        handler_id,
        user_msg,
        store=vector_store,
        matched_skills=matched_skills,
    )


def build_system_prompt_with_skills(
    handler: ItemHandler,
    message: str,
    agent: "Agent" = None,
    memories: str = "",
) -> tuple[str, list, list[dict]]:
    if agent is None:
        agent = agent_manager.get_or_create(handler)

    all_skills = agent.get_skills()
    matched_skills: list = []
    tools = select_tools_for_turn(
        agent.get_tools_for_litellm(),
        source="web",
        agent=agent,
    )

    parts = [get_system_prompt(agent)]

    if all_skills:
        parts.append("\n## Your Loaded Skills:\n")
        parts.append(
            "You have the following skills loaded. When user asks about your skills, list them:\n"
        )
        for skill in all_skills:
            parts.append(f"- **{skill.name}** (ID: {skill.skill_id})")
            if skill.description:
                parts.append(f"  Description: {skill.description}")
        parts.append("")

    if matched_skills:
        parts.append("\n## Skills Relevant to Current Query:\n")
        for skill in matched_skills:
            if skill.category in {"system", "persona"}:
                continue
            parts.append(f"### {skill.name}")
            if skill.description:
                parts.append(f"Description: {skill.description}")

            if skill.action and skill.action.prompt:
                parts.append(f"\n{skill.action.prompt}\n")

            if skill.content:
                content_without_frontmatter = skill.content
                match = re.match(r"^---\s*\n.*?\n---\s*\n", skill.content, re.DOTALL)
                if match:
                    content_without_frontmatter = skill.content[match.end() :].strip()
                if content_without_frontmatter:
                    parts.append(f"Additional Info:\n{content_without_frontmatter}")
            parts.append("")

    if memories:
        parts.append(
            "\n## Relevant Memories (historical background retrieved from past conversations;\n"
            "entries describe things that already happened or were said before - they are\n"
            "reference context, not pending work or current instructions):\n"
            f"{memories}"
        )

    return "\n".join(parts), matched_skills, tools


def _to_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _qq_display_text(text: str) -> str:
    """Raw QQ text for chat history / web UI display. The composed agent
    message ends with a `[Current QQ message]` marker; everything before it
    (sender cards, impression cards, live context) is model-only context.
    [CQ:image] segments are rewritten as markdown images so the web UI can
    render them inline.
    """
    def _cq_image_to_markdown(match: re.Match) -> str:
        params = match.group(1)
        url_match = re.search(r"(?:^|,)url=([^,\]]+)", params)
        if not url_match:
            return "[图片]"
        return f"![图片]({url_match.group(1).strip()})"

    marker = "[Current QQ message]"
    if marker in text:
        tail = text.rsplit(marker, 1)[1].strip()
        if tail:
            text = tail
    else:
        text = text.strip() or text
    return re.sub(r"\[CQ:image,([^\]]*)\]", _cq_image_to_markdown, text)


def _persist_and_broadcast_event(
    item_id: str,
    *,
    role: str | None,
    content: str,
    message_type: str,
    extra: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    event = append_chat_message(
        item_id,
        role=role,
        content=content,
        message_type=message_type,
        timestamp=timestamp,
        extra=extra,
    )
    stream_manager.broadcast_chat_event(item_id, event)
    return event


def _broadcast_agent_status(
    item_id: str,
    status: str,
    content: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "agent_status",
        "status": status,
        "content": content,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        event.update(extra)
    stream_manager.broadcast_chat_event(item_id, event)
    return event


def _should_hide_tool_details(tool_name: str) -> bool:
    return tool_name in SILENT_TOOL_NAMES


def _format_tool_result(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("success"):
            result_data = result.get("result", [])
            if isinstance(result_data, list):
                texts = [
                    item.get("text", "")
                    for item in result_data
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "\n".join(texts).strip()
            return str(result_data)
        return f"Error: {result.get('error', 'Unknown error')}"

    return str(result)



def _current_reply_ticket_id(agent: Any) -> str:
    context = getattr(agent, "_context", None)
    return str(getattr(context, "reply_ticket_id", "") or "").strip()


def _ticket_delivery_trace() -> dict[str, Any]:
    return {
        "type": "agent_tool_result",
        "content": "Message sent to current robot conversation.",
        "timestamp": datetime.now().isoformat(),
        "tool_name": "reply_ticket",
        "hidden": True,
    }


def _hidden_blocked_qq_send_trace() -> dict[str, Any]:
    return {
        "type": "agent_tool_result",
        "content": "Blocked accidental QQ send from non-QQ source.",
        "timestamp": datetime.now().isoformat(),
        "tool_name": ROBOT_SEND_TOOL_NAME,
        "hidden": True,
    }


def _deliver_reply_ticket_final_response(
    *,
    agent: Any,
    item_id: str,
    content: str,
    include_hidden_tool_results: bool,
    reply_ticket_id: str = "",
) -> list[dict[str, Any]]:
    from app.plugins.robot.internal_trace import sanitize_robot_visible_text
    from app.plugins.robot.reply_intent import is_no_reply_intent

    ticket_id = str(reply_ticket_id or "").strip() or _current_reply_ticket_id(agent)
    ticket = reply_ticket_manager.get(ticket_id)
    if not ticket or ticket.source_type != SOURCE_QQ:
        return []
    if is_no_reply_intent(content):
        # The agent chose silence (e.g. bare "NRN" on a job callback); never
        # deliver the marker itself. Finalize as completed so the ticket does
        # not linger as "running" on the task board.
        reply_ticket_manager.mark_completed(ticket_id)
        return []
    if ticket.external_report_sent:
        # The result already reached QQ via the send tool in an earlier turn;
        # finalize the ticket silently instead of sending a duplicate report.
        reply_ticket_manager.mark_delivered(ticket_id)
        return []
    visible_content = sanitize_robot_visible_text(content).strip()
    if not visible_content:
        return []
    delivered = reply_ticket_manager.deliver(ticket_id, visible_content)
    if not delivered:
        return []
    events: list[dict[str, Any]] = [
        _persist_and_broadcast_event(
            item_id,
            role="assistant",
            content=f"已回复 QQ：{visible_content}",
            message_type=ROBOT_QQ_REPLY_EVENT_TYPE,
            extra={"tool_name": "reply_ticket", "qq_delivery": True},
        )
    ]
    if include_hidden_tool_results:
        events.append(_ticket_delivery_trace())
    return events


def _complete_confirmed_external_delivery(ticket_id: str) -> bool:
    ticket = reply_ticket_manager.get(ticket_id)
    if not ticket:
        return True
    return reply_ticket_manager.mark_delivered(ticket_id)


def _ticket_has_open_plan_steps(ticket_id: str) -> bool:
    """票据的计划还有 pending/in_progress 步骤时，裸发送只是中间汇报，
    不能提前结束本轮（剩余步骤仍需继续跑）。"""
    from app.services.agent.reply_ticket import plan_has_open_steps

    ticket = reply_ticket_manager.get(ticket_id)
    return bool(ticket) and plan_has_open_steps(ticket.plan)


def _run_async_from_sync(coro_factory):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def runner():
        try:
            result_queue.put((True, asyncio.run(coro_factory())))
        except BaseException as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    ok, result = result_queue.get()
    if ok:
        return result
    raise result


def _build_completion_kwargs(
    handler: ItemHandler,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict],
    stream: bool,
) -> dict[str, Any]:
    kwargs = build_litellm_completion_kwargs(
        model=handler.model,
        messages=messages,
        stream=stream,
        timeout=REQUEST_TIMEOUT,
        tools=tools,
        api_key=handler.api_key,
        api_base=handler.api_url,
        model_parameters=getattr(handler, "model_parameters", {}),
        default_parameters={"temperature": 0.1},
    )
    if stream:
        kwargs["stream_options"] = {"include_usage": True}
    return kwargs


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _extract_json_payload(raw_content: str) -> Any | None:
    candidates: list[str] = []
    normalized = (raw_content or "").strip()
    if not normalized:
        return None

    candidates.append(normalized)
    if normalized.startswith("```"):
        fenced = re.sub(r"^```(?:json)?\s*", "", normalized, flags=re.IGNORECASE)
        fenced = re.sub(r"\s*```$", "", fenced)
        candidates.append(fenced.strip())

    first_object = normalized.find("{")
    last_object = normalized.rfind("}")
    if first_object != -1 and last_object > first_object:
        candidates.append(normalized[first_object : last_object + 1])

    first_array = normalized.find("[")
    last_array = normalized.rfind("]")
    if first_array != -1 and last_array > first_array:
        candidates.append(normalized[first_array : last_array + 1])

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def _normalize_task_title(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    normalized = value.strip()
    normalized = re.sub(
        r"^\s*(?:task|任务)?\s*\d+\s*[:：.\-、)]\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"^\s*[-*]\s*", "", normalized)
    return normalized.strip()[:120]


def _parse_task_titles(raw_content: str) -> list[str]:
    payload = _extract_json_payload(raw_content)
    if payload is None:
        return []

    if isinstance(payload, dict):
        entries = (
            payload.get("tasks")
            or payload.get("items")
            or payload.get("plan")
            or payload.get("steps")
            or []
        )
    elif isinstance(payload, list):
        entries = payload
    else:
        entries = []

    titles: list[str] = []
    seen: set[str] = set()

    for entry in entries:
        title = ""
        if isinstance(entry, str):
            title = _normalize_task_title(entry)
        elif isinstance(entry, dict):
            for key in ("title", "task", "name", "content", "summary", "step"):
                candidate = _normalize_task_title(entry.get(key))
                if candidate:
                    title = candidate
                    break

        if not title:
            continue

        fingerprint = title.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        titles.append(title)

        if len(titles) >= MAX_AUTO_TASKS:
            break

    return titles


def _build_fallback_task_titles(message: str) -> list[str]:
    normalized = (message or "").lower()
    prefers_chinese = _contains_cjk(message)

    if any(
        token in normalized
        for token in ("安装", "install", "java", "jdk", "软件", "package")
    ):
        return (
            ["确认当前安装状态与系统环境", "安装用户要求的软件", "验证版本并汇报结果"]
            if prefers_chinese
            else [
                "Check the current installation and system environment",
                "Install the requested software",
                "Verify the version and report the result",
            ]
        )

    if "python" in normalized or ".py" in normalized:
        return (
            ["编写 Python 代码", "执行 Python 脚本", "检查输出是否正确"]
            if prefers_chinese
            else ["Write the Python code", "Run the Python script", "Verify the output"]
        )

    if any(token in normalized for token in ("file", "folder", "目录", "文件")):
        return (
            ["确认目标文件与路径", "执行文件操作", "检查文件结果是否正确"]
            if prefers_chinese
            else ["Confirm the target file and path", "Perform the file operation", "Verify the file result"]
        )

    return (
        ["分析用户请求", "执行所需操作", "检查结果并反馈"]
        if prefers_chinese
        else ["Analyze the request", "Perform the required operation", "Verify the result and report back"]
    )


def _current_task_origin(agent: "Agent") -> dict[str, str]:
    context = getattr(agent, "_context", None)
    robot_id = str(getattr(context, "robot_id", "") or "").strip()
    robot_conversation_key = str(
        getattr(context, "robot_conversation_key", "") or ""
    ).strip()
    if robot_id:
        label = robot_conversation_key or "current QQ conversation"
        return {
            "type": "qq",
            "label": f"QQ {label}",
            "reply_rule": "reply through the locked current QQ robot context",
        }
    return {
        "type": "web",
        "label": "TermPaws web chat",
        "reply_rule": "reply in the current web chat response",
    }


def _append_conversation_memory(
    item_id: str,
    *,
    user_message: str,
    assistant_message: str,
    matched_skills: list,
    agent: "Agent" = None,
):
    handler_id = str(getattr(agent, "handler_id", "") or "").strip()
    if not handler_id:
        from app.services.agent.memory.scope import resolve_handler_id

        handler_id = resolve_handler_id(item_id)

    candidate = extract_important_info(
        user_message,
        assistant_message,
        item_id=item_id,
        handler_id=handler_id,
        matched_skills=matched_skills,
    )
    if not candidate:
        return

    persist_memory_candidate(
        handler_id,
        candidate,
        store=vector_store,
        source_item_id=item_id,
    )


def _inject_active_jobs_prompt_context(
    item_id: str,
    messages: list[dict[str, Any]],
) -> None:
    terminal_session = agent_session_manager.get_session(item_id)
    if not terminal_session:
        return
    terminal_session.inject_active_jobs_prompt_context(messages)


def _detect_tool_loop(
    tool_call_history: list[tuple[str, str]],
    tool_name: str,
    tool_args_str: str,
) -> tuple[bool, str]:
    tool_fingerprint = _tool_loop_fingerprint(tool_name, tool_args_str)
    tool_call_history.append((tool_name, tool_fingerprint))
    if len(tool_call_history) > LOOP_DETECTION_WINDOW:
        del tool_call_history[:-LOOP_DETECTION_WINDOW]

    call_counts: dict[tuple[str, str], int] = {}
    for call in tool_call_history:
        call_counts[call] = call_counts.get(call, 0) + 1

    for (repeated_name, _), count in call_counts.items():
        if count >= LOOP_THRESHOLD:
            return True, f"Detected repeated tool loop for {repeated_name} ({count} times)"

    return False, ""


def _tool_loop_fingerprint(tool_name: str, tool_args_str: str) -> str:
    try:
        payload = json.loads(tool_args_str or "{}")
    except json.JSONDecodeError:
        return " ".join(str(tool_args_str or "").split())

    if not isinstance(payload, dict):
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    normalized = dict(payload)
    normalized.pop("item_id", None)
    normalized.pop("_reply_ticket_id", None)
    # Descriptive fields vary even when the model repeats the same state
    # transition without making progress, so they stay out of loop fingerprints.
    for _volatile_key in ("note", "explanation"):
        normalized.pop(_volatile_key, None)

    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sanitize_model_error(reason: str) -> str:
    detail = re.sub(
        r"^Agent request failed:\s*",
        "",
        str(reason or "").strip(),
        flags=re.IGNORECASE,
    )
    detail = re.sub(
        r"(?i)\bbearer\s+[a-z0-9._~+/=-]+",
        "Bearer [REDACTED]",
        detail,
    )
    detail = re.sub(
        r"(?i)\bsk-[a-z0-9_-]{8,}\b",
        "[REDACTED]",
        detail,
    )
    detail = re.sub(
        r"(?i)\b(api[_-]?key|authorization)\s*[:=]\s*['\"]?[^,\s'\"}]+",
        r"\1=[REDACTED]",
        detail,
    )
    return re.sub(r"\s+", " ", detail).strip()[:1200]


def _stopped_turn_fallback(reason: str, *, prefers_chinese: bool) -> str:
    normalized_reason = str(reason or "").casefold()
    error_detail = _sanitize_model_error(reason)
    if "unsupportedparamserror" in normalized_reason or (
        "temperature" in normalized_reason and "support" in normalized_reason
    ):
        return (
            f"模型参数不兼容：{error_detail}"
            if prefers_chinese
            else f"Incompatible model parameters: {error_detail}"
        )
    if "no available channel" in normalized_reason or "serviceunavailable" in normalized_reason:
        return (
            f"模型通道不可用：{error_detail}"
            if prefers_chinese
            else f"Model channel unavailable: {error_detail}"
        )
    if any(
        marker in normalized_reason
        for marker in ("authentication", "unauthorized", "invalid api key", "status code: 401")
    ):
        return (
            f"模型接口认证失败：{error_detail}"
            if prefers_chinese
            else f"Model API authentication failed: {error_detail}"
        )
    if "rate limit" in normalized_reason or "status code: 429" in normalized_reason:
        return (
            f"模型接口请求过于频繁：{error_detail}"
            if prefers_chinese
            else f"Model API rate limit reached: {error_detail}"
        )
    if "timeout" in normalized_reason or "timed out" in normalized_reason:
        return (
            f"模型接口响应超时：{error_detail}"
            if prefers_chinese
            else f"Model API timed out: {error_detail}"
        )
    if "agent request failed:" in normalized_reason:
        return (
            f"模型请求失败：{error_detail}"
            if prefers_chinese
            else f"Model request failed: {error_detail}"
        )
    if normalized_reason.startswith("terminal unavailable:"):
        terminal_error = error_detail.split(":", 1)[-1].strip()
        return (
            f"任务未完成：{terminal_error} 本次任务已结束。"
            if prefers_chinese
            else f"Task failed because the terminal is unavailable: {terminal_error}"
        )
    if any(
        marker in normalized_reason
        for marker in (
            "repeat",
            "loop",
            "did not converge",
            "iteration",
            "no progress",
            "automatic continuation attempts",
        )
    ):
        return (
            "任务未完成：自动执行后仍没有得到可验证结果，任务已结束。"
            if prefers_chinese
            else "The task did not complete after automatic attempts and has ended."
        )
    return (
        f"任务未完成：{error_detail}"
        if prefers_chinese
        else f"Task failed: {error_detail}"
    )


def _generate_stopped_turn_report(
    handler: ItemHandler,
    messages: list[dict[str, Any]],
    *,
    reason: str,
    prefers_chinese: bool,
) -> str:
    fallback = _stopped_turn_fallback(reason, prefers_chinese=prefers_chinese)
    normalized_reason = str(reason or "").casefold()
    if normalized_reason.startswith(
        ("agent request failed:", "terminal unavailable:")
    ):
        return fallback
    final_messages = [*messages]
    final_messages.append(
        {
            "role": "system",
            "content": (
                "Tool execution has ended after bounded automatic continuation attempts. "
                f"Internal stop reason: {reason} "
                "Give the user one concise final failure status in the user's language. Summarize "
                "only confirmed results already present in the conversation and state the actual "
                "error or why the objective could not be completed. The task will be removed after "
                "this report, so do not say it is paused, waiting, preserved, or will resume later. "
                "Do not call tools. Do not mention iteration limits, MCP, reply tickets, internal "
                "workflows, or tool-loop detection."
            ),
        }
    )
    try:
        kwargs = _build_completion_kwargs(
            handler,
            messages=final_messages,
            tools=[],
            stream=False,
        )
        kwargs["max_tokens"] = 300
        response = completion(**kwargs)
        content = (
            response.choices[0].message.content
            if response and getattr(response, "choices", None)
            else ""
        )
        return str(content or "").strip() or fallback
    except Exception as exc:
        logger.warning("[Chat] Failed to generate stopped-turn report: %s", exc)
        return fallback


def _finalize_stopped_turn(
    *,
    agent: Any,
    handler: ItemHandler,
    item_id: str,
    messages: list[dict[str, Any]],
    reason: str,
    prefers_chinese: bool,
    include_hidden_tool_results: bool,
    reply_ticket_id: str = "",
    is_abort: bool = False,
) -> list[dict[str, Any]]:
    from app.plugins.robot.internal_trace import sanitize_robot_visible_text

    if is_abort:
        return [
            _persist_and_broadcast_event(
                item_id,
                role="assistant",
                content="当前轮已中断，任务仍在运行中。",
                message_type="agent_response",
            )
        ]

    report = _generate_stopped_turn_report(
        handler,
        messages,
        reason=reason,
        prefers_chinese=prefers_chinese,
    )
    visible_report = sanitize_robot_visible_text(report).strip()
    final_report = visible_report or "任务未能完成，已停止处理。"

    ticket_id = str(reply_ticket_id or "").strip() or _current_reply_ticket_id(agent)
    ticket = reply_ticket_manager.get(ticket_id)
    if ticket:
        reply_ticket_manager.mark_failed(ticket_id, reason or final_report)
        # 硬性停止（循环/预算耗尽/失败）是终态：计划一并取消，否则它保持
        # 未完成，resume sweep 会反复复活这个注定再失败的任务，用户就会
        # 看到"任务已结束"的消息一遍遍重发。
        if not is_abort:
            reply_ticket_manager.cancel_plan(ticket_id, explanation="stopped")
    delivered = False
    if ticket and ticket.source_type == SOURCE_QQ:
        delivered = reply_ticket_manager.deliver(ticket_id, final_report)
        if delivered:
            events = [
                _persist_and_broadcast_event(
                    item_id,
                    role="assistant",
                    content=f"已回复 QQ：{final_report}",
                    message_type=ROBOT_QQ_REPLY_EVENT_TYPE,
                    extra={"tool_name": "reply_ticket", "qq_delivery": True},
                )
            ]
            if include_hidden_tool_results:
                events.append(_ticket_delivery_trace())
            return events

    if ticket and ticket.source_type != SOURCE_QQ:
        reply_ticket_manager.mark_delivered(ticket_id)
    return [
        _persist_and_broadcast_event(
            item_id,
            role="assistant",
            content=final_report,
            message_type="agent_response",
        )
    ]


def _generate_stream_unserialized(
    message: str,
    history: list[ChatMessage],
    handler: ItemHandler,
    item_id: str,
    agent: "Agent" = None,
    include_hidden_tool_results: bool = False,
    latest_only_context: bool = False,
    source_type: str = SOURCE_WEB,
    reply_ticket_id: str = "",
    image_urls: list[str] | None = None,
    force_image_vision: bool = False,
    agent_image_urls: list[str] | None = None,
) -> Generator[str, None, None]:
    if agent is None:
        agent = agent_manager.get_or_create(handler)

    normalized_source_type = str(source_type or SOURCE_WEB).strip().lower()
    internal_agent_callback = _is_internal_agent_callback(
        message,
        normalized_source_type,
    )
    if normalized_source_type == SOURCE_WEB:
        clear_robot_context = getattr(agent, "clear_robot_context", None)
        if callable(clear_robot_context):
            clear_robot_context()

    reply_ticket = reply_ticket_manager.get(reply_ticket_id)
    if (
        reply_ticket is None
        or reply_ticket.item_id != str(item_id)
        or reply_ticket.status == "delivered"
    ):
        reply_ticket = reply_ticket_manager.create_for_agent(
            agent,
            item_id=item_id,
            handler_id=str(handler.id),
            message=message,
            source_type=normalized_source_type,
        )
    else:
        reply_ticket_manager.attach_to_agent(agent, reply_ticket.ticket_id)
    matched_skills: list = []
    turn_source = "qq" if normalized_source_type == SOURCE_QQ else "web"
    tools = select_tools_for_turn(
        agent.get_tools_for_litellm(),
        source=turn_source,
        agent=agent,
        reply_ticket_id=reply_ticket.ticket_id,
    )

    agent_context = getattr(agent, "_context", None)
    if agent_context is not None:
        agent_context.robot_backend_target_resolution_enabled = bool(
            normalized_source_type == SOURCE_WEB
            and any(
                tool.get("function", {}).get("name") == ROBOT_SEND_TOOL_NAME
                for tool in tools
            )
        )

    pending_context = build_pending_terminal_continuation_prompt(item_id, message)
    messages = build_chat_turn_messages(
        agent,
        item_id=item_id,
        message=message,
        query=message,
        latest_only_context=latest_only_context,
        pending_context=pending_context,
        image_urls=agent_image_urls or image_urls,
        force_image_vision=force_image_vision or bool(agent_image_urls),
    )
    # Authoritative live terminal state is cheap (one daemon HTTP call) and
    # prevents stale-memory answers about terminal status; inject it for
    # every turn instead of guessing intent from the message text.
    try:
        live_terminal_state = get_terminal_runtime_state(item_id)
        messages.insert(
            max(len(messages) - 1, 0),
            {
                "role": "system",
                "content": live_terminal_state.prompt_context(),
            },
        )
    except Exception:
        logger.exception(
            "[Chat] Failed to inject live terminal state: item=%s",
            item_id,
        )
    reply_ticket_prompt = reply_ticket_manager.build_prompt(reply_ticket.ticket_id)
    if reply_ticket_prompt:
        messages.insert(1, {"role": "system", "content": reply_ticket_prompt})
    _inject_active_jobs_prompt_context(item_id, messages)
    extract_integration_context_targets(agent, messages)
    record_integration_context_targets(agent, item_id)

    AgentMessageQueue.clear_abort(item_id)

    _stripped_message = str(message or "").lstrip()
    background_job_callback = internal_agent_callback and (
        _stripped_message.startswith(INTERNAL_QQ_BACKGROUND_JOB_PREFIX)
        or _stripped_message.startswith(INTERNAL_QQ_BACKGROUND_JOB_ROBOT_PREFIX)
        or _stripped_message.startswith(INTERNAL_QQ_BACKGROUND_JOB_BATCH_PREFIX)
    )
    if not internal_agent_callback or background_job_callback:
        user_message_type = "qq_user" if normalized_source_type == SOURCE_QQ else "chat_user"
        # The model gets the fully composed message (sender cards, live
        # context, memories...), but the chat history and the web UI should
        # only show the raw text the QQ user actually sent. Background job
        # callback rounds have no user text of their own; persist the
        # callback content itself so the history stays self-consistent.
        # (Otherwise the ticket's original request would be re-persisted
        # once per callback and later turns lose track of which round they
        # are answering.)
        display_content = message
        if background_job_callback:
            display_content = _qq_display_text(message)
        elif normalized_source_type == SOURCE_QQ:
            display_content = _qq_display_text(reply_ticket.request_message or message)
            if image_urls:
                image_marks = "\n".join(
                    f"![图片]({url})" for url in image_urls if url.strip()
                )
                if image_marks:
                    display_content = (
                        f"{display_content}\n{image_marks}"
                        if display_content.strip()
                        else image_marks
                    )
        user_event = _persist_and_broadcast_event(
            item_id,
            role="user",
            content=display_content,
            message_type=user_message_type,
            extra={"sender_key": reply_ticket.sender_key, "sender_label": reply_ticket.sender_label, "source": normalized_source_type} if normalized_source_type == SOURCE_QQ else None,
        )
        yield _to_sse(user_event)

    tool_call_history: list[tuple[str, str]] = []
    final_response = ""
    tool_called_this_turn = False
    called_tool_names: set[str] = set()
    called_tool_sequence: list[str] = []
    delivery_tool_sent_by_integration = False
    qq_message_sent_this_turn = False
    confirmed_external_delivery_to_qq = False
    pending_async_delivery = False
    thinking_only_retry_used = False
    plan_continuation_count = 0
    robot_final_text_retry_used = False
    plan_open_at_turn_start = _ticket_has_open_plan_steps(reply_ticket.ticket_id)
    plan_report_nudge_used = False
    from app.core.config import settings as chat_settings

    turn_started_at = time.monotonic()

    try:
        for iteration_index in range(MAX_ITERATIONS + 1):
            reply_ticket_manager.touch(reply_ticket.ticket_id)
            finalization_only = iteration_index == MAX_ITERATIONS
            iteration_tools = (
                []
                if finalization_only
                else select_tools_for_turn(
                    agent.get_tools_for_litellm(),
                    source=turn_source,
                    agent=agent,
                    reply_ticket_id=reply_ticket.ticket_id,
                )
            )
            if finalization_only:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "The tool-call budget is finished. Do not call any more tools. "
                            "Use the latest confirmed tool results to give the user one concise "
                            "final answer now. If the latest result confirms the requested check "
                            "or action succeeded, report that success instead of saying the task "
                            "is incomplete. Mention a failure only when the tool result actually "
                            "shows a failure."
                        ),
                    }
                )
            if AgentMessageQueue.is_aborted(item_id):
                stopped_events = _finalize_stopped_turn(
                    agent=agent,
                    handler=handler,
                    item_id=item_id,
                    messages=messages,
                    reason="Task was interrupted before completion.",
                    prefers_chinese=_contains_cjk(message),
                    include_hidden_tool_results=include_hidden_tool_results,
                    reply_ticket_id=reply_ticket.ticket_id,
                    is_abort=True,
                )
                for stopped_event in stopped_events:
                    yield _to_sse(stopped_event)
                yield _to_sse(
                    {
                        "type": "aborted",
                        "content": "Chat aborted",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return

            if (
                chat_settings.AGENT_TURN_TIMEOUT_SECONDS > 0
                and time.monotonic() - turn_started_at
                > chat_settings.AGENT_TURN_TIMEOUT_SECONDS
            ):
                logger.warning(
                    "[Chat] Turn exceeded %.0fs for item %s; stopping turn.",
                    chat_settings.AGENT_TURN_TIMEOUT_SECONDS,
                    item_id,
                )
                stopped_events = _finalize_stopped_turn(
                    agent=agent,
                    handler=handler,
                    item_id=item_id,
                    messages=messages,
                    reason="Task exceeded the maximum turn duration and was stopped.",
                    prefers_chinese=_contains_cjk(message),
                    include_hidden_tool_results=include_hidden_tool_results,
                    reply_ticket_id=reply_ticket.ticket_id,
                )
                for stopped_event in stopped_events:
                    yield _to_sse(stopped_event)
                yield _to_sse(
                    {
                        "type": "aborted",
                        "content": "Chat turn timed out",
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return

            response = None
            last_error: Exception | None = None

            for attempt in range(MAX_RETRIES):
                try:
                    response = completion(
                        **_build_completion_kwargs(
                            handler,
                            messages=messages,
                            tools=iteration_tools,
                            stream=True,
                        )
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "[Chat] Stream attempt %s/%s failed for item %s: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        item_id,
                        exc,
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)

            if response is None:
                error_text = str(last_error or "Unknown stream error")
                stopped_events = _finalize_stopped_turn(
                    agent=agent,
                    handler=handler,
                    item_id=item_id,
                    messages=messages,
                    reason=f"Agent request failed: {error_text}",
                    prefers_chinese=_contains_cjk(message),
                    include_hidden_tool_results=include_hidden_tool_results,
                    reply_ticket_id=reply_ticket.ticket_id,
                )
                for stopped_event in stopped_events:
                    yield _to_sse(stopped_event)
                yield _to_sse({"done": True})
                return

            iteration_content = ""
            iteration_reasoning = ""
            tool_calls_map: dict[int, dict[str, Any]] = {}
            _chunk_usage = None

            for chunk in response:
                if AgentMessageQueue.is_aborted(item_id):
                    stopped_events = _finalize_stopped_turn(
                        agent=agent,
                        handler=handler,
                        item_id=item_id,
                        messages=messages,
                        reason="Task was interrupted before completion.",
                        prefers_chinese=_contains_cjk(message),
                        include_hidden_tool_results=include_hidden_tool_results,
                        reply_ticket_id=reply_ticket.ticket_id,
                        is_abort=True,
                    )
                    for stopped_event in stopped_events:
                        yield _to_sse(stopped_event)
                    yield _to_sse(
                        {
                            "type": "aborted",
                            "content": "Chat aborted",
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    return

                _cu = getattr(chunk, "usage", None)
                if _cu:
                    _chunk_usage = _cu
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = getattr(choice, "delta", None)
                if not delta:
                    continue

                delta_content = getattr(delta, "content", None)
                if delta_content:
                    iteration_content += delta_content

                delta_reasoning = getattr(delta, "reasoning_content", None) or getattr(
                    delta, "reasoning", None
                )
                if delta_reasoning:
                    iteration_reasoning += delta_reasoning

                delta_tool_calls = getattr(delta, "tool_calls", None)
                if not delta_tool_calls:
                    continue

                for tool_call in delta_tool_calls:
                    index = getattr(tool_call, "index", 0)
                    accumulator = tool_calls_map.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )

                    tool_call_id = getattr(tool_call, "id", None)
                    if tool_call_id:
                        accumulator["id"] = tool_call_id

                    function = getattr(tool_call, "function", None)
                    if function is None:
                        continue

                    function_name = getattr(function, "name", None)
                    if function_name:
                        accumulator["function"]["name"] += function_name

                    function_arguments = getattr(function, "arguments", None)
                    if function_arguments:
                        accumulator["function"]["arguments"] += function_arguments

            try:
                from app.services.agent.token_usage import token_usage_tracker as _tt
                _su = _chunk_usage or getattr(response, "usage", None)
                if _su:
                    _tt.record(
                        str(getattr(agent, "handler_id", "") or handler.id),
                        str(handler.model or "unknown"),
                        int(getattr(_su, "prompt_tokens", 0) or 0),
                        int(getattr(_su, "completion_tokens", 0) or 0),
                        int(getattr(_su, "total_tokens", 0) or 0),
                        reply_ticket_id=str(
                            getattr(reply_ticket, "ticket_id", "") or ""
                        ),
                    )
            except Exception:
                pass

            iteration_content, dsml_tool_calls = extract_dsml_tool_calls(
                iteration_content,
                allowed_tool_names={
                    str(tool.get("function", {}).get("name") or "").strip()
                    for tool in iteration_tools
                    if str(tool.get("function", {}).get("name") or "").strip()
                },
            )
            ordered_tool_calls = [
                tool_calls_map[index]
                for index in sorted(tool_calls_map)
                if tool_calls_map[index].get("function", {}).get("name")
            ]
            if finalization_only:
                ordered_tool_calls = []
            elif not ordered_tool_calls:
                ordered_tool_calls = dsml_tool_calls

            if not ordered_tool_calls:
                # 计划驱动推进：计划还有未完成步骤时轮次不能就这么结束——
                # 由计划自身驱动继续执行，而不是靠 prompt 硬约束。
                # 超过 5 次推进仍不行动才放行，防止死循环。
                if (
                    not finalization_only
                    and _ticket_has_open_plan_steps(reply_ticket.ticket_id)
                    and plan_continuation_count < 5
                ):
                    plan_continuation_count += 1
                    messages.append({"role": "assistant", "content": iteration_content})
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "当前计划还有未完成步骤。先核对现实：如果这些步骤的工作实际上已经完成，"
                                "不要再执行——用 mcp_local_update_plan 把剩余步骤全部标为 completed"
                                "（计划会自动关闭），然后总结收尾；"
                                "只有确实没做完的步骤才继续执行，每完成一步用 update_plan 更新状态。"
                            ),
                        }
                    )
                    continue
                # 计划在本轮刚收尾但用户还没收到结果：推一次让 agent 汇报，
                # 不然"活干完了没吱声"（远端实测：巡检续跑把活干完但没人告诉用户）。
                if (
                    plan_open_at_turn_start
                    and not plan_report_nudge_used
                    and not finalization_only
                    and not _ticket_has_open_plan_steps(reply_ticket.ticket_id)
                    and not (delivery_tool_sent_by_integration or qq_message_sent_this_turn)
                    and normalized_source_type == SOURCE_QQ
                ):
                    plan_report_nudge_used = True
                    messages.append({"role": "assistant", "content": iteration_content})
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "计划已全部完成，但用户还没收到结果。现在用 "
                                "mcp_robot_send_message 把最终结果发给用户（发到它该在的"
                                "群/会话，显式带 target_type+target_id），发完返回 "
                                "[no_qq_reply]。"
                            ),
                        }
                    )
                    continue
                final_response = strip_think_tags(guard_fabricated_tool_trace(iteration_content))
                if finalization_only and not final_response:
                    break

                if normalized_source_type == SOURCE_WEB and not internal_agent_callback:
                    try:
                        from app.plugins.robot.explicit_target_backfill import (
                            run_explicit_target_backfill,
                        )

                        run_explicit_target_backfill(
                            item_id=str(item_id),
                            user_message=message,
                            final_text=final_response,
                            delivery_key=f"ticket:{reply_ticket.ticket_id}",
                        )
                    except Exception:
                        logger.exception(
                            "[Chat] Explicit QQ target backfill failed for item %s",
                            item_id,
                        )

                final_response = append_tool_call_footer(
                    final_response,
                    called_tool_sequence,
                )
                # QQ 轮次里发送工具是唯一出口，最终文本不会送达。
                # 两种形态都纠偏一次：(a) 发过几条但把后续写进了最终文本
                # （长故事断尾的根因）；(b) 什么都没发，直接把答复写成最终文本
                # （回调轮"改好啦"没进群的根因）。web 不受影响：最终文本就是回复。
                try:
                    from app.plugins.robot.reply_intent import is_no_reply_intent
                except Exception:
                    is_no_reply_intent = None
                if (
                    normalized_source_type == SOURCE_QQ
                    and not robot_final_text_retry_used
                    and not finalization_only
                    and final_response.strip()
                    and is_no_reply_intent is not None
                    and not is_no_reply_intent(final_response)
                ):
                    robot_final_text_retry_used = True
                    messages.append({"role": "assistant", "content": iteration_content})
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "你上一条文本不会送达用户——QQ 回复的唯一出口是 "
                                "mcp_robot_send_message。如果那是要给用户的内容，"
                                "现在用发送工具把它发出去（显式带 target_type+target_id，"
                                "一条气泡一次调用，发完为止）；"
                                "如果确实没有什么要发给用户，只返回 [no_qq_reply]。"
                            ),
                        }
                    )
                    continue
                if delivery_tool_sent_by_integration or qq_message_sent_this_turn:
                    logger.info(
                        "[Chat] Suppressed final response after source delivery tool sent for item %s",
                        item_id,
                    )
                    _broadcast_agent_status(item_id, "idle")
                    yield _to_sse({"done": True})
                    return

                if pending_async_delivery:
                    logger.info(
                        "[Chat] Suppressed final response: async tool results pending for item %s, will deliver on callback",
                        item_id,
                    )
                    _broadcast_agent_status(item_id, "idle")
                    yield _to_sse({"done": True})
                    return

                response_event = _persist_and_broadcast_event(
                    item_id,
                    role="assistant",
                    content=final_response,
                    message_type="agent_response",
                )
                yield _to_sse(response_event)
                # 计划未完成就标 delivered 会杀掉 plan 生命周期：
                # 有未完成步骤的票据保持 pending，由 plan resume sweep 自动续跑。
                if (
                    reply_ticket.source_type != SOURCE_QQ
                    and not _ticket_has_open_plan_steps(reply_ticket.ticket_id)
                ):
                    reply_ticket_manager.mark_delivered(reply_ticket.ticket_id)
                if not internal_agent_callback:
                    _append_conversation_memory(
                        item_id,
                        user_message=message,
                        assistant_message=final_response,
                        matched_skills=matched_skills,
                        agent=agent,
                    )

                if not final_response and not finalization_only and not thinking_only_retry_used:
                    messages.append(
                        {"role": "assistant", "content": iteration_content}
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Your previous response contained only internal thinking "
                                "without any action or reply. You must either call a tool "
                                "or provide a text response to the user now."
                            ),
                        }
                    )
                    thinking_only_retry_used = True
                    continue

                _broadcast_agent_status(item_id, "idle")

                import gc as _gc_turn
                try:
                    del messages
                except NameError:
                    pass
                _gc_turn.collect()
                yield _to_sse({"done": True})
                return

            thinking_text = iteration_reasoning.strip() or iteration_content.strip()
            if thinking_text:
                thinking_event = _persist_and_broadcast_event(
                    item_id,
                    role="assistant",
                    content=thinking_text,
                    message_type="agent_thinking",
                )
                yield _to_sse(thinking_event)

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": iteration_content,
                "tool_calls": [],
            }
            tool_messages: list[dict[str, Any]] = []

            background_job_started_this_iteration = False
            for tool_call in ordered_tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args_str = tool_call["function"]["arguments"]
                in_loop, loop_message = _detect_tool_loop(
                    tool_call_history,
                    tool_name,
                    tool_args_str,
                )
                if in_loop:
                    logger.warning(
                        "[Chat] %s item=%s ticket=%s tool=%s",
                        loop_message,
                        item_id,
                        reply_ticket.ticket_id,
                        tool_name,
                    )
                    for event in _finalize_stopped_turn(
                        agent=agent,
                        handler=handler,
                        item_id=item_id,
                        messages=messages,
                        reason=TOOL_LOOP_STOP_REASON,
                        prefers_chinese=_contains_cjk(message),
                        include_hidden_tool_results=include_hidden_tool_results,
                        reply_ticket_id=reply_ticket.ticket_id,
                    ):
                        yield _to_sse(event)
                    _broadcast_agent_status(item_id, "idle")
                    yield _to_sse({"done": True})
                    return

                try:
                    tool_args = parse_tool_arguments(tool_name, tool_args_str)
                except ToolArgumentParseError:
                    logger.warning(
                        "[Chat] Failed to parse tool arguments for %s: %r",
                        tool_name,
                        tool_args_str,
                    )
                    error_event = _persist_and_broadcast_event(
                        item_id,
                        role="assistant",
                        content=f"Failed to parse tool arguments for {tool_name}",
                        message_type="agent_error",
                        extra={"tool_name": tool_name},
                    )
                    yield _to_sse(error_event)
                    yield _to_sse(
                        {
                            "type": "error",
                            "content": error_event["content"],
                            "timestamp": error_event["timestamp"],
                        }
                    )
                    return

                if tool_name == ROBOT_SEND_TOOL_NAME and normalized_source_type == SOURCE_QQ:
                    tool_args = enforce_persona_identity_robot_tool_args(
                        agent,
                        message,
                        tool_args,
                    )

                if (
                    tool_name == ROBOT_SEND_TOOL_NAME
                    and reply_ticket.source_type != SOURCE_QQ
                    and not robot_send_has_explicit_destination(tool_args)
                ):
                    logger.info(
                        "[Chat] Blocked destination-less QQ send from source=%s item=%s",
                        reply_ticket.source_type,
                        item_id,
                    )
                    if include_hidden_tool_results:
                        yield _to_sse(_hidden_blocked_qq_send_trace())
                    _broadcast_agent_status(item_id, "idle")
                    yield _to_sse({"done": True})
                    return

                normalized_tool_args_str = json.dumps(tool_args, ensure_ascii=False)
                tool_args["item_id"] = item_id
                tool_args["_reply_ticket_id"] = reply_ticket.ticket_id
                terminal_session = None
                if tool_name in COMMAND_TOOL_NAMES or tool_name == RUN_JOB_TOOL_NAME:
                    terminal_session = agent_session_manager.get_or_create_session(
                        item_id,
                        str(handler.id),
                    )
                    terminal_input_error = terminal_session.validate_terminal_tool_input(
                        tool_name,
                        tool_args,
                    )
                    if terminal_input_error:
                        if is_terminal_unavailable_error(terminal_input_error):
                            for event in _finalize_stopped_turn(
                                agent=agent,
                                handler=handler,
                                item_id=item_id,
                                messages=messages,
                                reason=f"Terminal unavailable: {terminal_input_error}",
                                prefers_chinese=_contains_cjk(message),
                                include_hidden_tool_results=include_hidden_tool_results,
                                reply_ticket_id=reply_ticket.ticket_id,
                            ):
                                yield _to_sse(event)
                        else:
                            assistant_message["tool_calls"].append(
                                {
                                    "id": tool_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": normalized_tool_args_str,
                                    },
                                }
                            )
                            tool_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "content": (
                                        f"Error: {terminal_input_error} "
                                        "Choose the next action yourself. For an independent "
                                        "non-interactive operation, use mcp_local_run_job; "
                                        "otherwise report the actual failure."
                                    ),
                                }
                            )
                            continue
                        _broadcast_agent_status(item_id, "idle")
                        yield _to_sse({"done": True})
                        return

                hide_tool_details = _should_hide_tool_details(tool_name)
                if not hide_tool_details:
                    action_event = _persist_and_broadcast_event(
                        item_id,
                        role="assistant",
                        content=f"Executing tool: {tool_name}",
                        message_type="agent_action",
                        extra={"tool_name": tool_name},
                    )
                    yield _to_sse(action_event)

                result = _run_async_from_sync(
                    lambda tool_name=tool_name, tool_args=tool_args: agent.execute_tool(
                        tool_name,
                        tool_args,
                    )
                )
                tool_called_this_turn = True
                result_text = _format_tool_result(result)
                called_tool_names.add(tool_name)
                called_tool_sequence.append(tool_name)
                command_dispatch_failed = is_command_dispatch_failure_result(
                    tool_name,
                    result_text,
                    result,
                )
                command_dispatch_pending = is_command_dispatch_pending_result(
                    tool_name,
                    result_text,
                )
                if command_dispatch_pending or is_background_job_started_result(result):
                    pending_async_delivery = True
                if result_text and fallback_is_delivery_result(result_text):
                    if tool_name == ROBOT_SEND_TOOL_NAME:
                        confirmed_external_delivery_to_qq = True
                        qq_message_sent_this_turn = True
                    delivery_tool_sent_by_integration = True
                    reply_ticket_manager.mark_delivered(
                        reply_ticket.ticket_id
                    )
                    if tool_name == ROBOT_SEND_TOOL_NAME:
                        reply_event = _persist_and_broadcast_event(
                            item_id,
                            role="assistant",
                            content=robot_reply_event_content(tool_args, result_text),
                            message_type=ROBOT_QQ_REPLY_EVENT_TYPE,
                            extra={
                                "tool_name": tool_name,
                                "qq_delivery": True,
                            },
                        )
                        yield _to_sse(reply_event)
                        # No early return here: the agent continues the turn,
                        # delivers any remaining targets, and ends the task by
                        # reporting (or NRN) on its own. Duplicate delivery is
                        # already prevented by ticket.external_report_sent plus
                        # final-response suppression — not by cutting the turn.

                if result_text and not command_dispatch_pending:
                    if hide_tool_details:
                        if (
                            include_hidden_tool_results
                            and tool_name in HIDDEN_TOOL_RESULT_NAMES
                        ):
                            yield _to_sse(
                                {
                                    "type": "agent_tool_result",
                                    "content": result_text,
                                    "timestamp": datetime.now().isoformat(),
                                    "tool_name": tool_name,
                                    "hidden": True,
                                }
                            )
                    else:
                        result_event = _persist_and_broadcast_event(
                            item_id,
                            role="assistant",
                            content=result_text,
                            message_type="agent_tool_result",
                            extra={"tool_name": tool_name},
                        )
                        yield _to_sse(result_event)

                if command_dispatch_failed:
                    record_pending_terminal_continuation(
                        item_id=item_id,
                        command=str(tool_args.get("command") or ""),
                        reason="terminal_not_connected",
                        message=COMMAND_DISPATCH_FAILURE_MESSAGE,
                        tool_name=tool_name,
                    )
                    for event in _finalize_stopped_turn(
                        agent=agent,
                        handler=handler,
                        item_id=item_id,
                        messages=messages,
                        reason=(
                            f"Terminal unavailable: {COMMAND_DISPATCH_FAILURE_MESSAGE}"
                        ),
                        prefers_chinese=_contains_cjk(message),
                        include_hidden_tool_results=include_hidden_tool_results,
                        reply_ticket_id=reply_ticket.ticket_id,
                    ):
                        yield _to_sse(event)
                    _broadcast_agent_status(item_id, "idle")
                    yield _to_sse({"done": True})
                    return

                if is_background_job_started_result(result):
                    clear_pending_terminal_continuation(
                        item_id,
                        command=str(tool_args.get("command") or ""),
                    )
                    background_job_started_this_iteration = True
                    continue

                if confirmed_external_delivery_to_qq:
                    if _complete_confirmed_external_delivery(
                        reply_ticket.ticket_id
                    ):
                        delivery_tool_sent_by_integration = True

                # A bare send (no explicit destination) is a reply to the
                # current conversation — the whole point of the turn is done,
                # so end here and skip the extra wrap-up LLM call. Sends
                # carrying explicit targets continue the turn so the agent
                # can deliver the remaining targets. Never stop while the
                # ticket's plan has unfinished steps: the send was an
                # intermediate note and the remaining steps still need turns.
                if (
                    tool_name == ROBOT_SEND_TOOL_NAME
                    and qq_message_sent_this_turn
                    and not robot_send_has_explicit_destination(tool_args)
                    and not _ticket_has_open_plan_steps(reply_ticket.ticket_id)
                ):
                    _broadcast_agent_status(item_id, "idle")
                    yield _to_sse({"done": True})
                    return

                if (
                    tool_name in COMMAND_TOOL_NAMES
                    and result.get("success")
                ):
                    clear_pending_terminal_continuation(
                        item_id,
                        command=str(tool_args.get("command") or ""),
                    )
                    if terminal_session is None:
                        terminal_session = agent_session_manager.get_or_create_session(
                            item_id,
                            str(handler.id),
                        )
                    terminal_session.mark_terminal_command_dispatched(tool_name, tool_args)
                    waiting_event = _broadcast_agent_status(
                        item_id,
                        "waiting_terminal",
                        "命令已发送，等待终端反馈",
                        {"terminal_source": "raw_feedback", "tool_name": tool_name},
                    )
                    yield _to_sse(waiting_event)
                    yield _to_sse({"done": True})
                    return
                assistant_message["tool_calls"].append(
                    {
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": normalized_tool_args_str,
                        },
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result_text,
                    }
                )

            if background_job_started_this_iteration:
                _broadcast_agent_status(item_id, "idle")
                yield _to_sse({"done": True})
                return

            if not assistant_message["tool_calls"]:
                # Providers like ZAI reject assistant messages carrying an
                # empty tool_calls array. Drop the key; skip the message
                # entirely when it carries no content and produced no tool
                # results (e.g. every call was blocked by the tool-loop guard).
                assistant_message.pop("tool_calls", None)
            if (
                assistant_message.get("tool_calls")
                or str(assistant_message.get("content") or "").strip()
                or tool_messages
            ):
                messages.append(assistant_message)
            messages.extend(tool_messages)

        logger.warning(
            "[Chat] Tool iteration budget exhausted item=%s ticket=%s iterations=%s",
            item_id,
            reply_ticket.ticket_id,
            MAX_ITERATIONS,
        )
        for event in _finalize_stopped_turn(
            agent=agent,
            handler=handler,
            item_id=item_id,
            messages=messages,
            reason=TOOL_BUDGET_STOP_REASON,
            prefers_chinese=_contains_cjk(message),
            include_hidden_tool_results=include_hidden_tool_results,
            reply_ticket_id=reply_ticket.ticket_id,
        ):
            yield _to_sse(event)
        _broadcast_agent_status(item_id, "idle")
        yield _to_sse({"done": True})
    except Exception as exc:
        logger.exception("[Chat] Unexpected stream error for item %s", item_id)
        for event in _finalize_stopped_turn(
            agent=agent,
            handler=handler,
            item_id=item_id,
            messages=messages,
            reason=f"Unexpected Agent error: {exc}",
            prefers_chinese=_contains_cjk(message),
            include_hidden_tool_results=include_hidden_tool_results,
            reply_ticket_id=reply_ticket.ticket_id,
        ):
            yield _to_sse(event)
        yield _to_sse({"done": True})
    finally:
        reply_ticket_manager.detach_from_agent(agent, reply_ticket.ticket_id)


def generate_stream(
    message: str,
    history: list[ChatMessage],
    handler: ItemHandler,
    item_id: str,
    agent: "Agent" = None,
    include_hidden_tool_results: bool = False,
    latest_only_context: bool = False,
    source_type: str = SOURCE_WEB,
    reply_ticket_id: str = "",
    turn_serialized: bool = False,
    image_urls: list[str] | None = None,
    force_image_vision: bool = False,
    agent_image_urls: list[str] | None = None,
) -> Generator[str, None, None]:
    if turn_serialized:
        yield from _generate_stream_unserialized(
            message=message,
            history=history,
            handler=handler,
            item_id=item_id,
            agent=agent,
            include_hidden_tool_results=include_hidden_tool_results,
            latest_only_context=latest_only_context,
            source_type=source_type,
            reply_ticket_id=reply_ticket_id,
            image_urls=image_urls,
            force_image_vision=force_image_vision,
            agent_image_urls=agent_image_urls,
        )
        return
    with agent_turn_coordinator.turn(agent_turn_key(str(handler.id))):
        yield from _generate_stream_unserialized(
            message=message,
            history=history,
            handler=handler,
            item_id=item_id,
            agent=agent,
            include_hidden_tool_results=include_hidden_tool_results,
            latest_only_context=latest_only_context,
            source_type=source_type,
            reply_ticket_id=reply_ticket_id,
            image_urls=image_urls,
            force_image_vision=force_image_vision,
            agent_image_urls=agent_image_urls,
        )


def _require_chat_handler(
    session: SessionDep,
    item_id: str,
    current_user: CurrentUser,
) -> tuple[ItemHandler, Item]:
    prepared = _get_item_handler_cached(session, item_id, current_user)
    if not prepared:
        raise HTTPException(
            status_code=404,
            detail="No ItemHandler associated with this item.",
        )
    return prepared


def _run_chat_turn_sync(
    message: str,
    history: list,
    handler,
    item_id: str,
    agent,
    image_urls: list[str] | None = None,
) -> tuple[str, str]:
    """在 worker 线程里跑完整个 agent 轮次——同步生成器会阻塞数分钟，
    绝不能在事件循环里直接迭代（会把整个 backend 卡死）。"""
    content = ""
    error_message = ""
    for chunk in generate_stream(
        message=message,
        history=history,
        handler=handler,
        item_id=item_id,
        agent=agent,
        turn_serialized=True,
        image_urls=image_urls,
    ):
        if not chunk.startswith("data: "):
            continue
        payload = json.loads(chunk[6:].strip())
        if payload.get("type") == "agent_response":
            content = payload.get("content", content)
        elif payload.get("type") in {"agent_error", "error"}:
            error_message = payload.get("content", error_message)
    return content, error_message


@router.post("/{item_id}")
async def chat(
    item_id: str,
    request: ChatRequest,
    session: SessionDep,
    current_user: CurrentUser,
) -> dict:
    from fastapi.concurrency import run_in_threadpool

    prepared = _require_chat_handler(session, item_id, current_user)
    lease = await agent_turn_coordinator.acquire_async(
        agent_turn_key(str(prepared[0].id))
    )
    try:
        handler, _, agent = await prepare_chat_agent(
            session,
            item_id,
            current_user,
            prepared=prepared,
        )
        matched_skills: list = []

        content, error_message = await run_in_threadpool(
            _run_chat_turn_sync,
            request.message,
            request.history,
            handler,
            item_id,
            agent,
            request.images or None,
        )
    finally:
        lease.release()

    if error_message:
        raise HTTPException(status_code=500, detail=error_message)

    if not content and not error_message:
        content = "已提交后台任务，结果会自动推送～"

    return {
        "content": content,
        "model": handler.model,
        "matched_skills": [skill.skill_id for skill in matched_skills],
    }


@router.post("/{item_id}/stream")
async def chat_stream(
    item_id: str,
    request: ChatStreamRequest,
    session: SessionDep,
    current_user: CurrentUser,
):
    import queue as _queue

    prepared = _require_chat_handler(session, item_id, current_user)
    handler, _, agent = await prepare_chat_agent(
        session,
        item_id,
        current_user,
        prepared=prepared,
    )

    from app.services.agent.session import agent_session_manager

    agent_session = agent_session_manager.get_or_create_session(
        item_id, str(handler.id)
    )

    sse_queue: _queue.Queue[dict | None] = _queue.Queue()

    def _on_output(event: dict) -> None:
        sse_queue.put_nowait(event)

    def _on_complete(success: bool, error: str = "") -> None:
        if not success and error:
            sse_queue.put_nowait(
                {"type": "agent_error", "content": error}
            )
        sse_queue.put(None)

    user_event = _persist_and_broadcast_event(
        item_id,
        role="user",
        content=request.message,
        message_type="chat_user",
    )

    from app.services.agent.session import InputMessage, InputType

    input_msg = InputMessage(
        input_type=InputType.CHAT,
        content=request.message,
        query=request.message,
        completion_callback=_on_complete,
        source_label="web 聊天",
        image_urls=list(request.images or []),
    )
    agent_session.add_output_callback(_on_output)
    import asyncio as _aio

    await _aio.get_event_loop().run_in_executor(
        None,
        lambda: agent_session.process_input(input_msg),
    )

    def queued_stream() -> Generator[str, None, None]:
        try:
            yield _to_sse(user_event)
            while True:
                try:
                    event = sse_queue.get(timeout=180)
                except _queue.Empty:
                    yield _to_sse(
                        {
                            "type": "agent_warning",
                            "content": "Response timed out. The agent may be busy with another task.",
                        }
                    )
                    break
                if event is None:
                    break
                yield _to_sse(event)
            yield _to_sse({"done": True})
        finally:
            agent_session.remove_output_callback(_on_output)

    return StreamingResponse(
        queued_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{item_id}/skills")
async def get_matched_skills(
    item_id: str,
    query: str,
    session: SessionDep,
    current_user: CurrentUser,
):
    prepared = _require_chat_handler(session, item_id, current_user)
    lease = await agent_turn_coordinator.acquire_async(
        agent_turn_key(str(prepared[0].id))
    )
    try:
        _, _, agent = await prepare_chat_agent(
            session,
            item_id,
            current_user,
            prepared=prepared,
        )
        matched = list(agent.get_skills())
    finally:
        lease.release()
    return {
        "matched_skills": [
            {
                "skill_id": skill.skill_id,
                "name": skill.name,
                "description": skill.description,
            }
            for skill in matched
        ]
    }


class AgentMessageQueue:
    _abort_flags: dict[str, bool] = {}
    _lock = threading.Lock()

    @classmethod
    def set_abort(cls, item_id: str, abort: bool = True):
        with cls._lock:
            cls._abort_flags[item_id] = abort

    @classmethod
    def is_aborted(cls, item_id: str) -> bool:
        with cls._lock:
            return cls._abort_flags.get(item_id, False)

    @classmethod
    def clear_abort(cls, item_id: str):
        with cls._lock:
            cls._abort_flags.pop(item_id, None)


def _agent_message_callback(item_id: str, msg_queue: queue.Queue):
    def callback(message: dict):
        try:
            msg_queue.put_nowait(message)
        except queue.Full:
            try:
                msg_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                msg_queue.put_nowait(message)
            except queue.Full:
                logger.warning(
                    f"[Chat] Dropping agent event for item {item_id}: subscriber queue full"
                )

    return callback


@router.get("/{item_id}/agent-events")
async def agent_events(
    item_id: str,
    session: SessionDep,
    current_user: CurrentUser,
):
    if not get_item_handler_llm_config(session, item_id, current_user):
        raise HTTPException(status_code=404, detail="No handler found for this item")

    msg_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
    callback = _agent_message_callback(item_id, msg_queue)
    stream_manager.register_chat(item_id, callback)

    def event_generator():
        try:
            while True:
                try:
                    message = msg_queue.get(timeout=30)
                    yield _to_sse(message)
                except queue.Empty:
                    yield ": heartbeat\n\n"
        finally:
            stream_manager.unregister_chat(item_id, callback)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{item_id}/abort")
async def abort_chat(
    item_id: str,
    session: SessionDep,
    current_user: CurrentUser,
):
    if not get_item_handler_llm_config(session, item_id, current_user):
        raise HTTPException(status_code=404, detail="No handler found for this item")

    AgentMessageQueue.set_abort(item_id, True)
    stream_manager.reset_session(item_id)

    active_session = agent_session_manager.get_session(item_id)
    if active_session:
        active_session.abort(clear_queue=True)
        active_session.emit_output("\u672c\u6b21\u4f1a\u8bdd\u5df2\u4e2d\u65ad", "agent_warning")
        active_session.emit_status("idle", "")

    return {
        "success": True,
        "message": "Chat aborted; background jobs continue running",
        "cancel_result": None,
        "background_jobs_preserved": True,
    }
