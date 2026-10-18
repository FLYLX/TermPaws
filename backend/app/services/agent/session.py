import asyncio
import hashlib
import json
import logging
import os
import queue
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from litellm import completion

from app.core.config import settings
from app.core.tool_markup import extract_dsml_tool_calls
from app.services.agent.agent import Agent, agent_manager
from app.services.agent.history.chat import append_chat_message
from app.services.agent.integrations import (
    clear_integration_chat_contexts,
    clear_terminal_alert_integration_tools,
    ensure_integration_chat_context_tools,
    ensure_terminal_alert_integration_tools,
    extract_integration_context_targets,
    integration_message_sent,
    send_integration_final_response_fallback,
    setup_integration_chat_contexts,
    should_enable_terminal_alert_integrations,
)
from app.services.agent.pending_context import (
    attach_terminal_feedback_to_pending_continuation,
    build_pending_terminal_continuation_prompt,
    clear_pending_terminal_continuation,
    record_pending_terminal_continuation,
)
from app.services.agent.prompts import builder as prompt_builder
from app.services.agent.prompts.builder import (
    build_chat_turn_messages,
    build_terminal_turn_messages,
    is_critical_terminal_event,
)
from app.services.agent.prompts.system import get_system_prompt
from app.services.agent.robot_delivery import (
    ROBOT_QQ_REPLY_EVENT_TYPE,
    ROBOT_SEND_TOOL_NAME,
    robot_reply_event_content,
)
from app.services.agent.token_usage import token_usage_tracker
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
from app.services.terminal_command_state import terminal_command_state_manager
from app.services.terminal_runtime_state import get_terminal_runtime_state

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = settings.CHAT_LLM_TIMEOUT_SECONDS
MAX_ITERATIONS = 6
LOOP_DETECTION_WINDOW = 8
LOOP_THRESHOLD = 4
MAX_QUEUE_SIZE = 20
MAX_TOOL_CALLS = 15
MAX_REPEATED_LOG_READS = 2
MAX_NO_PROGRESS_STEPS = 5
MAX_TURN_DURATION_SECONDS = int(settings.AGENT_TURN_TIMEOUT_SECONDS)
PENDING_COMMAND_TIMEOUT_SECONDS = 20
PENDING_COMMAND_RECHECK_SECONDS = 0.25
PENDING_COMMAND_STALLED_CONFIRM_SECONDS = 2.0
READ_LOG_TOOL_NAME = "mcp_local_read_terminal_log"
COMMAND_TOOL_NAMES = {
    "mcp_local_execute_command",
    "mcp_local_interrupt_command",
}
COMMAND_DISPATCH_FAILURE_MESSAGE = "终端未连接或未打开，命令没有发送。请先启动或连接终端后再试。"
COMMAND_DISPATCH_FAILURE_MARKERS = (
    "命令发送失败",
    "终端未连接",
    "终端未打开",
    "终端未收到命令",
)
EXECUTE_COMMAND_TOOL_NAME = "mcp_local_execute_command"
RUN_JOB_TOOL_NAME = "mcp_local_run_job"
BACKGROUND_JOB_STARTED_MARKER = "background_job_started"
COMMAND_DISPATCH_FAILED_MARKER = "command_dispatch_failed"
TERMINAL_UNAVAILABLE_RESULT_MARKER = "terminal_unavailable"
BACKGROUND_JOB_STARTED_RESPONSE = (
    "\u540e\u53f0\u4efb\u52a1\u5df2\u542f\u52a8\uff0c"
    "\u5b8c\u6210\u540e\u6211\u4f1a\u6839\u636e\u7ed3\u679c\u7ee7\u7eed\u5904\u7406"
    "\u5e76\u56de\u5230\u5bf9\u5e94\u6765\u6e90\u3002"
)
COMMAND_DISPATCH_PENDING_MARKER = "命令已发送到终端，尚未确认执行结果:"
TERMINAL_INPUT_MODE_BUSY = "busy"
TERMINAL_INPUT_MODE_CONSOLE = "console"
TERMINAL_INPUT_CONTEXT_TTL_SECONDS = 6 * 60 * 60
DAEMON_JOBS_CONTEXT_REFRESH_SECONDS = 3.0

def _is_transient_llm_error(exc: BaseException) -> bool:
    """Return True for LLM API errors that are worth retrying."""
    name = type(exc).__name__
    msg = str(exc or "").lower()
    transient_names = {
        "BadRequestError",
        "RateLimitError",
        "ServiceUnavailableError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ZaiException",
    }
    if name in transient_names:
        return True
    transient_markers = (
        "rate limit",
        "timeout",
        "connection",
        "unavailable",
        "overloaded",
        "server_error",
        "openai_error",
        "bad request",
        "zaiexception",
    )
    return any(marker in msg for marker in transient_markers)



class _ManagedEventLoop:
    """Context manager for a thread-local event loop with safe cleanup.

    Prevents 'Cannot run the event loop while another loop is running' by
    verifying no loop is active, and always cleans up on exit so stale
    loop references never leak into the next turn.
    """

    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None

    def __enter__(self) -> asyncio.AbstractEventLoop:
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "AgentSession turn called from an async context. "
                "Use run_in_executor or process_input (queue-driven) instead."
            )
        except RuntimeError as exc:
            if "no running event loop" not in str(exc).lower() and "no current event loop" not in str(exc).lower():
                raise
        # Close any stale event loop left over from a previous crashed turn
        # so asyncio.new_event_loop() + run_until_complete() never hits
        # "Cannot run the event loop while another loop is running".
        try:
            stale = asyncio.get_event_loop_policy().get_event_loop()
            if stale is not None and not stale.is_running() and not stale.is_closed():
                stale.close()
        except Exception:
            pass
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        return self.loop

    def __exit__(self, *exc_info):
        if self.loop is not None:
            try:
                if not self.loop.is_closed():
                    self.loop.close()
            except Exception:
                pass
            asyncio.set_event_loop(None)
            self.loop = None
        return False


TERMINAL_BUSY_COMMAND_PATTERNS = (
    r"^(?:sudo\s+)?(?:apt|apt-get|aptitude)\s+(?:update|upgrade|full-upgrade|dist-upgrade|install|remove|autoremove)\b",
    r"^(?:sudo\s+)?(?:dnf|yum)\s+(?:install|update|upgrade|remove|groupinstall)\b",
    r"^(?:sudo\s+)?apk\s+(?:add|update|upgrade|del)\b",
    r"^(?:sudo\s+)?pacman\s+-S(?:yu?)?\b",
    r"^(?:sudo\s+)?zypper\s+(?:install|update|refresh|remove)\b",
    r"^(?:python\d*(?:\.\d+)?\s+-m\s+pip|pip\d*|uv\s+pip)\s+install\b",
    r"^uv\s+sync\b",
    r"^(?:npm|pnpm|yarn|bun)\s+(?:install|ci|add|update|upgrade)\b",
    r"^docker\s+(?:build|compose\s+build|compose\s+up)\b",
    r"^(?:curl|wget)\b.*(?:https?://|ftp://)",
    r"^git\s+(?:clone|pull|fetch|submodule\s+update)\b",
    r"^(?:make|cmake\s+--build|cargo\s+(?:build|install)|go\s+(?:build|install)|mvn|gradle|\./gradlew)\b",
)
TERMINAL_CONSOLE_COMMAND_PATTERNS = (
    r"\bjava\s+.*(?:-jar\s+\S*(?:server|paper|spigot|forge|fabric|bukkit|mohist|arclight|minecraft)\S*|nogui)\b",
    r"\bjava\s+.*@(?:\S*/)?libraries/\S*(?:minecraftforge|forge|fabric|minecraft)\S*/(?:unix_args|win_args)\.txt\b",
    r"^(?:sudo\s+)?(?:(?:bash|sh)\s+)?(?:\./)?(?:run|start|startserver|server)\.sh\b",
    r"\bbedrock_server\b",
    r"^(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?(?:dev|start|serve|watch)\b",
    r"^(?:vite|next\s+dev|nuxt\s+dev|astro\s+dev)\b",
    r"^(?:python\d*(?:\.\d+)?\s+-m\s+)?(?:http\.server|uvicorn|fastapi)\b",
    r"^(?:uvicorn|hypercorn|gunicorn)\b",
    r"^(?:tail\s+-f|journalctl\s+-f|docker\s+(?:compose\s+)?logs\s+-f|watch\s+)\b",
    r"^(?:bash|sh|zsh|fish|python\d*(?:\.\d+)?|node|deno)\s*$",
)
MINECRAFT_CONSOLE_COMMANDS = frozenset(
    {
        "advancement",
        "ban",
        "ban-ip",
        "banlist",
        "clear",
        "deop",
        "difficulty",
        "effect",
        "enchant",
        "execute",
        "experience",
        "fill",
        "forceload",
        "function",
        "gamemode",
        "gamerule",
        "give",
        "help",
        "kick",
        "kill",
        "list",
        "locate",
        "me",
        "msg",
        "op",
        "pardon",
        "pardon-ip",
        "particle",
        "playsound",
        "reload",
        "save-all",
        "save-off",
        "save-on",
        "say",
        "schedule",
        "scoreboard",
        "seed",
        "setblock",
        "setidletimeout",
        "setworldspawn",
        "spawnpoint",
        "spectate",
        "spreadplayers",
        "stop",
        "summon",
        "tag",
        "team",
        "teleport",
        "tell",
        "tellraw",
        "time",
        "title",
        "tp",
        "weather",
        "whitelist",
        "worldborder",
        "xp",
    }
)
SILENT_TOOL_NAMES = {
    "mcp_local_get_terminal_status",
    READ_LOG_TOOL_NAME,
    RUN_JOB_TOOL_NAME,
    "mcp_robot_send_message",
    "mcp_robot_sleep_conversation",
    "mcp_robot_save_memory",
}
TERMINAL_SOURCE_FILTERED = "filtered_output"
TERMINAL_SOURCE_RAW_FEEDBACK = "raw_feedback"
MINECRAFT_PLAYER_CHAT_LINE_RE = re.compile(
    r"(?m)^\s*(?:\[[^\]\n]+\]\s*){0,4}"
    r"(?:\[[^\]\n]*INFO[^\]\n]*\]\s*)?"
    r"(?:(?:\[[^\]\n]+\]|[^\n:：]{1,80})[:：]\s*)?"
    r"(?:\[Not Secure\]\s*)?<[^>\n]{1,64}>\s+\S"
)
MINECRAFT_PLAYER_CHAT_CAPTURE_RE = re.compile(
    r"(?m)^.*?<(?P<player>[^>\n]{1,64})>\s+(?P<message>\S.*)$"
)
MINECRAFT_RELAY_COMMAND_RE = re.compile(
    r"^\s*/?(?:(?:minecraft):)?(?:tell|msg|w|whisper)\s+"
    r"(?P<player>[A-Za-z0-9_]{1,64})\s+(?P<question>\S[\s\S]*)$",
    re.IGNORECASE,
)
TERMINAL_NO_REPLY_MARKERS = (
    "[no_terminal_reply]",
    "[no_reply]",
    "[no_qq_reply]",
    "nrn",
    "不回复",
    "不用回复",
    "不用回",
    "不打扰",
)


def is_command_dispatch_failure_result(
    tool_name: str,
    result_text: str,
    result: Any = None,
) -> bool:
    if tool_name not in COMMAND_TOOL_NAMES and tool_name != RUN_JOB_TOOL_NAME:
        return False
    if isinstance(result, dict):
        result_data = result.get("result")
        if isinstance(result_data, list):
            for item in result_data:
                if not isinstance(item, dict) or item.get("type") != "metadata":
                    continue
                if bool(item.get(COMMAND_DISPATCH_FAILED_MARKER)):
                    return True
                if bool(item.get(TERMINAL_UNAVAILABLE_RESULT_MARKER)):
                    return True
                if item.get("reason") in {
                    TERMINAL_UNAVAILABLE_RESULT_MARKER,
                    "terminal_status_unavailable",
                }:
                    return True
    return any(marker in (result_text or "") for marker in COMMAND_DISPATCH_FAILURE_MARKERS)


def is_terminal_unavailable_error(message: str) -> bool:
    normalized = str(message or "").casefold()
    return any(
        marker in normalized
        for marker in (
            "终端未连接",
            "终端未打开",
            "terminal is not connected",
            "terminal not connected",
            "terminal unavailable",
            "terminal_unavailable",
            "terminal_status_unavailable",
            "\u7ec8\u7aef\u672a\u542f\u52a8",
            "\u7ec8\u7aef\u672a\u8fde\u63a5",
            "\u547d\u4ee4\u6ca1\u6709\u53d1\u9001",
        )
    )


def is_command_dispatch_pending_result(tool_name: str, result_text: str) -> bool:
    return (
        tool_name == EXECUTE_COMMAND_TOOL_NAME
        and COMMAND_DISPATCH_PENDING_MARKER in (result_text or "")
    )


def is_background_job_started_result(result: Any) -> bool:
    if not isinstance(result, dict) or not result.get("success"):
        return False
    result_data = result.get("result")
    if not isinstance(result_data, list):
        return False
    return any(
        isinstance(item, dict) and bool(item.get(BACKGROUND_JOB_STARTED_MARKER))
        for item in result_data
    )


def _normalize_command_for_routing(command: str) -> str:
    return " ".join((command or "").strip().split()).lower()


def _classification_command_variants(command: str) -> list[str]:
    normalized = _normalize_command_for_routing(command)
    if not normalized:
        return []

    variants = [normalized]
    # Strip leading env assignments (optionally after sudo) so anchored
    # patterns still match: DEBIAN_FRONTEND=x apt-get install -> apt-get install
    env_stripped = re.sub(
        r"^((?:sudo\s+)?)(?:[a-z_][a-z0-9_]*=\S+\s+)+",
        r"\1",
        normalized,
        flags=re.IGNORECASE,
    )
    if env_stripped and env_stripped != normalized:
        variants.append(env_stripped)
    remainder = env_stripped or normalized
    while True:
        match = re.match(r"^(?:cd|pushd)\s+[^;&|]+\s*(?:&&|;)\s*(.+)$", remainder)
        if not match:
            break
        remainder = match.group(1).strip()
        if not remainder or remainder in variants:
            break
        variants.append(remainder)
    return variants


def classify_terminal_input_mode(command: str) -> str | None:
    variants = _classification_command_variants(command)
    if not variants:
        return None
    if any(
        re.search(pattern, variant)
        for variant in variants
        for pattern in TERMINAL_CONSOLE_COMMAND_PATTERNS
    ):
        return TERMINAL_INPUT_MODE_CONSOLE
    if any(
        re.search(pattern, variant)
        for variant in variants
        for pattern in TERMINAL_BUSY_COMMAND_PATTERNS
    ):
        return TERMINAL_INPUT_MODE_BUSY
    return None


def _coerce_timeout_seconds(value: Any, default: int = PENDING_COMMAND_TIMEOUT_SECONDS) -> int:
    if value is None or value == "":
        return default
    try:
        timeout = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(1, min(timeout, 600))


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _normalize_optional_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _first_command_token(command: str) -> str:
    normalized = _normalize_command_for_routing(command)
    if not normalized:
        return ""
    if normalized.startswith("/"):
        normalized = normalized[1:].strip()
    return normalized.split(maxsplit=1)[0]


def is_terminal_console_command(command: str) -> bool:
    stripped = (command or "").strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        return False
    if re.search(r"[;&|`$<>]", stripped):
        return False
    return _first_command_token(stripped) in MINECRAFT_CONSOLE_COMMANDS


def terminal_content_looks_like_server_chat(content: str) -> bool:
    return bool(MINECRAFT_PLAYER_CHAT_LINE_RE.search(content or ""))


def is_terminal_no_reply_intent(content: str) -> bool:
    normalized = " ".join((content or "").strip().lower().split())
    if not normalized:
        return True
    return len(normalized) <= 120 and any(
        marker.lower() in normalized for marker in TERMINAL_NO_REPLY_MARKERS
    )


def agent_has_litellm_tool(agent: Agent, tool_name: str) -> bool:
    try:
        tools = agent.get_tools_for_litellm()
    except Exception:
        return False
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == tool_name:
            return True
    return False


def should_retry_terminal_source_delivery(
    *,
    agent: Agent,
    terminal_content: str,
    terminal_source: str,
    final_response: str,
    retry_used: bool,
) -> bool:
    if retry_used or terminal_source != TERMINAL_SOURCE_FILTERED:
        return False
    if is_terminal_no_reply_intent(final_response):
        return False
    if not terminal_content_looks_like_server_chat(terminal_content):
        return False
    return agent_has_litellm_tool(agent, EXECUTE_COMMAND_TOOL_NAME)


def build_terminal_source_delivery_correction_message(final_response: str) -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "终端/服务器来源回复反思：你刚才生成了最终文字，但没有写回消息来源。"
            "当前终端输入像服务器内玩家聊天；如果这段文字是在回答服务器内的人，"
            "现在必须调用 `mcp_local_execute_command` 回到同一个终端。"
            "Minecraft/类 Minecraft 控制台用 `say <回复内容>`，需要私聊玩家时用 `tell <玩家名> <回复内容>`。"
            "只调用工具，不要再次输出普通最终文本。\n"
            f"刚才准备回复的内容:\n{final_response.strip()}"
        ),
    }


def _fake_llm_response():
    """TERMPAWS_FAKE_LLM=1 时的固定应答：无工具调用、无思考。
    TERMPAWS_FAKE_LLM_DELAY_MS 可让假轮次变慢，用于压"边处理边合并"。"""
    delay_ms = int(os.environ.get("TERMPAWS_FAKE_LLM_DELAY_MS") or 0)
    if delay_ms > 0:
        time.sleep(delay_ms / 1000)
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="[fake-llm] ok",
                    tool_calls=None,
                    reasoning_content=None,
                )
            )
        ],
        usage=None,
    )


class SessionState(Enum):
    IDLE = "idle"
    COLLECTING = "collecting"
    RUNNING = "running"
    INTERRUPTING = "interrupting"
    COOLDOWN = "cooldown"


class InputType(Enum):
    TERMINAL = "terminal"
    CHAT = "chat"
    SCHEDULED_TASK = "scheduled_task"


@dataclass
class InputMessage:
    input_type: InputType
    content: str
    raw_content: str = ""
    query: str = ""
    reply_ticket_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    callback: Callable | None = None
    completion_callback: Callable[[bool, str], None] | None = None
    scheduled_task_id: str = ""
    scheduled_execution_id: str = ""
    # Human-readable request shown in the plan table; structured alternative
    # to parsing the agent-facing wrapper text.
    request_display: str = ""
    # Source identity bound at enqueue time (e.g. "QQ 私聊 2537134688",
    # "web 聊天"). The merge layer never guesses origins: whoever enqueues
    # labels the input, and the batch prompt groups sections by this label so
    # the agent can see which message belongs to which reply route.
    source_label: str = ""
    # All reply routes merged into this input (ticket ids, in arrival order).
    # Set by _merge_input_batch; delivery fans out per route afterwards.
    merged_ticket_ids: tuple[str, ...] = ()
    image_urls: list[str] = field(default_factory=list)
    force_image_vision: bool = False


@dataclass
class TurnGuard:
    turn_id: str = field(default_factory=lambda: uuid4().hex[:8])
    started_at: datetime = field(default_factory=datetime.now)
    tool_call_count: int = 0
    tool_names: list[str] = field(default_factory=list)
    no_progress_steps: int = 0
    last_progress_token: str | None = None
    last_log_fingerprint: str | None = None
    repeated_log_reads: int = 0
    pending_hint: str = ""
    recent_steps: deque[str] = field(
        default_factory=lambda: deque(maxlen=LOOP_DETECTION_WINDOW)
    )

    def check_timeout(self) -> tuple[bool, str]:
        elapsed = (datetime.now() - self.started_at).total_seconds()
        if elapsed > MAX_TURN_DURATION_SECONDS:
            return True, f"\u5f53\u524d\u8f6e\u5904\u7406\u8d85\u65f6 ({MAX_TURN_DURATION_SECONDS}s)\uff0c\u5df2\u505c\u6b62"
        return False, ""

    def reset_timeout_window(self) -> None:
        self.started_at = datetime.now()
        self.no_progress_steps = 0

    def before_tool(self, tool_name: str, tool_args_str: str) -> tuple[bool, str]:
        self.tool_call_count += 1
        self.tool_names.append(str(tool_name or "").strip())
        if self.tool_call_count > MAX_TOOL_CALLS:
            return True, f"当前轮工具调用次数超过限制 ({MAX_TOOL_CALLS})，已停止"

        if tool_name != READ_LOG_TOOL_NAME:
            normalized_args = (tool_args_str or "").strip().replace("\n", " ")[:256]
            step_token = f"{tool_name}:{normalized_args}"
            self.recent_steps.append(step_token)
            if len(self.recent_steps) >= LOOP_THRESHOLD:
                recent = list(self.recent_steps)[-LOOP_THRESHOLD:]
                if all(token == step_token for token in recent):
                    self.pending_hint = (
                        f"[Hint] You have repeated the same action ({tool_name}) "
                        f"{LOOP_THRESHOLD} times with the same result. Consider a "
                        "different approach or report the current status."
                    )


        return False, ""

    def record_progress(self, progress_token: str | None) -> tuple[bool, str]:
        if not progress_token:
            self.no_progress_steps += 1
        elif progress_token == self.last_progress_token:
            self.no_progress_steps += 1
        else:
            self.last_progress_token = progress_token
            self.no_progress_steps = 0

        if self.no_progress_steps >= MAX_NO_PROGRESS_STEPS:
            self.pending_hint = (
                "[Hint] Multiple consecutive steps produced no new progress. "
                "Consider changing your approach, reporting the blocker, or "
                "completing the task with current results."
            )
            self.no_progress_steps = 0

        return False, ""

    def after_tool(self, tool_name: str, result_text: str) -> tuple[bool, str]:
        normalized_result = result_text.strip()
        result_fingerprint = hashlib.sha1(normalized_result[:4000].encode("utf-8")).hexdigest()

        if tool_name == READ_LOG_TOOL_NAME:
            if result_fingerprint == self.last_log_fingerprint:
                self.repeated_log_reads += 1
            else:
                self.last_log_fingerprint = result_fingerprint
                self.repeated_log_reads = 0

            if self.repeated_log_reads >= MAX_REPEATED_LOG_READS:
                self.pending_hint = (
                    "[Hint] Log content has not changed. Consider summarizing "
                    "based on existing output or taking a different action."
                )
                self.repeated_log_reads = 0

            return self.record_progress(f"log:{result_fingerprint}")

        return self.record_progress(f"{tool_name}:{result_fingerprint}")


@dataclass
class PendingCommand:
    tool_name: str
    command: str
    normalized_command: str
    integration_contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    log_line_cursor: int = 0
    dispatched_at: datetime = field(default_factory=datetime.now)
    echo_count: int = 0
    timeout_warned: bool = False
    input_mode: str | None = None
    timeout_seconds: int = PENDING_COMMAND_TIMEOUT_SECONDS
    auto_interrupt_on_timeout: bool = False
    integration_response_sent: bool = False
    reply_ticket_id: str = ""


@dataclass
class TerminalInputContext:
    command: str
    normalized_command: str
    input_mode: str
    created_at: datetime = field(default_factory=datetime.now)
    last_seen_at: datetime = field(default_factory=datetime.now)


@dataclass
class RunningTerminalJob:
    tool_name: str
    command: str
    normalized_command: str
    timeout_seconds: int = 600
    reply_ticket_id: str = ""
    source_type: str = ""
    source_label: str = ""
    started_at: datetime = field(default_factory=datetime.now)


@dataclass
class TerminalAnalysisResult:
    content: str | None
    terminal_source: str
    direct_response: bool = False


class AgentSession:
    def __init__(self, item_id: str, handler_id: str):
        self.item_id = item_id
        self.handler_id = handler_id
        self.agent: Agent | None = None
        self.state = SessionState.IDLE
        self.input_queue: queue.Queue[InputMessage] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.output_callbacks: list[Callable] = []
        self.lock = threading.RLock()
        self._abort_flag = False
        self._aborted_this_turn = False
        self._last_activity = datetime.now()
        self._current_turn_id: str | None = None
        self._last_status_signature: tuple[str, str, str] | None = None
        self._pending_command: PendingCommand | None = None
        self._pending_command_recheck_timer: threading.Timer | None = None
        self._terminal_input_context: TerminalInputContext | None = None
        self._running_terminal_job: RunningTerminalJob | None = None
        self._daemon_jobs_snapshot: list[dict[str, Any]] = []
        self._daemon_jobs_snapshot_at: datetime | None = None
        self._recent_finished_jobs: list[dict[str, Any]] = []
        # 会话级已验证可用工具白名单（tool -> 备注），由 job 沉淀钩子维护
        self._verified_tools: dict[str, str] = {}

        self._wake_event = threading.Event()
        self._consumer_shutdown = False
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop, daemon=True, name=f"agent-consumer-{item_id[:8]}"
        )
        self._consumer_thread.start()

        logger.info(f"[AgentSession] Created session for item={item_id}, handler={handler_id}")

    def get_agent(self) -> Agent | None:
        try:
            from sqlmodel import Session

            from app.core.db import engine
            from app.models import ItemHandler

            with Session(engine) as session:
                handler = session.get(ItemHandler, self.handler_id)
                if handler:
                    self.agent = agent_manager.get_or_create(handler)
                    logger.debug(f"[AgentSession] Refreshed agent for handler {self.handler_id}")
                    return self.agent
        except Exception as exc:
            logger.error(f"[AgentSession] Failed to get agent: {exc}")

        return self.agent

    def add_output_callback(self, callback: Callable):
        with self.lock:
            if callback not in self.output_callbacks:
                self.output_callbacks.append(callback)

    def remove_output_callback(self, callback: Callable):
        with self.lock:
            if callback in self.output_callbacks:
                self.output_callbacks.remove(callback)

    def _broadcast_event(self, event: dict[str, Any]):
        with self.lock:
            callbacks = self.output_callbacks.copy()

        for callback in callbacks:
            try:
                callback(event)
            except Exception as exc:
                logger.error(f"[AgentSession] Error in output callback: {exc}")

    def emit_output(
        self,
        message: str,
        msg_type: str = "agent_response",
        extra: dict[str, Any] | None = None,
        *,
        persist: bool = True,
    ):
        if not message:
            return

        event: dict[str, Any] = {
            "type": msg_type,
            "content": message,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            event.update(extra)

        if persist:
            try:
                append_chat_message(
                    self.item_id,
                    role=None,
                    content=message,
                    message_type=msg_type,
                    timestamp=event["timestamp"],
                    extra=extra,
                )
            except Exception as exc:
                logger.error(f"[AgentSession] Failed to persist output: {exc}")

        self._broadcast_event(event)

    def emit_status(
        self,
        status: str,
        content: str | None = None,
        extra: dict[str, Any] | None = None,
    ):
        payload = content if content is not None else ""
        extra_signature = ""
        if extra:
            try:
                extra_signature = json.dumps(extra, sort_keys=True, ensure_ascii=False)
            except TypeError:
                extra_signature = str(sorted(extra.items()))
        signature = (status, payload, extra_signature)
        if signature == self._last_status_signature:
            return

        self._last_status_signature = signature
        event: dict[str, Any] = {
            "type": "agent_status",
            "status": status,
            "content": payload,
            "timestamp": datetime.now().isoformat(),
        }
        if self._current_turn_id:
            event["turn_id"] = self._current_turn_id
        if extra:
            event.update(extra)
        self._broadcast_event(event)

    def _emit_collecting_terminal_status(self):
        self.emit_status(
            "collecting",
            "\u63a5\u6536\u8fc7\u6ee4\u540e\u7ec8\u7aef\u8f93\u51fa\u4e2d",
            {"terminal_source": TERMINAL_SOURCE_FILTERED},
        )

    def _emit_running_terminal_status(self, terminal_source: str):
        if terminal_source == TERMINAL_SOURCE_RAW_FEEDBACK:
            content = "\u5206\u6790\u547d\u4ee4\u539f\u751f\u65e5\u5fd7\u53cd\u9988\u4e2d"
        else:
            content = "\u5206\u6790\u8fc7\u6ee4\u540e\u7ec8\u7aef\u8f93\u51fa\u4e2d"

        self.emit_status(
            "running",
            content,
            {"terminal_source": terminal_source},
        )

    def _emit_waiting_terminal_status(self, tool_name: str | None = None):
        extra = {"terminal_source": TERMINAL_SOURCE_RAW_FEEDBACK}
        if tool_name:
            extra["tool_name"] = tool_name
        self.emit_status(
            "waiting_terminal",
            "\u547d\u4ee4\u5df2\u53d1\u9001\uff0c\u8bfb\u53d6\u539f\u751f\u65e5\u5fd7\u53cd\u9988\u4e2d",
            extra,
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join((value or "").strip().split()).lower()

    @staticmethod
    def _strip_timestamp_prefix(line: str) -> str:
        return re.sub(r"^\[[^\]]+\]\s*", "", line or "").strip()

    def _extract_command_text(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        for key in ("command", "cmd", "text"):
            value = tool_args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return f"{tool_name}: {json.dumps(tool_args, ensure_ascii=False, sort_keys=True)}"

    def _short_command(self, command: str, *, max_length: int = 120) -> str:
        compact = " ".join((command or "").strip().split())
        if len(compact) <= max_length:
            return compact
        return compact[: max_length - 1] + "…"

    def _set_terminal_input_context(self, command: str, input_mode: str | None) -> None:
        if input_mode != TERMINAL_INPUT_MODE_CONSOLE:
            return
        context = TerminalInputContext(
            command=command,
            normalized_command=self._normalize_text(command),
            input_mode=input_mode,
        )
        with self.lock:
            self._terminal_input_context = context

    def _clear_terminal_input_context_for_command(self, command: str) -> None:
        normalized_command = self._normalize_text(command)
        with self.lock:
            context = self._terminal_input_context
            if context and context.normalized_command == normalized_command:
                self._terminal_input_context = None

    def _get_terminal_input_context(self) -> TerminalInputContext | None:
        with self.lock:
            context = self._terminal_input_context
            if not context:
                return None
            age_seconds = (datetime.now() - context.last_seen_at).total_seconds()
            if age_seconds > TERMINAL_INPUT_CONTEXT_TTL_SECONDS:
                self._terminal_input_context = None
                return None
            return context

    def _get_running_terminal_job(self) -> RunningTerminalJob | None:
        with self.lock:
            return self._running_terminal_job

    def has_running_terminal_job(self) -> bool:
        return self._get_running_terminal_job() is not None

    def has_interactive_terminal_context(self) -> bool:
        context = self._get_terminal_input_context()
        pending = self._get_pending_command()
        return bool(
            (context and context.input_mode == TERMINAL_INPUT_MODE_CONSOLE)
            or (pending and pending.input_mode == TERMINAL_INPUT_MODE_CONSOLE)
        )

    def mark_terminal_job_started(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> None:
        command = self._extract_command_text(tool_name, tool_args)
        timeout_seconds = _coerce_timeout_seconds(
            tool_args.get("timeout_seconds"),
            default=600,
        )
        reply_ticket_id = str(tool_args.get("_reply_ticket_id") or "").strip()
        source_type = ""
        source_label = ""
        if reply_ticket_id:
            try:
                from app.services.agent.reply_ticket import reply_ticket_manager

                ticket = reply_ticket_manager.get(reply_ticket_id)
                if ticket:
                    source_type = str(ticket.source_type or "")
                    source_label = str(ticket.source_label or "")
            except Exception:
                source_type = ""
                source_label = ""
        with self.lock:
            self._running_terminal_job = RunningTerminalJob(
                tool_name=tool_name,
                command=command,
                normalized_command=self._normalize_text(command),
                timeout_seconds=timeout_seconds,
                reply_ticket_id=reply_ticket_id,
                source_type=source_type,
                source_label=source_label,
            )

    def clear_terminal_job(self, command: str | None = None) -> None:
        normalized_command = self._normalize_text(command or "")
        with self.lock:
            running_job = self._running_terminal_job
            if not running_job:
                return
            if normalized_command and running_job.normalized_command != normalized_command:
                return
            self._running_terminal_job = None

    def get_recent_finished_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self._recent_finished_jobs)

    def record_finished_job(
        self,
        command: str,
        *,
        job_id: str = "",
        success: bool = False,
        exit_code: int | None = None,
        output_tail: str = "",
    ) -> None:
        output_tail = str(output_tail or "")[-150:]
        with self.lock:
            self._recent_finished_jobs.append({
                "command": self._short_command(command, max_length=120),
                "job_id": job_id,
                "success": success,
                "exit_code": exit_code,
                "output_tail": output_tail,
                "finished_at": datetime.now(),
            })
            if len(self._recent_finished_jobs) > 3:
                self._recent_finished_jobs = self._recent_finished_jobs[-3:]
        try:
            from app.services.filters.env_facts import sediment_job_result

            summary = sediment_job_result(
                self.item_id,
                command,
                success=success,
                exit_code=exit_code,
                output_tail=output_tail,
                job_id=job_id,
            )
            verified = summary.get("verified") or summary.get("recovered")
            if verified:
                with self.lock:
                    self._verified_tools[verified] = "verified"
        except Exception:
            logger.debug("[AgentSession] env-facts sediment skipped", exc_info=True)

    def _find_duplicate_background_job(self, command: str) -> dict[str, Any] | None:
        normalized_command = self._normalize_text(command or "")
        if not normalized_command:
            return None

        running_job = self._get_running_terminal_job()
        if running_job and running_job.normalized_command == normalized_command:
            return {
                "command": running_job.command,
                "elapsed_seconds": (
                    datetime.now() - running_job.started_at
                ).total_seconds(),
                "source": "session",
            }

        for job in self._get_daemon_jobs_snapshot():
            job_command = str(job.get("command") or "")
            if self._normalize_text(job_command) == normalized_command:
                return dict(job)

        return None

    def _build_duplicate_background_job_warning(
        self,
        duplicate_job: dict[str, Any],
    ) -> str:
        elapsed_raw = duplicate_job.get("elapsed_seconds") or 0
        try:
            elapsed_seconds = int(float(elapsed_raw))
        except (TypeError, ValueError):
            elapsed_seconds = 0
        job_id = str(duplicate_job.get("job_id") or "").strip()
        job_suffix = f" job_id={job_id}" if job_id else ""
        return (
            "\u540e\u53f0\u4efb\u52a1\u5df2\u5728\u8fd0\u884c"
            f"{job_suffix}\uff0c\u5df2\u8fd0\u884c {elapsed_seconds}s\u3002"
            "\u76f8\u540c\u547d\u4ee4\u5df2\u62e6\u622a\uff0c\u4e0d\u4f1a\u91cd\u590d\u53d1\u9001\u3002"
            "\u5982\u679c\u8981\u542f\u52a8\u53e6\u4e00\u4e2a\u4e0d\u540c\u7684\u540e\u53f0\u4efb\u52a1\uff0c"
            "\u8bf7\u53d1\u9001\u4e0d\u540c\u7684\u547d\u4ee4\u3002"
        )

    def _validate_terminal_command_input(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str | None:
        if (
            tool_name == EXECUTE_COMMAND_TOOL_NAME
            and self.has_interactive_terminal_context()
        ):
            with self.lock:
                active_context = self._terminal_input_context
                if active_context:
                    active_context.last_seen_at = datetime.now()

        return None

    def validate_terminal_tool_input(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str | None:
        return self._validate_terminal_command_input(tool_name, tool_args)

    def mark_terminal_command_dispatched(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> None:
        self._set_pending_command(tool_name, tool_args)
        self._emit_waiting_terminal_status(tool_name)

    def _set_pending_command(self, tool_name: str, tool_args: dict[str, Any]):
        command = self._extract_command_text(tool_name, tool_args)
        input_mode = classify_terminal_input_mode(command)
        if tool_name == EXECUTE_COMMAND_TOOL_NAME:
            context = self._get_terminal_input_context()
            current_pending = self._get_pending_command()
            if (
                context and context.input_mode == TERMINAL_INPUT_MODE_CONSOLE
            ) or (
                current_pending
                and current_pending.input_mode == TERMINAL_INPUT_MODE_CONSOLE
            ):
                input_mode = TERMINAL_INPUT_MODE_CONSOLE
        timeout_seconds = _coerce_timeout_seconds(
            tool_args.get("timeout_seconds"),
            PENDING_COMMAND_TIMEOUT_SECONDS,
        )
        auto_interrupt_on_timeout = _coerce_bool(
            tool_args.get("auto_interrupt_on_timeout"),
            False,
        )
        pending = PendingCommand(
            tool_name=tool_name,
            command=command,
            normalized_command=self._normalize_text(command),
            integration_contexts=self._capture_tool_integration_contexts(tool_args),
            log_line_cursor=self._get_log_line_count(),
            input_mode=input_mode,
            timeout_seconds=timeout_seconds,
            auto_interrupt_on_timeout=auto_interrupt_on_timeout,
            reply_ticket_id=str(tool_args.get("_reply_ticket_id") or "").strip(),
        )
        self._set_terminal_input_context(command, input_mode)
        terminal_command_state_manager.record(
            self.item_id,
            command,
            source="agent",
            timeout_seconds=timeout_seconds,
        )
        with self.lock:
            self._pending_command = pending
        self._schedule_pending_command_recheck()

    @staticmethod
    def _parse_terminal_relay_command(command: str) -> tuple[str, str] | None:
        match = MINECRAFT_RELAY_COMMAND_RE.match(str(command or "").strip())
        if not match:
            return None
        player = str(match.group("player") or "").strip()
        question = str(match.group("question") or "").strip()
        if not player or not question:
            return None
        return player, question

    def _clear_pending_command(self):
        self._cancel_pending_command_recheck()
        with self.lock:
            self._pending_command = None

    def _get_pending_command(self) -> PendingCommand | None:
        with self.lock:
            return self._pending_command

    def has_pending_command(self) -> bool:
        return self._get_pending_command() is not None

    def _capture_tool_integration_contexts(
        self,
        tool_args: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        context_token = str(tool_args.get("_robot_context_token") or "").strip()
        if not context_token:
            return {}

        try:
            from app.services.agent.integrations.hooks import (
                get_integration_context_by_token,
            )

            context = get_integration_context_by_token(context_token)
            reply_target = getattr(context, "reply_target", None) if context else None
            robot_id = str(getattr(context, "robot_id", "") or "").strip()
            sender_key = str(getattr(context, "sender_key", "") or "").strip()
            if not robot_id or not sender_key or reply_target is None:
                return {}
            if hasattr(reply_target, "model_copy"):
                reply_target = reply_target.model_copy(deep=True)
            return {
                "robot": {
                    "robot_id": robot_id,
                    "sender_key": sender_key,
                    "reply_target": reply_target,
                    "conversation_key": str(
                        getattr(context, "conversation_key", "") or ""
                    ),
                    "conversation_generation": int(
                        getattr(context, "conversation_generation", 0) or 0
                    ),
                    # A terminal command result is a completion for an already accepted
                    # QQ request. It must be delivered even if the chat window slept
                    # before the shell produced output.
                    "reply_requires_awake": False,
                }
            }
        except Exception as exc:
            logger.warning(
                "[AgentSession] Failed to capture integration context for pending command item=%s: %s",
                self.item_id,
                exc,
            )
            return {}

    def _copy_pending_integration_contexts(
        self,
        pending: PendingCommand | None,
    ) -> dict[str, dict[str, Any]]:
        if not pending or not pending.integration_contexts:
            return {}
        contexts: dict[str, dict[str, Any]] = {}
        for name, context in pending.integration_contexts.items():
            copied = dict(context)
            reply_target = copied.get("reply_target")
            if hasattr(reply_target, "model_copy"):
                copied["reply_target"] = reply_target.model_copy(deep=True)
            contexts[name] = copied
        return contexts

    def _build_pending_source_context_prompt(
        self,
        pending: PendingCommand | None,
    ) -> str:
        contexts = self._copy_pending_integration_contexts(pending)
        if not pending or "robot" not in contexts:
            return ""
        robot_context = contexts["robot"]
        conversation_key = str(robot_context.get("conversation_key") or "").strip()
        sender_key = str(robot_context.get("sender_key") or "").strip()
        return (
            "Source routing for this terminal feedback:\n"
            "- The pending terminal command was started by a QQ robot conversation.\n"
            f"- Pending command: {pending.command}\n"
            f"- QQ conversation: {conversation_key or 'current locked conversation'}\n"
            f"- QQ sender_key: {sender_key or 'unknown'}\n"
            "- Use the terminal output only as evidence for that pending QQ request.\n"
            "- If you answer or update the user, call `mcp_robot_send_message` so the reply goes back to the locked QQ conversation. Do not leave the answer only in TermPaws.\n"
            "- Treat this pending QQ request as one-shot: send at most one concise QQ status/final message for this command result. If you already sent a QQ message in this turn, do not restate the same conclusion in the final assistant text.\n"
        )

    def _get_reply_ticket(self, ticket_id: str):
        ticket_id = str(ticket_id or "").strip()
        if not ticket_id:
            return None
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            return reply_ticket_manager.get(ticket_id)
        except Exception:
            return None

    @staticmethod
    def _record_scheduled_ticket_result(
        ticket_id: str,
        *,
        success: bool,
        error: str = "",
    ) -> None:
        if not ticket_id:
            return
        try:
            from app.services.agent.scheduled_tasks import (
                record_scheduled_ticket_result,
            )

            record_scheduled_ticket_result(
                ticket_id,
                success=success,
                error=error,
            )
        except Exception:
            logger.exception(
                "[AgentSession] Failed to update scheduled task result: ticket=%s",
                ticket_id,
            )

    def _build_reply_ticket_prompt(self, ticket_id: str) -> str:
        ticket_id = str(ticket_id or "").strip()
        if not ticket_id:
            return ""
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            return reply_ticket_manager.build_prompt(ticket_id)
        except Exception:
            return ""

    def _deliver_terminal_reply_ticket(
        self,
        ticket_id: str,
        content: str,
    ) -> bool:
        ticket = self._get_reply_ticket(ticket_id)
        if not ticket or not content.strip():
            return False
        if is_terminal_no_reply_intent(content):
            return True
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            if ticket.source_type == "qq":
                if ticket.external_report_sent:
                    return reply_ticket_manager.mark_delivered(ticket_id)
                from app.services.agent.integrations.hooks import (
                    deliver_integration_ticket,
                    integration_conversation_memory_append,
                    sanitize_integration_visible_text,
                )
                from app.services.agent.reply_ticket import ReplyTicketManager

                text = sanitize_integration_visible_text(content).strip()
                if not text:
                    return False
                if deliver_integration_ticket(ticket, text):
                    integration_conversation_memory_append(ticket, text)
                    self.emit_output(
                        f"已回复 QQ：{text}",
                        ROBOT_QQ_REPLY_EVENT_TYPE,
                        {"tool_name": "reply_ticket", "qq_delivery": True},
                    )
                    reply_ticket_manager.mark_delivered(ticket_id)
                    return True
        except Exception:
            logger.exception(
                "[AgentSession] Failed to deliver terminal reply ticket: item=%s ticket=%s",
                self.item_id,
                ticket_id,
            )
        return False

    def _fail_and_report_pending_reply(
        self,
        ticket_id: str,
        *,
        report: str,
        reason: str,
        notify: bool = True,
    ) -> bool:
        ticket = self._get_reply_ticket(ticket_id)
        if not ticket:
            return False
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            reply_ticket_manager.mark_failed(ticket_id, reason or report)
            if not notify:
                # Timeout/turn-guard cancels fail the ticket quietly: the task
                # state survives for later turns, no scary message to the user.
                return False
            # 对用户可见的终止汇报意味着任务真的结束了：计划一并取消，
            # 防止 resume sweep 复活已宣告失败的任务。
            reply_ticket_manager.cancel_plan(ticket_id, explanation="stopped")
            deliver_report = report
            if ticket.source_type == "qq":
                # QQ 侧只给类别不给原始异常（litellm 类名/报文位置等内部
                # 细节对群友毫无意义）；完整原因在日志和票据上。
                friendly = str(report or "").strip()
                if "：" in friendly:
                    friendly = friendly.split("：", 1)[0]
                deliver_report = f"{friendly}，稍后再试一次应该就好" if friendly else "任务中断了，稍后再试"
            delivered = reply_ticket_manager.deliver(ticket_id, deliver_report)
            if delivered and ticket.source_type == "qq":
                self.emit_output(
                    f"已回复 QQ：{report.strip()}",
                    ROBOT_QQ_REPLY_EVENT_TYPE,
                    {"tool_name": "reply_ticket", "qq_delivery": True},
                )
            elif not delivered:
                self.emit_output(
                    "任务失败汇报尚未送达。",
                    "agent_warning",
                )
            return delivered
        except Exception:
            logger.exception(
                "[AgentSession] Failed to report pending task failure: item=%s ticket=%s",
                self.item_id,
                ticket_id,
            )
            return False

    def _attach_reply_ticket_to_agent(self, agent: Agent, ticket_id: str) -> None:
        ticket_id = str(ticket_id or "").strip()
        if not ticket_id:
            return
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            reply_ticket_manager.attach_to_agent(agent, ticket_id)
        except Exception:
            context = getattr(agent, "_context", None)
            if context is not None:
                context.reply_ticket_id = ticket_id

    def _detach_reply_ticket_from_agent(self, agent: Agent, ticket_id: str) -> None:
        ticket_id = str(ticket_id or "").strip()
        if not ticket_id:
            return
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            reply_ticket_manager.detach_from_agent(agent, ticket_id)
        except Exception:
            context = getattr(agent, "_context", None)
            if (
                context is not None
                and str(getattr(context, "reply_ticket_id", "") or "") == ticket_id
            ):
                context.reply_ticket_id = ""

    def _robot_send_blocked_by_reply_ticket(
        self, ticket_id: str, tool_name: str, tool_args: dict | None = None
    ) -> str:
        if tool_name != ROBOT_SEND_TOOL_NAME:
            return ""
        ticket = self._get_reply_ticket(ticket_id)
        if not ticket:
            return ""
        if ticket.source_type == "qq":
            return ""
        # Non-QQ turns have no "current QQ conversation"; the send tool must
        # carry an explicit destination (target/targets/reply_to). The agent
        # decides who to send to — this only rejects destination-less calls.
        from app.services.agent.reply_ticket import robot_send_has_explicit_destination

        if robot_send_has_explicit_destination(tool_args or {}):
            return ""
        return (
            "\u5df2\u62e6\u622a QQ \u53d1\u9001\uff1a"
            "\u5f53\u524d\u540e\u53f0\u4efb\u52a1\u7684\u53d1\u8d77\u6765\u6e90"
            f"\u662f {ticket.source_label or ticket.source_type}\uff0c"
            "\u53ea\u80fd\u56de\u5230\u539f\u59cb\u6765\u6e90\uff0c\u4e0d\u80fd\u8f6c\u53d1\u5230 QQ\u3002"
        )

    def _send_pending_integration_response(
        self,
        pending: PendingCommand | None,
        content: str,
        *,
        message_sent: bool = False,
    ) -> bool:
        contexts = self._copy_pending_integration_contexts(pending)
        if not contexts or not content.strip():
            return False
        if is_terminal_no_reply_intent(content):
            if pending:
                self._mark_pending_integration_response_sent(pending)
            return True
        if pending and pending.integration_response_sent:
            return False
        if pending and message_sent:
            self._mark_pending_integration_response_sent(pending)
            return True
        try:
            delivered = send_integration_final_response_fallback(
                contexts,
                content=content,
                message_sent=message_sent,
            )
            if delivered and pending:
                self._mark_pending_integration_response_sent(pending)
            return delivered
        except Exception as exc:
            logger.warning(
                "[AgentSession] Failed to send pending integration response item=%s: %s",
                self.item_id,
                exc,
            )
            return False

    def _mark_pending_integration_response_sent(self, pending: PendingCommand) -> None:
        pending.integration_response_sent = True
        with self.lock:
            current_pending = self._pending_command
            if (
                current_pending
                and current_pending.normalized_command == pending.normalized_command
            ):
                current_pending.integration_response_sent = True

    def _cancel_pending_command_recheck(self):
        with self.lock:
            timer = self._pending_command_recheck_timer
            self._pending_command_recheck_timer = None

        if timer:
            timer.cancel()

    def _schedule_pending_command_recheck(
        self,
        delay_seconds: float = PENDING_COMMAND_RECHECK_SECONDS,
    ):
        pending = self._get_pending_command()
        if not pending or self._abort_flag:
            return

        with self.lock:
            if self._pending_command_recheck_timer:
                self._pending_command_recheck_timer.cancel()

            timer = threading.Timer(
                delay_seconds,
                self._run_pending_command_recheck,
                args=(pending.normalized_command,),
            )
            timer.daemon = True
            self._pending_command_recheck_timer = timer

        timer.start()

    @staticmethod
    def _background_job_redirect_hint() -> str:
        return (
            "规则：凡不需要后续交互输入、能无人值守跑完的命令"
            "（sleep、下载、安装、编译、批处理查询等）一律改用 "
            "mcp_local_run_job 后台执行——结果会自动回调，不占主终端也不会被超时打断；"
            "前台 execute_command 只用于需要持续交互的进程或几秒内出结果的短命令。"
            "如果这条命令属于前者，请现在改用 mcp_local_run_job 重试。"
        )

    def _build_missing_command_feedback(self, pending: PendingCommand) -> str:
        return (
            f"命令 `{pending.command}` 已发送，但在 "
            f"{pending.timeout_seconds} 秒内没有读取到新的原生日志反馈。"
            + self._background_job_redirect_hint()
        )


    def _send_interrupt_for_pending_timeout(self, pending: PendingCommand) -> bool:
        try:
            from app.services.socket_pool import InputSDK

            success = InputSDK().send(self.item_id, "\x03")
            logger.info(
                "[AgentSession] Auto interrupt sent for timed-out expected command item=%s, command=%s, success=%s",
                self.item_id,
                pending.command,
                success,
            )
            return bool(success)
        except Exception as exc:
            logger.warning(
                "[AgentSession] Failed to auto interrupt timed-out expected command item=%s, command=%s, error=%s",
                self.item_id,
                pending.command,
                exc,
            )
            return False

    def _build_expected_command_timeout_feedback(
        self,
        pending: PendingCommand,
        interrupted: bool,
    ) -> str:
        if pending.auto_interrupt_on_timeout:
            interrupt_text = (
                "已按显式配置发送 Ctrl+C。"
                if interrupted
                else "按显式配置尝试发送 Ctrl+C，但发送失败。"
            )
        else:
            interrupt_text = "未中断当前进程，请结合上一条命令和最新终端输出排查。"
        return (
            f"命令 `{pending.command}` 在 {pending.timeout_seconds} 秒内未完成。"
            f"{interrupt_text}"
            + self._background_job_redirect_hint()
        )

    def _run_pending_command_recheck(self, expected_command: str):
        with self.lock:
            self._pending_command_recheck_timer = None

        pending = self._get_pending_command()
        if (
            not pending
            or pending.normalized_command != expected_command
            or self._abort_flag
        ):
            return

        if self.is_running_turn() or not self.input_queue.empty():
            self._schedule_pending_command_recheck()
            return

        elapsed_seconds = (datetime.now() - pending.dispatched_at).total_seconds()
        current_log_line_count = self._get_log_line_count()
        has_new_log_lines = current_log_line_count > pending.log_line_cursor
        should_force_tail_check = (
            pending.echo_count > 0
            and elapsed_seconds >= PENDING_COMMAND_STALLED_CONFIRM_SECONDS
        )

        if not has_new_log_lines and not should_force_tail_check:
            if elapsed_seconds >= pending.timeout_seconds:
                logger.warning(
                    "[AgentSession] Pending command timed out without raw feedback for item=%s, command=%s",
                    self.item_id,
                    pending.command,
                )
                self._clear_pending_command()
                timeout_message = self._build_missing_command_feedback(pending)
                self._send_pending_integration_response(pending, timeout_message)
                self.emit_output(
                    timeout_message,
                    "agent_warning",
                    {"tool_name": pending.tool_name},
                )
                return

            self._schedule_pending_command_recheck()
            return

        logger.info(
            "[AgentSession] Triggering pending command recheck for item=%s, command=%s, new_lines=%s, echo_count=%s",
            self.item_id,
            pending.command,
            has_new_log_lines,
            pending.echo_count,
        )
        # Read actual log tail so the agent sees real terminal output
        recheck_content = self._consume_pending_log_delta()
        if not recheck_content:
            recheck_content = self._get_recent_pending_log_tail(30)
        self.process_input(
            InputMessage(
                input_type=InputType.TERMINAL,
                content=recheck_content,
                raw_content=recheck_content,
            )
        )
        if self.has_pending_command():
            self._schedule_pending_command_recheck()

    def _get_log_manager(self):
        from app.services.log_manager import LogManager

        return LogManager()

    def _get_log_line_count(self) -> int:
        try:
            return self._get_log_manager().get_line_count(self.item_id)
        except Exception as exc:
            logger.debug(
                "[AgentSession] Failed to read log line count for item=%s: %s",
                self.item_id,
                exc,
            )
            return 0

    def _consume_pending_log_delta(self) -> str:
        pending = self._get_pending_command()
        if not pending:
            return ""

        try:
            log_manager = self._get_log_manager()
            delta = log_manager.get_lines_since(self.item_id, pending.log_line_cursor) or ""
            new_cursor = log_manager.get_line_count(self.item_id)
        except Exception as exc:
            logger.debug(
                "[AgentSession] Failed to consume pending log delta for item=%s: %s",
                self.item_id,
                exc,
            )
            return ""

        with self.lock:
            current_pending = self._pending_command
            if current_pending and current_pending.normalized_command == pending.normalized_command:
                current_pending.log_line_cursor = new_cursor

        return delta

    def discard_pending_terminal_feedback_delta(self) -> None:
        pending = self._get_pending_command()
        if not pending:
            return
        try:
            new_cursor = self._get_log_line_count()
        except Exception as exc:
            logger.debug(
                "[AgentSession] Failed to advance pending log cursor for item=%s: %s",
                self.item_id,
                exc,
            )
            return
        with self.lock:
            current_pending = self._pending_command
            if current_pending and current_pending.normalized_command == pending.normalized_command:
                current_pending.log_line_cursor = max(
                    current_pending.log_line_cursor,
                    new_cursor,
                )
        self._emit_waiting_terminal_status(pending.tool_name)
    def _get_recent_pending_log_tail(self, lines: int = 8) -> str:
        try:
            return self._get_log_manager().get_last_lines(self.item_id, lines) or ""
        except Exception as exc:
            logger.debug(
                "[AgentSession] Failed to read recent log tail for item=%s: %s",
                self.item_id,
                exc,
            )
            return ""

    def _resolve_terminal_analysis_content(
        self,
        input_msg: InputMessage,
    ) -> TerminalAnalysisResult:
        pending = self._get_pending_command()
        if pending:
            analysis_content = self._consume_pending_log_delta().strip()
            if not analysis_content:
                analysis_content = self._get_recent_pending_log_tail().strip()
            if not analysis_content:
                analysis_content = (input_msg.raw_content or input_msg.content).strip()

            if not analysis_content:
                return TerminalAnalysisResult(
                    content=None,
                    terminal_source=TERMINAL_SOURCE_RAW_FEEDBACK,
                )

            should_hold, resolved_feedback, direct_response = (
                self._maybe_hold_for_pending_terminal_feedback(analysis_content)
            )
            if should_hold:
                return TerminalAnalysisResult(
                    content=None,
                    terminal_source=TERMINAL_SOURCE_RAW_FEEDBACK,
                )
            if resolved_feedback:
                return TerminalAnalysisResult(
                    content=resolved_feedback,
                    terminal_source=TERMINAL_SOURCE_RAW_FEEDBACK,
                    direct_response=direct_response,
                )

            return TerminalAnalysisResult(
                content=analysis_content,
                terminal_source=TERMINAL_SOURCE_RAW_FEEDBACK,
            )

        analysis_content = (input_msg.content or "").strip()
        return TerminalAnalysisResult(
            content=analysis_content or None,
            terminal_source=TERMINAL_SOURCE_FILTERED,
        )

    def _is_prompt_only_line(self, line: str) -> bool:
        normalized_line = self._normalize_text(self._strip_timestamp_prefix(line))
        if not normalized_line:
            return True
        if re.fullmatch(r"[#>$\s]+", normalized_line):
            return True
        if normalized_line.endswith(("#", "$", ">")) and any(
            marker in normalized_line for marker in "#$>"
        ):
            return bool(re.fullmatch(r"[\w\s./:@~()\\\-\[\]#>$]+", normalized_line))
        return False

    def _is_command_echo_line(self, line: str, pending: PendingCommand) -> bool:
        normalized_line = self._normalize_text(self._strip_timestamp_prefix(line))
        if not normalized_line:
            return False
        if normalized_line == pending.normalized_command:
            return True

        stripped_prompt_line = re.sub(r"^(?:[#>$]\s*)+", "", normalized_line).strip()
        if stripped_prompt_line == pending.normalized_command:
            return True

        if normalized_line.endswith(pending.normalized_command):
            prefix = normalized_line[: -len(pending.normalized_command)]
            if any(marker in prefix for marker in "#$>") and re.fullmatch(
                r"[\w\s./:@~()\\\-\[\]#>$]*",
                prefix,
            ):
                return True

        return False

    def _build_empty_command_feedback(self, pending: PendingCommand) -> str:
        return f"命令 `{pending.command}` 已执行完成，终端没有输出内容。"

    def _maybe_resolve_empty_command_feedback(
        self,
        pending: PendingCommand,
        lines: list[str],
        informative_lines: list[str],
    ) -> str | None:
        prompt_only_count = len(lines) - len(informative_lines)
        tail_text = self._get_recent_pending_log_tail()
        tail_lines = [line for line in tail_text.splitlines() if line.strip()]
        tail_informative_lines = [
            line for line in tail_lines if not self._is_prompt_only_line(line)
        ]
        tail_prompt_only_count = len(tail_lines) - len(tail_informative_lines)
        tail_echo_count = sum(
            1 for line in tail_informative_lines if self._is_command_echo_line(line, pending)
        )
        with self.lock:
            current_pending = self._pending_command
            prior_echo_count = (
                current_pending.echo_count
                if current_pending
                and current_pending.normalized_command == pending.normalized_command
                else 0
            )

        # Prompt has returned or the same echo has appeared repeatedly: treat as
        # a completed command with empty stdout/stderr instead of waiting forever.
        if (
            prompt_only_count > 0
            or tail_prompt_only_count > 0
            or len(informative_lines) > 1
            or tail_echo_count > 1
            or prior_echo_count > 0
        ):
            logger.info(
                "[AgentSession] Treating empty command output as completed for item=%s, command=%s",
                self.item_id,
                pending.command,
            )
            self._clear_pending_command()
            return self._build_empty_command_feedback(pending)

        return None

    def _maybe_hold_for_pending_terminal_feedback(
        self,
        content: str,
    ) -> tuple[bool, str | None, bool]:
        pending = self._get_pending_command()
        if not pending:
            return False, None, False

        lines = [line for line in content.splitlines() if line.strip()]
        informative_lines = [line for line in lines if not self._is_prompt_only_line(line)]
        if not informative_lines:
            logger.info(
                "[AgentSession] Prompt returned without command output for item=%s, command=%s",
                self.item_id,
                pending.command,
            )
            self._clear_pending_command()
            return False, self._build_empty_command_feedback(pending), True

        if not all(self._is_command_echo_line(line, pending) for line in informative_lines):
            logger.info(
                "[AgentSession] Received non-echo terminal feedback for item=%s, command=%s",
                self.item_id,
                pending.command,
            )
            if pending.input_mode == TERMINAL_INPUT_MODE_BUSY:
                if len(lines) > len(informative_lines):
                    self._clear_pending_command()
                    return False, None, False
                logger.info(
                    "[AgentSession] Holding busy command feedback until prompt returns for item=%s, command=%s",
                    self.item_id,
                    pending.command,
                )
                self._emit_waiting_terminal_status(pending.tool_name)
                return True, None, False
            if pending.input_mode == TERMINAL_INPUT_MODE_CONSOLE:
                self._set_terminal_input_context(pending.command, pending.input_mode)
            self._clear_pending_command()
            return False, None, False

        resolved_feedback = self._maybe_resolve_empty_command_feedback(
            pending,
            lines,
            informative_lines,
        )
        if resolved_feedback:
            return False, resolved_feedback, True

        warning_text: str | None = None
        with self.lock:
            current_pending = self._pending_command
            if current_pending and current_pending.normalized_command == pending.normalized_command:
                current_pending.echo_count += len(informative_lines)
                if (
                    not current_pending.timeout_warned
                    and (
                        datetime.now() - current_pending.dispatched_at
                    ).total_seconds()
                    > current_pending.timeout_seconds
                ):
                    current_pending.timeout_warned = True
                    warning_text = (
                        f"等待终端反馈超时，当前只收到命令回显：{current_pending.command}"
                    )

        logger.info(
            "[AgentSession] Suppressed echoed command for item=%s, command=%s",
            self.item_id,
            pending.command,
        )
        if warning_text:
            self.emit_output(
                warning_text,
                "agent_warning",
                {"tool_name": pending.tool_name},
            )
        self._emit_waiting_terminal_status(pending.tool_name)
        return True, None, False

    def _should_display_pending_held_output(
        self,
        content: str,
        pending: PendingCommand | None,
    ) -> bool:
        if not pending:
            return False
        lines = [line for line in content.splitlines() if line.strip()]
        informative_lines = [line for line in lines if not self._is_prompt_only_line(line)]
        return bool(informative_lines) and all(
            self._is_command_echo_line(line, pending) for line in informative_lines
        )

    def _emit_idle_or_waiting_status(self, queue_size: int):
        if queue_size > 0:
            self.emit_status("queued", "后续输出排队中", {"queue_size": queue_size})
            return

        pending = self._get_pending_command()
        if pending:
            self._emit_waiting_terminal_status(pending.tool_name)
            return

        self.emit_status("idle", "")

    def mark_collecting(self):
        with self.lock:
            if self.state in {SessionState.RUNNING, SessionState.INTERRUPTING}:
                return
            self.state = SessionState.COLLECTING
        self._emit_collecting_terminal_status()

    def mark_waiting_terminal_feedback(self):
        pending = self._get_pending_command()
        self._emit_waiting_terminal_status(pending.tool_name if pending else None)

    @staticmethod
    def _merge_text_parts(existing: str, incoming: str) -> str:
        left = (existing or "").strip("\n")
        right = (incoming or "").strip("\n")
        if left and right:
            return f"{left}\n{right}"
        return left or right

    def _merge_terminal_queue_input(self, input_msg: InputMessage) -> bool:
        queued_items = getattr(self.input_queue, "queue", None)
        if not queued_items:
            return False

        last_item = queued_items[-1]
        if not isinstance(last_item, InputMessage):
            return False
        if last_item.input_type != InputType.TERMINAL:
            return False
        if str(last_item.reply_ticket_id or "") != str(input_msg.reply_ticket_id or ""):
            return False

        last_item.content = self._merge_text_parts(last_item.content, input_msg.content)
        last_item.raw_content = self._merge_text_parts(
            last_item.raw_content,
            input_msg.raw_content,
        )
        if input_msg.query:
            last_item.query = self._merge_text_parts(last_item.query, input_msg.query)
        if input_msg.callback:
            last_item.callback = input_msg.callback
        last_item.timestamp = input_msg.timestamp
        return True

    def queue_input(self, input_msg: InputMessage) -> bool:
        with self.lock:
            try:
                merged = False
                if input_msg.input_type == InputType.TERMINAL:
                    merged = self._merge_terminal_queue_input(input_msg)

                if not merged:
                    self.input_queue.put_nowait(input_msg)
                queue_size = self.input_queue.qsize()
                logger.info(
                    f"[AgentSession] {'Merged' if merged else 'Queued'} "
                    f"{input_msg.input_type.value} input for item {self.item_id}, "
                    f"queue size: {queue_size}"
                )
            except queue.Full:
                logger.warning(f"[AgentSession] Input queue full for item {self.item_id}")
                self.emit_output("输入队列已满，后续输入已丢弃", "agent_warning")
                return False

        if input_msg.input_type == InputType.TERMINAL:
            terminal_source = (
                TERMINAL_SOURCE_RAW_FEEDBACK
                if self.has_pending_command() and (input_msg.raw_content or "").strip()
                else TERMINAL_SOURCE_FILTERED
            )
            self.emit_status(
                "queued",
                "后续命令反馈排队中"
                if terminal_source == TERMINAL_SOURCE_RAW_FEEDBACK
                else "后续过滤终端输出排队中",
                {
                    "queue_size": self.input_queue.qsize(),
                    "terminal_source": terminal_source,
                },
            )
        else:
            self.emit_status("queued", "消息已排队，等待当前轮结束", {"queue_size": self.input_queue.qsize()})

        return True

    def clear_queued_inputs(self) -> int:
        cleared = 0
        while True:
            try:
                self.input_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        return cleared

    def _begin_turn(self, input_msg: InputMessage):
        with self.lock:
            self.state = SessionState.RUNNING
            self._abort_flag = False
            self._current_turn_id = uuid4().hex[:8]

        if input_msg.input_type == InputType.TERMINAL:
            terminal_source = (
                TERMINAL_SOURCE_RAW_FEEDBACK
                if self.has_pending_command()
                else TERMINAL_SOURCE_FILTERED
            )
            self._emit_running_terminal_status(terminal_source)
        else:
            self.emit_status("running", "回复中")

    def _finish_turn(self):
        with self.lock:
            queue_size = self.input_queue.qsize()
            self.state = SessionState.IDLE if queue_size == 0 else SessionState.COOLDOWN
            self._current_turn_id = None
            was_aborted = getattr(self, "_aborted_this_turn", False)
            self._aborted_this_turn = False
        self._emit_idle_or_waiting_status(queue_size)
        if was_aborted:
            return
        try:
            from app.services.agent.mcp.local_server import (
                flush_all_job_results_for_item,
            )

            flush_all_job_results_for_item(self.item_id)
        except Exception:
            pass

    def _consumer_loop(self):
        """Single consumer: wait for wake signal, drain queue, process one turn, repeat."""
        while not self._consumer_shutdown:
            try:
                self._wake_event.wait(timeout=1.0)
                self._wake_event.clear()

                if self._consumer_shutdown:
                    break

                batch: list[InputMessage] = []
                try:
                    first = self.input_queue.get(timeout=0.1)
                    batch.append(first)
                except queue.Empty:
                    continue
                while True:
                    try:
                        batch.append(self.input_queue.get(timeout=0.3))
                    except queue.Empty:
                        break

                # 不按时间丢弃任何输入：队列上限（MAX_QUEUE_SIZE=20）就是唯一的
                # 背压阀。终端刷屏由 _merge_terminal_queue_input 在入队时合并兜底，
                # 用户消息无论排多久都必须被处理。
                if not batch:
                    with self.lock:
                        if self.input_queue.empty():
                            self.state = SessionState.IDLE
                            self._emit_idle_or_waiting_status(0)
                    continue

                # 过期的 plan-resume 输入直接丢弃：计划已被先前轮次跑完，
                # 再执行只会让 agent 重新造计划、把已完成的任务重做一遍。
                batch = [msg for msg in batch if not self._is_stale_plan_resume(msg)]
                if not batch:
                    continue

                # 不同来源路由的输入不合并：web/群A/群B 各有各的回复路由，
                # 揉成一轮会让模型只有一个"主票据"，把别的来源的结论发错
                # 会话。按路由分组，同一路由（同一会话的连发）仍然合并。
                for group in self._group_batch_by_route(batch):
                    self._run_batch(group)
            except Exception:
                # 循环体里任何一处（分组/_begin_turn/状态收尾）抛异常都不能
                # 弄死消费者线程——它一死，队列里的消息就永远没人处理。
                logger.exception("[AgentSession] consumer iteration crashed; keeping loop alive")
                with self.lock:
                    if self.state in {SessionState.RUNNING, SessionState.INTERRUPTING}:
                        self.state = SessionState.IDLE

        with self.lock:
            self.state = SessionState.IDLE

    @staticmethod
    def _turn_group_key(msg: "InputMessage") -> tuple[str, str]:
        """轮次分组键：纯路由信息，不看内容。同键输入才允许合并成一轮。"""
        if msg.input_type != InputType.CHAT:
            return ("other", str(msg.input_type))
        label = str(msg.source_label or "")
        if label.startswith("plan-resume "):
            return ("chat", label[len("plan-resume "):])
        if label.startswith("QQ "):
            return ("chat", label[3:])
        if label.startswith("web"):
            return ("chat", "web")
        return ("chat", label)

    def _group_batch_by_route(self, batch: list["InputMessage"]) -> list[list["InputMessage"]]:
        groups: dict[tuple[str, str], list[InputMessage]] = {}
        order: list[tuple[str, str]] = []
        for msg in batch:
            key = self._turn_group_key(msg)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(msg)
        return [groups[key] for key in order]

    def _run_batch(self, batch: list[InputMessage]) -> None:
        merged = batch[0] if len(batch) == 1 else self._merge_input_batch(batch)

        self._begin_turn(merged)
        process_error = ""
        try:
            self._process_input(merged)
        except Exception as exc:
            process_error = str(exc)
            logger.error(f"[AgentSession] Error processing queue: {exc}")
            self.emit_output(f"处理失败: {exc}", "agent_error")
        finally:
            self._finish_turn()
            for msg in batch:
                self._notify_input_complete(msg, not process_error, process_error)

    def ensure_consumer_alive(self) -> bool:
        """队列非空但消费者停摆时的自愈：线程死了就重启，活着就补 wake。

        返回是否进行了干预。由 watchdog 每轮巡检调用，保证「队列有货 +
        会话空闲」一定能启动下一轮。
        """
        if self._consumer_shutdown or self.input_queue.empty():
            return False
        with self.lock:
            alive = self._consumer_thread.is_alive()
            if not alive and not self._consumer_shutdown:
                self._consumer_thread = threading.Thread(
                    target=self._consumer_loop,
                    daemon=True,
                    name=f"agent-consumer-{self.item_id[:8]}",
                )
                self._consumer_thread.start()
                logger.warning(
                    "[AgentSession] consumer thread was dead with %d queued input(s); restarted for item=%s",
                    self.input_queue.qsize(),
                    self.item_id,
                )
                return True
        self._wake_event.set()
        return True

    def _is_stale_plan_resume(self, msg: InputMessage) -> bool:
        if not str(getattr(msg, "source_label", "") or "").startswith("plan-resume"):
            return False
        from app.services.agent.reply_ticket import (
            plan_has_open_steps,
            reply_ticket_manager,
        )

        ticket = reply_ticket_manager.get(str(getattr(msg, "reply_ticket_id", "") or ""))
        return ticket is None or not plan_has_open_steps(ticket.plan)

    def process_queue(self):
        """Legacy compat: wake the consumer."""
        self._wake_event.set()

    def _merge_input_batch(self, batch: list[InputMessage]) -> InputMessage:
        """Merge multiple queued inputs into a single InputMessage."""
        source_labels = {
            InputType.TERMINAL: "终端输出",
            InputType.CHAT: "聊天消息",
            InputType.SCHEDULED_TASK: "定时任务",
        }
        sections: list[str] = []
        merged_ticket_ids: list[str] = []
        merged_image_urls: list[str] = []
        merged_force_vision = False
        for msg in batch:
            for url in msg.image_urls:
                if url not in merged_image_urls:
                    merged_image_urls.append(url)
            merged_force_vision = merged_force_vision or msg.force_image_vision
            label = msg.source_label or source_labels.get(
                msg.input_type, msg.input_type.value
            )
            body = (msg.content or msg.raw_content or "").strip()
            if not body:
                continue
            sections.append(f"[{label}]\n" + body)
            if msg.reply_ticket_id and msg.reply_ticket_id not in merged_ticket_ids:
                merged_ticket_ids.append(msg.reply_ticket_id)

        # 合并批次里有 ≥2 条真实用户消息时，它们全是未回复的新消息，
        # 但 QQ 会话提示词默认"只有一条唤醒消息"——模型会只回最后一条，
        # 前面 @它的提问被静默跳过（"有些回复没回复"的根因）。在合并内容
        # 头部显式声明，让模型逐条回应。
        user_chat_count = sum(
            1
            for msg in batch
            if msg.input_type == InputType.CHAT
            and not str(msg.source_label or "").startswith("plan-resume")
        )
        if user_chat_count >= 2:
            sections.insert(
                0,
                "[系统提示]\n"
                f"本轮合并了 {user_chat_count} 条尚未回复的用户消息（见下方各分段）。"
                "它们都是需要回应的新消息，不是已处理的历史——每条的内容都要覆盖到；"
                "同一个意思的（哪怕来自不同人）融合成一句带过，不同的题目才分开简短作答。",
            )

        combined_content = "\n---\n".join(sections) if sections else ""

        # Pick primary reply_ticket: the most recent input carrying one.
        primary_ticket = ""
        for msg in reversed(batch):
            if msg.reply_ticket_id:
                primary_ticket = msg.reply_ticket_id
                break

        # Use the last callback (most recent sender expects a reply)
        last_callback = None
        for msg in reversed(batch):
            if msg.callback:
                last_callback = msg.callback
                break

        return InputMessage(
            input_type=InputType.TERMINAL,
            content=combined_content,
            raw_content=combined_content,
            query=f"batch:{len(batch)} inputs merged",
            reply_ticket_id=primary_ticket,
            callback=last_callback,
            merged_ticket_ids=tuple(merged_ticket_ids),
            image_urls=merged_image_urls,
            force_image_vision=merged_force_vision,
        )

    def process_input(self, input_msg: InputMessage):
        self.queue_input(input_msg)
        self._wake_event.set()

    @staticmethod
    def _notify_input_complete(
        input_msg: InputMessage,
        processed: bool,
        process_error: str = "",
    ) -> None:
        callback = input_msg.completion_callback
        if callback is None:
            return
        try:
            callback(processed, process_error)
        except Exception:
            logger.exception("[AgentSession] Input completion callback failed")

    def _process_input(self, input_msg: InputMessage):
        with agent_turn_coordinator.turn(agent_turn_key(self.handler_id)):
            self._process_input_serialized(input_msg)

    def _process_input_serialized(self, input_msg: InputMessage):
        if input_msg.callback:
            self.add_output_callback(input_msg.callback)
        from app.services.agent.stream_manager import stream_manager as _sm
        for _cb in _sm._get_callbacks(self.item_id):
            self.add_output_callback(_cb)
        agent = self.get_agent()
        if not agent:
            self.emit_output("Agent 不可用", "agent_error")
            return

        self._last_activity = datetime.now()

        if input_msg.input_type == InputType.TERMINAL:
            self._attach_reply_ticket_to_agent(agent, input_msg.reply_ticket_id)
            try:
                self._process_terminal_input(input_msg, agent)
            finally:
                self._detach_reply_ticket_from_agent(agent, input_msg.reply_ticket_id)
        elif input_msg.input_type == InputType.SCHEDULED_TASK:
            from app.services.agent.reply_ticket import reply_ticket_manager

            if not input_msg.reply_ticket_id:
                ticket = reply_ticket_manager.create_for_scheduled_task(
                    agent,
                    item_id=self.item_id,
                    handler_id=self.handler_id,
                    message=input_msg.content,
                    scheduled_task_id=input_msg.scheduled_task_id,
                    scheduled_execution_id=input_msg.scheduled_execution_id,
                    request_message=input_msg.request_display,
                )
                input_msg.reply_ticket_id = ticket.ticket_id
            else:
                self._attach_reply_ticket_to_agent(agent, input_msg.reply_ticket_id)
            try:
                self._process_chat_input(input_msg, agent)
            finally:
                self._detach_reply_ticket_from_agent(agent, input_msg.reply_ticket_id)
        else:
            self._process_chat_input(input_msg, agent)

    def _process_terminal_input(self, input_msg: InputMessage, agent: Agent):
        combined_input = "\n".join(
            part
            for part in (input_msg.content, input_msg.raw_content)
            if part and part.strip()
        )
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            if not input_msg.reply_ticket_id:
                pending = self._get_pending_command()
                if pending and pending.reply_ticket_id:
                    input_msg.reply_ticket_id = pending.reply_ticket_id
                    self._attach_reply_ticket_to_agent(agent, input_msg.reply_ticket_id)
            if not input_msg.reply_ticket_id and self._get_pending_command() is None:
                player_chat = MINECRAFT_PLAYER_CHAT_CAPTURE_RE.search(combined_input)
                if player_chat:
                    player = str(player_chat.group("player") or "").strip()
                    message = str(player_chat.group("message") or "").strip()
                    ticket = reply_ticket_manager.create_for_terminal(
                        agent,
                        item_id=self.item_id,
                        handler_id=self.handler_id,
                        player=player,
                        message=message,
                    )
                    input_msg.reply_ticket_id = ticket.ticket_id
        except Exception:
            logger.exception(
                "[AgentSession] Failed to resolve terminal pending-reply source: item=%s",
                self.item_id,
            )

        reply_ticket = self._get_reply_ticket(input_msg.reply_ticket_id)
        reply_ticket_is_web = bool(
            reply_ticket and getattr(reply_ticket, "source_type", "") == "web"
        )
        pending_before_analysis = self._get_pending_command()
        had_pending_command = pending_before_analysis is not None
        pending_integration_contexts = self._copy_pending_integration_contexts(
            pending_before_analysis
        )
        if reply_ticket_is_web or not pending_integration_contexts:
            clear_robot_context = getattr(agent, "clear_robot_context", None)
            if callable(clear_robot_context):
                clear_robot_context()
        analysis = self._resolve_terminal_analysis_content(input_msg)
        if input_msg.content:
            attach_terminal_feedback_to_pending_continuation(self.item_id, input_msg.content)
        should_emit_terminal_output = bool(input_msg.content) and (
            not had_pending_command
            or bool(analysis.content)
            or self._should_display_pending_held_output(
                input_msg.content,
                pending_before_analysis,
            )
        )
        if should_emit_terminal_output:
            self.emit_output(input_msg.content, "terminal_output")
        if not analysis.content:
            return
        if analysis.direct_response:
            integration_already_sent = bool(
                pending_before_analysis
                and pending_before_analysis.integration_response_sent
            )
            integration_delivered = self._send_pending_integration_response(
                pending_before_analysis,
                analysis.content,
            )
            if not (integration_already_sent or integration_delivered):
                self.emit_output(analysis.content, "agent_response")
            self._record_scheduled_ticket_result(
                input_msg.reply_ticket_id,
                success=True,
            )
            return

        transient_integration_tools_added = False
        _loop_mgr = _ManagedEventLoop()
        try:
            loop = _loop_mgr.__enter__()
            if pending_integration_contexts:
                setup_integration_chat_contexts(agent, pending_integration_contexts)
            loop.run_until_complete(agent.start_mcp_servers())
            if pending_integration_contexts:
                loop.run_until_complete(
                    ensure_integration_chat_context_tools(
                        agent,
                        pending_integration_contexts,
                    )
                )
            if (
                is_critical_terminal_event(analysis.content)
                and should_enable_terminal_alert_integrations(agent, analysis.content)
            ):
                transient_integration_tools_added = bool(
                    loop.run_until_complete(
                        ensure_terminal_alert_integration_tools(
                            agent,
                            analysis.content,
                        )
                    )
                )

            messages = self._build_terminal_messages(
                agent,
                input_msg,
                analysis.content,
                terminal_source=analysis.terminal_source,
                pending_command=pending_before_analysis,
            )
            extract_integration_context_targets(agent, messages)
            turn_guard = TurnGuard()
            self._current_turn_id = turn_guard.turn_id
            self._emit_running_terminal_status(analysis.terminal_source)
            terminal_delivery_retry_used = False
            robot_final_text_retry_used = False
            integration_tool_results: list[str] = []
            terminal_failure_report = ""
            terminal_failure_silent = False
            _response_emitted_in_loop = False

            for iteration_index in range(MAX_ITERATIONS):
                from app.services.agent.reply_ticket import reply_ticket_manager as _rtm

                _rtm.touch(input_msg.reply_ticket_id)
                timed_out, timeout_reason = turn_guard.check_timeout()
                if timed_out:
                    terminal_failure_report = "任务未能完成：本轮处理超时，已停止该任务。"
                    terminal_failure_silent = True
                    self._send_pending_integration_response(
                        pending_before_analysis,
                        timeout_reason,
                    )
                    self.emit_output(timeout_reason, "agent_warning")
                    break

                if self._abort_flag:
                    self.emit_output("当前轮已中断", "agent_warning")
                    self.emit_status("interrupted", "当前轮已中断")
                    break

                finalization_only = False
                if iteration_index == MAX_ITERATIONS - 1:
                    finalization_only = True
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "The tool-call budget is finished. Do not call any more "
                                "tools. Use the latest confirmed tool results to give the "
                                "user one concise final answer now."
                            ),
                        }
                    )

                # 每次迭代重算工具表：prepare_capabilities 中途加载的工具
                # 必须本轮可调用，否则模型只能反复 prepare 打转直到守卫停机。
                tools = [] if finalization_only else select_tools_for_turn(
                    agent.get_tools_for_litellm(),
                    source="terminal",
                    query=input_msg.query or analysis.content,
                    agent=agent,
                    reply_ticket_id=input_msg.reply_ticket_id,
                )
                response = self._call_llm(agent, messages, tools=tools)
                message = self._normalize_dsml_tool_message(
                    response.choices[0].message,
                    tools,
                )
                if not (hasattr(message, "tool_calls") and message.tool_calls):
                    raw_content = message.content or ""
                    if not raw_content and hasattr(message, "reasoning_content"):
                        raw_content = ""
                    if raw_content:
                        final_content = strip_think_tags(guard_fabricated_tool_trace(raw_content))
                        reply_ticket = self._get_reply_ticket(
                            input_msg.reply_ticket_id
                        )
                        if (
                            not robot_final_text_retry_used
                            and str(getattr(agent, "robot_context_token", "") or "").strip()
                            and final_content.strip()
                            and not is_terminal_no_reply_intent(final_content)
                        ):
                            # QQ 轮次里唯一出口是发送工具：最终文本不会送达。
                            # 模型把没说的内容写进最终文本时，纠偏让它用工具发完，
                            # 而不是默默丢弃（故事/长内容讲到一半被砍的主因）。
                            messages.append({"role": "assistant", "content": final_content})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "[系统] 你上一条文本不会送达用户——QQ 回复的唯一出口是 "
                                        "mcp_robot_send_message。如果那是你还没说完的内容，"
                                        "现在用发送工具把它发出去（一条气泡一次调用，发完为止）；"
                                        "如果确实已经全部说完了，只返回 [no_qq_reply]。"
                                    ),
                                }
                            )
                            robot_final_text_retry_used = True
                            continue
                        if should_retry_terminal_source_delivery(
                            agent=agent,
                            terminal_content=analysis.content,
                            terminal_source=analysis.terminal_source,
                            final_response=final_content,
                            retry_used=terminal_delivery_retry_used,
                        ):
                            messages.append({"role": "assistant", "content": final_content})
                            messages.append(
                                build_terminal_source_delivery_correction_message(final_content)
                            )
                            terminal_delivery_retry_used = True
                            self._emit_running_terminal_status(analysis.terminal_source)
                            continue
                        if pending_integration_contexts:
                            message_sent = (
                                integration_message_sent(integration_tool_results)
                            )
                            integration_delivered = self._send_pending_integration_response(
                                pending_before_analysis,
                                final_content,
                                message_sent=message_sent,
                            )
                            if message_sent or integration_delivered:
                                break
                        reply_ticket = self._get_reply_ticket(
                            input_msg.reply_ticket_id
                        )
                        deliver_content = append_tool_call_footer(
                            final_content,
                            turn_guard.tool_names,
                        )
                        if (
                            reply_ticket
                            and reply_ticket.source_type == "qq"
                            and self._deliver_terminal_reply_ticket(
                                input_msg.reply_ticket_id,
                                deliver_content,
                            )
                        ):
                            break
                        self.emit_output(deliver_content, "agent_response")
                        _response_emitted_in_loop = True
                        if reply_ticket and reply_ticket.source_type == "web":
                            try:
                                from app.services.agent.reply_ticket import (
                                    reply_ticket_manager,
                                )

                                reply_ticket_manager.mark_delivered(
                                    input_msg.reply_ticket_id
                                )
                            except Exception:
                                logger.exception(
                                    "[AgentSession] Failed to complete web reply ticket: item=%s ticket=%s",
                                    self.item_id,
                                    input_msg.reply_ticket_id,
                                )
                        self._record_scheduled_ticket_result(
                            input_msg.reply_ticket_id,
                            success=True,
                        )
                    else:
                        if turn_guard.tool_call_count > 0 and iteration_index < MAX_ITERATIONS - 1:
                            messages.append(
                                {
                                    "role": "system",
                                    "content": (
                                        "Your previous response had no visible text content. "
                                        "Summarize the tool results for the user in one or two "
                                        "sentences now. Do not call any tools."
                                    ),
                                }
                            )
                            continue
                    break

                next_messages = self._handle_tool_calls(
                    agent,
                    loop,
                    messages,
                    message,
                    turn_guard,
                    terminal_source=analysis.terminal_source,
                    tool_results_sink=integration_tool_results,
                    pending_command_for_delivery=pending_before_analysis,
                    reply_ticket_id=input_msg.reply_ticket_id,
                )
                if next_messages is None:
                    break
                messages = next_messages
            if terminal_failure_report:
                self._fail_and_report_pending_reply(
                    input_msg.reply_ticket_id,
                    report=terminal_failure_report,
                    reason=terminal_failure_report,
                    notify=not terminal_failure_silent,
                )

            if pending_integration_contexts:
                clear_integration_chat_contexts(agent, pending_integration_contexts)
            if transient_integration_tools_added:
                clear_terminal_alert_integration_tools(agent)
            _loop_mgr.__exit__(None, None, None)
        except Exception as exc:
            _loop_mgr.__exit__(*sys.exc_info())
            logger.error(f"[AgentSession] Terminal processing error: {exc}")
            if _is_transient_llm_error(exc) and input_msg.reply_ticket_id:
                logger.info(
                    "[AgentSession] Transient LLM error, keeping workflow alive: item=%s error=%s",
                    self.item_id,
                    exc,
                )
                self.emit_output(
                    f"模型调用临时失败，任务保留，稍后自动重试: {exc}",
                    "agent_warning",
                )
                self._schedule_delayed_retry(input_msg.reply_ticket_id)
            else:
                self.emit_output(f"处理失败: {exc}", "agent_error")
                self._fail_and_report_pending_reply(
                    input_msg.reply_ticket_id,
                    report=f"任务处理失败：{exc}",
                    reason=str(exc),
                )
            self._record_scheduled_ticket_result(
                input_msg.reply_ticket_id,
                success=False,
                error=str(exc),
            )

            if transient_integration_tools_added:
                clear_terminal_alert_integration_tools(agent)
            if pending_integration_contexts:
                clear_integration_chat_contexts(agent, pending_integration_contexts)
            _loop_mgr.__exit__(None, None, None)

    def _setup_robot_context_from_ticket(self, agent: Agent, ticket: Any) -> None:
        """Rebuild the robot integration context from a persisted QQ reply
        ticket, so a session turn that owns this ticket (e.g. a merged
        pending batch) can use QQ send tools aimed at the right conversation.
        The ticket carries every field the live dispatch path would have
        injected from the inbound message."""
        from app.plugins.robot.contracts import RobotReplyTarget
        from app.services.agent.integrations import setup_integration_chat_contexts

        reply_target = RobotReplyTarget.model_validate(ticket.reply_target or {})
        setup_integration_chat_contexts(
            agent,
            {
                "robot": {
                    "robot_id": ticket.robot_id,
                    "sender_key": ticket.sender_key,
                    "reply_target": reply_target,
                    "conversation_key": ticket.conversation_key,
                    "conversation_generation": ticket.conversation_generation,
                    "reply_requires_awake": False,
                }
            },
        )

    def _process_chat_input(self, input_msg: InputMessage, agent: Agent):
        if input_msg.callback:
            self.add_output_callback(input_msg.callback)

        try:
            clear_robot_context = getattr(agent, "clear_robot_context", None)
            if callable(clear_robot_context):
                clear_robot_context()
            is_plain_chat = input_msg.input_type == InputType.CHAT
            if is_plain_chat:
                try:
                    from app.services.agent.integrations.hooks import (
                        reset_integration_delivery_tracker,
                    )

                    reset_integration_delivery_tracker(f"item:{self.item_id}")
                except Exception:
                    logger.exception(
                        "[AgentSession] Failed to reset QQ delivery tracker: item=%s",
                        self.item_id,
                    )
                # Web chat turns run through the session queue and never get a
                # reply ticket from the route layer. The ticket owns the plan
                # scratchpad and background-job tracking (_reply_ticket_id),
                # so create one here — without it update_plan/run_job callbacks
                # have nothing to attach to.
                if not str(input_msg.reply_ticket_id or "").strip():
                    try:
                        from app.services.agent.reply_ticket import reply_ticket_manager

                        ticket = reply_ticket_manager.create_for_agent(
                            agent,
                            item_id=self.item_id,
                            handler_id=self.handler_id,
                            message=input_msg.content,
                            source_type="web",
                        )
                        input_msg.reply_ticket_id = ticket.ticket_id
                    except Exception:
                        logger.exception(
                            "[AgentSession] Failed to create web reply ticket: item=%s",
                            self.item_id,
                        )
            # QQ-sourced turn (e.g. merged pending batch riding this queue):
            # rebuild the robot context from its ticket so send tools target
            # the right conversation.
            qq_ticket = None
            if str(input_msg.reply_ticket_id or "").strip():
                try:
                    from app.services.agent.reply_ticket import reply_ticket_manager

                    candidate = reply_ticket_manager.get(input_msg.reply_ticket_id)
                    if (
                        candidate is not None
                        and candidate.source_type == "qq"
                        and candidate.robot_id
                        and candidate.reply_target
                    ):
                        qq_ticket = candidate
                except Exception:
                    qq_ticket = None
            if qq_ticket is not None:
                try:
                    self._setup_robot_context_from_ticket(agent, qq_ticket)
                except Exception:
                    logger.exception(
                        "[AgentSession] Failed to setup robot context from ticket: item=%s ticket=%s",
                        self.item_id,
                        input_msg.reply_ticket_id,
                    )
            _chat_loop_mgr = _ManagedEventLoop()
            loop = _chat_loop_mgr.__enter__()
            loop.run_until_complete(agent.start_mcp_servers())

            messages = self._build_chat_messages(agent, input_msg)
            turn_guard = TurnGuard()
            self._current_turn_id = turn_guard.turn_id
            self.emit_status("running", "回复中")

            turn_final_content = ""
            turn_completed = False
            plan_continuation_count = 0
            chat_robot_retry_used = False
            for _ in range(MAX_ITERATIONS):
                from app.services.agent.reply_ticket import reply_ticket_manager as _rtm

                _rtm.touch(input_msg.reply_ticket_id)
                timed_out, timeout_reason = turn_guard.check_timeout()
                if timed_out:
                    self.emit_output(timeout_reason, "agent_warning")
                    break

                if self._abort_flag:
                    self.emit_output("当前轮已中断", "agent_warning")
                    self.emit_status("interrupted", "当前轮已中断")
                    break

                # 每次迭代重算工具表：prepare_capabilities 中途加载的工具
                # 必须本轮可调用，否则模型只能反复 prepare 打转直到守卫停机。
                tools = select_tools_for_turn(
                    agent.get_tools_for_litellm(),
                    source="web",
                    query=input_msg.query or input_msg.content,
                    agent=agent,
                    reply_ticket_id=input_msg.reply_ticket_id,
                )
                if _ == 0:
                    self._log_prompt_budget(messages, tools, context="chat")
                response = self._call_llm(agent, messages, tools=tools)
                message = self._normalize_dsml_tool_message(
                    response.choices[0].message,
                    tools,
                )

                if not (hasattr(message, "tool_calls") and message.tool_calls):
                    # 计划还有未完成步骤时轮次不能就这么结束——由计划自身
                    # 驱动继续执行，超过 5 次推进仍不行动才放行（防死循环）。
                    from app.services.agent.reply_ticket import (
                        plan_has_open_steps,
                        reply_ticket_manager,
                    )

                    _plan_ticket = reply_ticket_manager.get(input_msg.reply_ticket_id)
                    if (
                        _plan_ticket is not None
                        and plan_has_open_steps(_plan_ticket.plan)
                        and plan_continuation_count < 5
                    ):
                        plan_continuation_count += 1
                        messages.append(
                            {"role": "assistant", "content": message.content or ""}
                        )
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
                    if (
                        not chat_robot_retry_used
                        and str(getattr(agent, "robot_context_token", "") or "").strip()
                        and (message.content or "").strip()
                        and not is_terminal_no_reply_intent(message.content or "")
                    ):
                        logger.info(
                            "[AgentSession] Robot final-text correction fired: item=%s",
                            self.item_id,
                        )
                        # QQ 轮次里唯一出口是发送工具：最终文本不会送达。
                        # 模型把没说完的内容写进最终文本时，纠偏让它用工具发完，
                        # 而不是默默丢弃（故事/长内容讲到一半被砍的主因）。
                        messages.append({"role": "assistant", "content": message.content or ""})
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "[系统] 你上一条文本不会送达用户——QQ 回复的唯一出口是 "
                                    "mcp_robot_send_message。如果那是你还没说完的内容，"
                                    "现在用发送工具把它发出去（一条气泡一次调用，发完为止）；"
                                    "如果确实已经全部说完了，只返回 [no_qq_reply]。"
                                ),
                            }
                        )
                        chat_robot_retry_used = True
                        continue
                    if message.content:
                        logger.info(
                            "[AgentSession] Chat turn final text (no correction): item=%s retry_used=%s token=%s",
                            self.item_id,
                            chat_robot_retry_used,
                            bool(str(getattr(agent, "robot_context_token", "") or "").strip()),
                        )
                        final_content = strip_think_tags(guard_fabricated_tool_trace(message.content))
                        turn_final_content = final_content
                        self.emit_output(
                            append_tool_call_footer(
                                final_content,
                                turn_guard.tool_names,
                            ),
                            "agent_response",
                        )
                    turn_completed = True
                    break

                next_messages = self._handle_tool_calls(
                    agent, loop, messages, message, turn_guard,
                    reply_ticket_id=input_msg.reply_ticket_id,
                )
                if next_messages is None:
                    break
                messages = next_messages

            if is_plain_chat and turn_completed:
                self._auto_persist_conversation_memory(
                    str(input_msg.content or ""), turn_final_content, agent
                )
                # Fan out the final text to every reply route merged into this
                # turn. The primary ticket is covered by the paths below; each
                # secondary ticket gets the same combined answer through its
                # own route (QQ -> bridge, web -> history+SSE). Tickets the
                # agent already served via the send tool carry
                # external_report_sent and finalize silently — never duplicate.
                if input_msg.merged_ticket_ids and turn_final_content.strip():
                    from app.services.agent.reply_ticket import reply_ticket_manager

                    for route_ticket_id in input_msg.merged_ticket_ids:
                        if route_ticket_id == input_msg.reply_ticket_id:
                            continue
                        try:
                            route_ticket = reply_ticket_manager.get(route_ticket_id)
                            if (
                                route_ticket is not None
                                and route_ticket.status not in {"delivered", "failed"}
                            ):
                                reply_ticket_manager.deliver(
                                    route_ticket_id,
                                    turn_final_content,
                                )
                        except Exception:
                            logger.exception(
                                "[AgentSession] Failed to fan out to merged ticket: item=%s ticket=%s",
                                self.item_id,
                                route_ticket_id,
                            )
                # QQ ticket turn finished without the agent calling the send
                # tool: deliver the final text through the ticket's own route
                # (integration hook -> QQ bridge). If the agent did send, the
                # ticket already carries external_report_sent and this is a
                # silent finalize instead of a duplicate.
                if qq_ticket is not None and turn_final_content.strip():
                    try:
                        from app.services.agent.reply_ticket import reply_ticket_manager

                        current = reply_ticket_manager.get(qq_ticket.ticket_id)
                        if current is not None and current.status != "delivered":
                            reply_ticket_manager.deliver(
                                qq_ticket.ticket_id,
                                turn_final_content,
                            )
                    except Exception:
                        logger.exception(
                            "[AgentSession] Failed to deliver QQ ticket after chat turn: item=%s ticket=%s",
                            self.item_id,
                            qq_ticket.ticket_id,
                        )
            _chat_loop_mgr.__exit__(None, None, None)
        except Exception as exc:
            _chat_loop_mgr.__exit__(*sys.exc_info())
            logger.error(f"[AgentSession] Chat processing error: {exc}")
            self.emit_output(f"处理失败: {exc}", "agent_error")

    def _build_terminal_messages(
        self,
        agent: Agent,
        input_msg: InputMessage,
        terminal_content: str | None = None,
        *,
        terminal_source: str = TERMINAL_SOURCE_FILTERED,
        pending_command: PendingCommand | None = None,
    ) -> list[dict]:
        effective_terminal_content = terminal_content or input_msg.content
        pending_command = pending_command or self._get_pending_command()
        messages = build_terminal_turn_messages(
            agent,
            item_id=self.item_id,
            terminal_content=effective_terminal_content,
            query=(input_msg.query or effective_terminal_content or "")[:500],
            terminal_source=terminal_source,
            pending_command=pending_command.command if pending_command else "",
            pending_source_context=self._build_pending_source_context_prompt(
                pending_command
            ),
        )

        terminal_text = (effective_terminal_content or "").strip()
        if terminal_text:
            has_system_prompt = any(
                message.get("role") == "system" for message in messages
            )
            if not has_system_prompt:
                system_parts = [get_system_prompt(agent)]
                try:
                    from app.services.agent.capability_state import ensure_loaded
                    from app.services.agent.tool_selection import (
                        TERMINAL_GUIDE_IDS,
                        loaded_guide_skills,
                    )

                    ensure_loaded(self.item_id, guides=TERMINAL_GUIDE_IDS)
                    for skill in loaded_guide_skills(agent, self.item_id):
                        action = getattr(skill, "action", None)
                        prompt = str(getattr(action, "prompt", "") or "").strip()
                        if prompt:
                            system_parts.append(prompt)
                except Exception:
                    pass
                messages = [
                    {"role": "system", "content": "\n\n".join(system_parts)}
                ] + messages

            reply_ticket_prompt = self._build_reply_ticket_prompt(
                input_msg.reply_ticket_id
            )
            if reply_ticket_prompt:
                insert_at = 1 if messages and messages[0].get("role") == "system" else 0
                messages.insert(
                    insert_at,
                    {"role": "system", "content": reply_ticket_prompt},
                )

            last_user_content = ""
            if messages and messages[-1].get("role") == "user":
                last_user_content = str(messages[-1].get("content") or "")
            if terminal_text not in last_user_content:
                source_label = prompt_builder.FILTERED_TERMINAL_LABEL
                if terminal_source == TERMINAL_SOURCE_RAW_FEEDBACK:
                    source_label = prompt_builder.RAW_TERMINAL_LABEL
                messages.append(
                    {
                        "role": "user",
                        "content": f"{source_label}:\n{terminal_text}",
                    }
                )
        self.inject_active_jobs_prompt_context(messages)
        # QQ/终端路径也要看到"已有进行中的计划"——此前只 web chat 路径有，
        # QQ 轮次对别的票据上的 plan 完全失明，正是重复开任务的机制缺口。
        self.inject_open_plans_prompt_context(
            messages, exclude_ticket_id=input_msg.reply_ticket_id
        )
        return messages
    def _get_daemon_jobs_snapshot(self) -> list[dict[str, Any]]:
        now = datetime.now()
        with self.lock:
            snapshot_at = self._daemon_jobs_snapshot_at
            if (
                snapshot_at is not None
                and (now - snapshot_at).total_seconds()
                < DAEMON_JOBS_CONTEXT_REFRESH_SECONDS
            ):
                return [dict(job) for job in self._daemon_jobs_snapshot]

        jobs: list[dict[str, Any]] = []
        try:
            import uuid

            from sqlmodel import Session

            from app.core.db import engine
            from app.models import Item
            from app.services import DaemonConfig, connection_manager

            try:
                item_uuid = uuid.UUID(str(self.item_id))
            except ValueError:
                return []

            with Session(engine) as db:
                item = db.get(Item, item_uuid)
                if (
                    item is None
                    or not item.socket_host
                    or not item.socket_port
                    or not item.api_key
                ):
                    return []
                daemon_config = DaemonConfig(
                    item.socket_host,
                    item.socket_port,
                    item.api_key,
                )

            connection = connection_manager.get_or_create_connection(daemon_config)
            result = connection.list_jobs_http(item_uuid=str(self.item_id))
            raw_jobs = result.get("jobs") if result.get("success") else []
            if isinstance(raw_jobs, list):
                jobs = [dict(job) for job in raw_jobs if isinstance(job, dict)]
        except Exception as exc:
            logger.debug(
                "[AgentSession] Failed to fetch daemon jobs snapshot for item=%s: %s",
                self.item_id,
                exc,
            )
            jobs = []

        with self.lock:
            self._daemon_jobs_snapshot = [dict(job) for job in jobs]
            self._daemon_jobs_snapshot_at = now
        return jobs

    def build_active_jobs_prompt_context(self) -> str:
        lines: list[str] = ["[Execution State]"]
        has_content = False

        main_cmd_state = terminal_command_state_manager.snapshot(self.item_id)
        main_cmd = str(main_cmd_state.get("command") or "").strip()
        if main_cmd:
            source = main_cmd_state.get("source") or "execute_command"
            lines.append(f"Main terminal last command ({source}): {self._short_command(main_cmd, max_length=120)}")
            has_content = True

        running_job = self._get_running_terminal_job()
        if running_job:
            elapsed = int((datetime.now() - running_job.started_at).total_seconds())
            source_hint = f" source={running_job.source_label}" if running_job.source_label else ""
            lines.append(f"Running job: elapsed={elapsed}s{source_hint} cmd={self._short_command(running_job.command, max_length=120)}")
            has_content = True

        daemon_jobs = self._get_daemon_jobs_snapshot()
        if len(daemon_jobs) > 1:
            lines.append(
                f"⚠ 已有 {len(daemon_jobs)} 个后台任务并发在跑——不要再发起新的后台任务，"
                "先等它们完成或用 cancel_job 清掉重复/卡死的任务。"
            )
        for job in daemon_jobs[:10]:
            command = self._short_command(str(job.get("command") or ""), max_length=120)
            try:
                elapsed = int(float(job.get("elapsed_seconds") or 0))
            except (TypeError, ValueError):
                elapsed = 0
            lines.append(f"Background job [running]: job_id={job.get('job_id', '')} elapsed={elapsed}s cmd={command}")
            output_tail = str(job.get("output_tail") or "").strip()
            if output_tail:
                if len(output_tail) > 600:
                    output_tail = output_tail[-600:]
                lines.append(f"  output_tail: {output_tail}")
            has_content = True

        with self.lock:
            finished = list(self._recent_finished_jobs)
        for job in finished:
            status = "OK" if job.get("success") else f"FAIL(exit={job.get('exit_code')})"
            lines.append(f"Background job [{status}]: job_id={job.get('job_id', '')} cmd={job.get('command', '')}")
            job_output = str(job.get("output_tail") or "").strip()
            if job_output and not job.get("success"):
                lines.append(f"  output_tail: {job_output[-100:]}")
            has_content = True

        # 固定状态区：已确认的环境约束（缺失工具黑名单）+ 已验证可用工具。
        # 这些事实持久在 DB 规则表里，不随对话窗口滚动丢失。
        try:
            import uuid as _uuid

            from sqlmodel import Session as _Session

            from app.core.db import engine as _engine
            from app.models import Item as _Item
            from app.services.filters.env_facts import list_missing_tools

            with _Session(_engine) as _db:
                _item = _db.get(_Item, _uuid.UUID(str(self.item_id)))
                _env_rules = dict(_item.output_filter_rules or {}) if _item else {}
            missing_tools = list_missing_tools(_env_rules)
        except Exception:
            missing_tools = []
        with self.lock:
            verified_tools = sorted(self._verified_tools)
        if missing_tools or verified_tools:
            lines.append("Known environment constraints:")
            for entry in missing_tools:
                lines.append(
                    f"- 缺失工具: {entry['tool']}（已实测不存在，累计{entry['count']}次，"
                    "调用会被过滤器直接拦截；复验用 command -v）"
                )
            if verified_tools:
                lines.append(f"- 已验证可用工具: {', '.join(verified_tools)}")
            has_content = True

        # 会话状态卡：欠账/在途票据/重复提示合并成一处紧凑注入。
        # 只在 QQ 轮次注入（web 轮次看到 QQ 欠账会诱发错发，实测过）。
        state_lines = self._build_conversation_state_lines()
        if state_lines:
            lines.append("[会话状态卡]")
            lines.extend(state_lines)
            lines.append(
                "纪律：每个来源的结论各回各家（显式 target 逐个发送）；"
                "其它渠道的内容只是背景，不要代发；已回过的别重复发。"
            )
            has_content = True

        if not has_content:
            return ""

        lines.append(
            "Decision: will the command finish on its own without needing your later input? -> run_job. "
            "Will it enter a state waiting for your input or remain open for future commands? -> execute_command. "
            "Do not start an exact duplicate of a listed running command. "
            "If a background job is already running, do NOT call list_jobs again in the same turn; "
            "the job callback will automatically start the next turn when it finishes."
        )
        return "\n".join(lines)

    def _build_conversation_state_lines(self) -> list[str]:
        """一张卡收齐当前 QQ 会话的全部动态状态（欠账/在途票据/重复）。

        分散注入的提示会被几十次工具调用冲淡；合并成固定位置、固定格式
        的一张卡，模型每轮在同一个地方看到全部事实。
        """
        try:
            agent = self.agent
            context = getattr(agent, "_context", None)
            robot_id = str(getattr(context, "robot_id", "") or "").strip()
        except Exception:
            return []
        if not robot_id:
            return []
        from app.services.agent.reply_ticket import reply_ticket_manager

        lines: list[str] = []
        ticket_id = str(getattr(context, "reply_ticket_id", "") or "").strip()
        ticket = reply_ticket_manager.get(ticket_id) if ticket_id else None
        conversation_key = str(
            getattr(ticket, "conversation_key", "") or ""
        ).strip()

        for entry in reply_ticket_manager.pending_report_debts(self.item_id)[:5]:
            target = f"{entry.get('target_type')}:{entry.get('target_id')}"
            request = str(entry.get("request") or "")[:40]
            kind = str(entry.get("kind") or "")
            if kind == "undelivered":
                lines.append(f"- 欠账 {target}：「{request}」汇报未送达，通道恢复后补发")
            elif kind == "inherited":
                lines.append(
                    f"- 欠账 {target}：票据 {entry.get('ticket_id')} 的「{request}」"
                    "已由你接管，对方还没收到结果"
                )
            else:
                lines.append(f"- 在途 {target}：「{request}」完成后回这个来源")

        if ticket is not None and conversation_key:
            try:
                siblings = reply_ticket_manager.active_sibling_summary(
                    robot_id=robot_id,
                    conversation_key=conversation_key,
                    exclude_ticket_id=ticket.ticket_id,
                )
            except Exception:
                siblings = []
            for sibling in siblings[:3]:
                progress = "；".join(sibling["open_steps"]) if sibling["open_steps"] else (
                    "已汇报" if sibling["reported"] else "待处理"
                )
                lines.append(
                    f"- 同会话票据 {sibling['ticket_id']}「{sibling['request']}」（{progress}）"
                    "——同一件事就接管合并汇报一次，别重复执行"
                )
            duplicate_of = str(getattr(ticket, "duplicate_of", "") or "").strip()
            if duplicate_of:
                lines.append(
                    f"- 本条与活跃票据 {duplicate_of[:8]} 内容完全相同（疑似重复）"
                    "——重复就引用进度回一句，有新意图再正常处理"
                )
        return lines

    def inject_active_jobs_prompt_context(self, messages: list[dict]) -> None:
        context = self.build_active_jobs_prompt_context()
        if not context:
            return
        messages.insert(
            max(len(messages) - 1, 0),
            {
                "role": "system",
                "content": context,
            },
        )

    def _build_chat_messages(self, agent: Agent, input_msg: InputMessage) -> list[dict]:
        pending_context = build_pending_terminal_continuation_prompt(
            self.item_id,
            input_msg.content,
        )
        messages = build_chat_turn_messages(
            agent,
            item_id=self.item_id,
            message=input_msg.content,
            query=input_msg.query or input_msg.content,
            pending_context=pending_context,
            image_urls=input_msg.image_urls or None,
            force_image_vision=input_msg.force_image_vision,
        )
        query = input_msg.query or input_msg.content
        # Always inject authoritative live terminal state (cheap daemon call);
        # never guess "status query" intent from the message text.
        try:
            state = get_terminal_runtime_state(self.item_id)
            messages.insert(
                max(len(messages) - 1, 0),
                {"role": "system", "content": state.prompt_context()},
            )
        except Exception:
            logger.exception(
                "[AgentSession] Failed to inject live terminal state: item=%s",
                self.item_id,
            )
        self.inject_active_jobs_prompt_context(messages)
        self.inject_open_plans_prompt_context(
            messages, exclude_ticket_id=input_msg.reply_ticket_id
        )
        return messages

    def _log_prompt_budget(
        self, messages: list[dict], tools: list, *, context: str
    ) -> None:
        """每轮 prompt 体积体检：总量 + 最大的三条消息 + 工具 schema 大小。

        轮次慢的主因是 prompt 体积（远端均值 ~60k tok/轮），没有分项数据
        就无从压缩。每轮一行日志，长期数据可直接定位肥段。
        """
        try:
            import json as _json

            def _size(msg: dict) -> int:
                content = msg.get("content")
                if isinstance(content, list):
                    return sum(
                        len(str(part.get("text", "")))
                        for part in content
                        if isinstance(part, dict)
                    )
                return len(str(content or ""))

            sizes = sorted(
                ((_size(m), str(m.get("role")), str(m.get("content"))[:60].replace("\n", " ")) for m in messages),
                reverse=True,
            )
            total = sum(size for size, _role, _preview in sizes)
            tools_chars = len(_json.dumps(tools, ensure_ascii=False))
            top = [
                f"{role}:{size}c" for size, role, _preview in sizes[:3]
            ]
            logger.info(
                "[AgentSession] prompt budget: context=%s messages=%s total=%sc top=%s tools=%s/%sc",
                context,
                len(messages),
                total,
                top,
                len(tools),
                tools_chars,
            )
            # 超大消息解剖：逐段打印前缀定位肥段（临时诊断）
            for size, role, _preview in sizes[:1]:
                if size > 50000:
                    fat = next(
                        m for m in messages if _size(m) == size
                    )
                    content = fat.get("content")
                    if isinstance(content, list):
                        content = "\n".join(
                            str(p.get("text", "")) for p in content if isinstance(p, dict)
                        )
                    content = str(content or "")
                    pos = 0
                    for _i in range(25):
                        nxt = content.find("\n\n", pos)
                        if nxt < 0:
                            nxt = min(pos + 2000, len(content))
                        block = content[pos:nxt]
                        if len(block) > 200:
                            logger.info(
                                "[AgentSession] fat-block@%s len=%s | %r",
                                pos,
                                len(block),
                                block[:100],
                            )
                        pos = nxt + 2
                        if pos >= len(content):
                            break
        except Exception:
            pass

    def _auto_persist_conversation_memory(
        self, user_message: str, assistant_message: str, agent: Agent
    ) -> None:
        """显式记忆指令（"记住..."）的自动落盘。

        此前只有旧的同步路由做这一步；stream/QQ 全指望 agent 自己调
        save_memory——它没调就丢了（实测"记住路由器密码"就没存上）。
        persist_memory_candidate 按内容哈希去重，agent 已手动存的不会重复。
        """
        try:
            if not user_message.strip() or not assistant_message.strip():
                return
            # 定义在 chat 路由模块里；函数内延迟导入避免模块级循环依赖。
            from app.api.routes.chat import extract_important_info
            from app.services.agent.memory.vector_store import vector_store
            from app.services.agent.prompts import persist_memory_candidate

            handler_id = str(getattr(agent, "handler_id", "") or "").strip()
            if not handler_id:
                from app.services.agent.memory.scope import resolve_handler_id

                handler_id = resolve_handler_id(self.item_id)
            candidate = extract_important_info(
                user_message,
                assistant_message,
                item_id=self.item_id,
                handler_id=handler_id,
                matched_skills=None,
            )
            if candidate:
                persist_memory_candidate(
                    handler_id,
                    candidate,
                    store=vector_store,
                    source_item_id=self.item_id,
                )
        except Exception:
            logger.debug(
                "[AgentSession] auto-persist conversation memory failed: item=%s",
                self.item_id,
            )

    def inject_open_plans_prompt_context(
        self, messages: list[dict], exclude_ticket_id: str = ""
    ) -> None:
        """新轮次必须看到同 item 其他票据上还在进行的计划——否则用户
        重复发送同一任务时 agent 会毫不知情地再建一份一模一样的计划，
        条带出现重复任务、两份计划互相抢进度。"""
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            others = [
                entry
                for entry in reply_ticket_manager.active_plans_for_item(self.item_id)
                if str(entry.get("ticket_id") or "") != str(exclude_ticket_id or "")
            ]
        except Exception:
            return
        if not others:
            return
        lines = ["该终端已有进行中的任务计划（来自之前的消息）："]
        for entry in others[:3]:
            steps = entry.get("plan") or []
            done = sum(
                1
                for s in steps
                if str(s.get("status")) in {"completed", "cancelled"}
            )
            lines.append(
                f"- 计划 {str(entry.get('ticket_id'))[:8]}："
                f"「{str(entry.get('request') or '')[:60]}」（{done}/{len(steps)} 已完成）"
            )
            for s in steps:
                if str(s.get("status")) not in {"completed", "cancelled"}:
                    lines.append(f"    [{s.get('status')}] {str(s.get('step') or '')[:50]}")
        lines.append(
            "如果用户这条消息与上面某个计划是同一个任务：不要新建计划，直接继续执行该计划，"
            "并用 update_plan 的 supersede_ticket_ids 参数（填入上面的计划 id）关闭旧计划，"
            "把它接管到你当前的票据上；只有确认是不同任务才新建独立计划。"
        )
        messages.insert(
            max(len(messages) - 1, 0),
            {"role": "system", "content": "\n".join(lines)},
        )

    def _get_skill_prompt(
        self,
        agent: Agent,
        query: str = "",
        *,
        skill_query: str | None = None,
    ) -> str:
        prompt_parts = [get_system_prompt(agent)]

        from app.services.agent.tool_selection import loaded_guide_skills

        skills = loaded_guide_skills(agent, self.item_id)
        for skill in skills:
            if skill.category == "system":
                continue
            if skill.action and skill.action.prompt:
                prompt_parts.append(skill.action.prompt)

        base_prompt = "\n\n".join(part.strip() for part in prompt_parts if part and part.strip())
        if query:
            return f"{base_prompt}\n\nUser query: {query}"
        return base_prompt

    def _get_relevant_memories(self, query: str, n_results: int = 3) -> str:
        try:
            from app.services.agent.memory.vector_store import vector_store

            memories = vector_store.search_memories(
                handler_id=str(self.handler_id),
                query=query,
                n_results=n_results,
            )
            if not memories:
                return ""

            return "\n".join([f"- {memory['content']}" for memory in memories])
        except Exception as exc:
            logger.warning(f"[AgentSession] Failed to retrieve memories: {exc}")
            return ""

    def _query_matches_tool(self, query: str, tool_name: str) -> bool:
        tool_base = tool_name.replace("mcp_local_", "").replace("_", " ")
        return tool_base in query.lower()

    def _should_hide_tool_details(self, tool_name: str) -> bool:
        return tool_name in SILENT_TOOL_NAMES

    def _call_llm(
        self,
        agent: Agent,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
    ):
        # 通道压测用的假模型：零延迟返回固定文本，不走网络。只在显式设置
        # TERMPAWS_FAKE_LLM=1 时生效（生产永远不设置）。测的是
        # 队列/合并/轮次通道，不是模型质量。
        if os.environ.get("TERMPAWS_FAKE_LLM") == "1":
            return _fake_llm_response()

        effective_tools = agent.get_tools_for_litellm() if tools is None else tools
        kwargs = build_litellm_completion_kwargs(
            model=agent._context.model,
            messages=messages,
            timeout=REQUEST_TIMEOUT,
            tools=effective_tools,
            api_key=agent._context.api_key,
            api_base=agent._context.api_url,
            model_parameters=getattr(agent._context, "model_parameters", {}),
        )

        last_exc: BaseException | None = None
        for attempt in range(3):
            try:
                response = completion(**kwargs)
                try:
                    usage = getattr(response, "usage", None)
                    if usage:
                        token_usage_tracker.record(
                            str(agent.handler_id or self.handler_id),
                            str(agent._context.model or "unknown"),
                            int(getattr(usage, "prompt_tokens", 0) or 0),
                            int(getattr(usage, "completion_tokens", 0) or 0),
                            int(getattr(usage, "total_tokens", 0) or 0),
                            reply_ticket_id=str(
                                getattr(agent._context, "reply_ticket_id", "") or ""
                            ),
                        )
                except Exception:
                    pass
                return response
            except Exception as exc:
                if not _is_transient_llm_error(exc):
                    raise
                last_exc = exc
                if attempt < 2:
                    import time as _time
                    _time.sleep(2 * (attempt + 1))
                    logger.warning(
                        "[AgentSession] Transient LLM error (attempt %d/3): %s; retrying...",
                        attempt + 1,
                        exc,
                    )
        raise last_exc

    @staticmethod
    def _normalize_dsml_tool_message(message: Any, tools: list[dict]) -> Any:
        raw_content = str(getattr(message, "content", "") or "")
        allowed_tool_names = {
            str(tool.get("function", {}).get("name") or "").strip()
            for tool in tools
            if str(tool.get("function", {}).get("name") or "").strip()
        }
        visible_content, dsml_calls = extract_dsml_tool_calls(
            raw_content,
            allowed_tool_names=allowed_tool_names,
        )
        native_tool_calls = list(getattr(message, "tool_calls", None) or [])
        if native_tool_calls:
            return SimpleNamespace(
                content=visible_content,
                tool_calls=native_tool_calls,
            )
        if not dsml_calls and visible_content == raw_content:
            return message
        return SimpleNamespace(
            content=visible_content,
            tool_calls=[
                SimpleNamespace(
                    id=tool_call["id"],
                    type=tool_call["type"],
                    function=SimpleNamespace(**tool_call["function"]),
                )
                for tool_call in dsml_calls
            ],
        )

    def _handle_tool_calls(
        self,
        agent: Agent,
        loop: asyncio.AbstractEventLoop,
        messages: list[dict],
        message: Any,
        turn_guard: TurnGuard,
        terminal_source: str | None = None,
        tool_results_sink: list[str] | None = None,
        pending_command_for_delivery: PendingCommand | None = None,
        reply_ticket_id: str = "",
    ) -> list[dict] | None:
        tool_calls = list(message.tool_calls or [])
        if tool_calls:
            if terminal_source:
                self._emit_running_terminal_status(terminal_source)
            else:
                self.emit_status("running", "回复中")

        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [],
        }
        tool_messages: list[dict[str, Any]] = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments

            should_stop, reason = turn_guard.before_tool(tool_name, tool_args_str)
            if should_stop:
                self.emit_output(reason, "agent_warning", {"tool_name": tool_name})
                return None

            try:
                tool_args = parse_tool_arguments(tool_name, tool_args_str)
            except ToolArgumentParseError:
                logger.error(
                    "[AgentSession] Failed to parse tool args for %s: %r",
                    tool_name,
                    tool_args_str,
                )
                self.emit_output(
                    f"工具参数解析失败: {tool_name}",
                    "agent_error",
                    {"tool_name": tool_name},
                )
                return None

            normalized_tool_args_str = json.dumps(tool_args, ensure_ascii=False)
            robot_send_source_error = self._robot_send_blocked_by_reply_ticket(
                reply_ticket_id,
                tool_name,
                tool_args,
            )
            if robot_send_source_error:
                logger.info(
                    "[AgentSession] Blocked robot send for non-QQ reply ticket: item=%s tool=%s reason=%s",
                    self.item_id,
                    tool_name,
                    robot_send_source_error,
                )
                return None
            tool_args["item_id"] = self.item_id
            if reply_ticket_id:
                tool_args["_reply_ticket_id"] = reply_ticket_id
            terminal_input_error = self._validate_terminal_command_input(tool_name, tool_args)
            terminal_validation_result = None
            if terminal_input_error:
                if is_terminal_unavailable_error(terminal_input_error):
                    failure_reported = self._fail_and_report_pending_reply(
                        reply_ticket_id,
                        report=terminal_input_error,
                        reason=terminal_input_error,
                    )
                    self._send_pending_integration_response(
                        pending_command_for_delivery,
                        terminal_input_error,
                    )
                    if not failure_reported:
                        self.emit_output(
                            terminal_input_error,
                            "agent_warning",
                            {"tool_name": tool_name},
                        )
                    return None
                terminal_validation_result = {
                    "success": False,
                    "error": (
                        f"{terminal_input_error} Choose the next action yourself. "
                        "Use mcp_local_run_job for an independent non-interactive "
                        "operation, or report the actual failure."
                    ),
                }
            hide_tool_details = self._should_hide_tool_details(tool_name)

            if terminal_validation_result is None:
                if hide_tool_details:
                    self.emit_status("tool", "读取日志中", {"tool_name": tool_name})
                else:
                    self.emit_status("tool", f"调用工具：{tool_name}", {"tool_name": tool_name})
                    self.emit_output(
                        f"执行工具: {tool_name}",
                        "agent_action",
                        {"tool_name": tool_name},
                    )

            if terminal_validation_result is not None:
                result = terminal_validation_result
            else:
                result = loop.run_until_complete(agent.execute_tool(tool_name, tool_args))
                logger.info(f"[AgentSession] Tool {tool_name} executed")
            if tool_name == RUN_JOB_TOOL_NAME:
                turn_guard.reset_timeout_window()

            result_text = self._format_tool_result(result).strip()
            robot_delivery_result = self._is_successful_robot_send(
                tool_name, result_text
            ) or (
                tool_name == ROBOT_SEND_TOOL_NAME
                and isinstance(result, dict)
                and isinstance(result.get("result"), list)
                and any(
                    isinstance(item, dict) and bool(item.get("qq_delivered"))
                    for item in result["result"]
                )
            )
            if (
                tool_results_sink is not None
                and result_text
            ):
                tool_results_sink.append(result_text)
            if robot_delivery_result:
                if reply_ticket_id:
                    try:
                        from app.services.agent.reply_ticket import (
                            reply_ticket_manager,
                        )

                        reply_ticket_manager.mark_external_report_sent(reply_ticket_id)
                    except Exception:
                        pass

                try:
                    from app.services.agent.reply_ticket import (
                        reply_ticket_manager,
                    )

                    reply_ticket_manager.mark_delivered(reply_ticket_id)
                except Exception:
                    logger.exception(
                        "[AgentSession] Failed to mark QQ reply delivered: item=%s ticket=%s",
                        self.item_id,
                        reply_ticket_id,
                    )

                # 前端时间线事件由发送函数自己落（_emit_qq_reply_timeline），
                # 这里不再按结果文本重复生成。
                # Report sent -- stop the turn immediately to prevent
                # duplicate reports.
                return None
            command_dispatch_failed = is_command_dispatch_failure_result(
                tool_name,
                result_text,
                result,
            )
            command_dispatch_pending = is_command_dispatch_pending_result(tool_name, result_text)
            if result_text and not hide_tool_details and not command_dispatch_pending:
                self.emit_output(
                    result_text,
                    "agent_tool_result",
                    {"tool_name": tool_name},
                )

            if command_dispatch_failed:
                record_pending_terminal_continuation(
                    item_id=self.item_id,
                    command=str(tool_args.get("command") or ""),
                    reason="terminal_not_connected",
                    message=COMMAND_DISPATCH_FAILURE_MESSAGE,
                    tool_name=tool_name,
                )
                failure_reported = self._fail_and_report_pending_reply(
                    reply_ticket_id,
                    report=COMMAND_DISPATCH_FAILURE_MESSAGE,
                    reason=COMMAND_DISPATCH_FAILURE_MESSAGE,
                )
                if not failure_reported:
                    self.emit_output(
                        COMMAND_DISPATCH_FAILURE_MESSAGE,
                        "agent_warning",
                        {"tool_name": tool_name},
                    )
                return None

            if is_background_job_started_result(result):
                clear_pending_terminal_continuation(
                    self.item_id,
                    command=str(tool_args.get("command") or ""),
                )
                return None

            assistant_message["tool_calls"].append(
                {
                    "id": tool_call.id,
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
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

            should_stop_after_tool, reason_after_tool = turn_guard.after_tool(
                tool_name,
                result_text,
            )
            if (
                tool_name in COMMAND_TOOL_NAMES
                and result.get("success")
            ):
                clear_pending_terminal_continuation(
                    self.item_id,
                    command=str(tool_args.get("command") or ""),
                )
                self._set_pending_command(tool_name, tool_args)
                self._emit_waiting_terminal_status(tool_name)
                return None

            if should_stop_after_tool:
                self.emit_output(
                    reason_after_tool,
                    "agent_warning",
                    {"tool_name": tool_name},
                )
                return None

        if turn_guard.pending_hint and tool_messages:
            tool_messages[-1]["content"] += "\n" + turn_guard.pending_hint
            turn_guard.pending_hint = ""

        messages.append(assistant_message)
        messages.extend(tool_messages)
        return messages

    def _is_successful_robot_send(self, tool_name: str, result_text: str) -> bool:
        """Detect a successful QQ send without fragile text matching.

        A robot-send tool call counts as a real delivery unless the result is
        an explicit error or an explicit "no message sent" outcome. This makes
        workflow completion robust so a finished report reliably closes the
        task instead of looping forever on background-job wakeups.
        """
        if tool_name != ROBOT_SEND_TOOL_NAME:
            return False
        text = (result_text or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if lowered.startswith("error"):
            return False
        if "no qq message sent" in lowered:
            return False
        if "message sent" in lowered or "???" in text or "???" in text:
            return True
        try:
            return integration_message_sent([text])
        except Exception:
            return False

    def _format_tool_result(self, result: dict) -> str:
        if not isinstance(result, dict):
            return str(result)
        if result.get("success"):
            result_data = result.get("result", [])
            if isinstance(result_data, list):
                texts = [
                    item.get("text", "")
                    for item in result_data
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "\n".join(texts)
            return str(result_data)
        return f"Error: {result.get('error', 'Unknown error')}"

    def abort(self, clear_queue: bool = False):
        cleared = 0
        with self.lock:
            self._abort_flag = True
            self._aborted_this_turn = True
            self._pending_command = None
            if self.state == SessionState.RUNNING:
                self.state = SessionState.INTERRUPTING
            elif clear_queue:
                self.state = SessionState.IDLE
            if clear_queue:
                cleared = self.clear_queued_inputs()
        self._cancel_pending_command_recheck()
        self.emit_status("interrupting", "中断当前轮中")
        if clear_queue and cleared > 0:
            self.emit_output(f"已清空 {cleared} 条排队输入", "agent_warning")
        logger.info(f"[AgentSession] Aborted session for item {self.item_id}")

    def _schedule_delayed_retry(self, ticket_id: str, delay_seconds: float = 10.0):
        """Disabled: callback-driven design. LLM errors are reported, not auto-retried.

        Background job results are the only mechanism that starts new turns.
        Auto-retry threads caused ghost tasks after user cancellation.
        """
        logger.info(
            "[AgentSession] Transient LLM error for ticket=%s; NOT auto-retrying "
            "(callback-driven design). The next job callback will resume the workflow.",
            ticket_id,
        )

    def is_idle(self) -> bool:
        with self.lock:
            return self.state == SessionState.IDLE

    def is_running_turn(self) -> bool:
        with self.lock:
            return self.state in {SessionState.RUNNING, SessionState.INTERRUPTING}

    def shutdown(self):
        self._abort_flag = True
        self._consumer_shutdown = True
        self._wake_event.set()
        self._cancel_pending_command_recheck()


class AgentSessionManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._sessions: dict[str, AgentSession] = {}
        self._global_lock = threading.RLock()
        logger.info("[AgentSessionManager] Initialized")

    def get_or_create_session(self, item_id: str, handler_id: str) -> AgentSession:
        with self._global_lock:
            if item_id not in self._sessions:
                self._sessions[item_id] = AgentSession(item_id, handler_id)
            session = self._sessions[item_id]
        # 工具执行（prepare_capabilities 等）按 item 反查 handler；resume
        # sweep 等不走 prepare_chat_agent 的路径也要保证这个映射存在。
        if item_id and handler_id:
            from app.services.agent.agent import item_handler_context

            item_handler_context.set_handler(item_id, handler_id)
        return session

    def get_session(self, item_id: str) -> AgentSession | None:
        with self._global_lock:
            return self._sessions.get(item_id)

    def remove_session(self, item_id: str):
        with self._global_lock:
            session = self._sessions.pop(item_id, None)
            if session:
                session.shutdown()
                logger.info(f"[AgentSessionManager] Removed session for item {item_id}")

    def process_terminal_output(self, item_id: str, handler_id: str, output: str, query: str = ""):
        session = self.get_or_create_session(item_id, handler_id)
        input_msg = InputMessage(
            input_type=InputType.TERMINAL,
            content=output,
            query=query,
        )
        session.process_input(input_msg)

    def process_chat_message(
        self,
        item_id: str,
        handler_id: str,
        message: str,
        callback: Callable | None = None,
        query: str = "",
    ):
        session = self.get_or_create_session(item_id, handler_id)
        input_msg = InputMessage(
            input_type=InputType.CHAT,
            content=message,
            query=query,
            callback=callback,
        )
        session.process_input(input_msg)


agent_session_manager = AgentSessionManager()
