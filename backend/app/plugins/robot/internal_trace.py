from __future__ import annotations

import ast
import json
import re
from typing import Any

from app.core.tool_markup import (
    contains_dsml_tool_markup,
    strip_dsml_tool_markup,
)

ROBOT_SEND_TOOL_NAME = "mcp_robot_send_message"
ROBOT_SLEEP_TOOL_NAME = "mcp_robot_sleep_conversation"
NO_QQ_REPLY_MARKER = "[no_qq_reply]"
_DEGRADED_SEND_PREAMBLE_RE = re.compile(
    r"(?:\b(?:need|must|should)\b.{0,80}\b(?:tool|function)\s*call\b"
    r"|\b(?:send|delivery)\s+tool\b"
    r"|(?:调用|使用).{0,16}(?:发送)?工具)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_structured_text_string(value: str) -> Any | None:
    stripped = value.strip()
    if len(stripped) < 2 or stripped[0] not in "[{" or stripped[-1] not in "]}":
        return None
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None
    return parsed if isinstance(parsed, (dict, list, tuple)) else None


def _extract_degraded_send_payload(value: str) -> str:
    """Recover visible text when a provider prints send-tool args as content."""
    if "{" not in value or not _DEGRADED_SEND_PREAMBLE_RE.search(value):
        return value

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", value):
        candidate = value[match.start():]
        try:
            payload, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if candidate[end:].strip().strip("`").strip():
            continue
        if not isinstance(payload, dict) or not ({"text", "messages"} & payload.keys()):
            continue

        preamble = value[: match.start()].strip().strip("`").strip()
        if not preamble or not _DEGRADED_SEND_PREAMBLE_RE.search(preamble):
            continue
        extracted = normalize_robot_message_text(payload)
        if extracted.strip():
            return extracted.strip()
    return value


def normalize_robot_message_text(value: Any, *, _depth: int = 0) -> str:
    """Extract plain text from MCP/LLM content blocks without leaking reprs."""
    if _depth > 5 or value is None:
        return ""
    if isinstance(value, str):
        parsed = _parse_structured_text_string(value)
        if parsed is not None:
            extracted = normalize_robot_message_text(parsed, _depth=_depth + 1)
            if extracted:
                return extracted
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "message", "data"):
            if key not in value:
                continue
            extracted = normalize_robot_message_text(
                value.get(key),
                _depth=_depth + 1,
            )
            if extracted:
                return extracted
        return ""
    if isinstance(value, (list, tuple)):
        parts = [
            normalize_robot_message_text(item, _depth=_depth + 1).strip()
            for item in value
        ]
        return "\n".join(part for part in parts if part)
    return str(value)


def _is_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _join_visible_lines(lines: list[str]) -> str:
    message = ""
    for line in lines:
        if not message:
            message = line
        elif _is_cjk(message[-1]) or _is_cjk(line[0]):
            message = f"{message}{line}"
        else:
            message = f"{message} {line}"
    return message.strip()


def _normalized_lines(value: Any) -> list[str]:
    text = normalize_robot_message_text(value)
    return [line.strip() for line in text.splitlines() if line.strip()]


def _is_tool_execution_line(line: str) -> bool:
    return line.startswith("Executing tool:") or line.startswith("执行工具:")


def _contains_robot_send_tool_execution(line: str) -> bool:
    return ROBOT_SEND_TOOL_NAME in line and (
        "Executing tool:" in line or "执行工具:" in line
    )


def _contains_robot_sleep_tool_execution(line: str) -> bool:
    return ROBOT_SLEEP_TOOL_NAME in line and (
        "Executing tool:" in line or "鎵ц宸ュ叿:" in line
    )

def _is_robot_send_success_line(line: str) -> bool:
    return (
        line.startswith("Message sent to QQ ")
        or line.startswith("Message sent to current robot conversation.")
        or line.startswith("Broadcast sent to ")
        or line.startswith("Multi-target QQ send to ")
        or line.startswith("已送达用户")
    )


def _contains_robot_send_success(line: str) -> bool:
    return (
        "Message sent to QQ " in line
        or "Message sent to current robot conversation." in line
        or "Broadcast sent to " in line
        or "已送达用户" in line
    )


def _is_no_qq_reply_marker_line(line: str) -> bool:
    return line == NO_QQ_REPLY_MARKER or line.endswith(f": {NO_QQ_REPLY_MARKER}")


_INTERNAL_CONTEXT_CARD_PREFIXES = (
    "[Pending QQ messages",
    "[Internal corrective turn]",
    "[Current QQ conversation impression card",
    "[Recent QQ live context",
)

_INTERNAL_CONTEXT_CARD_LINE_PREFIXES = (
    "These messages arrived while the bot was already thinking",
    "Treat them as current live QQ messages",
    "source=QQ;",
    "Messages from the same sender are one evolving intent",
    "If multiple distinct senders each asked the bot something",
    "Default to one concise QQ bubble",
)

_PENDING_BATCH_ENTRY_RE = re.compile(r"^\d+\.\s*sender=.+sender_key=.+")

# Reasoning-model leaks: strip <think>...</think> blocks and stray think tags
# so chain-of-thought never reaches QQ-visible text.
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def _strip_think_markup(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub(" ", str(text or ""))
    cleaned = THINK_TAG_RE.sub("", cleaned)
    return cleaned


def _is_internal_context_card_line(line: str) -> bool:
    if line.startswith(_INTERNAL_CONTEXT_CARD_PREFIXES):
        return True
    if line.startswith(_INTERNAL_CONTEXT_CARD_LINE_PREFIXES):
        return True
    return bool(_PENDING_BATCH_ENTRY_RE.search(line))


def _is_internal_send_trace_line(line: str) -> bool:
    return (
        _contains_robot_send_tool_execution(line)
        or _contains_robot_sleep_tool_execution(line)
        or _contains_robot_send_success(line)
        or _is_no_qq_reply_marker_line(line)
        or "No QQ message sent:" in line
        or _is_internal_context_card_line(line)
    )


def is_robot_internal_trace_text(value: str) -> bool:
    if not _strip_think_markup(value).strip():
        return bool(str(value or "").strip())
    lines = _normalized_lines(value)
    if not lines:
        return False

    if lines[0].startswith(_INTERNAL_CONTEXT_CARD_PREFIXES):
        return True

    normalized = normalize_robot_message_text(value)
    if contains_dsml_tool_markup(normalized):
        return not strip_dsml_tool_markup(normalized).strip()

    if _is_robot_send_success_line(lines[0]):
        return True

    if _is_tool_execution_line(lines[0]):
        if ROBOT_SEND_TOOL_NAME not in lines[0]:
            return False
        return all(
            _is_tool_execution_line(line) or _is_robot_send_success_line(line)
            for line in lines
        )

    return False


_SYSTEM_PROMPT_LEAK_MARKERS = (
    "Current QQ message",
    "Current QQ sender",
    "Replied QQ message",
    "trigger=background_job_result",
    "authoritative for this turn",
)


def _is_system_prompt_leak(text: str) -> bool:
    hits = sum(1 for marker in _SYSTEM_PROMPT_LEAK_MARKERS if marker in text)
    return hits >= 2


def sanitize_robot_visible_text(value: Any) -> str:
    """Remove robot tool traces from text that may be sent or shown as chat memory."""
    normalized_raw = str(value or "")
    if _is_system_prompt_leak(normalized_raw):
        return ""
    cleaned_lines: list[str] = []
    previous_blank = False
    normalized = normalize_robot_message_text(value)
    normalized = _strip_think_markup(normalized)
    normalized = _extract_degraded_send_payload(normalized)
    normalized = strip_dsml_tool_markup(normalized)
    for raw_line in normalized.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped and _is_internal_send_trace_line(stripped):
            continue

        if NO_QQ_REPLY_MARKER in line:
            line = line.replace(NO_QQ_REPLY_MARKER, "").rstrip()
            stripped = line.strip()
            if not stripped:
                continue

        if not stripped:
            if cleaned_lines and not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        cleaned_lines.append(line)
        previous_blank = False

    while cleaned_lines and not cleaned_lines[-1].strip():
        cleaned_lines.pop()

    return "\n".join(cleaned_lines).strip()


def compact_robot_visible_message_text(value: Any) -> str:
    """Keep one outgoing QQ message as one visual line.

    Multi-message replies should use the MCP `messages` array. Newlines inside a
    single message are presentation noise from the model, not separate QQ sends.
    """
    normalized = normalize_robot_message_text(value)
    return _join_visible_lines(
        [
            line.strip()
            for line in normalized.replace("\r\n", "\n")
            .replace("\r", "\n")
            .split("\n")
            if line.strip()
        ]
    )
