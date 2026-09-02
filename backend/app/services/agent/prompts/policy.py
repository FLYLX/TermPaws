from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PromptTurnType(str, Enum):
    CHAT = "chat"
    TERMINAL_FILTERED = "terminal_filtered"
    TERMINAL_RAW_FEEDBACK = "terminal_raw_feedback"


@dataclass(frozen=True)
class PromptMemoryPolicy:
    include_session_summary: bool
    include_recent_history: bool
    include_long_term: bool
    max_recent_messages: int
    max_long_term_memories: int
    allowed_long_term_types: tuple[str, ...]


def resolve_prompt_memory_policy(turn_type: PromptTurnType) -> PromptMemoryPolicy:
    if turn_type == PromptTurnType.CHAT:
        return PromptMemoryPolicy(
            include_session_summary=False,
            # 闲聊需要"最近几条"上下文：此前为 False，且 stream 路由丢弃了
            # 前端传来的 history，等于闲聊零上下文、答不上任何追问。
            include_recent_history=True,
            include_long_term=True,
            max_recent_messages=6,
            max_long_term_memories=3,
            allowed_long_term_types=("fact", "preference", "error", "context"),
        )

    if turn_type == PromptTurnType.TERMINAL_RAW_FEEDBACK:
        return PromptMemoryPolicy(
            include_session_summary=False,
            include_recent_history=False,
            include_long_term=False,
            max_recent_messages=6,
            max_long_term_memories=0,
            allowed_long_term_types=(),
        )

    return PromptMemoryPolicy(
        include_session_summary=False,
        include_recent_history=False,
        include_long_term=True,
        max_recent_messages=6,
        max_long_term_memories=12,
        allowed_long_term_types=("preference", "error", "context"),
    )


@dataclass(frozen=True)
class MemoryCandidate:
    content: str
    memory_type: str
    ttl_days: int | None
    metadata: dict[str, Any]



EXPLICIT_MEMORY_CUES = (
    "记住",
    "记下来",
    "保存这个",
    "请保存",
    "remember this",
    "save this",
    "记一下",
    "帮我记",
    "帮我记住",
    "你记住",
    "记得",
    "以后记得",
)

PREFERENCE_MEMORY_CUES = (
    "偏好",
    "以后都",
    "始终",
    "默认",
    "人格",
    "人设",
    "语气",
    "口吻",
    "说话方式",
    "prefer",
    "preference",
    "persona",
    "tone",
    "style",
)

TASK_MEMORY_CUES = (
    "任务",
    "待办",
    "todo",
    "接下来",
    "当前目标",
)

ERROR_MEMORY_CUES = (
    "错误",
    "报错",
    "异常",
    "bug",
    "问题",
)

EXPLICIT_MEMORY_PATTERNS = (
    r"(?:帮我|给我)?\s*记(?:一下|下|住|着)?(?:这[个件条])?[:：,，\s]*(.+)$",
    r"(?:你\s*)?记得[:：,，\s]*(.+)$",
    r"(?:请\s*)?记住(?:这[个件条])?[:：,，\s]*(.+)$",
    r"(?:请\s*)?记下来(?:这[个件条])?[:：,，\s]*(.+)$",
    r"(?:请\s*)?保存(?:这个|这条|这段)?[:：,，\s]*(.+)$",
    r"(?:please\s+)?remember(?:\s+this)?[:\s,-]*(.+)$",
    r"(?:please\s+)?save(?:\s+this)?[:\s,-]*(.+)$",
)

GENERIC_MEMORY_PAYLOADS = {
    "这个",
    "这个结果",
    "这个结论",
    "这个信息",
    "这个设置",
    "这个配置",
    "这件事",
    "这条",
    "这段",
    "this",
    "that",
    "it",
}

CONFIRMATION_CUES = (
    "对",
    "对的",
    "是的",
    "没错",
    "就是这个",
    "就是这个路径",
    "这个结论对",
    "确认",
    "可以，就这个",
    "yes",
    "correct",
    "that's right",
)

TASK_RESOLVED_CUES = (
    "任务完成",
    "完成了",
    "已完成",
    "搞定了",
    "done",
)

ERROR_RESOLVED_CUES = (
    "错误解决",
    "问题解决",
    "已经解决",
    "已解决",
    "修复了",
    "修好了",
    "fixed",
    "resolved",
)

WAITING_STATE_PATTERNS = (
    "命令已发送",
    "等待终端反馈",
    "等待终端结果确认",
    "暂无新反馈",
    "读取原生日志反馈中",
    "读取日志中",
    "排队中",
    "queued",
    "waiting",
)

SENSITIVE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:api[_-]?key|token|secret|password)\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
)

LOG_NOISE_PATTERNS = (
    re.compile(r"^\[[^\]]+\]\s*#", re.MULTILINE),
    re.compile(r"^\[[^\]]+\]", re.MULTILINE),
    re.compile(r"^(?:Traceback|Error:|Exception:)", re.MULTILINE),
    re.compile(r"Executing tool:\s*mcp_robot_send_message", re.IGNORECASE),
    re.compile(r"Executing tool:\s*mcp_robot_sleep_conversation", re.IGNORECASE),
    re.compile(r"Message sent to current robot conversation\.", re.IGNORECASE),
    re.compile(r"Message sent to QQ\s+\w+", re.IGNORECASE),
    re.compile(r"\[no_qq_reply\]", re.IGNORECASE),
)

MEMORY_TTL_DAYS = {
    "preference": None,
    "fact": 90,
    "error": None,
    "context": 30,
}
SUPPORTED_LONG_TERM_MEMORY_TYPES = frozenset(MEMORY_TTL_DAYS)

MAX_MEMORY_CONTENT_LENGTH = 240
MAX_MEMORY_LINE_COUNT = 4
STATUS_UPDATE_STOPWORDS = {
    "这个",
    "这个任务",
    "这个错误",
    "这个问题",
    "任务",
    "错误",
    "问题",
    "已经",
    "已",
    "了",
    "就",
    "是",
    "就是",
    "that",
    "this",
}

STATUS_BY_MEMORY_TYPE: dict[str, tuple[str, ...]] = {
    "error": ("active", "resolved"),
}

AUTO_MEMORY_CQ_AT_RE = re.compile(
    r"\[CQ:at,[^\]]*?(?:qq|user_id)=([^,\]]+)[^\]]*\]",
    re.IGNORECASE,
)
AUTO_MEMORY_CQ_RE = re.compile(r"\[CQ:[^\]]+\]", re.IGNORECASE)

def resolve_memory_ttl_days(memory_type: str) -> int | None:
    return MEMORY_TTL_DAYS.get(memory_type, 30)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _extract_explicit_memory_payload(user_message: str) -> str:
    raw_message = (user_message or "").strip()
    if not raw_message:
        return ""

    for pattern in EXPLICIT_MEMORY_PATTERNS:
        match = re.search(pattern, raw_message, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        payload = match.group(1).strip(" \t\r\n:：,，;；。")
        # 剥掉指令尾巴（"记住后回复我/告诉我"等），它们是说话方式不是要记的内容。
        payload = re.sub(
            r"(?:[。，,；;]?\s*(?:记住|记好|记下|记下来|记一下)(?:了|好|后|以后)?(?:再|然后|并)?(?:告诉|回复|通知|跟……说|跟)(?:我|大家)?[。！!~]*)+$",
            "",
            payload,
            flags=re.IGNORECASE,
        ).strip(" \t\r\n:：,，;；。")
        if payload:
            return payload

    return ""


def _is_generic_memory_payload(payload: str) -> bool:
    normalized = _normalize_text(payload).lower().strip(" .,:;")
    return not normalized or normalized in GENERIC_MEMORY_PAYLOADS


def _contains_sensitive_content(content: str) -> bool:
    return any(pattern.search(content) for pattern in SENSITIVE_PATTERNS)


def _looks_like_log_noise(content: str) -> bool:
    stripped = (content or "").strip()
    if not stripped:
        return True

    line_count = len([line for line in stripped.splitlines() if line.strip()])
    if line_count > MAX_MEMORY_LINE_COUNT:
        return True

    if "```" in stripped:
        return True

    lowered = stripped.lower()
    if any(phrase in lowered for phrase in ("stdout", "stderr", "traceback", "stack trace")):
        return True

    return any(pattern.search(stripped) for pattern in LOG_NOISE_PATTERNS)


def _is_waiting_state_content(content: str) -> bool:
    normalized = _normalize_text(content).lower()
    return any(pattern.lower() in normalized for pattern in WAITING_STATE_PATTERNS)


def _build_memory_content(payload: str, memory_type: str) -> str:
    normalized = _normalize_text(payload)
    if memory_type == "preference" and not normalized.startswith("用户偏好"):
        return f"用户偏好：{normalized}"
    if memory_type == "error":
        normalized = re.sub(r"^(?:这个|当前)?(?:错误|报错|异常|问题)(?:是|为|[:：])?\s*", "", normalized)
        if normalized.startswith("已知错误："):
            return normalized
        return f"已知错误：{normalized}"
    return normalized


def _build_content_hash(content: str) -> str:
    normalized = _normalize_text(content)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def should_reject_long_term_memory(content: str) -> bool:
    normalized = (content or "").strip()
    if not normalized:
        return True
    if len(normalized) > MAX_MEMORY_CONTENT_LENGTH:
        return True
    if _is_waiting_state_content(normalized):
        return True
    if _looks_like_log_noise(normalized):
        return True
    if _contains_sensitive_content(normalized):
        return True
    return False


def _normalize_memory_key_fragment(value: str) -> str:
    normalized = _normalize_text(value).lower()
    normalized = re.sub(r"[`\"'“”‘’]", "", normalized)
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff._/-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized[:96]


def _infer_preference_memory_key(content: str) -> str | None:
    normalized = _normalize_text(content).lower()
    matched_axes: list[str] = []
    if "中文" in normalized or "english" in normalized or "英文" in normalized:
        matched_axes.append("preference.language")
    if "简洁" in normalized or "详细" in normalized or "啰嗦" in normalized or "verbosity" in normalized:
        matched_axes.append("preference.verbosity")
    if "不要" in normalized and ("思考" in normalized or "推理" in normalized or "cot" in normalized):
        matched_axes.append("preference.reasoning_visibility")
    if "聊天框" in normalized and ("状态" in normalized or "过程" in normalized):
        matched_axes.append("preference.chat_visibility")

    if len(matched_axes) == 1:
        return matched_axes[0]
    return None


def _extract_fact_subject(content: str) -> str:
    normalized = _normalize_text(content)
    subject_patterns = (
        r"^([\u4e00-\u9fffA-Za-z0-9_./\\-]+)\s*(?:位于|在|路径是|是|为|使用)\s+.+$",
        r"^([\u4e00-\u9fffA-Za-z0-9_./\\-]+)\s*(?:对应|指向)\s+.+$",
        r"^(?:当前)?([\u4e00-\u9fffA-Za-z0-9_./\\-]+(?:目录|路径|端口|脚本|文件))\s*(?:是|为|在)\s+.+$",
    )
    for pattern in subject_patterns:
        match = re.match(pattern, normalized)
        if match:
            return match.group(1).strip()
    return ""


def _extract_error_subject(content: str) -> str:
    normalized = _normalize_text(content)
    if normalized.startswith("已知错误："):
        return normalized.removeprefix("已知错误：").strip()
    match = re.match(r"^(?:当前)?(?:错误|报错|异常|问题)[:：]?\s*(.+)$", normalized)
    if match:
        return match.group(1).strip()
    return ""


def _infer_error_memory_key(content: str) -> str | None:
    subject = _extract_error_subject(content)
    if not subject:
        return None
    subject_key = _normalize_memory_key_fragment(subject)
    if subject_key:
        return f"error.{subject_key}"
    return None


def infer_memory_key(content: str, memory_type: str) -> str | None:
    if memory_type == "preference":
        return _infer_preference_memory_key(content)
    if memory_type == "fact":
        subject = _extract_fact_subject(content)
        if subject:
            subject_key = _normalize_memory_key_fragment(subject)
            if subject_key:
                return f"fact.{subject_key}"
    if memory_type == "error":
        return _infer_error_memory_key(content)
    return None


def _is_confirmation_message(user_message: str) -> bool:
    normalized = _normalize_text(user_message).lower()
    if not normalized:
        return False
    return any(cue in normalized for cue in CONFIRMATION_CUES)


def _looks_like_stable_fact(content: str) -> bool:
    normalized = _normalize_text(content)
    if normalized.startswith("用户偏好："):
        return True

    stable_patterns = (
        r"/[\w./-]+",
        r"\b[\w.-]+\.(?:py|js|ts|tsx|json|yaml|yml|toml|md)\b",
        r"(?:路径|目录|端口|配置|脚本|文件)\s*(?:是|为|在|位于)",
        r"[\u4e00-\u9fffA-Za-z0-9_./\\-]+\s*(?:位于|在|路径是|是|为|使用)\s+.+",
    )
    return any(re.search(pattern, normalized) for pattern in stable_patterns)


def should_auto_persist_conversation_memory(user_message: str, assistant_message: str) -> bool:
    if not user_message or not assistant_message.strip():
        return False

    normalized = user_message.lower()
    return any(cue in user_message or cue in normalized for cue in EXPLICIT_MEMORY_CUES)


def infer_explicit_memory_type(user_message: str) -> str:
    normalized = (user_message or "").lower()
    if any(cue in user_message or cue in normalized for cue in PREFERENCE_MEMORY_CUES):
        return "preference"
    if any(cue in user_message or cue in normalized for cue in TASK_MEMORY_CUES):
        return "context"
    if any(cue in user_message or cue in normalized for cue in ERROR_MEMORY_CUES):
        return "error"
    return "fact"


def _contains_any(value: str, cues: tuple[str, ...]) -> bool:
    normalized = (value or "").casefold()
    return any(cue.casefold() in normalized for cue in cues)


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)


def _normalize_memory_message_text(value: str) -> str:
    normalized = AUTO_MEMORY_CQ_AT_RE.sub(
        lambda match: f" @QQ({match.group(1).strip()}) ",
        str(value or ""),
    )
    normalized = AUTO_MEMORY_CQ_RE.sub(" ", normalized)
    return _normalize_text(normalized)


def _extract_memory_tokens(value: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_./:-]+", _normalize_text(value))
        if token and token.lower() not in STATUS_UPDATE_STOPWORDS and len(token) > 1
    }
    return tokens


def _get_memory_sort_key(memory: dict[str, Any]) -> tuple[str, str]:
    metadata = memory.get("metadata") or {}
    return (
        str(metadata.get("updated_at") or ""),
        str(metadata.get("created_at") or ""),
    )


def _resolve_target_memory_type(user_message: str) -> str | None:
    normalized = _normalize_text(user_message).lower()
    if any(cue in normalized for cue in ERROR_RESOLVED_CUES) or any(
        cue in normalized for cue in ("错误", "报错", "异常", "bug", "问题")
    ):
        return "error"
    return None


def _resolve_status_transition(user_message: str, memory_type: str) -> str | None:
    normalized = _normalize_text(user_message).lower()
    if memory_type == "error":
        if any(cue in normalized for cue in ERROR_RESOLVED_CUES):
            return "resolved"
        return None
    return None


def _is_active_memory(memory: dict[str, Any], memory_type: str) -> bool:
    status = resolve_memory_status(memory)
    if memory_type == "error":
        return status not in {"resolved"}
    return True


def _select_memory_for_status_update(
    user_message: str,
    memories: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not memories:
        return None

    active_memories = [memory for memory in memories if _is_active_memory(memory, memory.get("metadata", {}).get("memory_type", ""))]
    candidates = active_memories or memories

    if len(candidates) == 1:
        return candidates[0]

    user_tokens = _extract_memory_tokens(user_message)
    scored: list[tuple[int, tuple[str, str], dict[str, Any]]] = []
    for memory in candidates:
        content = str(memory.get("content") or "")
        score = len(_extract_memory_tokens(content) & user_tokens)
        scored.append((score, _get_memory_sort_key(memory), memory))

    scored.sort(key=lambda item: (item[0], item[1][0], item[1][1]), reverse=True)
    if not scored:
        return None

    top_score = scored[0][0]
    second_score = scored[1][0] if len(scored) > 1 else -1
    if top_score > 0 and top_score > second_score:
        return scored[0][2]
    return None


def _apply_memory_status_to_content(content: str, memory_type: str, status: str) -> str:
    normalized = _normalize_text(content)
    normalized = re.sub(r"（已完成|已解决）$", "", normalized).strip()
    if memory_type == "error" and status == "resolved":
        return f"{normalized}（已解决）"
    return normalized


def get_allowed_memory_statuses(memory_type: str) -> tuple[str, ...]:
    return STATUS_BY_MEMORY_TYPE.get(memory_type, ())


def resolve_memory_status(memory: dict[str, Any]) -> str | None:
    metadata = memory.get("metadata") or {}
    memory_type = str(metadata.get("memory_type") or "")
    normalized_status = str(metadata.get("status") or "").lower()
    if normalized_status in get_allowed_memory_statuses(memory_type):
        return normalized_status

    content = _normalize_text(str(memory.get("content") or ""))
    if memory_type == "error":
        if content.endswith("（已解决）"):
            return "resolved"
        return "active"
    return None


def build_manual_status_update(
    memory: dict[str, Any],
    target_status: str,
    *,
    source: str = "manual_status_action",
) -> tuple[str, dict[str, Any]] | None:
    metadata = dict(memory.get("metadata") or {})
    memory_type = str(metadata.get("memory_type") or "")
    allowed_statuses = get_allowed_memory_statuses(memory_type)
    normalized_target_status = str(target_status or "").lower()
    if normalized_target_status not in allowed_statuses:
        return None

    current_status = resolve_memory_status(memory)
    existing_content = str(memory.get("content") or "").strip()
    updated_content = _apply_memory_status_to_content(
        existing_content,
        memory_type,
        normalized_target_status,
    )
    if should_reject_long_term_memory(updated_content):
        return None

    if current_status == normalized_target_status and updated_content == _normalize_text(existing_content):
        return updated_content, {
            **metadata,
            "status": normalized_target_status,
        }

    updated_metadata = {
        **metadata,
        "type": source,
        "source": source,
        "verified": True,
        "status": normalized_target_status,
        "status_updated_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "content_hash": _build_content_hash(updated_content),
    }
    return updated_content, updated_metadata


CASUAL_FACT_START_CUES = (
    "我家",
    "我的",
    "我想",
    "我喜欢",
    "我打算",
    "我在",
    "我是",
    "我有",
    "我买",
    "我养",
    "我住",
    "我公司",
)
_CASUAL_QUESTION_TAILS = ("吗", "嘛", "呢", "吧", "？", "?")


def build_casual_fact_memory_candidate(user_message: str) -> MemoryCandidate | None:
    """不经意说出的第一人称事实也值得记住（"我家楼下新开了家拉面"）。

    只收第一人称陈述句：疑问/反问/日志噪音/敏感内容一律不收（复用既有
    拒绝规则）。这类记忆由 30 分钟一轮的 dedup sweep 合并清理，
    prompt 侧另有 3 条/6K 封顶，不会撑爆上下文。
    """
    text = _normalize_text(user_message)
    if len(text) < 8:
        return None
    if not text.startswith(CASUAL_FACT_START_CUES):
        return None
    if "?" in text or "\uff1f" in text or text.endswith(_CASUAL_QUESTION_TAILS):
        return None
    content = _build_memory_content(text, "fact")
    if should_reject_long_term_memory(content):
        return None
    metadata: dict[str, Any] = {
        "type": "conversation_casual",
        "source": "chat_user",
        "verified": False,
        "content_hash": _build_content_hash(content),
    }
    memory_key = infer_memory_key(content, "fact")
    if memory_key:
        metadata["memory_key"] = memory_key
    return MemoryCandidate(
        content=content,
        memory_type="fact",
        ttl_days=resolve_memory_ttl_days("fact"),
        metadata=metadata,
    )


def build_conversation_memory_candidate(
    user_message: str,
    assistant_message: str,
    *,
    matched_skills: list[Any] | None = None,
) -> MemoryCandidate | None:
    if not should_auto_persist_conversation_memory(user_message, assistant_message):
        return None

    payload = _normalize_memory_message_text(
        _extract_explicit_memory_payload(user_message)
    )
    if _is_generic_memory_payload(payload):
        logger.debug("[MemoryPolicy] Skip explicit memory candidate: generic payload")
        return None
    memory_type = infer_explicit_memory_type(user_message)
    content = _build_memory_content(payload, memory_type)
    if should_reject_long_term_memory(content):
        logger.debug("[MemoryPolicy] Skip explicit memory candidate: rejected by policy")
        return None

    metadata: dict[str, Any] = {
        "type": "conversation_explicit",
        "source": "chat_user",
        "verified": memory_type == "preference",
        "content_hash": _build_content_hash(content),
    }
    if memory_type == "error":
        metadata["status"] = "active"
    if matched_skills:
        skill_ids = [
            skill.skill_id
            for skill in matched_skills
            if getattr(skill, "skill_id", None)
        ]
        if skill_ids:
            metadata["skills"] = skill_ids

    memory_key = infer_memory_key(content, memory_type)
    if memory_key:
        metadata["memory_key"] = memory_key

    return MemoryCandidate(
        content=content,
        memory_type=memory_type,
        ttl_days=resolve_memory_ttl_days(memory_type),
        metadata=metadata,
    )


def build_confirmation_memory_candidate(
    user_message: str,
    previous_assistant_message: str,
    *,
    matched_skills: list[Any] | None = None,
) -> MemoryCandidate | None:
    if not _is_confirmation_message(user_message):
        return None

    content = _normalize_text(previous_assistant_message)
    if should_reject_long_term_memory(content):
        return None
    if not _looks_like_stable_fact(content):
        return None

    memory_type = "fact"
    if content.startswith("用户偏好："):
        memory_type = "preference"

    metadata: dict[str, Any] = {
        "type": "conversation_confirmed",
        "source": "assistant_response",
        "verified": True,
        "confirmation_message": _normalize_text(user_message),
        "content_hash": _build_content_hash(content),
    }
    if matched_skills:
        skill_ids = [
            skill.skill_id
            for skill in matched_skills
            if getattr(skill, "skill_id", None)
        ]
        if skill_ids:
            metadata["skills"] = skill_ids

    memory_key = infer_memory_key(content, memory_type)
    if memory_key:
        metadata["memory_key"] = memory_key

    return MemoryCandidate(
        content=content,
        memory_type=memory_type,
        ttl_days=resolve_memory_ttl_days(memory_type),
        metadata=metadata,
    )


def build_status_update_memory_candidate(
    handler_id: str,
    user_message: str,
    *,
    store: Any,
    matched_skills: list[Any] | None = None,
) -> MemoryCandidate | None:
    target_type = _resolve_target_memory_type(user_message)
    if not target_type:
        return None

    next_status = _resolve_status_transition(user_message, target_type)
    if not next_status:
        return None

    try:
        memories = store.get_all_memories(handler_id, memory_type=target_type)
    except Exception as exc:
        logger.warning(
            "[MemoryPolicy] Failed to inspect existing %s memories for handler=%s: %s",
            target_type,
            handler_id,
            exc,
        )
        return None

    target_memory = _select_memory_for_status_update(user_message, memories)
    if not target_memory:
        return None

    existing_metadata = dict(target_memory.get("metadata") or {})
    existing_content = str(target_memory.get("content") or "").strip()
    updated_content = _apply_memory_status_to_content(
        existing_content,
        target_type,
        next_status,
    )
    if should_reject_long_term_memory(updated_content):
        return None

    metadata: dict[str, Any] = {
        **existing_metadata,
        "type": "conversation_status_update",
        "source": "chat_user",
        "verified": True,
        "status": next_status,
        "status_updated_by": _normalize_text(user_message),
        "status_updated_at": datetime.now().isoformat(),
        "content_hash": _build_content_hash(updated_content),
    }
    if matched_skills:
        skill_ids = [
            skill.skill_id
            for skill in matched_skills
            if getattr(skill, "skill_id", None)
        ]
        if skill_ids:
            metadata["skills"] = skill_ids

    memory_key = existing_metadata.get("memory_key") or infer_memory_key(
        updated_content,
        target_type,
    )
    if memory_key:
        metadata["memory_key"] = memory_key

    return MemoryCandidate(
        content=updated_content,
        memory_type=target_type,
        ttl_days=resolve_memory_ttl_days(target_type),
        metadata=metadata,
    )


def persist_memory_candidate(
    handler_id: str,
    candidate: MemoryCandidate | None,
    *,
    store: Any,
    source_item_id: str = "",
) -> str | None:
    if candidate is None:
        return None
    if candidate.memory_type not in SUPPORTED_LONG_TERM_MEMORY_TYPES:
        logger.info(
            "[MemoryPolicy] Skip unsupported long-term memory type=%s for handler=%s",
            candidate.memory_type,
            handler_id,
        )
        return None
    source_item_id = str(source_item_id or "").strip()
    if source_item_id:
        candidate.metadata.setdefault("source_item_id", source_item_id)
    content_hash = candidate.metadata.get("content_hash")
    try:
        existing_memories = store.get_all_memories(handler_id, memory_type=candidate.memory_type)
    except Exception as exc:
        logger.warning(
            "[MemoryPolicy] Failed to inspect existing memories for handler=%s: %s",
            handler_id,
            exc,
        )
        existing_memories = []

    if content_hash:
        for memory in existing_memories:
            existing_hash = (memory.get("metadata") or {}).get("content_hash")
            if existing_hash == content_hash:
                logger.info(
                    "[MemoryPolicy] Skip duplicate long-term memory by content hash for handler=%s",
                    handler_id,
                )
                return None

    memory_key = candidate.metadata.get("memory_key")
    if memory_key:
        same_key_memories = [
            memory
            for memory in existing_memories
            if (memory.get("metadata") or {}).get("memory_key") == memory_key
        ]
        if same_key_memories:
            primary = same_key_memories[0]
            primary_id = primary.get("id")
            merged_metadata = {
                **(primary.get("metadata") or {}),
                **dict(candidate.metadata),
                "updated_at": datetime.now().isoformat(),
            }

            if primary_id and hasattr(store, "update_memory"):
                try:
                    updated = store.update_memory(
                        memory_id=primary_id,
                        content=candidate.content,
                        metadata=merged_metadata,
                    )
                except Exception as exc:
                    logger.warning(
                        "[MemoryPolicy] Failed to update keyed long-term memory for handler=%s: %s",
                        handler_id,
                        exc,
                    )
                    updated = False
                if updated:
                    if hasattr(store, "delete_memory"):
                        for memory in same_key_memories[1:]:
                            duplicate_id = memory.get("id")
                            if duplicate_id and duplicate_id != primary_id:
                                try:
                                    store.delete_memory(duplicate_id)
                                except Exception:
                                    logger.debug(
                                        "[MemoryPolicy] Failed to delete duplicate keyed memory %s",
                                        duplicate_id,
                                    )
                    return str(primary_id)

    try:
        metadata = {
            **dict(candidate.metadata),
            "updated_at": datetime.now().isoformat(),
        }
        return store.add_memory(
            handler_id=handler_id,
            content=candidate.content,
            memory_type=candidate.memory_type,
            metadata=metadata,
            ttl_days=candidate.ttl_days,
        )
    except Exception as exc:
        logger.warning(
            "[MemoryPolicy] Failed to persist long-term memory for handler=%s: %s",
            handler_id,
            exc,
        )
        return None
