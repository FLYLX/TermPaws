"""环境事实沉淀：把 job 失败/成功自动转化为持久的环境约束。

背景：QQ 机器人轮次的上下文窗口只有 6 条消息，agent 在精简容器里
反复重试不存在的工具（curl/wget/unzip），每次重新发现都要烧掉一整
个 run_job 周期。本模块在 job 完成回调里自动沉淀两类事实：

1. 缺失工具黑名单：输出命中 "not found"/"not recognized" 且任务失败
   → 往 item.output_filter_rules 写一条 __env_missing_<tool> block
   规则（复用现有 OutputFilter 闸门，不新建拦截器）。规则 pattern
   带放行口：which/command -v/--version/包管理器安装命令不拦，避免
   环境变更后永远锁死。
2. 工具恢复：含该工具的命令在命令位置执行成功 → 删除对应规则。

所有事实的唯一真相源是 item.output_filter_rules（DB，跨会话持久），
会话级可用工具白名单由调用方（AgentSession）另行维护。
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

ENV_RULE_PREFIX = "__env_missing_"
MAX_ENV_RULES = 10

_NOT_FOUND_RES = (
    re.compile(r"([A-Za-z0-9_.+@-]+):\s*(?:command\s+)?not found", re.IGNORECASE),
    re.compile(r"'([^'\r\n]+)'\s+is not recognized as", re.IGNORECASE),
    re.compile(r"'([^'\r\n]+)'\s*不是内部或外部命令"),
)

_VERIFY_OR_INSTALL_RE = (
    r"command\s+-v|which\s|apt(?:-get)?\s+install|apk\s+add|dnf\s+install|"
    r"yum\s+install|pip3?\s+install|--version\b"
)

_COMMAND_POSITION_TEMPLATE = r"(?:^|[\n|;&]\s*){tool}(?=\s|$)"

_PROBE_RES = (
    re.compile(r"(?:^|[\n|;&]\s*)(?:which|command\s+-v)\s+([A-Za-z0-9_.+@-]+)", re.IGNORECASE),
    re.compile(r"(?:^|[\n|;&]\s*)([A-Za-z0-9_.+@-]+)\s+--version\b", re.IGNORECASE),
)

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.+@-]{1,40}$")


def extract_missing_tool(output: str) -> str:
    """从失败输出里提取缺失的工具名，提取不到返回空串。"""
    text = str(output or "")
    if not text.strip():
        return ""
    for pattern in _NOT_FOUND_RES:
        match = pattern.search(text)
        if not match:
            continue
        candidate = str(match.group(1) or "").strip()
        if _TOOL_NAME_RE.match(candidate):
            return candidate
    return ""


def extract_verified_tool(command: str) -> str:
    """从成功的探测命令里提取被验证可用的工具名。"""
    text = str(command or "")
    for pattern in _PROBE_RES:
        match = pattern.search(text)
        if match:
            candidate = str(match.group(1) or "").strip()
            if _TOOL_NAME_RE.match(candidate):
                return candidate
    return ""


def _missing_tool_pattern(tool: str) -> str:
    escaped = re.escape(tool)
    return (
        rf"(?s)^(?!.*(?:{_VERIFY_OR_INSTALL_RE})).*"
        rf"(?<![\w./-]){escaped}(?![\w.-])"
    )


def build_missing_tool_rule(tool: str, *, job_id: str = "", count: int = 1) -> dict[str, Any]:
    job_short = str(job_id or "")[:8] or "unknown"
    return {
        "regex_patterns": [_missing_tool_pattern(tool)],
        "action_type": "block",
        "reason": (
            f"{tool} 已实测不存在于该终端环境（命令失败，累计{count}次，"
            f"最近 job {job_short}）。不要再用 {tool} 发起命令；改用其他方案"
            f"（如 python3）。若环境可能已变更，先用 command -v {tool} 复验，"
            "执行成功系统会自动解除本拦截。"
        ),
        "count": count,
        "last_job_id": str(job_id or ""),
    }


def env_rule_tool(rule_name: str, rules: dict[str, Any]) -> str:
    """若规则名是环境黑名单规则，返回对应工具名，否则返回空串。"""
    if not isinstance(rule_name, str) or not rule_name.startswith(ENV_RULE_PREFIX):
        return ""
    tool = rule_name[len(ENV_RULE_PREFIX):]
    return tool if _TOOL_NAME_RE.match(tool) else ""


def list_missing_tools(rules: dict[str, Any]) -> list[dict[str, Any]]:
    """从规则表里列出所有环境黑名单条目（供 prompt 注入用）。"""
    entries: list[dict[str, Any]] = []
    for name, rule in (rules or {}).items():
        tool = env_rule_tool(name, rules)
        if not tool or not isinstance(rule, dict):
            continue
        entries.append({
            "tool": tool,
            "count": int(rule.get("count") or 1),
            "reason": str(rule.get("reason") or ""),
        })
    return entries


def _command_uses_tool(command: str, tool: str) -> bool:
    pattern = _COMMAND_POSITION_TEMPLATE.format(tool=re.escape(tool))
    return bool(re.search(pattern, str(command or "")))


def sediment_job_result(
    item_id: str,
    command: str,
    *,
    success: bool,
    exit_code: int | None,
    output_tail: str,
    job_id: str = "",
) -> dict[str, str]:
    """job 完成沉淀钩子：黑名单写入 / 恢复移除。绝不向调用方抛异常。"""
    summary = {"blacklisted": "", "recovered": "", "verified": ""}
    try:
        import uuid

        from sqlmodel import Session

        from app.core.db import engine
        from app.models import Item

        item_uuid = uuid.UUID(str(item_id))
    except Exception:
        return summary

    try:
        with Session(engine) as db:
            item = db.get(Item, item_uuid)
            if item is None:
                return summary
            rules = dict(item.output_filter_rules or {})
            changed = False

            if success and exit_code == 0:
                verified = extract_verified_tool(command)
                for name in list(rules):
                    tool = env_rule_tool(name, rules)
                    if tool and (_command_uses_tool(command, tool) or verified == tool):
                        rules.pop(name, None)
                        summary["recovered"] = tool
                        changed = True
                if verified:
                    summary["verified"] = verified
            else:
                missing = extract_missing_tool(output_tail)
                if missing:
                    name = f"{ENV_RULE_PREFIX}{missing}"
                    existing = rules.get(name)
                    count = 1
                    if isinstance(existing, dict):
                        count = int(existing.get("count") or 1) + 1
                    env_rules = [n for n in rules if env_rule_tool(n, rules)]
                    if existing is None and len(env_rules) >= MAX_ENV_RULES:
                        logger.info(
                            "[EnvFacts] env rule cap reached (%s), skip %s for item=%s",
                            MAX_ENV_RULES, missing, item_id,
                        )
                    else:
                        rules[name] = build_missing_tool_rule(
                            missing, job_id=job_id, count=count
                        )
                        summary["blacklisted"] = missing
                        changed = True

            if changed:
                if any(env_rule_tool(n, rules) for n in rules):
                    item.output_filter_enabled = True
                item.output_filter_rules = rules
                db.add(item)
                db.commit()
    except Exception as exc:
        logger.warning("[EnvFacts] sediment failed for item=%s: %s", item_id, exc)
    return summary
