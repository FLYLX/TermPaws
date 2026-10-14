from __future__ import annotations

from typing import Any, Literal

from app.services.agent.capability_state import (
    ensure_loaded,
    get_loaded_guides,
    get_loaded_tools,
)

TurnSource = Literal["web", "qq", "terminal"]

LOCAL_PREFIX = "mcp_local_"
ROBOT_PREFIX = "mcp_robot_"

PREPARE_CAPABILITIES_TOOL = "mcp_local_prepare_capabilities"

# Tools whose full schemas are always exposed: delivery, conversation memory,
# plan management, and the progressive-loading meta tool itself.
CORE_TOOL_NAMES = frozenset(
    {
        "mcp_robot_send_message",
        "mcp_robot_read_conversation_memory",
        "mcp_robot_save_memory",
        "mcp_robot_recall_memory",
        "mcp_robot_list_memories",
        "mcp_robot_compress_memories",
        "mcp_robot_sleep_conversation",
        "mcp_local_update_plan",
        PREPARE_CAPABILITIES_TOOL,
    }
)

TERMINAL_TOOLSET = frozenset(
    {
        "mcp_local_execute_command",
        "mcp_local_run_job",
        "mcp_local_list_jobs",
        "mcp_local_cancel_job",
        "mcp_local_interrupt_command",
        "mcp_local_get_terminal_status",
        "mcp_local_read_terminal_log",
        "mcp_local_add_terminal_input_filter_rule",
        "mcp_local_list_terminal_input_filter_rules",
        "mcp_local_list_terminal_filter_rules",
        "mcp_local_delete_terminal_input_filter_rule",
        "mcp_local_clear_terminal_input_filter_rules",
    }
)

# 干活主力工具（执行/后台任务/查状态/读日志/打断）：QQ/web 回合也直接挂载。
# 让它们走 prepare_capabilities 按需加载的代价是：每个任务轮先烧 1-2 次
# 迭代加载工具，flash 模型经常在加载完之前就耗尽迭代或走丢——简单任务
# 死在起跑线上。过滤规则管理类保持按需加载。
TERMINAL_WORK_TOOLSET = frozenset(
    {
        "mcp_local_execute_command",
        "mcp_local_run_job",
        "mcp_local_list_jobs",
        "mcp_local_cancel_job",
        "mcp_local_interrupt_command",
        "mcp_local_get_terminal_status",
        "mcp_local_read_terminal_log",
    }
)

TERMINAL_GUIDE_IDS = ("terminal_mcp", "minecraft_server_helper")

_GUIDE_EXCLUDED_CATEGORIES = {"system", "persona"}


def _tool_name(tool: dict[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def _agent_item_id(agent: Any) -> str:
    context = getattr(agent, "_context", None)
    if context is None:
        return ""
    return str(getattr(context, "item_id", "") or "")


def select_tools_for_turn(
    tools: list[dict[str, Any]],
    *,
    source: TurnSource,
    query: str = "",
    agent: Any = None,
    reply_ticket_id: str = "",
) -> list[dict[str, Any]]:
    """Progressive tool exposure: core tools + capabilities loaded for the item.

    No keyword matching. The agent sees a catalog of on-demand capabilities in
    the system prompt and loads what it needs via prepare_capabilities.
    Terminal-sourced turns auto-load the terminal toolset.
    """
    del query, reply_ticket_id
    item_id = _agent_item_id(agent)
    if source == "terminal" and item_id:
        ensure_loaded(item_id, tools=TERMINAL_TOOLSET, guides=TERMINAL_GUIDE_IDS)
    loaded = get_loaded_tools(item_id) if item_id else frozenset()
    if source == "terminal":
        loaded = loaded | TERMINAL_TOOLSET
    elif item_id:
        # QQ/web 回合直接挂载干活主力工具（见 TERMINAL_WORK_TOOLSET 注释）。
        loaded = loaded | TERMINAL_WORK_TOOLSET

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for tool in tools:
        name = _tool_name(tool)
        if not name or name in seen:
            continue
        if name in CORE_TOOL_NAMES or name in loaded:
            selected.append(tool)
            seen.add(name)
    return selected


def _guide_skills(agent: Any) -> list[Any]:
    get_skills = getattr(agent, "get_skills", None)
    if not callable(get_skills):
        return []
    guides: list[Any] = []
    try:
        skills = list(get_skills())
    except Exception:
        return []
    for skill in skills:
        category = str(getattr(skill, "category", "") or "")
        if category in _GUIDE_EXCLUDED_CATEGORIES:
            continue
        action = getattr(skill, "action", None)
        prompt = str(getattr(action, "prompt", "") or "").strip() if action else ""
        if not prompt:
            continue
        guides.append(skill)
    return guides


def loaded_guide_skills(agent: Any, item_id: str) -> list[Any]:
    loaded = get_loaded_guides(item_id)
    if not loaded:
        return []
    return [skill for skill in _guide_skills(agent) if str(getattr(skill, "skill_id", "")) in loaded]


def build_capability_catalog(agent: Any, *, source: TurnSource) -> str:
    """Compact catalog of on-demand tools and guides for the system prompt."""
    item_id = _agent_item_id(agent)
    try:
        all_tools = list(agent.get_tools_for_litellm())
    except Exception:
        all_tools = []
    catalog_tools = [
        tool
        for tool in all_tools
        if _tool_name(tool) and _tool_name(tool) not in CORE_TOOL_NAMES
    ]
    guides = _guide_skills(agent)
    if not catalog_tools and not guides:
        return ""

    lines = [
        "## On-demand capabilities (not loaded yet)",
        (
            f"To use one, call `{PREPARE_CAPABILITIES_TOOL}` with item_id and the "
            "exact names; full schemas/guides are returned and stay loaded for this item."
        ),
    ]
    if catalog_tools:
        lines.append("Tools:")
        for tool in sorted(catalog_tools, key=_tool_name):
            function = tool.get("function", {})
            description = str(function.get("description") or "").strip()
            description = " ".join(description.split())
            if len(description) > 60:
                description = description[:57] + "..."
            lines.append(f"- {_tool_name(tool)}: {description}")
    if guides:
        lines.append("Guides:")
        for skill in guides:
            skill_id = str(getattr(skill, "skill_id", "") or "")
            description = str(getattr(skill, "description", "") or "").strip()
            description = " ".join(description.split())
            if len(description) > 60:
                description = description[:57] + "..."
            lines.append(f"- {skill_id}: {description or getattr(skill, 'name', '')}")
    del source
    return "\n".join(lines)