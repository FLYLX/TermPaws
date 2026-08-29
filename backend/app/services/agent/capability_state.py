"""Per-item registry of progressively loaded capabilities.

Tools and skill guides are exposed to the agent through a compact catalog.
When the agent calls `mcp_local_prepare_capabilities`, the requested tool
schemas / guide prompts are marked loaded for the item. Loaded tools are
further gated by a hot-LRU window: freshly loaded tools and recently used
tools stay, stale ones age out back to the catalog.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterable

HOT_TTL_SECONDS = 1800.0
HOT_MAX_TOOLS = 8
FRESH_MAX_TOOLS = 5

_LOCK = threading.Lock()
_STATE: dict[str, dict] = {}


def _entry(item_id: str) -> dict:
    key = str(item_id or "")
    with _LOCK:
        return _STATE.setdefault(
            key, {"tools": set(), "guides": set(), "usage": {}, "loaded_at": {}}
        )


def ensure_loaded(item_id: str, *, tools: Iterable[str] = (), guides: Iterable[str] = ()) -> None:
    if not item_id:
        return
    entry = _entry(item_id)
    now = time.time()
    with _LOCK:
        for name in tools:
            name = str(name)
            if not name.strip():
                continue
            entry["tools"].add(name)
            # 每次 prepare 都刷新新鲜度：agent 显式 prepare 就是"我现在要用"
            # 的声明。setdefault 会让老工具沉到 fresh 窗口末尾被挤出轮次
            # 工具表，模型调用时拿到 unknown tool，误判"调用无响应"而空转。
            entry["loaded_at"][name] = now
        entry["guides"].update(str(name) for name in guides if str(name).strip())


def record_tool_use(item_id: str, tool_name: str) -> None:
    if not item_id or not tool_name:
        return
    entry = _entry(item_id)
    with _LOCK:
        usage = entry["usage"].setdefault(str(tool_name), {"last_used": 0.0, "count": 0})
        usage["last_used"] = time.time()
        usage["count"] += 1


def get_loaded_tools(item_id: str) -> frozenset[str]:
    entry = _STATE.get(str(item_id or ""))
    if not entry:
        return frozenset()
    now = time.time()
    with _LOCK:
        loaded = entry["tools"]
        usage = entry["usage"]
        loaded_at = entry["loaded_at"]
        # fresh（未用过的）：按加载时间倒序，限 FRESH_MAX_TOOLS
        fresh = sorted(
            (name for name in loaded if name not in usage),
            key=lambda name: loaded_at.get(name, 0.0),
            reverse=True,
        )[:FRESH_MAX_TOOLS]
        # hot（30min 内用过）：按调用频次+最近使用排序，填满剩余名额
        hot = sorted(
            (
                name
                for name in loaded
                if name in usage and now - usage[name]["last_used"] <= HOT_TTL_SECONDS
            ),
            key=lambda name: (usage[name]["count"], usage[name]["last_used"]),
            reverse=True,
        )
    selected = fresh + [name for name in hot if name not in fresh]
    return frozenset(selected[:HOT_MAX_TOOLS])


def get_loaded_guides(item_id: str) -> frozenset[str]:
    with _LOCK:
        return frozenset(_STATE.get(str(item_id or ""), {}).get("guides", ()))


def reset_item(item_id: str) -> None:
    with _LOCK:
        _STATE.pop(str(item_id or ""), None)
