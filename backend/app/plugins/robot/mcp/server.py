from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import threading
import uuid
from datetime import datetime, timedelta
from typing import Any

from app.plugins.robot.contracts import RobotReplyTarget
from app.plugins.robot.internal_trace import (
    compact_robot_visible_message_text,
    is_robot_internal_trace_text,
    normalize_robot_message_text,
    sanitize_robot_visible_text,
)
from app.plugins.robot.mcp.context import get_robot_mcp_context
from app.plugins.robot.memory_scope import (
    memory_content_is_question_like,
    memory_scope_for_content,
    speaker_global_key_from_context,
)
from app.plugins.robot.memory_scope import (
    memory_conversation_key as scoped_memory_conversation_key,
)
from app.plugins.robot.memory_scope import (
    memory_scope_rank as scoped_memory_scope_rank,
)
from app.plugins.robot.message_chunks import is_group_reply_target
from app.plugins.robot.reply_intent import is_no_reply_intent

logger = logging.getLogger(__name__)


DEFAULT_MEMORY_RECENT_LINES = 8
ACTIVE_CONTEXT_MEMORY_MAX_LINES = 12
GENERAL_MEMORY_MAX_LINES = 500
DEFAULT_LONG_TERM_MEMORY_LIST_RESULTS = 10
MIN_LONG_TERM_MEMORY_RELEVANCE = 0.32
MIN_LONG_TERM_MEMORY_QUERY_SCORE = 0.7
LONG_TERM_MEMORY_TYPES = {"fact", "preference", "error", "context"}
LONG_TERM_MEMORY_TYPE_ORDER = ("preference", "fact", "context", "error")
LONG_TERM_MEMORY_TYPE_RANK = {
    "preference": 5,
    "fact": 4,
    "context": 3,
    "error": 2,
}
ROBOT_DELIVERY_TRACK_TTL_SECONDS = 600
ROBOT_DELIVERY_TRACK_MAX_ENTRIES = 512


class RobotMCPServer:
    _delivered_targets_lock = threading.Lock()
    # delivery key -> (last update time, {(target_type, target_id): last sent text})
    _delivered_targets: dict[str, tuple[datetime, dict[tuple[str, str], str]]] = {}
    _turn_send_counts_lock = threading.Lock()
    # reply ticket -> (last update time, bubbles sent this turn)
    _turn_send_counts: dict[str, tuple[datetime, int]] = {}

    @classmethod
    def _bump_turn_send_count(cls, ticket_id: str, count: int) -> int:
        if not ticket_id:
            return count
        now = datetime.now()
        with cls._turn_send_counts_lock:
            stale = [
                key
                for key, (updated_at, _) in cls._turn_send_counts.items()
                if (now - updated_at).total_seconds() > 1800
            ]
            for key in stale:
                cls._turn_send_counts.pop(key, None)
            _updated, total = cls._turn_send_counts.get(ticket_id, (now, 0))
            total += count
            cls._turn_send_counts[ticket_id] = (now, total)
            return total

    def __init__(self) -> None:
        self._tools: dict[str, dict] = {}
        self._register_builtin_tools()

    @staticmethod
    def _memory_scope(item_id: str) -> str:
        """Long-term memories are keyed by handler id: every item driven by
        one ItemHandler (web item, QQ-bound items) shares its memory pool."""
        try:
            from app.services.agent.memory.scope import resolve_handler_id

            return resolve_handler_id(item_id)
        except Exception:
            return str(item_id or "")

    def _register_builtin_tools(self) -> None:
        self.register_tool(
            name="send_message",
            description=(
                "Send a QQ message through the TermPaws connector. Every call "
                "must carry an explicit destination: target_type+target_id, "
                "including replies to the current conversation (pass ITS id - "
                "there is no implicit default destination). To find destinations, "
                "use the roster in the context card or list_contacts; ids not "
                "seen there are rejected. "
                "Outside an active QQ "
                "context never infer destinations from chat history: use "
                "target_type+target_id only for explicit user-provided group/QQ "
                "numbers, or reply_to for a backend web chat conversation visibly "
                "present (ask the user if ambiguous). Deliver one message to "
                "several explicit targets via the targets array in a single call. "
                "Provide robot_id when multiple robots exist."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Only for a single complete short sentence. 多句内容"
                            "一律用 messages 数组，不要用 text 拼接/塞长段。"
                        ),
                    },
                    "messages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": (
                            "回复的主形态：每条是一句口语短句，作为独立气泡按顺序"
                            "逐条发出（像人打字，一句一条）。多句回复都用它；"
                            "句数没有上限，每句短即可。"
                        ),
                    },
                    "target_type": {
                        "type": "string",
                        "enum": ["group", "private"],
                        "description": (
                            "Optional in QQ-triggered context, required in "
                            "backend chat. 'group' or 'private'."
                        ),
                    },
                    "target_id": {
                        "type": "string",
                        "description": (
                            "Optional in QQ-triggered context, required in "
                            "backend chat. Group number or QQ number."
                        ),
                    },
                    "conversation": {
                        "type": "string",
                        "description": (
                            "Deprecated for sending; use target_type+target_id "
                            "for explicit targets outside the current QQ context."
                        ),
                    },
                    "reply_to": {
                        "type": "string",
                        "description": (
                            "Backend web chat only: sender name or group reference "
                            "uniquely matching a QQ conversation visible in the "
                            "current context. Never infer invisible targets."
                        ),
                    },
                    "robot_id": {
                        "type": "string",
                        "description": (
                            "Robot UUID. Omit when an active QQ context exists or "
                            "exactly one enabled robot is accessible."
                        ),
                    },
                    "broadcast": {
                        "type": "boolean",
                        "description": (
                            "Deprecated. Broadcast from prior chat history is blocked "
                            "to prevent replying to the wrong QQ conversation."
                        ),
                    },
                    "targets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "target_type": {
                                    "type": "string",
                                    "enum": ["group", "private"],
                                },
                                "target_id": {"type": "string"},
                            },
                            "required": ["target_type", "target_id"],
                        },
                        "minItems": 1,
                        "maxItems": 10,
                        "description": (
                            "Backend chat only: send the same text/messages once "
                            "to each listed target in this single call. Do not "
                            "combine with other target parameters. Result reports "
                            "one status line per target."
                        ),
                    },
                },
            },
            handler=self._send_message,
            skip_memory=True,
        )
        self.register_tool(
            name="read_conversation_memory",
            description=(
                "Read or search the conversation-local QQ .log memory. In an "
                "incoming QQ-triggered turn, call with no target arguments, and "
                "only when the user explicitly asks about earlier QQ context or "
                "the message cannot be answered without it - never just to decide "
                "whether to reply or to verify a send. In backend chat use a "
                "visible conversation/reply_to, or explicit target_type/target_id "
                "plus robot_id."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "conversation": {
                        "type": "string",
                        "description": (
                            "Optional key like 'group:123456' or 'private:654321'; "
                            "must match the current conversation in an active QQ "
                            "context."
                        ),
                    },
                    "reply_to": {
                        "type": "string",
                        "description": (
                            "Optional natural reference resolved against visible "
                            "QQ context in backend chat."
                        ),
                    },
                    "target_type": {
                        "type": "string",
                        "enum": ["group", "private"],
                        "description": "Optional explicit QQ target type.",
                    },
                    "target_id": {
                        "type": "string",
                        "description": "Optional explicit QQ group number or QQ number.",
                    },
                    "robot_id": {
                        "type": "string",
                        "description": "Optional robot UUID when no active QQ context exists.",
                    },
                    "lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 500,
                        "description": (
                            "Max recent or matching lines to return (default 8; "
                            "capped to 12 in active QQ context)."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional case-insensitive text filter.",
                    },
                },
            },
            handler=self._read_conversation_memory,
            skip_memory=True,
        )
        self.register_tool(
            name="list_memories",
            description=(
                "List durable long-term memories in the current QQ robot scope "
                "without a query. Use when the user asks what the bot remembers "
                "or to view all memories. Scoped to the current item and active "
                "robot/conversation/sender. Not raw chat history - use "
                "read_conversation_memory for exact recent lines."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Maximum memories to return. Default is 10.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": sorted(LONG_TERM_MEMORY_TYPES),
                        "description": "Optional memory type filter.",
                    },
                },
            },
            handler=self._list_memories,
            skip_memory=True,
        )
        self.register_tool(
            name="recall_memory",
            description=(
                "Search long-term memory for stable facts, preferences, errors, "
                "and reusable context (not raw .log). Use when the current "
                "message refers to a known preference/person/fact or needs "
                "durable context. Omit target arguments: results are scoped to "
                "the current item, preferring the current conversation. For a "
                "general list use list_memories."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query built from the current user message.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": sorted(LONG_TERM_MEMORY_TYPES),
                        "description": "Optional memory type filter.",
                    },
                },
                "required": ["query"],
            },
            handler=self._recall_memory,
            skip_memory=True,
        )
        self.register_tool(
            name="save_memory",
            description=(
                "Save concise durable information from the current QQ chat to "
                "long-term memory: explicit remember requests, stable "
                "names/nicknames, bot identity rules, durable preferences, "
                "relationships, reusable facts, recurring group context. "
                "Execution state belongs to the task queue. Never save trivial "
                "chatter, images, reactions, raw logs, or secrets. Write times "
                "as absolute dates (e.g. 'user said on 2026-09-02: trip in "
                "October'), never relative words like 'next month/yesterday' - "
                "future you won't know what day that was."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Concise normalized memory text to save.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": sorted(LONG_TERM_MEMORY_TYPES),
                        "description": "Memory type. Default is fact.",
                    },
                    "ttl_days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3650,
                        "description": "Retention days for fact/context. REQUIRED for time-bound facts (plans, trips, temporary states) - estimate it yourself (e.g. 'travels next month' -> 45). Permanent facts, preferences and errors need none.",
                    },
                },
                "required": ["content"],
            },
            handler=self._save_memory,
            skip_memory=True,
        )
        self.register_tool(
            name="compress_memories",
            description=(
                "Merge redundant, overlapping, or outdated long-term memories "
                "into one concise replacement: the new memory is saved first, "
                "then the listed old ones are deleted. Use on duplicates, stale "
                "versions of the same fact, or noisy chatter. Never merge "
                "different users' personal memories. Collect ids via "
                "list_memories/recall_memory first."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "description": (
                            "Ids of the memories to merge (full id or the "
                            "8-character prefix shown by list_memories)."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "Concise merged replacement memory text.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": sorted(LONG_TERM_MEMORY_TYPES),
                        "description": (
                            "Memory type. Default is the majority type of the "
                            "merged memories, falling back to fact."
                        ),
                    },
                    "ttl_days": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3650,
                        "description": "Optional retention days for fact/context. Preference and error memories are permanent.",
                    },
                },
                "required": ["memory_ids", "content"],
            },
            handler=self._compress_memories,
            skip_memory=True,
        )
        self.register_tool(
            name="sleep_conversation",
            description=(
                "Put the current QQ conversation to sleep, in an incoming "
                "QQ-triggered context only: the sender asks the bot to sleep/be "
                "quiet/stop replying, or a trigger=active_chat_window message is "
                "ordinary chatter not addressed to the bot. Closes the controller "
                "until someone mentions or replies to the bot."
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=self._sleep_conversation,
            skip_memory=True,
        )
        self.register_tool(
            name="list_contacts",
            description=(
                "List the groups and friends this QQ account is actually in "
                "(live from the platform), with names and ids. Use it to find "
                "the right target when the user asks you to contact another "
                "group/person ('去隔壁群问问'), or when you are unsure which "
                "conversation an id belongs to. Groups you can see here are "
                "yours to talk to: after listing, you may send_message to any "
                "of them with target_type+target_id, even from an active QQ "
                "turn."
            ),
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=self._list_contacts,
            skip_memory=True,
        )

    # robot_id -> (fetched_at, {(target_type, target_id), ...}, names);
    # targets the platform itself confirmed this account belongs to.
    # Seeing is owning: anything in the live contact list is a legitimate
    # send destination. Names feed the per-turn roster line so the agent
    # always knows which conversation an id belongs to without a tool call.
    _verified_contacts_lock = threading.Lock()
    _verified_contacts: dict[
        str, tuple[datetime, set[tuple[str, str]], dict[tuple[str, str], str]]
    ] = {}
    _VERIFIED_CONTACTS_TTL_SECONDS = 1800

    @classmethod
    def _remember_verified_contacts(
        cls,
        robot_id: str,
        targets: set[tuple[str, str]],
        names: dict[tuple[str, str], str] | None = None,
    ) -> None:
        if not robot_id or not targets:
            return
        with cls._verified_contacts_lock:
            cls._verified_contacts[robot_id] = (
                datetime.now(),
                set(targets),
                dict(names or {}),
            )

    @classmethod
    def roster_line(cls, robot_id: str, *, allow_fetch: bool = True) -> str:
        """One-line "我在哪些群/有哪些好友" for the turn context card.

        Served from the verified-contacts cache; on first use (or after
        expiry) fetches the live list once via the bridge so the roster is
        available without the agent remembering to call list_contacts.
        """
        robot_id = str(robot_id or "").strip()
        if not robot_id:
            return ""
        with cls._verified_contacts_lock:
            entry = cls._verified_contacts.get(robot_id)
            fresh = bool(
                entry
                and (datetime.now() - entry[0]).total_seconds()
                <= cls._VERIFIED_CONTACTS_TTL_SECONDS
            )
        if not fresh and allow_fetch:
            try:
                cls()._refresh_contacts(robot_id)
            except Exception:
                pass
            with cls._verified_contacts_lock:
                entry = cls._verified_contacts.get(robot_id)
        if not entry:
            return ""
        _fetched_at = entry[0]
        targets = entry[1]
        names = entry[2] if len(entry) > 2 else {}
        groups = sorted(
            (t for t in targets if t[0] == "group"),
            key=lambda t: names.get(t, ""),
        )
        privates = sorted(
            (t for t in targets if t[0] == "private"),
            key=lambda t: names.get(t, ""),
        )
        parts = []
        if groups:
            parts.append(
                "群: " + ", ".join(f"{names.get(t, '')}({t[1]})" for t in groups)
            )
        if privates:
            parts.append(
                "私聊: "
                + ", ".join(f"{names.get(t, '')}({t[1]})" for t in privates[:10])
            )
        return "; ".join(parts)

    @classmethod
    def _is_verified_contact(
        cls, robot_id: str, target_type: str, target_id: str
    ) -> bool:
        if not robot_id or not target_id:
            return False
        with cls._verified_contacts_lock:
            entry = cls._verified_contacts.get(robot_id)
            if not entry:
                return False
            fetched_at, targets = entry[0], entry[1]
            if (datetime.now() - fetched_at).total_seconds() > cls._VERIFIED_CONTACTS_TTL_SECONDS:
                cls._verified_contacts.pop(robot_id, None)
                return False
            return (target_type, target_id) in targets

    def _refresh_contacts(
        self, robot_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
        from app.plugins.robot.bridge_client import robot_bridge_client

        groups: list[dict[str, Any]] = []
        friends: list[dict[str, Any]] = []
        errors: list[str] = []
        try:
            group_resp = robot_bridge_client.call_api(robot_id, "get_group_list")
            if group_resp.get("success") and isinstance(group_resp.get("data"), list):
                groups = group_resp["data"]
            elif group_resp.get("error"):
                errors.append(f"get_group_list: {group_resp['error']}")
        except Exception as exc:
            errors.append(f"get_group_list: {exc}")
        try:
            friend_resp = robot_bridge_client.call_api(robot_id, "get_friend_list")
            if friend_resp.get("success") and isinstance(friend_resp.get("data"), list):
                friends = friend_resp["data"]
            elif friend_resp.get("error"):
                errors.append(f"get_friend_list: {friend_resp['error']}")
        except Exception as exc:
            errors.append(f"get_friend_list: {exc}")

        verified: set[tuple[str, str]] = set()
        names: dict[tuple[str, str], str] = {}
        for group in groups:
            group_id = str(group.get("group_id") or "").strip()
            if group_id:
                verified.add(("group", group_id))
                names[("group", group_id)] = str(group.get("group_name") or "").strip()
        for friend in friends:
            user_id = str(friend.get("user_id") or "").strip()
            if user_id:
                verified.add(("private", user_id))
                names[("private", user_id)] = str(
                    friend.get("nickname") or friend.get("remark") or ""
                ).strip()
        if verified:
            self._remember_verified_contacts(robot_id, verified, names)
        return groups, friends, errors

    def _list_contacts(self, args: dict) -> list[dict[str, str]]:
        context_token = str(args.get("_robot_context_token") or "").strip()
        context = get_robot_mcp_context(context_token)
        robot_id = str(getattr(context, "robot_id", "") or "").strip()
        if not robot_id:
            robot_id = str(args.get("_robot_id") or "").strip()
        if not robot_id:
            return [{"type": "text", "text": "Error: robot context unavailable"}]

        groups, friends, errors = self._refresh_contacts(robot_id)
        lines: list[str] = []
        if groups:
            lines.append(f"我所在的群（{len(groups)} 个）：")
            for group in groups:
                group_id = str(group.get("group_id") or "").strip()
                name = str(group.get("group_name") or "").strip() or "( unnamed )"
                if not group_id:
                    continue
                member_count = group.get("member_count")
                count_note = f"，{member_count} 人" if member_count else ""
                lines.append(f"- {name}（群号 {group_id}{count_note}）")
        if friends:
            lines.append(f"好友（{len(friends)} 个）：")
            for friend in friends[:50]:
                user_id = str(friend.get("user_id") or "").strip()
                name = str(
                    friend.get("nickname") or friend.get("remark") or ""
                ).strip() or "( unnamed )"
                if not user_id:
                    continue
                lines.append(f"- {name}（QQ {user_id}）")
        if not lines:
            detail = "; ".join(errors) if errors else "empty list"
            return [{"type": "text", "text": f"没拿到联系人列表：{detail}"}]

        lines.append("")
        lines.append(
            "以上都是这个账号真实所在的会话。注意：不带 target 的 send_message "
            "永远发回当前对话；要给上面列表里的其他群/好友发消息，那次调用必须显式带上 "
            "target_type（group/private）+ target_id（群号/QQ号），否则内容再对也会发错地方。"
            "每个群的“性格”从它的聊天内容里感受，列表只回答“我在哪”。"
        )
        if errors:
            lines.append(f"（部分查询失败：{'; '.join(errors)}）")
        return [{"type": "text", "text": "\n".join(lines)}]

    @classmethod
    def _target_label(cls, robot_id: str, target: Any) -> str:
        """'群名(群号)' for receipts, so the agent sees WHERE a send landed
        and can catch its own targeting mistakes immediately."""
        target_id = str(getattr(target, "target_id", "") or "").strip()
        target_type = str(getattr(target, "target_type", "") or "").strip()
        name = ""
        with cls._verified_contacts_lock:
            entry = cls._verified_contacts.get(str(robot_id or ""))
            if entry and len(entry) > 2:
                name = entry[2].get((target_type, target_id), "")
        return f"{name}({target_id})" if name else target_id

    def _normalize_target_type(self, value: Any) -> str:
        raw = str(value or "").strip().lower()
        if raw in {"group", "qq_group"}:
            return "group"
        if raw in {"private", "friend", "user", "direct", "c2c"}:
            return "private"
        return ""

    def _parse_conversation_target(self, value: Any) -> tuple[str, str]:
        raw = str(value or "").strip()
        if not raw or ":" not in raw:
            return "", ""
        target_type, target_id = raw.split(":", 1)
        return self._normalize_target_type(target_type), target_id.strip()

    def _context_reference(self, args: dict) -> str:
        return str(
            args.get("reply_to")
            or args.get("recipient")
            or args.get("conversation")
            or ""
        ).strip()

    def _build_explicit_target(self, args: dict) -> RobotReplyTarget | None:
        target_type = self._normalize_target_type(
            args.get("target_type") or args.get("mcp_target_type")
        )
        target_id = str(args.get("target_id") or args.get("mcp_target_id") or "").strip()
        if not target_type and not target_id:
            return None
        if not target_type:
            raise ValueError("target_type is required when target_id is provided")
        if not target_id:
            raise ValueError("target_id is required when target_type is provided")
        return RobotReplyTarget(
            target_type=target_type,
            target_id=target_id,
            metadata={"manual_target": True, "mcp_explicit_target": True},
        )

    def _build_explicit_targets(self, args: dict) -> list[RobotReplyTarget]:
        raw_targets = args.get("targets")
        if not isinstance(raw_targets, list) or not raw_targets:
            return []
        targets: list[RobotReplyTarget] = []
        seen: set[str] = set()
        for index, raw_target in enumerate(raw_targets):
            if not isinstance(raw_target, dict):
                raise ValueError(
                    f"targets[{index}] must be an object with "
                    "target_type and target_id"
                )
            target_type = self._normalize_target_type(raw_target.get("target_type"))
            target_id = str(raw_target.get("target_id") or "").strip()
            if target_type not in {"group", "private"} or not target_id:
                raise ValueError(
                    f"targets[{index}] requires target_type "
                    "('group' or 'private') and a non-empty target_id"
                )
            key = f"{target_type}:{target_id}"
            if key in seen:
                continue
            seen.add(key)
            targets.append(
                RobotReplyTarget(
                    target_type=target_type,
                    target_id=target_id,
                    metadata={"manual_target": True, "mcp_explicit_targets": True},
                )
            )
        return targets

    def _build_conversation_target(self, args: dict) -> RobotReplyTarget | None:
        target_type, target_id = self._parse_conversation_target(args.get("conversation"))
        if target_type not in {"group", "private"} or not target_id:
            return None
        return RobotReplyTarget(
            target_type=target_type,
            target_id=target_id,
            metadata={"manual_target": True, "mcp_explicit_conversation": True},
        )

    def _context_targets(self, args: dict) -> list[dict[str, str]]:
        raw_targets = args.get("_robot_known_targets")
        if not isinstance(raw_targets, list):
            return []

        targets: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw_target in raw_targets:
            if not isinstance(raw_target, dict):
                continue
            target_type = self._normalize_target_type(raw_target.get("target_type"))
            target_id = str(raw_target.get("target_id") or "").strip()
            if target_type not in {"group", "private"} or not target_id:
                continue
            key = f"{target_type}:{target_id}"
            if key in seen:
                continue
            seen.add(key)
            conversation = str(raw_target.get("conversation") or key).strip() or key
            targets.append(
                {
                    "conversation": conversation,
                    "target_type": target_type,
                    "target_id": target_id,
                    "sender": str(raw_target.get("sender") or "").strip(),
                    "robot_id": str(raw_target.get("robot_id") or "").strip(),
                }
            )
        return targets

    def _context_target_from_active_context(
        self,
        context: Any,
    ) -> dict[str, str] | None:
        if context is None:
            return None
        target = context.reply_target
        conversation_data = target.metadata.get("conversation")
        if isinstance(conversation_data, dict):
            target_type = self._normalize_target_type(
                conversation_data.get("target_type")
                or conversation_data.get("type")
                or conversation_data.get("conversation_type")
            )
            target_id = str(
                conversation_data.get("target_id")
                or conversation_data.get("id")
                or conversation_data.get("conversation_id")
                or ""
            ).strip()
            if target_type in {"group", "private"} and target_id:
                return {
                    "conversation": f"{target_type}:{target_id}",
                    "target_type": target_type,
                    "target_id": target_id,
                    "sender": "",
                    "robot_id": context.robot_id,
                }

        target_type = self._normalize_target_type(target.target_type)
        target_id = str(target.target_id or "").strip()
        target_data = target.metadata.get("target")
        if isinstance(target_data, dict):
            if bool(target_data.get("private")):
                target_type = "private"
            elif bool(target_data.get("channel")):
                return None
            target_id = str(
                target_data.get("group_id")
                or target_data.get("user_id")
                or target_data.get("parent_id")
                or target_data.get("id")
                or target_id
            ).strip()
        if target_type not in {"group", "private"} or not target_id:
            return None
        return {
            "conversation": f"{target_type}:{target_id}",
            "target_type": target_type,
            "target_id": target_id,
            "sender": "",
            "robot_id": context.robot_id,
        }

    def _broadcast_context_targets(
        self,
        *,
        args: dict,
        text: str,
        context: Any,
    ) -> tuple[list[dict[str, str]], int]:
        targets = self._context_targets(args)
        active_target = self._context_target_from_active_context(context)
        if active_target is not None:
            active_key = f"{active_target['target_type']}:{active_target['target_id']}"
            if all(
                f"{target['target_type']}:{target['target_id']}" != active_key
                for target in targets
            ):
                targets.insert(0, active_target)

        if not targets:
            raise ValueError(
                "broadcast requires at least one visible QQ conversation target"
            )

        from app.plugins.robot.bridge_client import robot_bridge_client

        sent: list[dict[str, str]] = []
        delivered: list[tuple[str, RobotReplyTarget]] = []
        for target_data in targets:
            target = RobotReplyTarget(
                target_type=target_data["target_type"],
                target_id=target_data["target_id"],
                metadata={
                    "manual_target": True,
                    "mcp_context_target": True,
                    "mcp_context_broadcast": True,
                },
            )
            fallback_robot_id = target_data.get("robot_id", "")
            if not fallback_robot_id and context is not None:
                fallback_robot_id = context.robot_id
            robot_id = self._get_accessible_robot_id(
                args,
                fallback_robot_id=fallback_robot_id,
            )
            self._record_tool_send_event(
                robot_id=robot_id,
                target=target,
                text=text,
                mode="context_broadcast",
                args=args,
            )
            robot_bridge_client.send_message(robot_id, target, text)
            delivered.append((robot_id, target))
            sent.append(
                {
                    "robot_id": robot_id,
                    "conversation": target_data["conversation"],
                    "target_type": target.target_type,
                    "target_id": target.target_id,
                }
            )
        for robot_id, target in delivered:
            self._remember_sent_message(robot_id, target, text, args=args)
        return sent

    def _target_match_score(self, target: dict[str, str], reference: str) -> int:
        normalized_reference = reference.strip().casefold()
        if not normalized_reference:
            return 0

        conversation = target.get("conversation", "")
        target_type = target.get("target_type", "")
        target_id = target.get("target_id", "")
        sender = target.get("sender", "")
        exact_candidates = {
            conversation,
            f"{target_type}:{target_id}",
            target_id,
            sender,
        }
        if normalized_reference in {candidate.casefold() for candidate in exact_candidates if candidate}:
            return 100

        partial_candidates = [
            conversation,
            f"{target_type}:{target_id}",
            f"{target_type} {target_id}",
            target_id,
            sender,
            target_type,
        ]
        return sum(
            1
            for candidate in partial_candidates
            if candidate
            and (
                normalized_reference in candidate.casefold()
                or candidate.casefold() in normalized_reference
            )
        )

    def _resolve_context_target(self, args: dict) -> tuple[RobotReplyTarget | None, str]:
        targets = self._context_targets(args)
        if not targets:
            return None, ""

        reference = self._context_reference(args)
        if not reference and len(targets) == 1:
            target_data = targets[0]
            return (
                RobotReplyTarget(
                    target_type=target_data["target_type"],
                    target_id=target_data["target_id"],
                    metadata={
                        "manual_target": True,
                        "mcp_context_target": True,
                    },
                ),
                target_data.get("robot_id", ""),
            )

        if reference:
            parsed_type, parsed_id = self._parse_conversation_target(reference)
            if parsed_type and parsed_id:
                reference = f"{parsed_type}:{parsed_id}"

            scored = [
                (self._target_match_score(target, reference), target)
                for target in targets
            ]
            matches = [target for score, target in scored if score > 0]
            if len(matches) == 1:
                target_data = matches[0]
                return (
                    RobotReplyTarget(
                        target_type=target_data["target_type"],
                        target_id=target_data["target_id"],
                        metadata={
                            "manual_target": True,
                            "mcp_context_target": True,
                            "mcp_context_reference": reference,
                        },
                    ),
                    target_data.get("robot_id", ""),
                )
            if len(matches) > 1:
                choices = ", ".join(
                    f"{target['conversation']} sender={target.get('sender') or '-'}"
                    for target in matches[:5]
                )
                raise ValueError(
                    "The reply target is ambiguous in context. Choose one of: "
                    f"{choices}"
                )
            choices = ", ".join(
                f"{target['conversation']} sender={target.get('sender') or '-'}"
                for target in targets[:5]
            )
            raise ValueError(
                "No QQ conversation matched reply_to/conversation "
                f"`{reference}`. Choose one of: {choices}"
            )

        if len(targets) > 1:
            choices = ", ".join(
                f"{target['conversation']} sender={target.get('sender') or '-'}"
                for target in targets[:5]
            )
            raise ValueError(
                "Multiple QQ conversations are available in context. Set reply_to "
                f"to one of: {choices}"
            )
        return None, ""

    def _same_target(self, left: RobotReplyTarget, right: dict[str, str]) -> bool:
        left_type = self._normalize_target_type(left.target_type)
        left_id = str(left.target_id or "").strip()
        left_data = left.metadata.get("conversation")
        if isinstance(left_data, dict):
            left_type = self._normalize_target_type(
                left_data.get("target_type")
                or left_data.get("type")
                or left_data.get("conversation_type")
                or left_type
            )
            left_id = str(
                left_data.get("target_id")
                or left_data.get("id")
                or left_data.get("conversation_id")
                or left_id
            ).strip()
        return left_type == right.get("target_type") and left_id == right.get("target_id")

    def _reference_matches_active_target(
        self,
        reference: str,
        active_target: dict[str, str],
    ) -> bool:
        normalized = reference.strip().casefold()
        if not normalized:
            return True
        candidates = {
            active_target.get("conversation", ""),
            f"{active_target.get('target_type', '')}:{active_target.get('target_id', '')}",
            active_target.get("target_id", ""),
        }
        return normalized in {candidate.casefold() for candidate in candidates if candidate}

    @staticmethod
    def _target_matches_known(
        target: RobotReplyTarget,
        known_targets: list[dict[str, str]],
    ) -> bool:
        target_id = str(target.target_id or "").strip()
        target_type = str(target.target_type or "").strip()
        for known in known_targets:
            if (
                str(known.get("target_id") or "").strip() == target_id
                and str(known.get("target_type") or "").strip() == target_type
            ):
                return True
        return False

    def _active_context_target_override_error(
        self,
        *,
        context: Any,
        explicit_target: RobotReplyTarget | None,
        broadcast: bool,
        context_reference: str,
        explicit_targets: bool = False,
        known_targets: list[dict[str, str]] | None = None,
        explicit_target_list: list[RobotReplyTarget] | None = None,
    ) -> str:
        if context is None:
            return ""

        active_target = self._context_target_from_active_context(context)
        active_label = (
            active_target["conversation"] if active_target is not None else "current"
        )
        robot_id = str(getattr(context, "robot_id", "") or "").strip()
        error = (
            "Error: target not recognized in this turn. To reply to the current "
            f"conversation ({active_label}), omit reply_to/conversation/target_type/"
            "target_id. To reach another group or friend, call list_contacts "
            "first and pick the target from your actual contact list - contacts "
            "you can see there are unlocked for sending. Broadcast is not allowed."
        )

        known = list(known_targets or [])

        def _allowed(target: RobotReplyTarget) -> bool:
            if self._target_matches_known(target, known):
                return True
            return self._is_verified_contact(
                robot_id,
                str(target.target_type or "").strip(),
                str(target.target_id or "").strip(),
            )

        # 合并轮次（多个 QQ 来源）时，模型可以自行决定给哪个来源回、回不回：
        # 显式目标落在本轮已知来源集合内即放行；广播仍然禁止（无法界定范围）。
        if broadcast:
            return error
        if explicit_targets:
            entries = list(explicit_target_list or [])
            if entries and all(_allowed(entry) for entry in entries):
                return ""
            return error
        if explicit_target is not None:
            if active_target is not None and self._same_target(
                explicit_target,
                active_target,
            ):
                return ""
            if _allowed(explicit_target):
                return ""
            return error
        if context_reference and (
            active_target is None
            or not self._reference_matches_active_target(context_reference, active_target)
        ):
            return error
        return ""

    def _active_context_delivery_error(self, context: Any) -> str:
        if context is None or not bool(getattr(context, "reply_requires_awake", False)):
            return ""
        try:
            conversation_generation = int(
                getattr(context, "conversation_generation", 0) or 0
            )
        except (TypeError, ValueError):
            conversation_generation = 0

        from app.plugins.robot.service import robot_service

        if robot_service.conversation_controller_allows_completion_reply(
            getattr(context, "robot_id", ""),
            getattr(context, "conversation_key", ""),
            conversation_generation,
            requires_awake=True,
        ):
            return ""
        return (
            "Send skipped: this turn was superseded by a newer QQ message, so the "
            "conversation belongs to a newer turn now. Do NOT retry mcp_robot_send_message "
            "and do NOT mention this failure to the user; the newer turn will deliver the "
            "reply. End this turn silently."
        )

    def _get_accessible_robot_id(
        self,
        args: dict,
        *,
        fallback_robot_id: str = "",
    ) -> str:
        explicit_robot_id = str(args.get("robot_id") or "").strip()
        if explicit_robot_id:
            return self._assert_robot_access(explicit_robot_id, args)
        if fallback_robot_id:
            return fallback_robot_id
        return self._single_accessible_robot_id(args)

    def _assert_robot_access(self, robot_id: str, args: dict) -> str:
        try:
            parsed_robot_id = uuid.UUID(robot_id)
        except ValueError as exc:
            raise ValueError("robot_id must be a valid UUID") from exc

        from sqlmodel import Session

        from app.core.db import engine
        from app.models import Robot

        with Session(engine) as session:
            robot = session.get(Robot, parsed_robot_id)
            if robot is None:
                raise ValueError(f"Robot not found: {robot_id}")
            if not robot.is_enabled:
                raise ValueError(f"Robot is disabled: {robot_id}")
            self._assert_user_can_use_robot(robot, args)
        return str(parsed_robot_id)

    def _single_accessible_robot_id(self, args: dict) -> str:
        from sqlmodel import Session, select

        from app.core.db import engine
        from app.models import Robot

        user_id = str(args.get("_TermPaws_user_id") or "").strip()
        is_superuser = bool(args.get("_TermPaws_is_superuser"))
        if not user_id and not is_superuser:
            raise ValueError(
                "robot_id is required outside an active robot conversation context"
            )

        with Session(engine) as session:
            statement = select(Robot).where(Robot.is_enabled == True)  # noqa: E712
            if not is_superuser:
                statement = statement.where(Robot.owner_id == uuid.UUID(user_id))
            robots = list(session.exec(statement).all())

        if len(robots) == 1:
            return str(robots[0].id)
        if not robots:
            raise ValueError("No enabled QQ robot is available for this user")

        choices = ", ".join(f"{robot.name} ({robot.id})" for robot in robots[:5])
        raise ValueError(
            "Multiple enabled QQ robots are available; provide robot_id. "
            f"Available robots: {choices}"
        )

    def _assert_user_can_use_robot(self, robot: Any, args: dict) -> None:
        if bool(args.get("_TermPaws_is_superuser")):
            return
        user_id = str(args.get("_TermPaws_user_id") or "").strip()
        if not user_id:
            raise ValueError("Not authorized to use this robot")
        if str(robot.owner_id) != user_id:
            raise ValueError("Not authorized to use this robot")

    def _record_tool_send_event(
        self,
        *,
        robot_id: str,
        target: RobotReplyTarget,
        text: str,
        mode: str,
        args: dict,
        status: str = "ok",
        error: str = "",
    ) -> None:
        from app.plugins.robot.debug_log import preview_text, record_robot_event

        payload = {
            "mode": mode,
            "target_type": target.target_type,
            "target_id": target.target_id,
            "reply_to": str(args.get("reply_to") or "").strip() or None,
            "conversation": str(args.get("conversation") or "").strip() or None,
            "has_context_token": bool(str(args.get("_robot_context_token") or "").strip()),
            "context_target_count": len(self._context_targets(args)),
        }
        if error:
            payload["error"] = error
        record_robot_event(
            robot_id,
            direction="agent_to_mcp",
            event="robot_tool_send_requested",
            status=status,
            message=preview_text(text),
            payload=payload,
        )

    def _remember_sent_message(
        self,
        robot_id: str,
        target: RobotReplyTarget,
        text: str,
        args: dict | None = None,
    ) -> None:
        try:
            from app.plugins.robot.conversation_memory import (
                conversation_key_from_reply_target,
                robot_conversation_memory,
            )

            robot_conversation_memory.append_assistant_message(
                robot_id,
                conversation_key_from_reply_target(target),
                text,
            )
        except Exception:
            logger.exception("[RobotMCPServer] Failed to write sent QQ memory")
        if args:
            self._track_delivered_target(args, target, text)

    def _remember_sent_messages(
        self,
        robot_id: str,
        target: RobotReplyTarget,
        messages: list[str],
        args: dict | None = None,
    ) -> None:
        for message in messages:
            self._remember_sent_message(robot_id, target, message, args=args)

    @staticmethod
    def delivery_track_key_from_args(args: dict | None) -> str:
        if not isinstance(args, dict):
            return ""
        ticket_id = str(args.get("_reply_ticket_id") or "").strip()
        if ticket_id:
            return f"ticket:{ticket_id}"
        item_id = str(args.get("item_id") or "").strip()
        if item_id:
            return f"item:{item_id}"
        return ""

    @classmethod
    def _prune_delivered_targets_locked(cls, now: datetime) -> None:
        cutoff = now - timedelta(seconds=ROBOT_DELIVERY_TRACK_TTL_SECONDS)
        stale_keys = [
            key
            for key, (updated_at, _) in cls._delivered_targets.items()
            if updated_at < cutoff
        ]
        for key in stale_keys:
            cls._delivered_targets.pop(key, None)
        while len(cls._delivered_targets) > ROBOT_DELIVERY_TRACK_MAX_ENTRIES:
            oldest_key = min(
                cls._delivered_targets,
                key=lambda key: cls._delivered_targets[key][0],
            )
            cls._delivered_targets.pop(oldest_key, None)

    @classmethod
    def mark_target_delivered(
        cls,
        key: str,
        target_type: str,
        target_id: str,
        text: str = "",
    ) -> None:
        key = str(key or "").strip()
        target_type = str(target_type or "").strip().lower()
        target_id = str(target_id or "").strip()
        if not key or not target_type or not target_id:
            return
        now = datetime.now()
        with cls._delivered_targets_lock:
            cls._prune_delivered_targets_locked(now)
            entry = cls._delivered_targets.setdefault(key, (now, {}))
            entry[1][(target_type, target_id)] = str(text or "")
            cls._delivered_targets[key] = (now, entry[1])

    @classmethod
    def _track_delivered_target(
        cls,
        args: dict,
        target: RobotReplyTarget,
        text: str,
    ) -> None:
        key = cls.delivery_track_key_from_args(args)
        if not key:
            return
        cls.mark_target_delivered(
            key,
            str(target.target_type or ""),
            str(target.target_id or ""),
            text,
        )

    @classmethod
    def get_delivered_targets(cls, key: str) -> set[tuple[str, str]]:
        key = str(key or "").strip()
        if not key:
            return set()
        now = datetime.now()
        with cls._delivered_targets_lock:
            cls._prune_delivered_targets_locked(now)
            entry = cls._delivered_targets.get(key)
            if entry is None:
                return set()
            return set(entry[1].keys())

    @classmethod
    def get_delivered_texts(cls, key: str) -> dict[tuple[str, str], str]:
        key = str(key or "").strip()
        if not key:
            return {}
        now = datetime.now()
        with cls._delivered_targets_lock:
            cls._prune_delivered_targets_locked(now)
            entry = cls._delivered_targets.get(key)
            if entry is None:
                return {}
            return dict(entry[1])

    @classmethod
    def reset_delivered_targets(cls, key: str) -> None:
        key = str(key or "").strip()
        if not key:
            return
        with cls._delivered_targets_lock:
            cls._delivered_targets.pop(key, None)

    @classmethod
    def _clear_delivered_targets_for_test(cls) -> None:
        with cls._delivered_targets_lock:
            cls._delivered_targets.clear()
        with RobotMCPServer._recent_sends_lock:
            RobotMCPServer._recent_sends.clear()

    # 完全相同的内容在短时间窗内的重复发送一律抑制——不是内容判断，
    # 是防"模型连续两轮调了同一个发送"的流控（和队列去重同类）。
    _RECENT_SEND_DUP_WINDOW_SECONDS = 30.0
    _recent_sends: dict[tuple[str, str, str], datetime] = {}
    _recent_sends_lock = threading.Lock()

    @staticmethod
    def _bubble_form_feedback(messages: list[str]) -> str:
        """长段单气泡的反馈提示：只反馈不拦截，agent 自己看着办。

        单条 text 超过 80 字且含 2+ 句标点，说明把多句挤进了一个气泡——
        告诉它下次用 messages 数组按句拆。消息本身已经发出去了。
        """
        text = str(messages[0] or "") if len(messages) == 1 else ""
        if not text:
            return ""
        sentence_marks = sum(text.count(ch) for ch in "。！？!？")
        if len(text) > 80 and sentence_marks >= 2:
            return (
                f"[形态提示] 你刚发的这条是 {len(text)} 字的长段，挤了 {sentence_marks} 句。"
                "一句一个气泡才是正常形态——下次这种内容用 messages 数组按句拆开发。"
            )
        return ""

    def _send_messages(
        self,
        bridge_client: Any,
        robot_id: str,
        target: RobotReplyTarget,
        messages: list[str],
    ) -> list[str]:
        sent_messages: list[str] = []
        now = datetime.now()
        with self._recent_sends_lock:
            for key, ts in list(self._recent_sends.items()):
                if (now - ts).total_seconds() > self._RECENT_SEND_DUP_WINDOW_SECONDS:
                    self._recent_sends.pop(key, None)
        for message in messages:
            normalized = str(message or "").strip()
            dup_key = (
                str(robot_id),
                str(target.target_type or ""),
                str(target.target_id or ""),
                normalized,
            )
            with self._recent_sends_lock:
                last_sent = self._recent_sends.get(dup_key)
                if (
                    normalized
                    and last_sent is not None
                    and (now - last_sent).total_seconds()
                    < self._RECENT_SEND_DUP_WINDOW_SECONDS
                ):
                    logger.info(
                        "[RobotMCPServer] Suppressed duplicate send to %s:%s: %r",
                        target.target_type,
                        target.target_id,
                        normalized[:40],
                    )
                    continue
            bridge_client.send_message(robot_id, target, message)
            # 发送成功才登记——失败的消息不算发过，允许重试。
            with self._recent_sends_lock:
                self._recent_sends[dup_key] = datetime.now()
            sent_messages.append(message)
        return sent_messages

    def _send_explicit_targets(
        self,
        *,
        robot_bridge_client: Any,
        robot_id: str,
        targets: list[RobotReplyTarget],
        messages: list[str],
        text: str,
        args: dict,
    ) -> list[dict[str, str]]:
        report_lines: list[str] = []
        delivered: list[tuple[RobotReplyTarget, list[str]]] = []
        for target in targets:
            label = f"{target.target_type}:{target.target_id}"
            self._record_tool_send_event(
                robot_id=robot_id,
                target=target,
                text=text,
                mode="explicit_targets",
                args=args,
            )
            try:
                sent_messages = self._send_messages(
                    robot_bridge_client,
                    robot_id,
                    target,
                    messages,
                )
            except Exception as exc:
                logger.warning(
                    "[RobotMCPServer] Failed to send robot message to %s: %s",
                    label,
                    exc,
                )
                self._record_tool_send_event(
                    robot_id=robot_id,
                    target=target,
                    text=text,
                    mode="explicit_targets",
                    args=args,
                    status="error",
                    error=str(exc),
                )
                report_lines.append(f"error: {label}: {exc}")
                continue
            delivered.append((target, sent_messages))
            report_lines.append(f"sent: {label}")
        for target, sent_messages in delivered:
            self._remember_sent_messages(robot_id, target, sent_messages, args=args)
        return [
            {
                "type": "text",
                "text": (
                    f"Multi-target QQ send to {len(targets)} target(s):\n"
                    + "\n".join(report_lines)
                ),
            }
        ]

    @staticmethod
    def _raw_message_texts(args: dict) -> list[str]:
        raw_messages = args.get("messages")
        if isinstance(raw_messages, list):
            return [normalize_robot_message_text(value) for value in raw_messages]
        raw_text = args.get("text")
        if raw_text is not None:
            return [normalize_robot_message_text(raw_text)]
        for fallback_key in ("content", "message", "body"):
            fallback_value = args.get(fallback_key)
            if fallback_value is not None:
                return [normalize_robot_message_text(fallback_value)]
        return []

    @staticmethod
    def _sanitize_outgoing_messages(raw_messages: list[str]) -> list[str]:
        messages: list[str] = []
        for raw_message in raw_messages:
            text = compact_robot_visible_message_text(
                sanitize_robot_visible_text(raw_message)
            )
            if text and not is_no_reply_intent(text) and not is_robot_internal_trace_text(text):
                messages.append(text)
        return messages

    @staticmethod
    def _uses_messages_array(args: dict) -> bool:
        return isinstance(args.get("messages"), list)

    @staticmethod
    def _context_allows_multiple_reply_messages(context: Any) -> bool:
        if context is None:
            return True
        reply_target = getattr(context, "reply_target", None)
        metadata = getattr(reply_target, "metadata", None)
        return bool(
            isinstance(metadata, dict)
            and metadata.get("allow_multiple_reply_messages")
        )

    @classmethod
    def _coalesce_current_context_messages(
        cls,
        context: Any,
        messages: list[str],
    ) -> list[str]:
        if len(messages) <= 1 or cls._context_allows_multiple_reply_messages(context):
            return messages

        unique_messages: list[str] = []
        seen: set[str] = set()
        for message in messages:
            normalized = message.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_messages.append(normalized)
        # 只做去重，不做合并：句数没有上限，agent 按语义决定发几条。
        # 之前 >3 条就用空格拼成一段，正是"长段之间有空格"的来源。
        return unique_messages

    @staticmethod
    def _memory_line_limit(
        args: dict,
        *,
        max_lines: int = GENERAL_MEMORY_MAX_LINES,
    ) -> int:
        raw_value = (
            args.get("lines")
            or args.get("limit")
            or DEFAULT_MEMORY_RECENT_LINES
        )
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = DEFAULT_MEMORY_RECENT_LINES
        return max(1, min(max_lines, value))

    @staticmethod
    def _filter_memory_lines(content: str, query: str, *, lines: int) -> str:
        normalized_query = query.casefold()
        matches = [
            line
            for line in content.splitlines()
            if normalized_query in line.casefold()
        ]
        return sanitize_robot_visible_text("\n".join(matches[-lines:]))

    def _active_context_memory_target_error(
        self,
        active_target: dict[str, str] | None,
    ) -> str:
        active_label = (
            active_target["conversation"] if active_target is not None else "current"
        )
        return (
            f"Error: target not recognized in this turn. Omit reply_to/"
            f"conversation/target_type/target_id to read the current "
            f"conversation ({active_label}); for another conversation, call "
            "list_contacts first - contacts you can see there are unlocked."
        )

    def _sleep_conversation(self, args: dict) -> list[dict[str, str]]:
        context_token = str(args.get("_robot_context_token") or "").strip()
        context = get_robot_mcp_context(context_token)
        if context is None:
            return [
                {
                    "type": "text",
                    "text": "Error: sleep_conversation requires an active QQ context.",
                }
            ]
        active_target = self._context_target_from_active_context(context)
        conversation_key = str(getattr(context, "conversation_key", "") or "").strip()
        if not conversation_key and active_target is not None:
            conversation_key = active_target["conversation"]
        if not conversation_key:
            return [
                {
                    "type": "text",
                    "text": "Error: current QQ conversation is unavailable.",
                }
            ]

        from app.plugins.robot.service import robot_service

        robot_service.sleep_conversation_controller(
            getattr(context, "robot_id", ""),
            conversation_key,
            reason="mcp_sleep_conversation",
        )
        return [
            {
                "type": "text",
                "text": "No QQ message sent: current conversation is sleeping.",
            }
        ]
    def _long_term_memory_list_limit(self, args: dict) -> int:
        raw_value = args.get("limit") or DEFAULT_LONG_TERM_MEMORY_LIST_RESULTS
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            value = DEFAULT_LONG_TERM_MEMORY_LIST_RESULTS
        return max(1, value)

    @staticmethod
    def _memory_distance(memory: dict[str, Any]) -> float:
        try:
            return float(memory.get("distance"))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _memory_similarity_label(memory: dict[str, Any]) -> str:
        distance = memory.get("distance")
        try:
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        except (TypeError, ValueError):
            return ""
        return f", score {similarity:.2f}"

    @staticmethod
    def _memory_metadata(memory: dict[str, Any]) -> dict[str, Any]:
        metadata = memory.get("metadata") or {}
        return metadata if isinstance(metadata, dict) else {}

    @classmethod
    def _memory_type(cls, memory: dict[str, Any]) -> str:
        return str(cls._memory_metadata(memory).get("memory_type") or "fact")

    @staticmethod
    def _parse_memory_datetime(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    @classmethod
    def _memory_timestamp(cls, memory: dict[str, Any]) -> float:
        metadata = cls._memory_metadata(memory)
        parsed = cls._parse_memory_datetime(
            metadata.get("updated_at") or metadata.get("created_at")
        )
        if parsed is None:
            return 0.0
        try:
            return parsed.timestamp()
        except OSError:
            return 0.0

    @classmethod
    def _memory_expired(cls, memory: dict[str, Any]) -> bool:
        if cls._memory_type(memory) in {"preference", "error"}:
            return False
        parsed = cls._parse_memory_datetime(
            cls._memory_metadata(memory).get("expires_at")
        )
        if parsed is None:
            return False
        now = datetime.now(parsed.tzinfo) if parsed.tzinfo else datetime.now()
        return parsed < now

    @classmethod
    def _memory_inactive(cls, memory: dict[str, Any]) -> bool:
        metadata = cls._memory_metadata(memory)
        memory_type = str(metadata.get("memory_type") or "")
        status = str(metadata.get("status") or "").lower()
        return memory_type == "task" and status == "completed"

    @staticmethod
    def _memory_tokens(value: str) -> set[str]:
        normalized = str(value or "").casefold()
        chinese = re.findall(r"[\u4e00-\u9fff]", normalized)
        tokens = set(chinese)
        tokens.update(
            "".join(chinese[index : index + 2])
            for index in range(max(len(chinese) - 1, 0))
        )
        tokens.update(re.findall(r"[a-z0-9][a-z0-9_.:/-]*", normalized))
        return {token for token in tokens if token}

    @staticmethod
    def _memory_content_key(memory: dict[str, Any]) -> str:
        content = str(memory.get("content") or "").casefold()
        return re.sub(r"\s+", " ", content).strip()

    @classmethod
    def _memory_query_score(cls, memory: dict[str, Any], query: str) -> float:
        query_tokens = cls._memory_tokens(query)
        if not query_tokens:
            return 0.0
        content = str(memory.get("content") or "")
        content_folded = content.casefold()
        score = 0.0
        normalized_query = query.casefold().strip()
        if normalized_query and normalized_query in content_folded:
            score += 3.0
        content_tokens = cls._memory_tokens(content)
        if content_tokens and query_tokens:
            score += (len(query_tokens & content_tokens) / len(query_tokens)) * 4.0
        metadata = cls._memory_metadata(memory)
        for key in ("memory_key", "speaker_key", "speaker_global_key", "conversation_key"):
            value = str(metadata.get(key) or "")
            if value and query.casefold() in value.casefold():
                score += 0.8
        return score

    @classmethod
    def _memory_recency_score(cls, memory: dict[str, Any]) -> float:
        timestamp = cls._memory_timestamp(memory)
        if timestamp <= 0:
            return 0.0
        try:
            age_days = max(0.0, (datetime.now().timestamp() - timestamp) / 86400)
        except OSError:
            return 0.0
        if age_days <= 7:
            return 0.7
        if age_days <= 30:
            return 0.4
        if age_days <= 90:
            return 0.2
        return 0.0

    @classmethod
    def _memory_hybrid_rank_score(
        cls,
        memory: dict[str, Any],
        *,
        query: str,
        scope_rank: int,
    ) -> float:
        metadata = cls._memory_metadata(memory)
        verified_bonus = 1.0 if metadata.get("verified") is True else 0.0
        vector_score = max(0.0, 1.0 - cls._memory_distance(memory))
        return (
            cls._memory_query_score(memory, query) * 12.0
            + vector_score * 10.0
            + scope_rank * 2.0
            + LONG_TERM_MEMORY_TYPE_RANK.get(cls._memory_type(memory), 1) * 0.5
            + verified_bonus * 2.0
            + cls._memory_recency_score(memory)
        )

    @classmethod
    def _memory_usable(cls, memory: dict[str, Any]) -> bool:
        return (
            isinstance(memory, dict)
            and bool(str(memory.get("content") or "").strip())
            and not cls._memory_expired(memory)
            and not cls._memory_inactive(memory)
        )

    @staticmethod
    def _memory_conversation_key(memory: dict[str, Any]) -> str:
        metadata = memory.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return scoped_memory_conversation_key(metadata)

    @staticmethod
    def _memory_scope_rank(
        memory: dict[str, Any],
        *,
        robot_id: str,
        conversation_key: str,
        speaker_global_key: str = "",
    ) -> int:
        return scoped_memory_scope_rank(
            memory,
            robot_id=robot_id,
            conversation_key=conversation_key,
            speaker_global_key=speaker_global_key,
        )

    def _collect_scoped_long_term_memory_candidates(
        self,
        *,
        store: Any,
        item_id: str,
        memory_type: str | None,
        robot_id: str,
        conversation_key: str,
        speaker_global_key: str,
    ) -> list[dict[str, Any]]:
        memory_types = (memory_type,) if memory_type else LONG_TERM_MEMORY_TYPE_ORDER
        candidates: list[dict[str, Any]] = []
        for current_type in memory_types:
            try:
                memories = store.get_all_memories(self._memory_scope(item_id), memory_type=current_type)
            except Exception as exc:
                logger.debug(
                    "[RobotMCP] Failed to load scoped memory candidates item=%s type=%s: %s",
                    item_id,
                    current_type,
                    exc,
                )
                continue
            has_context = bool(robot_id or conversation_key or speaker_global_key)
            for memory in memories:
                if not self._memory_usable(memory):
                    continue
                rank = self._memory_scope_rank(
                    memory,
                    robot_id=robot_id,
                    conversation_key=conversation_key,
                    speaker_global_key=speaker_global_key,
                )
                # 作用域过滤只在"有明确身份/会话上下文"时生效（区分不同群友）；
                # 没有任何上下文（web/后台）时，rank<0 不代表不该看——
                # 不做排除，否则 web 用户自己的记忆都会凭空消失。
                if rank < 0 and has_context:
                    continue
                candidates.append(memory)
        return candidates

    def _select_recalled_long_term_memories(
        self,
        *,
        query: str,
        limit: int | None,
        memories: list[dict[str, Any]],
        robot_id: str,
        conversation_key: str,
        speaker_global_key: str,
    ) -> list[dict[str, Any]]:
        by_key: dict[str, dict[str, Any]] = {}
        for memory in memories:
            if not self._memory_usable(memory):
                continue
            if memory_content_is_question_like(str(memory.get("content") or "")):
                continue
            memory_id = str(memory.get("id") or "").strip()
            key = memory_id or f"content:{self._memory_content_key(memory)}"
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = memory
                continue
            if existing.get("distance") is None and memory.get("distance") is not None:
                by_key[key] = {**existing, "distance": memory.get("distance")}

        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for memory in by_key.values():
            query_score = self._memory_query_score(memory, query)
            similarity = max(0.0, 1.0 - self._memory_distance(memory))
            if (
                query_score < MIN_LONG_TERM_MEMORY_QUERY_SCORE
                and similarity < MIN_LONG_TERM_MEMORY_RELEVANCE
            ):
                continue
            scope_rank = self._memory_scope_rank(
                memory,
                robot_id=robot_id,
                conversation_key=conversation_key,
                speaker_global_key=speaker_global_key,
            )
            # 同收集阶段：没有任何身份/会话上下文（web/后台）时不做作用域排除，
            # 否则 web 用户的 speaker 作用域记忆会凭空消失。
            has_context = bool(robot_id or conversation_key or speaker_global_key)
            if scope_rank < 0 and has_context:
                continue
            ranked.append(
                (
                    self._memory_hybrid_rank_score(
                        memory,
                        query=query,
                        scope_rank=scope_rank,
                    ),
                    str(memory.get("id") or ""),
                    memory,
                )
            )

        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected: list[dict[str, Any]] = []
        seen_content: set[str] = set()
        for _, _, memory in ranked:
            content_key = self._memory_content_key(memory)
            if not content_key or content_key in seen_content:
                continue
            seen_content.add(content_key)
            selected.append(memory)
            if limit is not None and len(selected) >= limit:
                break
        return selected

    def _save_memory(self, args: dict) -> list[dict[str, str]]:
        content = sanitize_robot_visible_text(str(args.get("content") or "")).strip()
        if not content:
            return [{"type": "text", "text": "Error: content required"}]

        memory_type = str(args.get("memory_type") or "fact").strip() or "fact"
        if memory_type not in LONG_TERM_MEMORY_TYPES:
            return [{"type": "text", "text": f"Error: invalid memory_type: {memory_type}"}]

        item_id = str(args.get("_TermPaws_item_id") or args.get("item_id") or "").strip()
        if not item_id:
            return [{"type": "text", "text": "Error: item_id unavailable"}]

        try:
            from app.services.agent.prompts import policy as memory_policy

            rejection = memory_policy.memory_content_rejection_reason(content)
            if rejection:
                return [
                    {
                        "type": "text",
                        "text": f"记忆未保存：{rejection}",
                    }
                ]

            default_ttl_days = memory_policy.resolve_memory_ttl_days(memory_type)
            raw_ttl_days = args.get("ttl_days")
            if default_ttl_days is None:
                ttl_days = None
            elif raw_ttl_days in (None, ""):
                ttl_days = default_ttl_days
            else:
                ttl_days = int(raw_ttl_days)
            if ttl_days is not None:
                ttl_days = max(1, min(3650, ttl_days))

            context_token = str(args.get("_robot_context_token") or "").strip()
            context = get_robot_mcp_context(context_token)
            active_target = self._context_target_from_active_context(context)
            conversation_key = ""
            robot_id = ""
            sender_key = ""
            if context is not None:
                robot_id = str(context.robot_id or "").strip()
                sender_key = str(context.sender_key or "").strip()
                conversation_key = str(
                    getattr(context, "conversation_key", "")
                    or (active_target or {}).get("conversation")
                    or ""
                ).strip()

            metadata: dict[str, Any] = {
                "type": "robot_agent_saved",
                "source": "qq_robot_agent",
                "verified": False,
                "content_hash": memory_policy._build_content_hash(content),
            }
            memory_key = memory_policy.infer_memory_key(content, memory_type)
            if memory_key:
                metadata["memory_key"] = memory_key
            if memory_type == "error":
                metadata["status"] = "active"
            if robot_id:
                metadata["robot_id"] = robot_id
            if conversation_key:
                metadata["robot_conversation_key"] = conversation_key
                metadata["conversation_key"] = conversation_key
            if sender_key:
                metadata["speaker_key"] = sender_key
            speaker_global_key = speaker_global_key_from_context(
                sender_key,
                getattr(context, "reply_target", None) if context is not None else None,
            )
            if speaker_global_key:
                metadata["speaker_global_key"] = speaker_global_key
            metadata["memory_scope"] = memory_scope_for_content(content, memory_type)

            candidate = memory_policy.MemoryCandidate(
                content=content,
                memory_type=memory_type,
                ttl_days=ttl_days,
                metadata=metadata,
            )

            from app.services.agent.memory.vector_store import vector_store

            memory_id = memory_policy.persist_memory_candidate(
                self._memory_scope(item_id),
                candidate,
                store=vector_store,
                source_item_id=item_id,
            )
        except Exception as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]

        if not memory_id:
            return [{"type": "text", "text": "Memory not saved: duplicate or rejected."}]
        return [
            {
                "type": "text",
                "text": f"Memory saved to TermPaws long-term memory (ID: {str(memory_id)[:8]}...).",
            }
        ]

    def _compress_memories(self, args: dict) -> list[dict[str, str]]:
        raw_ids = args.get("memory_ids")
        if not isinstance(raw_ids, list):
            return [{"type": "text", "text": "Error: memory_ids must be a list"}]
        requested_ids = [
            value for value in (str(item or "").strip() for item in raw_ids) if value
        ]
        if len(set(requested_ids)) < 2:
            return [
                {"type": "text", "text": "Error: provide at least 2 distinct memory ids"}
            ]

        content = sanitize_robot_visible_text(str(args.get("content") or "")).strip()
        if not content:
            return [{"type": "text", "text": "Error: content required"}]

        item_id = str(args.get("_TermPaws_item_id") or args.get("item_id") or "").strip()
        if not item_id:
            return [{"type": "text", "text": "Error: item_id unavailable"}]

        memory_type = str(args.get("memory_type") or "").strip()
        if memory_type and memory_type not in LONG_TERM_MEMORY_TYPES:
            return [{"type": "text", "text": f"Error: invalid memory_type: {memory_type}"}]

        from app.services.agent.prompts import policy as memory_policy

        rejection = memory_policy.memory_content_rejection_reason(content)
        if rejection:
            return [{"type": "text", "text": f"压缩结果未保存：{rejection}"}]

        context_token = str(args.get("_robot_context_token") or "").strip()
        context = get_robot_mcp_context(context_token)
        active_target = self._context_target_from_active_context(context)
        robot_id = ""
        conversation_key = ""
        sender_key = ""
        speaker_global_key = ""
        if context is not None:
            robot_id = str(context.robot_id or "").strip()
            sender_key = str(context.sender_key or "").strip()
            conversation_key = str(
                getattr(context, "conversation_key", "")
                or (active_target or {}).get("conversation")
                or ""
            ).strip()
            speaker_global_key = speaker_global_key_from_context(
                sender_key,
                getattr(context, "reply_target", None),
            )

        try:
            from app.plugins.robot.memory_migration import (
                ensure_legacy_robot_memories_upgraded,
            )
            from app.services.agent.memory.vector_store import vector_store
            from app.services.agent.prompts import policy as memory_policy

            ensure_legacy_robot_memories_upgraded(self._memory_scope(item_id), store=vector_store)
            all_memories = vector_store.get_all_memories(self._memory_scope(item_id))
        except Exception as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]

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
            return [
                {
                    "type": "text",
                    "text": f"Error: memory ids not found: {', '.join(unmatched)}",
                }
            ]
        if len(sources) < 2:
            return [
                {"type": "text", "text": "Error: provide at least 2 distinct memory ids"}
            ]

        for memory in sources:
            rank = scoped_memory_scope_rank(
                memory,
                robot_id=robot_id,
                conversation_key=conversation_key,
                speaker_global_key=speaker_global_key,
            )
            if rank < 0:
                return [
                    {
                        "type": "text",
                        "text": "Error: some memories are outside the current QQ scope.",
                    }
                ]

        if not memory_type:
            type_counts: dict[str, int] = {}
            for memory in sources:
                source_type = self._memory_type(memory)
                type_counts[source_type] = type_counts.get(source_type, 0) + 1
            memory_type = max(
                LONG_TERM_MEMORY_TYPE_ORDER,
                key=lambda candidate: type_counts.get(candidate, 0),
            )
            if memory_type not in LONG_TERM_MEMORY_TYPES:
                memory_type = "fact"

        default_ttl_days = memory_policy.resolve_memory_ttl_days(memory_type)
        raw_ttl_days = args.get("ttl_days")
        if default_ttl_days is None:
            ttl_days = None
        elif raw_ttl_days in (None, ""):
            ttl_days = default_ttl_days
        else:
            ttl_days = int(raw_ttl_days)
        if ttl_days is not None:
            ttl_days = max(1, min(3650, ttl_days))

        def _uniform_metadata_value(key: str) -> str:
            values = {
                str((memory.get("metadata") or {}).get(key) or "").strip()
                for memory in sources
            }
            values.discard("")
            return values.pop() if len(values) == 1 else ""

        metadata: dict[str, Any] = {
            "type": "robot_agent_saved",
            "source": "qq_robot_agent_compress",
            "verified": False,
            "content_hash": memory_policy._build_content_hash(content),
            "source_item_id": item_id,
        }
        memory_key = memory_policy.infer_memory_key(content, memory_type)
        if memory_key:
            metadata["memory_key"] = memory_key
        if memory_type == "error":
            metadata["status"] = "active"
        if robot_id:
            metadata["robot_id"] = robot_id
        inherited_conversation_key = _uniform_metadata_value(
            "conversation_key"
        ) or _uniform_metadata_value("robot_conversation_key")
        if inherited_conversation_key:
            metadata["robot_conversation_key"] = inherited_conversation_key
            metadata["conversation_key"] = inherited_conversation_key
        inherited_speaker_key = _uniform_metadata_value("speaker_key")
        if inherited_speaker_key:
            metadata["speaker_key"] = inherited_speaker_key
        inherited_speaker_global_key = _uniform_metadata_value("speaker_global_key")
        if inherited_speaker_global_key:
            metadata["speaker_global_key"] = inherited_speaker_global_key
        metadata["memory_scope"] = memory_scope_for_content(content, memory_type)

        try:
            new_memory_id = vector_store.add_memory(
                handler_id=self._memory_scope(item_id),
                content=content,
                memory_type=memory_type,
                metadata=metadata,
                ttl_days=ttl_days,
                allow_duplicate=True,
                run_maintenance=False,
            )
        except Exception as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]
        if not new_memory_id:
            return [
                {
                    "type": "text",
                    "text": "Error: compressed memory was not saved; old memories left untouched.",
                }
            ]

        deleted = 0
        for memory in sources:
            try:
                if vector_store.delete_memory(str(memory.get("id") or "")):
                    deleted += 1
            except Exception:
                continue

        return [
            {
                "type": "text",
                "text": (
                    f"Compressed {len(sources)} memories into one "
                    f"(ID: {str(new_memory_id)[:8]}...); deleted {deleted} old memories."
                ),
            }
        ]

    def _list_memories(self, args: dict) -> list[dict[str, str]]:
        memory_type = str(args.get("memory_type") or "").strip() or None
        if memory_type is not None and memory_type not in LONG_TERM_MEMORY_TYPES:
            return [{"type": "text", "text": f"Error: invalid memory_type: {memory_type}"}]

        item_id = str(args.get("_TermPaws_item_id") or args.get("item_id") or "").strip()
        if not item_id:
            return [{"type": "text", "text": "Error: item_id unavailable"}]

        context_token = str(args.get("_robot_context_token") or "").strip()
        context = get_robot_mcp_context(context_token)
        if context is None:
            return [
                {
                    "type": "text",
                    "text": "Error: list_memories requires an active QQ robot conversation context.",
                }
            ]

        robot_id = str(context.robot_id or "").strip()
        active_target = self._context_target_from_active_context(context)
        conversation_key = str(
            getattr(context, "conversation_key", "")
            or (active_target or {}).get("conversation", "")
        ).strip()
        speaker_global_key = speaker_global_key_from_context(
            str(getattr(context, "sender_key", "") or ""),
            getattr(context, "reply_target", None),
        )

        try:
            from app.plugins.robot.memory_migration import (
                ensure_legacy_robot_memories_upgraded,
            )
            from app.services.agent.memory.vector_store import vector_store

            ensure_legacy_robot_memories_upgraded(
                self._memory_scope(item_id), store=vector_store
            )
            vector_store.maintain_memories(self._memory_scope(item_id))
            memories = self._collect_scoped_long_term_memory_candidates(
                store=vector_store,
                item_id=item_id,
                memory_type=memory_type,
                robot_id=robot_id,
                conversation_key=conversation_key,
                speaker_global_key=speaker_global_key,
            )
        except Exception as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]

        by_key: dict[str, dict[str, Any]] = {}
        for memory in memories:
            memory_id = str(memory.get("id") or "").strip()
            content = str(memory.get("content") or "").strip()
            key = memory_id or f"content:{content}"
            by_key.setdefault(key, memory)

        ranked = sorted(
            by_key.values(),
            key=lambda memory: (
                self._memory_scope_rank(
                    memory,
                    robot_id=robot_id,
                    conversation_key=conversation_key,
                    speaker_global_key=speaker_global_key,
                ),
                self._memory_timestamp(memory),
                LONG_TERM_MEMORY_TYPE_RANK.get(self._memory_type(memory), 1),
                str(memory.get("id") or ""),
            ),
            reverse=True,
        )
        limit = self._long_term_memory_list_limit(args)
        selected = ranked[:limit]
        if not selected:
            return [
                {
                    "type": "text",
                    "text": "No long-term memories are visible in the current QQ scope.",
                }
            ]

        lines = [
            f"Long-term memories visible in current QQ scope (showing {len(selected)} of {len(ranked)}):"
        ]
        for index, memory in enumerate(selected, start=1):
            metadata = self._memory_metadata(memory)
            sender = str(
                metadata.get("speaker")
                or metadata.get("speaker_label")
                or metadata.get("sender")
                or ""
            ).strip()
            content = str(memory.get("content") or "").strip()
            prefix = f"{sender}: " if sender and not content.startswith(sender) else ""
            memory_id = str(memory.get("id") or "")
            created = str(
                metadata.get("created_at") or metadata.get("updated_at") or ""
            )[:10]
            date_tag = f" {created}" if created else ""
            lines.append(
                f"{index}. [{memory_id[:8]}{date_tag}] [{self._memory_type(memory)}] {prefix}{content}"
            )
        try:
            clusters = vector_store.find_duplicate_clusters(self._memory_scope(item_id))
        except Exception:
            clusters = []
        if clusters:
            lines.append("")
            lines.append(
                "疑似重复（同一件事的多种说法）——是否合并由你判断，确认后用 compress_memories："
            )
            for cluster in clusters:
                preview = " ≈ ".join(
                    f"[{str(m.get('id') or '')[:8]}] {str(m.get('content') or '')[:30]}"
                    for m in cluster[:4]
                )
                lines.append(f"- {preview}")
        return [{"type": "text", "text": "\n".join(lines)}]

    def _recall_memory(self, args: dict) -> list[dict[str, str]]:
        query = str(args.get("query") or "").strip()
        if not query:
            return [{"type": "text", "text": "Error: query required"}]

        memory_type = str(args.get("memory_type") or "").strip() or None
        if memory_type is not None and memory_type not in LONG_TERM_MEMORY_TYPES:
            return [{"type": "text", "text": f"Error: invalid memory_type: {memory_type}"}]

        item_id = str(args.get("_TermPaws_item_id") or args.get("item_id") or "").strip()
        if not item_id:
            return [{"type": "text", "text": "Error: item_id unavailable"}]

        context_token = str(args.get("_robot_context_token") or "").strip()
        context = get_robot_mcp_context(context_token)
        active_target = self._context_target_from_active_context(context)
        conversation_key = ""
        robot_id = ""
        speaker_global_key = ""
        if context is not None:
            robot_id = str(context.robot_id or "").strip()
            conversation_key = str(
                getattr(context, "conversation_key", "")
                or (active_target or {}).get("conversation")
                or ""
            ).strip()
            speaker_global_key = speaker_global_key_from_context(
                str(getattr(context, "sender_key", "") or ""),
                getattr(context, "reply_target", None),
            )

        try:
            from app.plugins.robot.memory_migration import (
                ensure_legacy_robot_memories_upgraded,
            )
            from app.services.agent.memory.vector_store import vector_store

            ensure_legacy_robot_memories_upgraded(
                self._memory_scope(item_id), store=vector_store
            )
            scoped_memories = self._collect_scoped_long_term_memory_candidates(
                store=vector_store,
                item_id=item_id,
                memory_type=memory_type,
                robot_id=robot_id,
                conversation_key=conversation_key,
                speaker_global_key=speaker_global_key,
            )
            vector_memories = vector_store.search_memories(
                handler_id=self._memory_scope(item_id),
                query=query,
                n_results=max(1, len(scoped_memories)),
                memory_type=memory_type,
                include_expired=False,
                active_only=True,
                min_similarity=MIN_LONG_TERM_MEMORY_RELEVANCE,
                candidate_multiplier=1,
            )
        except Exception as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]

        selected = self._select_recalled_long_term_memories(
            query=query,
            limit=None,
            memories=[*vector_memories, *scoped_memories],
            robot_id=robot_id,
            conversation_key=conversation_key,
            speaker_global_key=speaker_global_key,
        )
        if not selected:
            scope = f" for {conversation_key}" if conversation_key else ""
            return [
                {
                    "type": "text",
                    "text": f"No TermPaws long-term memory matched {query!r}{scope}.",
                }
            ]

        lines = ["Relevant long-term memory:"]
        for memory in selected:
            metadata = memory.get("metadata") or {}
            content = sanitize_robot_visible_text(str(memory.get("content") or "")).strip()
            if not content:
                continue
            sender = str(
                metadata.get("speaker")
                or metadata.get("speaker_label")
                or metadata.get("sender")
                or ""
            ).strip()
            prefix = f"{sender}: " if sender and not content.startswith(sender) else ""
            memory_id = str(memory.get("id") or "")
            created = str(
                metadata.get("created_at") or metadata.get("updated_at") or ""
            )[:10]
            date_tag = f" {created}" if created else ""
            lines.append(f"- [{memory_id[:8]}{date_tag}] {prefix}{content}")
        return [{"type": "text", "text": "\n".join(lines)}]

    def _read_conversation_memory(self, args: dict) -> list[dict[str, str]]:
        context_token = str(args.get("_robot_context_token") or "").strip()
        context = get_robot_mcp_context(context_token)
        target_args = args
        if context is not None:
            has_target_type = bool(
                str(args.get("target_type") or args.get("mcp_target_type") or "").strip()
            )
            has_target_id = bool(
                str(args.get("target_id") or args.get("mcp_target_id") or "").strip()
            )
            if has_target_type != has_target_id:
                target_args = dict(args)
                for key in (
                    "target_type",
                    "target_id",
                    "mcp_target_type",
                    "mcp_target_id",
                ):
                    target_args.pop(key, None)
        try:
            explicit_target = self._build_explicit_target(target_args)
            conversation_target = self._build_conversation_target(target_args)
        except Exception as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]

        target: RobotReplyTarget | None = None
        fallback_robot_id = ""
        if context is not None:
            active_target = self._context_target_from_active_context(context)
            if active_target is None:
                return [
                    {
                        "type": "text",
                        "text": (
                            "Error: current QQ conversation memory is unavailable "
                            "for this target type."
                        ),
                    }
                ]
            requested_target = explicit_target or conversation_target
            if requested_target is not None and not self._same_target(
                requested_target,
                active_target,
            ):
                # 读过别的会话的记忆是无害操作：目标只要是本轮见过的来源或
                # list_contacts 确认过的真实联系人就放行。
                robot_id_for_check = str(getattr(context, "robot_id", "") or "").strip()
                known = args.get("_robot_known_targets") or []
                if not (
                    self._target_matches_known(requested_target, known)
                    or self._is_verified_contact(
                        robot_id_for_check,
                        str(requested_target.target_type or "").strip(),
                        str(requested_target.target_id or "").strip(),
                    )
                ):
                    return [
                        {
                            "type": "text",
                            "text": self._active_context_memory_target_error(active_target),
                        }
                    ]
            context_reference = str(
                args.get("reply_to") or args.get("conversation") or ""
            ).strip()
            if context_reference and not self._reference_matches_active_target(
                context_reference,
                active_target,
            ):
                return [
                    {
                        "type": "text",
                        "text": self._active_context_memory_target_error(active_target),
                    }
                ]
            robot_id = context.robot_id
            conversation_key = active_target["conversation"]
        else:
            if explicit_target is not None:
                target = explicit_target
            else:
                try:
                    target, fallback_robot_id = self._resolve_context_target(args)
                except Exception as exc:
                    if conversation_target is None:
                        return [{"type": "text", "text": f"Error: {exc}"}]
                    target = conversation_target
                if target is None and conversation_target is not None:
                    target = conversation_target
            if target is None:
                return [
                    {
                        "type": "text",
                        "text": (
                            "Error: no active robot conversation context and no "
                            "matching QQ conversation memory target. Use reply_to "
                            "or conversation from visible context, or provide "
                            "target_type, target_id, and robot_id."
                        ),
                    }
                ]
            try:
                robot_id = self._get_accessible_robot_id(
                    args,
                    fallback_robot_id=fallback_robot_id,
                )
            except Exception as exc:
                return [{"type": "text", "text": f"Error: {exc}"}]
            from app.plugins.robot.conversation_memory import (
                conversation_key_from_reply_target,
            )

            conversation_key = conversation_key_from_reply_target(target)

        line_cap = (
            ACTIVE_CONTEXT_MEMORY_MAX_LINES
            if context is not None
            else GENERAL_MEMORY_MAX_LINES
        )
        lines = self._memory_line_limit(args, max_lines=line_cap)
        query = str(args.get("query") or "").strip()
        from app.plugins.robot.conversation_memory import (
            normalize_conversation_key,
            robot_conversation_memory,
        )

        conversation_key = normalize_conversation_key(conversation_key)
        try:
            if query:
                memory = self._filter_memory_lines(
                    robot_conversation_memory.read(robot_id, conversation_key),
                    query,
                    lines=lines,
                )
                scope = f"matching query {query!r}"
            else:
                if context is not None:
                    from app.plugins.robot.conversation_memory import (
                        recent_dialogue_scan_lines,
                        select_recent_dialogue_lines,
                    )

                    raw_memory = sanitize_robot_visible_text(
                        robot_conversation_memory.read_recent(
                            robot_id,
                            conversation_key,
                            lines=recent_dialogue_scan_lines(lines),
                        )
                    )
                    memory = "\n".join(
                        select_recent_dialogue_lines(raw_memory, lines=lines)
                    )
                else:
                    memory = robot_conversation_memory.read_recent(
                        robot_id,
                        conversation_key,
                        lines=lines,
                    )
                scope = f"recent {lines} line(s)"
        except Exception as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]

        memory = sanitize_robot_visible_text(memory)
        if not memory.strip():
            if query:
                return [
                    {
                        "type": "text",
                        "text": (
                            "No QQ conversation .log memory matched "
                            f"{query!r} for {conversation_key}."
                        ),
                    }
                ]
            return [
                {
                    "type": "text",
                    "text": (
                        "No QQ conversation .log memory found for "
                        f"{conversation_key}."
                    ),
                }
            ]
        return [
            {
                "type": "text",
                "text": (
                    "QQ conversation .log memory "
                    f"({conversation_key}, {scope}):\n{memory}"
                ),
            }
        ]

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: dict,
        handler: callable,
        skip_memory: bool = False,
    ) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler,
            "skip_memory": skip_memory,
        }

    def _send_message(self, args: dict) -> list[dict[str, str]]:
        raw_messages = self._raw_message_texts(args)
        messages = self._sanitize_outgoing_messages(raw_messages)
        text = "\n".join(messages).strip()
        raw_text = "\n".join(str(value or "") for value in raw_messages).strip()
        if not messages:
            if any(is_robot_internal_trace_text(str(value or "")) for value in raw_messages):
                return [
                    {
                        "type": "text",
                        "text": "No QQ message sent: internal tool trace.",
                    }
                ]
            if any(is_no_reply_intent(str(value or "")) for value in raw_messages):
                return [
                    {
                        "type": "text",
                        "text": "No QQ message sent: no reply needed.",
                    }
                ]
            received_keys = [k for k in args if k not in ("_robot_context_token",) and args[k] is not None]
            raw_preview = raw_text[:80] if raw_text else "(empty)"
            return [{"type": "text", "text": (
                f"Error: text required. Your message was empty after sanitization. "
                f"Received fields: {received_keys}. Raw preview: {raw_preview}. "
                f"Provide a plain user-facing summary in the 'text' field. "
                f"Do NOT include tool execution traces, terminal output, or internal markers."
            )}]
        if is_no_reply_intent(raw_text) and raw_text == text:
            return [
                {
                    "type": "text",
                    "text": "No QQ message sent: no reply needed.",
                }
            ]
        if any(is_robot_internal_trace_text(message) for message in messages):
            return [
                {
                    "type": "text",
                    "text": "No QQ message sent: internal tool trace.",
                }
            ]

        context_token = str(args.get("_robot_context_token") or "").strip()
        context = get_robot_mcp_context(context_token)
        messages = self._coalesce_current_context_messages(context, messages)
        text = "\n".join(messages).strip()
        if context is not None:
            has_explicit_target_type = bool(
                self._normalize_target_type(
                    args.get("target_type") or args.get("mcp_target_type")
                )
            )
            has_explicit_target_id = bool(
                str(
                    args.get("target_id") or args.get("mcp_target_id") or ""
                ).strip()
            )
            if has_explicit_target_type != has_explicit_target_id:
                for key in (
                    "target_type",
                    "target_id",
                    "mcp_target_type",
                    "mcp_target_id",
                ):
                    args.pop(key, None)
        try:
            explicit_target = self._build_explicit_target(args)
        except Exception as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]
        try:
            explicit_targets = self._build_explicit_targets(args)
        except Exception as exc:
            return [{"type": "text", "text": f"Error: {exc}"}]
        if explicit_targets:
            conflicting_keys = [
                key
                for key in (
                    "reply_to",
                    "recipient",
                    "conversation",
                    "target_type",
                    "target_id",
                    "mcp_target_type",
                    "mcp_target_id",
                    "broadcast",
                )
                if args.get(key)
            ]
            if conflicting_keys:
                return [
                    {
                        "type": "text",
                        "text": (
                            "Error: targets cannot be combined with "
                            f"{', '.join(conflicting_keys)}. Use targets alone "
                            "for a deterministic multi-target send."
                        ),
                    }
                ]
        broadcast = bool(args.get("broadcast"))
        if broadcast and explicit_target is not None:
            return [
                {
                    "type": "text",
                    "text": (
                        "Error: broadcast uses QQ conversations from context; "
                        "do not combine it with target_type/target_id."
                    ),
                }
            ]
        if context is None and broadcast:
            return [
                {
                    "type": "text",
                    "text": (
                        "Error: broadcast from prior QQ context is disabled. "
                        "Use target_type and target_id for one explicit QQ target."
                    ),
                }
            ]

        context_target: RobotReplyTarget | None = None
        context_target_robot_id = ""
        context_reference = self._context_reference(args)
        active_context_override_error = self._active_context_target_override_error(
            context=context,
            explicit_target=explicit_target,
            broadcast=broadcast,
            context_reference=context_reference,
            explicit_targets=bool(explicit_targets),
            known_targets=self._context_targets(args),
            explicit_target_list=explicit_targets,
        )
        if active_context_override_error:
            return [{"type": "text", "text": active_context_override_error}]
        active_context_delivery_error = self._active_context_delivery_error(context)
        if active_context_delivery_error:
            return [{"type": "text", "text": active_context_delivery_error}]
        if context is not None:
            context_reference = ""

        if (
            explicit_target is None
            and not broadcast
            and context is None
            and context_reference
        ):
            try:
                context_target, context_target_robot_id = self._resolve_context_target(args)
            except Exception as exc:
                return [{"type": "text", "text": f"Error: {exc}"}]

        if (
            context is None
            and explicit_target is None
            and context_target is None
            and not broadcast
            and not explicit_targets
        ):
            return [
                {
                    "type": "text",
                    "text": (
                        "Error: no active QQ robot conversation context. "
                        "Provide target_type and target_id to send directly "
                        "(e.g. from reply ticket source info)."
                    ),
                }
            ]

        attempted_robot_id = ""
        attempted_target: RobotReplyTarget | None = None
        attempted_mode = ""
        try:
            from app.plugins.robot.bridge_client import robot_bridge_client

            if explicit_targets:
                robot_id = self._get_accessible_robot_id(
                    args,
                    fallback_robot_id=context.robot_id if context else "",
                )
                attempted_robot_id = robot_id
                attempted_mode = "explicit_targets"
                return self._send_explicit_targets(
                    robot_bridge_client=robot_bridge_client,
                    robot_id=robot_id,
                    targets=explicit_targets,
                    messages=messages,
                    text=text,
                    args=args,
                )

            if broadcast:
                sent = self._broadcast_context_targets(
                    args=args,
                    text=text,
                    context=context,
                )
                return [
                    {
                        "type": "text",
                        "text": f"Broadcast sent to {len(sent)} QQ conversation(s).",
                    }
                ]

            if explicit_target is not None:
                robot_id = self._get_accessible_robot_id(
                    args,
                    fallback_robot_id=context.robot_id if context else "",
                )
                attempted_robot_id = robot_id
                attempted_target = explicit_target
                attempted_mode = "explicit_target"
                self._record_tool_send_event(
                    robot_id=robot_id,
                    target=explicit_target,
                    text=text,
                    mode="explicit_target",
                    args=args,
                )
                sent_messages = self._send_messages(
                    robot_bridge_client,
                    robot_id,
                    explicit_target,
                    messages,
                )
                self._remember_sent_messages(robot_id, explicit_target, sent_messages, args=args)
                bubble_count = len(sent_messages)
                turn_total = self._bump_turn_send_count(
                    str(args.get("_reply_ticket_id") or ""), bubble_count
                )
                result_text = (
                    f"已送达【{self._target_label(robot_id, explicit_target)}】："
                    f"本次 {bubble_count} 条气泡完整送达（本轮累计 {turn_total} 条）。"
                    "不要重复发送已发内容；"
                    "如果话还没说完（比如故事讲到一半、列表没列完），继续调用本工具发下一条；"
                    "内容真正讲完才能结束本轮——提前收尾等于把话说到一半扔下用户。"
                )
                form_note = self._bubble_form_feedback(messages)
                if form_note:
                    result_text += f"\n{form_note}"
                return [
                    {
                        "type": "text",
                        "text": result_text,
                    }
                ]

            if context_target is not None:
                fallback_robot_id = context_target_robot_id
                if not fallback_robot_id and context is not None:
                    fallback_robot_id = context.robot_id
                robot_id = self._get_accessible_robot_id(
                    args,
                    fallback_robot_id=fallback_robot_id,
                )
                attempted_robot_id = robot_id
                attempted_target = context_target
                attempted_mode = "context_target"
                self._record_tool_send_event(
                    robot_id=robot_id,
                    target=context_target,
                    text=text,
                    mode="context_target",
                    args=args,
                )
                sent_messages = self._send_messages(
                    robot_bridge_client,
                    robot_id,
                    context_target,
                    messages,
                )
                self._remember_sent_messages(robot_id, context_target, sent_messages, args=args)
                return [
                    {
                        "type": "text",
                        "text": (
                            f"Message sent to QQ {context_target.target_type} "
                            f"{self._target_label(robot_id, context_target)} from chat context."
                        ),
                    }
                ]

            if context is None:
                return [{"type": "text", "text": "Error: robot context unavailable"}]

            # 没有隐式目的地：每次发送都是一次显式选择（像人选好聊天窗口再打字）。
            # 当前对话也在可选列表里，回它就把它的 target 写明白——这样跨群任务
            # 不可能再"忘了带 target 发回自己家"。
            current_label = self._target_label(context.robot_id, context.reply_target)
            roster = self.roster_line(str(context.robot_id or ""), allow_fetch=False)
            roster_note = f"你在的会话：{roster}。" if roster else ""
            return [
                {
                    "type": "text",
                    "text": (
                        "Error: 发送缺少目的地。请显式传 target_type+target_id："
                        f"回当前对话就传 {context.reply_target.target_type} "
                        f"{context.reply_target.target_id}（{current_label}）；"
                        f"发别处就从你实际在的会话里选。{roster_note}"
                        "没见过/不在列表里的 id 会被拒绝，先 list_contacts 确认。"
                    ),
                }
            ]

        except Exception as exc:
            logger.warning("[RobotMCPServer] Failed to send robot message: %s", exc)
            if attempted_robot_id and attempted_target is not None:
                self._record_tool_send_event(
                    robot_id=attempted_robot_id,
                    target=attempted_target,
                    text=text,
                    mode=attempted_mode or "unknown",
                    args=args,
                    status="error",
                    error=str(exc),
                )
            return [{"type": "text", "text": f"Error: {exc}"}]

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": tool["inputSchema"],
                "skip_memory": tool.get("skip_memory", False),
            }
            for tool in self._tools.values()
        ]

    def call_tool(self, name: str, args: dict) -> list[dict[str, str]]:
        if name not in self._tools:
            return [{"type": "text", "text": f"Tool '{name}' not found"}]
        try:
            return self._tools[name]["handler"](args)
        except Exception as exc:
            logger.warning("[RobotMCPServer] Tool '%s' failed: %s", name, exc)
            return [{"type": "text", "text": f"Error: {exc}"}]

    async def run(self) -> None:
        logger.info("[RobotMCPServer] Starting stdio server")
        loop = asyncio.get_running_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                request = json.loads(line.strip())
                response = await self._handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError as exc:
                sys.stdout.write(json.dumps({"error": f"Invalid JSON: {exc}"}) + "\n")
                sys.stdout.flush()
            except Exception as exc:
                sys.stdout.write(json.dumps({"error": str(exc)}) + "\n")
                sys.stdout.flush()

    async def _handle_request(self, request: dict) -> dict:
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "TermPaws-robot", "version": "1.0.0"},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": self.list_tools()},
            }

        if method == "tools/call":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": self.call_tool(
                        params.get("name", ""),
                        params.get("arguments", {}),
                    )
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


robot_mcp_server = RobotMCPServer()


def main() -> None:
    asyncio.run(robot_mcp_server.run())


if __name__ == "__main__":
    main()
