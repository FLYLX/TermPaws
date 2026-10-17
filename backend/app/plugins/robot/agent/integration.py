from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from app.plugins.robot import is_robot_plugin_enabled
from app.plugins.robot.internal_trace import (
    NO_QQ_REPLY_MARKER,
    compact_robot_visible_message_text,
    is_robot_internal_trace_text,
    sanitize_robot_visible_text,
)
from app.plugins.robot.mcp.context import (
    RobotMCPContext,
    build_robot_reply_context_summary,
    extract_robot_context_targets_from_text,
    register_robot_mcp_context,
    unregister_robot_mcp_context,
)
from app.plugins.robot.reply_intent import is_no_reply_intent

if TYPE_CHECKING:
    from app.plugins.robot.contracts import RobotReplyTarget
    from app.services.agent.agent import Agent
    from app.services.agent.skills.definition import SkillDefinition

logger = logging.getLogger(__name__)

ROBOT_MCP_SERVER_NAME = "robot"
ROBOT_SEND_TOOL_NAME = "mcp_robot_send_message"
QQ_MCP_SKILL_ID = "qq_mcp"
ROBOT_MESSAGING_SKILL_ID = "robot_messaging"
ROBOT_MESSAGING_COMPAT_SKILL_IDS = {QQ_MCP_SKILL_ID, ROBOT_MESSAGING_SKILL_ID}
ROBOT_MESSAGE_STAMP_RE = re.compile(r"\[Robot message; (?P<body>[^\]]+)\]")
ROBOT_READ_ONLY_TOOL_RESULT_PREFIXES = (
    "QQ conversation .log memory ",
    "No QQ conversation .log memory ",
    "Robot memory recall ",
    "No robot memory ",
    "Memory saved",
)


def _has_tool(tools: list[dict[str, Any]], tool_name: str) -> bool:
    return any(tool.get("function", {}).get("name") == tool_name for tool in tools)


def _robot_context(agent: Agent) -> Any:
    return getattr(agent, "_context", None)


def _robot_conversation_key(reply_target: RobotReplyTarget, sender_key: str) -> str:
    metadata = reply_target.metadata if reply_target else {}
    conversation = metadata.get("conversation") if isinstance(metadata, dict) else {}
    if isinstance(conversation, dict):
        conversation_type = str(conversation.get("type") or "").strip().lower()
        conversation_id = str(conversation.get("id") or "").strip()
        if conversation_type and conversation_id:
            return f"{conversation_type}:{conversation_id}"

    summary = build_robot_reply_context_summary(reply_target, sender_key)
    match = re.search(r"^- conversation:\s*(.+)$", summary, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return sender_key


def _is_robot_send_tool_result(value: str) -> bool:
    return is_robot_internal_trace_text(value)


def _is_robot_read_only_tool_result(value: str) -> bool:
    normalized = str(value or "").strip()
    return any(
        normalized.startswith(prefix)
        for prefix in ROBOT_READ_ONLY_TOOL_RESULT_PREFIXES
    )


def is_robot_read_only_tool_result(value: str) -> bool:
    return _is_robot_read_only_tool_result(value)


def _parse_robot_message_stamp_body(body: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in body.split(";"):
        key, separator, value = part.strip().partition("=")
        if not separator:
            continue
        normalized_key = key.strip().lower()
        normalized_value = value.strip()
        if normalized_key and normalized_value:
            parsed[normalized_key] = normalized_value
    return parsed


def _robot_message_conversation_key(content: str) -> str:
    match = ROBOT_MESSAGE_STAMP_RE.search(content or "")
    if not match:
        return ""
    stamp = _parse_robot_message_stamp_body(match.group("body"))
    return stamp.get("conversation", "").strip()


def _event_is_assistant_context(event: dict[str, Any]) -> bool:
    return event.get("role") == "assistant" or event.get("type") in {
        "agent_action",
        "agent_tool_result",
        "agent_response",
        "agent_warning",
        "agent_error",
    }


def _reply_target_is_direct_wakeup(robot_reply_target: RobotReplyTarget | None) -> bool:
    if robot_reply_target is None:
        return False

    metadata = robot_reply_target.metadata
    return bool(metadata.get("mentioned_bot") or metadata.get("replied_to_bot"))


def _int_context_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _should_send_final_response_fallback(
    *,
    robot_id: str | None,
    robot_reply_target: RobotReplyTarget | None,
    content: str,
    robot_message_sent: bool,
    reply_requires_awake: bool = False,
) -> bool:
    return (
        bool(robot_id)
        and robot_reply_target is not None
        and bool(content.strip())
        and not robot_message_sent
        and (
            _reply_target_is_direct_wakeup(robot_reply_target)
            or reply_requires_awake
        )
    )


class RobotAgentIntegration:
    name = "robot"

    def build_system_prompt(self, agent: Agent) -> str:
        from app.plugins.robot.prompts import build_robot_context_prompt

        return build_robot_context_prompt(agent)

    def build_history_prompt(
        self,
        agent: Agent,
        *,
        message: str,
        context_messages: list[dict[str, str]],
    ) -> str:
        has_robot_context = "[Robot message;" in message or any(
            "[Robot message;" in context_message.get("content", "")
            for context_message in context_messages
        )
        if not has_robot_context:
            return ""

        from app.plugins.robot.prompts import build_robot_history_prompt

        return build_robot_history_prompt(agent, has_robot_context=has_robot_context)

    def history_scope_key(self, agent: Agent) -> str:
        context = _robot_context(agent)
        if context is None:
            return ""
        return str(getattr(context, "robot_conversation_key", "") or "").strip()

    def annotate_history_events(
        self,
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        annotated_events: list[dict[str, Any]] = []
        active_robot_conversation = ""

        for event in events:
            content = str(event.get("content", ""))
            if _event_is_assistant_context(event):
                sanitized_content = sanitize_robot_visible_text(content)
                if not sanitized_content:
                    continue
                if sanitized_content != content.strip():
                    event = {**event, "content": sanitized_content}
                    content = sanitized_content

            event_conversation = _robot_message_conversation_key(content)
            associated_conversation = ""

            if event_conversation:
                active_robot_conversation = event_conversation
                associated_conversation = event_conversation
            elif "[Robot message;" in content:
                active_robot_conversation = ""
                associated_conversation = "__unknown_robot_conversation__"
            elif _event_is_assistant_context(event):
                associated_conversation = active_robot_conversation
            elif event.get("role") in {"user", "terminal"} or event.get("type") in {
                "chat_user",
                "terminal_output",
            }:
                active_robot_conversation = ""

            if associated_conversation:
                event = {**event, "_integration_robot_conversation_key": associated_conversation}
            annotated_events.append(event)

        return annotated_events

    def history_event_matches_scope(
        self,
        event: dict[str, Any],
        scope_key: str,
    ) -> bool:
        if not scope_key:
            return True

        associated_conversation = str(
            event.get("_integration_robot_conversation_key", "") or ""
        ).strip()
        if associated_conversation:
            return associated_conversation == scope_key

        content = str(event.get("content", ""))
        event_conversation = _robot_message_conversation_key(content)
        if event_conversation:
            return event_conversation == scope_key
        if "[Robot message;" in content:
            return False
        return True

    def extract_context_targets(
        self,
        agent: Agent,
        messages: list[dict[str, Any]],
    ) -> None:
        context = _robot_context(agent)
        if context is None:
            return
        if not getattr(context, "robot_id", "") and not bool(
            getattr(context, "robot_backend_target_resolution_enabled", False)
        ):
            context.robot_known_targets = []
            return

        targets_by_key: dict[str, dict[str, str]] = {}
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else ""
            if not isinstance(content, str):
                continue
            for target in extract_robot_context_targets_from_text(content):
                target_data = dict(target)
                if getattr(context, "robot_id", "") and not target_data.get("robot_id"):
                    target_data["robot_id"] = context.robot_id
                key = f"{target_data.get('target_type')}:{target_data.get('target_id')}"
                targets_by_key[key] = target_data

        # Seed the robot's other active conversations (e.g. QQ groups seen by
        # this item) so web turns know they are deliverable targets even when
        # the chat history carries no robot message stamps for them. Text
        # parsing results keep priority for the same target key.
        for seeded_target in self._conversation_controller_targets(context):
            seeded_key = (
                f"{seeded_target.get('target_type')}:{seeded_target.get('target_id')}"
            )
            targets_by_key.setdefault(seeded_key, seeded_target)

        context.robot_known_targets = list(targets_by_key.values())

    def _conversation_controller_targets(self, context: Any) -> list[dict[str, str]]:
        robot_id = str(getattr(context, "robot_id", "") or "").strip()
        item_id = str(getattr(context, "item_id", "") or "").strip()
        if not robot_id and not item_id:
            return []

        try:
            from app.plugins.robot.service import robot_service

            snapshots = robot_service.conversation_controller_snapshots(
                robot_ids={robot_id} if robot_id else None,
                item_ids={item_id} if item_id else None,
            )
        except Exception:
            logger.warning(
                "[RobotIntegration] Failed to seed robot context targets from "
                "conversation controllers",
                exc_info=True,
            )
            return []

        targets: list[dict[str, str]] = []
        for snapshot in snapshots:
            target_type = str(snapshot.get("conversation_type") or "").strip().lower()
            target_id = str(snapshot.get("conversation_id") or "").strip()
            if target_type not in {"group", "private"} or not target_id:
                continue
            conversation = (
                str(snapshot.get("conversation_key") or "").strip()
                or f"{target_type}:{target_id}"
            )
            sender = ""
            pending_messages = snapshot.get("pending_messages")
            if isinstance(pending_messages, list) and pending_messages:
                last_pending = pending_messages[-1]
                if isinstance(last_pending, dict):
                    sender = str(
                        last_pending.get("sender_label")
                        or last_pending.get("sender_key")
                        or ""
                    ).strip()
            targets.append(
                {
                    "conversation": conversation,
                    "target_type": target_type,
                    "target_id": target_id,
                    "sender": sender,
                    "robot_id": str(snapshot.get("robot_id") or robot_id).strip(),
                }
            )
        return targets

    def record_context_targets(self, agent: Agent, item_id: str) -> None:
        context = _robot_context(agent)
        if context is None:
            return

        targets = getattr(context, "robot_known_targets", None) or []
        if not targets:
            return

        robot_ids = {
            str(target.get("robot_id") or "").strip()
            for target in targets
            if isinstance(target, dict) and str(target.get("robot_id") or "").strip()
        }
        if not robot_ids and getattr(context, "robot_id", ""):
            robot_ids.add(str(context.robot_id))

        if not robot_ids:
            return

        from app.plugins.robot.debug_log import record_robot_event

        payload_targets = [
            {
                "conversation": target.get("conversation"),
                "target_type": target.get("target_type"),
                "target_id": target.get("target_id"),
                "sender": target.get("sender"),
            }
            for target in targets
            if isinstance(target, dict)
        ]
        for robot_id in robot_ids:
            record_robot_event(
                robot_id,
                direction="agent_internal",
                event="robot_context_targets",
                message=f"{len(payload_targets)} QQ context target(s)",
                payload={
                    "item_id": item_id,
                    "targets": payload_targets,
                },
            )

    def setup_chat_context(self, agent: Agent, context: dict[str, Any]) -> bool:
        if not is_robot_plugin_enabled():
            return False
        agent_context = _robot_context(agent)
        if agent_context is None:
            return False

        robot_id = str(context.get("robot_id") or "").strip()
        sender_key = str(context.get("sender_key") or "").strip()
        reply_target = context.get("reply_target")
        if not robot_id or not sender_key or reply_target is None:
            return False

        self.clear_chat_context(agent, context)
        conversation_key = str(context.get("conversation_key") or "").strip()
        if not conversation_key:
            conversation_key = _robot_conversation_key(reply_target, sender_key)
        conversation_generation = _int_context_value(context.get("conversation_generation"))
        reply_requires_awake = bool(context.get("reply_requires_awake"))
        request_message = str(context.get("request_message") or "").strip()
        mcp_context = RobotMCPContext(
            robot_id=robot_id,
            sender_key=sender_key,
            reply_target=reply_target.model_copy(deep=True),
            conversation_key=conversation_key,
            conversation_generation=conversation_generation,
            reply_requires_awake=reply_requires_awake,
            request_message=request_message,
        )
        agent_context.robot_id = robot_id
        agent_context.robot_sender_key = sender_key
        agent_context.robot_conversation_key = conversation_key
        agent_context.robot_context_token = register_robot_mcp_context(mcp_context)
        reply_context_summary = build_robot_reply_context_summary(
            reply_target,
            sender_key,
        )
        try:
            from app.plugins.robot.mcp.server import robot_mcp_server

            roster = robot_mcp_server.roster_line(robot_id)
        except Exception:
            roster = ""
        if roster:
            # 渐进披露的一级：一行花名册让 agent 每轮都知道“我在哪些群/
            # 有哪些好友”，细节由 list_contacts 按需展开。
            reply_context_summary += f"\n- my conversations (sendable): {roster}"
        agent_context.robot_reply_context_summary = reply_context_summary
        agent_context.robot_mcp_server_transient = False
        return True

    async def ensure_chat_context_tools(
        self,
        agent: Agent,
        context: dict[str, Any],
    ) -> bool:
        if not is_robot_plugin_enabled():
            return False
        agent_context = _robot_context(agent)
        if agent_context is None or not getattr(agent_context, "robot_id", ""):
            return False
        return await self._ensure_messaging_tools(agent)

    def clear_chat_context(self, agent: Agent, context: dict[str, Any]) -> None:
        agent_context = _robot_context(agent)
        if agent_context is None:
            return

        should_remove_robot_mcp = bool(getattr(agent_context, "robot_mcp_server_transient", False))
        unregister_robot_mcp_context(getattr(agent_context, "robot_context_token", ""))
        agent_context.robot_id = ""
        agent_context.robot_sender_key = ""
        agent_context.robot_conversation_key = ""
        agent_context.robot_context_token = ""
        agent_context.robot_reply_context_summary = ""
        agent_context.robot_known_targets = []
        agent_context.robot_backend_target_resolution_enabled = False
        agent_context.robot_mcp_server_transient = False
        if should_remove_robot_mcp and ROBOT_MCP_SERVER_NAME in agent._mcp_servers:
            agent._mcp_servers = [
                server_name
                for server_name in agent._mcp_servers
                if server_name != ROBOT_MCP_SERVER_NAME
            ]
            agent._load_mcp_tools()

    def inject_tool_args(
        self,
        agent: Agent,
        *,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> None:
        args.pop("_robot_context_token", None)
        args.pop("_TermPaws_user_id", None)
        args.pop("_TermPaws_is_superuser", None)
        args.pop("_TermPaws_item_id", None)
        args.pop("_robot_known_targets", None)

        context = _robot_context(agent)

        # 线程无关的兜底：派发 worker 里每个异步工具调用都跑在新线程
        # （_run_async_from_sync），回合线程上的上下文在那里不可见。
        # reply ticket 自带完整 robot 上下文，是随数据走的可靠载体。
        ticket_token = ""
        if not (context is not None and getattr(context, "robot_context_token", "")):
            ticket_id = str(args.get("_reply_ticket_id") or "").strip()
            if ticket_id:
                try:
                    from app.services.agent.reply_ticket import reply_ticket_manager
                    from app.plugins.robot.mcp.context import (
                        get_or_register_context_for_ticket,
                    )

                    ticket = reply_ticket_manager.get(ticket_id)
                    if ticket is not None:
                        ticket_token = get_or_register_context_for_ticket(ticket)
                except Exception:
                    ticket_token = ""

        if context is None and not ticket_token:
            return

        robot_id = str(getattr(context, "robot_id", "") or "") if context is not None else ""
        context_token = (
            str(getattr(context, "robot_context_token", "") or "")
            if context is not None
            else ""
        ) or ticket_token

        if (
            server_name == "local"
            and tool_name in {"execute_command", "run_job"}
            and (robot_id or ticket_token)
            and context_token
        ):
            args["_robot_context_token"] = context_token
            return

        if server_name != ROBOT_MCP_SERVER_NAME:
            return

        if context is not None and getattr(context, "current_user_id", ""):
            args["_TermPaws_user_id"] = context.current_user_id
            args["_TermPaws_is_superuser"] = context.current_user_is_superuser
        if context is not None and getattr(context, "item_id", ""):
            args["_TermPaws_item_id"] = context.item_id
        if context is not None and getattr(context, "robot_known_targets", None):
            args["_robot_known_targets"] = [
                dict(target) for target in context.robot_known_targets
            ]
        if (robot_id or ticket_token) and context_token:
            args["_robot_context_token"] = context_token

        if getattr(context, "current_user_id", ""):
            args["_TermPaws_user_id"] = context.current_user_id
            args["_TermPaws_is_superuser"] = context.current_user_is_superuser
        if getattr(context, "item_id", ""):
            args["_TermPaws_item_id"] = context.item_id
        if getattr(context, "robot_known_targets", None):
            args["_robot_known_targets"] = [
                dict(target) for target in context.robot_known_targets
            ]
        if getattr(context, "robot_id", "") and getattr(context, "robot_context_token", ""):
            args["_robot_context_token"] = context.robot_context_token

    def should_retry_delivery(
        self,
        agent: Agent,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        final_response: str,
        retry_used: bool,
    ) -> bool:
        return (
            is_robot_plugin_enabled()
            and bool(final_response.strip())
            and not retry_used
            and not is_no_reply_intent(final_response)
            and _has_tool(tools, ROBOT_SEND_TOOL_NAME)
            and self._has_delivery_context(agent, messages)
        )

    def delivery_correction_message(self, final_response: str) -> dict[str, str]:
        from app.plugins.robot.prompts import build_robot_delivery_reflection_prompt

        reflection_prompt = build_robot_delivery_reflection_prompt(final_response)
        if reflection_prompt:
            return {"role": "system", "content": reflection_prompt}

        return {
            "role": "system",
            "content": (
                "Robot message delivery reflection:\n"
                "You produced a final assistant response without calling "
                "`mcp_robot_send_message`:\n"
                f"{final_response.strip()}\n\n"
                "Re-evaluate whether QQ should receive that text. If it should, "
                "call `mcp_robot_send_message` now using the QQ conversation "
                "visible in context. If not, respond with a concise internal note "
                "explaining that no QQ message was sent. Do not output the "
                "reflection itself."
            ),
        }

    def record_delivery_correction(
        self,
        agent: Agent,
        item_id: str,
        final_response: str,
    ) -> None:
        context = _robot_context(agent)
        if context is None:
            return

        robot_ids = {
            str(target.get("robot_id") or "").strip()
            for target in getattr(context, "robot_known_targets", []) or []
            if isinstance(target, dict) and str(target.get("robot_id") or "").strip()
        }
        if not robot_ids and getattr(context, "robot_id", ""):
            robot_ids.add(str(context.robot_id))

        if not robot_ids:
            return

        from app.plugins.robot.debug_log import preview_text, record_robot_event

        for robot_id in robot_ids:
            record_robot_event(
                robot_id,
                direction="agent_internal",
                event="robot_delivery_correction",
                message=preview_text(final_response),
                payload={
                    "item_id": item_id,
                    "reason": "plain_final_response_without_robot_tool_call",
                },
            )

    def is_delivery_tool_result(self, value: str) -> bool:
        return _is_robot_send_tool_result(value)

    def fallback_response_content(
        self,
        context: dict[str, Any],
        *,
        tool_results: list[str],
        warnings: list[str],
        done_seen: bool,
    ) -> str:
        if not context.get("robot_id"):
            return ""

        for result in reversed(tool_results):
            normalized = result.strip()
            if _is_robot_send_tool_result(normalized):
                return normalized

        for warning in reversed(warnings):
            if warning.strip():
                return warning.strip()

        if done_seen:
            return NO_QQ_REPLY_MARKER
        return ""

    def record_no_final_response(
        self,
        context: dict[str, Any],
        *,
        fallback_content: str,
        tool_results: list[str],
        warnings: list[str],
    ) -> None:
        robot_id = str(context.get("robot_id") or "").strip()
        if not robot_id:
            return

        from app.plugins.robot.debug_log import preview_text, record_robot_event

        record_robot_event(
            robot_id,
            direction="agent_internal",
            event="agent_no_final_response",
            message=preview_text(fallback_content),
            payload={
                "tool_result_count": len(tool_results),
                "warning_count": len(warnings),
                "used_fallback": bool(fallback_content),
            },
        )

    def send_final_response_fallback(
        self,
        context: dict[str, Any],
        *,
        content: str,
        message_sent: bool,
    ) -> bool:
        return False

    def should_enable_for_terminal_alert(self, agent: Agent, content: str) -> bool:
        if not is_robot_plugin_enabled():
            return False
        context = _robot_context(agent)
        if context is None:
            return False
        return (
            bool(
                ROBOT_MESSAGING_COMPAT_SKILL_IDS.intersection(
                    set(getattr(context, "enabled_skills", []) or [])
                )
            )
            or ROBOT_MCP_SERVER_NAME in set(getattr(context, "enabled_mcp_servers", []) or [])
        )

    async def ensure_terminal_alert_tools(self, agent: Agent, content: str) -> bool:
        if not self.should_enable_for_terminal_alert(agent, content):
            return False
        return await self._ensure_messaging_tools(agent)

    def clear_terminal_alert_tools(self, agent: Agent) -> None:
        context = _robot_context(agent)
        if context is None or not getattr(context, "robot_mcp_server_transient", False):
            return
        if getattr(context, "robot_id", ""):
            return

        context.robot_mcp_server_transient = False
        if ROBOT_MCP_SERVER_NAME in agent._mcp_servers:
            agent._mcp_servers = [
                server_name
                for server_name in agent._mcp_servers
                if server_name != ROBOT_MCP_SERVER_NAME
            ]
            agent._load_mcp_tools()

    def builtin_skill_definitions(self) -> list[SkillDefinition]:
        from app.plugins.robot.prompts import build_robot_messaging_skill_definition

        skill = build_robot_messaging_skill_definition()
        return [skill] if skill is not None else []

    def builtin_mcp_server_factories(self) -> dict[str, Any]:
        if not is_robot_plugin_enabled():
            return {}

        def factory() -> Any:
            from app.plugins.robot.mcp.server import RobotMCPServer

            return RobotMCPServer()

        return {ROBOT_MCP_SERVER_NAME: factory}

    def build_source_route(self, agent: Agent, *, source: str) -> str:
        context = _robot_context(agent)
        if context is None:
            return ""
        robot_id = str(getattr(context, "robot_id", "") or "").strip()
        if not robot_id:
            return ""
        robot_conversation_key = str(
            getattr(context, "robot_conversation_key", "") or ""
        ).strip()
        route = robot_conversation_key or "current QQ conversation"
        return (
            "Current source route:\n"
            f"- current source: QQ robot conversation ({route})\n"
            "- reply contract: to reply to the current QQ conversation, call `mcp_robot_send_message` in your current response (pass only `text`; it is routed to this conversation automatically). Do not end the turn with plain reply text - final text is NOT auto-delivered. Do not leave the answer only in the TermPaws web chat.\n"
            "- reply only to the source: do NOT broadcast the answer to the terminal "
            "or game server console (e.g., say/tell commands). The user must explicitly "
            "ask to '在服务器说/广播/公告' to trigger a console say; reporting a task result "
            "(e.g. 服务器开好了) is NOT a reason to say in console—report only to the QQ/web source."
        )

    def build_ticket_prompt(self, ticket: Any) -> str:
        source_type = str(getattr(ticket, "source_type", "") or "").strip().lower()
        if source_type != "qq":
            return ""
        conv_key = str(getattr(ticket, "conversation_key", "") or "")
        target_hint = ""
        if ":" in conv_key:
            parts = conv_key.split(":", 1)
            target_hint = (
                f"- To send: call `mcp_robot_send_message` with "
                f"target_type=\"{parts[0]}\" target_id=\"{parts[1]}\". "
                f"This works even without active QQ context.\n"
            )
        ticket_id = str(getattr(ticket, "ticket_id", "") or "")
        duplicate_hint = ""
        duplicate_of = str(getattr(ticket, "duplicate_of", "") or "").strip()
        if duplicate_of:
            duplicate_hint = (
                f"- POSSIBLE DUPLICATE: 同会话已有一张内容完全相同的活跃票据 "
                f"{duplicate_of[:8]}。先判断：若只是重复请求（重发/催问），不要重复执行，"
                "直接说明已在处理或引用进度即可；若带有新意图再正常处理。\n"
            )
        siblings_hint = ""
        robot_id = str(getattr(ticket, "robot_id", "") or "").strip()
        if robot_id and conv_key:
            try:
                from app.services.agent.reply_ticket import reply_ticket_manager

                siblings = reply_ticket_manager.active_sibling_summary(
                    robot_id=robot_id,
                    conversation_key=conv_key,
                    exclude_ticket_id=ticket_id,
                )
            except Exception:
                siblings = []
            if siblings:
                lines = [
                    f"  - 票据 {s['ticket_id']}「{s['request']}」"
                    + (
                        f"（进行中：{'；'.join(s['open_steps'])}）"
                        if s["open_steps"]
                        else ("（已汇报）" if s["reported"] else "（待处理）")
                    )
                    for s in siblings[:4]
                ]
                siblings_hint = (
                    "- 同会话其它在途票据：\n"
                    + "\n".join(lines)
                    + "\n  判断是否与本次是同一件事：是就 update_plan(supersede_ticket_ids=[对方票据号]) "
                    "接管并合并汇报一次，或仅回复一句进度；不要重复执行、不要分多次汇报同一结果。\n"
                )
        return (
            "Authoritative reply ticket:\n"
            f"- ticket_id: {ticket_id}\n"
            f"- source: QQ ({conv_key or 'current conversation'})\n"
            f"{target_hint}"
            f"{duplicate_hint}"
            f"{siblings_hint}"
            "- REPLY ROUTING: to reply to this QQ conversation, call "
            "`mcp_robot_send_message` in your current response (pass only `text`). "
            "Plain final text is NOT auto-delivered. Do not leave answers only in TermPaws.\n"
            "- If the user asks you to send elsewhere, do it then report back here.\n"
            "- Do not duplicate: if already sent via tool, do not restate in final text.\n"
        )

    def memory_scope_rank(self, agent: Agent, memory: dict[str, Any]) -> int:
        context = _robot_context(agent)
        metadata = memory.get("metadata") if isinstance(memory, dict) else {}
        if not isinstance(metadata, dict):
            return 0
        has_robot_scope = bool(
            str(metadata.get("robot_id") or "").strip()
            or str(metadata.get("robot_conversation_key") or "").strip()
        )
        if context is None:
            return -1 if has_robot_scope else 0
        robot_id = str(getattr(context, "robot_id", "") or "").strip()
        conversation_key = str(getattr(context, "robot_conversation_key", "") or "").strip()
        sender_key = str(getattr(context, "robot_sender_key", "") or "").strip()
        if not robot_id and not conversation_key:
            return -1 if has_robot_scope else 0
        try:
            from app.plugins.robot.memory_scope import (
                memory_scope_rank as plugin_memory_scope_rank,
                speaker_global_key_from_context,
            )

            return plugin_memory_scope_rank(
                memory,
                robot_id=robot_id,
                conversation_key=conversation_key,
                speaker_global_key=speaker_global_key_from_context(sender_key),
            )
        except Exception:
            return 0

    def filter_skills(
        self,
        skills: list[Any],
        agent: Agent,
    ) -> list[Any]:
        context = _robot_context(agent)
        if context is None:
            return skills
        robot_prompt_active = bool(
            str(getattr(context, "robot_id", "") or "").strip()
        )
        if not robot_prompt_active:
            return skills
        return [
            skill for skill in skills
            if getattr(skill, "skill_id", "") not in ROBOT_MESSAGING_COMPAT_SKILL_IDS
        ]

    def deliver_ticket(self, ticket: Any, text: str) -> bool:
        source_type = str(getattr(ticket, "source_type", "") or "").strip().lower()
        if source_type != "qq":
            return False
        from app.plugins.robot.bridge_client import robot_bridge_client
        from app.plugins.robot.contracts import RobotReplyTarget

        reply_target_dict = getattr(ticket, "reply_target", None)
        if not isinstance(reply_target_dict, dict):
            return False
        target = RobotReplyTarget.model_validate(reply_target_dict)
        robot_id = str(getattr(ticket, "robot_id", "") or "")
        if not robot_id:
            return False
        robot_bridge_client.send_message(robot_id, target, text)
        return True

    def conversation_memory_append(self, ticket: Any, text: str) -> None:
        source_type = str(getattr(ticket, "source_type", "") or "").strip().lower()
        if source_type != "qq":
            return
        conversation_key = str(getattr(ticket, "conversation_key", "") or "")
        robot_id = str(getattr(ticket, "robot_id", "") or "")
        if not conversation_key or not robot_id:
            return
        try:
            from app.plugins.robot.conversation_memory import robot_conversation_memory

            robot_conversation_memory.append_assistant_message(
                robot_id,
                conversation_key,
                text,
            )
        except Exception:
            pass

    def is_delivery_event(self, payload: dict[str, Any]) -> bool:
        event_type = str(payload.get("type") or "").strip().lower()
        tool_name = str(payload.get("tool_name") or "").strip().lower()
        return (
            event_type == "agent_qq_reply"
            or tool_name == "reply_ticket"
            or tool_name == "mcp_robot_send_message"
        )

    def is_internal_trace(self, text: str) -> bool:
        from app.plugins.robot.internal_trace import is_robot_internal_trace_text

        return is_robot_internal_trace_text(text)

    def is_read_only_result(self, text: str) -> bool:
        # Defined at module level in this file; internal_trace has no such
        # export (the old import raised ImportError mid-delivery).
        return is_robot_read_only_tool_result(text)

    def sanitize_visible_text(self, text: str) -> str:
        from app.plugins.robot.internal_trace import sanitize_robot_visible_text

        return sanitize_robot_visible_text(text)

    def get_context_by_token(self, token: str) -> Any:
        from app.plugins.robot.mcp.context import get_robot_mcp_context

        return get_robot_mcp_context(token)

    def reset_delivery_tracker(self, key: str) -> None:
        from app.plugins.robot.mcp.server import RobotMCPServer

        RobotMCPServer.reset_delivered_targets(key)

    def extract_targets_from_text(
        self, *, item_id: str, user_message: str, final_text: str, delivery_key: str
    ) -> None:
        from app.plugins.robot.explicit_target_backfill import (
            run_explicit_target_backfill,
        )

        run_explicit_target_backfill(
            item_id=item_id,
            user_message=user_message,
            final_text=final_text,
            delivery_key=delivery_key,
        )

    def is_conversation_processing(self, integration_id: str, conversation_key: str) -> bool:
        from app.plugins.robot.service import robot_service

        # A conversation counts as processing when a dispatch job is running
        # or queued for it. The raw controller `processing` flag is not used
        # here: job-result buffering relies on the turn-end flush hook, which
        # only fires for dispatch jobs, so only an active/queued dispatch job
        # guarantees buffered results get drained.
        return robot_service.conversation_has_active_dispatch(
            integration_id, conversation_key
        )

    def register_background_job_reply(
        self, *, integration_id: str, item_id: str, sender_key: str,
        reply_target: dict, conversation_key: str, conversation_generation: int,
        command: str,
    ) -> str:
        from app.plugins.robot.service import robot_service

        return robot_service.register_background_job_reply(
            robot_id=integration_id,
            item_id=item_id,
            sender_key=sender_key,
            reply_target=reply_target,
            conversation_key=conversation_key,
            conversation_generation=conversation_generation,
            command=command,
        )

    def enqueue_background_job_result(
        self, *, integration_id: str, item_id: str, reply_target: dict,
        sender_key: str, conversation_key: str, conversation_generation: int,
        reply_requires_awake: bool, reply_ticket_id: str, message: str,
        pending_reply_id: str = "",
    ) -> bool:
        from app.plugins.robot.service import robot_service

        return robot_service.enqueue_background_job_result(
            robot_id=integration_id,
            item_id=item_id,
            reply_target=reply_target,
            sender_key=sender_key,
            conversation_key=conversation_key,
            conversation_generation=conversation_generation,
            reply_requires_awake=reply_requires_awake,
            reply_ticket_id=reply_ticket_id,
            message=message,
            pending_reply_id=pending_reply_id,
        )

    def clear_background_job_reply(
        self, *, integration_id: str, conversation_key: str, pending_reply_id: str,
    ) -> None:
        from app.plugins.robot.service import robot_service

        robot_service.clear_background_job_reply(
            robot_id=integration_id,
            conversation_key=conversation_key,
            pending_reply_id=pending_reply_id,
        )

    def reap_stuck_dispatch_jobs(self) -> int:
        from app.plugins.robot.service import robot_service

        return robot_service.reap_stuck_dispatch_jobs()

    def sweep_pending_inputs(self) -> dict[str, int]:
        from app.plugins.robot.service import robot_service

        return robot_service.sweep_pending_conversations()

    async def _ensure_messaging_tools(self, agent: Agent) -> bool:
        if not is_robot_plugin_enabled():
            return False
        from app.services.agent.mcp.server_manager import mcp_server_manager

        context = _robot_context(agent)
        if context is None:
            return False

        added_transient = False
        if ROBOT_MCP_SERVER_NAME not in agent._mcp_servers:
            agent._mcp_servers.append(ROBOT_MCP_SERVER_NAME)
            context.robot_mcp_server_transient = True
            added_transient = True

        if not mcp_server_manager.is_server_running(ROBOT_MCP_SERVER_NAME):
            started = await mcp_server_manager.start_server(ROBOT_MCP_SERVER_NAME)
            if not started:
                logger.warning(
                    "[RobotIntegration] Robot MCP server is not available for handler %s",
                    agent.handler_id,
                )
                return False

        agent._load_mcp_tools()
        return added_transient

    def _has_delivery_context(
        self,
        agent: Agent,
        messages: list[dict[str, Any]],
    ) -> bool:
        context = _robot_context(agent)
        if context is not None:
            return bool(
                getattr(context, "robot_id", "")
                or (
                    bool(
                        getattr(
                            context,
                            "robot_backend_target_resolution_enabled",
                            False,
                        )
                    )
                    and getattr(context, "robot_known_targets", None)
                )
            )

        return False

    def _record_final_response_fallback(
        self,
        robot_id: str | None,
        *,
        robot_reply_target: RobotReplyTarget,
        content: str,
    ) -> None:
        if not robot_id:
            return

        from app.plugins.robot.debug_log import preview_text, record_robot_event

        metadata = robot_reply_target.metadata
        record_robot_event(
            robot_id,
            direction="agent_internal",
            event="agent_final_response_bridge_fallback",
            message=preview_text(content),
            payload={
                "target_type": robot_reply_target.target_type,
                "target_id": robot_reply_target.target_id,
                "mentioned_bot": bool(metadata.get("mentioned_bot")),
                "replied_to_bot": bool(metadata.get("replied_to_bot")),
            },
        )


def get_robot_agent_integration() -> RobotAgentIntegration:
    return RobotAgentIntegration()


def register_robot_agent_integration() -> None:
    if not is_robot_plugin_enabled():
        return
    from app.services.agent.integrations.registry import (
        register_agent_integration as register_core_agent_integration,
    )

    register_core_agent_integration(get_robot_agent_integration())


def register_agent_integration() -> None:
    register_robot_agent_integration()


__all__ = [
    "ROBOT_MCP_SERVER_NAME",
    "ROBOT_MESSAGING_SKILL_ID",
    "QQ_MCP_SKILL_ID",
    "ROBOT_SEND_TOOL_NAME",
    "RobotAgentIntegration",
    "get_robot_agent_integration",
    "register_agent_integration",
    "register_robot_agent_integration",
    "_is_robot_send_tool_result",
    "_should_send_final_response_fallback",
]
