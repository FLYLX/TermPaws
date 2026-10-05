from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .contracts import DeliveryRetryDecision, MCPServerFactory
from .registry import get_agent_integration, get_agent_integrations

if TYPE_CHECKING:
    from app.services.agent.agent import Agent
    from app.services.agent.skills.definition import SkillDefinition


def build_integration_system_prompt(agent: Agent) -> str:
    return "\n\n".join(
        prompt
        for integration in get_agent_integrations()
        if (prompt := integration.build_system_prompt(agent).strip())
    )


def build_integration_history_prompt(
    agent: Agent,
    *,
    message: str,
    context_messages: list[dict[str, str]],
) -> str:
    return "\n\n".join(
        prompt
        for integration in get_agent_integrations()
        if (
            prompt := integration.build_history_prompt(
                agent,
                message=message,
                context_messages=context_messages,
            ).strip()
        )
    )


def annotate_integration_history_events(
    agent: Agent,
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    annotated_events = events
    scopes: dict[str, str] = {}
    for integration in get_agent_integrations():
        scope_key = integration.history_scope_key(agent)
        if scope_key:
            scopes[integration.name] = scope_key
        annotated_events = integration.annotate_history_events(annotated_events)
    return annotated_events, scopes


def integration_history_event_matches_scopes(
    event: dict[str, Any],
    scopes: dict[str, str],
) -> bool:
    if not scopes:
        return True
    for integration in get_agent_integrations():
        scope_key = scopes.get(integration.name, "")
        if scope_key and not integration.history_event_matches_scope(event, scope_key):
            return False
    return True


def extract_integration_context_targets(
    agent: Agent,
    messages: list[dict[str, Any]],
) -> None:
    for integration in get_agent_integrations():
        integration.extract_context_targets(agent, messages)


def record_integration_context_targets(agent: Agent, item_id: str) -> None:
    for integration in get_agent_integrations():
        integration.record_context_targets(agent, item_id)


def inject_integration_tool_args(
    agent: Agent,
    *,
    server_name: str,
    tool_name: str,
    args: dict[str, Any],
) -> None:
    for integration in get_agent_integrations():
        integration.inject_tool_args(
            agent,
            server_name=server_name,
            tool_name=tool_name,
            args=args,
        )


def get_delivery_retry_decision(
    agent: Agent,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    final_response: str,
    retry_used_by_integration: dict[str, bool],
) -> DeliveryRetryDecision | None:
    for integration in get_agent_integrations():
        retry_used = retry_used_by_integration.get(integration.name, False)
        if not integration.should_retry_delivery(
            agent,
            messages=messages,
            tools=tools,
            final_response=final_response,
            retry_used=retry_used,
        ):
            continue
        retry_used_by_integration[integration.name] = True
        return DeliveryRetryDecision(
            integration_name=integration.name,
            correction_message=integration.delivery_correction_message(final_response),
        )
    return None


def record_integration_delivery_correction(
    agent: Agent,
    *,
    integration_name: str,
    item_id: str,
    final_response: str,
) -> None:
    integration = get_agent_integration(integration_name)
    if integration is None:
        return
    integration.record_delivery_correction(agent, item_id, final_response)


def setup_integration_chat_contexts(
    agent: Agent,
    contexts: dict[str, dict[str, Any]],
) -> None:
    for name, context in contexts.items():
        integration = get_agent_integration(name)
        if integration is not None:
            integration.setup_chat_context(agent, context)


async def ensure_integration_chat_context_tools(
    agent: Agent,
    contexts: dict[str, dict[str, Any]],
) -> None:
    for name, context in contexts.items():
        integration = get_agent_integration(name)
        if integration is not None:
            await integration.ensure_chat_context_tools(agent, context)


def clear_integration_chat_contexts(
    agent: Agent,
    contexts: dict[str, dict[str, Any]],
) -> None:
    for name, context in contexts.items():
        integration = get_agent_integration(name)
        if integration is not None:
            integration.clear_chat_context(agent, context)


def integration_message_sent(tool_results: list[str]) -> bool:
    return any(
        integration.is_delivery_tool_result(result)
        for integration in get_agent_integrations()
        for result in tool_results
    )


def send_integration_final_response_fallback(
    contexts: dict[str, dict[str, Any]],
    *,
    content: str,
    message_sent: bool,
) -> bool:
    delivered = message_sent
    for name, context in contexts.items():
        integration = get_agent_integration(name)
        if integration is None:
            continue
        if integration.send_final_response_fallback(
            context,
            content=content,
            message_sent=delivered,
        ):
            delivered = True
    return delivered


def integration_fallback_response_content(
    contexts: dict[str, dict[str, Any]],
    *,
    tool_results: list[str],
    warnings: list[str],
    done_seen: bool,
) -> str:
    for name, context in contexts.items():
        integration = get_agent_integration(name)
        if integration is None:
            continue
        fallback = integration.fallback_response_content(
            context,
            tool_results=tool_results,
            warnings=warnings,
            done_seen=done_seen,
        )
        if fallback:
            return fallback
    return ""


def record_integration_no_final_response(
    contexts: dict[str, dict[str, Any]],
    *,
    fallback_content: str,
    tool_results: list[str],
    warnings: list[str],
) -> None:
    for name, context in contexts.items():
        integration = get_agent_integration(name)
        if integration is not None:
            integration.record_no_final_response(
                context,
                fallback_content=fallback_content,
                tool_results=tool_results,
                warnings=warnings,
            )


def fallback_is_delivery_result(value: str) -> bool:
    return any(
        integration.is_delivery_tool_result(value)
        for integration in get_agent_integrations()
    )


def get_builtin_mcp_server_factory(name: str) -> MCPServerFactory | None:
    for integration in get_agent_integrations():
        factory = integration.builtin_mcp_server_factories().get(name)
        if factory is not None:
            return factory
    return None


def get_builtin_skill_definitions() -> list[SkillDefinition]:
    skills: list[SkillDefinition] = []
    for integration in get_agent_integrations():
        skills.extend(integration.builtin_skill_definitions())
    return skills


def should_enable_terminal_alert_integrations(agent: Agent, content: str) -> bool:
    return any(
        integration.should_enable_for_terminal_alert(agent, content)
        for integration in get_agent_integrations()
    )


async def ensure_terminal_alert_integration_tools(
    agent: Agent,
    content: str,
) -> bool:
    added = False
    for integration in get_agent_integrations():
        if await integration.ensure_terminal_alert_tools(agent, content):
            added = True
    return added


def clear_terminal_alert_integration_tools(agent: Agent) -> None:
    for integration in get_agent_integrations():
        integration.clear_terminal_alert_tools(agent)


def build_integration_source_route(agent: Agent, *, source: str) -> str:
    for integration in get_agent_integrations():
        prompt = integration.build_source_route(agent, source=source).strip()
        if prompt:
            return prompt
    return ""


def build_integration_ticket_prompt(ticket: Any) -> str:
    for integration in get_agent_integrations():
        prompt = integration.build_ticket_prompt(ticket).strip()
        if prompt:
            return prompt
    return ""


def integration_memory_scope_rank(agent: Agent, memory: dict[str, Any]) -> int:
    best = 0
    for integration in get_agent_integrations():
        rank = integration.memory_scope_rank(agent, memory)
        if rank < 0:
            return rank
        if rank > best:
            best = rank
    return best


def integration_filter_skills(
    skills: list[SkillDefinition],
    agent: Agent,
) -> list[SkillDefinition]:
    filtered = skills
    for integration in get_agent_integrations():
        filtered = integration.filter_skills(filtered, agent)
    return filtered


def deliver_integration_ticket(ticket: Any, text: str) -> bool:
    for integration in get_agent_integrations():
        if integration.deliver_ticket(ticket, text):
            return True
    return False


def integration_conversation_memory_append(ticket: Any, text: str) -> None:
    for integration in get_agent_integrations():
        integration.conversation_memory_append(ticket, text)


def parse_integration_inbound_message(raw: dict[str, Any]) -> Any:
    for integration in get_agent_integrations():
        parsed = integration.parse_inbound_message(raw)
        if parsed is not None:
            return parsed
    return None


def notify_integration_terminal_output(
    item_id: str,
    content: str,
    *,
    source: str,
) -> None:
    for integration in get_agent_integrations():
        integration.on_terminal_output(item_id, content, source=source)


def notify_integration_item_event(
    item_id: str,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    payload = payload or {}
    for integration in get_agent_integrations():
        integration.on_item_event(item_id, event, payload)


def is_integration_delivery_event(payload: dict[str, Any]) -> bool:
    return any(
        integration.is_delivery_event(payload)
        for integration in get_agent_integrations()
    )


def is_integration_internal_trace(text: str) -> bool:
    return any(
        integration.is_internal_trace(text)
        for integration in get_agent_integrations()
    )


def is_integration_read_only_result(text: str) -> bool:
    return any(
        integration.is_read_only_result(text)
        for integration in get_agent_integrations()
    )


def sanitize_integration_visible_text(text: str) -> str:
    result = text
    for integration in get_agent_integrations():
        result = integration.sanitize_visible_text(result)
    return result


def get_integration_context_by_token(token: str) -> Any:
    for integration in get_agent_integrations():
        ctx = integration.get_context_by_token(token)
        if ctx is not None:
            return ctx
    return None


def reset_integration_delivery_tracker(key: str) -> None:
    for integration in get_agent_integrations():
        integration.reset_delivery_tracker(key)


def extract_integration_targets_from_text(
    *, item_id: str, user_message: str, final_text: str, delivery_key: str
) -> None:
    for integration in get_agent_integrations():
        integration.extract_targets_from_text(
            item_id=item_id,
            user_message=user_message,
            final_text=final_text,
            delivery_key=delivery_key,
        )


def is_integration_conversation_processing(
    integration_id: str, conversation_key: str
) -> bool:
    return any(
        integration.is_conversation_processing(integration_id, conversation_key)
        for integration in get_agent_integrations()
    )


def register_integration_background_job_reply(
    *, integration_id: str, item_id: str, sender_key: str,
    reply_target: dict, conversation_key: str, conversation_generation: int,
    command: str,
) -> str:
    for integration in get_agent_integrations():
        result = integration.register_background_job_reply(
            integration_id=integration_id,
            item_id=item_id,
            sender_key=sender_key,
            reply_target=reply_target,
            conversation_key=conversation_key,
            conversation_generation=conversation_generation,
            command=command,
        )
        if result:
            return result
    return ""


def enqueue_integration_background_job_result(
    *, integration_id: str, item_id: str, reply_target: dict,
    sender_key: str, conversation_key: str, conversation_generation: int,
    reply_requires_awake: bool, reply_ticket_id: str, message: str,
    pending_reply_id: str = "",
) -> bool:
    for integration in get_agent_integrations():
        if integration.enqueue_background_job_result(
            integration_id=integration_id,
            item_id=item_id,
            reply_target=reply_target,
            sender_key=sender_key,
            conversation_key=conversation_key,
            conversation_generation=conversation_generation,
            reply_requires_awake=reply_requires_awake,
            reply_ticket_id=reply_ticket_id,
            message=message,
            pending_reply_id=pending_reply_id,
        ):
            return True
    return False


def clear_integration_background_job_reply(
    *, integration_id: str, conversation_key: str, pending_reply_id: str,
) -> None:
    for integration in get_agent_integrations():
        integration.clear_background_job_reply(
            integration_id=integration_id,
            conversation_key=conversation_key,
            pending_reply_id=pending_reply_id,
        )
