from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from app.services.agent.agent import Agent
    from app.services.agent.skills.definition import SkillDefinition

MCPServerFactory = Callable[[], Any]


class AgentIntegration(Protocol):
    name: str

    def build_system_prompt(self, agent: Agent) -> str: ...

    def build_history_prompt(
        self,
        agent: Agent,
        *,
        message: str,
        context_messages: list[dict[str, str]],
    ) -> str: ...

    def history_scope_key(self, agent: Agent) -> str: ...

    def annotate_history_events(
        self,
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...

    def history_event_matches_scope(
        self,
        event: dict[str, Any],
        scope_key: str,
    ) -> bool: ...

    def extract_context_targets(
        self,
        agent: Agent,
        messages: list[dict[str, Any]],
    ) -> None: ...

    def record_context_targets(self, agent: Agent, item_id: str) -> None: ...

    def setup_chat_context(self, agent: Agent, context: dict[str, Any]) -> bool: ...

    async def ensure_chat_context_tools(
        self,
        agent: Agent,
        context: dict[str, Any],
    ) -> bool: ...

    def clear_chat_context(self, agent: Agent, context: dict[str, Any]) -> None: ...

    def inject_tool_args(
        self,
        agent: Agent,
        *,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> None: ...

    def should_retry_delivery(
        self,
        agent: Agent,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        final_response: str,
        retry_used: bool,
    ) -> bool: ...

    def delivery_correction_message(self, final_response: str) -> dict[str, str]: ...

    def record_delivery_correction(
        self,
        agent: Agent,
        item_id: str,
        final_response: str,
    ) -> None: ...

    def is_delivery_tool_result(self, value: str) -> bool: ...

    def fallback_response_content(
        self,
        context: dict[str, Any],
        *,
        tool_results: list[str],
        warnings: list[str],
        done_seen: bool,
    ) -> str: ...

    def record_no_final_response(
        self,
        context: dict[str, Any],
        *,
        fallback_content: str,
        tool_results: list[str],
        warnings: list[str],
    ) -> None: ...

    def send_final_response_fallback(
        self,
        context: dict[str, Any],
        *,
        content: str,
        message_sent: bool,
    ) -> bool: ...

    def should_enable_for_terminal_alert(self, agent: Agent, content: str) -> bool: ...

    async def ensure_terminal_alert_tools(self, agent: Agent, content: str) -> bool: ...

    def clear_terminal_alert_tools(self, agent: Agent) -> None: ...

    def builtin_skill_definitions(self) -> list[SkillDefinition]: ...

    def builtin_mcp_server_factories(self) -> dict[str, MCPServerFactory]: ...

    def build_source_route(self, agent: Agent, *, source: str) -> str: ...

    def build_ticket_prompt(self, ticket: Any) -> str: ...

    def memory_scope_rank(self, agent: Agent, memory: dict[str, Any]) -> int: ...

    def filter_skills(
        self,
        skills: list[SkillDefinition],
        agent: Agent,
    ) -> list[SkillDefinition]: ...

    def deliver_ticket(self, ticket: Any, text: str) -> bool: ...

    def parse_inbound_message(self, raw: dict[str, Any]) -> Any: ...

    def conversation_memory_append(self, ticket: Any, text: str) -> None: ...

    def on_terminal_output(self, item_id: str, content: str, *, source: str) -> None: ...

    def on_item_event(self, item_id: str, event: str, payload: dict[str, Any]) -> None: ...

    def is_delivery_event(self, payload: dict[str, Any]) -> bool: ...

    def is_internal_trace(self, text: str) -> bool: ...

    def is_read_only_result(self, text: str) -> bool: ...

    def sanitize_visible_text(self, text: str) -> str: ...

    def get_context_by_token(self, token: str) -> Any: ...

    def reset_delivery_tracker(self, key: str) -> None: ...

    def extract_targets_from_text(
        self, *, item_id: str, user_message: str, final_text: str, delivery_key: str
    ) -> None: ...

    def is_conversation_processing(self, integration_id: str, conversation_key: str) -> bool: ...

    def register_background_job_reply(
        self, *, integration_id: str, item_id: str, sender_key: str,
        reply_target: dict, conversation_key: str, conversation_generation: int,
        command: str,
    ) -> str: ...

    def enqueue_background_job_result(
        self, *, integration_id: str, item_id: str, reply_target: dict,
        sender_key: str, conversation_key: str, conversation_generation: int,
        reply_requires_awake: bool, reply_ticket_id: str, message: str,
        pending_reply_id: str = "",
    ) -> bool: ...

    def clear_background_job_reply(
        self, *, integration_id: str, conversation_key: str, pending_reply_id: str,
    ) -> None: ...

    def reap_stuck_dispatch_jobs(self) -> int: ...


class NoopAgentIntegration:
    name = "noop"

    def build_system_prompt(self, agent: Agent) -> str:
        return ""

    def build_history_prompt(
        self,
        agent: Agent,
        *,
        message: str,
        context_messages: list[dict[str, str]],
    ) -> str:
        return ""

    def history_scope_key(self, agent: Agent) -> str:
        return ""

    def annotate_history_events(
        self,
        events: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return events

    def history_event_matches_scope(
        self,
        event: dict[str, Any],
        scope_key: str,
    ) -> bool:
        return True

    def extract_context_targets(
        self,
        agent: Agent,
        messages: list[dict[str, Any]],
    ) -> None:
        return None

    def record_context_targets(self, agent: Agent, item_id: str) -> None:
        return None

    def setup_chat_context(self, agent: Agent, context: dict[str, Any]) -> bool:
        return False

    async def ensure_chat_context_tools(
        self,
        agent: Agent,
        context: dict[str, Any],
    ) -> bool:
        return False

    def clear_chat_context(self, agent: Agent, context: dict[str, Any]) -> None:
        return None

    def inject_tool_args(
        self,
        agent: Agent,
        *,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
    ) -> None:
        return None

    def should_retry_delivery(
        self,
        agent: Agent,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        final_response: str,
        retry_used: bool,
    ) -> bool:
        return False

    def delivery_correction_message(self, final_response: str) -> dict[str, str]:
        return {}

    def record_delivery_correction(
        self,
        agent: Agent,
        item_id: str,
        final_response: str,
    ) -> None:
        return None

    def is_delivery_tool_result(self, value: str) -> bool:
        return False

    def fallback_response_content(
        self,
        context: dict[str, Any],
        *,
        tool_results: list[str],
        warnings: list[str],
        done_seen: bool,
    ) -> str:
        return ""

    def record_no_final_response(
        self,
        context: dict[str, Any],
        *,
        fallback_content: str,
        tool_results: list[str],
        warnings: list[str],
    ) -> None:
        return None

    def send_final_response_fallback(
        self,
        context: dict[str, Any],
        *,
        content: str,
        message_sent: bool,
    ) -> bool:
        return False

    def should_enable_for_terminal_alert(self, agent: Agent, content: str) -> bool:
        return False

    async def ensure_terminal_alert_tools(self, agent: Agent, content: str) -> bool:
        return False

    def clear_terminal_alert_tools(self, agent: Agent) -> None:
        return None

    def builtin_skill_definitions(self) -> list[SkillDefinition]:
        return []

    def builtin_mcp_server_factories(self) -> dict[str, MCPServerFactory]:
        return {}

    def build_source_route(self, agent: Agent, *, source: str) -> str:
        return ""

    def build_ticket_prompt(self, ticket: Any) -> str:
        return ""

    def memory_scope_rank(self, agent: Agent, memory: dict[str, Any]) -> int:
        return 0

    def filter_skills(
        self,
        skills: list[SkillDefinition],
        agent: Agent,
    ) -> list[SkillDefinition]:
        return skills

    def deliver_ticket(self, ticket: Any, text: str) -> bool:
        return False

    def parse_inbound_message(self, raw: dict[str, Any]) -> Any:
        return None

    def conversation_memory_append(self, ticket: Any, text: str) -> None:
        return None

    def on_terminal_output(self, item_id: str, content: str, *, source: str) -> None:
        return None

    def on_item_event(self, item_id: str, event: str, payload: dict[str, Any]) -> None:
        return None

    def is_delivery_event(self, payload: dict[str, Any]) -> bool:
        return False

    def is_internal_trace(self, text: str) -> bool:
        return False

    def is_read_only_result(self, text: str) -> bool:
        return False

    def sanitize_visible_text(self, text: str) -> str:
        return text

    def get_context_by_token(self, token: str) -> Any:
        return None

    def reset_delivery_tracker(self, key: str) -> None:
        return None

    def extract_targets_from_text(
        self, *, item_id: str, user_message: str, final_text: str, delivery_key: str
    ) -> None:
        return None

    def is_conversation_processing(self, integration_id: str, conversation_key: str) -> bool:
        return False

    def register_background_job_reply(
        self, *, integration_id: str, item_id: str, sender_key: str,
        reply_target: dict, conversation_key: str, conversation_generation: int,
        command: str,
    ) -> str:
        return ""

    def enqueue_background_job_result(
        self, *, integration_id: str, item_id: str, reply_target: dict,
        sender_key: str, conversation_key: str, conversation_generation: int,
        reply_requires_awake: bool, reply_ticket_id: str, message: str,
        pending_reply_id: str = "",
    ) -> bool:
        return False

    def clear_background_job_reply(
        self, *, integration_id: str, conversation_key: str, pending_reply_id: str,
    ) -> None:
        return None

    def reap_stuck_dispatch_jobs(self) -> int:
        return 0


@dataclass(frozen=True)
class DeliveryRetryDecision:
    integration_name: str
    correction_message: dict[str, str]
