from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.models import ItemHandler
from app.services.agent.integrations import (
    get_agent_integration,
    inject_integration_tool_args,
)
from app.services.agent.mcp.server_manager import mcp_server_manager
from app.services.agent.skills.definition import SkillDefinition
from app.services.agent.skills.loader import skill_loader
from app.services.filters.output_filter import OutputFilter, OutputFilterConfig

if TYPE_CHECKING:
    from app.models import Item
    from app.services.agent.mcp.types import MCPTool

logger = logging.getLogger(__name__)

SKILL_ID_ALIASES = {
    "robot_messaging": "qq_mcp",
    "terminal_critical_alert": "terminal_mcp",
    "mutsumi_tone": "mutsumi_persona",
    "tokisaki_kurumi_tone": "tokisaki_kurumi_persona",
    "hirasawa_yui_tone": "hirasawa_yui_persona",
}


@dataclass
class AgentContext:
    handler_id: str
    item_id: str = ""
    model: str | None = None
    api_key: str | None = None
    api_url: str | None = None
    model_parameters: dict[str, Any] = field(default_factory=dict)
    enabled_skills: list[str] = field(default_factory=list)
    enabled_mcp_servers: list[str] = field(default_factory=list)
    enabled_knowledge_files: list[str] = field(default_factory=list)
    skill_revision: int = 0
    output_filter_enabled: bool = False
    output_filter_rules: dict = field(default_factory=dict)
    integration_contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_user_id: str = ""
    current_user_is_superuser: bool = False
    reply_ticket_id: str = ""

    def _integration_state(self, name: str) -> dict[str, Any]:
        return self.integration_contexts.setdefault(name, {})

    def _get_robot_value(self, key: str, default: Any = "") -> Any:
        return self.integration_contexts.get("robot", {}).get(key, default)

    def _set_robot_value(self, key: str, value: Any) -> None:
        self._integration_state("robot")[key] = value

    @property
    def robot_id(self) -> str:
        return str(self._get_robot_value("robot_id", "") or "")

    @robot_id.setter
    def robot_id(self, value: str) -> None:
        self._set_robot_value("robot_id", value)

    @property
    def robot_sender_key(self) -> str:
        return str(self._get_robot_value("sender_key", "") or "")

    @robot_sender_key.setter
    def robot_sender_key(self, value: str) -> None:
        self._set_robot_value("sender_key", value)

    @property
    def robot_context_token(self) -> str:
        return str(self._get_robot_value("context_token", "") or "")

    @robot_context_token.setter
    def robot_context_token(self, value: str) -> None:
        self._set_robot_value("context_token", value)

    @property
    def robot_conversation_key(self) -> str:
        return str(self._get_robot_value("conversation_key", "") or "")

    @robot_conversation_key.setter
    def robot_conversation_key(self, value: str) -> None:
        self._set_robot_value("conversation_key", value)

    @property
    def robot_reply_context_summary(self) -> str:
        return str(self._get_robot_value("reply_context_summary", "") or "")

    @robot_reply_context_summary.setter
    def robot_reply_context_summary(self, value: str) -> None:
        self._set_robot_value("reply_context_summary", value)

    @property
    def robot_mcp_server_transient(self) -> bool:
        return bool(self._get_robot_value("mcp_server_transient", False))

    @robot_mcp_server_transient.setter
    def robot_mcp_server_transient(self, value: bool) -> None:
        self._set_robot_value("mcp_server_transient", bool(value))

    @property
    def robot_known_targets(self) -> list[dict[str, str]]:
        value = self._get_robot_value("known_targets", [])
        return value if isinstance(value, list) else []

    @robot_known_targets.setter
    def robot_known_targets(self, value: list[dict[str, str]]) -> None:
        self._set_robot_value("known_targets", value)

    @property
    def robot_backend_target_resolution_enabled(self) -> bool:
        return bool(self._get_robot_value("backend_target_resolution_enabled", False))

    @robot_backend_target_resolution_enabled.setter
    def robot_backend_target_resolution_enabled(self, value: bool) -> None:
        self._set_robot_value("backend_target_resolution_enabled", bool(value))


class Agent:
    _instances: dict[str, Agent] = {}

    def __new__(cls, handler_id: str):
        if handler_id in cls._instances:
            return cls._instances[handler_id]
        instance = super().__new__(cls)
        cls._instances[handler_id] = instance
        return instance

    def __init__(self, handler_id: str):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True
        self.handler_id = handler_id
        self._context: AgentContext | None = None
        self._skills: dict[str, SkillDefinition] = {}
        self._mcp_servers: list[str] = []
        self._mcp_tools: list[dict] = []
        self._mcp_tools_raw: list[MCPTool] = []
        logger.info(f"[Agent] Created agent for handler {handler_id}")

    @classmethod
    def from_handler(
        cls,
        handler: ItemHandler,
        *,
        instance_key: str | None = None,
    ) -> Agent:
        agent = cls(instance_key or str(handler.id))
        agent._context = AgentContext(
            handler_id=str(handler.id),
            model=handler.model,
            api_key=handler.api_key,
            api_url=handler.api_url,
            model_parameters=handler.model_parameters or {},
            enabled_skills=handler.enabled_skills or [],
            enabled_mcp_servers=(
                handler.enabled_mcp_servers
                if handler.enabled_mcp_servers is not None
                else ["local"]
            ),
            enabled_knowledge_files=handler.enabled_knowledge_files or [],
        )
        agent._load_skills()
        return agent

    @classmethod
    def release_instance(cls, agent: Agent | Any) -> None:
        instance_key = str(getattr(agent, "handler_id", "") or "")
        if instance_key and cls._instances.get(instance_key) is agent:
            cls._instances.pop(instance_key, None)

    def _load_skills(self):
        self._skills.clear()
        self._mcp_servers.clear()
        self._mcp_tools.clear()
        self._mcp_tools_raw.clear()

        if self._context:
            if self._context.enabled_mcp_servers:
                for server_name in self._context.enabled_mcp_servers:
                    if server_name not in self._mcp_servers:
                        self._mcp_servers.append(server_name)
                        logger.info(f"[Agent] Added MCP server from handler config: {server_name}")

            if self._context.enabled_skills:
                for skill_id in self._context.enabled_skills:
                    resolved_skill_id = SKILL_ID_ALIASES.get(skill_id, skill_id)
                    skill = skill_loader.get(resolved_skill_id)
                    if skill:
                        self._skills[resolved_skill_id] = skill
                        if skill.mcp_servers:
                            for server_name in skill.mcp_servers:
                                if server_name not in self._mcp_servers:
                                    self._mcp_servers.append(server_name)
                                    logger.info(
                                        "[Agent] Added MCP server from skill '%s': %s",
                                        resolved_skill_id,
                                        server_name,
                                    )
                    else:
                        logger.warning(f"[Agent] Skill '{skill_id}' not found")

        self._load_mcp_tools()
        if self._context:
            self._context.skill_revision = skill_loader.revision

        logger.info(
            "[Agent] Loaded %s skills (%s), %s MCP servers (%s), %s tools for handler %s",
            len(self._skills),
            sorted(self._skills),
            len(self._mcp_servers),
            self._mcp_servers,
            len(self._mcp_tools),
            self.handler_id,
        )

    def _load_mcp_tools(self):
        self._mcp_tools.clear()
        self._mcp_tools_raw.clear()
        seen_tool_names: set[str] = set()

        for server_name in self._mcp_servers:
            tools = mcp_server_manager.get_tools_for_server(server_name)
            logger.info(f"[Agent] Got {len(tools)} tools from server '{server_name}'")
            for tool in tools:
                full_tool_name = f"mcp_{tool.server_name}_{tool.name}"
                if full_tool_name in seen_tool_names:
                    continue
                seen_tool_names.add(full_tool_name)
                self._mcp_tools_raw.append(tool)
                self._mcp_tools.append(tool.to_litellm_tool())

    @property
    def enabled_knowledge_files(self) -> list[str]:
        if not self._context:
            return []
        return self._context.enabled_knowledge_files

    async def start_mcp_servers(self):
        running_servers: list[str] = []
        for server_name in self._mcp_servers:
            if mcp_server_manager.is_server_running(server_name):
                running_servers.append(server_name)
            elif mcp_server_manager.is_builtin_server_available(server_name):
                if await mcp_server_manager.start_server(server_name):
                    running_servers.append(server_name)
                else:
                    logger.warning(f"[Agent] MCP server '{server_name}' not running")
            else:
                logger.warning(f"[Agent] MCP server '{server_name}' not running")

        if not running_servers:
            return

        # 幂等捷径：运行中的 server 集合没变就不重建工具表——这个函数
        # 每条聊天消息都会被调用一次，全量重建纯属浪费。
        current = frozenset(running_servers)
        if getattr(self, "_mcp_tools_loaded_for", None) == current:
            return
        self._mcp_tools_loaded_for = current

        self._load_mcp_tools()
        for server_name in running_servers:
            tools_count = len(
                [
                    t
                    for t in self._mcp_tools
                    if t.get("function", {}).get("name", "").startswith(f"mcp_{server_name}_")
                ]
            )
            logger.info(f"[Agent] MCP server '{server_name}' running, loaded {tools_count} tools")

    def set_item_context(self, item_id: str, item: Item | None = None):
        if self._context:
            self._context.item_id = item_id
            if item:
                self._context.output_filter_enabled = item.output_filter_enabled
                self._context.output_filter_rules = item.output_filter_rules or {}

    def set_user_context(self, user_id: str, is_superuser: bool = False) -> None:
        if self._context:
            self._context.current_user_id = user_id
            self._context.current_user_is_superuser = is_superuser

    def set_robot_known_targets_from_messages(self, messages: list[dict]) -> None:
        integration = get_agent_integration("robot")
        if integration is None:
            return
        integration.extract_context_targets(self, messages)

    def set_robot_context(
        self,
        *,
        robot_id: str,
        sender_key: str,
        reply_target: Any,
    ) -> None:
        integration = get_agent_integration("robot")
        if integration is None:
            return
        integration.setup_chat_context(
            self,
            {
                "robot_id": robot_id,
                "sender_key": sender_key,
                "reply_target": reply_target,
            },
        )

    async def ensure_robot_context_tools(self) -> None:
        integration = get_agent_integration("robot")
        if integration is None:
            return
        await integration.ensure_chat_context_tools(self, {})

    async def ensure_robot_messaging_tools(self) -> bool:
        integration = get_agent_integration("robot")
        ensure_messaging_tools = getattr(integration, "_ensure_messaging_tools", None)
        if not callable(ensure_messaging_tools):
            return False
        return bool(await ensure_messaging_tools(self))

    def clear_transient_robot_messaging_tools(self) -> None:
        integration = get_agent_integration("robot")
        if integration is not None:
            integration.clear_terminal_alert_tools(self)

    def clear_robot_context(self) -> None:
        integration = get_agent_integration("robot")
        if integration is not None:
            integration.clear_chat_context(self, {})

    def _get_output_filter(self) -> OutputFilter | None:
        if not self._context or not self._context.output_filter_enabled:
            return None

        from app.services.filters.output_filter import FilterRule

        filters = []
        for filter_name, filter_config in self._context.output_filter_rules.items():
            if isinstance(filter_config, dict):
                filter_rule = FilterRule(
                    name=filter_name,
                    regex_patterns=filter_config.get("regex_patterns", []),
                    action_type=filter_config.get("action_type", "ignore"),
                    replace_rules=filter_config.get("action", {}).get("replace_rules", {}),
                )
                filters.append(filter_rule)

        config = OutputFilterConfig(enabled=True, filters=filters)
        return OutputFilter(config)

    def get_skills(self) -> list[SkillDefinition]:
        return list(self._skills.values())

    def reload_skills(self):
        self._load_skills()

    def refresh_from_handler(self, handler: ItemHandler):
        if self._context is None:
            self._context = AgentContext(handler_id=str(handler.id))

        self._context.model = handler.model
        self._context.api_key = handler.api_key
        self._context.api_url = handler.api_url
        self._context.model_parameters = handler.model_parameters or {}
        self._context.enabled_skills = handler.enabled_skills or []
        self._context.enabled_mcp_servers = (
            handler.enabled_mcp_servers
            if handler.enabled_mcp_servers is not None
            else ["local"]
        )
        self._context.enabled_knowledge_files = handler.enabled_knowledge_files or []
        self._load_skills()

    def update_skills(self, enabled_skills: list[str]):
        skill_loader.reload()
        if self._context:
            self._context.enabled_skills = enabled_skills
        self._load_skills()

    def update_mcp_servers(self, enabled_mcp_servers: list[str]):
        if self._context:
            self._context.enabled_mcp_servers = enabled_mcp_servers
        self._load_skills()

    def match_skills(self, query: str) -> list[SkillDefinition]:
        """Deprecated compatibility shim.

        Regex/keyword-based skill matching was removed: skill guides are now
        exposed through the capability catalog and loaded on demand by the
        agent (see app.services.agent.tool_selection). Always returns [].
        """
        del query
        return []

    def get_mcp_servers(self) -> list[str]:
        return self._mcp_servers.copy()

    def get_skip_memory_tools(self) -> list[str]:
        return [
            f"mcp_{tool.server_name}_{tool.name}"
            for tool in self._mcp_tools_raw
            if tool.skip_memory
        ]

    def get_tools_for_litellm(self) -> list[dict]:
        return self._mcp_tools.copy()

    async def execute_tool(self, tool_name: str, args: dict) -> dict:
        if not tool_name.startswith("mcp_"):
            return {"success": False, "error": f"Tool '{tool_name}' is not an MCP tool"}

        parts = tool_name.split("_", 2)
        if len(parts) < 3:
            return {"success": False, "error": f"Invalid MCP tool name: {tool_name}"}

        server_name = parts[1]
        actual_tool_name = parts[2]

        if server_name not in self._mcp_servers:
            return {"success": False, "error": f"MCP server '{server_name}' not available for this agent"}

        inject_integration_tool_args(
            self,
            server_name=server_name,
            tool_name=actual_tool_name,
            args=args,
        )
        if (
            self._context
            and self._context.reply_ticket_id
            and "_reply_ticket_id" not in args
        ):
            args["_reply_ticket_id"] = self._context.reply_ticket_id

        for tool in self._mcp_tools:
            if tool.get("function", {}).get("name") == tool_name:
                input_schema = tool.get("function", {}).get("parameters", {})
                properties = input_schema.get("properties", {})
                if "item_id" in properties and "item_id" not in args:
                    if self._context and self._context.item_id:
                        args["item_id"] = self._context.item_id
                break

        if "command" in args:
            output_filter = self._get_output_filter()
            if output_filter:
                filter_result = output_filter.filter(args["command"])

                if filter_result.is_blocked:
                    logger.warning(f"[Agent] Command blocked by output filter: {filter_result.reason}")
                    return {
                        "success": False,
                        "error": f"Command blocked by security filter: {filter_result.reason}",
                        "blocked": True,
                    }

                if filter_result.action.value == "modified":
                    logger.info(f"[Agent] Command modified by output filter: {args['command']} -> {filter_result.command}")
                    args["command"] = filter_result.command

        try:
            result = await mcp_server_manager.call_tool(server_name, actual_tool_name, args)
            logger.info(f"[Agent] Executed MCP tool '{tool_name}' with args: {args}")
            if self._context and self._context.item_id:
                from app.services.agent.capability_state import record_tool_use

                record_tool_use(self._context.item_id, tool_name)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"[Agent] Error executing MCP tool '{tool_name}': {e}")
            return {"success": False, "error": str(e)}


class AgentManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
        return cls._instance

    def get_or_create(self, handler: ItemHandler) -> Agent:
        handler_id = str(handler.id)

        if handler_id in self._agents:
            agent = self._agents[handler_id]
            context = agent._context
            if context is None:
                agent.refresh_from_handler(handler)
                return agent

            if (
                context.model != handler.model
                or context.api_key != handler.api_key
                or context.api_url != handler.api_url
                or context.model_parameters != (handler.model_parameters or {})
                or context.enabled_skills != (handler.enabled_skills or [])
                or context.enabled_mcp_servers != (handler.enabled_mcp_servers or [])
                or context.enabled_knowledge_files != (handler.enabled_knowledge_files or [])
                or context.skill_revision != skill_loader.revision
            ):
                if context.skill_revision == skill_loader.revision:
                    skill_loader.reload()
                agent.refresh_from_handler(handler)
            return agent

        agent = Agent.from_handler(handler)
        self._agents[handler_id] = agent
        return agent

    def refresh_cached(self, handler: ItemHandler) -> Agent | None:
        handler_id = str(handler.id)
        agent = self._agents.get(handler_id)
        if not agent:
            return None

        skill_loader.reload()
        agent.refresh_from_handler(handler)
        return agent


class ItemHandlerContext:
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
        self._item_handlers: dict[str, str] = {}
        self._global_lock = threading.RLock()
        logger.info("[ItemHandlerContext] Initialized")

    def set_handler(self, item_id: str, handler_id: str):
        with self._global_lock:
            self._item_handlers[item_id] = handler_id
            logger.info(f"[ItemHandlerContext] Set handler={handler_id} for item={item_id}")

    def get_handler(self, item_id: str) -> str | None:
        with self._global_lock:
            return self._item_handlers.get(item_id)

    def remove_handler(self, item_id: str):
        with self._global_lock:
            if item_id in self._item_handlers:
                del self._item_handlers[item_id]
                logger.info(f"[ItemHandlerContext] Removed handler for item={item_id}")


agent_manager = AgentManager()
item_handler_context = ItemHandlerContext()
