import logging
import time

from sqlmodel import Session, select

from app.core.db import engine
from app.models import Item, ItemHandlerItem
from app.services.agent.prompts.builder import is_critical_terminal_event

logger = logging.getLogger(__name__)

# 空闲 agent 被"未经过滤的终端输出"触发轮次的冷却：只放行严重事件且限速，
# 否则高音量终端（如游戏服务器日志）会让 agent 每 ~1.2s 跑一轮完整 LLM turn，
# Python 对象高速churn导致内存高水位、并持续烧 LLM token。
_UNSOLICITED_TURN_COOLDOWN_SECONDS = 120.0


class AgentInputBridge:
    """
    Convert terminal stream payloads into agent session input.

    Responsibilities:
    - resolve which handler is attached to an item
    - load item filter configuration
    - pass raw + filtered terminal output into stream_manager
    """

    def __init__(self) -> None:
        self._last_unsolicited_turn_at: dict[str, float] = {}

    def _should_skip_unsolicited_output(
        self, item_uuid: str, item: Item | None, raw_output: str
    ) -> bool:
        from app.services.agent.session import agent_session_manager

        session = agent_session_manager.get_session(item_uuid)
        if session and session.has_pending_command():
            return False
        if item and item.input_filter_enabled:
            return False
        if is_critical_terminal_event(raw_output):
            return False
        # 有 handler 负责的终端：agent 的工作对象就是它，输出是它的眼睛——
        # 按冷却放行非严重输出，而不是"仅严重事件可见"让它变瞎。
        # 无 handler 的终端维持抑制（没人在看，别烧 token）。
        if not self._load_handler_id(item_uuid):
            return True
        now = time.monotonic()
        last = self._last_unsolicited_turn_at.get(item_uuid, 0.0)
        if now - last < _UNSOLICITED_TURN_COOLDOWN_SECONDS:
            return True
        self._last_unsolicited_turn_at[item_uuid] = now
        return False

    def handle_stream(self, item_uuid: str, data: dict[str, object]) -> None:
        if data.get("source") == "job":
            logger.info(
                "[AgentInputBridge] Skipping daemon job stream for item %s", item_uuid
            )
            return
        logger.info(
            "[AgentInputBridge] handle_stream: item=%s stdout_len=%s stderr_len=%s",
            item_uuid,
            len(str(data.get("stdout") or "")),
            len(str(data.get("stderr") or "")),
        )
        try:
            from app.services.agent import item_handler_context
            from app.services.agent.stream_manager import stream_manager

            raw_output = self._build_raw_output(data)
            if not raw_output.strip():
                return

            item = self._load_item(item_uuid)
            if self._should_skip_unsolicited_output(item_uuid, item, raw_output):
                logger.info(
                    "[AgentInputBridge] Suppressed unsolicited terminal output for item %s "
                    "(no pending command, filter disabled, non-critical)",
                    item_uuid,
                )
                return
            filtered_output = self._build_filtered_output(item, data, raw_output)

            handler_id = item_handler_context.get_handler(
                item_uuid
            ) or self._load_handler_id(item_uuid)
            if not handler_id:
                logger.info(
                    "[AgentInputBridge] No handler found for item %s", item_uuid
                )
                return

            stream_manager.process_stream(
                item_uuid,
                filtered_output,
                handler_id,
                raw_output=raw_output,
            )
            logger.info(
                "[AgentInputBridge] Processed stream for item=%s, handler=%s, filtered_len=%s, raw_len=%s",
                item_uuid,
                handler_id,
                len(filtered_output),
                len(raw_output),
            )
        except Exception as exc:
            logger.error(
                "[AgentInputBridge] Failed to bridge stream for item %s: %s",
                item_uuid,
                exc,
                exc_info=True,
            )

    def _load_handler_id(self, item_uuid: str) -> str | None:
        import uuid as _uuid

        try:
            item_key = _uuid.UUID(str(item_uuid))
        except (TypeError, ValueError):
            item_key = str(item_uuid)
        with Session(engine) as session:
            handler_item = session.exec(
                select(ItemHandlerItem).where(ItemHandlerItem.item_id == item_key)
            ).first()
            if handler_item is None and item_key is not item_uuid:
                handler_item = session.exec(
                    select(ItemHandlerItem).where(ItemHandlerItem.item_id == item_uuid)
                ).first()
            if handler_item:
                return str(handler_item.item_handler_id)
        return None

    def _load_item(self, item_uuid: str) -> Item | None:
        with Session(engine) as session:
            return session.exec(select(Item).where(Item.id == item_uuid)).first()

    def _build_raw_output(self, data: dict[str, object]) -> str:
        stdout = data.get("stdout", "")
        stderr = data.get("stderr", "")
        return f"{stdout}{stderr}"

    def _build_filtered_output(
        self,
        item: Item | None,
        data: dict[str, object],
        raw_output: str,
    ) -> str:
        if not item or not item.input_filter_enabled:
            return raw_output

        try:
            from app.services.filters.input_filter import InputFilter, InputFilterConfig

            config = InputFilterConfig.from_item(item)
            input_filter = InputFilter(config)
            event = input_filter.filter(data)
            return event.raw_content if event else ""
        except Exception as exc:
            logger.debug(
                "[AgentInputBridge] Filter error for item %s: %s", item.id, exc
            )
            return raw_output
