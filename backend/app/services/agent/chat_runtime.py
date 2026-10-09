from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from fastapi import HTTPException
from sqlmodel import Session, select

from app.models import Item, ItemHandler, ItemHandlerItem, User
from app.services.agent.agent import Agent, agent_manager, item_handler_context
from app.services.agent.integrations import (
    clear_integration_chat_contexts,
    ensure_integration_chat_context_tools,
    fallback_is_delivery_result,
    integration_fallback_response_content,
    integration_message_sent,
    record_integration_no_final_response,
    send_integration_final_response_fallback,
    setup_integration_chat_contexts,
)
from app.services.agent.integrations.hooks import (
    is_integration_delivery_event,
    sanitize_integration_visible_text,
    is_integration_internal_trace,
    is_integration_read_only_result,
)
from app.services.agent.persona_guard import enforce_persona_identity_response

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from app.services.agent.agent import Agent


@dataclass(frozen=True)
class ChatResponseResult:
    content: str
    robot_message_sent: bool = False


def _looks_like_internal_tool_trace(value: str) -> bool:
    return is_integration_internal_trace(str(value or ""))


def _is_integration_internal_response(value: str) -> bool:
    normalized = str(value or "").strip()
    if not normalized:
        return False
    if _looks_like_internal_tool_trace(normalized) or fallback_is_delivery_result(
        normalized
    ):
        return True
    return is_integration_read_only_result(normalized)


def _sanitize_integration_response(value: str) -> str:
    return sanitize_integration_visible_text(str(value or ""))


# item → (handler, item) 解析缓存：每条聊天消息 3 次 SELECT 在突发流量下
# 会直接打满连接池。60s TTL 兜住 handler/归属变更的过期窗口；权限判断
# 用的是缓存里的 owner_id，语义与实时查询一致。
_HANDLER_CACHE_TTL_SECONDS = 60.0
_handler_cache_lock = threading.Lock()
_handler_cache: dict[str, tuple[float, ItemHandler, Item]] = {}


def invalidate_handler_cache(item_id: str | None = None) -> None:
    with _handler_cache_lock:
        if item_id:
            _handler_cache.pop(str(item_id), None)
        else:
            _handler_cache.clear()


def _get_item_handler_cached(
    session: Session,
    item_id: str,
    user: User,
) -> tuple[ItemHandler, Item] | None:
    key = str(item_id)
    now = time.monotonic()
    with _handler_cache_lock:
        cached = _handler_cache.get(key)
    if cached is not None and now - cached[0] < _HANDLER_CACHE_TTL_SECONDS:
        _t, handler, item = cached
    else:
        resolved = get_item_handler_llm_config(session, item_id, user)
        if resolved is None:
            return None
        _t_handler, _t_item = resolved
        with _handler_cache_lock:
            _handler_cache[key] = (now, _t_handler, _t_item)
        handler, item = _t_handler, _t_item
    # 权限判断必须每请求执行（用缓存里的归属数据，语义不变）
    if not user.is_superuser and item.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return handler, item


def get_item_handler_llm_config(
    session: Session,
    item_id: str,
    user: User,
) -> tuple[ItemHandler, Item] | None:
    item = session.get(Item, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if not user.is_superuser and item.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    handler_item = session.exec(
        select(ItemHandlerItem).where(ItemHandlerItem.item_id == item_id)
    ).first()
    if not handler_item:
        return None

    handler = session.get(ItemHandler, handler_item.item_handler_id)
    if not handler:
        return None

    return handler, item


async def prepare_chat_agent(
    session: Session,
    item_id: str,
    current_user: User,
    prepared: tuple[ItemHandler, Item] | None = None,
    isolated: bool = False,
) -> tuple[ItemHandler, Item, Agent]:
    result = prepared or get_item_handler_llm_config(session, item_id, current_user)
    if not result:
        raise HTTPException(
            status_code=404,
            detail="No ItemHandler associated with this item. Please associate an ItemHandler first.",
        )

    handler, item = result
    if not handler.model:
        raise HTTPException(
            status_code=400,
            detail=f"ItemHandler '{handler.name}' has no model configured.",
        )

    agent = (
        Agent.from_handler(
            handler,
            instance_key=f"{handler.id}:turn:{uuid4().hex}",
        )
        if isolated
        else agent_manager.get_or_create(handler)
    )
    agent.set_item_context(item_id, item)
    agent.set_user_context(str(current_user.id), bool(current_user.is_superuser))
    clear_robot_context = getattr(agent, "clear_robot_context", None)
    if callable(clear_robot_context):
        clear_robot_context()
    item_handler_context.set_handler(item_id, str(handler.id))
    await agent.start_mcp_servers()
    return handler, item, agent


async def _collect_chat_response_unserialized(
    *,
    session: Session,
    item_id: str,
    current_user: User,
    message: str,
    history: list[Any] | None = None,
    robot_id: str | None = None,
    robot_sender_key: str | None = None,
    robot_reply_target: Any | None = None,
    robot_conversation_key: str | None = None,
    robot_conversation_generation: int = 0,
    robot_reply_requires_awake: bool = False,
    robot_request_message: str = "",
    reply_ticket_id: str = "",
    return_result: bool = False,
    prepared: tuple[ItemHandler, Item] | None = None,
    prepared_agent: tuple[ItemHandler, Item, Agent] | None = None,
    image_urls: list[str] | None = None,
    force_image_vision: bool = False,
    agent_image_urls: list[str] | None = None,
) -> str | ChatResponseResult:
    from app.api.routes.chat import generate_stream

    if prepared_agent is not None:
        handler, _, agent = prepared_agent
    else:
        handler, _, agent = await prepare_chat_agent(
            session,
            item_id,
            current_user,
            prepared=prepared,
        )

    integration_contexts: dict[str, dict[str, Any]] = {}
    if robot_id and robot_sender_key and robot_reply_target:
        integration_contexts["robot"] = {
            "robot_id": robot_id,
            "sender_key": robot_sender_key,
            "reply_target": robot_reply_target,
            "conversation_key": robot_conversation_key or "",
            "conversation_generation": robot_conversation_generation,
            "reply_requires_awake": robot_reply_requires_awake,
            "request_message": robot_request_message,
        }
        setup_integration_chat_contexts(agent, integration_contexts)
        await ensure_integration_chat_context_tools(agent, integration_contexts)
        logger.info("[ChatRuntime] robot context registered: robot_id=%s source=qq", robot_id)
    else:
        logger.warning("[ChatRuntime] robot context NOT registered: robot_id=%s sender=%s target=%s -> source=web", robot_id, robot_sender_key, type(robot_reply_target).__name__)

    content = ""
    error_message = ""
    tool_results: list[str] = []
    warnings: list[str] = []
    done_seen = False
    robot_message_sent = False
    try:
        for chunk in generate_stream(
            message=message,
            history=history or [],
            handler=handler,
            item_id=item_id,
            agent=agent,
            include_hidden_tool_results=True,
            latest_only_context=bool(integration_contexts),
            source_type="qq" if integration_contexts else "web",
            reply_ticket_id=reply_ticket_id,
            turn_serialized=True,
            image_urls=image_urls,
            force_image_vision=force_image_vision,
            agent_image_urls=agent_image_urls,
        ):
            if not chunk.startswith("data: "):
                continue

            payload = json.loads(chunk[6:].strip())
            if payload.get("type") == "agent_response":
                candidate = str(payload.get("content") or "")
                if integration_contexts:
                    sanitized_candidate = _sanitize_integration_response(candidate)
                    if _is_integration_internal_response(candidate):
                        tool_results.append(candidate)
                    elif sanitized_candidate:
                        content = sanitized_candidate
                    elif candidate:
                        tool_results.append(candidate)
                elif candidate:
                    content = candidate
            elif payload.get("type") in {"agent_error", "error"}:
                error_message = str(payload.get("content") or error_message)
            elif payload.get("type") == "agent_tool_result":
                tool_results.append(str(payload.get("content") or ""))
            elif payload.get("type") == "agent_warning":
                warnings.append(str(payload.get("content") or ""))
            elif payload.get("done") is True:
                done_seen = True
            elif is_integration_delivery_event(payload):
                done_seen = True
                if integration_contexts:
                    robot_message_sent = True
                continue
    finally:
        if integration_contexts:
            clear_integration_chat_contexts(agent, integration_contexts)

    if error_message:
        raise HTTPException(status_code=500, detail=error_message)
    if integration_contexts and _is_integration_internal_response(content):
        tool_results.append(content)
        content = ""
    if integration_contexts and content:
        content = _sanitize_integration_response(content)
    if content:
        content = enforce_persona_identity_response(agent, message, content)
    robot_message_sent = robot_message_sent or (bool(integration_contexts) and integration_message_sent(tool_results))
    if robot_message_sent and reply_ticket_id:
        try:
            from app.services.agent.reply_ticket import reply_ticket_manager

            reply_ticket_manager.mark_external_report_sent(reply_ticket_id)
        except Exception:
            pass
    if robot_message_sent and content.strip():
        logger.info(
            "[ChatRuntime] Suppressed final response after robot delivery tool sent for item %s",
            item_id,
        )
        content = ""
    if content.strip() and not robot_message_sent:
        robot_message_sent = send_integration_final_response_fallback(
            integration_contexts,
            content=content,
            message_sent=robot_message_sent,
        )

    if not content:
        fallback_content = integration_fallback_response_content(
            integration_contexts,
            tool_results=tool_results,
            warnings=warnings,
            done_seen=done_seen,
        )
        if fallback_content:
            record_integration_no_final_response(
                integration_contexts,
                fallback_content=fallback_content,
                tool_results=tool_results,
                warnings=warnings,
            )
            fallback_delivered = fallback_is_delivery_result(fallback_content)
            result = ChatResponseResult(
                content="" if fallback_delivered else fallback_content,
                robot_message_sent=robot_message_sent or fallback_delivered,
            )
            return result if return_result else result.content
        raise HTTPException(
            status_code=502,
            detail="Agent did not return any response content.",
        )
    result = ChatResponseResult(content=content, robot_message_sent=robot_message_sent)
    return result if return_result else result.content


async def collect_chat_response(
    *,
    session: Session,
    item_id: str,
    current_user: User,
    message: str,
    history: list[Any] | None = None,
    robot_id: str | None = None,
    robot_sender_key: str | None = None,
    robot_reply_target: Any | None = None,
    robot_conversation_key: str | None = None,
    robot_conversation_generation: int = 0,
    robot_reply_requires_awake: bool = False,
    robot_request_message: str = "",
    reply_ticket_id: str = "",
    return_result: bool = False,
    image_urls: list[str] | None = None,
    force_image_vision: bool = False,
    agent_image_urls: list[str] | None = None,
) -> str | ChatResponseResult:
    from app.services.agent.turn_coordinator import (
        agent_turn_coordinator,
        agent_turn_key,
    )

    prepared_agent = None
    try:
        prepared = get_item_handler_llm_config(session, item_id, current_user)
    except AttributeError:
        prepared = None

    if prepared:
        handler_id = str(prepared[0].id)
    else:
        prepared_agent = await prepare_chat_agent(session, item_id, current_user)
        handler = prepared_agent[0]
        handler_id = str(getattr(handler, "id", "") or handler.name)
    # 并发单元是会话，不是 handler：QQ 各群/私聊的回合互不阻塞，
    # 同一会话内保持串行（时序正确性）。回合级 robot 状态是线程隔离的
    # （AgentContext._robot_turn_state），并行回合结构上互不干扰。
    lease_key = agent_turn_key(handler_id)
    if robot_conversation_key:
        lease_key = f"{lease_key}:qq:{robot_conversation_key}"
    lease = await agent_turn_coordinator.acquire_async(lease_key)
    try:
        return await _collect_chat_response_unserialized(
            session=session,
            item_id=item_id,
            current_user=current_user,
            message=message,
            history=history,
            robot_id=robot_id,
            robot_sender_key=robot_sender_key,
            robot_reply_target=robot_reply_target,
            robot_conversation_key=robot_conversation_key,
            robot_conversation_generation=robot_conversation_generation,
            robot_reply_requires_awake=robot_reply_requires_awake,
            robot_request_message=robot_request_message,
            reply_ticket_id=reply_ticket_id,
            return_result=return_result,
            prepared=prepared,
            prepared_agent=prepared_agent,
            image_urls=image_urls,
            force_image_vision=force_image_vision,
            agent_image_urls=agent_image_urls,
        )
    finally:
        lease.release()
