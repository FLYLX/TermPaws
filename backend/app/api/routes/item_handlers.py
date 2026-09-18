import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlmodel import col, select

from app.api.deps import CurrentUser, SessionDep
from app.models import (
    ItemHandler,
    ItemHandlerCreate,
    ItemHandlerItem,
    ItemHandlerPublic,
    ItemHandlerUpdate,
    ItemHandlerUser,
    Message,
)
from app.services.agent.agent import agent_manager
from app.services.agent.knowledge import knowledge_base_service
from app.services.agent.model_parameters import normalize_model_parameters_for_model
from app.services.llm_health_service import llm_health_service

router = APIRouter(prefix="/item-handlers", tags=["item-handlers"])


class KnowledgeFileItem(BaseModel):
    path: str
    name: str
    size: int | None = None
    modified_at: int | None = None
    enabled: bool
    indexed: bool
    missing: bool
    chunk_count: int


class KnowledgeFileListResponse(BaseModel):
    data: list[KnowledgeFileItem]
    count: int
    enabled_count: int


class ItemHandlerLlmStatusItem(BaseModel):
    item_handler_id: uuid.UUID
    status: str
    reachable: bool
    message: str | None = None
    checked_at: datetime
    cached: bool = False


class ItemHandlerLlmStatusListResponse(BaseModel):
    data: list[ItemHandlerLlmStatusItem]
    count: int


class ItemHandlerSummaryPublic(ItemHandlerPublic):
    item_count: int = 0
    user_count: int = 0


def _normalized_api_key(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _build_item_handler_public(item_handler: ItemHandler) -> ItemHandlerPublic:
    return ItemHandlerPublic.model_validate(
        item_handler,
        update={"has_api_key": bool(_normalized_api_key(item_handler.api_key))},
    )


def _get_item_handler_or_404(session: SessionDep, item_handler_id: uuid.UUID) -> ItemHandler:
    item_handler = session.get(ItemHandler, item_handler_id)
    if not item_handler:
        raise HTTPException(status_code=404, detail="Item handler not found")
    return item_handler


def _assert_item_handler_permission(item_handler: ItemHandler, current_user: CurrentUser) -> None:
    if not current_user.is_superuser and item_handler.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")


def _build_item_handler_summaries(
    session: SessionDep,
    item_handlers: list[ItemHandler],
) -> list[ItemHandlerSummaryPublic]:
    handler_ids = [handler.id for handler in item_handlers]
    if not handler_ids:
        return []

    item_count_rows = session.exec(
        select(ItemHandlerItem.item_handler_id, func.count())
        .where(col(ItemHandlerItem.item_handler_id).in_(handler_ids))
        .group_by(ItemHandlerItem.item_handler_id)
    ).all()
    user_count_rows = session.exec(
        select(ItemHandlerUser.item_handler_id, func.count())
        .where(col(ItemHandlerUser.item_handler_id).in_(handler_ids))
        .group_by(ItemHandlerUser.item_handler_id)
    ).all()
    item_counts = dict(item_count_rows)
    user_counts = dict(user_count_rows)

    return [
        ItemHandlerSummaryPublic.model_validate(
            handler,
            update={
                "has_api_key": bool(_normalized_api_key(handler.api_key)),
                "item_count": item_counts.get(handler.id, 0),
                "user_count": user_counts.get(handler.id, 0),
            },
        )
        for handler in item_handlers
    ]


@router.get("/", response_model=list[ItemHandlerSummaryPublic])
def read_item_handlers(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> Any:
    """
    Retrieve item handlers.
    """

    if current_user.is_superuser:
        statement = (
            select(ItemHandler)
            .order_by(col(ItemHandler.created_at).desc())
            .offset(skip)
            .limit(limit)
        )
    else:
        statement = (
            select(ItemHandler)
            .where(ItemHandler.owner_id == current_user.id)
            .order_by(col(ItemHandler.created_at).desc())
            .offset(skip)
            .limit(limit)
        )

    return _build_item_handler_summaries(session, session.exec(statement).all())


@router.get("/llm/status", response_model=ItemHandlerLlmStatusListResponse)
def read_item_handler_llm_statuses(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
    force: bool = False,
) -> ItemHandlerLlmStatusListResponse:
    if current_user.is_superuser:
        statement = (
            select(ItemHandler)
            .order_by(col(ItemHandler.created_at).desc())
            .offset(skip)
            .limit(limit)
        )
    else:
        statement = (
            select(ItemHandler)
            .where(ItemHandler.owner_id == current_user.id)
            .order_by(col(ItemHandler.created_at).desc())
            .offset(skip)
            .limit(limit)
        )

    item_handlers = session.exec(statement).all()
    statuses = [
        ItemHandlerLlmStatusItem.model_validate(item)
        for item in llm_health_service.get_statuses(item_handlers, force=force)
    ]
    return ItemHandlerLlmStatusListResponse(
        data=statuses,
        count=len(statuses),
    )


@router.get("/{id}", response_model=ItemHandlerSummaryPublic)
def read_item_handler(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> Any:
    """
    Get item handler by ID.
    """

    item_handler = _get_item_handler_or_404(session, id)
    _assert_item_handler_permission(item_handler, current_user)
    return _build_item_handler_summaries(session, [item_handler])[0]


@router.post("/", response_model=ItemHandlerPublic)
def create_item_handler(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    item_handler_in: ItemHandlerCreate,
) -> Any:
    """
    Create new item handler.
    """

    existing_item_handler = session.exec(
        select(ItemHandler).where(
            ItemHandler.name == item_handler_in.name,
            ItemHandler.owner_id == current_user.id,
        )
    ).first()

    if existing_item_handler:
        raise HTTPException(
            status_code=400,
            detail="An item handler with this name already exists for your account.",
        )

    create_data = item_handler_in.model_dump()
    create_data["api_key"] = _normalized_api_key(create_data.get("api_key"))
    create_data["model_parameters"] = normalize_model_parameters_for_model(
        create_data.get("model"),
        create_data.get("model_parameters"),
    )
    item_handler = ItemHandler(**create_data, owner_id=current_user.id)

    session.add(item_handler)
    session.commit()
    session.refresh(item_handler)

    return _build_item_handler_public(item_handler)


@router.put("/{id}", response_model=ItemHandlerPublic)
def update_item_handler(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    item_handler_in: ItemHandlerUpdate,
) -> Any:
    """
    Update an item handler.
    """

    item_handler = _get_item_handler_or_404(session, id)
    _assert_item_handler_permission(item_handler, current_user)

    if item_handler_in.name and item_handler_in.name != item_handler.name:
        existing_item_handler = session.exec(
            select(ItemHandler).where(
                ItemHandler.name == item_handler_in.name,
                ItemHandler.owner_id == current_user.id,
                ItemHandler.id != id,
            )
        ).first()

        if existing_item_handler:
            raise HTTPException(
                status_code=400,
                detail="An item handler with this name already exists for your account.",
            )

    update_dict = item_handler_in.model_dump(exclude_unset=True)
    clear_api_key = bool(update_dict.pop("clear_api_key", False))
    if clear_api_key:
        update_dict["api_key"] = None
    elif "api_key" in update_dict:
        replacement_api_key = _normalized_api_key(update_dict.get("api_key"))
        if replacement_api_key:
            update_dict["api_key"] = replacement_api_key
        else:
            update_dict.pop("api_key", None)
    if "enabled_knowledge_files" in update_dict:
        update_dict["enabled_knowledge_files"] = knowledge_base_service.normalize_enabled_files(
            update_dict.get("enabled_knowledge_files")
        )
    if "model" in update_dict or "model_parameters" in update_dict:
        update_dict["model_parameters"] = normalize_model_parameters_for_model(
            update_dict.get("model", item_handler.model),
            update_dict.get("model_parameters", item_handler.model_parameters),
        )
    item_handler.sqlmodel_update(update_dict)

    session.add(item_handler)
    session.commit()
    session.refresh(item_handler)

    if update_dict:
        agent_manager.refresh_cached(item_handler)

    from app.services.agent.chat_runtime import invalidate_handler_cache

    invalidate_handler_cache()

    return _build_item_handler_public(item_handler)


@router.delete("/{id}")
def delete_item_handler(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> Message:
    """
    Delete an item handler.
    """

    item_handler = _get_item_handler_or_404(session, id)
    _assert_item_handler_permission(item_handler, current_user)

    session.delete(item_handler)
    session.commit()

    from app.services.agent.chat_runtime import invalidate_handler_cache

    invalidate_handler_cache()

    return Message(message="Item handler deleted successfully")


@router.get("/{id}/knowledge/files", response_model=KnowledgeFileListResponse)
def list_item_handler_knowledge_files(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> KnowledgeFileListResponse:
    item_handler = _get_item_handler_or_404(session, id)
    _assert_item_handler_permission(item_handler, current_user)

    files = [
        KnowledgeFileItem.model_validate(item)
        for item in knowledge_base_service.list_files(
            item_handler.enabled_knowledge_files or [],
        )
    ]
    return KnowledgeFileListResponse(
        data=files,
        count=len(files),
        enabled_count=len([item for item in files if item.enabled]),
    )
