import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field
from sqlmodel import col, select

from app.api.deps import CurrentUser, SessionDep
from app.models import (
    Item,
    Message,
    Robot,
    RobotCreate,
    RobotItem,
    RobotPublic,
    RobotsPublic,
    RobotUpdate,
)
from app.services.agent.reply_ticket import SOURCE_QQ, reply_ticket_manager

from . import is_robot_plugin_enabled
from .api_support import (
    assert_bridge_permission,
    assert_item_permission,
    assert_robot_permission,
    clear_default_targets,
    current_robot_config,
    ensure_unique_binding_route_key,
    ensure_unique_robot_name,
    get_item_or_404,
    get_robot_or_404,
    normalize_binding_payload,
    normalize_robot_stack,
    serialize_binding,
    validate_binding_payload,
)
from .bridge_client import robot_bridge_client
from .contracts import RobotDispatchResponse, RobotInboundMessage, RobotReplyTarget
from .conversation_memory import (
    conversation_key_from_reply_target,
    robot_conversation_memory,
)
from .debug_log import get_robot_events, record_robot_event
from .platforms import (
    RobotPlatformPublic,
    get_robot_platform,
    get_robot_runtime_config,
    list_supported_robot_platforms,
    normalize_robot_config,
    normalize_robot_platform_id,
    resolve_robot_identity,
)
from .schemas import (
    RobotItemBindingCreate,
    RobotItemBindingPublic,
    RobotItemBindingUpdate,
)
from .service import robot_service


def ensure_robot_plugin_enabled() -> None:
    if not is_robot_plugin_enabled():
        raise HTTPException(status_code=404, detail="Robot plugin is disabled")


router = APIRouter(
    prefix="/robots",
    tags=["robots"],
    dependencies=[Depends(ensure_robot_plugin_enabled)],
)


REPLY_TICKET_PROCESSING_STALE = timedelta(minutes=30)


def _reply_ticket_snapshot_stale(ticket: dict[str, object]) -> bool:
    raw = ticket.get("updated_at") or ticket.get("created_at")
    if not raw:
        return True
    try:
        updated_at = datetime.fromisoformat(str(raw))
    except ValueError:
        return True
    if updated_at.tzinfo is not None:
        updated_at = updated_at.astimezone(timezone.utc).replace(tzinfo=None)
    return datetime.now() - updated_at > REPLY_TICKET_PROCESSING_STALE


def _merge_active_reply_tickets_into_controller_snapshots(
    snapshots_by_robot: dict[str, list[dict[str, object]]],
    *,
    item_id: uuid.UUID | str,
    robot_ids: set[uuid.UUID | str],
) -> None:
    allowed_robot_ids = {str(robot_id) for robot_id in robot_ids}
    tickets = reply_ticket_manager.snapshot(str(item_id))
    for ticket in tickets:
        robot_id = str(ticket.get("robot_id") or "")
        conversation_key = str(ticket.get("conversation_key") or "")
        status = str(ticket.get("status") or "")
        if (
            ticket.get("source_type") != SOURCE_QQ
            or status in {"delivered", "failed"}
            or robot_id not in allowed_robot_ids
            or not conversation_key
            or _reply_ticket_snapshot_stale(ticket)
        ):
            continue

        robot_snapshots = snapshots_by_robot.setdefault(robot_id, [])
        controller = next(
            (
                snapshot
                for snapshot in robot_snapshots
                if str(snapshot.get("conversation_key") or "") == conversation_key
            ),
            None,
        )
        if controller is None:
            conversation_type, _, conversation_id = conversation_key.partition(":")
            updated_at = str(ticket.get("updated_at") or ticket.get("created_at") or "")
            controller = {
                "robot_id": robot_id,
                "conversation_key": conversation_key,
                "conversation_type": conversation_type,
                "conversation_id": conversation_id,
                "item_id": str(item_id),
                "status": "processing",
                "awake": True,
                "sleeping": False,
                "processing": True,
                "generation": int(ticket.get("conversation_generation") or 0),
                "expires_at": None,
                "processing_expires_at": None,
                "processing_seconds_remaining": 0,
                "updated_at": updated_at,
                "seconds_remaining": 0,
                "pending_count": 0,
                "pending_messages": [],
            }
            robot_snapshots.append(controller)

        pending_messages = list(controller.get("pending_messages") or [])
        ticket_id = str(ticket.get("ticket_id") or "")
        if any(
            str(message.get("reply_ticket_id") or "") == ticket_id
            for message in pending_messages
            if isinstance(message, dict)
        ):
            continue

        request_message = re.sub(
            r"\s+",
            " ",
            str(ticket.get("request_message") or ""),
        ).strip()
        command = re.sub(r"\s+", " ", str(ticket.get("command") or "")).strip()
        preview = request_message or command or "等待回复原来源"
        if len(preview) > 180:
            preview = f"{preview[:177]}..."
        if len(command) > 140:
            command = f"{command[:137]}..."
        workflow = ticket.get("workflow")
        workflow = workflow if isinstance(workflow, dict) else {}

        pending_messages.append(
            {
                "index": len(pending_messages) + 1,
                "item_id": str(item_id),
                "route_key": "reply_ticket",
                "sender_key": str(ticket.get("sender_key") or ""),
                "sender_label": str(
                    ticket.get("sender_label")
                    or ticket.get("sender_key")
                    or ticket.get("source_label")
                    or "QQ"
                ),
                "trigger_reason": "reply_ticket",
                "message_preview": preview,
                "enqueued_at": str(ticket.get("created_at") or ""),
                "direct_wakeup": True,
                "reply_ticket_id": ticket_id,
                "reply_ticket_status": str(ticket.get("status") or "pending"),
                "task_request_id": str(ticket.get("task_request_id") or ""),
                "command_preview": command,
                "delivery_error": str(ticket.get("delivery_error") or ""),
                "workflow_id": str(workflow.get("workflow_id") or ""),
                "workflow_objective": str(workflow.get("objective") or ""),
                "workflow_status": str(workflow.get("status") or ""),
                "workflow_current_step": str(
                    workflow.get("current_step") or ""
                ),
                "workflow_latest_progress": str(
                    workflow.get("latest_progress") or ""
                ),
                "workflow_blocker": str(workflow.get("blocker") or ""),
                "workflow_steps": workflow.get("steps")
                if isinstance(workflow.get("steps"), list)
                else [],
            }
        )
        pending_messages.sort(key=lambda message: str(message.get("enqueued_at") or ""))
        for index, message in enumerate(pending_messages, start=1):
            message["index"] = index
        controller["pending_messages"] = pending_messages
        controller["pending_count"] = len(pending_messages)

CONVERSATION_KEY_PATTERN = r"^(group|private|channel):[^/\\\r\n]+$"


class RobotConversationMemoryImportBody(BaseModel):
    content: str = Field(..., max_length=2 * 1024 * 1024)
    append: bool = False



class RobotDebugSendBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=1200)


class RobotManualSendBody(BaseModel):
    target_type: str = Field(..., min_length=1, max_length=32)
    target_id: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1, max_length=4000)


def _remember_sent_robot_message(
    robot_id: uuid.UUID | str,
    target: RobotReplyTarget,
    text: str,
) -> None:
    try:
        robot_conversation_memory.append_assistant_message(
            str(robot_id),
            conversation_key_from_reply_target(target),
            text,
        )
    except Exception as exc:
        record_robot_event(
            str(robot_id),
            direction="backend",
            event="conversation_memory_write_failed",
            status="error",
            message=str(exc) or exc.__class__.__name__,
        )


def _generate_robot_access_token() -> str:
    return secrets.token_urlsafe(32)


def _ensure_onebot_access_token(
    platform_id: str,
    config: dict,
) -> dict:
    if normalize_robot_platform_id(platform_id) != "onebot_v11":
        return config

    credentials = config.get("credentials")
    if not isinstance(credentials, dict):
        credentials = {}
    access_token = str(credentials.get("access_token") or "").strip()
    if access_token:
        return config

    return {
        **config,
        "credentials": {
            **credentials,
            "access_token": _generate_robot_access_token(),
        },
    }


def _event_seen(events: list[dict], names: set[str]) -> bool:
    return any(str(event.get("event") or "") in names for event in events)


def _latest_reply_target_from_debug_events(robot_id: uuid.UUID) -> RobotReplyTarget | None:
    for event in get_robot_events(str(robot_id), limit=200):
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        raw_target = payload.get("reply_target")
        if not isinstance(raw_target, dict):
            continue
        try:
            return RobotReplyTarget.model_validate(raw_target)
        except Exception:
            continue
    return None


def _manual_send_target(body: RobotManualSendBody) -> RobotReplyTarget:
    raw_target_type = body.target_type.strip().lower()
    target_id = body.target_id.strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="Target ID is required")

    private_types = {"private", "friend", "user", "direct", "c2c"}
    group_types = {"group", "qq_group"}
    if raw_target_type in private_types:
        target_type = "private"
    elif raw_target_type in group_types:
        target_type = "group"
    else:
        raise HTTPException(
            status_code=400,
            detail="Target type must be group or private",
        )

    return RobotReplyTarget(
        target_type=target_type,
        target_id=target_id,
        metadata={"manual_target": True},
    )


@router.get("/platforms", response_model=list[RobotPlatformPublic])
def list_robot_platform_metadata() -> list[RobotPlatformPublic]:
    return list_supported_robot_platforms()


@router.get("/images/{name}")
def get_robot_image(name: str) -> FileResponse:
    """入站图片的本地缓存。uuid 文件名不可猜测，无需鉴权（同下载票据模式）。"""
    from .image_store import image_file_path

    path = image_file_path(name)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@router.get("/stickers")
def list_robot_stickers(current_user: CurrentUser) -> dict:
    """表情包库清单（前端表情库面板）。"""
    del current_user
    from .stickers import list_stickers

    return {"tags": list_stickers()}


@router.get("/stickers/file/{tag}/{name}")
def get_robot_sticker_file(tag: str, name: str) -> FileResponse:
    """表情包文件。uuid 文件名不可猜测，无需鉴权（同下载票据模式）。"""
    from .stickers import sticker_file_path

    path = sticker_file_path(tag, name)
    if path is None:
        raise HTTPException(status_code=404, detail="Sticker not found")
    return FileResponse(path)


@router.delete("/stickers/{tag}/{name}")
def delete_robot_sticker(tag: str, name: str, current_user: CurrentUser) -> dict:
    del current_user
    from .stickers import delete_sticker

    if not delete_sticker(tag, name):
        raise HTTPException(status_code=404, detail="Sticker not found")
    return {"message": "deleted"}


@router.post("/stickers")
async def upload_robot_sticker(
    current_user: CurrentUser,
    tag: str = Form(...),
    file: UploadFile = File(...),
) -> dict:
    """上传一张表情包到指定标签。"""
    del current_user
    from .image_store import _EXT_BY_MIME
    from .stickers import normalize_tag, save_sticker

    normalized = normalize_tag(tag)
    if not normalized:
        raise HTTPException(status_code=400, detail="Invalid tag")
    payload = await file.read()
    if not payload or len(payload) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Empty or oversized file")
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    ext = _EXT_BY_MIME.get(content_type)
    if ext is None:
        if payload[:3] == b"\xff\xd8\xff":
            ext = ".jpg"
        elif payload[:8] == b"\x89PNG\r\n\x1a\n":
            ext = ".png"
        elif payload[:6] in (b"GIF87a", b"GIF89a"):
            ext = ".gif"
        elif payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
            ext = ".webp"
        else:
            raise HTTPException(status_code=400, detail="Unsupported image format")
    path = save_sticker(normalized, payload, ext)
    if path is None:
        raise HTTPException(status_code=400, detail="Failed to save sticker")
    return {"tag": normalized, "name": path.name}


@router.get("/bridge/health")
def get_bridge_health(current_user: CurrentUser) -> dict:
    del current_user

    try:
        health = robot_bridge_client.get_health()
        health["backend_dispatch_queue"] = robot_service.dispatch_queue_snapshot()
        return health
    except Exception as e:
        return {
            "error": str(e),
            "connected": False,
            "bridge_url": robot_bridge_client.base_url,
            "backend_dispatch_queue": robot_service.dispatch_queue_snapshot(),
        }


@router.get("/bridge/runtime-config")
def get_bridge_runtime_config(
    session: SessionDep,
    x_TermPaws_bridge_token: str | None = Header(default=None),
) -> dict:
    assert_bridge_permission(x_TermPaws_bridge_token)

    robots = session.exec(
        select(Robot).where(
            Robot.is_enabled == True,  # noqa: E712
            Robot.provider == "nonebot2",
        )
    ).all()

    loaded_robots: list[dict] = []
    robot_id_by_identity: dict[str, str] = {}
    identity_by_robot_id: dict[str, str] = {}
    errors: dict[str, str] = {}

    for robot in robots:
        robot_id = str(robot.id)
        try:
            platform_id = normalize_robot_platform_id(robot.platform or robot.protocol)
            runtime_config = get_robot_runtime_config(robot)
            identity = resolve_robot_identity(platform_id, robot)
        except Exception as exc:
            errors[robot_id] = str(exc)
            continue
        if identity in robot_id_by_identity:
            errors[robot_id] = f"Duplicate robot identity `{identity}`"
            continue

        loaded_robots.append(
            {
                "id": robot_id,
                "platform": platform_id,
                "protocol": platform_id,
                "provider": robot.provider,
                "name": robot.name,
                "runtime_config": runtime_config,
                "identity": identity,
            }
        )
        robot_id_by_identity[identity] = robot_id
        identity_by_robot_id[robot_id] = identity

    return {
        "robots": loaded_robots,
        "robot_id_by_identity": robot_id_by_identity,
        "identity_by_robot_id": identity_by_robot_id,
        "errors": errors,
    }


@router.get("/", response_model=RobotsPublic)
def read_robots(
    session: SessionDep,
    current_user: CurrentUser,
    skip: int = 0,
    limit: int = 100,
) -> RobotsPublic:
    if current_user.is_superuser:
        statement = (
            select(Robot)
            .order_by(col(Robot.created_at).desc())
            .offset(skip)
            .limit(limit)
        )
        robots = session.exec(statement).all()
    else:
        statement = (
            select(Robot)
            .where(Robot.owner_id == current_user.id)
            .order_by(col(Robot.created_at).desc())
            .offset(skip)
            .limit(limit)
        )
        robots = session.exec(statement).all()

    return RobotsPublic(data=robots, count=len(robots))


@router.get("/items/{item_id}/conversation-controllers")
def get_item_robot_conversation_controllers(
    session: SessionDep,
    current_user: CurrentUser,
    item_id: uuid.UUID,
) -> dict:
    item = get_item_or_404(session, item_id)
    assert_item_permission(item, current_user)

    bindings = session.exec(
        select(RobotItem).where(RobotItem.item_id == item_id)
    ).all()
    visible_bindings: list[tuple[RobotItem, Robot]] = []
    robot_ids: set[uuid.UUID] = set()
    for binding in bindings:
        robot = session.get(Robot, binding.robot_id)
        if not robot:
            continue
        if not current_user.is_superuser and robot.owner_id != current_user.id:
            continue
        visible_bindings.append((binding, robot))
        robot_ids.add(robot.id)

    snapshots_by_robot: dict[str, list[dict[str, object]]] = {}
    for snapshot in robot_service.conversation_controller_snapshots(
        robot_ids,
        item_ids={item.id},
    ):
        snapshots_by_robot.setdefault(str(snapshot.get("robot_id") or ""), []).append(
            snapshot
        )

    _merge_active_reply_tickets_into_controller_snapshots(
        snapshots_by_robot,
        item_id=item.id,
        robot_ids=robot_ids,
    )

    robots: list[dict[str, object]] = []
    for binding, robot in visible_bindings:
        robots.append(
            {
                "robot_id": str(robot.id),
                "robot_name": robot.name,
                "is_enabled": robot.is_enabled,
                "allow_chat": binding.allow_chat,
                "route_key": robot_service.build_route_key(item, binding),
                "reply_context_window_seconds": robot_service.reply_context_window_seconds(
                    robot
                ),
                "conversation_controllers": snapshots_by_robot.get(str(robot.id), []),
            }
        )

    return {"item_id": str(item.id), "robots": robots, "count": len(robots)}


@router.get("/{id}", response_model=RobotPublic)
def read_robot(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> RobotPublic:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)
    return robot


@router.post("/", response_model=RobotPublic)
def create_robot(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    robot_in: RobotCreate,
) -> RobotPublic:
    ensure_unique_robot_name(session, current_user, robot_in.name)

    payload = robot_in.model_dump()
    _, update = normalize_robot_stack(
        str(payload.get("platform") or ""),
        payload.get("config"),
    )
    update["config"] = _ensure_onebot_access_token(
        str(update["platform"]),
        update["config"],
    )

    robot = Robot.model_validate(
        robot_in,
        update={
            "owner_id": current_user.id,
            **update,
        },
    )
    session.add(robot)
    session.commit()
    session.refresh(robot)
    robot_bridge_client.notify_reload()
    return robot


@router.put("/{id}", response_model=RobotPublic)
def update_robot(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    robot_in: RobotUpdate,
) -> RobotPublic:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    if robot_in.name and robot_in.name != robot.name:
        ensure_unique_robot_name(
            session,
            current_user,
            robot_in.name,
            exclude_robot_id=robot.id,
        )

    update_dict = robot_in.model_dump(exclude_unset=True)
    try:
        merged_platform = normalize_robot_platform_id(
            str(update_dict.get("platform") or robot.platform or robot.protocol or "")
        )
        if "config" in update_dict:
            incoming_config = update_dict.get("config")
            if isinstance(incoming_config, dict):
                existing_config = current_robot_config(robot)
                incoming_credentials = incoming_config.get("credentials")
                if isinstance(incoming_credentials, dict):
                    merged_credentials = dict(existing_config.get("credentials") or {})
                    platform = get_robot_platform(merged_platform)
                    secret_keys = {field.key for field in platform.fields if field.secret}
                    for key, value in incoming_credentials.items():
                        if value not in (None, "") or key not in secret_keys:
                            merged_credentials[str(key)] = value
                    incoming_config = {
                        **incoming_config,
                        "credentials": merged_credentials,
                    }
            merged_config = normalize_robot_config(
                merged_platform,
                incoming_config,
            )
        else:
            merged_config = current_robot_config(robot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    merged_config = _ensure_onebot_access_token(merged_platform, merged_config)
    _, normalized_update = normalize_robot_stack(merged_platform, merged_config)

    if "name" in update_dict and update_dict["name"] is not None:
        normalized_update["name"] = str(update_dict["name"]).strip()
    if "is_enabled" in update_dict:
        normalized_update["is_enabled"] = update_dict["is_enabled"]

    robot.sqlmodel_update(normalized_update)
    session.add(robot)
    session.commit()
    session.refresh(robot)
    robot_bridge_client.notify_reload()
    return robot


@router.delete("/{id}")
def delete_robot(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> Message:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    session.delete(robot)
    session.commit()
    robot_bridge_client.notify_reload()
    return Message(message="Robot deleted successfully")


@router.get("/{id}/items", response_model=list[RobotItemBindingPublic])
def list_robot_item_bindings(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> list[RobotItemBindingPublic]:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    bindings = session.exec(select(RobotItem).where(RobotItem.robot_id == id)).all()
    return [serialize_binding(session, binding) for binding in bindings]


@router.post("/{id}/items", response_model=RobotItemBindingPublic)
def bind_item_to_robot(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: RobotItemBindingCreate,
) -> RobotItemBindingPublic:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    item = get_item_or_404(session, body.item_id)
    assert_item_permission(item, current_user)

    existing = session.exec(
        select(RobotItem).where(
            RobotItem.robot_id == id,
            RobotItem.item_id == body.item_id,
        )
    ).first()
    if existing:
        raise HTTPException(
            status_code=400, detail="Item is already bound to this robot"
        )

    payload = normalize_binding_payload(body)
    validate_binding_payload(payload)
    route_key = ensure_unique_binding_route_key(
        session,
        id,
        item,
        chat_alias=payload.get("chat_alias"),
    )

    binding = RobotItem(
        robot_id=id,
        item_id=body.item_id,
        allow_chat=payload.get("allow_chat", True),
        receive_filtered_output=False,
        chat_alias=payload.get("chat_alias"),
        is_default_target=payload.get("is_default_target", False),
    )

    if binding.is_default_target:
        clear_default_targets(session, id)

    session.add(binding)
    session.commit()
    session.refresh(binding)

    response = serialize_binding(session, binding)
    response.route_key = route_key
    return response


@router.put("/{id}/items/{item_id}", response_model=RobotItemBindingPublic)
def update_robot_item_binding(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    item_id: uuid.UUID,
    body: RobotItemBindingUpdate,
) -> RobotItemBindingPublic:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    binding = session.exec(
        select(RobotItem).where(
            RobotItem.robot_id == id,
            RobotItem.item_id == item_id,
        )
    ).first()
    if not binding:
        raise HTTPException(status_code=404, detail="Robot item binding not found")

    item = get_item_or_404(session, item_id)
    assert_item_permission(item, current_user)

    payload = normalize_binding_payload(body)
    merged_payload = {
        "allow_chat": binding.allow_chat,
        "receive_filtered_output": False,
        "chat_alias": binding.chat_alias,
        "is_default_target": binding.is_default_target,
    }
    merged_payload.update(payload)
    merged_payload["receive_filtered_output"] = False
    validate_binding_payload(merged_payload)

    if "chat_alias" in payload:
        ensure_unique_binding_route_key(
            session,
            id,
            item,
            chat_alias=merged_payload.get("chat_alias"),
            exclude_item_id=item_id,
        )

    if merged_payload.get("is_default_target"):
        clear_default_targets(session, id, exclude_item_id=item_id)

    binding.allow_chat = merged_payload["allow_chat"]
    binding.receive_filtered_output = False
    binding.chat_alias = merged_payload["chat_alias"]
    binding.is_default_target = merged_payload["is_default_target"]

    if not binding.allow_chat:
        binding.is_default_target = False

    session.add(binding)
    session.commit()
    session.refresh(binding)
    return serialize_binding(session, binding)


@router.delete("/{id}/items/{item_id}")
def unbind_item_from_robot(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    item_id: uuid.UUID,
) -> Message:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    binding = session.exec(
        select(RobotItem).where(
            RobotItem.robot_id == id,
            RobotItem.item_id == item_id,
        )
    ).first()
    if not binding:
        raise HTTPException(status_code=404, detail="Robot item binding not found")

    session.delete(binding)
    session.commit()
    return Message(message="Robot item binding deleted successfully")


@router.post("/{id}/dispatch", response_model=RobotDispatchResponse)
def dispatch_robot_message(
    session: SessionDep,
    id: uuid.UUID,
    body: RobotInboundMessage,
    x_TermPaws_bridge_token: str | None = Header(default=None),
) -> RobotDispatchResponse:
    assert_bridge_permission(x_TermPaws_bridge_token)
    robot = get_robot_or_404(session, id)
    return robot_service.handle_inbound_message(session, robot, body)


@router.post("/{id}/debug-events")
def ingest_robot_debug_event(
    session: SessionDep,
    id: uuid.UUID,
    body: dict,
    x_TermPaws_bridge_token: str | None = Header(default=None),
) -> dict:
    assert_bridge_permission(x_TermPaws_bridge_token)
    get_robot_or_404(session, id)
    record_robot_event(
        str(id),
        direction=str(body.get("direction") or "bridge"),
        event=str(body.get("event") or "debug_event"),
        status=str(body.get("status") or "ok"),
        message=body.get("message"),
        payload=body.get("payload") if isinstance(body.get("payload"), dict) else {},
    )
    return {"success": True}


@router.get("/{id}/connection")
def get_robot_connection_status(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> dict:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    if not robot.is_enabled:
        return {"robot_id": str(id), "connected": False, "reason": "robot_disabled"}

    try:
        health_data = robot_bridge_client.get_health()

        robots_status = health_data.get("robots", {})
        robot_status = robots_status.get(str(id), {})

        return {
            "robot_id": str(id),
            "identity": robot_status.get("identity"),
            "connected": robot_status.get("connected", False),
            "platform": robot.platform,
        }
    except Exception as e:
        return {"robot_id": str(id), "connected": False, "error": str(e)}


@router.get("/{id}/diagnose")
def diagnose_robot_chain(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    request: Request,
) -> dict:
    from app.services import backend_conn_pool

    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)
    platform_id = normalize_robot_platform_id(robot.platform or robot.protocol)
    uses_napcat_socket = platform_id == "onebot_v11"

    result: dict = {
        "robot_id": str(id),
        "robot_name": robot.name,
        "robot_enabled": robot.is_enabled,
        "platform": robot.platform,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "chain": {},
    }

    result["chain"]["robot_config"] = {
        "status": "ok" if robot.is_enabled else "disabled",
        "app_id": robot.app_id,
        "provider": robot.provider,
    }

    recent_events = get_robot_events(str(id), limit=200)
    bridge_status: dict = {"status": "unknown", "connected": False}
    socket_status: dict = {
        "status": "unknown" if uses_napcat_socket else "not_applicable",
        "connected": not uses_napcat_socket,
    }
    try:
        health_data = robot_bridge_client.get_health()

        robots_status = health_data.get("robots", {})
        robot_status = robots_status.get(str(id), {})
        backend_status = health_data.get("backend", {})
        connection_errors = health_data.get("connection_errors", {})
        raw_connection_error = connection_errors.get(str(id))
        historical_error = (
            raw_connection_error.get("message")
            if isinstance(raw_connection_error, dict)
            else raw_connection_error
        )
        historical_error_at = (
            raw_connection_error.get("timestamp")
            if isinstance(raw_connection_error, dict)
            else None
        )
        current_error = robot_status.get("error")
        connected = robot_status.get("connected", False)
        onebot_socket = (
            robot_status.get("onebot_socket")
            if isinstance(robot_status.get("onebot_socket"), dict)
            else {}
        )

        bridge_status = {
            "status": "ok",
            "connected": connected,
            "identity": robot_status.get("identity"),
            "backend_reachable": backend_status.get("reachable", False),
            "error": None if connected else current_error or historical_error,
            "stale_error": historical_error if connected else None,
            "stale_error_at": historical_error_at if connected else None,
            "bridge_checked_at": health_data.get("checked_at"),
            "live": bool(health_data.get("live", False)),
        }
        if uses_napcat_socket:
            socket_connected = bool(onebot_socket.get("connected"))
            reverse_ws_url = (
                onebot_socket.get("reverse_ws_url")
                or onebot_socket.get("ws_url")
                or robot_status.get("reverse_ws_url")
                or health_data.get("onebot_reverse_ws_url")
                or _default_reverse_ws_url(request)
            )
            socket_status = {
                "status": "ok" if socket_connected else "waiting",
                "connected": socket_connected,
                "server_url": reverse_ws_url,
                "reverse_ws_url": reverse_ws_url,
                "ws_url": reverse_ws_url,
                "last_event": onebot_socket.get("event"),
                "last_event_at": onebot_socket.get("last_event_at"),
                "self_id": onebot_socket.get("self_id"),
                "last_message_seen": _event_seen(
                    recent_events,
                    {"platform_message", "inbound_message"},
                ),
                "last_socket_receive_seen": _event_seen(
                    recent_events,
                    {"platform_event", "platform_message"},
                ),
            }
    except Exception as e:
        bridge_status = {"status": "error", "error": str(e), "connected": False}
        socket_status = {
            "status": "error" if uses_napcat_socket else "not_applicable",
            "error": str(e),
            "connected": False if uses_napcat_socket else True,
        }

    result["chain"]["qq_to_bridge"] = bridge_status
    result["chain"]["napcat_socket"] = socket_status

    bindings = session.exec(select(RobotItem).where(RobotItem.robot_id == id)).all()
    items_status: list[dict] = []

    for binding in bindings:
        item = session.get(Item, binding.item_id)
        if not item:
            continue

        item_status: dict = {
            "item_id": str(item.id),
            "item_title": item.title,
            "allow_chat": binding.allow_chat,
            "receive_filtered_output": False,
            "is_default_target": binding.is_default_target,
            "route_key": robot_service.build_route_key(item, binding),
        }

        daemon_status: dict = {"status": "unknown", "online": False}
        if item.api_key:
            daemon_state = backend_conn_pool.get_daemon_main_conn_state(item.api_key)
            if daemon_state and daemon_state.is_connected():
                daemon_status = {"status": "online", "online": True}
            else:
                daemon_status = {"status": "offline", "online": False}
        else:
            daemon_status = {"status": "not_configured", "online": False}

        item_status["daemon"] = daemon_status
        item_status["agent_route"] = {
            "status": "ok" if binding.allow_chat else "disabled",
            "chat_enabled": binding.allow_chat,
            "route_key": item_status["route_key"],
            "default_target": binding.is_default_target,
            "filtered_output_enabled": False,
        }
        items_status.append(item_status)

    result["chain"]["items"] = items_status

    all_ok = (
        robot.is_enabled
        and bridge_status.get("connected", False)
        and (not uses_napcat_socket or socket_status.get("connected", False))
        and any(item.get("daemon", {}).get("online", False) for item in items_status)
    )
    result["overall_status"] = "ok" if all_ok else "degraded"

    return result


@router.post("/{id}/reload")
def reload_robot_bridge(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> dict:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    success, detail = robot_bridge_client.notify_reload()
    if success:
        record_robot_event(
            str(id),
            direction="backend_to_bridge",
            event="manual_reload",
            message=detail,
        )
        return {"success": True, "message": detail}

    record_robot_event(
        str(id),
        direction="backend_to_bridge",
        event="manual_reload",
        status="error",
        message=detail,
    )
    return {"success": False, "error": detail}


@router.post("/{id}/debug/send")
def send_robot_debug_message(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: RobotDebugSendBody,
) -> dict:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message text is required")

    target = _latest_reply_target_from_debug_events(id)
    if target is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "No recent robot conversation target found. Send a message from QQ "
                "first, then try the debug send again."
            ),
        )

    try:
        robot_bridge_client.send_message(robot.id, target, text)
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        record_robot_event(
            str(id),
            direction="backend_to_bridge",
            event="manual_debug_send",
            status="error",
            message=detail,
            payload={
                "target_type": target.target_type,
                "target_id": target.target_id,
            },
        )
        raise HTTPException(status_code=502, detail=detail) from exc

    _remember_sent_robot_message(id, target, text)
    record_robot_event(
        str(id),
        direction="backend_to_bridge",
        event="manual_debug_send",
        message=text,
        payload={
            "target_type": target.target_type,
            "target_id": target.target_id,
        },
    )
    return {
        "success": True,
        "target_type": target.target_type,
        "target_id": target.target_id,
    }


@router.post("/{id}/messages/send")
def send_robot_manual_message(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: RobotManualSendBody,
) -> dict:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message text is required")

    target = _manual_send_target(body)
    try:
        robot_bridge_client.send_message(robot.id, target, text)
    except Exception as exc:
        detail = str(exc) or exc.__class__.__name__
        record_robot_event(
            str(id),
            direction="backend_to_bridge",
            event="manual_send",
            status="error",
            message=detail,
            payload={
                "target_type": target.target_type,
                "target_id": target.target_id,
            },
        )
        raise HTTPException(status_code=502, detail=detail) from exc

    _remember_sent_robot_message(id, target, text)
    record_robot_event(
        str(id),
        direction="backend_to_bridge",
        event="manual_send",
        message=text,
        payload={
            "target_type": target.target_type,
            "target_id": target.target_id,
        },
    )
    return {
        "success": True,
        "target_type": target.target_type,
        "target_id": target.target_id,
    }


def _default_reverse_ws_url(request: Request) -> str:
    """静态推导 NapCat 反向 WS 地址（还没连接时也要显示给用户）。"""
    host = request.headers.get("host") or request.url.hostname or "localhost"
    return f"ws://{host}/robot-bridge/onebot/v11/ws"


@router.get("/{id}/debug")
def get_robot_debug(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    request: Request,
    limit: int = 100,
) -> dict:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)

    bridge_health: dict = {"status": "unknown"}
    try:
        bridge_health = robot_bridge_client.get_health()
    except Exception as e:
        bridge_health = {"status": "error", "error": str(e)}

    robot_health = bridge_health.get("robots", {}).get(str(id), {})
    onebot_socket = (
        robot_health.get("onebot_socket")
        if isinstance(robot_health.get("onebot_socket"), dict)
        else {}
    )
    reverse_ws_url = (
        onebot_socket.get("reverse_ws_url")
        or robot_health.get("reverse_ws_url")
        or bridge_health.get("onebot_reverse_ws_url")
        or onebot_socket.get("ws_url")
        or robot_health.get("ws_url")
        or _default_reverse_ws_url(request)
    )
    if reverse_ws_url:
        onebot_socket = {
            "server_url": reverse_ws_url,
            "reverse_ws_url": reverse_ws_url,
            "ws_url": reverse_ws_url,
            **onebot_socket,
        }
    connection_errors = bridge_health.get("connection_errors", {})
    raw_connection_error = connection_errors.get(str(id))
    historical_error = (
        raw_connection_error.get("message")
        if isinstance(raw_connection_error, dict)
        else raw_connection_error
    )
    connected = robot_health.get("connected", False)
    events = get_robot_events(str(id), limit=limit)
    return {
        "robot": {
            "id": str(robot.id),
            "name": robot.name,
            "platform": robot.platform,
            "provider": robot.provider,
            "is_enabled": robot.is_enabled,
            "app_id": robot.app_id,
        },
        "bridge": {
            "url": robot_bridge_client.base_url,
            "napcat_ws_url": reverse_ws_url,
            "onebot_reverse_ws_url": reverse_ws_url,
            "status": bridge_health.get(
                "status", "ok" if "error" not in bridge_health else "error"
            ),
            "loaded_robot_count": bridge_health.get("loaded_robot_count", 0),
            "connected_bot_count": bridge_health.get("connected_bot_count", 0),
            "onebot_client_count": bridge_health.get("onebot_client_count", 0),
            "onebot_clients": bridge_health.get("onebot_clients", []),
            "connected": robot_health.get("connected", False),
            "identity": robot_health.get("identity"),
            "bot": robot_health.get("bot"),
            "bots": bridge_health.get("bots", []),
            "onebot_socket": onebot_socket,
            "checked_at": bridge_health.get("checked_at"),
            "last_platform_event_at": robot_health.get("last_platform_event_at"),
            "last_message_event_at": robot_health.get("last_message_event_at"),
            "error": None
            if connected
            else robot_health.get("error")
            or historical_error
            or bridge_health.get("error"),
            "stale_error": historical_error if connected else None,
        },
        "diagnostics": {
            "event_count": len(events),
            "last_event_at": events[0].get("timestamp") if events else None,
            "qq_event_hint": (
                "Robot server is connected, but no QQ connector message event has reached TermPaws yet. "
                "Check the reverse WebSocket endpoint and whether the logged-in QQ account is receiving messages."
                if connected and not robot_health.get("last_message_event_at")
                else None
            ),
        },
        "events": events,
    }


def _normalize_api_conversation_key(conversation_key: str) -> str:
    normalized = conversation_key.strip()
    if not re.match(CONVERSATION_KEY_PATTERN, normalized):
        raise HTTPException(
            status_code=400,
            detail=(
                "conversation_key must look like group:<id>, private:<id>, "
                "or channel:<id>"
            ),
        )
    return normalized


@router.get("/{id}/conversation-memory")
def list_robot_conversation_memory(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> dict:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)
    entries = robot_conversation_memory.list_conversations(str(id))
    return {
        "data": [entry.__dict__ for entry in entries],
        "count": len(entries),
    }


@router.get("/{id}/conversation-memory/{conversation_key}")
def read_robot_conversation_memory(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    conversation_key: str,
) -> dict:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)
    normalized_key = _normalize_api_conversation_key(conversation_key)
    info = robot_conversation_memory.info(str(id), normalized_key)
    return {
        "memory": robot_conversation_memory.read(str(id), normalized_key),
        "info": info.__dict__,
    }


@router.get("/{id}/conversation-memory/{conversation_key}/export")
def export_robot_conversation_memory(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    conversation_key: str,
) -> Response:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)
    normalized_key = _normalize_api_conversation_key(conversation_key)
    filename = robot_conversation_memory.export_filename(normalized_key)
    return PlainTextResponse(
        robot_conversation_memory.read(str(id), normalized_key),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{id}/conversation-memory/{conversation_key}/import")
def import_robot_conversation_memory(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    conversation_key: str,
    body: RobotConversationMemoryImportBody,
) -> dict:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)
    normalized_key = _normalize_api_conversation_key(conversation_key)
    try:
        info = robot_conversation_memory.replace(
            str(id),
            normalized_key,
            body.content,
            append=body.append,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "info": info.__dict__}


@router.delete("/{id}/conversation-memory/{conversation_key}")
def delete_robot_conversation_memory(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    conversation_key: str,
) -> dict:
    robot = get_robot_or_404(session, id)
    assert_robot_permission(robot, current_user)
    deleted = robot_conversation_memory.delete(
        str(id),
        _normalize_api_conversation_key(conversation_key),
    )
    return {"success": True, "deleted": deleted}
