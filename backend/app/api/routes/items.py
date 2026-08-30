import json
import logging
import re
import uuid
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel
from sqlmodel import col, func, select

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.models import (
    Item,
    ItemCreate,
    ItemPublic,
    ItemStatus,
    ItemUpdate,
    Message,
    User,
)
from app.services import (
    DaemonConfig,
    backend_conn_pool,
    connection_manager,
    item_file_service,
    log_manager,
    socket_pool_facade,
    sync_daemon_connection_state,
)
from app.services.item_file_service import ItemFileServiceError
from app.services.filters import (
    InputFilter,
    InputFilterConfig,
    OutputFilter,
    OutputFilterConfig,
)
from app.services.llm_generation_service import (
    LlmGenerationError,
    generate_json_payload,
    get_item_handler_for_item,
)
from app.services.socket_pool.terminal_stream_pipeline import sanitize_terminal_text
from app.services.terminal_service import TerminalService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/items", tags=["items"])

FILTER_ACTION_TYPES = {"block", "ignore", "log", "replace"}


def _normalize_ws_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if url.startswith("http://"):
        return f"ws://{url[7:]}"
    if url.startswith("https://"):
        return f"wss://{url[8:]}"
    return url


def _format_ws_host(host: str) -> str:
    host = host.strip()
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _build_terminal_ws_url(request: Request, item: Item) -> str:
    public_url = (settings.DAEMON_PUBLIC_URL or "").strip()
    if public_url:
        return _normalize_ws_url(public_url)

    configured_host = (settings.DAEMON_PUBLIC_HOST or "").strip()
    host = configured_host or request.url.hostname or item.socket_host or "localhost"
    port = settings.DAEMON_PUBLIC_PORT or settings.DAEMON_HOST_PORT or item.socket_port or 9000
    scheme = settings.DAEMON_PUBLIC_WS_SCHEME or "ws"

    if "://" in host:
        parsed = urlsplit(host)
        if parsed.scheme:
            scheme = "wss" if parsed.scheme == "https" else "ws"
        host = parsed.hostname or host
        port = settings.DAEMON_PUBLIC_PORT or parsed.port or port

    return f"{scheme}://{_format_ws_host(host)}:{port}"


class FilePathRequest(BaseModel):
    path: str


class TerminalCommandStateRequest(BaseModel):
    command: str


class UploadTicketRequest(FilePathRequest):
    allow_overwrite: bool = False


class RenamePathRequest(FilePathRequest):
    target_path: str


class CopyPathRequest(FilePathRequest):
    target_path: str


class FileWriteRequest(FilePathRequest):
    content: str
    encoding: str = "utf-8"


class FileTicketVerifyRequest(BaseModel):
    ticket: str
    op: str


class GenerateFilterRequest(BaseModel):
    target: Literal["input", "output"]
    instruction: str
    existing_rules: dict[str, Any] | None = None


class GenerateFilterResponse(BaseModel):
    success: bool = True
    target: Literal["input", "output"]
    rules: dict[str, Any]
    explanation: str = ""
    item_handler_id: str
    model: str


def _get_daemon_status(item: Item) -> dict:
    """获取item对应daemon的连接状态"""
    if not item.socket_host or not item.socket_port or not item.api_key:
        return {"daemon_id": None, "daemon_online": False, "daemon_status": "not_configured"}
    
    daemon_id = f"{item.socket_host}:{item.socket_port}:{item.api_key}"
    daemon_state = backend_conn_pool.get_daemon_main_conn_state(item.api_key)
    connection = connection_manager.get_connection(daemon_id)
    if connection and connection.is_connected():
        synced_state = backend_conn_pool.create_daemon_main_conn_state(
            item.api_key,
            f"http://{item.socket_host}:{item.socket_port}",
        )
        synced_state.set_connected()
        return {"daemon_id": daemon_id, "daemon_online": True, "daemon_status": "connected"}
    
    if daemon_state and daemon_state.is_connected():
        return {"daemon_id": daemon_id, "daemon_online": True, "daemon_status": "connected"}
    return {"daemon_id": daemon_id, "daemon_online": False, "daemon_status": "disconnected"}


def _get_item_data(item: Item, session: SessionDep = None) -> dict:
    """获取item的完整数据（包含daemon状态和连接信息）"""
    item_data = ItemPublic.model_validate(item).model_dump()
    item_data["daemon_url"] = f"http://{item.socket_host}:{item.socket_port}"
    
    daemon_status = _get_daemon_status(item)
    item_data["daemon_id"] = daemon_status["daemon_id"]
    item_data["daemon_online"] = daemon_status["daemon_online"]
    item_data["daemon_status"] = daemon_status["daemon_status"]
    
    daemon_id, token = socket_pool_facade.get_item_token(str(item.id))
    if token:
        item_data['token'] = token
    
    subscribers = _get_item_subscribers_internal(item)
    
    if session and subscribers.get("subscribers"):
        user_ids = set()
        for sub in subscribers["subscribers"]:
            if sub.get("user_uuid"):
                try:
                    user_ids.add(uuid.UUID(sub["user_uuid"]))
                except (ValueError, TypeError):
                    pass
        
        if user_ids:
            users = session.exec(select(User).where(User.id.in_(user_ids))).all()
            user_map = {str(user.id): user.full_name or user.email for user in users}
            
            for sub in subscribers["subscribers"]:
                user_uuid = sub.get("user_uuid", "")
                sub["user_name"] = user_map.get(user_uuid, user_uuid[:8] + "...")
    
    item_data["subscribers"] = subscribers.get("subscribers", [])
    item_data["browser_count"] = subscribers.get("browser_count", 0)
    item_data["backend_connected"] = subscribers.get("backend_connected", False)
    
    return item_data


def _get_item_subscribers_internal(item: Item) -> dict:
    """获取item的订阅者信息（内部方法）"""
    daemon_status = _get_daemon_status(item)
    if not daemon_status["daemon_online"]:
        return {
            "subscribers": [],
            "browser_count": 0,
            "backend_connected": False
        }
    
    connection = connection_manager.get_connection(
        f"{item.socket_host}:{item.socket_port}:{item.api_key}"
    )
    
    if not connection or not connection.is_connected():
        return {
            "subscribers": [],
            "browser_count": 0,
            "backend_connected": False
        }
    
    result = connection.get_item_subscribers_http(str(item.id))
    return result


def _check_item_permission(item: Item, current_user: CurrentUser):
    """检查用户对item的权限"""
    if not current_user.is_superuser and item.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")


def _get_item_daemon_connection(item: Item):
    daemon_status = _get_daemon_status(item)
    if not daemon_status["daemon_online"]:
        return daemon_status, None

    connection = connection_manager.get_connection(
        f"{item.socket_host}:{item.socket_port}:{item.api_key}"
    )
    if not connection or not connection.is_connected():
        return daemon_status, None
    return daemon_status, connection


def _sanitize_filter_name(name: str, index: int, existing: set[str]) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9_]+", "_", (name or "").strip().lower()).strip(
        "_"
    )
    if not candidate:
        candidate = f"rule_{index}"

    if candidate not in existing:
        return candidate

    suffix = 2
    while f"{candidate}_{suffix}" in existing:
        suffix += 1
    return f"{candidate}_{suffix}"


def _normalize_generated_filter_rules(raw_rules: Any) -> dict[str, Any]:
    if not isinstance(raw_rules, dict):
        raise LlmGenerationError(
            "Generated rules must be a JSON object.",
            status_code=502,
        )

    normalized_rules: dict[str, Any] = {}
    for index, (raw_name, raw_rule) in enumerate(raw_rules.items(), start=1):
        if not isinstance(raw_rule, dict):
            raise LlmGenerationError(
                f"Generated filter '{raw_name}' is invalid.",
                status_code=502,
            )

        action_type = str(raw_rule.get("action_type", "")).strip().lower()
        if action_type not in FILTER_ACTION_TYPES:
            raise LlmGenerationError(
                f"Generated filter '{raw_name}' uses an unsupported action_type.",
                status_code=502,
            )

        raw_patterns = raw_rule.get("regex_patterns", [])
        if not isinstance(raw_patterns, list):
            raise LlmGenerationError(
                f"Generated filter '{raw_name}' must contain regex_patterns.",
                status_code=502,
            )

        patterns: list[str] = []
        for raw_pattern in raw_patterns:
            if not isinstance(raw_pattern, str):
                continue
            pattern = raw_pattern.strip()
            if not pattern:
                continue
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise LlmGenerationError(
                    f"Generated regex '{pattern}' is invalid: {exc}",
                    status_code=502,
                ) from exc
            patterns.append(pattern)

        if not patterns:
            continue

        filter_name = _sanitize_filter_name(
            str(raw_name),
            index,
            set(normalized_rules),
        )
        rule_payload: dict[str, Any] = {
            "regex_patterns": patterns,
            "action_type": action_type,
        }

        if action_type == "replace":
            replace_rules: dict[str, str] = {}
            action = raw_rule.get("action", {})
            if isinstance(action, dict):
                raw_replace_rules = action.get("replace_rules", {})
                if isinstance(raw_replace_rules, dict):
                    for pattern, replacement in raw_replace_rules.items():
                        if not isinstance(pattern, str) or not isinstance(
                            replacement, str
                        ):
                            continue
                        replace_rules[pattern] = replacement

            if replace_rules:
                rule_payload["action"] = {"replace_rules": replace_rules}

        normalized_rules[filter_name] = rule_payload

    if not normalized_rules:
        raise LlmGenerationError(
            "LLM did not generate any usable filter rules.",
            status_code=502,
        )

    return normalized_rules


def _build_filter_generation_prompts(
    *,
    target: Literal["input", "output"],
    existing_rules: dict[str, Any],
    instruction: str,
) -> tuple[str, str]:
    target_label = (
        "terminal output before it reaches Agent/timeline/robot routing"
        if target == "input"
        else "Agent-generated shell commands before terminal execution"
    )
    target_behavior = (
        "Path: terminal output -> input filter -> Agent/timeline/robot routing. "
        "Use 'block' to drop dangerous or useless output, 'ignore' to strip "
        "noisy fragments, 'log' to keep the content but mark it as important, "
        "and 'replace' to redact secrets."
        if target == "input"
        else "Path: Agent-generated command -> output filter -> terminal execution. "
        "Use 'block' to deny dangerous commands, 'ignore' to strip harmless "
        "noise fragments, 'log' to keep the command but flag it, and 'replace' "
        "to redact or rewrite sensitive values."
    )
    schema = {
        "rules": {
            "rule_name": {
                "regex_patterns": ["regex pattern"],
                "action_type": "block|ignore|log|replace",
                "action": {"replace_rules": {"regex pattern": "replacement"}},
            }
        },
        "explanation": "short explanation",
    }
    system_prompt = (
        "You design TermPaws filter rules.\n"
        "Return only a valid JSON object with the exact schema described by the user.\n"
        "Generate a small set of precise rules. Avoid duplicate or overly broad regex.\n"
        "Prefer 'block' only for clearly dangerous matches. Use 'replace' for redaction.\n"
        "Every regex must be valid for Python's re module."
    )
    user_prompt = (
        f"Target: {target} filter for {target_label}.\n"
        f"Behavior: {target_behavior}\n\n"
        "Current rules JSON:\n"
        f"{json.dumps(existing_rules or {}, ensure_ascii=False, indent=2)}\n\n"
        "Requested changes:\n"
        f"{instruction.strip()}\n\n"
        "Return JSON only using this shape:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
    return system_prompt, user_prompt


@router.get("/", response_model=dict[str, Any])
def read_items(
    session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100
) -> Any:
    if current_user.is_superuser:
        count_statement = select(func.count()).select_from(Item)
        count = session.exec(count_statement).one()
        statement = (
            select(Item).order_by(col(Item.created_at).desc()).offset(skip).limit(limit)
        )
        items = session.exec(statement).all()
    else:
        count_statement = (
            select(func.count())
            .select_from(Item)
            .where(Item.owner_id == current_user.id)
        )
        count = session.exec(count_statement).one()
        statement = (
            select(Item)
            .where(Item.owner_id == current_user.id)
            .order_by(col(Item.created_at).desc())
            .offset(skip).limit(limit)
        )
        items = session.exec(statement).all()

    items_with_data = [_get_item_data(item, session) for item in items]
    return {"data": items_with_data, "count": count}


@router.get("/{id}", response_model=dict[str, Any])
def read_item(session: SessionDep, current_user: CurrentUser, id: uuid.UUID) -> Any:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    return _get_item_data(item)


@router.post("/daemon/reconnect")
def reconnect_daemon(
    session: SessionDep, current_user: CurrentUser, daemon_id: str
) -> Any:
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    parts = daemon_id.split(":")
    if len(parts) < 3:
        raise HTTPException(status_code=400, detail="Invalid daemon_id format")
    
    host = parts[0]
    port = int(parts[1])
    api_key = ":".join(parts[2:])
    
    config = DaemonConfig(ip=host, port=port, api_key=api_key)
    result = connection_manager.reconnect_connection(config)
    if result.get("success"):
        sync_daemon_connection_state(config)
    return result


def _assert_unique_title_per_daemon(
    session: SessionDep,
    *,
    title: str,
    socket_host: str | None,
    socket_port: int | None,
    api_key: str | None,
    exclude_id: uuid.UUID | None = None,
) -> None:
    """同一 daemon（host:port:key）内终端标题必须唯一（workdir 以标题命名）。"""
    from sqlmodel import select

    stmt = select(Item).where(
        Item.title == title,
        Item.socket_host == socket_host,
        Item.socket_port == socket_port,
    )
    for other in session.exec(stmt).all():
        if exclude_id is None or other.id != exclude_id:
            raise HTTPException(
                status_code=400,
                detail=f"同一 daemon 下已存在标题为「{title}」的终端，请换个标题",
            )


@router.post("/", response_model=ItemPublic)
def create_item(
    *, session: SessionDep, current_user: CurrentUser, item_in: ItemCreate
) -> Any:
    _assert_unique_title_per_daemon(
        session,
        title=item_in.title,
        socket_host=item_in.socket_host,
        socket_port=item_in.socket_port,
        api_key=item_in.api_key,
    )
    item = Item.model_validate(item_in, update={"owner_id": current_user.id})
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


@router.put("/{id}", response_model=ItemPublic)
def update_item(
    *,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    item_in: ItemUpdate,
) -> Any:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    update_dict = item_in.model_dump(exclude_unset=True)
    item.sqlmodel_update(update_dict)
    _assert_unique_title_per_daemon(
        session,
        title=item.title,
        socket_host=item.socket_host,
        socket_port=item.socket_port,
        api_key=item.api_key,
        exclude_id=item.id,
    )
    session.add(item)
    session.commit()
    session.refresh(item)
    return item


@router.post("/{id}/start")
async def start_item(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Message:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    daemon_config = DaemonConfig(
        ip=item.socket_host,
        port=item.socket_port,
        api_key=item.api_key
    )
    
    terminal_service = TerminalService(connection_manager, socket_pool_facade)
    result = terminal_service.start_terminal(
        item_uuid=str(item.id),
        user_uuid=str(current_user.id),
        daemon_config=daemon_config
    )
    
    if result["success"]:
        item.status = ItemStatus.running
        item.socket_connected = True
        session.commit()
        return Message(message=result.get("message", "Item started successfully"))
    else:
        item.status = ItemStatus.error
        item.socket_connected = False
        session.commit()
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to start item"))


@router.post("/{id}/stop")
async def stop_item(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Message:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    terminal_service = TerminalService(connection_manager, socket_pool_facade)
    result = terminal_service.stop_terminal(
        daemon_id=f"{item.socket_host}:{item.socket_port}:{item.api_key}",
        item_uuid=str(item.id)
    )
    
    if result["success"]:
        item.status = ItemStatus.stopped
        item.socket_connected = False
        session.commit()
        return Message(message=result.get("message", "Item stopped successfully"))
    else:
        item.status = ItemStatus.error
        session.commit()
        raise HTTPException(status_code=500, detail=result.get("error", "Failed to stop item"))


@router.post("/{id}/restart")
async def restart_item(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Message:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    item.status = ItemStatus.stopping
    session.commit()
    
    daemon_config = DaemonConfig(
        ip=item.socket_host,
        port=item.socket_port,
        api_key=item.api_key
    )
    
    terminal_service = TerminalService(connection_manager, socket_pool_facade)
    
    stop_result = terminal_service.stop_terminal(
        daemon_id=f"{item.socket_host}:{item.socket_port}:{item.api_key}",
        item_uuid=str(item.id)
    )
    
    if not stop_result["success"]:
        logger.warning(f"Stop failed during restart: {stop_result.get('error')}")
    
    start_result = terminal_service.start_terminal(
        item_uuid=str(item.id),
        user_uuid=str(current_user.id),
        daemon_config=daemon_config
    )
    
    if start_result["success"]:
        item.status = ItemStatus.running
        item.socket_connected = True
        session.commit()
        return Message(message=start_result.get("message", "Item restarted successfully"))
    else:
        item.status = ItemStatus.error
        session.commit()
        raise HTTPException(status_code=500, detail=start_result.get("error", "Failed to restart item"))


@router.delete("/{id}")
def delete_item(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> Message:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    socket_pool_facade.cleanup_item_runtime(str(item.id))
    backend_conn_pool.remove_room_listen_conn(str(item.id))
    
    session.delete(item)
    session.commit()
    return Message(message="Item deleted successfully")


@router.get("/{id}/terminal-token")
def get_terminal_token(
    request: Request, session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> dict[str, Any]:
    from app.services.auth_service import auth_service
    
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    daemon_id, daemon_token = socket_pool_facade.get_item_token(str(item.id))

    if (
        not daemon_token
        and item.socket_host
        and item.socket_port
        and item.api_key
    ):
        daemon_config = DaemonConfig(
            ip=item.socket_host,
            port=item.socket_port,
            api_key=item.api_key,
        )
        terminal_service = TerminalService(connection_manager, socket_pool_facade)
        terminal_service.restore_running_terminal(
            item_uuid=str(item.id),
            owner_uuid=str(item.owner_id),
            daemon_config=daemon_config,
        )
        daemon_id, daemon_token = socket_pool_facade.get_item_token(str(item.id))

    if not daemon_token:
        raise HTTPException(status_code=400, detail="Item not running or token not available")
    
    daemon_status = _get_daemon_status(item)
    if not daemon_status["daemon_online"]:
        raise HTTPException(status_code=400, detail="Daemon is not connected")
    
    if not item.api_key:
        raise HTTPException(status_code=400, detail="Item has no daemon key")
    temp_token_info = auth_service.generate_terminal_token_hmac(
        item_uuid=str(id),
        user_id=str(current_user.id),
        api_key=item.api_key,
        expire_minutes=5,
    )
    
    logger.info(f"Generated terminal temp token for user={current_user.id}, item={id}")
    
    return {
        "success": True,
        "temp_token": temp_token_info["token"],
        "item_uuid": str(id),
        "user_uuid": str(current_user.id),
        "ws_url": _build_terminal_ws_url(request, item),
        "daemon_id": daemon_id,
        "expire_seconds": temp_token_info["expires_in"]
    }


@router.post("/{id}/verify-terminal-token")
def verify_terminal_token(
    session: SessionDep,
    id: uuid.UUID,
    temp_token: str = Body(...),
    item_uuid: str = Body(...)
) -> dict[str, Any]:
    from app.services.auth_service import auth_service
    
    if str(id) != item_uuid:
        return {"success": False, "error": "Item UUID mismatch"}

    item = session.get(Item, id)
    if not item or not item.api_key:
        result = {"success": False, "error": "Item has no daemon key"}
    else:
        result = auth_service.verify_terminal_token_hmac(
            temp_token, item_uuid, item.api_key
        )
    
    if result["success"]:
        logger.info(f"Terminal temp token verified for item={id}, user={result['user_id']}")
    else:
        logger.warning(f"Terminal temp token verification failed: {result['error']}")
    
    return result


@router.get("/{id}/files/tree")
def get_item_file_tree(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    path: str = "/",
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.list_tree(item=item, path=path)
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.get("/{id}/files/default-path")
def get_item_default_file_path(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.get_default_path(item=item)
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.get("/{id}/files/content")
def get_item_file_content(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    path: str,
    preview_bytes: int = 256 * 1024,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.get_content(item=item, path=path, preview_bytes=preview_bytes)
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.post("/{id}/files/write")
def write_item_file_content(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: FileWriteRequest,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.write_content(
            item=item,
            path=body.path,
            content=body.content,
            encoding=body.encoding,
        )
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.post("/{id}/files/download-ticket")
def issue_item_download_ticket(
    request: Request,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: FilePathRequest,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.issue_download_ticket(
            item=item,
            actor_user_id=str(current_user.id),
            path=body.path,
            request_host=request.url.hostname or "",
        )
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.post("/{id}/files/upload-ticket")
def issue_item_upload_ticket(
    request: Request,
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: UploadTicketRequest,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.issue_upload_ticket(
            item=item,
            actor_user_id=str(current_user.id),
            path=body.path,
            allow_overwrite=body.allow_overwrite,
            request_host=request.url.hostname or "",
        )
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.post("/{id}/files/mkdir")
def create_item_directory(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: FilePathRequest,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.create_directory(item=item, path=body.path)
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.post("/{id}/files/rename")
def rename_item_path(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: RenamePathRequest,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.rename_path(
            item=item,
            path=body.path,
            target_path=body.target_path,
        )
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.post("/{id}/files/copy")
def copy_item_path(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: CopyPathRequest,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.copy_path(
            item=item,
            path=body.path,
            target_path=body.target_path,
        )
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.post("/{id}/files/delete")
def delete_item_path(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: FilePathRequest,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    try:
        return item_file_service.delete_path(item=item, path=body.path)
    except ItemFileServiceError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.post("/file-ticket/verify")
def verify_file_ticket(
    request: Request,
    body: FileTicketVerifyRequest,
) -> dict[str, Any]:
    from app.services.auth_service import auth_service

    daemon_api_key = request.headers.get("X-Daemon-Api-Key", "").strip()
    if not daemon_api_key:
        raise HTTPException(status_code=401, detail="Missing daemon api key")

    result = auth_service.validate_file_ticket(
        ticket=body.ticket,
        op=body.op,
        daemon_api_key=daemon_api_key,
        mark_used=True,
    )
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])

    return result


@router.get("/{id}/subscribers")
def get_item_subscribers(
    session: SessionDep, current_user: CurrentUser, id: uuid.UUID
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    daemon_status = _get_daemon_status(item)
    if not daemon_status["daemon_online"]:
        return {
            "success": True,
            "item_uuid": str(id),
            "subscribers": [],
            "browser_count": 0,
            "backend_connected": False,
            "daemon_online": False
        }
    
    connection = connection_manager.get_connection(
        f"{item.socket_host}:{item.socket_port}:{item.api_key}"
    )
    
    if not connection or not connection.is_connected():
        return {
            "success": True,
            "item_uuid": str(id),
            "subscribers": [],
            "browser_count": 0,
            "backend_connected": False,
            "daemon_online": True,
            "error": "Backend not connected to daemon"
        }
    
    result = connection.get_item_subscribers_http(str(id))
    
    result["daemon_online"] = True
    return result


@router.post("/{id}/disconnect-subscriber")
def disconnect_item_subscriber(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    user_uuid: str = Body(default=None),
    ip_address: str = Body(default=None)
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    if not user_uuid and not ip_address:
        raise HTTPException(status_code=400, detail="Must provide user_uuid or ip_address")
    
    daemon_status = _get_daemon_status(item)
    if not daemon_status["daemon_online"]:
        raise HTTPException(status_code=400, detail="Daemon is not connected")
    
    connection = connection_manager.get_connection(
        f"{item.socket_host}:{item.socket_port}:{item.api_key}"
    )
    
    if not connection or not connection.is_connected():
        raise HTTPException(status_code=400, detail="Backend not connected to daemon")
    
    result = connection.disconnect_connection_http(
        item_uuid=str(id),
        user_uuid=user_uuid,
        ip_address=ip_address
    )
    
    return result


@router.get("/{id}/plan")
def get_item_plan(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> dict[str, Any]:
    """Current agent plan scratchpad for this item (Codex-style update_plan).

    The plan lives on the item's most recently updated non-terminal reply
    ticket; the response shape is unchanged for the frontend."""
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    from app.services.agent.reply_ticket import reply_ticket_manager

    _ticket, plan = reply_ticket_manager.latest_plan_for_item(str(id))
    return {
        "item_id": str(id),
        "plan": plan,
        "plans": reply_ticket_manager.active_plans_for_item(str(id)),
        "updated": bool(plan),
    }


@router.post("/{id}/plan/{ticket_id}/cancel")
def cancel_item_plan(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    ticket_id: str,
) -> dict[str, Any]:
    """取消指定任务：中断轮次、杀掉后台 job、票据标记取消、步骤一并取消。"""
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    from app.services.agent.reply_ticket import reply_ticket_manager

    ticket = reply_ticket_manager.get(ticket_id)
    if not ticket or ticket.item_id != str(id):
        raise HTTPException(status_code=404, detail="Plan not found")

    # 1. 先中断正在执行的轮次：否则运行中的轮次会继续执行已取消计划的
    #    后续步骤，甚至用 update_plan 把计划"复活"。
    from app.services.agent.session import agent_session_manager

    agent_session = agent_session_manager.get_session(str(id))
    if agent_session is not None:
        try:
            agent_session.abort(clear_queue=True)
            agent_session.emit_output("任务已由用户取消", "agent_warning")
            agent_session.emit_status("idle", "")
        except Exception:
            logger.exception("[Items] Failed to abort session on plan cancel: item=%s", id)

    # 2. 杀掉票据上的 daemon 后台 job 并摘除——否则远端进程继续跑，
    #    完成后轮询线程/resume sweep 还会把结果送回来。
    cancelled_jobs: list[str] = []
    pending_job_ids = list(getattr(ticket, "pending_job_ids", None) or [])
    if pending_job_ids:
        _daemon_status, connection = _get_item_daemon_connection(item)
        for job_id in pending_job_ids:
            if connection is not None:
                try:
                    connection.cancel_job_http(item_uuid=str(id), job_id=job_id)
                except Exception:
                    logger.exception(
                        "[Items] Failed to cancel daemon job %s on plan cancel: item=%s",
                        job_id,
                        id,
                    )
            reply_ticket_manager.take_pending_job(ticket_id, job_id)
            cancelled_jobs.append(job_id)

    # 3. 票据标记取消，未完成步骤一并取消。
    reply_ticket_manager.mark_failed(ticket_id, "用户取消了该任务")
    finalized_plan: list[dict] = []
    with reply_ticket_manager._lock:
        current = reply_ticket_manager._tickets.get(
            reply_ticket_manager._resolve_ticket_id_locked(ticket_id)
        )
        if current and current.plan:
            finalized_plan = [
                {
                    **step,
                    "status": (
                        step.get("status")
                        if step.get("status") in {"completed", "cancelled"}
                        else "cancelled"
                    ),
                }
                for step in current.plan
            ]
    # update_ticket_plan 会同步持久化——直接改内存不落盘的话，重启恢复会
    # 把已取消的计划当成未完成计划复活。
    reply_ticket_manager.update_ticket_plan(ticket_id, [])

    # 4. 广播取消后的终态快照，前端条带立即消失（不用等轮询）。
    try:
        from app.services.agent.stream_manager import stream_manager

        stream_manager.broadcast_chat_event(
            str(id),
            {
                "type": "plan_updated",
                "item_id": str(id),
                "plan": finalized_plan,
                "explanation": "cancelled",
            },
        )
    except Exception:
        pass
    return {"success": True, "ticket_id": ticket_id, "cancelled_jobs": cancelled_jobs}


@router.get("/{id}/jobs")
def list_item_jobs(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    daemon_status, connection = _get_item_daemon_connection(item)
    if not connection:
        return {
            "success": True,
            "item_uuid": str(id),
            "jobs": [],
            "count": 0,
            "daemon_online": daemon_status["daemon_online"],
            "error": (
                "Backend not connected to daemon"
                if daemon_status["daemon_online"]
                else "Daemon is not connected"
            ),
        }

    result = connection.list_jobs_http(item_uuid=str(id))
    result["daemon_online"] = True
    return result


@router.post("/{id}/jobs/{job_id}/cancel")
def cancel_item_job(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    job_id: str,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    daemon_status, connection = _get_item_daemon_connection(item)
    if not connection:
        raise HTTPException(
            status_code=400,
            detail=(
                "Daemon is not connected"
                if not daemon_status["daemon_online"]
                else "Backend not connected to daemon"
            ),
        )

    result = connection.cancel_job_http(item_uuid=str(id), job_id=job_id)
    if not result.get("success") and result.get("error") != "No running job for item":
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Failed to cancel job"),
        )

    from app.services.agent.session import agent_session_manager

    agent_session = agent_session_manager.get_session(str(id))
    if agent_session:
        agent_session.clear_terminal_job()

    result["daemon_online"] = True
    return result


@router.get("/{id}/output")
def get_item_output(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    lines: int = 64
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    if lines < 1 or lines > 1000:
        raise HTTPException(status_code=400, detail="lines must be between 1 and 1000")
    
    output = log_manager.get_last_lines(str(id), lines)
    
    return {
        "success": True,
        "item_uuid": str(id),
        "lines": lines,
        "output": sanitize_terminal_text(output) if output else output
    }


@router.get("/{id}/terminal-command-state")
def get_terminal_command_state(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    from app.services.terminal_command_state import terminal_command_state_manager

    return terminal_command_state_manager.snapshot(str(id))


@router.post("/{id}/terminal-command-state")
def record_terminal_command_state(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: TerminalCommandStateRequest,
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    command = body.command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="command is required")

    from app.services.terminal_command_state import terminal_command_state_manager

    return terminal_command_state_manager.record(
        str(id),
        command,
        source="web",
    )


@router.post("/{id}/generate-filter", response_model=GenerateFilterResponse)
def generate_item_filter(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    body: GenerateFilterRequest,
) -> GenerateFilterResponse:
    try:
        item_handler, item = get_item_handler_for_item(session, id, current_user)

        existing_rules = body.existing_rules
        if existing_rules is None:
            existing_rules = (
                item.input_filter_rules
                if body.target == "input"
                else item.output_filter_rules
            )

        system_prompt, user_prompt = _build_filter_generation_prompts(
            target=body.target,
            existing_rules=existing_rules or {},
            instruction=body.instruction,
        )
        payload = generate_json_payload(
            item_handler,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        rules = _normalize_generated_filter_rules(payload.get("rules", payload))
    except LlmGenerationError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)

    return GenerateFilterResponse(
        target=body.target,
        rules=rules,
        explanation=str(payload.get("explanation", "")).strip(),
        item_handler_id=str(item_handler.id),
        model=item_handler.model or "",
    )


@router.post("/{id}/test-input-filter")
def test_input_filter(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    test_text: str = Body(..., embed=True),
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    config = InputFilterConfig.from_item(item)
    input_filter = InputFilter(config)
    
    stream_data = {"stdout": test_text, "stderr": ""}
    result = input_filter.filter(stream_data)
    
    if result is None:
        return {
            "success": True,
            "result": "",
            "event_type": "blocked",
            "matched_filters": [],
            "matches": [],
        }
    
    return {
        "success": True,
        "result": result.raw_content,
        "event_type": result.event_type.value,
        "matched_filters": result.matched_filters,
        "matches": result.matches[:20],
    }


@router.post("/{id}/test-output-filter")
def test_output_filter(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
    command: str = Body(..., embed=True),
) -> dict[str, Any]:
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)
    
    config = OutputFilterConfig.from_item(item)
    output_filter = OutputFilter(config)
    
    result = output_filter.filter(command)
    
    return {
        "success": True,
        "original_command": command,
        "result": "" if result.is_blocked else result.command,
        "action": result.action.value,
        "reason": result.reason,
        "is_allowed": result.is_allowed,
        "is_blocked": result.is_blocked,
        "matched_filters": result.matched_filters,
        "matches": result.matches[:20],
    }


@router.get("/{id}/token-usage")
def get_token_usage(
    current_user: CurrentUser,
    id: uuid.UUID,
) -> dict[str, Any]:
    from app.services.agent.memory.scope import resolve_handler_id
    from app.services.agent.token_usage import token_usage_tracker

    handler_id = resolve_handler_id(str(id))
    stats = token_usage_tracker.get_handler_stats(handler_id)
    if stats is None:
        return {
            "handler_id": handler_id,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "total_turns": 0,
            "first_seen": "",
            "last_seen": "",
            "models": [],
        }
    return stats


@router.get("/{id}/token-usage/by-task")
def get_token_usage_by_task(
    session: SessionDep,
    current_user: CurrentUser,
    id: uuid.UUID,
) -> dict[str, Any]:
    """Token usage grouped by reply ticket (one task per agent turn)."""
    item = session.get(Item, id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    _check_item_permission(item, current_user)

    from app.services.agent.memory.scope import resolve_handler_id
    from app.services.agent.reply_ticket import reply_ticket_manager
    from app.services.agent.token_usage import token_usage_tracker

    tasks: list[dict[str, Any]] = []
    for row in token_usage_tracker.get_task_stats(resolve_handler_id(str(id)), limit=50):
        ticket = reply_ticket_manager.get(row["reply_ticket_id"])
        plan: list[dict[str, Any]] = []
        if ticket is not None:
            plan = [
                {
                    "step": str(entry.get("step") or ""),
                    "status": str(entry.get("status") or ""),
                }
                for entry in (ticket.plan or [])
                if isinstance(entry, dict)
            ]
        tasks.append(
            {
                "reply_ticket_id": row["reply_ticket_id"],
                "source_type": str(ticket.source_type) if ticket else "",
                "source_label": str(ticket.source_label) if ticket else "",
                "request_message": str(ticket.request_message or "")[:80]
                if ticket
                else "",
                "plan": plan,
                "ticket_status": str(ticket.status) if ticket else "",
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "total_tokens": row["total_tokens"],
                "turns": row["turns"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
            }
        )
    return {"handler_id": resolve_handler_id(str(id)), "tasks": tasks}
