from __future__ import annotations

import inspect
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import APIRouter, FastAPI, Header, HTTPException, WebSocket

from app.core.config import settings
from app.plugins.robot.bridge.rate_limit import send_text_with_rate_limit
from app.plugins.robot.contracts import (
    RobotBridgeCallApiRequest,
    RobotBridgeReloadResponse,
    RobotBridgeSendRequest,
    RobotDispatchResponse,
    RobotInboundMessage,
)
from app.plugins.robot.debug_log import (
    preview_text,
    record_loaded_robot_event,
    record_robot_event,
)
from app.plugins.robot.platforms import (
    build_inbound_message,
    normalize_robot_platform_id,
    resolve_bot_identity,
    resolve_platform_from_bot,
)
from app.services.runtime_monitor import collect_backend_runtime_stats

logger = logging.getLogger(__name__)

_bridge_router: APIRouter | None = None
_loaded_robots: list = []
_robot_id_by_identity: dict[str, str] = {}
_identity_by_robot_id: dict[str, str] = {}
_seen_connected_robot_ids: set[str] = set()
_last_platform_event_at_by_robot_id: dict[str, str] = {}
_last_message_event_at_by_robot_id: dict[str, str] = {}
_onebot_socket_status_by_robot_id: dict[str, dict[str, Any]] = {}
_stale_onebot_message_max_age_seconds = 60.0
_stale_onebot_message_future_grace_seconds = 900.0
_onebot_message_timezone_offset_candidates_seconds = (8 * 3600,)
_initialized = False
_connection_errors: dict[str, dict[str, str]] = {}


def _user_tmp_dir() -> Path:
    base = Path(tempfile.gettempdir())
    if os.name == "nt":
        return base
    # /tmp is shared and tmpfs can race on concurrent creation; keep per-user.
    per_user = base / f"termpaws-{os.getuid()}"
    per_user.mkdir(mode=0o700, exist_ok=True)
    return per_user


_error_file_path: str = str(_user_tmp_dir() / "robot_bridge_errors.json")
_identity_file_path: str = str(_user_tmp_dir() / "robot_bridge_identities.json")
_singleton_lock_path: str = str(_user_tmp_dir() / "termpaws_robot_bridge.lock")
_owner_info_path: str = str(_user_tmp_dir() / "termpaws_robot_bridge_owner.json")
_singleton_lock_file: Any | None = None
_singleton_lock_owner = False
_singleton_lock_guard = threading.Lock()
_ipc_server: uvicorn.Server | None = None
_ipc_thread: threading.Thread | None = None
_ipc_base_url: str | None = None
_startup_callback: Any | None = None
_shutdown_callback: Any | None = None
_loaded_robot_config_signature: dict[str, Any] = {}
_app_ref: Any | None = None


def set_bridge_app(app: Any) -> None:
    global _app_ref
    _app_ref = app


async def hot_reload_bridge() -> dict[str, Any]:
    """Restart the embedded bridge in place (no process restart)."""
    global _initialized, _bridge_router
    await stop_embedded_bridge()
    _initialized = False
    _bridge_router = None
    router = init_embedded_bridge()
    if _app_ref is not None:
        _app_ref.router.routes[:] = [
            route
            for route in _app_ref.router.routes
            if not str(getattr(route, "path", "")).startswith("/robot-bridge")
        ]
        if router is not None:
            _app_ref.include_router(router)
    await start_embedded_bridge()
    loaded, _, _, _ = _current_enabled_robot_config_signature()
    return {
        "success": True,
        "detail": f"Bridge reloaded in place with {len(loaded)} robot(s)",
    }


def _normalize_connection_errors(raw: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        return {}
    normalized: dict[str, dict[str, str]] = {}
    for robot_id, value in raw.items():
        if isinstance(value, dict):
            message = str(value.get("message") or value.get("error") or "")
            timestamp = str(value.get("timestamp") or "")
        else:
            message = str(value)
            timestamp = ""
        if message:
            normalized[str(robot_id)] = {
                "message": message,
                "timestamp": timestamp,
            }
    return normalized


def _dispatch_failure_payload(inbound: RobotInboundMessage) -> dict[str, str]:
    return {
        "target_type": inbound.reply_target.target_type,
        "target_id": inbound.reply_target.target_id,
    }


def _load_connection_errors() -> dict[str, dict[str, str]]:
    global _connection_errors
    try:
        import json

        with open(_error_file_path) as f:
            _connection_errors = _normalize_connection_errors(json.load(f))
    except Exception:
        _connection_errors = {}
    return _connection_errors


def _save_connection_error(robot_id: str, error: str) -> None:
    global _connection_errors
    _connection_errors[robot_id] = {
        "message": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        import json

        with open(_error_file_path, "w") as f:
            json.dump(_connection_errors, f)
    except Exception:
        pass


def _is_rate_limit_error(error: str) -> bool:
    return "100017" in error or "频率限制" in error or "rate limit" in error.lower()


def _acquire_singleton_lock() -> bool:
    global _singleton_lock_file, _singleton_lock_owner
    with _singleton_lock_guard:
        if _singleton_lock_owner:
            return True

        try:
            Path(_singleton_lock_path).parent.mkdir(parents=True, exist_ok=True)
            lock_file = open(_singleton_lock_path, "a+")
        except OSError as exc:
            logger.warning(
                "[Bridge] Cannot open lock file %s: %s; bridge stays idle",
                _singleton_lock_path,
                exc,
            )
            return False
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock_file.close()
            return False

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(os.getpid()))
        lock_file.flush()
        _singleton_lock_file = lock_file
        _singleton_lock_owner = True
        return True


def _release_singleton_lock() -> None:
    global _singleton_lock_file, _singleton_lock_owner
    with _singleton_lock_guard:
        lock_file = _singleton_lock_file
        if lock_file is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            lock_file.close()
        except OSError:
            pass
        _singleton_lock_file = None
        _singleton_lock_owner = False


def _write_owner_info(base_url: str) -> None:
    try:
        import json

        with open(_owner_info_path, "w") as f:
            json.dump(
                {
                    "pid": os.getpid(),
                    "base_url": base_url,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                f,
            )
    except Exception:
        logger.exception("[Bridge] Failed to write IPC owner info")


def _read_owner_info() -> dict[str, Any] | None:
    try:
        import json

        with open(_owner_info_path) as f:
            raw = json.load(f)
        if isinstance(raw, dict) and raw.get("base_url"):
            return raw
    except Exception:
        return None
    return None


def _remove_owner_info() -> None:
    try:
        os.remove(_owner_info_path)
    except FileNotFoundError:
        pass
    except Exception:
        logger.exception("[Bridge] Failed to remove IPC owner info")


def _load_enabled_robot_configs_from_db() -> tuple[
    list, dict[str, str], dict[str, str]
]:
    from sqlmodel import Session, select

    from app.core.db import engine
    from app.models import Robot
    from app.plugins.robot.platforms import (
        get_robot_platform,
        get_robot_runtime_config,
    )

    loaded_robots: list = []
    robot_id_by_identity: dict[str, str] = {}
    identity_by_robot_id: dict[str, str] = {}

    with Session(engine) as session:
        robots = session.exec(
            select(Robot).where(
                Robot.is_enabled == True,  # noqa: E712
                Robot.provider == "nonebot2",
            )
        ).all()

    for robot in robots:
        try:
            platform_id = normalize_robot_platform_id(robot.platform or robot.protocol)
            get_robot_platform(platform_id)
            get_robot_runtime_config(robot)
            identity = resolve_robot_identity(platform_id, robot)
        except Exception as exc:
            logger.warning(
                "[Bridge] Skip robot %s because runtime config is invalid: %s",
                robot.id,
                exc,
            )
            continue

        if identity in robot_id_by_identity:
            logger.warning(
                "[Bridge] Duplicate robot identity %s detected, keep first robot only",
                identity,
            )
            continue

        loaded_robots.append(robot)
        robot_id_by_identity[identity] = str(robot.id)
        identity_by_robot_id[str(robot.id)] = identity

    return loaded_robots, robot_id_by_identity, identity_by_robot_id


def _robot_config_signature(robots: list) -> dict[str, Any]:
    return {
        str(robot.id): {
            "platform": normalize_robot_platform_id(robot.platform or robot.protocol),
            "config": robot.config if isinstance(robot.config, dict) else {},
            "enabled": bool(robot.is_enabled),
        }
        for robot in robots
    }


def _current_enabled_robot_config_signature() -> tuple[
    list, dict[str, str], dict[str, str], dict[str, Any]
]:
    robots, robot_id_by_identity, identity_by_robot_id = (
        _load_enabled_robot_configs_from_db()
    )
    return (
        robots,
        robot_id_by_identity,
        identity_by_robot_id,
        _robot_config_signature(robots),
    )


async def _forward_to_owner(
    path: str,
    *,
    body: Any | None = None,
    timeout: float = 10.0,
    method: str = "POST",
) -> dict[str, Any]:
    owner_info = _read_owner_info()
    if not owner_info:
        raise HTTPException(
            status_code=503, detail="Robot bridge owner is not available"
        )

    base_url = str(owner_info["base_url"]).rstrip("/")
    shared_secret = settings.ROBOT_BRIDGE_SHARED_SECRET or settings.SECRET_KEY
    headers = {"X-TermPaws-Bridge-Token": shared_secret}
    if body is not None:
        headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=timeout) as client:
        if body is None:
            if method.upper() == "GET":
                response = await client.get(f"{base_url}{path}", headers=headers)
            else:
                response = await client.post(f"{base_url}{path}", headers=headers)
        else:
            response = await client.post(
                f"{base_url}{path}",
                headers=headers,
                content=body.model_dump_json()
                if hasattr(body, "model_dump_json")
                else None,
                json=None if hasattr(body, "model_dump_json") else body,
            )
        response.raise_for_status()
        return response.json()


def _start_ipc_server(send_handler, health_handler, reload_handler) -> str:
    global _ipc_base_url, _ipc_server, _ipc_thread
    if _ipc_base_url:
        return _ipc_base_url

    app = FastAPI()

    @app.post("/internal/send")
    async def ipc_send(
        body: RobotBridgeSendRequest,
        x_termpaws_bridge_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _assert_bridge_permission(x_termpaws_bridge_token)
        return await send_handler(body)

    @app.get("/internal/health")
    async def ipc_health(
        x_termpaws_bridge_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _assert_bridge_permission(x_termpaws_bridge_token)
        return await health_handler()

    @app.post("/internal/reload")
    async def ipc_reload(
        x_termpaws_bridge_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _assert_bridge_permission(x_termpaws_bridge_token)
        return await reload_handler()

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    started = threading.Event()
    failed: list[BaseException] = []

    def run_server() -> None:
        try:
            server.config.load()
            server.lifespan = server.config.lifespan_class(server.config)
            import asyncio

            async def serve() -> None:
                await server.startup()
                started.set()
                if server.should_exit:
                    return
                await server.main_loop()
                await server.shutdown()

            asyncio.run(serve())
        except BaseException as exc:
            failed.append(exc)
            started.set()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    if not started.wait(timeout=5.0):
        raise RuntimeError("Timed out starting robot bridge IPC server")
    if failed:
        raise RuntimeError(f"Failed to start robot bridge IPC server: {failed[0]}")
    if not server.servers:
        raise RuntimeError("Robot bridge IPC server did not expose a socket")

    sockname = server.servers[0].sockets[0].getsockname()
    base_url = f"http://127.0.0.1:{sockname[1]}"
    _ipc_server = server
    _ipc_thread = thread
    _ipc_base_url = base_url
    _write_owner_info(base_url)
    logger.info("[Bridge] IPC owner server started at %s", base_url)
    return base_url


def _stop_ipc_server() -> None:
    global _ipc_base_url, _ipc_server, _ipc_thread
    if _ipc_server is not None:
        _ipc_server.should_exit = True
    if _ipc_thread is not None and _ipc_thread.is_alive():
        _ipc_thread.join(timeout=3.0)
    _ipc_server = None
    _ipc_thread = None
    _ipc_base_url = None
    _remove_owner_info()


def _serialize_bot_snapshot(bot: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "self_id": str(getattr(bot, "self_id", "") or ""),
        "adapter": None,
        "class": bot.__class__.__name__,
    }
    try:
        snapshot["adapter"] = str(bot.adapter.get_name())
    except Exception:
        snapshot["adapter"] = None

    bot_info = getattr(bot, "bot_info", None)
    if bot_info is not None:
        snapshot["bot_info"] = {
            key: value
            for key, value in {
                "id": getattr(bot_info, "id", None),
                "username": getattr(bot_info, "username", None),
                "name": getattr(bot_info, "name", None),
            }.items()
            if value is not None
        }
    return snapshot


def _serialize_event_payload(event: Any) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        try:
            data = event.model_dump()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    if hasattr(event, "dict"):
        try:
            data = event.dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _summarize_platform_event(event: Any) -> dict[str, Any]:
    payload = _serialize_event_payload(event)
    sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
    summary = {
        "post_type": payload.get("post_type"),
        "meta_event_type": payload.get("meta_event_type"),
        "message_type": payload.get("message_type"),
        "sub_type": payload.get("sub_type"),
        "user_id": payload.get("user_id"),
        "group_id": payload.get("group_id"),
        "message_id": payload.get("message_id"),
        "sender_user_id": sender.get("user_id"),
        "sender_nickname": sender.get("nickname"),
        "sender_card": sender.get("card"),
        "sender_role": sender.get("role"),
        "raw_message": payload.get("raw_message"),
    }
    return {key: value for key, value in summary.items() if value not in (None, "")}


def _is_onebot_heartbeat_event(event: Any) -> bool:
    payload = _serialize_event_payload(event)
    return (
        str(payload.get("post_type") or "").strip().lower() == "meta_event"
        and str(payload.get("meta_event_type") or "").strip().lower() == "heartbeat"
    )


def _coerce_unix_event_time(value: Any) -> float | None:
    try:
        event_time = float(value)
    except (TypeError, ValueError):
        return None
    if event_time <= 0:
        return None
    if event_time > 10_000_000_000:
        event_time = event_time / 1000
    return event_time


def _onebot_event_time(event: Any, payload: dict[str, Any]) -> float | None:
    event_time = _coerce_unix_event_time(payload.get("time"))
    if event_time is not None:
        return event_time
    return _coerce_unix_event_time(getattr(event, "time", None))


def _stale_onebot_message_event_payload(
    event: Any,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    if str(payload.get("post_type") or "").strip().lower() != "message":
        return None
    event_time = _onebot_event_time(event, payload)
    if event_time is None:
        return None

    received_at = time.time()
    event_age_seconds = received_at - event_time
    age_candidates = [(event_age_seconds, 0)]
    age_candidates.extend(
        (event_age_seconds - offset_seconds, offset_seconds)
        for offset_seconds in _onebot_message_timezone_offset_candidates_seconds
    )

    for adjusted_age_seconds, _offset_seconds in age_candidates:
        if (
            -_stale_onebot_message_future_grace_seconds
            <= adjusted_age_seconds
            <= _stale_onebot_message_max_age_seconds
        ):
            return None

    adjusted_age_seconds, timezone_offset_seconds = min(
        age_candidates,
        key=lambda candidate: abs(candidate[0]),
    )
    return {
        "event_time": event_time,
        "received_at": received_at,
        "event_age_seconds": event_age_seconds,
        "adjusted_event_age_seconds": adjusted_age_seconds,
        "timezone_offset_seconds": timezone_offset_seconds,
        "max_age_seconds": _stale_onebot_message_max_age_seconds,
        "future_grace_seconds": _stale_onebot_message_future_grace_seconds,
        "event_summary": _summarize_platform_event(event),
    }


def _decode_websocket_body(message: dict[str, Any]) -> tuple[str | None, int | None]:
    text = message.get("text")
    if isinstance(text, str):
        return text, None

    body = message.get("bytes")
    if isinstance(body, bytes):
        try:
            return body.decode("utf-8"), len(body)
        except UnicodeDecodeError:
            return None, len(body)

    return None, None


def _summarize_websocket_asgi_message(
    message: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    text, byte_length = _decode_websocket_body(message)
    payload: dict[str, Any] = {"asgi_type": message.get("type")}
    if byte_length is not None:
        payload["bytes_length"] = byte_length
    if text is not None:
        payload["text_preview"] = preview_text(text)
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = None
        if isinstance(raw, dict):
            params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
            for key, value in {
                "self_id": raw.get("self_id") or params.get("self_id"),
                "post_type": raw.get("post_type"),
                "meta_event_type": raw.get("meta_event_type"),
                "message_type": raw.get("message_type"),
                "sub_type": raw.get("sub_type"),
                "user_id": raw.get("user_id"),
                "group_id": raw.get("group_id"),
                "message_id": raw.get("message_id"),
                "raw_message": raw.get("raw_message"),
                "action": raw.get("action"),
                "echo": raw.get("echo"),
                "status": raw.get("status"),
                "retcode": raw.get("retcode"),
                "interval": raw.get("interval"),
            }.items():
                if value not in (None, ""):
                    payload[key] = value
    if message.get("code") is not None:
        payload["code"] = message.get("code")
    if message.get("reason"):
        payload["reason"] = message.get("reason")
    return text, payload


def _resolve_robot_id_from_onebot_socket_message(
    platform_id: str,
    message: dict[str, Any],
) -> str | None:
    text, _ = _decode_websocket_body(message)
    if not text:
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    self_id = raw.get("self_id") or params.get("self_id")
    if self_id is None:
        return None
    return _robot_id_by_identity.get(f"{platform_id}:{str(self_id).strip()}")


def _record_websocket_debug_event(
    robot_id: str | None,
    platform_id: str,
    *,
    direction: str,
    event: str,
    status: str = "ok",
    message: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    robot_ids = (
        [robot_id]
        if robot_id
        else [
            candidate_robot_id
            for candidate_robot_id, identity in _identity_by_robot_id.items()
            if identity.startswith(f"{platform_id}:")
        ]
    )
    for candidate_robot_id in robot_ids:
        _record_bridge_event(
            candidate_robot_id,
            direction=direction,
            event=event,
            status=status,
            message=message,
            payload=payload,
        )


def _should_record_websocket_debug_event(
    event_name: str,
    payload: dict[str, Any],
) -> bool:
    if event_name in {
        "websocket_connect",
        "websocket_accept",
        "websocket_disconnect",
        "websocket_close",
        "websocket_closed",
        "websocket_error",
    }:
        return True
    if payload.get("post_type") == "message":
        return True
    meta_event_type = str(payload.get("meta_event_type") or "").strip().lower()
    if meta_event_type and meta_event_type != "heartbeat":
        return True
    retcode = payload.get("retcode")
    if retcode not in (None, 0, "0"):
        return True
    action = str(payload.get("action") or "")
    return action.startswith("send_")


def _websocket_connection_payload(websocket: WebSocket, path: str) -> dict[str, Any]:
    client = websocket.client
    return {
        "socket_path": path,
        "client": f"{client.host}:{client.port}" if client else None,
    }


def _update_onebot_socket_status(
    robot_id: str | None,
    *,
    connected: bool,
    event: str,
    direction: str,
    payload: dict[str, Any] | None = None,
) -> None:
    if not robot_id:
        return

    now = datetime.now(timezone.utc).isoformat()
    existing = _onebot_socket_status_by_robot_id.get(robot_id, {})
    status: dict[str, Any] = {
        **existing,
        "connected": connected,
        "event": event,
        "direction": direction,
        "last_event_at": now,
    }
    if payload:
        for key in (
            "socket_path",
            "client",
            "asgi_type",
            "self_id",
            "post_type",
            "meta_event_type",
            "message_type",
            "action",
            "echo",
            "code",
            "reason",
        ):
            value = payload.get(key)
            if value not in (None, ""):
                status[key] = value
    _onebot_socket_status_by_robot_id[robot_id] = status
    if connected:
        _seen_connected_robot_ids.add(robot_id)
    else:
        _seen_connected_robot_ids.discard(robot_id)


async def _call_optional_adapter_hook(adapter: Any, hook_name: str) -> bool:
    adapter_name = (
        str(adapter.get_name())
        if hasattr(adapter, "get_name")
        else adapter.__class__.__name__
    )
    hook = getattr(adapter, hook_name, None)
    if not callable(hook):
        logger.info(
            "[Bridge] Adapter %s has no %s hook; skip",
            adapter_name,
            hook_name,
        )
        return False

    result = hook()
    if inspect.isawaitable(result):
        await result
    return True


async def _run_logged_onebot_reverse_ws(
    *,
    websocket: WebSocket,
    path: str,
    nonebot_app: Any,
) -> None:
    platform_id = "onebot_v11"
    scope = dict(websocket.scope)
    scope["path"] = path
    scope["root_path"] = ""
    state: dict[str, Any] = {
        "robot_id": None,
        "disconnect_recorded": False,
        "close_recorded": False,
    }
    connection_payload = _websocket_connection_payload(websocket, path)
    _record_websocket_debug_event(
        None,
        platform_id,
        direction="platform_to_bridge",
        event="websocket_connect",
        payload=connection_payload,
    )

    async def receive() -> dict[str, Any]:
        raw_message = await websocket.receive()
        resolved_robot_id = _resolve_robot_id_from_onebot_socket_message(
            platform_id,
            raw_message,
        )
        if resolved_robot_id:
            state["robot_id"] = resolved_robot_id
        text, payload = _summarize_websocket_asgi_message(raw_message)
        payload.update(connection_payload)
        asgi_type = raw_message.get("type")
        event_name = (
            "websocket_disconnect"
            if asgi_type == "websocket.disconnect"
            else "websocket_receive"
        )
        if event_name == "websocket_disconnect":
            state["disconnect_recorded"] = True
        _update_onebot_socket_status(
            state.get("robot_id"),
            connected=event_name != "websocket_disconnect",
            event=event_name,
            direction="platform_to_bridge",
            payload=payload,
        )
        if _should_record_websocket_debug_event(event_name, payload):
            _record_websocket_debug_event(
                state.get("robot_id"),
                platform_id,
                direction="platform_to_bridge",
                event=event_name,
                message=preview_text(text) if text else None,
                payload=payload,
            )
        return raw_message

    async def send(raw_message: dict[str, Any]) -> None:
        text, payload = _summarize_websocket_asgi_message(raw_message)
        payload.update(connection_payload)
        asgi_type = raw_message.get("type")
        event_name = {
            "websocket.accept": "websocket_accept",
            "websocket.send": "websocket_send",
            "websocket.close": "websocket_close",
        }.get(str(asgi_type), "websocket_send")
        if event_name == "websocket_close":
            state["close_recorded"] = True
        _update_onebot_socket_status(
            state.get("robot_id"),
            connected=event_name != "websocket_close",
            event=event_name,
            direction="bridge_to_platform",
            payload=payload,
        )
        if _should_record_websocket_debug_event(event_name, payload):
            _record_websocket_debug_event(
                state.get("robot_id"),
                platform_id,
                direction="bridge_to_platform",
                event=event_name,
                message=preview_text(text) if text else None,
                payload=payload,
            )
        await websocket.send(raw_message)

    try:
        await nonebot_app(scope, receive, send)
    except Exception as exc:
        _record_websocket_debug_event(
            state.get("robot_id"),
            platform_id,
            direction="platform_to_bridge",
            event="websocket_error",
            status="error",
            message=str(exc),
            payload=connection_payload,
        )
        _update_onebot_socket_status(
            state.get("robot_id"),
            connected=False,
            event="websocket_error",
            direction="platform_to_bridge",
            payload=connection_payload,
        )
        raise
    finally:
        if not state.get("disconnect_recorded") and not state.get("close_recorded"):
            _update_onebot_socket_status(
                state.get("robot_id"),
                connected=False,
                event="websocket_closed",
                direction="platform_to_bridge",
                payload=connection_payload,
            )
            _record_websocket_debug_event(
                state.get("robot_id"),
                platform_id,
                direction="platform_to_bridge",
                event="websocket_closed",
                payload=connection_payload,
            )


def _build_ready_event_payload(
    platform_id: str,
    bot_identity: str,
    event: Any,
) -> tuple[str, dict[str, Any]]:
    event_payload = _serialize_event_payload(event)
    user = (
        event_payload.get("user") if isinstance(event_payload.get("user"), dict) else {}
    )
    payload = {
        "platform": platform_id,
        "bot_identity": bot_identity,
        "event_type": event.__class__.__name__,
        "session_id": event_payload.get("session_id"),
        "version": event_payload.get("version"),
        "shard": event_payload.get("shard"),
        "user_id": user.get("id"),
        "username": user.get("username"),
        "bot": user.get("bot"),
    }
    message = (
        f"QQ bot ready: {payload.get('username') or payload.get('user_id') or bot_identity}"
        f" session={payload.get('session_id') or '-'}"
    )
    return message, {key: value for key, value in payload.items() if value is not None}


def _save_identities() -> None:
    try:
        import json

        with open(_identity_file_path, "w") as f:
            json.dump(_identity_by_robot_id, f)
    except Exception:
        pass


def _load_identities() -> dict[str, str]:
    try:
        import json

        with open(_identity_file_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _loguru_sink(message):
    record = message.record
    level = record.get("level")
    level_name = level.name if level else None
    if level_name == "ERROR":
        text = record.get("message", "")
        if "robot bridge" in text.lower():
            exc_info = record.get("exception")
            identities = _load_identities()
            for robot_id in identities:
                exc_text = ""
                if exc_info:
                    exc_type = exc_info.type if hasattr(exc_info, "type") else None
                    exc_value = exc_info.value if hasattr(exc_info, "value") else None
                    if exc_type and exc_value:
                        exc_text = f"{exc_type.__name__}: {exc_value}"
                error_msg = exc_text if exc_text else text
                _save_connection_error(robot_id, error_msg)
                break


def _assert_bridge_permission(header_value: str | None) -> None:
    expected = settings.ROBOT_BRIDGE_SHARED_SECRET or settings.SECRET_KEY
    if not expected or header_value != expected:
        raise HTTPException(status_code=403, detail="Invalid robot bridge token")


def _record_bridge_event(
    robot_id: str,
    *,
    direction: str,
    event: str,
    status: str = "ok",
    message: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    _last_platform_event_at_by_robot_id[robot_id] = timestamp
    if event == "platform_message":
        _last_message_event_at_by_robot_id[robot_id] = timestamp
    record_robot_event(
        robot_id,
        direction=direction,
        event=event,
        status=status,
        message=message,
        payload=payload,
    )


async def _dispatch_to_backend(
    robot_id: str,
    payload: RobotInboundMessage,
) -> RobotDispatchResponse:
    shared_secret = settings.ROBOT_BRIDGE_SHARED_SECRET or settings.SECRET_KEY
    timeout = settings.ROBOT_BACKEND_DISPATCH_TIMEOUT_SECONDS
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.ROBOT_BACKEND_URL.rstrip('/')}"
                f"{settings.API_V1_STR}/robots/{robot_id}/dispatch",
                headers={
                    "X-TermPaws-Bridge-Token": shared_secret,
                    "Content-Type": "application/json",
                },
                content=payload.model_dump_json(),
            )
            response.raise_for_status()
            return RobotDispatchResponse.model_validate(response.json())
    except httpx.TimeoutException as exc:
        raise TimeoutError(
            f"Backend dispatch timed out after {timeout:g}s"
        ) from exc


def get_bridge_router() -> APIRouter | None:
    global _bridge_router
    return _bridge_router


def get_loaded_robots() -> tuple[list, dict[str, str], dict[str, str]]:
    return _loaded_robots, _robot_id_by_identity, _identity_by_robot_id


async def start_embedded_bridge() -> None:
    if _startup_callback is not None:
        await _startup_callback()


async def stop_embedded_bridge() -> None:
    if _shutdown_callback is not None:
        await _shutdown_callback()


def _build_idle_bridge_router(reason: str) -> APIRouter:
    router = APIRouter(prefix="/robot-bridge", tags=["robot-bridge"])

    @router.post("/internal/send")
    async def internal_send(
        body: RobotBridgeSendRequest,
        x_termpaws_bridge_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _assert_bridge_permission(x_termpaws_bridge_token)
        return await _forward_to_owner("/internal/send", body=body)

    @router.post("/internal/reload", response_model=RobotBridgeReloadResponse)
    async def internal_reload(
        x_termpaws_bridge_token: str | None = Header(default=None),
    ) -> RobotBridgeReloadResponse:
        _assert_bridge_permission(x_termpaws_bridge_token)
        try:
            result = await _forward_to_owner("/internal/reload")
        except HTTPException as exc:
            if exc.status_code != 503:
                raise
            robots, _, _, _ = _current_enabled_robot_config_signature()
            if robots:
                result = await hot_reload_bridge()
                return RobotBridgeReloadResponse(
                    success=bool(result.get("success")),
                    detail=str(result.get("detail") or ""),
                )
            return RobotBridgeReloadResponse(
                success=True,
                detail=f"Bridge is idle ({reason}); no enabled robots are configured",
            )
        return RobotBridgeReloadResponse.model_validate(result)

    @router.get("/internal/health")
    async def internal_health(
        x_termpaws_bridge_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        _assert_bridge_permission(x_termpaws_bridge_token)
        try:
            owner_health = await _forward_to_owner(
                "/internal/health", timeout=5.0, method="GET"
            )
            owner_health["proxy_worker"] = {
                "pid": os.getpid(),
                "status": "forwarded_to_owner",
                "reason": reason,
            }
            return owner_health
        except Exception as exc:
            checked_at = datetime.now(timezone.utc).isoformat()
            robots, _, identity_by_robot_id, _ = (
                _current_enabled_robot_config_signature()
            )
            return {
                "checked_at": checked_at,
                "live": True,
                "status": "idle",
                "reason": reason,
                "owner_error": str(exc),
                "loaded_robot_count": len(identity_by_robot_id),
                "connected_bot_count": 0,
                "platforms": sorted(
                    {
                        normalize_robot_platform_id(robot.platform or robot.protocol)
                        for robot in robots
                    }
                ),
                "connected_identities": [],
                "robots": {
                    robot_id: {
                        "identity": identity,
                        "connected": False,
                        "error": f"Bridge is idle ({reason})",
                    }
                    for robot_id, identity in identity_by_robot_id.items()
                },
                "backend": {"reachable": True},
                "connection_errors": _load_connection_errors(),
                "runtime": collect_backend_runtime_stats("robot").model_dump(mode="json"),
            }

    return router


def init_embedded_bridge() -> APIRouter | None:
    global \
        _bridge_router, \
        _loaded_robots, \
        _robot_id_by_identity, \
        _identity_by_robot_id, \
        _initialized
    global _startup_callback, _shutdown_callback
    global _loaded_robot_config_signature

    if not settings.ROBOT_PLUGIN_ENABLED:
        logger.info("[Bridge] Robot plugin disabled, skip bridge initialization")
        return None

    if _initialized:
        return _bridge_router

    _initialized = True

    if not _acquire_singleton_lock():
        logger.info(
            "[Bridge] Another backend process already owns the robot bridge lock; "
            "skip initializing NoneBot in this worker"
        )
        _bridge_router = _build_idle_bridge_router(
            "robot_bridge_lock_owned_by_another_process"
        )
        return _bridge_router

    try:
        import nonebot
    except ImportError:
        logger.info("[Bridge] nonebot not installed, skip bridge initialization")
        _bridge_router = _build_idle_bridge_router("nonebot_not_installed")
        return _bridge_router

    try:
        from nonebot import get_asgi, get_bots, on, on_message
        from nonebot.adapters import Bot, Event

        globals()["Bot"] = Bot
        globals()["Event"] = Event

        from app.plugins.robot.platforms import (
            build_nonebot_init_kwargs,
            register_nonebot_adapters,
        )

        (
            _loaded_robots,
            _robot_id_by_identity,
            _identity_by_robot_id,
            _loaded_robot_config_signature,
        ) = _current_enabled_robot_config_signature()
        _save_identities()

        if not _loaded_robots:
            logger.info("[Bridge] No enabled robots found, skip bridge initialization")
            _release_singleton_lock()
            _bridge_router = _build_idle_bridge_router("no_enabled_robots")
            return _bridge_router

        init_kwargs = build_nonebot_init_kwargs(_loaded_robots)
        init_kwargs.setdefault("driver", "~fastapi+~httpx+~websockets")
        nonebot.init(**init_kwargs)
        driver = nonebot.get_driver()
        register_nonebot_adapters(driver, _loaded_robots)

        event_probe = on(priority=1, block=False)
        bridge_handler = on_message(priority=10, block=False)

        def _resolve_bot_for_robot(robot_id: str) -> Bot | None:
            identity = _identity_by_robot_id.get(robot_id)
            if not identity:
                return None

            for bot in get_bots().values():
                try:
                    if resolve_bot_identity(bot) == identity:
                        return bot
                except Exception:
                    continue
            return None

        @event_probe.handle()
        async def probe_robot_event(bot: Bot, event: Event) -> None:
            loaded_robot_ids = list(_identity_by_robot_id.keys())
            platform_id = resolve_platform_from_bot(bot)
            if not platform_id:
                return
            try:
                bot_identity = resolve_bot_identity(bot)
            except Exception:
                return
            robot_id = _robot_id_by_identity.get(bot_identity)
            if robot_id:
                _seen_connected_robot_ids.add(robot_id)
                event_name = event.__class__.__name__
                if platform_id == "onebot_v11" and _is_onebot_heartbeat_event(event):
                    return
                if "ready" in event_name.lower():
                    message, payload = _build_ready_event_payload(
                        platform_id,
                        bot_identity,
                        event,
                    )
                    _record_bridge_event(
                        robot_id,
                        direction="platform_to_bridge",
                        event="bot_ready",
                        message=message,
                        payload=payload,
                    )
                    return
                _record_bridge_event(
                    robot_id,
                    direction="platform_to_bridge",
                    event="platform_event",
                    message=event_name,
                    payload={
                        "platform": platform_id,
                        "bot_identity": bot_identity,
                        "event_type": event_name,
                        "event": str(event),
                    },
                )
            elif loaded_robot_ids:
                record_loaded_robot_event(
                    loaded_robot_ids,
                    direction="platform_to_bridge",
                    event="platform_event_unmapped",
                    status="error",
                    message=event.__class__.__name__,
                    payload={
                        "platform": platform_id,
                        "bot_identity": bot_identity,
                    },
                )

        @bridge_handler.handle()
        async def handle_robot_message(bot: Bot, event: Event) -> None:
            loaded_robot_ids = list(_identity_by_robot_id.keys())
            event_summary = _summarize_platform_event(event)
            logger.info(
                "[Bridge] Received platform event: bot=%s event=%s peer=%s raw=%s",
                getattr(bot, "self_id", None),
                event.__class__.__name__,
                event_summary,
                preview_text(event),
            )
            platform_id = resolve_platform_from_bot(bot)
            if not platform_id:
                record_loaded_robot_event(
                    loaded_robot_ids,
                    direction="platform_to_bridge",
                    event="event_ignored",
                    status="error",
                    message="Unsupported bot adapter",
                    payload={
                        "bot_class": bot.__class__.__module__,
                        "event_summary": event_summary,
                    },
                )
                return

            try:
                bot_identity = resolve_bot_identity(bot)
            except Exception:
                record_loaded_robot_event(
                    loaded_robot_ids,
                    direction="platform_to_bridge",
                    event="event_ignored",
                    status="error",
                    message="Failed to resolve bot identity",
                    payload={
                        "platform": platform_id,
                        "event_summary": event_summary,
                    },
                )
                logger.exception("[Bridge] Failed to resolve bot identity")
                return

            robot_id = _robot_id_by_identity.get(bot_identity)
            if not robot_id:
                record_loaded_robot_event(
                    loaded_robot_ids,
                    direction="platform_to_bridge",
                    event="event_ignored",
                    status="error",
                    message=f"No TermPaws robot is mapped to identity {bot_identity}",
                    payload={
                        "platform": platform_id,
                        "bot_identity": bot_identity,
                        "known_identities": list(_robot_id_by_identity.keys()),
                        "event_summary": event_summary,
                    },
                )
                logger.warning(
                    "[Bridge] No TermPaws robot is mapped to identity %s",
                    bot_identity,
                )
                return

            _seen_connected_robot_ids.add(robot_id)
            if platform_id == "onebot_v11":
                event_payload = _serialize_event_payload(event)
                stale_payload = _stale_onebot_message_event_payload(event, event_payload)
                if stale_payload is not None:
                    _record_bridge_event(
                        robot_id,
                        direction="platform_to_bridge",
                        event="stale_message_ignored",
                        status="ignored",
                        message=event.__class__.__name__,
                        payload={
                            "platform": platform_id,
                            **stale_payload,
                        },
                    )
                    return
            inbound = build_inbound_message(platform_id, bot, event)
            if inbound is None:
                record_robot_event(
                    robot_id,
                    direction="platform_to_bridge",
                    event="event_ignored",
                    status="error",
                    message="Inbound message is empty or unsupported",
                    payload={
                        "event_type": event.__class__.__name__,
                        "event": str(event),
                        "event_summary": event_summary,
                    },
                )
                return
            _record_bridge_event(
                robot_id,
                direction="platform_to_bridge",
                event="platform_message",
                message=inbound.text,
                payload={
                    "platform": platform_id,
                    "sender_key": inbound.sender_key,
                    "target_type": inbound.reply_target.target_type,
                    "target_id": inbound.reply_target.target_id,
                    "reply_target": inbound.reply_target.model_dump(mode="json"),
                    "event_summary": event_summary,
                },
            )
            logger.info(
                "[Bridge] Platform message robot=%s sender=%s target=%s text=%s",
                robot_id,
                inbound.sender_key,
                inbound.reply_target.target_id,
                preview_text(inbound.text),
            )

            try:
                dispatch = await _dispatch_to_backend(robot_id, inbound)
            except Exception as exc:
                error_message = str(exc) or exc.__class__.__name__
                record_robot_event(
                    robot_id,
                    direction="bridge_to_backend",
                    event="dispatch_failed",
                    status="error",
                    message=error_message,
                )
                logger.exception(
                    "[Bridge] Failed to dispatch message for robot %s", robot_id
                )
                record_robot_event(
                    robot_id,
                    direction="bridge_to_platform",
                    event="dispatch_failure_notification_suppressed",
                    status="ignored",
                    message=error_message,
                    payload=_dispatch_failure_payload(inbound),
                )
                return

            if dispatch.ignored:
                return

            for chunk in dispatch.reply_chunks:
                logger.info(
                    "[Bridge] Sending platform reply robot=%s target=%s text=%s",
                    robot_id,
                    inbound.reply_target.target_id,
                    preview_text(chunk),
                )
                record_robot_event(
                    robot_id,
                    direction="bridge_to_platform",
                    event="platform_send",
                    message=chunk,
                    payload={
                        "target_type": inbound.reply_target.target_type,
                        "target_id": inbound.reply_target.target_id,
                    },
                )
                try:
                    await send_text_with_rate_limit(
                        bot,
                        inbound.reply_target,
                        chunk,
                        robot_id=robot_id,
                    )
                except Exception as send_exc:
                    record_robot_event(
                        robot_id,
                        direction="bridge_to_platform",
                        event="platform_send_failed",
                        status="error",
                        message=str(send_exc) or send_exc.__class__.__name__,
                        payload={
                            "target_type": inbound.reply_target.target_type,
                            "target_id": inbound.reply_target.target_id,
                        },
                    )
                    logger.exception(
                        "[Bridge] Failed to send platform reply for robot %s",
                        robot_id,
                    )

        nonebot_app = get_asgi()

        _bridge_router = APIRouter(prefix="/robot-bridge", tags=["robot-bridge"])

        @_bridge_router.websocket("/onebot/v11/ws")
        async def onebot_v11_reverse_ws(websocket: WebSocket) -> None:
            await _run_logged_onebot_reverse_ws(
                websocket=websocket,
                path="/onebot/v11/ws",
                nonebot_app=nonebot_app,
            )

        @_bridge_router.websocket("/onebot/v11/ws/")
        async def onebot_v11_reverse_ws_slash(websocket: WebSocket) -> None:
            await _run_logged_onebot_reverse_ws(
                websocket=websocket,
                path="/onebot/v11/ws/",
                nonebot_app=nonebot_app,
            )

        def _get_robot_id_for_adapter(adapter_name: str) -> str | None:
            adapter_lower = adapter_name.lower().replace(" ", "_")
            for robot_id, identity in _identity_by_robot_id.items():
                if identity.startswith(adapter_lower):
                    return robot_id
            return None

        async def startup_nonebot():
            global _connection_errors
            logger.info("[Bridge] Starting NoneBot adapters...")
            for adapter in driver._adapters.values():
                adapter_name = str(adapter.get_name())
                try:
                    hook_called = await _call_optional_adapter_hook(adapter, "startup")
                    if hook_called:
                        logger.info("[Bridge] Adapter %s started", adapter_name)
                    robot_id = _get_robot_id_for_adapter(adapter_name)
                    if robot_id and robot_id in _connection_errors:
                        del _connection_errors[robot_id]
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(
                        f"[Bridge] Failed to start adapter {adapter_name}: {error_msg}"
                    )
                    robot_id = _get_robot_id_for_adapter(adapter_name)
                    if robot_id:
                        _save_connection_error(robot_id, error_msg)
            _start_ipc_server(_send_from_owner, _health_from_owner, _reload_owner)

        async def shutdown_nonebot():
            logger.info("[Bridge] Stopping NoneBot adapters...")
            _stop_ipc_server()
            for adapter in driver._adapters.values():
                try:
                    await _call_optional_adapter_hook(adapter, "shutdown")
                except Exception as e:
                    logger.error(f"[Bridge] Failed to stop adapter: {e}")

        _startup_callback = startup_nonebot
        _shutdown_callback = shutdown_nonebot

        async def _send_from_owner(body: RobotBridgeSendRequest) -> dict[str, Any]:
            robot_id = str(body.robot_id)
            bot = _resolve_bot_for_robot(robot_id)
            if bot is None:
                raise HTTPException(
                    status_code=404, detail="Robot is not loaded in bridge"
                )

            record_robot_event(
                robot_id,
                direction="bridge_to_platform",
                event="internal_send",
                message=body.text,
                payload={
                    "target_type": body.target.target_type,
                    "target_id": body.target.target_id,
                },
            )
            await send_text_with_rate_limit(
                bot,
                body.target,
                body.text,
                robot_id=robot_id,
            )
            return {"success": True}

        @_bridge_router.post("/internal/send")
        async def internal_send(
            body: RobotBridgeSendRequest,
            x_termpaws_bridge_token: str | None = Header(default=None),
        ) -> dict[str, Any]:
            _assert_bridge_permission(x_termpaws_bridge_token)
            try:
                return await _send_from_owner(body)
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                return await _forward_to_owner("/internal/send", body=body)

        async def _call_api_from_owner(body: RobotBridgeCallApiRequest) -> dict[str, Any]:
            robot_id = str(body.robot_id)
            bot = _resolve_bot_for_robot(robot_id)
            if bot is None:
                raise HTTPException(
                    status_code=404, detail="Robot is not loaded in bridge"
                )
            action = str(body.action or "").strip()
            if not action or not action.replace("_", "").isalnum() or len(action) > 64:
                raise HTTPException(status_code=400, detail="Invalid action name")
            try:
                data = await bot.call_api(action, **(body.params or {}))
            except Exception as exc:
                return {"success": False, "data": None, "error": str(exc)}
            return {"success": True, "data": data, "error": None}

        @_bridge_router.post("/internal/call_api")
        async def internal_call_api(
            body: RobotBridgeCallApiRequest,
            x_termpaws_bridge_token: str | None = Header(default=None),
        ) -> dict[str, Any]:
            _assert_bridge_permission(x_termpaws_bridge_token)
            try:
                return await _call_api_from_owner(body)
            except HTTPException as exc:
                if exc.status_code != 404:
                    raise
                return await _forward_to_owner("/internal/call_api", body=body)

        async def _health_from_owner() -> dict[str, Any]:
            checked_at = datetime.now(timezone.utc).isoformat()
            _load_connection_errors()
            connected_bot_count = len(get_bots())
            bot_snapshots: list[dict[str, Any]] = []
            bot_snapshot_by_identity: dict[str, dict[str, Any]] = {}
            for bot in get_bots().values():
                snapshot = _serialize_bot_snapshot(bot)
                bot_snapshots.append(snapshot)
                try:
                    bot_snapshot_by_identity[resolve_bot_identity(bot)] = snapshot
                except Exception:
                    continue
            connected_identities = [
                identity
                for robot_id, identity in _identity_by_robot_id.items()
                if robot_id in _seen_connected_robot_ids
                or identity in bot_snapshot_by_identity
            ]

            robot_status: dict[str, dict[str, Any]] = {}
            for robot_id, identity in _identity_by_robot_id.items():
                robot_status[robot_id] = {
                    "identity": identity,
                    "connected": identity in connected_identities,
                    "bot": bot_snapshot_by_identity.get(identity),
                    "onebot_socket": _onebot_socket_status_by_robot_id.get(
                        robot_id,
                        {"connected": False},
                    ),
                    "last_platform_event_at": _last_platform_event_at_by_robot_id.get(
                        robot_id
                    ),
                    "last_message_event_at": _last_message_event_at_by_robot_id.get(
                        robot_id
                    ),
                    "error": (
                        None
                        if identity in connected_identities
                        else (_connection_errors.get(robot_id) or {}).get("message")
                    ),
                    "last_error": _connection_errors.get(robot_id),
                }

            backend_status: dict[str, Any] = {"reachable": False}
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(
                        f"{settings.ROBOT_BACKEND_URL.rstrip('/')}{settings.API_V1_STR}/utils/health-check/",
                    )
                    backend_status = {
                        "reachable": resp.status_code == 200,
                        "status_code": resp.status_code,
                    }
            except Exception as exc:
                backend_status["error"] = str(exc)

            return {
                "checked_at": checked_at,
                "live": True,
                "loaded_robot_count": len(_identity_by_robot_id),
                "connected_bot_count": connected_bot_count,
                "platforms": sorted(
                    {
                        normalize_robot_platform_id(robot.platform or robot.protocol)
                        for robot in _loaded_robots
                    }
                ),
                "connected_identities": connected_identities,
                "bots": bot_snapshots,
                "robots": robot_status,
                "backend": backend_status,
                "connection_errors": _connection_errors,
                "runtime": collect_backend_runtime_stats("robot").model_dump(mode="json"),
                "ipc_owner": {
                    "pid": os.getpid(),
                    "base_url": _ipc_base_url,
                },
            }

        async def _reload_owner() -> dict[str, Any]:
            _, _, _, current_signature = _current_enabled_robot_config_signature()
            if current_signature != _loaded_robot_config_signature:
                return await hot_reload_bridge()

            errors: list[str] = []
            called_hooks = 0
            skipped_hooks = 0
            for adapter in driver._adapters.values():
                adapter_name = str(adapter.get_name())
                try:
                    logger.info(
                        "[Bridge] Soft-reloading adapter %s: shutdown", adapter_name
                    )
                    if await _call_optional_adapter_hook(adapter, "shutdown"):
                        called_hooks += 1
                    else:
                        skipped_hooks += 1
                except Exception as exc:
                    error_msg = (
                        f"{adapter_name} shutdown failed: {type(exc).__name__}: {exc}"
                    )
                    logger.warning("[Bridge] %s", error_msg)
                    errors.append(error_msg)

            for adapter in driver._adapters.values():
                adapter_name = str(adapter.get_name())
                try:
                    logger.info(
                        "[Bridge] Soft-reloading adapter %s: startup", adapter_name
                    )
                    if await _call_optional_adapter_hook(adapter, "startup"):
                        called_hooks += 1
                    else:
                        skipped_hooks += 1
                except Exception as exc:
                    error_msg = (
                        f"{adapter_name} startup failed: {type(exc).__name__}: {exc}"
                    )
                    logger.error("[Bridge] %s", error_msg)
                    errors.append(error_msg)

            if errors:
                return {
                    "success": False,
                    "detail": "; ".join(errors),
                }
            return {
                "success": True,
                "detail": (
                    "Bridge adapter reload checked: "
                    f"{called_hooks} hook(s) called, {skipped_hooks} hook(s) skipped"
                ),
            }

        @_bridge_router.post(
            "/internal/reload", response_model=RobotBridgeReloadResponse
        )
        async def internal_reload(
            x_termpaws_bridge_token: str | None = Header(default=None),
        ) -> RobotBridgeReloadResponse:
            _assert_bridge_permission(x_termpaws_bridge_token)
            return RobotBridgeReloadResponse.model_validate(await _reload_owner())

        @_bridge_router.get("/internal/health")
        async def internal_health(
            x_termpaws_bridge_token: str | None = Header(default=None),
        ) -> dict[str, Any]:
            _assert_bridge_permission(x_termpaws_bridge_token)
            return await _health_from_owner()

        logger.info(
            "[Bridge] Embedded bridge initialized with %d robot(s), platforms: %s",
            len(_loaded_robots),
            sorted(
                {
                    normalize_robot_platform_id(r.platform or r.protocol)
                    for r in _loaded_robots
                }
            ),
        )
        return _bridge_router

    except Exception as e:
        logger.exception("[Bridge] Failed to initialize embedded bridge: %s", e)
        return None


def resolve_robot_identity(platform_id: str, robot) -> str:
    from app.plugins.robot.platforms import resolve_robot_identity as _resolve

    return _resolve(platform_id, robot)
