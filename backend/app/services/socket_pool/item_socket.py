import logging
import threading
from typing import Any

import socketio

from ..protocol import ProtocolEvents
from .socket_models import TerminalStatus
from .subscription_center import subscription_center

logger = logging.getLogger(__name__)


class ItemSocket:
    """
    代表单个终端的Socket连接 - 按 UPDATE.MD 规范

    支持两种订阅者类型：
    1. backend: 永久订阅者，监听输出写日志
    2. browser: 临时订阅者，实时渲染终端

    事件流：
    ItemSocket 接收事件 -> 发布到 SubscriptionCenter -> 订阅者收到通知
    """
    def __init__(self, item_uuid: str, token: str, daemon_url: str, user_uuid: str, subscriber_type: str = "browser"):
        self.item_uuid = item_uuid
        self.token = token
        self.daemon_url = daemon_url
        self.user_uuid = user_uuid
        self.subscriber_type = subscriber_type
        self.status = TerminalStatus.STOPPED
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=1,
            reconnection_delay_max=5,
        )
        self._connected_event = threading.Event()
        self._connect_error = None
        self.setup_event_handlers()

    def setup_event_handlers(self):
        @self.sio.event
        def connect():
            logger.info(f"[ItemSocket] Socket connect event for item={self.item_uuid}")
            self._emit_terminal_connect()

        @self.sio.event
        def disconnect():
            self.status = TerminalStatus.STOPPED
            self._connected_event.clear()
            subscription_center.publish_disconnected(self.item_uuid, {"item_uuid": self.item_uuid})

        @self.sio.event
        def terminal_connected(data):
            logger.info(f"[ItemSocket] Terminal connected: {data}")
            self.status = TerminalStatus.RUNNING
            self._connected_event.set()
            subscription_center.publish_connected(self.item_uuid, data)

        @self.sio.event
        def auth_error(data):
            logger.error(f"[ItemSocket] Auth error: {data}")
            self._connect_error = data
            self._connected_event.set()
            subscription_center.publish_auth_error(self.item_uuid, data)
            threading.Thread(target=self._recover_auth, daemon=True).start()

        @self.sio.on(ProtocolEvents.STREAM)
        def on_stream(data):
            stdout = data.get("stdout", "")
            stderr = data.get("stderr", "")
            stdin = data.get("stdin", "")

            if stdout:
                logger.info(f"[ItemSocket] Received stdout for item={self.item_uuid}: {len(stdout)} chars")
            if stderr:
                logger.debug(f"[ItemSocket] Received stderr for item={self.item_uuid}: {len(stderr)} chars")
            if stdin:
                logger.debug(f"[ItemSocket] Received stdin for item={self.item_uuid}: {len(stdin)} chars")

            subscription_center.publish_stream(self.item_uuid, data)

    def connect(self, api_key: str, timeout: float = 10.0) -> bool:
        try:
            self._connected_event.clear()
            self._connect_error = None
            self.status = TerminalStatus.STARTING

            logger.info(f"[ItemSocket] Connecting to {self.daemon_url} for item {self.item_uuid}, user {self.user_uuid}, type {self.subscriber_type}")

            self.sio.connect(
                self.daemon_url,
                transports=["websocket"],
                auth={"api_key": api_key},
                wait=True,
                wait_timeout=timeout
            )

            logger.info("[ItemSocket] Socket connected, waiting for terminal_connect confirmation")

            logger.info(f"[ItemSocket] Waiting for terminal_connected event (timeout={timeout}s)")

            if self._connected_event.wait(timeout=timeout):
                if self._connect_error:
                    logger.error(f"[ItemSocket] Connection rejected: {self._connect_error}")
                    self.status = TerminalStatus.ERROR
                    return False
                logger.info(f"[ItemSocket] Connection confirmed for item={self.item_uuid}")
                return True
            else:
                logger.error("[ItemSocket] Connection timeout waiting for terminal_connected")
                self.status = TerminalStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"[ItemSocket] Failed to connect: {str(e)}")
            self.status = TerminalStatus.ERROR
            return False

    def _emit_terminal_connect(self) -> None:
        try:
            self.sio.emit(ProtocolEvents.TERMINAL_CONNECT, {
                "item_uuid": self.item_uuid,
                "token": self.token,
                "user_uuid": self.user_uuid,
                "subscriber_type": self.subscriber_type
            })
        except Exception as exc:
            logger.error(f"[ItemSocket] Failed to emit terminal_connect: {exc}")

    def _recover_auth(self) -> None:
        """token 失效（daemon 重启清空内存 token 库）后的自愈：
        走标准 restore 流程——从 daemon 拿当前 token 并重建订阅；终端死了就幂等重启。"""
        try:
            import uuid as _uuid

            from sqlmodel import Session

            from app.core.db import engine
            from app.models import Item
            from app.services import DaemonConfig, connection_manager, socket_pool_facade
            from app.services.terminal_service import TerminalService

            with Session(engine) as db:
                item = db.get(Item, _uuid.UUID(str(self.item_uuid)))
            if not item or not item.socket_host or not item.socket_port or not item.api_key:
                return
            daemon_config = DaemonConfig(item.socket_host, item.socket_port, item.api_key)
            service = TerminalService(connection_manager, socket_pool_facade)
            restored = service.restore_running_terminal(
                item_uuid=self.item_uuid,
                owner_uuid=str(item.owner_id),
                daemon_config=daemon_config,
            )
            if restored:
                logger.info(f"[ItemSocket] auth recovered via restore for item={self.item_uuid}")
                return

            started = service.start_terminal(self.item_uuid, str(item.owner_id), daemon_config)
            if started.get("success"):
                logger.info(f"[ItemSocket] auth recovered via terminal restart for item={self.item_uuid}")
            else:
                logger.warning(f"[ItemSocket] auth recovery start failed for item={self.item_uuid}: {started.get('error')}")
        except Exception as exc:
            logger.warning(f"[ItemSocket] auth recovery failed for item={self.item_uuid}: {exc}")

    def disconnect(self):
        try:
            self.sio.disconnect()
        except Exception:
            pass
        self.status = TerminalStatus.STOPPED

    def emit(self, event: str, data: Any) -> bool:
        if self.status != TerminalStatus.RUNNING:
            logger.warning(f"[ItemSocket] emit failed: status={self.status}, not RUNNING")
            return False
        try:
            logger.info(f"[ItemSocket] emit event={event} to item={self.item_uuid}")
            self.sio.emit(event, data)
            return True
        except Exception as e:
            logger.error(f"[ItemSocket] emit error: {e}")
            return False

    def write(self, command: str, *, source: str = "backend") -> bool:
        logger.info(f"[ItemSocket] write() called: item={self.item_uuid}, command={command[:50]}...")
        sent = self.emit(ProtocolEvents.WRITE, {
            "command": command
        })
        if sent:
            subscription_center.publish_stream(
                self.item_uuid,
                {"stdin": command, "stdout": "", "stderr": "", "source": source},
            )
        return sent

    def is_connected(self) -> bool:
        return self.status == TerminalStatus.RUNNING

    def get_status(self) -> TerminalStatus:
        return self.status
