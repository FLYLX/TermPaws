import concurrent.futures
import logging
import threading
import uuid
import httpx
from datetime import datetime
from typing import Any, Callable, Dict

import socketio

from ..protocol import ProtocolEvents
from .connection_models import ConnectionStatus, DaemonConfig

logger = logging.getLogger(__name__)


class DaemonConnection:
    """
    鍗曚釜Daemon鑺傜偣鐨刉ebSocket杩炴帴灏佽
    
    浣跨敤WebSocket杩涜鎵€鏈夐€氫俊锛?
    - 鍙戦€佽姹傚苟绛夊緟鍝嶅簲
    - 鎺ユ敹浜嬩欢锛堝stdout锛?
    - 缁存姢杩炴帴鐘舵€?
    """
    def __init__(self, config: DaemonConfig):
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self._auth_completed = threading.Event()
        
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,  # 0=无限重连：daemon 重启/网络抖动后必须能自愈，
            # 之前 5 次重试耗尽后永久死亡，导致命令永远"超时"（其实执行了但没回声）
            reconnection_delay=2,
            reconnection_delay_max=10,
            logger=False,
            engineio_logger=False
        )
        
        self.callbacks: Dict[str, Callable] = {}
        self.pending_requests: Dict[str, concurrent.futures.Future] = {}
        self.last_heartbeat = None
        self._lock = threading.Lock()
        
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        @self.sio.event
        def connect():
            self.status = ConnectionStatus.CONNECTED
            self.last_heartbeat = datetime.now()
            logger.info(f"[WebSocket] Connected to {self.config.base_url}")
            
            if "connect" in self.callbacks:
                self.callbacks["connect"]()

        @self.sio.event
        def disconnect():
            self.status = ConnectionStatus.DISCONNECTED
            self._auth_completed.clear()
            logger.info(f"[WebSocket] Disconnected from {self.config.base_url}")
            
            with self._lock:
                for future in self.pending_requests.values():
                    if not future.done():
                        future.set_exception(Exception("Connection disconnected"))
                self.pending_requests.clear()
            
            if "disconnect" in self.callbacks:
                self.callbacks["disconnect"]()

        @self.sio.on("auth")
        def on_auth(data):
            if data.get("success"):
                logger.info(f"[WebSocket] Authenticated with daemon: {self.config.base_url}")
                self._auth_completed.set()
                self._sync_all_connections()
            else:
                logger.error(f"[WebSocket] Authentication failed: {data.get('message')}")

        @self.sio.on("terminal/start")
        def on_terminal_start(data):
            self._handle_response("terminal/start", data)

        @self.sio.on("terminal/stop")
        def on_terminal_stop(data):
            self._handle_response("terminal/stop", data)

        @self.sio.on("terminal/restart")
        def on_terminal_restart(data):
            self._handle_response("terminal/restart", data)

        @self.sio.on("terminal/status")
        def on_terminal_status(data):
            self._handle_response("terminal/status", data)

        @self.sio.on("terminal/list")
        def on_terminal_list(data):
            self._handle_response("terminal/list", data)

        @self.sio.on("connections/get")
        def on_connections_get(data):
            self._handle_response("connections/get", data)

        @self.sio.on("connections/get_all")
        def on_connections_get_all(data):
            self._handle_response("connections/get_all", data)

        @self.sio.on("connections/disconnect")
        def on_connections_disconnect(data):
            self._handle_response("connections/disconnect", data)

        @self.sio.on("item/subscribers")
        def on_item_subscribers(data):
            self._handle_response("item/subscribers", data)

        @self.sio.on("connection_update")
        def on_connection_update(data):
            if "connection_update" in self.callbacks:
                self.callbacks["connection_update"](data)

        @self.sio.on("stream")
        def on_stream(data):
            if ProtocolEvents.STREAM in self.callbacks:
                self.callbacks[ProtocolEvents.STREAM](data)

        @self.sio.on("terminal_connected")
        def on_terminal_connected(data):
            logger.info(f"[WebSocket] Terminal connected: {data.get('item_uuid')}")

        @self.sio.on("auth_error")
        def on_auth_error(data):
            logger.error(f"[WebSocket] Auth error: {data.get('message')}")

    def _handle_response(self, event: str, data: Dict[str, Any]):
        request_id = data.get("request_id")
        if request_id:
            with self._lock:
                future = self.pending_requests.pop(request_id, None)
            if future and not future.done():
                future.set_result(data)
        else:
            if event in self.callbacks:
                self.callbacks[event](data)

    def _generate_request_id(self) -> str:
        return str(uuid.uuid4())

    def _emit_and_wait_sync(self, event: str, data: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """鍚屾鍙戦€佽姹傚苟绛夊緟鍝嶅簲 - 浣跨敤绾跨▼瀹夊叏鐨?Future"""
        if self.status != ConnectionStatus.CONNECTED:
            return {"success": False, "error": "Not connected to daemon"}
        
        request_id = self._generate_request_id()
        data["request_id"] = request_id
        
        future = concurrent.futures.Future()
        with self._lock:
            self.pending_requests[request_id] = future
        
        try:
            self.sio.emit(event, data)
            result = future.result(timeout=timeout)
            return result
        except concurrent.futures.TimeoutError:
            with self._lock:
                self.pending_requests.pop(request_id, None)
            return {"success": False, "error": "Request timeout"}
        except Exception as e:
            with self._lock:
                self.pending_requests.pop(request_id, None)
            return {"success": False, "error": str(e)}

    def connect(self) -> bool:
        try:
            self._auth_completed.clear()
            self.sio.connect(
                self.config.base_url,
                transports=["websocket"],
                auth={"api_key": self.config.api_key}
            )
            
            self.sio.emit("auth", {
                "backend_id": f"{self.config.ip}:{self.config.api_key[:8]}"
            })
            
            if self._auth_completed.wait(timeout=10):
                logger.info(f"[WebSocket] Auth completed for {self.config.base_url}")
                return True
            else:
                logger.warning(f"[WebSocket] Auth timeout for {self.config.base_url}, but connection is established")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.config.base_url}: {str(e)}")
            self.status = ConnectionStatus.ERROR
            return False

    def disconnect(self):
        try:
            self.sio.disconnect()
        except Exception:
            pass
        self.status = ConnectionStatus.DISCONNECTED
        self._auth_completed.clear()

    def on(self, event: str, callback: Callable):
        self.callbacks[event] = callback

    def is_connected(self) -> bool:
        return self.status == ConnectionStatus.CONNECTED and self.sio.connected

    def get_status(self) -> ConnectionStatus:
        return self.status

    def terminal_start_http(self, user_uuid: str, item_uuid: str, working_directory: str = None, command: str = None, item_title: str = None) -> Dict[str, Any]:
        """
        鍚姩缁堢 - 鍚屾鏂规硶
        
        杩斿洖:
            success: 鏄惁鎴愬姛
            item_uuid: 缁堢UUID
            token: 璁块棶浠ょ墝
            message: 娑堟伅
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to daemon"}
        
        data = {
            "user_uuid": user_uuid,
            "item_uuid": item_uuid
        }
        if working_directory:
            data["working_directory"] = working_directory
        if command:
            data["command"] = command
        if item_title:
            data["item_title"] = item_title

        return self._emit_and_wait_sync("terminal/start", data)

    def terminal_stop_http(self, item_uuid: str) -> Dict[str, Any]:
        """
        鍋滄缁堢 - 鍚屾鏂规硶
        
        杩斿洖:
            success: 鏄惁鎴愬姛
            item_uuid: 缁堢UUID
            message: 娑堟伅
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to daemon"}
        return self._emit_and_wait_sync("terminal/stop", {"item_uuid": item_uuid})

    def terminal_restart_http(self, item_uuid: str, user_uuid: str = None, working_directory: str = None, command: str = None) -> Dict[str, Any]:
        """
        閲嶅惎缁堢 - 鍚屾鏂规硶
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to daemon"}
        
        data = {"item_uuid": item_uuid}
        if user_uuid:
            data["user_uuid"] = user_uuid
        if working_directory:
            data["working_directory"] = working_directory
        if command:
            data["command"] = command
        
        return self._emit_and_wait_sync("terminal/restart", data)

    def terminal_status_http(
        self,
        item_uuid: str,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        鏌ヨ缁堢鐘舵€?- 鍚屾鏂规硶
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to daemon"}
        return self._emit_and_wait_sync(
            "terminal/status",
            {"item_uuid": item_uuid},
            timeout=timeout,
        )

    def terminal_list_http(self) -> Dict[str, Any]:
        """
        鑾峰彇缁堢鍒楄〃 - 鍚屾鏂规硶
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to daemon"}
        return self._emit_and_wait_sync("terminal/list", {})

    def get_connections_http(self, item_uuid: str) -> Dict[str, Any]:
        """
        鑾峰彇鎸囧畾item鐨勮繛鎺ユ睜 - 鍚屾鏂规硶
        
        杩斿洖:
            success: 鏄惁鎴愬姛
            item_uuid: 缁堢UUID
            connections: 杩炴帴姹犺〃 {sid: {user_uuid, ip}}
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to daemon"}
        return self._emit_and_wait_sync("connections/get", {"item_uuid": item_uuid})

    def get_all_connections_http(self) -> Dict[str, Any]:
        """
        鑾峰彇鎵€鏈夎繛鎺ユ睜 - 鍚屾鏂规硶
        
        杩斿洖:
            success: 鏄惁鎴愬姛
            connections: 鎵€鏈夎繛鎺ユ睜琛?
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to daemon"}
        return self._emit_and_wait_sync("connections/get_all", {})

    def disconnect_connection_http(self, item_uuid: str, user_uuid: str = None, ip_address: str = None) -> Dict[str, Any]:
        """
        鏂紑鎸囧畾杩炴帴 - 鍚屾鏂规硶
        
        杩斿洖:
            success: 鏄惁鎴愬姛
            item_uuid: 缁堢UUID
            message: 娑堟伅
            connections: 鏇存柊鍚庣殑杩炴帴姹犺〃
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to daemon"}
        
        data = {"item_uuid": item_uuid}
        if user_uuid:
            data["user_uuid"] = user_uuid
        if ip_address:
            data["ip_address"] = ip_address
        
        return self._emit_and_wait_sync("connections/disconnect", data)

    def run_job_http(
        self,
        *,
        item_uuid: str,
        user_uuid: str,
        command: str,
        working_directory: str | None = None,
        timeout_seconds: int = 600,
        tail_lines: int = 80,
        env: dict[str, str] | None = None,
        item_title: str | None = None,
    ) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/internal/items/{item_uuid}/jobs/run"
        payload: Dict[str, Any] = {
            "user_uuid": user_uuid,
            "command": command,
            "working_directory": working_directory,
            "timeout_seconds": timeout_seconds,
            "tail_lines": tail_lines,
            "env": env or {},
            "item_title": item_title,
        }
        headers = {"X-API-Key": self.config.api_key}
        # Async start returns immediately; no need to wait out the job.
        request_timeout = 30.0
        logger.info(
            "[DaemonConnection] Running daemon job over HTTP: url=%s item=%s timeout=%s command=%r",
            url,
            item_uuid,
            timeout_seconds,
            command,
        )
        try:
            with httpx.Client(timeout=request_timeout) as client:
                response = client.post(url, json=payload, headers=headers)
            try:
                result = response.json()
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid daemon job response: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "body": response.text[:1000],
                }
            if response.status_code >= 400:
                detail = result.get("detail") if isinstance(result, dict) else None
                return {
                    "success": False,
                    "error": detail or f"Daemon job request failed: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "response": result,
                }
            return result
        except httpx.TimeoutException:
            logger.error("[DaemonConnection] Daemon job request timed out: item=%s command=%r", item_uuid, command)
            return {"success": False, "error": "Daemon job request timed out"}
        except httpx.HTTPError as exc:
            logger.error("[DaemonConnection] Daemon job request failed: item=%s error=%s", item_uuid, exc)
            return {"success": False, "error": f"Daemon job request failed: {exc}"}

    def list_jobs_http(self, *, item_uuid: str) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/internal/items/{item_uuid}/jobs"
        headers = {"X-API-Key": self.config.api_key}
        logger.debug(
            "[DaemonConnection] Listing daemon jobs over HTTP: url=%s item=%s",
            url,
            item_uuid,
        )
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, headers=headers)
            try:
                result = response.json()
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid daemon job list response: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "body": response.text[:1000],
                }
            if response.status_code >= 400:
                detail = result.get("detail") if isinstance(result, dict) else None
                return {
                    "success": False,
                    "error": detail or f"Daemon job list failed: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "response": result,
                }
            return result
        except httpx.TimeoutException:
            logger.error("[DaemonConnection] Daemon job list timed out: item=%s", item_uuid)
            return {"success": False, "error": "Daemon job list timed out"}
        except httpx.HTTPError as exc:
            logger.error("[DaemonConnection] Daemon job list failed: item=%s error=%s", item_uuid, exc)
            return {"success": False, "error": str(exc)}

    def get_job_result_http(
        self,
        *,
        item_uuid: str,
        job_id: str,
    ) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/internal/items/{item_uuid}/jobs/result"
        payload: Dict[str, Any] = {"job_id": job_id}
        headers = {"X-API-Key": self.config.api_key}
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload, headers=headers)
            try:
                result = response.json()
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid daemon job result response: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "body": response.text[:1000],
                }
            if response.status_code >= 400:
                detail = result.get("detail") if isinstance(result, dict) else None
                return {
                    "success": False,
                    "error": detail or f"Daemon job result request failed: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "response": result,
                }
            return result
        except httpx.TimeoutException:
            return {"success": False, "error": "Daemon job result request timed out"}
        except httpx.HTTPError as exc:
            return {"success": False, "error": f"Daemon job result request failed: {exc}"}

    def cancel_job_http(
        self,
        *,
        item_uuid: str,
        job_id: str | None = None,
    ) -> Dict[str, Any]:
        url = f"{self.config.base_url}/api/internal/items/{item_uuid}/jobs/cancel"
        payload: Dict[str, Any] = {"job_id": job_id}
        headers = {"X-API-Key": self.config.api_key}
        logger.info(
            "[DaemonConnection] Cancelling daemon job over HTTP: url=%s item=%s job_id=%s",
            url,
            item_uuid,
            job_id,
        )
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(url, json=payload, headers=headers)
            try:
                result = response.json()
            except ValueError:
                return {
                    "success": False,
                    "error": f"Invalid daemon job cancel response: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "body": response.text[:1000],
                }
            if response.status_code >= 400:
                detail = result.get("detail") if isinstance(result, dict) else None
                return {
                    "success": False,
                    "error": detail or f"Daemon job cancel failed: HTTP {response.status_code}",
                    "status_code": response.status_code,
                    "response": result,
                }
            return result
        except httpx.TimeoutException:
            logger.error("[DaemonConnection] Daemon job cancel timed out: item=%s job_id=%s", item_uuid, job_id)
            return {"success": False, "error": "Daemon job cancel timed out"}
        except httpx.HTTPError as exc:
            logger.error("[DaemonConnection] Daemon job cancel failed: item=%s error=%s", item_uuid, exc)
            return {"success": False, "error": str(exc)}

    def get_item_subscribers_http(
        self,
        item_uuid: str,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """
        鑾峰彇item鐨勮闃呰€呬俊鎭?- 鍚屾鏂规硶
        
        杩斿洖:
            success: 鏄惁鎴愬姛
            item_uuid: 缁堢UUID
            subscribers: 璁㈤槄鑰呭垪琛?[{sid, user_uuid, ip, type, join_time, last_active_time}]
            browser_count: 娴忚鍣ㄨ繛鎺ユ暟
            backend_connected: Backend鏄惁杩炴帴
            room_info: Room淇℃伅
        """
        if not self.is_connected():
            return {"success": False, "error": "Not connected to daemon"}
        return self._emit_and_wait_sync(
            "item/subscribers",
            {"item_uuid": item_uuid},
            timeout=timeout,
        )

    def _sync_all_connections(self):
        def run_sync():
            try:
                result = self._emit_and_wait_sync("connections/get_all", {}, timeout=10.0)
                if result.get("success"):
                    rooms = result.get("connections", {})
                    
                    logger.info("\n%s", "@" * 80)
                    logger.info("[DaemonConnection] Received full connection snapshot from daemon")
                    logger.info("%s", "@" * 80)
                    logger.info("  Source: %s", self.config.base_url)
                    logger.info("  Item count: %s", len(rooms))

                    if rooms:
                        logger.info("  Room details:")
                        for item_uuid, item_conns in rooms.items():
                            logger.info("    Item: %s", item_uuid)
                            if item_conns:
                                for sid, conn_info in item_conns.items():
                                    logger.info(
                                        "      - SID: %s... | User: %s | IP: %s",
                                        sid[:16],
                                        conn_info.get("user_uuid", "unknown"),
                                        conn_info.get("ip", "unknown"),
                                    )
                            else:
                                logger.info("      - No active connections")

                    if "connection_update" in self.callbacks:
                        self.callbacks["connection_update"]({
                            "type": "full_sync",
                            "rooms": rooms
                        })
                        logger.info("  [OK] Triggered connection_update callback")

                    logger.info("%s\n", "@" * 80)
            except Exception as e:
                logger.error(f"[WebSocket] Failed to sync all connections: {str(e)}")
        
        thread = threading.Thread(target=run_sync, daemon=True)
        thread.start()

    def emit(self, event: str, data: Any) -> bool:
        if self.status != ConnectionStatus.CONNECTED:
            return False
        try:
            self.sio.emit(event, data)
            return True
        except Exception:
            return False

