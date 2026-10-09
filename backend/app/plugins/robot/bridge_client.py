from __future__ import annotations

import logging
import uuid

import httpx

from app.core.config import settings

from .contracts import (
    RobotBridgeCallApiRequest,
    RobotBridgeReloadResponse,
    RobotBridgeSendRequest,
    RobotReplyTarget,
)
from .debug_log import record_robot_event
from .internal_trace import (
    compact_robot_visible_message_text,
    is_robot_internal_trace_text,
    normalize_robot_message_text,
    sanitize_robot_visible_text,
)
from .message_chunks import split_robot_message_for_target

logger = logging.getLogger(__name__)


class RobotBridgeClient:
    def __init__(self) -> None:
        self._base_url = settings.ROBOT_BRIDGE_URL.rstrip("/")
        self._shared_secret = settings.ROBOT_BRIDGE_SHARED_SECRET or settings.SECRET_KEY

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-TermPaws-Bridge-Token": self._shared_secret,
        }

    def _response_error_detail(self, response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text.strip()
        if isinstance(data, dict):
            detail = data.get("detail") or data.get("error") or data.get("message")
            if detail is not None:
                return str(detail)
        return str(data)

    def _friendly_error(self, exc: Exception, *, detail: str = "") -> str:
        message = str(exc)
        if detail:
            message = f"{message}: {detail}"
        if "Name or service not known" in message or "[Errno -2]" in message:
            return (
                f"{self._base_url}: {message}. Backend cannot resolve this "
                "ROBOT_BRIDGE_URL. Set ROBOT_BRIDGE_URL in the root .env to "
                "the bridge address reachable from the backend runtime."
            )
        return f"{self._base_url}: {message}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        **kwargs,
    ) -> httpx.Response:
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.request(
                    method,
                    f"{self._base_url}{path}",
                    headers=self._headers(),
                    **kwargs,
                )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    self._friendly_error(
                        exc,
                        detail=self._response_error_detail(response),
                    )
                ) from exc
            return response
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(self._friendly_error(exc)) from exc

    def get_health(self, timeout: float = 5.0) -> dict:
        response = self._request("GET", "/internal/health", timeout=timeout)
        data = response.json()
        if isinstance(data, dict):
            return data
        return {"data": data}

    def call_api(
        self,
        robot_id: uuid.UUID | str,
        action: str,
        params: dict | None = None,
        *,
        timeout: float = 15.0,
    ) -> dict:
        """Call a platform (onebot) action on the robot's live connection,
        e.g. get_group_list / get_friend_list. Returns {"success", "data",
        "error"}."""
        payload = RobotBridgeCallApiRequest(
            robot_id=uuid.UUID(str(robot_id)),
            action=action,
            params=params or {},
        )
        response = self._request(
            "POST", "/internal/call_api", timeout=timeout,
            json=payload.model_dump(mode="json"),
        )
        data = response.json()
        return data if isinstance(data, dict) else {"success": False, "data": None, "error": "bad bridge response"}

    def send_message(self, robot_id: uuid.UUID | str, target: RobotReplyTarget, text: str) -> None:
        robot_id_str = str(robot_id)
        original_text = normalize_robot_message_text(text)
        text = compact_robot_visible_message_text(sanitize_robot_visible_text(original_text))
        if not text or is_robot_internal_trace_text(text):
            record_robot_event(
                robot_id_str,
                direction="backend_to_bridge",
                event="send_message_blocked",
                status="ignored",
                message=original_text,
                payload={
                    "target_type": target.target_type,
                    "target_id": target.target_id,
                    "reason": "internal_robot_tool_trace",
                },
            )
            return
        if text != original_text.strip():
            record_robot_event(
                robot_id_str,
                direction="backend_to_bridge",
                event="send_message_sanitized",
                status="ok",
                message=text,
                payload={
                    "target_type": target.target_type,
                    "target_id": target.target_id,
                },
            )

        chunks = split_robot_message_for_target(target, text)
        if not chunks:
            return

        record_robot_event(
            robot_id_str,
            direction="backend_to_bridge",
            event="send_message",
            message=text,
            payload={
                "target_type": target.target_type,
                "target_id": target.target_id,
                "chunk_count": len(chunks),
            },
        )
        chunk_index = 0
        chunk_text = ""
        try:
            for chunk_index, chunk_text in enumerate(chunks, start=1):
                payload = RobotBridgeSendRequest(
                    robot_id=robot_id,
                    target=target,
                    text=chunk_text,
                )
                if len(chunks) > 1:
                    record_robot_event(
                        robot_id_str,
                        direction="backend_to_bridge",
                        event="send_message_chunk",
                        message=chunk_text,
                        payload={
                            "target_type": target.target_type,
                            "target_id": target.target_id,
                            "chunk_index": chunk_index,
                            "chunk_count": len(chunks),
                        },
                    )
                self._request(
                    "POST",
                    "/internal/send",
                    timeout=10.0,
                    content=payload.model_dump_json(),
                )
        except Exception as exc:
            record_robot_event(
                robot_id_str,
                direction="backend_to_bridge",
                event="send_message",
                status="error",
                message=str(exc),
                payload={
                    "target_type": target.target_type,
                    "target_id": target.target_id,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "chunk_text": chunk_text,
                },
            )
            raise

    def notify_reload(self) -> tuple[bool, str]:
        if not settings.ROBOT_BRIDGE_AUTO_RELOAD:
            return False, "Bridge auto reload is disabled"

        try:
            response = self._request("POST", "/internal/reload", timeout=5.0)
            result = RobotBridgeReloadResponse.model_validate(response.json())
            return result.success, result.detail
        except Exception as exc:
            logger.info("[RobotBridge] reload notification skipped: %s", exc)
            return False, str(exc)


robot_bridge_client = RobotBridgeClient()
