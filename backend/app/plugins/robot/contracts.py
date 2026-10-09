from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class RobotReplyTarget(BaseModel):
    target_type: str
    target_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def remaining_reply_budget(self) -> int | None:
        max_replies = self.metadata.get("reply_max_replies")
        used_replies = self.metadata.get("reply_used_replies")
        if max_replies is None:
            return None
        try:
            return max(0, int(max_replies) - int(used_replies or 0))
        except (TypeError, ValueError):
            return None


class RobotInboundMessage(BaseModel):
    sender_key: str
    text: str
    reply_target: RobotReplyTarget


class RobotDispatchResponse(BaseModel):
    success: bool
    ignored: bool = False
    reason: str | None = None
    error: str | None = None
    item_id: str | None = None
    route_key: str | None = None
    reply_chunks: list[str] = Field(default_factory=list)


class RobotBridgeSendRequest(BaseModel):
    robot_id: uuid.UUID
    target: RobotReplyTarget
    text: str


class RobotBridgeReloadResponse(BaseModel):
    success: bool
    detail: str


class RobotBridgeCallApiRequest(BaseModel):
    robot_id: uuid.UUID
    action: str
    params: dict = Field(default_factory=dict)


class RobotBridgeCallApiResponse(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None
