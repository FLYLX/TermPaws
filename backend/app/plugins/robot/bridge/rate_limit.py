from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from app.plugins.robot.contracts import RobotReplyTarget
from app.plugins.robot.debug_log import preview_text, record_robot_event
from app.plugins.robot.platforms import (
    resolve_bot_identity,
    send_text_with_bot,
)

logger = logging.getLogger(__name__)

DEFAULT_SEND_INTERVAL_SECONDS = 0.8
RATE_LIMIT_RETRY_DELAYS = (3.0, 8.0)


@dataclass
class _SendBucket:
    lock: asyncio.Lock
    next_at: float = 0.0


_buckets: dict[str, _SendBucket] = {}


def _target_key(target: RobotReplyTarget) -> str:
    target_data = target.metadata.get("target")
    if isinstance(target_data, dict):
        target_id = target_data.get("id") or target_data.get("parent_id") or target.target_id
        parent_id = target_data.get("parent_id") or ""
        return f"universal:{parent_id}:{target_id}"
    return f"{target.target_type}:{target.target_id}"


def _send_key(bot: Any, target: RobotReplyTarget) -> str:
    try:
        bot_identity = resolve_bot_identity(bot)
    except Exception:
        bot_identity = str(getattr(bot, "self_id", "unknown"))
    return f"{bot_identity}:{_target_key(target)}"


def _is_platform_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return (
        "100017" in text
        or "22009" in text
        or "msg limit exceed" in text.lower()
        or "rate limit" in text.lower()
    )


def _send_interval_for_bot(_bot: Any) -> float:
    return DEFAULT_SEND_INTERVAL_SECONDS


async def send_text_with_rate_limit(
    bot: Any,
    target: RobotReplyTarget,
    text: str,
    *,
    robot_id: str | None = None,
    image_b64: str = "",
) -> None:
    normalized_text = (text or "").strip()
    if not normalized_text and not image_b64:
        return

    key = _send_key(bot, target)
    bucket = _buckets.setdefault(key, _SendBucket(lock=asyncio.Lock()))

    async with bucket.lock:
        now = time.monotonic()
        if bucket.next_at > now:
            await asyncio.sleep(bucket.next_at - now)

        interval = _send_interval_for_bot(bot)
        attempts = 1 + len(RATE_LIMIT_RETRY_DELAYS)
        for attempt in range(attempts):
            try:
                await send_text_with_bot(bot, target, normalized_text, image_b64=image_b64)
                logger.info(
                    "[Bridge] Platform send success target=%s:%s text=%s",
                    target.target_type,
                    target.target_id,
                    preview_text(normalized_text),
                )
                if robot_id:
                    record_robot_event(
                        robot_id,
                        direction="bridge_to_platform",
                        event="platform_send_success",
                        message=normalized_text,
                        payload={
                            "target_type": target.target_type,
                            "target_id": target.target_id,
                        },
                    )
                bucket.next_at = time.monotonic() + interval
                return
            except Exception as exc:
                if not _is_platform_rate_limit(exc) or attempt >= len(RATE_LIMIT_RETRY_DELAYS):
                    if robot_id:
                        record_robot_event(
                            robot_id,
                            direction="bridge_to_platform",
                            event="platform_send_failed",
                            status="error",
                            message=str(exc),
                            payload={
                                "target_type": target.target_type,
                                "target_id": target.target_id,
                            },
                        )
                    raise

                delay = RATE_LIMIT_RETRY_DELAYS[attempt]
                bucket.next_at = time.monotonic() + delay
                if robot_id:
                    record_robot_event(
                        robot_id,
                        direction="bridge_to_platform",
                        event="platform_rate_limited",
                        status="error",
                        message=str(exc),
                        payload={
                            "retry_after_seconds": delay,
                            "target_type": target.target_type,
                            "target_id": target.target_id,
                        },
                    )
                logger.warning(
                    "[Bridge] Platform send rate limited; retrying in %.1fs: %s",
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
