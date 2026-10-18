"""QQ 表情包库：按 tag 分目录落盘，随机取用。

入站图片经 sticker_tagger 打标签后存到这里；出站时 send_message 带
sticker_tag 随机抽一张。与 image_store 的 TTL 缓存不同，表情包库是
永久资产，不自动清理。
"""
from __future__ import annotations

import logging
import os
import random
import re
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_TAG_LENGTH = 32
_INVALID_TAG_CHARS = re.compile(r"[^0-9A-Za-z一-鿿_-]+")


def normalize_tag(tag: str) -> str:
    """标签归一化：去空白/非法字符，小写拉丁。空或全非法返回空串。"""
    cleaned = _INVALID_TAG_CHARS.sub("", str(tag or "").strip()).lower()
    return cleaned[:MAX_TAG_LENGTH]


def _store_dir() -> Path:
    home = os.environ.get("TERMPAWS_HOME", str(Path.home() / ".termpaws"))
    path = Path(home) / "stickers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tag_dir(tag: str) -> Path:
    return _store_dir() / tag


def save_sticker(tag: str, payload: bytes, ext: str) -> Path | None:
    """存一张表情包，返回路径。tag 非法或内容为空返回 None。"""
    tag = normalize_tag(tag)
    if not tag or not payload:
        return None
    ext = ext if ext.startswith(".") else f".{ext}"
    directory = _tag_dir(tag)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(payload)
    logger.info("[Stickers] saved tag=%s file=%s (%d bytes)", tag, path.name, len(payload))
    return path


def random_sticker(tag: str) -> Path | None:
    """按 tag 随机抽一张。精确目录优先，没有则退化到包含该串的目录。"""
    tag = normalize_tag(tag)
    if not tag:
        return None
    exact = _tag_dir(tag)
    candidates = sorted(exact.glob("*")) if exact.is_dir() else []
    if not candidates:
        candidates = [
            p
            for directory in _store_dir().iterdir()
            if directory.is_dir() and tag in directory.name
            for p in directory.iterdir()
            if p.is_file()
        ]
    candidates = [p for p in candidates if p.is_file()]
    return random.choice(candidates) if candidates else None


def list_tags() -> dict[str, int]:
    """全部标签及图数。"""
    result: dict[str, int] = {}
    for directory in sorted(_store_dir().iterdir()):
        if directory.is_dir():
            result[directory.name] = sum(1 for p in directory.iterdir() if p.is_file())
    return result


def sticker_count() -> int:
    return sum(list_tags().values())
