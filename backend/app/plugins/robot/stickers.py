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
    """存一张表情包，返回路径。tag 非法或内容为空返回 None。

    内容去重：同一张图（字节级相同）在库里已有就跳过，不再重复入库
    （同一张梗图被反复转发/重发很常见）。
    """
    tag = normalize_tag(tag)
    if not tag or not payload:
        return None
    existing = _find_duplicate(payload)
    if existing is not None:
        logger.info(
            "[Stickers] duplicate skipped: tag=%s already at %s",
            tag,
            existing.parent.name,
        )
        return None
    ext = ext if ext.startswith(".") else f".{ext}"
    directory = _tag_dir(tag)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid.uuid4().hex}{ext}"
    path.write_bytes(payload)
    logger.info("[Stickers] saved tag=%s file=%s (%d bytes)", tag, path.name, len(payload))
    return path


def _find_duplicate(payload: bytes) -> Path | None:
    """全库字节级查重，返回已有文件路径。库存量级小，逐文件比对即可。"""
    size = len(payload)
    for directory in _store_dir().iterdir():
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            try:
                if (
                    path.is_file()
                    and path.stat().st_size == size
                    and path.read_bytes() == payload
                ):
                    return path
            except OSError:
                continue
    return None


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


def list_stickers() -> list[dict]:
    """前端表情库面板用：标签 + 文件名列表。"""
    result: list[dict] = []
    for directory in sorted(_store_dir().iterdir()):
        if directory.is_dir():
            files = sorted(p.name for p in directory.iterdir() if p.is_file())
            if files:
                result.append({"tag": directory.name, "files": files})
    return result


def sticker_file_path(tag: str, name: str) -> Path | None:
    """标签+文件名 -> 本地文件；非法成分（穿越/隐藏）返回 None。"""
    tag_dir_name = normalize_tag(tag)
    clean_name = Path(str(name or "")).name
    if not tag_dir_name or not clean_name or clean_name.startswith("."):
        return None
    path = _tag_dir(tag_dir_name) / clean_name
    return path if path.exists() and path.is_file() else None


def delete_sticker(tag: str, name: str) -> bool:
    """删除一张表情包；标签目录空了顺带删目录。"""
    path = sticker_file_path(tag, name)
    if path is None:
        return False
    path.unlink()
    parent = path.parent
    try:
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass
    logger.info("[Stickers] deleted %s/%s", parent.name, name)
    return True
