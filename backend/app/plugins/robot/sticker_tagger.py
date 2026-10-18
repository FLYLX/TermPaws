"""入站图片的表情包自动打标：后台线程跑视觉模型，是表情包就收进库里。

打标只依赖 handler 的模型配置（图识开关开的才跑），与对话轮次解耦——
机器人没被唤醒消息也会收图。判断词只有"是不是表情包/什么情绪"，
不打标签的内容（普通照片、截图）直接丢弃不入库。
"""
from __future__ import annotations

import base64
import logging
import threading
import uuid

from .image_store import image_file_path
from .stickers import save_sticker

logger = logging.getLogger(__name__)

_TAGGER_SEMAPHORE = threading.BoundedSemaphore(2)
_TAG_PROMPT = (
    "判断这张图片是不是表情包/梗图/ emoticon 贴纸。"
    "如果是，只输出一个 1~4 字的中文情绪或主题标签（如：开心、无语、嘲讽、卖萌、震惊、生气）。"
    "如果不是（普通照片、聊天截图、文件图片等），只输出 none。"
    "不要输出任何其它内容。"
)


def _is_meme_response(text: str) -> str:
    """解析打标结果：返回合法 tag 或空串。"""
    from .stickers import normalize_tag

    cleaned = str(text or "").strip()
    for ch in ("。", ".", ",", "，", "!", "！", "\"", "'"):
        cleaned = cleaned.strip(ch)
    cleaned = cleaned.strip()
    if not cleaned or cleaned.lower() in {"none", "no", "不是", "无"}:
        return ""
    tag = normalize_tag(cleaned)
    return tag if len(cleaned) <= 12 else ""


def _tag_one_image(handler_model: str | None, api_key: str | None,
                   api_url: str | None, model_parameters: dict | None,
                   image_name: str, item_id: str) -> None:
    path = image_file_path(image_name)
    if path is None:
        return
    try:
        import litellm

        from app.services.llm_completion import build_litellm_completion_kwargs

        payload = path.read_bytes()
        b64 = base64.b64encode(payload).decode()
        suffix = path.suffix.lstrip(".") or "jpeg"
        kwargs = build_litellm_completion_kwargs(
            model=handler_model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _TAG_PROMPT},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/{suffix};base64,{b64}"}},
                ],
            }],
            api_key=api_key,
            api_base=api_url,
            timeout=60,
            model_parameters=model_parameters,
        )
        response = litellm.completion(**kwargs)
        answer = response.choices[0].message.content or ""
        tag = _is_meme_response(answer)
        if tag:
            saved = save_sticker(tag, payload, path.suffix)
            if saved:
                logger.info(
                    "[StickerTagger] item=%s image=%s -> tag=%s",
                    item_id, image_name, tag,
                )
    except Exception as exc:
        logger.info("[StickerTagger] tag failed for %s: %s", image_name, exc)


def maybe_tag_inbound_images(item_id: str, image_names: list[str]) -> None:
    """入站图片后台打标入口。静默失败，绝不阻塞消息链路。"""
    if not image_names:
        return

    def _worker() -> None:
        try:
            from sqlmodel import Session

            from app.core.db import engine
            from app.models import ItemHandler, ItemHandlerItem

            with Session(engine) as db:
                handler_item = db.exec(
                    __import__("sqlmodel").select(ItemHandlerItem).where(
                        ItemHandlerItem.item_id == item_id
                    )
                ).first()
                handler = (
                    db.get(ItemHandler, handler_item.item_handler_id)
                    if handler_item
                    else None
                )
            model = getattr(handler, "model", None)
            api_key = getattr(handler, "api_key", None)
            api_url = getattr(handler, "api_url", None)
            model_parameters = getattr(handler, "model_parameters", None) or {}
            if not model:
                return
            for name in image_names:
                with _TAGGER_SEMAPHORE:
                    _tag_one_image(model, api_key, api_url, model_parameters,
                                   name, item_id)
        except Exception:
            logger.debug("[StickerTagger] worker failed", exc_info=True)

    threading.Thread(
        target=_worker,
        name=f"sticker-tagger-{uuid.uuid4().hex[:8]}",
        daemon=True,
    ).start()
