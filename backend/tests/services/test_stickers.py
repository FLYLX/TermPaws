"""表情包库 + 出站链路测试。"""
from __future__ import annotations

import asyncio

from app.plugins.robot.contracts import RobotReplyTarget
from app.plugins.robot.platforms import _onebot_message_segments
from app.plugins.robot.stickers import (
    list_tags,
    normalize_tag,
    random_sticker,
    save_sticker,
)


def test_normalize_tag_strips_illegal_chars() -> None:
    assert normalize_tag("开心") == "开心"
    assert normalize_tag(" Happy! ") == "happy"
    assert normalize_tag("???") == ""
    assert normalize_tag("") == ""
    assert len(normalize_tag("很长" * 30)) <= 32


def test_sticker_store_roundtrip(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TERMPAWS_HOME", str(tmp_path))
    payload = b"\xff\xd8\xff\xfakejpeg"
    saved = save_sticker("开心", payload, ".jpg")
    assert saved is not None and saved.exists()

    # 随机取能拿到
    picked = random_sticker("开心")
    assert picked is not None and picked.read_bytes() == payload

    # 重复内容不再入库（同标签、跨标签都挡）
    assert save_sticker("开心", payload, ".jpg") is None
    assert save_sticker("难过", payload, ".jpg") is None
    assert list_tags() == {"开心": 1}

    # 退化匹配：标签目录名包含查询串
    save_sticker("开心死了", payload + b"x", ".jpg")
    assert random_sticker("开心") is not None

    # 不存在返回 None
    assert random_sticker("不存在") is None
    assert save_sticker("???", payload, ".jpg") is None

    tags = list_tags()
    assert tags == {"开心": 1, "开心死了": 1}


def test_onebot_segments_with_image() -> None:
    # 无图时保持纯文本行为
    assert _onebot_message_segments("你好", "") == "你好"
    # 有图时：文本段 + 图片段
    segments = _onebot_message_segments("看这个", "aGVsbG8=")
    assert isinstance(segments, list)
    assert segments[0] == {"type": "text", "data": {"text": "看这个"}}
    assert segments[1]["type"] == "image"
    assert segments[1]["data"]["file"] == "base64://aGVsbG8="
    # 纯图（无文本）
    only_image = _onebot_message_segments("", "aGVsbG8=")
    assert isinstance(only_image, list) and len(only_image) == 1
    assert only_image[0]["type"] == "image"


def test_send_text_with_bot_image_only() -> None:
    from app.plugins.robot.platforms import send_text_with_bot

    calls: list[dict] = []

    class _Adapter:
        def get_name(self):
            return "OneBot V11"

    class _Bot:
        adapter = _Adapter()
        self_id = "123456"

        async def call_api(self, action, **kwargs):
            calls.append({"action": action, **kwargs})

    bot = _Bot()
    bot.adapter.connections = {"123456": object()}
    target = RobotReplyTarget(target_type="group", target_id="1001")
    asyncio.run(send_text_with_bot(bot, target, "", image_b64="aGVsbG8="))
    assert len(calls) == 1
    assert calls[0]["action"] == "send_group_msg"
    assert calls[0]["message"][-1]["type"] == "image"


def test_sticker_tagger_response_parse() -> None:
    from app.plugins.robot.sticker_tagger import _is_meme_response

    assert _is_meme_response("开心") == "开心"
    assert _is_meme_response(" 卖萌。") == "卖萌"
    assert _is_meme_response("none") == ""
    assert _is_meme_response("None") == ""
    assert _is_meme_response("") == ""
    assert _is_meme_response("这不是表情包因为图片里是一只猫的照片") == ""
