from app.services.agent.robot_delivery import robot_reply_event_content


def test_robot_reply_event_content_formats_sent_text() -> None:
    assert (
        robot_reply_event_content(
            {"text": "你好\n主人"},
            "Message sent to current robot conversation.",
        )
        == "已回复 QQ：你好 主人"
    )


def test_robot_reply_event_content_unwraps_structured_text() -> None:
    assert robot_reply_event_content(
        {"text": {"type": "text", "text": "服务器启动成功"}},
        "Message sent to current robot conversation.",
    ).endswith("服务器启动成功")


def test_robot_reply_event_content_formats_duplicate_skip() -> None:
    assert (
        robot_reply_event_content(
            {"text": "同一个回复"},
            "Message sent to current robot conversation. Duplicate QQ reply suppressed.",
        )
        == "已跳过重复 QQ 回复：同一个回复"
    )
