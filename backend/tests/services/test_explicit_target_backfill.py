from app.plugins.robot.explicit_target_backfill import (
    extract_explicit_qq_targets,
)
from app.plugins.robot.mcp.server import RobotMCPServer


def test_extract_multiple_explicit_targets() -> None:
    message = "把「大家好」分别发到私聊 2537134688 和群 770362397"
    targets = extract_explicit_qq_targets(message)
    assert targets == [("private", "2537134688"), ("group", "770362397")]


def test_extract_multiple_explicit_targets_english_keywords() -> None:
    message = "send hello to group 770362397 and private 2537134688"
    targets = extract_explicit_qq_targets(message)
    assert ("group", "770362397") in targets
    assert ("private", "2537134688") in targets
    assert len(targets) == 2


def test_extract_dedupes_repeated_target() -> None:
    message = "发到群 770362397，再确认一下群组：770362397 收到没"
    assert extract_explicit_qq_targets(message) == [("group", "770362397")]


def test_extract_single_target_stays_single() -> None:
    assert extract_explicit_qq_targets("帮我在群 770362397 里打个招呼") == [
        ("group", "770362397")
    ]


def test_extract_ignores_plain_numbers() -> None:
    assert extract_explicit_qq_targets("1+1 等于几？顺便算算 12345 + 67890") == []
    # ids shorter than 5 digits must not match either
    assert extract_explicit_qq_targets("发到群 1234 看看") == []


def test_delivery_tracker_records_per_ticket_targets() -> None:
    RobotMCPServer._clear_delivered_targets_for_test()
    try:
        RobotMCPServer.mark_target_delivered(
            "ticket:t1", "private", "2537134688", "大家好"
        )
        RobotMCPServer.mark_target_delivered("ticket:t1", "group", "770362397", "大家好")
        RobotMCPServer.mark_target_delivered("ticket:t2", "group", "11111111", "别的")

        assert RobotMCPServer.get_delivered_targets("ticket:t1") == {
            ("private", "2537134688"),
            ("group", "770362397"),
        }
        assert RobotMCPServer.get_delivered_targets("ticket:t2") == {
            ("group", "11111111")
        }
        assert RobotMCPServer.get_delivered_targets("ticket:missing") == set()
        assert RobotMCPServer.get_delivered_texts("ticket:t1")[("group", "770362397")] == (
            "大家好"
        )

        RobotMCPServer.reset_delivered_targets("ticket:t1")
        assert RobotMCPServer.get_delivered_targets("ticket:t1") == set()
    finally:
        RobotMCPServer._clear_delivered_targets_for_test()


def test_delivery_track_key_prefers_ticket_over_item() -> None:
    assert (
        RobotMCPServer.delivery_track_key_from_args(
            {"_reply_ticket_id": "abc", "item_id": "xyz"}
        )
        == "ticket:abc"
    )
    assert RobotMCPServer.delivery_track_key_from_args({"item_id": "xyz"}) == "item:xyz"
    assert RobotMCPServer.delivery_track_key_from_args({}) == ""
