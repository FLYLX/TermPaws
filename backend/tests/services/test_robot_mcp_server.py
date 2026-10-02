import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.plugins.robot.contracts import RobotReplyTarget
from app.plugins.robot.conversation_memory import robot_conversation_memory
from app.plugins.robot.mcp.context import (
    RobotMCPContext,
    register_robot_mcp_context,
    unregister_robot_mcp_context,
)
from app.plugins.robot.mcp.server import RobotMCPServer
from app.services.agent.memory.vector_store import vector_store


@pytest.fixture(autouse=True)
def clear_robot_delivery_tracker():
    RobotMCPServer._clear_delivered_targets_for_test()
    yield
    RobotMCPServer._clear_delivered_targets_for_test()


def _test_memory_dir(name: str) -> Path:
    base = Path(os.environ.get("ROBOT_CONVERSATION_MEMORY_DIR") or tempfile.gettempdir())
    path = base / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_robot_mcp_send_message_requires_context_or_explicit_target() -> None:
    server = RobotMCPServer()

    result = server.call_tool("send_message", {"text": "hello"})

    assert result[0]["type"] == "text"
    assert "no active QQ robot conversation context" in result[0]["text"]
    assert "target_type and target_id" in result[0]["text"]


def test_robot_mcp_send_message_resolves_visible_web_context_target(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: dict[str, object] = {}

    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": fallback_robot_id or "robot-1",
    )

    def fake_send_message(robot_id, reply_target, text):
        sent["robot_id"] = robot_id
        sent["reply_target"] = reply_target
        sent["text"] = text

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    result = server.call_tool(
        "send_message",
        {
            "text": "服务器没开",
            "reply_to": "baka",
            "_robot_known_targets": [
                {
                    "conversation": "group:770362397",
                    "target_type": "group",
                    "target_id": "770362397",
                    "sender": "baka (1874419565)",
                    "robot_id": "robot-1",
                }
            ],
        },
    )

    assert result[0]["text"].startswith("Message sent to QQ group 770362397 from chat context.")
    assert sent["robot_id"] == "robot-1"
    assert sent["text"] == "服务器没开"
    assert isinstance(sent["reply_target"], RobotReplyTarget)
    assert sent["reply_target"].target_type == "group"
    assert sent["reply_target"].target_id == "770362397"


def test_robot_mcp_send_message_noops_no_reply_intent(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[str] = []

    def fake_send_message(_robot_id, _reply_target, text):
        sent.append(text)

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    result = server.call_tool(
        "send_message",
        {
            "text": "[no_qq_reply]",
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result == [
        {"type": "text", "text": "No QQ message sent: no reply needed."}
    ]
    assert sent == []


def test_robot_mcp_send_message_resends_identical_text(monkeypatch) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="123456",
        metadata={"target": {"id": "123456"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:123456:user-1",
            reply_target=target,
        )
    )
    sent: list[str] = []

    def fake_send_message(_robot_id, _reply_target, text):
        sent.append(text)

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    try:
        first = server.call_tool(
            "send_message",
            {"text": "同一个回复", "_robot_context_token": token},
        )
        second = server.call_tool(
            "send_message",
            {"text": "同一个回复", "_robot_context_token": token},
        )
    finally:
        unregister_robot_mcp_context(token)

    assert first[0]["text"].startswith("Message sent to current robot conversation.")
    # 同一内容 30s 内的第二次发送被抑制（流控，防模型连续重复发）
    assert second[0]["text"].startswith("Message sent to current robot conversation.")
    assert sent == ["同一个回复"]


def test_robot_mcp_send_failure_allows_retry(monkeypatch) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="123456",
        metadata={"target": {"id": "123456"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:123456:user-1",
            reply_target=target,
        )
    )
    attempts = {"count": 0}
    sent: list[str] = []

    def fake_send_message(_robot_id, _reply_target, text):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("bridge down")
        sent.append(text)

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    try:
        first = server.call_tool(
            "send_message",
            {"text": "失败后重试", "_robot_context_token": token},
        )
        second = server.call_tool(
            "send_message",
            {"text": "失败后重试", "_robot_context_token": token},
        )
    finally:
        unregister_robot_mcp_context(token)

    assert first[0]["text"] == "Error: bridge down"
    assert second[0]["text"].startswith("Message sent to current robot conversation.")
    assert sent == ["失败后重试"]


def test_robot_mcp_send_message_blocks_internal_tool_trace(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[str] = []

    def fake_send_message(_robot_id, _reply_target, text):
        sent.append(text)

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    result = server.call_tool(
        "send_message",
        {
            "text": (
                "Executing tool: mcp_robot_send_message\n\n"
                "Message sent to current robot conversation."
            ),
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result == [
        {"type": "text", "text": "No QQ message sent: internal tool trace."}
    ]
    assert sent == []


def test_robot_mcp_send_message_blocks_dsml_tool_trace(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[str] = []

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        lambda _robot_id, _reply_target, text: sent.append(text),
    )

    result = server.call_tool(
        "send_message",
        {
            "text": (
                '<｜｜DSML｜｜tool_calls>\n'
                '<｜｜DSML｜｜invoke name="mcp_local_run_job">\n'
                '<｜｜DSML｜｜parameter name="action" string="true">'
                'complete_current_step</｜｜DSML｜｜parameter>\n'
                '</｜｜DSML｜｜invoke>\n'
                '</｜｜DSML｜｜tool_calls>'
            ),
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result == [
        {"type": "text", "text": "No QQ message sent: internal tool trace."}
    ]
    assert sent == []


def test_robot_mcp_send_message_sanitizes_mixed_internal_trace(
    monkeypatch,
) -> None:
    server = RobotMCPServer()
    sent: list[str] = []
    monkeypatch.setattr(
        robot_conversation_memory,
        "base_dir",
        _test_memory_dir("robot-mcp-send-memory"),
    )

    def fake_send_message(_robot_id, _reply_target, text):
        sent.append(text)

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "text": (
                "在呢。需要做什么测试？\n\n"
                "Executing tool: mcp_robot_send_message\n\n"
                "Message sent to current robot conversation.\n\n"
                "[no_qq_reply]"
            ),
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result[0]["text"].startswith("Message sent to QQ group 123456.")
    assert sent == ["在呢。需要做什么测试？"]
    memory = robot_conversation_memory.read("robot-2", "group:123456")
    assert "在呢。需要做什么测试？" in memory
    assert "Executing tool" not in memory
    assert "Message sent to current robot conversation" not in memory
    assert "[no_qq_reply]" not in memory


def test_robot_mcp_send_message_extracts_degraded_send_payload(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[str] = []

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        lambda _robot_id, _reply_target, text: sent.append(text),
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "text": (
                "No, user asked to expose memory. need send tool call.\n"
                '{"text":"我记得你喜欢蓝色，也在管理 Minecraft 服务器。"}'
            ),
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result[0]["text"].startswith("Message sent to QQ group 123456.")
    assert sent == ["我记得你喜欢蓝色，也在管理 Minecraft 服务器。"]


def test_robot_mcp_send_message_uses_explicit_target(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: dict[str, object] = {}

    def fake_send_message(robot_id, reply_target, text):
        sent["robot_id"] = robot_id
        sent["reply_target"] = reply_target
        sent["text"] = text

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "text": "notify group",
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result[0]["text"].startswith("Message sent to QQ group 123456.")
    assert sent["robot_id"] == "robot-2"
    assert sent["text"] == "notify group"
    assert isinstance(sent["reply_target"], RobotReplyTarget)
    assert sent["reply_target"].target_type == "group"
    assert sent["reply_target"].target_id == "123456"


def test_robot_mcp_send_message_uses_llm_chosen_messages(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[tuple[str, str, str, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target.target_type, reply_target.target_id, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "messages": ["我先看一下", "等我确认一下状态"],
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result[0]["text"].startswith("Message sent to QQ group 123456.")
    assert sent == [
        ("robot-2", "group", "123456", "我先看一下"),
        ("robot-2", "group", "123456", "等我确认一下状态"),
    ]


def test_robot_mcp_send_message_unwraps_structured_text_blocks(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[str] = []

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        lambda _robot_id, _reply_target, text: sent.append(text),
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "messages": [
                {"text": "RCON 也开了，现在可以进游戏了~"},
                {"type": "text", "text": "服务器启动成功啦~ Done (2.901s)！"},
                "{'text': '第三条也是纯文本'}",
            ],
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result[0]["text"].startswith("Message sent to QQ group 123456.")
    assert sent == [
        "RCON 也开了，现在可以进游戏了~",
        "服务器启动成功啦~ Done (2.901s)！",
        "第三条也是纯文本",
    ]


def test_robot_mcp_send_message_long_text_sends_without_hard_cap(monkeypatch) -> None:
    """单气泡长度不再有硬上限：长文照发，节奏完全由 agent 判断。"""
    server = RobotMCPServer()
    sent: list[str] = []

    def fake_send_message(_robot_id, _reply_target, text):
        sent.append(text)

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "text": "x" * 500,
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result[0]["type"] == "text"
    assert "Message sent" in result[0]["text"]
    assert sent == ["x" * 500]


def test_robot_mcp_send_message_compacts_paragraphs_inside_each_message(
    monkeypatch,
) -> None:
    server = RobotMCPServer()
    sent: list[str] = []

    def fake_send_message(_robot_id, _reply_target, text):
        sent.append(text)

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "messages": ["我先看一下\n\n可能是桥接还没接上", "等我确认一下状态"],
            "target_type": "group",
            "target_id": "123456",
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result[0]["text"].startswith("Message sent to QQ group 123456.")
    assert sent == ["我先看一下可能是桥接还没接上", "等我确认一下状态"]


def test_robot_mcp_send_message_accepts_stamped_target_aliases(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[tuple[str, RobotReplyTarget, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    alias_result = server.call_tool(
        "send_message",
        {
            "text": "alias target",
            "mcp_target_type": "private",
            "mcp_target_id": "654321",
            "_TermPaws_user_id": "user-1",
        },
    )
    conversation_result = server.call_tool(
        "send_message",
        {
            "text": "conversation target",
            "conversation": "group:123456",
            "_TermPaws_user_id": "user-1",
            "_robot_known_targets": [
                {
                    "conversation": "group:123456",
                    "target_type": "group",
                    "target_id": "123456",
                    "sender": "Alice (10001)",
                }
            ],
        },
    )

    assert alias_result[0]["text"].startswith("Message sent to QQ private 654321.")
    assert conversation_result[0]["text"].startswith(
        "Message sent to QQ group 123456 from chat context."
    )
    assert [(robot_id, target.target_type, target.target_id, text) for robot_id, target, text in sent] == [
        ("robot-2", "private", "654321", "alias target"),
        ("robot-2", "group", "123456", "conversation target"),
    ]


def test_robot_mcp_send_message_allows_visible_context_reply_to_without_active_context(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[tuple[str, RobotReplyTarget, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": fallback_robot_id or "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "text": "tell Alice",
            "reply_to": "Alice",
            "_TermPaws_user_id": "user-1",
            "_robot_known_targets": [
                {
                    "conversation": "group:123456",
                    "target_type": "group",
                    "target_id": "123456",
                    "sender": "Alice (10001)",
                    "robot_id": "robot-1",
                },
                {
                    "conversation": "private:654321",
                    "target_type": "private",
                    "target_id": "654321",
                    "sender": "Bob (654321)",
                    "robot_id": "robot-1",
                },
            ],
        },
    )

    assert result[0]["text"].startswith("Message sent to QQ group 123456 from chat context.")
    assert len(sent) == 1
    assert sent[0][0] == "robot-1"
    assert sent[0][1].target_type == "group"
    assert sent[0][1].target_id == "123456"
    assert sent[0][2] == "tell Alice"


def test_robot_mcp_send_message_blocks_only_context_target_without_active_context(
    monkeypatch,
) -> None:
    server = RobotMCPServer()
    sent: list[tuple[str, RobotReplyTarget, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": fallback_robot_id or "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "text": "single target",
            "_TermPaws_user_id": "user-1",
            "_robot_known_targets": [
                {
                    "conversation": "private:654321",
                    "target_type": "private",
                    "target_id": "654321",
                    "sender": "Bob (654321)",
                    "robot_id": "robot-1",
                }
            ],
        },
    )

    assert result[0]["type"] == "text"
    assert "no active QQ robot conversation context" in result[0]["text"]
    assert sent == []


def test_robot_mcp_send_message_uses_registered_context(monkeypatch) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="universal",
        target_id="group-1",
        metadata={"target": {"id": "group-1"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-1",
            sender_key="onebot_v11:group:group-1:user-1",
            reply_target=target,
        )
    )
    sent: dict[str, object] = {}

    def fake_send_message(robot_id, reply_target, text):
        sent["robot_id"] = robot_id
        sent["reply_target"] = reply_target
        sent["text"] = text

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    try:
        result = server.call_tool(
            "send_message",
            {"text": "notify user", "_robot_context_token": token},
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["text"].startswith("Message sent to current robot conversation.")
    assert sent["robot_id"] == "robot-1"
    assert sent["text"] == "notify user"
    assert isinstance(sent["reply_target"], RobotReplyTarget)
    assert sent["reply_target"].target_id == "group-1"


def test_robot_mcp_send_message_allows_up_to_three_bubbles(monkeypatch) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="group-1",
        metadata={"conversation": {"type": "group", "id": "group-1"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-1",
            sender_key="onebot_v11:group:group-1:user-1",
            reply_target=target,
        )
    )
    sent: list[str] = []
    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        lambda _robot_id, _reply_target, text: sent.append(text),
    )

    try:
        result = server.call_tool(
            "send_message",
            {
                "messages": ["I am here.", "What do you need?"],
                "_robot_context_token": token,
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["text"].startswith("Message sent to current robot conversation.")
    assert sent == ["I am here.", "What do you need?"]


def test_robot_mcp_send_message_keeps_multi_sender_batch_bubbles(monkeypatch) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="group-1",
        metadata={
            "conversation": {"type": "group", "id": "group-1"},
            "allow_multiple_reply_messages": True,
        },
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-1",
            sender_key="onebot_v11:group:group-1:user-2",
            reply_target=target,
        )
    )
    sent: list[str] = []
    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        lambda _robot_id, _reply_target, text: sent.append(text),
    )

    try:
        result = server.call_tool(
            "send_message",
            {
                "messages": ["Alice: first answer", "Bob: second answer"],
                "_robot_context_token": token,
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["text"].startswith("Message sent to current robot conversation.")
    assert sent == ["Alice: first answer", "Bob: second answer"]


def test_robot_mcp_send_message_blocks_sleeping_active_context(monkeypatch) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
            conversation_key="group:current-group",
            conversation_generation=99,
            reply_requires_awake=True,
        )
    )
    sent: list[tuple[str, RobotReplyTarget, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target, text))

    monkeypatch.setattr(
        "app.plugins.robot.service.robot_service.conversation_controller_allows_reply",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    try:
        result = server.call_tool(
            "send_message",
            {"text": "late reply", "_robot_context_token": token},
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result == [
        {
            "type": "text",
            "text": (
                "Send skipped: this turn was superseded by a newer QQ message, so the "
                "conversation belongs to a newer turn now. Do NOT retry mcp_robot_send_message "
                "and do NOT mention this failure to the user; the newer turn will deliver the "
                "reply. End this turn silently."
            ),
        }
    ]
    assert sent == []

def test_robot_mcp_sleep_conversation_uses_active_context(monkeypatch) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
            conversation_key="group:current-group",
            conversation_generation=99,
            reply_requires_awake=True,
        )
    )
    calls: list[tuple[str, str, str]] = []

    def fake_sleep(robot_id, conversation_key, *, reason):
        calls.append((robot_id, conversation_key, reason))
        return True

    monkeypatch.setattr(
        "app.plugins.robot.service.robot_service.sleep_conversation_controller",
        fake_sleep,
    )

    try:
        result = server.call_tool(
            "sleep_conversation",
            {"_robot_context_token": token},
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result == [
        {
            "type": "text",
            "text": "No QQ message sent: current conversation is sleeping.",
        }
    ]
    assert calls == [("robot-current", "group:current-group", "mcp_sleep_conversation")]

def test_robot_mcp_recalls_long_term_memory_scoped_to_active_context(monkeypatch) -> None:
    server = RobotMCPServer()
    calls: list[dict] = []

    def fake_search_memories(**kwargs):
        calls.append(kwargs)
        return [
            {
                "id": "current",
                "content": "nickname is XiaoChai",
                "metadata": {
                    "memory_type": "fact",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:current-group",
                    "verified": True,
                },
                "distance": 0.05,
            },
            {
                "id": "other-group",
                "content": "other group secret",
                "metadata": {
                    "memory_type": "fact",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                },
                "distance": 0.01,
            },
            {
                "id": "same-speaker-other-group",
                "content": "call this QQ user Master",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                    "speaker_global_key": "onebot_v11:user:user-1",
                    "memory_scope": "speaker",
                },
                "distance": 0.03,
            },
            {
                "id": "other-speaker-other-group",
                "content": "call another user Boss",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                    "speaker_global_key": "onebot_v11:user:user-2",
                    "memory_scope": "speaker",
                },
                "distance": 0.01,
            },
            {
                "id": "global",
                "content": "general robot preference",
                "metadata": {"memory_type": "preference"},
                "distance": 0.1,
            },
            {
                "id": "other-robot",
                "content": "other robot memory",
                "metadata": {
                    "memory_type": "fact",
                    "robot_id": "robot-other",
                },
                "distance": 0.02,
            },
        ]

    def fake_get_all_memories(*_args, **_kwargs):
        return []

    monkeypatch.setattr(vector_store, "search_memories", fake_search_memories)
    monkeypatch.setattr(vector_store, "get_all_memories", fake_get_all_memories)
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
            conversation_key="group:current-group",
        )
    )

    try:
        result = server.call_tool(
            "recall_memory",
            {
                "_robot_context_token": token,
                "_TermPaws_item_id": "item-1",
                "query": "nickname",
                "n_results": 3,
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert calls[0]["handler_id"] == "item-1"
    assert calls[0]["query"] == "nickname"
    assert calls[0]["n_results"] == 1
    text = result[0]["text"]
    assert "nickname is XiaoChai" in text
    assert "call this QQ user Master" in text
    assert "general robot preference" in text
    assert "other group secret" not in text
    assert "call another user Boss" not in text
    assert "other robot memory" not in text
    assert "verified" not in text
    assert "group:current-group" not in text
    assert text.index("nickname is XiaoChai") < text.index("call this QQ user Master")
    assert text.index("call this QQ user Master") < text.index("general robot preference")


def test_robot_mcp_lists_long_term_memory_scoped_to_active_context(monkeypatch) -> None:
    server = RobotMCPServer()
    tools = {tool["name"]: tool for tool in server.list_tools()}
    assert "list_memories" in tools
    assert tools["list_memories"]["skip_memory"] is True

    memories_by_type = {
        "fact": [
            {
                "id": "current-conversation",
                "content": "the bot name is XiaoChai",
                "metadata": {
                    "memory_type": "fact",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:current-group",
                    "memory_scope": "conversation",
                    "speaker": "Alice (user-1)",
                },
            },
            {
                "id": "other-robot",
                "content": "other robot memory",
                "metadata": {
                    "memory_type": "fact",
                    "robot_id": "robot-other",
                },
            },
        ],
        "preference": [
            {
                "id": "same-speaker-other-group",
                "content": "call this QQ user Master",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                    "speaker_global_key": "onebot_v11:user:user-1",
                    "memory_scope": "speaker",
                },
            },
            {
                "id": "other-speaker",
                "content": "call another QQ user Boss",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                    "speaker_global_key": "onebot_v11:user:user-2",
                    "memory_scope": "speaker",
                },
            },
        ],
        "context": [
            {
                "id": "other-conversation",
                "content": "other group server port is 19999",
                "metadata": {
                    "memory_type": "context",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                    "memory_scope": "conversation",
                },
            }
        ],
    }

    def fake_get_all_memories(item_id, memory_type=None):
        assert item_id == "item-1"
        return memories_by_type.get(memory_type, [])

    monkeypatch.setattr(vector_store, "get_all_memories", fake_get_all_memories)
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
            conversation_key="group:current-group",
        )
    )

    try:
        result = server.call_tool(
            "list_memories",
            {
                "_robot_context_token": token,
                "_TermPaws_item_id": "item-1",
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    text = result[0]["text"]
    assert "the bot name is XiaoChai" in text
    assert "call this QQ user Master" in text
    assert "other robot memory" not in text
    assert "call another QQ user Boss" not in text
    assert "other group server port is 19999" not in text
    assert text.index("call this QQ user Master") < text.index("the bot name is XiaoChai")


def test_robot_mcp_list_memories_requires_active_qq_context() -> None:
    server = RobotMCPServer()

    result = server.call_tool(
        "list_memories",
        {"_TermPaws_item_id": "item-1"},
    )

    assert "requires an active QQ robot conversation context" in result[0]["text"]


def test_robot_mcp_recall_uses_scoped_memory_fallback(monkeypatch) -> None:
    server = RobotMCPServer()
    calls: list[tuple[str, str | None]] = []

    def fake_search_memories(**_kwargs):
        return [
            {
                "id": "other-vector",
                "content": "other group has a server port",
                "metadata": {
                    "memory_type": "context",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                    "memory_scope": "conversation",
                },
                "distance": 0.01,
            }
        ]

    scoped_by_type = {
        "context": [
            {
                "id": "current-group-context",
                "content": "current group server port is 28888",
                "metadata": {
                    "memory_type": "context",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:current-group",
                    "memory_scope": "conversation",
                    "verified": True,
                },
            },
            {
                "id": "other-group-context",
                "content": "other group server port is 19999",
                "metadata": {
                    "memory_type": "context",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                    "memory_scope": "conversation",
                },
            },
        ],
        "preference": [
            {
                "id": "same-speaker-preference",
                "content": "call this QQ user Master",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                    "speaker_global_key": "onebot_v11:user:user-1",
                    "memory_scope": "speaker",
                },
            },
            {
                "id": "other-speaker-preference",
                "content": "call another QQ user Boss",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": "robot-current",
                    "robot_conversation_key": "group:other-group",
                    "speaker_global_key": "onebot_v11:user:user-2",
                    "memory_scope": "speaker",
                },
            },
        ],
    }

    def fake_get_all_memories(item_id, memory_type=None):
        calls.append((item_id, memory_type))
        return scoped_by_type.get(memory_type, [])

    monkeypatch.setattr(vector_store, "search_memories", fake_search_memories)
    monkeypatch.setattr(vector_store, "get_all_memories", fake_get_all_memories)
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
            conversation_key="group:current-group",
        )
    )

    try:
        result = server.call_tool(
            "recall_memory",
            {
                "_robot_context_token": token,
                "_TermPaws_item_id": "item-1",
                "query": "server port",
                "n_results": 3,
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert ("item-1", "context") in calls
    assert ("item-1", "preference") in calls
    text = result[0]["text"]
    assert "current group server port is 28888" in text
    assert "call this QQ user Master" not in text
    assert "other group has a server port" not in text
    assert "other group server port is 19999" not in text
    assert "call another QQ user Boss" not in text


def test_robot_mcp_relation_enumeration_recalls_old_group_facts(monkeypatch) -> None:
    server = RobotMCPServer()
    memories = [
        {
            "id": "new-question",
            "content": "Ac国掌管ac的神 (3385417251): 谁是猫娘",
            "metadata": {
                "memory_type": "fact",
                "memory_scope": "conversation",
                "robot_id": "robot-current",
                "robot_conversation_key": "group:770362397",
                "robot_memory_schema_version": 3,
            },
            "distance": 0.01,
        },
        {
            "id": "candy",
            "content": "和煦的糖果风 (641681910): 我是猫娘",
            "metadata": {
                "memory_type": "fact",
                "memory_scope": "speaker",
                "robot_id": "robot-current",
                "robot_conversation_key": "group:770362397",
                "speaker_global_key": "onebot_v11:user:641681910",
                "robot_memory_schema_version": 3,
            },
            "distance": 0.08,
        },
        {
            "id": "hamburger",
            "content": "月影寒波（简称汉堡）是猫娘",
            "metadata": {
                "memory_type": "fact",
                "memory_scope": "speaker",
                "robot_id": "robot-current",
                "robot_conversation_key": "group:770362397",
                "speaker_global_key": "onebot_v11:user:162221240",
                "robot_memory_schema_version": 3,
            },
            "distance": 0.12,
        },
        {
            "id": "hamburger-import",
            "content": "月影寒波（简称汉堡）是猫娘",
            "metadata": {
                "memory_type": "fact",
                "memory_scope": "conversation",
                "robot_id": "robot-current",
                "robot_conversation_key": "group:770362397",
                "robot_memory_schema_version": 3,
            },
            "distance": 0.1,
        },
    ]

    monkeypatch.setattr(vector_store, "search_memories", lambda **_kwargs: memories)
    monkeypatch.setattr(
        vector_store,
        "get_all_memories",
        lambda _item_id, memory_type=None: [
            memory
            for memory in memories
            if memory_type is None
            or memory["metadata"]["memory_type"] == memory_type
        ],
    )
    target = RobotReplyTarget(
        target_type="group",
        target_id="770362397",
        metadata={"target": {"id": "770362397"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:770362397:2537134688",
            reply_target=target,
            conversation_key="group:770362397",
        )
    )

    try:
        result = server.call_tool(
            "recall_memory",
            {
                "_robot_context_token": token,
                "_TermPaws_item_id": "item-1",
                "query": "还有谁是猫娘",
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    text = result[0]["text"]
    assert "谁是猫娘" not in text
    assert "和煦的糖果风" in text
    assert "月影寒波" in text
    assert text.count("月影寒波（简称汉堡）是猫娘") == 1


def test_robot_mcp_save_memory_persists_scoped_long_term_memory(monkeypatch) -> None:
    server = RobotMCPServer()
    tools = {tool["name"]: tool for tool in server.list_tools()}
    assert "save_memory" in tools
    assert tools["save_memory"]["skip_memory"] is True

    captured: dict[str, object] = {}

    def fake_get_all_memories(item_id: str, memory_type: str | None = None):
        captured["existing_item_id"] = item_id
        captured["existing_memory_type"] = memory_type
        return []

    def fake_add_memory(**kwargs):
        captured.update(kwargs)
        return "memory-123456"

    monkeypatch.setattr(vector_store, "get_all_memories", fake_get_all_memories)
    monkeypatch.setattr(vector_store, "add_memory", fake_add_memory)

    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
            conversation_key="group:current-group",
        )
    )

    try:
        result = server.call_tool(
            "save_memory",
            {
                "_robot_context_token": token,
                "_TermPaws_item_id": "item-1",
                "content": "\u4f60\u53eb\u5927\u72d7",
                "memory_type": "fact",
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["type"] == "text"
    assert "Memory saved" in result[0]["text"]
    assert captured["existing_item_id"] == "item-1"
    assert captured["existing_memory_type"] == "fact"
    assert captured["handler_id"] == "item-1"
    assert captured["content"] == "\u4f60\u53eb\u5927\u72d7"
    assert captured["memory_type"] == "fact"
    metadata = captured["metadata"]
    assert metadata["type"] == "robot_agent_saved"
    assert metadata["source"] == "qq_robot_agent"
    assert metadata["verified"] is False
    assert metadata["robot_id"] == "robot-current"
    assert metadata["robot_conversation_key"] == "group:current-group"
    assert metadata["conversation_key"] == "group:current-group"
    assert metadata["speaker_key"] == "onebot_v11:group:current-group:user-1"
    assert metadata["speaker_global_key"] == "onebot_v11:user:user-1"
    assert metadata["memory_scope"] == "robot"
    assert metadata["content_hash"]


def test_robot_mcp_save_memory_does_not_expose_removed_task_type() -> None:
    server = RobotMCPServer()
    save_tool = next(tool for tool in server.list_tools() if tool["name"] == "save_memory")

    assert "task" not in save_tool["inputSchema"]["properties"]["memory_type"]["enum"]


def test_robot_mcp_reads_registered_context_conversation_memory(
    monkeypatch,
    tmp_path,
) -> None:
    server = RobotMCPServer()
    monkeypatch.setattr(robot_conversation_memory, "base_dir", tmp_path)
    robot_conversation_memory.append_user_message(
        "robot-current",
        "group:current-group",
        "hello current keyword",
        sender="Alice (10001)",
    )
    robot_conversation_memory.append_assistant_message(
        "robot-current",
        "group:current-group",
        "assistant current reply",
    )
    robot_conversation_memory.append_user_message(
        "robot-current",
        "group:other-group",
        "other keyword",
        sender="Bob",
    )
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
        )
    )

    try:
        result = server.call_tool(
            "read_conversation_memory",
            {"_robot_context_token": token, "query": "keyword"},
        )
        blocked = server.call_tool(
            "read_conversation_memory",
            {"_robot_context_token": token, "conversation": "group:other-group"},
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["type"] == "text"
    assert "group:current-group" in result[0]["text"]
    assert "hello current keyword" in result[0]["text"]
    assert "assistant current reply" not in result[0]["text"]
    assert "other keyword" not in result[0]["text"]
    assert "active QQ-triggered context is locked to group:current-group" in blocked[0]["text"]


def test_robot_mcp_read_memory_active_context_caps_requested_lines(
    monkeypatch,
) -> None:
    server = RobotMCPServer()
    monkeypatch.setattr(
        robot_conversation_memory,
        "base_dir",
        _test_memory_dir("robot-mcp-read-memory-cap"),
    )
    robot_conversation_memory.replace(
        "robot-current",
        "group:current-group",
        "".join(
            f"[2026-01-01T00:00:{index:02d}+00:00] user Alice: message-{index:02d}\n"
            for index in range(30)
        ),
    )
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
        )
    )

    try:
        result = server.call_tool(
            "read_conversation_memory",
            {"_robot_context_token": token, "lines": 500},
        )
    finally:
        unregister_robot_mcp_context(token)

    text = result[0]["text"]
    assert "recent 12 line(s)" in text
    assert "message-17" not in text
    assert "message-18" in text
    assert "message-29" in text


def test_robot_mcp_read_memory_ignores_partial_target_in_active_context(
    monkeypatch,
    tmp_path,
) -> None:
    server = RobotMCPServer()
    monkeypatch.setattr(robot_conversation_memory, "base_dir", tmp_path)
    robot_conversation_memory.append_user_message(
        "robot-current",
        "group:current-group",
        "older catgirl memory",
        sender="Alice (10001)",
    )
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
            conversation_key="group:current-group",
        )
    )

    try:
        result = server.call_tool(
            "read_conversation_memory",
            {
                "_robot_context_token": token,
                "target_type": "group",
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert "target_id is required" not in result[0]["text"]
    assert "older catgirl memory" in result[0]["text"]


def test_robot_mcp_read_memory_sanitizes_old_internal_trace(
    monkeypatch,
) -> None:
    server = RobotMCPServer()
    monkeypatch.setattr(
        robot_conversation_memory,
        "base_dir",
        _test_memory_dir("robot-mcp-read-memory"),
    )
    robot_conversation_memory.replace(
        "robot-current",
        "private:2537134688",
        (
            "[2026-01-01T00:00:00+00:00] user FLY: 测试\n"
            "[2026-01-01T00:00:01+00:00] assistant: 在呢。需要做什么测试？\n"
            "Executing tool: mcp_robot_send_message\n"
            "Message sent to current robot conversation.\n"
            "[no_qq_reply]\n"
        ),
    )
    target = RobotReplyTarget(
        target_type="private",
        target_id="2537134688",
        metadata={"target": {"id": "2537134688", "private": True}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:private:2537134688",
            reply_target=target,
        )
    )

    try:
        result = server.call_tool(
            "read_conversation_memory",
            {"_robot_context_token": token},
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["type"] == "text"
    assert "在呢。需要做什么测试？" in result[0]["text"]
    assert "Executing tool" not in result[0]["text"]
    assert "Message sent to current robot conversation" not in result[0]["text"]
    assert "[no_qq_reply]" not in result[0]["text"]


def test_robot_conversation_memory_lists_group_and_private_logs() -> None:
    from app.plugins.robot.conversation_memory import RobotConversationMemoryManager

    manager = RobotConversationMemoryManager(
        base_dir=_test_memory_dir("robot-list-memory")
    )
    manager.replace("robot-1", "group:123456", "[seed] user Alice: hello\n")
    manager.replace("robot-1", "private:654321", "[seed] user Bob: hi\n")

    entries = manager.list_conversations("robot-1")

    assert {entry.conversation_key for entry in entries} == {
        "group:123456",
        "private:654321",
    }
    assert {entry.filename for entry in entries} == {
        "123456.log",
        "private-654321.log",
    }
    assert all(entry.exists for entry in entries)


def test_robot_mcp_send_message_prefers_active_context_over_ambiguous_history(
    monkeypatch,
) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
        )
    )
    sent: dict[str, object] = {}

    def fake_send_message(robot_id, reply_target, text):
        sent["robot_id"] = robot_id
        sent["reply_target"] = reply_target
        sent["text"] = text

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    try:
        result = server.call_tool(
            "send_message",
            {
                "text": "reply current chat",
                "_robot_context_token": token,
                "_robot_known_targets": [
                    {
                        "conversation": "group:other-group",
                        "target_type": "group",
                        "target_id": "other-group",
                        "sender": "Alice (10001)",
                        "robot_id": "robot-current",
                    },
                    {
                        "conversation": "private:654321",
                        "target_type": "private",
                        "target_id": "654321",
                        "sender": "Bob (654321)",
                        "robot_id": "robot-current",
                    },
                ],
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["text"].startswith("Message sent to current robot conversation.")
    assert sent["robot_id"] == "robot-current"
    assert sent["text"] == "reply current chat"
    assert isinstance(sent["reply_target"], RobotReplyTarget)
    assert sent["reply_target"].target_type == "group"
    assert sent["reply_target"].target_id == "current-group"


def test_robot_mcp_blocks_reply_to_other_context_from_active_robot_turn(
    monkeypatch,
) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={
            "conversation": {
                "type": "group",
                "id": "current-group",
                "target_type": "group",
                "target_id": "current-group",
            }
        },
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
        )
    )
    sent: list[tuple[str, RobotReplyTarget, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    try:
        result = server.call_tool(
            "send_message",
            {
                "text": "wrong chat",
                "_robot_context_token": token,
                "reply_to": "group:other-group",
                "_robot_known_targets": [
                    {
                        "conversation": "group:other-group",
                        "target_type": "group",
                        "target_id": "other-group",
                        "sender": "Alice (10001)",
                        "robot_id": "robot-current",
                    }
                ],
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["type"] == "text"
    assert "active QQ-triggered context is locked to group:current-group" in result[0]["text"]
    assert sent == []


def test_robot_mcp_blocks_explicit_other_target_from_active_robot_turn(
    monkeypatch,
) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
        )
    )
    sent: list[tuple[str, RobotReplyTarget, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    try:
        result = server.call_tool(
            "send_message",
            {
                "text": "wrong explicit target",
                "_robot_context_token": token,
                "target_type": "group",
                "target_id": "other-group",
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["type"] == "text"
    assert "active QQ-triggered context is locked to group:current-group" in result[0]["text"]
    assert sent == []


def test_robot_mcp_blocks_broadcast_from_active_robot_turn(monkeypatch) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="private",
        target_id="654321",
        metadata={"target": {"id": "654321", "private": True}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:private:654321",
            reply_target=target,
        )
    )
    sent: list[tuple[str, RobotReplyTarget, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    try:
        result = server.call_tool(
            "send_message",
            {
                "text": "broadcast from active",
                "_robot_context_token": token,
                "broadcast": True,
                "_robot_known_targets": [
                    {
                        "conversation": "group:123456",
                        "target_type": "group",
                        "target_id": "123456",
                        "sender": "Alice (10001)",
                        "robot_id": "robot-current",
                    }
                ],
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["type"] == "text"
    assert "active QQ-triggered context is locked to private:654321" in result[0]["text"]
    assert sent == []


def test_robot_mcp_send_message_blocks_broadcast_without_active_context(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[tuple[str, str, str, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target.target_type, reply_target.target_id, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": fallback_robot_id or "robot-default",
    )

    result = server.call_tool(
        "send_message",
        {
            "text": "critical alert",
            "broadcast": True,
            "_TermPaws_user_id": "user-1",
            "_robot_known_targets": [
                {
                    "conversation": "group:123456",
                    "target_type": "group",
                    "target_id": "123456",
                    "sender": "Alice (10001)",
                    "robot_id": "robot-1",
                },
                {
                    "conversation": "private:654321",
                    "target_type": "private",
                    "target_id": "654321",
                    "sender": "Bob (654321)",
                    "robot_id": "robot-2",
                },
            ],
        },
    )

    assert result[0]["type"] == "text"
    assert "broadcast from prior QQ context is disabled" in result[0]["text"]
    assert sent == []


@pytest.mark.parametrize(
    "partial_target",
    [
        {"target_type": "group"},
        {"target_id": "other-group"},
    ],
)
def test_robot_mcp_ignores_partial_target_in_active_robot_turn(
    monkeypatch,
    partial_target,
) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
        )
    )
    sent: list[tuple[str, RobotReplyTarget, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    try:
        result = server.call_tool(
            "send_message",
            {
                "text": "我是一只猪",
                "_robot_context_token": token,
                **partial_target,
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert result[0]["text"].startswith("Message sent to current robot conversation.")
    assert len(sent) == 1
    assert sent[0][0] == "robot-current"
    assert sent[0][1].target_id == "current-group"
    assert sent[0][2] == "我是一只猪"


def test_robot_mcp_broadcast_requires_context_target() -> None:
    server = RobotMCPServer()

    result = server.call_tool(
        "send_message",
        {"text": "critical alert", "broadcast": True},
    )

    assert result[0]["type"] == "text"
    assert "broadcast from prior QQ context is disabled" in result[0]["text"]


def _compress_test_context():
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    return register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
            conversation_key="group:current-group",
        )
    )


def _compress_test_memory(memory_id: str, content: str, **metadata_overrides):
    metadata = {
        "memory_type": "fact",
        "robot_id": "robot-current",
        "robot_conversation_key": "group:current-group",
        "memory_scope": "conversation",
        "robot_memory_schema_version": 3,
    }
    metadata.update(metadata_overrides)
    return {"id": memory_id, "content": content, "metadata": metadata}


def test_robot_mcp_compress_memories_merges_and_deletes(monkeypatch) -> None:
    server = RobotMCPServer()
    tools = {tool["name"]: tool for tool in server.list_tools()}
    assert "compress_memories" in tools
    assert tools["compress_memories"]["skip_memory"] is True

    existing = [
        _compress_test_memory(
            "aaaa1111-0000-4000-8000-000000000001",
            "端口是 43906",
        ),
        _compress_test_memory(
            "bbbb2222-0000-4000-8000-000000000002",
            "服务器端口其实是 43906",
        ),
    ]
    added: dict[str, object] = {}
    deleted: list[str] = []

    monkeypatch.setattr(
        vector_store,
        "get_all_memories",
        lambda *_args, **_kwargs: existing,
    )

    def fake_add_memory(**kwargs):
        added.update(kwargs)
        return "cccc3333-0000-4000-8000-000000000003"

    def fake_delete_memory(memory_id):
        deleted.append(str(memory_id))
        return True

    monkeypatch.setattr(vector_store, "add_memory", fake_add_memory)
    monkeypatch.setattr(vector_store, "delete_memory", fake_delete_memory)

    token = _compress_test_context()
    try:
        result = server.call_tool(
            "compress_memories",
            {
                "_robot_context_token": token,
                "_TermPaws_item_id": "item-1",
                "memory_ids": ["aaaa1111", "bbbb2222"],
                "content": "服务器端口是 43906",
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    text = result[0]["text"]
    assert "Compressed 2" in text
    assert "deleted 2" in text
    assert added["handler_id"] == "item-1"
    assert added["content"] == "服务器端口是 43906"
    assert added["memory_type"] == "fact"
    assert added["allow_duplicate"] is True
    metadata = added["metadata"]
    assert metadata["source"] == "qq_robot_agent_compress"
    assert metadata["robot_id"] == "robot-current"
    assert metadata["conversation_key"] == "group:current-group"
    assert metadata["robot_conversation_key"] == "group:current-group"
    assert deleted == [memory["id"] for memory in existing]


def test_robot_mcp_compress_memories_rejects_out_of_scope(monkeypatch) -> None:
    server = RobotMCPServer()
    existing = [
        _compress_test_memory(
            "aaaa1111-0000-4000-8000-000000000001",
            "端口是 43906",
        ),
        _compress_test_memory(
            "bbbb2222-0000-4000-8000-000000000002",
            "别人的专属偏好",
            memory_type="preference",
            memory_scope="speaker",
            speaker_global_key="onebot_v11:user:user-2",
        ),
    ]
    called: list[str] = []

    monkeypatch.setattr(
        vector_store,
        "get_all_memories",
        lambda *_args, **_kwargs: existing,
    )
    monkeypatch.setattr(
        vector_store,
        "add_memory",
        lambda **_kwargs: called.append("add") or "new-id",
    )
    monkeypatch.setattr(
        vector_store,
        "delete_memory",
        lambda _memory_id: called.append("delete") or True,
    )

    token = _compress_test_context()
    try:
        result = server.call_tool(
            "compress_memories",
            {
                "_robot_context_token": token,
                "_TermPaws_item_id": "item-1",
                "memory_ids": ["aaaa1111", "bbbb2222"],
                "content": "合并后的记忆",
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert "outside the current QQ scope" in result[0]["text"]
    assert called == []


def test_robot_mcp_compress_memories_reports_unmatched_ids(monkeypatch) -> None:
    server = RobotMCPServer()
    existing = [
        _compress_test_memory(
            "aaaa1111-0000-4000-8000-000000000001",
            "端口是 43906",
        ),
    ]
    called: list[str] = []

    monkeypatch.setattr(
        vector_store,
        "get_all_memories",
        lambda *_args, **_kwargs: existing,
    )
    monkeypatch.setattr(
        vector_store,
        "add_memory",
        lambda **_kwargs: called.append("add") or "new-id",
    )
    monkeypatch.setattr(
        vector_store,
        "delete_memory",
        lambda _memory_id: called.append("delete") or True,
    )

    token = _compress_test_context()
    try:
        result = server.call_tool(
            "compress_memories",
            {
                "_robot_context_token": token,
                "_TermPaws_item_id": "item-1",
                "memory_ids": ["aaaa1111", "zzzz9999"],
                "content": "合并后的记忆",
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert "not found" in result[0]["text"]
    assert "zzzz9999" in result[0]["text"]
    assert called == []


from types import SimpleNamespace


def _fake_qq_context():
    return SimpleNamespace(reply_target=SimpleNamespace(metadata={}))


def test_coalesce_allows_up_to_three_bubbles(monkeypatch) -> None:
    server = RobotMCPServer()
    messages = ["在呢", "刚卡了一下", "你说啥来着"]
    result = server._coalesce_current_context_messages(_fake_qq_context(), messages)
    assert result == messages


def test_coalesce_never_merges_bubbles() -> None:
    server = RobotMCPServer()
    messages = ["一", "二", "三", "四", "五"]
    result = server._coalesce_current_context_messages(_fake_qq_context(), messages)
    # 句数无上限：只做去重，绝不合并成一段
    assert result == ["一", "二", "三", "四", "五"]


def test_coalesce_dedupes_bubbles() -> None:
    server = RobotMCPServer()
    messages = ["在呢", "在呢", "你说啥"]
    result = server._coalesce_current_context_messages(_fake_qq_context(), messages)
    assert result == ["在呢", "你说啥"]


def test_robot_mcp_send_message_fans_out_to_explicit_targets(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[tuple[str, str, str, str]] = []

    def fake_send_message(robot_id, reply_target, text):
        sent.append((robot_id, reply_target.target_type, reply_target.target_id, text))

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "text": "服务器已开",
            "targets": [
                {"target_type": "private", "target_id": "2537134688"},
                {"target_type": "group", "target_id": "770362397"},
            ],
            "_TermPaws_user_id": "user-1",
        },
    )

    assert result[0]["type"] == "text"
    assert "sent: private:2537134688" in result[0]["text"]
    assert "sent: group:770362397" in result[0]["text"]
    assert sent == [
        ("robot-2", "private", "2537134688", "服务器已开"),
        ("robot-2", "group", "770362397", "服务器已开"),
    ]


def test_robot_mcp_send_message_targets_partial_failure(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[str] = []

    def fake_send_message(_robot_id, reply_target, _text):
        if reply_target.target_type == "private":
            raise RuntimeError("bridge offline")
        sent.append(reply_target.target_id)

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )

    result = server.call_tool(
        "send_message",
        {
            "text": "hello",
            "targets": [
                {"target_type": "private", "target_id": "2537134688"},
                {"target_type": "group", "target_id": "770362397"},
            ],
            "_TermPaws_user_id": "user-1",
        },
    )

    assert "error: private:2537134688: bridge offline" in result[0]["text"]
    assert "sent: group:770362397" in result[0]["text"]
    assert sent == ["770362397"]


def test_robot_mcp_send_message_targets_resends_identical_text(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[str] = []

    def fake_send_message(_robot_id, reply_target, _text):
        sent.append(reply_target.target_id)

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )
    args = {
        "text": "hello",
        "targets": [{"target_type": "group", "target_id": "770362397"}],
        "_TermPaws_user_id": "user-1",
    }

    first = server.call_tool("send_message", args)
    assert "sent: group:770362397" in first[0]["text"]

    second = server.call_tool("send_message", args)
    # 完全相同的内容 30s 内重发被抑制（去重流控），第二次没有实际发出
    assert sent == ["770362397"]


def test_robot_mcp_send_message_targets_rejects_conflicting_args(monkeypatch) -> None:
    server = RobotMCPServer()
    sent: list[str] = []

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        lambda robot_id, reply_target, text: sent.append(text),
    )
    monkeypatch.setattr(
        server,
        "_get_accessible_robot_id",
        lambda args, fallback_robot_id="": "robot-2",
    )
    base_args = {
        "text": "hello",
        "targets": [{"target_type": "group", "target_id": "770362397"}],
        "_TermPaws_user_id": "user-1",
    }

    for conflicting in (
        {"target_type": "group", "target_id": "123456"},
        {"reply_to": "baka"},
        {"conversation": "group:123456"},
        {"broadcast": True},
    ):
        result = server.call_tool("send_message", {**base_args, **conflicting})
        assert "targets cannot be combined with" in result[0]["text"]

    assert sent == []


def test_robot_mcp_send_message_targets_rejects_invalid_entries() -> None:
    server = RobotMCPServer()

    result = server.call_tool(
        "send_message",
        {
            "text": "hello",
            "targets": [{"target_type": "group"}],
            "_TermPaws_user_id": "user-1",
        },
    )

    assert "targets[0]" in result[0]["text"]
    assert "target_id" in result[0]["text"]


def test_robot_mcp_send_message_targets_blocked_in_active_context(monkeypatch) -> None:
    server = RobotMCPServer()
    target = RobotReplyTarget(
        target_type="group",
        target_id="current-group",
        metadata={"target": {"id": "current-group"}},
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-current",
            sender_key="onebot_v11:group:current-group:user-1",
            reply_target=target,
        )
    )
    sent: list[str] = []
    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        lambda robot_id, reply_target, text: sent.append(text),
    )
    try:
        result = server.call_tool(
            "send_message",
            {
                "text": "hello",
                "targets": [{"target_type": "group", "target_id": "770362397"}],
                "_robot_context_token": token,
            },
        )
    finally:
        unregister_robot_mcp_context(token)

    assert "active QQ-triggered context is locked to group:current-group" in result[0]["text"]
    assert sent == []
