import asyncio
import sys
import time
import types
from datetime import timedelta
from typing import Any

from sqlmodel import Session

from app.models import RobotItem
from app.plugins.robot.contracts import RobotInboundMessage, RobotReplyTarget
from app.plugins.robot.conversation_memory import robot_conversation_memory
from app.plugins.robot.platforms import (
    _event_mentions_bot,
    _event_replies_to_bot,
    _extract_conversation_metadata,
    _extract_event_mentions,
    _extract_sender_key,
    _extract_sender_metadata,
    build_inbound_message,
    send_text_with_bot,
)
from app.plugins.robot.service import RobotService, robot_service
from app.services.agent.chat_runtime import ChatResponseResult
from app.services.agent.memory.vector_store import vector_store
from tests.utils.item import create_random_item
from tests.utils.robot import create_random_robot


def test_dispatch_workers_keep_parallel_capacity_when_configured_as_one(
    monkeypatch,
) -> None:
    from app.plugins.robot import service as service_module

    started: list[str] = []

    class FakeThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            started.append(self.name)

    service = RobotService()
    monkeypatch.setattr(service_module.settings, "ROBOT_BACKEND_DISPATCH_WORKERS", 1)
    monkeypatch.setattr(service_module.threading, "Thread", FakeThread)

    service._ensure_dispatch_workers()

    assert len(started) == 2
    assert service.dispatch_queue_snapshot()["workers"] == 2


def test_dispatch_filtered_output_does_not_send_to_robot(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)

    binding = RobotItem(
        robot_id=robot.id,
        item_id=item.id,
        allow_chat=True,
        receive_filtered_output=True,
        chat_alias="alpha",
        is_default_target=True,
    )
    db.add(binding)
    db.commit()

    def fail_send():
        raise AssertionError("terminal output should not be sent to robot")

    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fail_send,
    )
    robot_service.dispatch_filtered_output(
        str(item.id),
        "line-one\nline-two",
        item_title=item.title,
    )


def test_enqueue_background_job_result_targets_original_conversation(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    captured = _capture_queued_chat(monkeypatch)
    target = RobotReplyTarget(
        target_type="private",
        target_id="2537134688",
        metadata={"mentioned_bot": True},
    )

    queued = robot_service.enqueue_background_job_result(
        robot_id=robot.id,
        item_id=item.id,
        sender_key="2537134688",
        reply_target=target,
        conversation_key="private:2537134688",
        conversation_generation=12,
        message="[Background terminal job result]\nTemurin installed",
    )

    assert queued is True
    job = captured["job"]
    assert job.robot_id == robot.id
    assert job.item_id == item.id
    assert job.sender_key == "2537134688"
    assert job.reply_target.target_id == "2537134688"
    assert job.conversation_key == "private:2537134688"
    assert job.conversation_generation == 12
    assert job.direct_reply_trigger is True
    assert job.reply_requires_awake is False
    assert "Temurin installed" in job.message


def test_background_job_reply_is_visible_in_pending_controller_snapshot(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    _capture_queued_chat(monkeypatch)
    target = RobotReplyTarget(
        target_type="group",
        target_id="770362397",
        metadata={
            "mentioned_bot": True,
            "sender": {
                "user_id": "2537134688",
                "display_name": "Ac国常务腐管理",
            },
        },
    )

    pending_id = robot_service.register_background_job_reply(
        robot_id=robot.id,
        item_id=item.id,
        sender_key="onebot_v11:group:770362397:2537134688",
        reply_target=target,
        conversation_key="group:770362397",
        conversation_generation=3,
        command="apt-get install -y temurin-17-jdk",
    )

    assert pending_id
    snapshots = robot_service.conversation_controller_snapshots(
        {robot.id},
        {item.id},
    )
    assert len(snapshots) == 1
    pending_messages = snapshots[0]["pending_messages"]
    assert len(pending_messages) == 1
    pending = pending_messages[0]
    assert pending["trigger_reason"] == "background_job"
    assert pending["sender_label"] == "Ac国常务腐管理 (2537134688)"
    assert pending["item_id"] == str(item.id)
    assert pending["pending_reply_id"] == pending_id
    assert "temurin-17-jdk" in pending["message_preview"]

    queued = robot_service.enqueue_background_job_result(
        robot_id=robot.id,
        item_id=item.id,
        sender_key="onebot_v11:group:770362397:2537134688",
        reply_target=target,
        conversation_key="group:770362397",
        conversation_generation=3,
        message="[Background terminal job result]\nTemurin installed",
        pending_reply_id=pending_id,
        reply_ticket_id="ticket-java",
    )

    assert queued is True
    snapshots = robot_service.conversation_controller_snapshots(
        {robot.id},
        {item.id},
    )
    assert snapshots[0]["pending_messages"][0]["pending_reply_id"] == pending_id

    captured = _capture_queued_chat(monkeypatch)
    robot_service.enqueue_background_job_result(
        robot_id=robot.id,
        item_id=item.id,
        sender_key="onebot_v11:group:770362397:2537134688",
        reply_target=target,
        conversation_key="group:770362397",
        conversation_generation=3,
        message="[Background terminal job result]\nTemurin installed",
        pending_reply_id=pending_id,
        reply_ticket_id="ticket-java",
    )
    job = captured["job"]
    assert job.reply_ticket_id == "ticket-java"

    async def fake_chat_with_item(**kwargs):
        assert kwargs["reply_ticket_id"] == "ticket-java"
        return ChatResponseResult(content="", robot_message_sent=True)

    monkeypatch.setattr(robot_service, "_chat_with_item", fake_chat_with_item)
    robot_service._process_chat_job(job)

    delivered_snapshots = robot_service.conversation_controller_snapshots(
        {robot.id},
        {item.id},
    )
    assert (
        delivered_snapshots == []
        or delivered_snapshots[0]["pending_messages"] == []
    )


TEST_BOT_SELF_ID = "10001"


def _message(
    text: str,
    *,
    sender_key: str = "onebot_v11:group:g1:u1",
    target: dict[str, Any] | None = None,
    sender: dict[str, Any] | None = None,
    conversation: dict[str, Any] | None = None,
    mentions: list[dict[str, Any]] | None = None,
    mentioned_bot: bool = False,
    replied_to_bot: bool = False,
    bot_self_ids: list[str] | None = None,
    reply: dict[str, Any] | None = None,
) -> RobotInboundMessage:
    if mentioned_bot and mentions is None:
        mentions = [{"qq": TEST_BOT_SELF_ID, "name": "TestBot"}]
    if mentioned_bot and bot_self_ids is None:
        bot_self_ids = [TEST_BOT_SELF_ID]
    metadata: dict[str, Any] = {}
    if target is not None:
        metadata["target"] = target
    if sender is not None:
        metadata["sender"] = sender
    if conversation is not None:
        metadata["conversation"] = conversation
    if mentions is not None:
        metadata["mentions"] = mentions
    if mentioned_bot:
        metadata["mentioned_bot"] = True
    if replied_to_bot:
        metadata["replied_to_bot"] = True
    if bot_self_ids is not None:
        metadata["bot_self_ids"] = bot_self_ids
    if reply is not None:
        metadata["reply"] = reply

    return RobotInboundMessage(
        sender_key=sender_key,
        text=text,
        reply_target=RobotReplyTarget(
            target_type="universal",
            target_id="g1",
            metadata=metadata,
        ),
    )


def _capture_queued_chat(monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_enqueue_chat_job(job) -> bool:
        captured["job"] = job
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    return captured


def _simulate_worker_pickup(monkeypatch, job) -> None:
    active_jobs = dict(robot_service._active_dispatch_jobs)
    tracking_key = job.job_id or "test-active-job"
    active_jobs[tracking_key] = (time.monotonic(), job)
    monkeypatch.setattr(robot_service, "_active_dispatch_jobs", active_jobs)


def _process_captured_chat_job(monkeypatch, job) -> list[str]:
    messages: list[str] = []

    async def fake_chat_with_item(**kwargs):
        messages.append(str(kwargs["message"]))
        return ChatResponseResult(content="", robot_message_sent=False)

    monkeypatch.setattr(robot_service, "_chat_with_item", fake_chat_with_item)
    robot_service._process_chat_job(job)
    return messages


class _FakeBot:
    self_id = "10001"


class _FakeOneBotAdapter:
    connections = {"10001": object()}

    def get_name(self) -> str:
        return "OneBot V11"


class _FakeOneBotBot:
    self_id = "10001"

    def __init__(self) -> None:
        self.adapter = _FakeOneBotAdapter()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_api(self, action: str, **kwargs: Any) -> None:
        self.calls.append((action, kwargs))


class _FakeSegment:
    def __init__(self, segment_type: str, data: dict[str, Any]) -> None:
        self.type = segment_type
        self.data = data


class _FakeEvent:
    def __init__(
        self,
        message: list[Any],
        *,
        to_me: bool = False,
        raw_message: str = "",
        reply: Any | None = None,
        sender: dict[str, Any] | None = None,
        user_id: str = "",
        message_type: str = "",
        group_id: str = "",
        guild_id: str = "",
        channel_id: str = "",
    ) -> None:
        self._message = message
        self.to_me = to_me
        self.raw_message = raw_message
        self.reply = reply
        self.sender = sender or {}
        self.user_id = user_id
        self.message_type = message_type
        self.group_id = group_id
        self.guild_id = guild_id
        self.channel_id = channel_id

    def get_message(self) -> list[Any]:
        return self._message

    def get_user_id(self) -> str:
        return self.user_id

    def get_plaintext(self) -> str:
        texts: list[str] = []
        for segment in self._message:
            if getattr(segment, "type", "") == "text":
                texts.append(str(getattr(segment, "data", {}).get("text") or ""))
        return "".join(texts)


class _FakeAlconnaTarget:
    id = "123456"
    source = None

    def dump(self) -> dict[str, Any]:
        return {"id": self.id}


def _mock_alconna(
    monkeypatch,
    *,
    message_id: str = "message-1",
    target: Any | None = None,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "nonebot_plugin_alconna",
        types.SimpleNamespace(
            get_message_id=lambda event, bot: message_id,
            get_target=lambda event, bot: target or _FakeAlconnaTarget(),
        ),
    )


def test_send_text_with_bot_uses_onebot_group_and_private_actions() -> None:
    bot = _FakeOneBotBot()

    asyncio.run(
        send_text_with_bot(
            bot,
            RobotReplyTarget(target_type="group", target_id="123456"),
            " 群消息 ",
        )
    )
    asyncio.run(
        send_text_with_bot(
            bot,
            RobotReplyTarget(target_type="private", target_id="654321"),
            "私信消息",
        )
    )

    assert bot.calls == [
        (
            "send_group_msg",
            {
                "group_id": 123456,
                "message": "群消息",
            },
        ),
        (
            "send_private_msg",
            {
                "user_id": 654321,
                "message": "私信消息",
            },
        ),
    ]


def test_send_text_with_bot_converts_onebot_group_cq_at_to_segment() -> None:
    bot = _FakeOneBotBot()

    asyncio.run(
        send_text_with_bot(
            bot,
            RobotReplyTarget(target_type="group", target_id="123456"),
            "[CQ:at,qq=3385417251] hello",
        )
    )

    assert bot.calls == [
        (
            "send_group_msg",
            {
                "group_id": 123456,
                "message": [
                    {"type": "at", "data": {"qq": "3385417251"}},
                    {"type": "text", "data": {"text": " hello"}},
                ],
            },
        )
    ]


def test_send_text_with_bot_converts_onebot_metadata_group_cq_at_to_segment() -> None:
    bot = _FakeOneBotBot()

    asyncio.run(
        send_text_with_bot(
            bot,
            RobotReplyTarget(
                target_type="universal",
                target_id="universal-target",
                metadata={
                    "target": {
                        "id": "universal-target",
                        "message_type": "group",
                        "group_id": "123456",
                    }
                },
            ),
            "reply [CQ:at,qq=3385417251]",
        )
    )

    assert bot.calls == [
        (
            "send_group_msg",
            {
                "group_id": 123456,
                "message": [
                    {"type": "text", "data": {"text": "reply "}},
                    {"type": "at", "data": {"qq": "3385417251"}},
                ],
            },
        )
    ]


def test_event_mentions_bot_detects_onebot_at_segment() -> None:
    event = _FakeEvent(
        [
            _FakeSegment("at", {"qq": "10001"}),
            _FakeSegment("text", {"text": " hello"}),
        ]
    )

    assert _event_mentions_bot(_FakeBot(), event) is True


def test_event_mentions_bot_ignores_other_at_segment() -> None:
    event = _FakeEvent([_FakeSegment("at", {"qq": "10002"})])

    assert _event_mentions_bot(_FakeBot(), event) is False


def test_event_mentions_bot_ignores_other_at_even_when_to_me_flag_is_set() -> None:
    event = _FakeEvent([_FakeSegment("at", {"qq": "10002"})], to_me=True)

    assert _event_mentions_bot(_FakeBot(), event) is False


def test_event_mentions_bot_ignores_group_to_me_without_self_at() -> None:
    event = _FakeEvent(
        [_FakeSegment("text", {"text": "hello"})],
        to_me=True,
        message_type="group",
        raw_message="[CQ:at,qq=10002] hello",
    )

    assert _event_mentions_bot(_FakeBot(), event) is False


def test_service_direct_trigger_requires_bot_self_id_mention() -> None:
    message = _message(
        "[CQ:at,qq=20002] hello",
        sender_key="onebot_v11:group:g1:u1",
        target={"id": "g1", "message_type": "group", "group_id": "g1"},
        mentioned_bot=True,
        bot_self_ids=["10001"],
        mentions=[{"id": "20002", "qq": "20002"}],
    )

    assert robot_service._message_directly_addresses_bot(message) is False


def test_service_direct_trigger_accepts_bot_self_id_mention() -> None:
    message = _message(
        "[CQ:at,qq=10001] hello",
        sender_key="onebot_v11:group:g1:u1",
        target={"id": "g1", "message_type": "group", "group_id": "g1"},
        mentioned_bot=True,
        bot_self_ids=["10001"],
        mentions=[{"id": "10001", "qq": "10001"}],
    )

    assert robot_service._message_directly_addresses_bot(message) is True

def test_extract_event_mentions_reads_onebot_at_segments() -> None:
    event = _FakeEvent(
        [
            _FakeSegment(
                "at",
                {"qq": "1512220570", "name": "她喜欢我才钓着我"},
            ),
            _FakeSegment("text", {"text": " 这个人是小男娘"}),
        ]
    )

    assert _extract_event_mentions(event) == [
        {
            "id": "1512220570",
            "qq": "1512220570",
            "name": "她喜欢我才钓着我",
        }
    ]


def test_extract_event_mentions_falls_back_to_raw_cq_at() -> None:
    event = _FakeEvent(
        [],
        raw_message="[CQ:at,qq=1512220570,name=她喜欢我才钓着我] 这个人是小男娘",
    )

    assert _extract_event_mentions(event) == [
        {
            "id": "1512220570",
            "qq": "1512220570",
            "name": "她喜欢我才钓着我",
        }
    ]


def test_event_mentions_bot_detects_to_me_flag() -> None:
    event = _FakeEvent([], to_me=True)

    assert _event_mentions_bot(_FakeBot(), event) is True


def test_build_inbound_message_keeps_empty_bot_mention(monkeypatch) -> None:
    _mock_alconna(monkeypatch)
    event = _FakeEvent(
        [_FakeSegment("at", {"qq": "10001"})],
        user_id="10002",
        message_type="group",
        group_id="123456",
    )

    inbound = build_inbound_message("onebot_v11", _FakeBot(), event)

    assert inbound is not None
    assert inbound.text == "[CQ:at,qq=10001]"
    assert inbound.reply_target.metadata["mentioned_bot"] is True
    assert inbound.reply_target.metadata["bot_self_ids"] == ["10001"]
    assert inbound.reply_target.metadata["message"]["segments"] == [
        {"type": "at", "data": {"qq": "10001"}}
    ]
    assert inbound.reply_target.metadata["conversation"]["id"] == "123456"


def test_build_inbound_message_preserves_onebot_at_in_text(monkeypatch) -> None:
    _mock_alconna(monkeypatch)
    event = _FakeEvent(
        [
            _FakeSegment("at", {"qq": "20002"}),
            _FakeSegment("text", {"text": " hello"}),
        ],
        user_id="10002",
        message_type="group",
        group_id="123456",
    )

    inbound = build_inbound_message("onebot_v11", _FakeBot(), event)

    assert inbound is not None
    assert inbound.text == "[CQ:at,qq=20002] hello"
    assert inbound.reply_target.metadata["mentions"] == [
        {"id": "20002", "qq": "20002"}
    ]
    assert inbound.reply_target.metadata["message"]["plain_text"] == "hello"
    assert inbound.reply_target.metadata["message"]["segments"] == [
        {"type": "at", "data": {"qq": "20002"}},
        {"type": "text", "data": {"text": " hello"}},
    ]


def test_build_inbound_message_drops_empty_non_mention(monkeypatch) -> None:
    _mock_alconna(monkeypatch)
    event = _FakeEvent(
        [],
        user_id="10002",
        message_type="group",
        group_id="123456",
    )

    assert build_inbound_message("onebot_v11", _FakeBot(), event) is None


def test_event_replies_to_bot_detects_reply_sender() -> None:
    event = _FakeEvent([], reply={"sender": {"user_id": "10001"}})

    assert _event_replies_to_bot(_FakeBot(), event) is True


def test_event_replies_to_bot_ignores_reply_event_self_id() -> None:
    event = _FakeEvent([], reply={"self_id": "10001", "message_id": "42"})

    assert _event_replies_to_bot(_FakeBot(), event) is False


def test_event_replies_to_bot_detects_reply_segment_sender_qq() -> None:
    event = _FakeEvent([_FakeSegment("reply", {"qq": "10001"})])

    assert _event_replies_to_bot(_FakeBot(), event) is True


def test_event_replies_to_bot_ignores_other_sender() -> None:
    event = _FakeEvent([_FakeSegment("reply", {"user_id": "10002"})])

    assert _event_replies_to_bot(_FakeBot(), event) is False


def test_build_inbound_message_keeps_replied_message_context(monkeypatch) -> None:
    _mock_alconna(monkeypatch, message_id="message-2")
    event = _FakeEvent(
        [
            _FakeSegment("reply", {"id": "message-1", "qq": "10001"}),
            _FakeSegment("text", {"text": "is it running?"}),
        ],
        user_id="10002",
        message_type="group",
        group_id="123456",
        reply={
            "message_id": "message-1",
            "sender": {"user_id": "10001", "nickname": "Bot"},
            "raw_message": "run.sh is ready, start it?",
        },
    )

    inbound = build_inbound_message("onebot_v11", _FakeBot(), event)

    assert inbound is not None
    assert inbound.reply_target.metadata["replied_to_bot"] is True
    assert inbound.reply_target.metadata["reply"] == {
        "message_id": "message-1",
        "source": "message-1",
        "text": "run.sh is ready, start it?",
        "sender": {
            "user_id": "10001",
            "display_name": "Bot",
            "nickname": "Bot",
        },
        "segment": {"id": "message-1", "qq": "10001"},
    }


def test_extract_sender_metadata_prefers_group_card() -> None:
    event = _FakeEvent(
        [],
        user_id="10002",
        message_type="group",
        sender={
            "user_id": 10002,
            "nickname": "Nick",
            "card": "Alice",
            "role": "admin",
        },
    )

    assert _extract_sender_metadata("onebot_v11", event) == {
        "platform": "onebot_v11",
        "user_id": "10002",
        "display_name": "Alice",
        "nickname": "Nick",
        "card": "Alice",
        "role": "admin",
        "message_type": "group",
    }


def test_agent_message_context_prefix_includes_mentioned_targets() -> None:
    message = _message(
        "这个人是小男娘",
        sender={
            "user_id": "2537134688",
            "display_name": "只有白龙马知道唐三藏动了情",
        },
        conversation={"type": "group", "id": "369040885"},
        mentions=[
            {
                "id": "1512220570",
                "qq": "1512220570",
                "name": "她喜欢我才钓着我",
            }
        ],
    )

    prefix = robot_service._agent_message_context_prefix(
        message,
        trigger_reason="active_chat_window",
    )

    assert "mentions=她喜欢我才钓着我 (1512220570)" in prefix


def test_agent_message_context_marks_bot_self_mention_for_agent() -> None:
    message = _message(
        "[CQ:at,qq=10001] hello",
        sender={"user_id": "u1", "display_name": "Alice"},
        conversation={"type": "group", "id": "g1"},
        mentions=[{"id": "10001", "qq": "10001", "name": "Bot"}],
        mentioned_bot=True,
        bot_self_ids=["10001"],
    )

    text = robot_service._agent_message_with_context(
        message,
        message.text,
        trigger_reason="mention_bot",
    )

    assert "mentions=Bot (10001) (you)" in text
    assert "[Robot identity; background only]" in text
    assert "- self_id: 10001 (this QQ id is you, the bot)" in text
    assert "@/回复指向上述 self_id 或名字就是在叫你" in text
    assert "[Current QQ sender: Alice (u1);" in text
    assert "都指这个人" in text


def test_agent_message_context_uses_bot_qq_name_for_self_mention() -> None:
    message = _message(
        "[CQ:at,qq=10001] hello",
        sender={"user_id": "u1", "display_name": "Alice"},
        conversation={"type": "group", "id": "g1"},
        mentions=[{"id": "10001", "qq": "10001"}],
        mentioned_bot=True,
        bot_self_ids=["10001"],
    )
    message.reply_target.metadata["bot_identity"] = {
        "self_ids": ["10001"],
        "display_name": "柴郡一号机",
        "aliases": ["柴郡一号机", "唯"],
    }

    text = robot_service._agent_message_with_context(
        message,
        message.text,
        trigger_reason="mention_bot",
    )

    assert "mentions=柴郡一号机 (10001) (you)" in text
    assert "- QQ display name: 柴郡一号机 (this name is you)" in text
    assert "- known QQ names: 柴郡一号机, 唯" in text


def test_robot_identity_falls_back_to_configured_name_and_self_id(db: Session) -> None:
    robot = create_random_robot(db)
    robot.name = "柴郡一号机"
    robot.config = {
        "credentials": {"self_id": "2900669542"},
        "options": {},
    }
    message = _message(
        "hello",
        sender={"user_id": "2537134688", "display_name": "FLY"},
        conversation={"type": "private", "id": "2537134688"},
    )

    robot_service._enrich_message_bot_identity(robot, message)

    assert message.reply_target.metadata["bot_self_ids"] == ["2900669542"]
    assert message.reply_target.metadata["bot_identity"] == {
        "self_ids": ["2900669542"],
        "display_name": "柴郡一号机",
        "aliases": ["柴郡一号机"],
    }


def test_agent_message_identity_question_is_anchored_to_current_sender() -> None:
    message = _message(
        "我是谁",
        sender={"user_id": "20002", "display_name": "EX_GuguX"},
        conversation={"type": "group", "id": "g1"},
        mentions=[{"id": "10001", "qq": "10001", "name": "Bot"}],
        mentioned_bot=True,
        bot_self_ids=["10001"],
    )

    text = robot_service._agent_message_with_context(
        message,
        message.text,
        trigger_reason="mention_bot",
    )

    assert "[Current QQ sender: EX_GuguX (20002);" in text
    assert "都指这个人" in text
    assert text.endswith("[Current QQ message]\n我是谁")


def test_agent_message_context_includes_replied_message_reference() -> None:
    message = _message(
        "is it running?",
        sender={"user_id": "u1", "display_name": "Alice"},
        conversation={"type": "group", "id": "g1"},
        replied_to_bot=True,
        bot_self_ids=["10001"],
        reply={
            "message_id": "message-1",
            "text": "run.sh is ready, start it?",
            "sender": {"user_id": "10001", "display_name": "Bot"},
        },
    )

    text = robot_service._agent_message_with_context(
        message,
        message.text,
        trigger_reason="reply_to_bot",
    )

    assert "[Replied QQ message; background only" in text
    assert "- message_id: message-1" in text
    assert "- sender: Bot (10001)" in text
    assert "- text: run.sh is ready, start it?" in text
    assert "[Current QQ message]\nis it running?" in text


def test_robot_reply_context_summary_marks_reply_to_self() -> None:
    from app.plugins.robot.mcp.context import build_robot_reply_context_summary

    message = _message(
        "continue",
        sender={"user_id": "u1", "display_name": "Alice"},
        conversation={"type": "group", "id": "g1"},
        replied_to_bot=True,
        bot_self_ids=["10001"],
    )
    message.reply_target.metadata["bot_identity"] = {
        "self_ids": ["10001"],
        "display_name": "柴郡一号机",
        "aliases": ["柴郡一号机"],
    }

    summary = build_robot_reply_context_summary(message.reply_target, message.sender_key)

    assert "- bot_self_id: 10001 (this QQ id is you, the bot)" in summary
    assert "- bot_display_name: 柴郡一号机 (this QQ name is you)" in summary
    assert "- addressed_to_bot: true" in summary
    assert "- direct_reason: reply_to_bot" in summary
    assert "- mentioned_self: false" in summary
    assert "- replied_to_self: true" in summary
    assert "QQ mentions/replies to this self_id are addressing you" in summary
    assert "'我/我的/我是谁' refer to Alice (u1)" in summary
    assert "Never answer with a tautology" in summary
    assert "reference rule" in summary
    assert "pronouns" in summary


def test_onebot_conversation_metadata_uses_group_id() -> None:
    event = _FakeEvent(
        [],
        user_id="10002",
        message_type="group",
        group_id="123456",
    )
    target_data = {"id": "universal-target"}

    conversation = _extract_conversation_metadata("onebot_v11", event, target_data)

    assert conversation == {
        "platform": "onebot_v11",
        "type": "group",
        "id": "123456",
        "target_type": "group",
        "target_id": "123456",
        "group_id": "123456",
        "user_id": "10002",
        "message_type": "group",
    }
    assert (
        _extract_sender_key(
            "onebot_v11",
            event,
            target_data,
            conversation_data=conversation,
        )
        == "onebot_v11:group:123456:10002"
    )


def test_onebot_conversation_metadata_uses_private_user_id() -> None:
    event = _FakeEvent(
        [],
        user_id="654321",
        message_type="private",
    )
    target_data = {"id": "universal-target", "private": True}

    conversation = _extract_conversation_metadata("onebot_v11", event, target_data)

    assert conversation == {
        "platform": "onebot_v11",
        "type": "private",
        "id": "654321",
        "target_type": "private",
        "target_id": "654321",
        "user_id": "654321",
        "message_type": "private",
    }
    assert (
        _extract_sender_key(
            "onebot_v11",
            event,
            target_data,
            conversation_data=conversation,
        )
        == "onebot_v11:private:654321"
    )


def test_plain_group_message_is_ignored_by_default(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    def fail_enqueue_chat_job(_job):
        raise AssertionError("plain group chatter should remain asleep by default")

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fail_enqueue_chat_job)

    memory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        robot_conversation_memory,
        "append_user_message",
        lambda *args, **kwargs: memory_calls.append((args, kwargs)),
    )

    response = robot_service.handle_inbound_message(db, robot, _message("hello"))

    assert response.success is True
    assert response.ignored is True
    assert response.reason == "conversation_sleeping"
    assert response.reply_chunks == []
    assert len(memory_calls) == 1
    assert str(memory_calls[0][0][0]) == str(robot.id)
    assert memory_calls[0][0][1] == "group:g1"
    assert memory_calls[0][0][2] == "hello"
    assert memory_calls[0][1]["sender"] == "onebot_v11:group:g1:u1"


def test_mentioned_group_message_routes_to_default_item_agent(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)
    memory_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        robot_conversation_memory,
        "append_user_message",
        lambda *args, **kwargs: memory_calls.append((args, kwargs)),
    )


    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello", mentioned_bot=True),
    )

    assert response.success is True
    assert response.item_id == str(item.id)
    assert response.route_key == "alpha"
    assert captured["job"].message == "hello"
    assert captured["job"].conversation_key == "group:g1"
    assert captured["job"].sender_key == "onebot_v11:group:g1:u1"
    assert isinstance(captured["job"].reply_target, RobotReplyTarget)
    assert response.reply_chunks == []
    assert len(memory_calls) == 1
    assert str(memory_calls[0][0][0]) == str(robot.id)
    assert memory_calls[0][0][1] == "group:g1"
    assert memory_calls[0][0][2] == "hello"
    assert memory_calls[0][1]["sender"] == "onebot_v11:group:g1:u1"


def test_robot_dispatch_queues_before_slow_memory_and_context_work(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    def fail_impression(**_kwargs):
        raise AssertionError("impression card must run in the worker, not dispatch")

    monkeypatch.setattr(robot_service, "_conversation_impression_card", fail_impression)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello", mentioned_bot=True),
    )

    assert response.success is True
    assert response.reason == "queued"
    assert captured["job"].message == "hello"
    assert captured["job"].message_text == "hello"
    assert captured["job"].trigger_reason == "mention_bot"
    assert captured["job"].inbound_message is not None


def test_robot_message_includes_current_conversation_impression_card(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    def fake_get_all_memories(item_id, memory_type=None):
        assert item_id == str(item.id)
        if memory_type not in {None, "preference"}:
            return []
        return [
            {
                "id": "current",
                "content": "用户偏好：Alice 喜欢短回复",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": str(robot.id),
                    "robot_conversation_key": "group:g1",
                    "updated_at": "2026-07-07T00:00:00",
                },
            },
            {
                "id": "same-speaker-other-group",
                "content": "用户偏好：Alice 以后叫她主人",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": str(robot.id),
                    "robot_conversation_key": "group:g2",
                    "speaker_global_key": "onebot_v11:user:u1",
                    "memory_scope": "speaker",
                    "updated_at": "2026-07-07T00:00:00",
                },
            },
            {
                "id": "other-speaker-same-group",
                "content": "用户偏好：Bob 以后叫他大主人",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": str(robot.id),
                    "robot_conversation_key": "group:g1",
                    "speaker_global_key": "onebot_v11:user:u2",
                    "memory_scope": "speaker",
                    "updated_at": "2026-07-07T00:00:01",
                },
            },
            {
                "id": "other-group",
                "content": "other group secret",
                "metadata": {
                    "memory_type": "preference",
                    "robot_id": str(robot.id),
                    "robot_conversation_key": "group:g2",
                    "updated_at": "2026-07-07T00:00:00",
                },
            },
        ]

    monkeypatch.setattr(vector_store, "get_all_memories", fake_get_all_memories)
    monkeypatch.setattr(vector_store, "search_memories", lambda **_kwargs: [])

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello",
            target={"id": "g1"},
            sender={"user_id": "u1", "display_name": "Alice"},
            mentioned_bot=True,
        ),
    )

    assert response.success is True
    messages = _process_captured_chat_job(monkeypatch, captured["job"])
    text = messages[0]
    assert "Current QQ conversation impression card" in text
    assert "Alice 喜欢短回复" in text
    assert "Alice 以后叫她主人" in text
    assert text.index("Alice 以后叫她主人") < text.index("Alice 喜欢短回复")
    assert "Bob 以后叫他大主人" not in text
    assert "other group secret" not in text
    assert text.endswith("[Current QQ message]\nhello")


def test_robot_impression_card_prioritizes_query_relevant_memory(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    scoped_metadata = {
        "robot_id": str(robot.id),
        "robot_conversation_key": "group:g1",
    }
    stable = {
        "id": "stable-preference",
        "content": "用户偏好：回答保持简洁",
        "metadata": {**scoped_metadata, "memory_type": "preference"},
    }
    relevant = {
        "id": "relevant-port",
        "content": "Minecraft 服务端口是 43906",
        "metadata": {**scoped_metadata, "memory_type": "fact"},
        "distance": 0.1,
    }
    unrelated = {
        "id": "unrelated-recent",
        "content": "最近修改了机器人颜色",
        "metadata": {
            **scoped_metadata,
            "memory_type": "fact",
            "updated_at": "2026-07-16T12:00:00",
        },
    }
    monkeypatch.setattr(
        vector_store,
        "get_all_memories",
        lambda *_args, **_kwargs: [stable, unrelated, relevant],
    )
    monkeypatch.setattr(
        vector_store,
        "search_memories",
        lambda **_kwargs: [relevant],
    )

    card = robot_service._conversation_impression_card(
        item_id=item.id,
        robot=robot,
        conversation_key="group:g1",
        sender_key="onebot_v11:user:u1",
        query="服务器端口是多少",
    )

    assert "回答保持简洁" in card
    assert "服务端口是 43906" in card
    assert card.index("服务端口是 43906") < card.index("最近修改了机器人颜色")


def test_robot_impression_card_relation_query_keeps_old_fact_and_drops_question(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    scoped_metadata = {
        "robot_id": str(robot.id),
        "robot_conversation_key": "group:770362397",
        "robot_memory_schema_version": 3,
    }
    question = {
        "id": "new-question",
        "content": "New+7 (2206406352): 猫娘是谁",
        "metadata": {
            **scoped_metadata,
            "memory_type": "fact",
            "memory_scope": "conversation",
        },
        "distance": 0.01,
    }
    old_fact = {
        "id": "old-fact",
        "content": "月影寒波（简称汉堡）是猫娘",
        "metadata": {
            **scoped_metadata,
            "memory_type": "fact",
            "memory_scope": "speaker",
            "speaker_global_key": "onebot_v11:user:162221240",
        },
        "distance": 0.12,
    }
    candy = {
        "id": "candy",
        "content": "和煦的糖果风 (641681910): 我是猫娘",
        "metadata": {
            **scoped_metadata,
            "memory_type": "fact",
            "memory_scope": "speaker",
            "speaker_global_key": "onebot_v11:user:641681910",
        },
        "distance": 0.08,
    }
    monkeypatch.setattr(
        vector_store,
        "get_all_memories",
        lambda *_args, **_kwargs: [question, old_fact, candy],
    )
    monkeypatch.setattr(
        vector_store,
        "search_memories",
        lambda **_kwargs: [question, candy, old_fact],
    )

    card = robot_service._conversation_impression_card(
        item_id=item.id,
        robot=robot,
        conversation_key="group:770362397",
        sender_key="onebot_v11:group:770362397:2537134688",
        query="还有谁是猫娘",
    )

    assert "猫娘是谁" not in card
    assert "月影寒波（简称汉堡）是猫娘" in card
    assert "和煦的糖果风" in card


def test_robot_message_passes_sender_prefix_to_agent(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello",
            target={"id": "g1"},
            sender={
                "user_id": "u1",
                "display_name": "Alice",
            },
            mentioned_bot=True,
        ),
    )

    assert response.success is True
    text = captured["job"].message
    assert text.startswith(
        "[Robot message; conversation=group:g1; trigger=mention_bot; sender=Alice (u1); mentions=TestBot (10001) (you)]\n"
    )
    assert "[Recent QQ live context; background only" not in text
    assert "Alice (u1): hello" not in text
    assert text.endswith("[Current QQ message]\nhello")
    assert response.reply_chunks == []


def test_private_robot_message_passes_context_stamp_to_agent(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello",
            sender_key="onebot_v11:private:u1",
            target={"id": "u1", "private": True},
            sender={
                "user_id": "u1",
                "display_name": "Alice",
            },
        ),
    )

    assert response.success is True
    text = captured["job"].message
    assert text.startswith(
        "[Robot message; conversation=private:u1; trigger=private_chat; sender=Alice (u1)]\n"
    )
    assert "[Recent QQ live context; background only" not in text
    assert "Alice (u1): hello" not in text
    assert text.endswith("[Current QQ message]\nhello")
    assert captured["job"].direct_reply_trigger is True
    assert captured["job"].reply_requires_awake is True
    assert captured["job"].conversation_generation > 0
    assert response.reply_chunks == []


def test_robot_message_includes_recent_live_context_without_current_duplicate(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    monkeypatch.setattr(
        robot_conversation_memory,
        "read_recent",
        lambda *_args, **_kwargs: (
            "[2026-07-10T16:32:32+00:00] user FLY (2537134688): 换国内源吧\n"
            "[2026-07-10T16:33:33+00:00] user FLY (2537134688): 换好了吗"
        ),
    )

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "换好了吗",
            sender_key="onebot_v11:private:2537134688",
            target={"id": "2537134688", "private": True},
            sender={
                "user_id": "2537134688",
                "display_name": "FLY",
            },
        ),
    )

    assert response.success is True
    messages = _process_captured_chat_job(monkeypatch, captured["job"])
    text = messages[0]
    assert "[Recent QQ live context; latest" in text
    assert "换国内源吧" in text
    assert text.count("换好了吗") == 1
    assert text.endswith("[Current QQ message]\n换好了吗")


def test_recent_live_context_uses_progressive_budget(
    db: Session,
    monkeypatch,
) -> None:
    robot = create_random_robot(db)
    calls: list[int] = []

    def fake_read_recent(*_args, **kwargs):
        calls.append(int(kwargs.get("lines") or 0))
        return "[2026-07-10T16:32:32+00:00] user FLY (2537134688): 上一条"

    monkeypatch.setattr(robot_conversation_memory, "read_recent", fake_read_recent)

    clear_direct_card = robot_service._recent_live_context_card(
        robot=robot,
        conversation_key="private:2537134688",
        trigger_reason="private_chat",
        current_message_text="hello",
    )
    expanded_card = robot_service._recent_live_context_card(
        robot=robot,
        conversation_key="private:2537134688",
        trigger_reason="private_chat",
        current_message_text="为什么",
    )
    active_card = robot_service._recent_live_context_card(
        robot=robot,
        conversation_key="group:g1",
        trigger_reason="active_chat_window",
        current_message_text="今天群里在聊服务器配置",
    )

    # Directly-addressed turns always get the full window; the message text
    # is never pattern-matched to shrink the budget.
    assert "latest 12 line(s)" in clear_direct_card
    assert "上一条" in clear_direct_card
    assert "latest 12 line(s)" in expanded_card
    assert "上一条" in expanded_card
    assert "latest 6 line(s)" in active_card
    assert calls == [48, 48, 24]


def test_recent_live_context_keeps_user_lines_when_bot_replies_repeat(
    db: Session,
    monkeypatch,
) -> None:
    robot = create_random_robot(db)
    repeated_reply = "我刚才没对齐，我会改回来"
    memory = "\n".join(
        [
            "[2026-07-17T09:33:30+00:00] user FLY (2537134688): 你怎么不说话了",
            *[
                f"[2026-07-17T09:33:{31 + index:02d}+00:00] assistant: {repeated_reply}"
                for index in range(8)
            ],
            "[2026-07-17T09:33:42+00:00] user FLY (2537134688): 回冬啊",
        ]
    )
    monkeypatch.setattr(
        robot_conversation_memory,
        "read_recent",
        lambda *_args, **_kwargs: memory,
    )

    card = robot_service._recent_live_context_card(
        robot=robot,
        conversation_key="group:770362397",
        trigger_reason="active_chat_window",
        current_message_text="回冬啊",
    )

    # Active-window turns always get the small sample budget; if the sample
    # is ambiguous the agent escalates via read_conversation_memory instead
    # of the budget being guessed from the message text.
    assert "latest 6 line(s)" in card
    assert "你怎么不说话了" in card
    assert card.count(repeated_reply) == 1
    assert "回冬啊" not in card
    assert "ask one brief clarification" in card


def test_prepare_queued_chat_message_excludes_later_pending_messages(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    bot_self_id = str(robot.config["credentials"]["self_id"])
    target = RobotReplyTarget(
        target_type="group",
        target_id="g1",
        metadata={
            "target": {"id": "g1"},
            "conversation": {"type": "group", "id": "g1"},
            "sender": {"user_id": "u1", "display_name": "Alice"},
            "mentions": [{"qq": bot_self_id}],
            "mentioned_bot": True,
            "bot_self_ids": [bot_self_id],
        },
    )

    response = robot_service.handle_inbound_message(
        db,
        robot,
        RobotInboundMessage(
            sender_key="onebot_v11:group:g1:u1",
            text="先说第一件事",
            reply_target=target.model_copy(deep=True),
        ),
    )
    assert response.ignored is False
    job = captured["job"]
    robot_service._record_pending_chat_input(
        robot=robot,
        conversation_key=job.conversation_key,
        item_id=item.id,
        route_key="alpha",
        message_text="回冬啊",
        sender_key="onebot_v11:group:g1:u1",
        sender_label="Alice (u1)",
        trigger_reason="active_chat_window",
        reply_target=target,
    )
    monkeypatch.setattr(
        robot_conversation_memory,
        "read_recent",
        lambda *_args, **_kwargs: (
            "[2026-07-17T09:33:30+00:00] user Alice (u1): 你怎么不说话了\n"
            "[2026-07-17T09:33:31+00:00] user Alice (u1): 先说第一件事\n"
            "[2026-07-17T09:33:32+00:00] user Alice (u1): 回冬啊"
        ),
    )

    prepared = robot_service._prepare_queued_chat_message(
        robot=robot,
        item=item,
        job=job,
    )

    assert "你怎么不说话了" in prepared
    assert prepared.count("先说第一件事") == 1
    assert "回冬啊" not in prepared


def test_term_command_routes_to_named_item_agent(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=False,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("/term alpha status?"),
    )

    assert response.success is True
    assert response.item_id == str(item.id)
    assert response.route_key == "alpha"
    assert captured["job"].message == "status?"
    assert captured["job"].sender_key == "onebot_v11:group:g1:u1"
    assert isinstance(captured["job"].reply_target, RobotReplyTarget)
    assert response.reply_chunks == []


def test_bot_at_prefix_does_not_break_term_command(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=False,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "[CQ:at,qq=10001] /term alpha status?",
            mentioned_bot=True,
            bot_self_ids=["10001"],
        ),
    )

    assert response.success is True
    assert response.item_id == str(item.id)
    assert response.route_key == "alpha"
    assert captured["job"].message == "status?"
    assert response.reply_chunks == []


def test_bot_at_prefix_is_preserved_for_chat_message(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "[CQ:at,qq=10001] hello",
            mentioned_bot=True,
            bot_self_ids=["10001"],
        ),
    )

    assert response.success is True
    assert captured["job"].message == "[CQ:at,qq=10001] hello"
    assert response.reply_chunks == []

def test_reply_message_type_filter_ignores_unselected_group_message(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {"reply_message_types": ["private"]},
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    def fail_enqueue_chat_job(_job):
        raise AssertionError("group messages should be filtered before chat dispatch")

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fail_enqueue_chat_job)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello group"),
    )

    assert response.success is True
    assert response.ignored is True
    assert response.reason == "conversation_sleeping"
    assert response.reply_chunks == []


def test_configured_group_message_still_requires_awake_controller(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {"reply_message_types": ["group"]},
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    def fail_enqueue_chat_job(_job):
        raise AssertionError("sleeping group chat must not reach the agent")

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fail_enqueue_chat_job)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello group"),
    )

    assert response.success is True
    assert response.ignored is True
    assert response.reason == "conversation_sleeping"
    assert response.reply_chunks == []
    snapshots = robot_service.conversation_controller_snapshots({robot.id})
    assert len(snapshots) == 1
    assert snapshots[0]["conversation_key"] == "group:g1"
    assert snapshots[0]["status"] == "sleeping"


def test_reply_message_type_filter_allows_configured_private_message(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {"reply_message_types": ["private"]},
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello private",
            sender_key="onebot_v11:private:u1",
            target={"id": "u1", "private": True},
        ),
    )

    assert response.success is True
    assert response.ignored is False
    assert captured["job"].message == "hello private"
    assert response.reply_chunks == []


def test_reply_message_type_filter_allows_command_when_scope_is_disabled(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {"reply_message_types": ["command"]},
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=False,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("/term alpha status?"),
    )

    assert response.success is True
    assert response.ignored is False
    assert captured["job"].message == "status?"
    assert response.reply_chunks == []


def test_reply_message_type_filter_ignores_unmentioned_when_only_mention_allowed(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {"reply_message_types": ["mention"]},
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    def fail_enqueue_chat_job(_job):
        raise AssertionError("unmentioned group messages should be filtered")

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fail_enqueue_chat_job)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello group"),
    )

    assert response.success is True
    assert response.ignored is True
    assert response.reason == "conversation_sleeping"
    assert response.reply_chunks == []


def test_reply_message_type_filter_allows_mention_when_group_is_disabled(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {"reply_message_types": ["mention"]},
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello mention", mentioned_bot=True),
    )

    assert response.success is True
    assert response.ignored is False
    assert captured["job"].message == "hello mention"
    assert response.reply_chunks == []


def test_reply_message_type_filter_ignores_at_target_by_default_when_only_mention_allowed(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {"reply_message_types": ["mention"]},
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    def fail_enqueue_chat_job(_job):
        raise AssertionError("@ target should be filtered unless mention_match_mode is any")

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fail_enqueue_chat_job)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "这个人是小男娘",
            sender={
                "user_id": "2537134688",
                "display_name": "只有白龙马知道唐三藏动了情",
            },
            conversation={"type": "group", "id": "369040885"},
            mentions=[
                {
                    "id": "1512220570",
                    "qq": "1512220570",
                    "name": "她喜欢我才钓着我",
                }
            ],
        ),
    )

    assert response.success is True
    assert response.ignored is True
    assert response.reason == "conversation_sleeping"
    assert response.reply_chunks == []


def test_reply_message_type_filter_ignores_at_target_even_when_mention_match_mode_is_any(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "mention_match_mode": "any",
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    def fail_enqueue_chat_job(_job):
        raise AssertionError("@ target should not wake this robot")

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fail_enqueue_chat_job)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello other mention",
            sender={
                "user_id": "2537134688",
                "display_name": "Alice",
            },
            conversation={"type": "group", "id": "369040885"},
            mentions=[
                {
                    "id": "1512220570",
                    "qq": "1512220570",
                    "name": "Bob",
                }
            ],
        ),
    )

    assert response.success is True
    assert response.ignored is True
    assert response.reason == "conversation_sleeping"
    assert response.reply_chunks == []


def test_mention_only_keeps_context_active_after_agent_sends_qq_message(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    now = robot_service._now()
    current_time = {"value": now}
    monkeypatch.setattr(robot_service, "_now", lambda: current_time["value"])
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    before_activation = robot_service.handle_inbound_message(
        db,
        robot,
        _message("plain before", target={"id": "g1"}),
    )
    assert before_activation.ignored is True
    assert before_activation.reason == "conversation_sleeping"

    mentioned = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello mention", target={"id": "g1"}, mentioned_bot=True),
    )
    assert mentioned.ignored is False
    robot_service._apply_reply_context_result(
        robot,
        queued_jobs[0].conversation_key,
        robot_message_sent=False,
        reply_target=queued_jobs[0].reply_target,
    )

    no_tool_reply = robot_service.handle_inbound_message(
        db,
        robot,
        _message("plain after no tool reply", target={"id": "g1"}),
    )
    assert no_tool_reply.ignored is True
    assert no_tool_reply.reason == "conversation_sleeping"

    mentioned_again = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello mention again", target={"id": "g1"}, mentioned_bot=True),
    )
    assert mentioned_again.ignored is False
    robot_service._apply_reply_context_result(
        robot,
        queued_jobs[-1].conversation_key,
        robot_message_sent=True,
        reply_target=queued_jobs[-1].reply_target,
    )

    current_time["value"] = now + timedelta(seconds=14)

    within_window = robot_service.handle_inbound_message(
        db,
        robot,
        _message("plain after mention", target={"id": "g1"}),
    )
    assert within_window.ignored is False
    assert [job.message for job in queued_jobs] == [
        "hello mention",
        "hello mention again",
        "plain after mention",
    ]
    robot_service._apply_reply_context_result(
        robot,
        queued_jobs[-1].conversation_key,
        robot_message_sent=True,
        reply_target=queued_jobs[-1].reply_target,
    )

    current_time["value"] = now + timedelta(seconds=30)

    expired = robot_service.handle_inbound_message(
        db,
        robot,
        _message("plain expired", target={"id": "g1"}),
    )
    assert expired.ignored is True
    assert expired.reason == "conversation_sleeping"


def test_reply_context_window_is_scoped_to_current_conversation(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    activated = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello mention",
            sender_key="onebot_v11:group:g1:u1",
            target={"id": "g1"},
            sender={"user_id": "u1", "display_name": "Alice"},
            mentioned_bot=True,
        ),
    )
    assert activated.ignored is False
    robot_service._apply_reply_context_result(
        robot,
        queued_jobs[0].conversation_key,
        robot_message_sent=True,
        reply_target=queued_jobs[0].reply_target,
    )

    other_group = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "plain in another group",
            sender_key="onebot_v11:group:g2:u2",
            target={"id": "g2"},
            sender={"user_id": "u2", "display_name": "Bob"},
        ),
    )
    assert other_group.ignored is True
    assert other_group.reason == "conversation_sleeping"

    other_private = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "plain in private",
            sender_key="onebot_v11:private:u2",
            target={"id": "u2", "private": True},
            sender={"user_id": "u2", "display_name": "Bob"},
        ),
    )
    assert other_private.ignored is False
    assert queued_jobs[-1].conversation_key == "private:u2"
    assert queued_jobs[-1].direct_reply_trigger is True
    assert queued_jobs[-1].reply_requires_awake is True

    same_group = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "plain in same group",
            sender_key="onebot_v11:group:g1:u3",
            target={"id": "g1"},
            sender={"user_id": "u3", "display_name": "Carol"},
        ),
    )
    assert same_group.ignored is False
    same_group_job = queued_jobs[-1]
    assert queued_jobs[0].message.startswith(
        "[Robot message; conversation=group:g1; trigger=mention_bot; sender=Alice (u1); mentions=TestBot (10001) (you)]\n"
    )
    assert "Alice (u1): hello mention" not in queued_jobs[0].message
    assert queued_jobs[0].message.endswith("[Current QQ message]\nhello mention")
    assert same_group_job.conversation_key == "group:g1"
    assert same_group_job.reply_context_active is True
    assert same_group_job.direct_reply_trigger is False
    assert "[Recent QQ conversation context" not in same_group_job.message
    assert "- Alice (u1): hello mention" not in same_group_job.message
    assert "plain in another group" not in same_group_job.message
    assert "plain in private" not in same_group_job.message
    assert same_group_job.message.startswith(
        "[Robot message; conversation=group:g1; trigger=active_chat_window; sender=Carol (u3)]\n"
    )

    assert same_group_job.message.endswith("[Current QQ message]\nplain in same group")


def test_direct_wakeup_agent_message_excludes_recent_same_conversation_context(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    same_group_plain = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "plain before",
            sender_key="onebot_v11:group:g1:u1",
            target={"id": "g1"},
            sender={"user_id": "u1", "display_name": "Alice"},
        ),
    )
    assert same_group_plain.ignored is True

    other_group_plain = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "plain in another group",
            sender_key="onebot_v11:group:g2:u2",
            target={"id": "g2"},
            sender={"user_id": "u2", "display_name": "Eve"},
        ),
    )
    assert other_group_plain.ignored is True

    mentioned = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello mention",
            sender_key="onebot_v11:group:g1:u3",
            target={"id": "g1"},
            sender={"user_id": "u3", "display_name": "Bob"},
            mentioned_bot=True,
        ),
    )

    assert mentioned.ignored is False
    assert len(queued_jobs) == 1
    assert "[Recent QQ conversation context" not in queued_jobs[0].message
    assert "[Recent QQ live context; background only" not in queued_jobs[0].message
    assert "Alice (u1): plain before" not in queued_jobs[0].message
    assert "plain in another group" not in queued_jobs[0].message
    assert queued_jobs[0].message.startswith(
        "[Robot message; conversation=group:g1; trigger=mention_bot; sender=Bob (u3); mentions=TestBot (10001) (you)]\n"
    )
    assert queued_jobs[0].message.endswith("[Current QQ message]\nhello mention")

def test_reply_context_uses_onebot_group_id_before_universal_target_id(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)

    mentioned = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello in group one",
            sender_key="onebot_v11:group:g1:u1",
            target={
                "id": "shared-target",
                "message_type": "group",
                "group_id": "g1",
            },
            conversation={
                "platform": "onebot_v11",
                "type": "group",
                "id": "g1",
                "target_type": "group",
                "target_id": "g1",
                "group_id": "g1",
                "user_id": "u1",
                "message_type": "group",
            },
            sender={"user_id": "u1", "display_name": "Alice"},
            mentioned_bot=True,
        ),
    )
    assert mentioned.ignored is False
    assert queued_jobs[0].conversation_key == "group:g1"
    robot_service._apply_reply_context_result(
        robot,
        queued_jobs[0].conversation_key,
        robot_message_sent=True,
        reply_target=queued_jobs[0].reply_target,
    )

    other_group = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "plain in group two",
            sender_key="onebot_v11:group:g2:u2",
            target={
                "id": "shared-target",
                "message_type": "group",
                "group_id": "g2",
            },
            conversation={
                "platform": "onebot_v11",
                "type": "group",
                "id": "g2",
                "target_type": "group",
                "target_id": "g2",
                "group_id": "g2",
                "user_id": "u2",
                "message_type": "group",
            },
            sender={"user_id": "u2", "display_name": "Bob"},
        ),
    )
    assert other_group.ignored is True
    assert other_group.reason == "conversation_sleeping"
    assert [job.conversation_key for job in queued_jobs] == ["group:g1"]


def test_conversation_controller_rejects_stale_generation_after_sleep(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    now = robot_service._now()
    current_time = {"value": now}
    monkeypatch.setattr(robot_service, "_now", lambda: current_time["value"])
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello mention", target={"id": "g1"}, mentioned_bot=True),
    )
    assert first.ignored is False
    first_job = queued_jobs[-1]
    first_generation = first_job.conversation_generation
    assert first_job.reply_requires_awake is True
    assert first_generation > 0
    assert robot_service.conversation_controller_allows_reply(
        robot.id,
        first_job.conversation_key,
        first_generation,
        requires_awake=True,
    )

    robot_service._apply_reply_context_result(
        robot,
        first_job.conversation_key,
        robot_message_sent=True,
        reply_target=first_job.reply_target,
        conversation_generation=first_generation,
    )
    current_time["value"] = now + timedelta(seconds=30)

    assert not robot_service.conversation_controller_allows_reply(
        robot.id,
        first_job.conversation_key,
        first_generation,
        requires_awake=True,
    )

    second = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello again", target={"id": "g1"}, mentioned_bot=True),
    )
    assert second.ignored is False
    second_job = queued_jobs[-1]
    assert second_job.conversation_generation > first_generation
    assert not robot_service.conversation_controller_allows_reply(
        robot.id,
        first_job.conversation_key,
        first_generation,
        requires_awake=True,
    )
    assert robot_service.conversation_controller_allows_reply(
        robot.id,
        second_job.conversation_key,
        second_job.conversation_generation,
        requires_awake=True,
    )


def test_private_message_wakes_its_conversation_controller(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["private"],
            "reply_context_window_seconds": 10,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)

    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello private",
            sender_key="onebot_v11:private:u1",
            target={"id": "u1", "private": True},
            sender={"user_id": "u1", "display_name": "Alice"},
        ),
    )
    assert first.ignored is False
    first_job = queued_jobs[-1]
    assert first_job.conversation_key == "private:u1"
    assert first_job.direct_reply_trigger is True
    assert first_job.reply_requires_awake is True
    assert first_job.conversation_generation > 0

    robot_service._apply_reply_context_result(
        robot,
        first_job.conversation_key,
        robot_message_sent=True,
        reply_target=first_job.reply_target,
        conversation_generation=first_job.conversation_generation,
    )

    robot_service.sleep_conversation_controller(
        robot.id,
        first_job.conversation_key,
        reason="test_private_sleep",
    )
    assert not robot_service.conversation_controller_allows_reply(
        robot.id,
        first_job.conversation_key,
        first_job.conversation_generation,
        requires_awake=True,
    )

    wake = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "hello again",
            sender_key="onebot_v11:private:u1",
            target={"id": "u1", "private": True},
            sender={"user_id": "u1", "display_name": "Alice"},
        ),
    )
    assert wake.ignored is False
    assert len(queued_jobs) == 2
    assert queued_jobs[-1].conversation_generation > first_job.conversation_generation


def test_direct_wakeup_job_reaches_agent_even_if_controller_window_expires(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 10,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    now = robot_service._now()
    current_time = {"value": now}
    monkeypatch.setattr(robot_service, "_now", lambda: current_time["value"])
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("wake after delay", target={"id": "g1"}, mentioned_bot=True),
    )
    assert response.ignored is False
    job = queued_jobs[-1]
    assert job.direct_reply_trigger is True
    assert job.reply_requires_awake is True

    current_time["value"] = now + timedelta(seconds=11)
    captured_messages: list[str] = []

    async def fake_chat_with_item(**kwargs):
        captured_messages.append(str(kwargs["message"]))
        return ChatResponseResult(content="", robot_message_sent=False)

    monkeypatch.setattr(robot_service, "_chat_with_item", fake_chat_with_item)

    robot_service._process_chat_job(job)

    assert captured_messages
    assert all("wake after delay" in message for message in captured_messages)


def test_visible_agent_response_without_tool_send_is_not_auto_delivered(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 10,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []
    sent_messages: list[tuple[Any, Any, str]] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    def fake_send_message(robot_id, target, text) -> None:
        sent_messages.append((robot_id, target, text))

    async def fake_chat_with_item(**_kwargs):
        return ChatResponseResult(content="install done", robot_message_sent=False)

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    monkeypatch.setattr(robot_service, "_chat_with_item", fake_chat_with_item)
    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("install done?", target={"id": "g1"}, mentioned_bot=True),
    )
    assert response.ignored is False
    job = queued_jobs[-1]

    robot_service._process_chat_job(job)

    assert sent_messages == []


def test_visible_agent_response_is_not_auto_delivered_after_long_processing(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 10,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []
    sent_messages: list[tuple[Any, Any, str]] = []
    now = robot_service._now()
    current_time = {"value": now}

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    def fake_send_message(robot_id, target, text) -> None:
        sent_messages.append((robot_id, target, text))

    async def fake_chat_with_item(**_kwargs):
        current_time["value"] = now + timedelta(seconds=180)
        return ChatResponseResult(content="Forge 已经安装好了", robot_message_sent=False)

    monkeypatch.setattr(robot_service, "_now", lambda: current_time["value"])
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    monkeypatch.setattr(robot_service, "_chat_with_item", fake_chat_with_item)
    monkeypatch.setattr(
        "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
        fake_send_message,
    )

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("forge 装好了吗", target={"id": "g1"}, mentioned_bot=True),
    )
    assert response.ignored is False
    job = queued_jobs[-1]

    robot_service._process_chat_job(job)

    assert sent_messages == []


def test_direct_wakeup_countdown_starts_after_agent_result(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 10,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    now = robot_service._now()
    current_time = {"value": now}
    monkeypatch.setattr(robot_service, "_now", lambda: current_time["value"])
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("wake before countdown", target={"id": "g1"}, mentioned_bot=True),
    )
    assert response.ignored is False
    job = queued_jobs[-1]
    assert job.direct_reply_trigger is True
    assert job.reply_requires_awake is True

    processing_snapshot = robot_service.conversation_controller_snapshots({robot.id})[0]
    assert processing_snapshot["conversation_key"] == "group:g1"
    assert processing_snapshot["status"] == "processing"
    assert processing_snapshot["awake"] is True
    assert processing_snapshot["processing"] is True
    assert processing_snapshot["expires_at"] is None
    assert processing_snapshot["seconds_remaining"] == 0
    assert processing_snapshot["processing_seconds_remaining"] == 120

    current_time["value"] = now + timedelta(seconds=11)
    assert robot_service.conversation_controller_allows_reply(
        robot.id,
        job.conversation_key,
        job.conversation_generation,
        requires_awake=True,
    )

    robot_service._apply_reply_context_result(
        robot,
        job.conversation_key,
        robot_message_sent=True,
        reply_target=job.reply_target,
        conversation_generation=job.conversation_generation,
    )

    awake_snapshot = robot_service.conversation_controller_snapshots({robot.id})[0]
    assert awake_snapshot["status"] == "awake"
    assert awake_snapshot["awake"] is True
    assert awake_snapshot["processing"] is False
    assert awake_snapshot["expires_at"] is not None
    assert awake_snapshot["seconds_remaining"] == 10

    current_time["value"] = now + timedelta(seconds=22)
    assert not robot_service.conversation_controller_allows_reply(
        robot.id,
        job.conversation_key,
        job.conversation_generation,
        requires_awake=True,
    )
    sleeping_snapshot = robot_service.conversation_controller_snapshots({robot.id})[0]
    assert sleeping_snapshot["status"] == "sleeping"
    assert sleeping_snapshot["awake"] is False
    assert sleeping_snapshot["processing"] is False


def test_direct_wakeup_messages_queue_without_superseding_active_reply(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    bot_self_id = str(robot.config["credentials"]["self_id"])
    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "第一个问题",
            sender_key="onebot_v11:group:g1:u1",
            target={"id": "g1"},
            sender={"user_id": "u1", "display_name": "Alice"},
            mentions=[{"qq": bot_self_id}],
            mentioned_bot=True,
            bot_self_ids=[bot_self_id],
        ),
    )
    assert first.ignored is False
    first_job = queued_jobs[-1]
    _simulate_worker_pickup(monkeypatch, first_job)

    second = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "第二个人也问",
            sender_key="onebot_v11:group:g1:u2",
            target={"id": "g1"},
            sender={"user_id": "u2", "display_name": "Bob"},
            mentions=[{"qq": bot_self_id}],
            mentioned_bot=True,
            bot_self_ids=[bot_self_id],
        ),
    )
    third = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "第三个人继续问",
            sender_key="onebot_v11:group:g1:u3",
            target={"id": "g1"},
            sender={"user_id": "u3", "display_name": "Carol"},
            mentions=[{"qq": bot_self_id}],
            mentioned_bot=True,
            bot_self_ids=[bot_self_id],
        ),
    )

    assert second.reason == "queued_pending"
    assert third.reason == "queued_pending"
    # All chat messages merge into the pending queue while a turn is in
    # flight, regardless of trigger; the batch drains at turn end.
    assert len(queued_jobs) == 1
    assert robot_service.conversation_controller_allows_completion_reply(
        robot.id,
        first_job.conversation_key,
        first_job.conversation_generation,
        requires_awake=True,
    )
    snapshots = robot_service.conversation_controller_snapshots(
        {robot.id},
        item_ids={item.id},
    )
    assert len(snapshots) == 1
    assert snapshots[0]["pending_count"] == 2


def test_pending_followup_includes_recent_live_context_without_duplicate(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    target = RobotReplyTarget(
        target_type="group",
        target_id="770362397",
        metadata={
            "conversation": {"type": "group", "id": "770362397"},
            "sender": {"user_id": "2206406352", "display_name": "New+7"},
        },
    )
    queued_jobs: list[Any] = []
    current_question = "谁是猫娘"
    fact = "猫娘就是月影汉堡猫娘"
    confirmation = "收到，猫娘就是月影汉堡猫娘"

    monkeypatch.setattr(
        robot_service,
        "_dispatch_pending_batch_to_session_queue",
        lambda **kwargs: queued_jobs.append(kwargs) or True,
    )
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    monkeypatch.setattr(
        robot_conversation_memory,
        "read_recent",
        lambda *_args, **_kwargs: (
            f"[2026-07-17T15:36:11+00:00] user New+7: {fact}\n"
            f"[2026-07-17T15:36:12+00:00] assistant: {confirmation}\n"
            f"[2026-07-17T15:36:31+00:00] user New+7: {current_question}"
        ),
    )
    robot_service._record_pending_chat_input(
        robot=robot,
        conversation_key="group:770362397",
        item_id=item.id,
        route_key="alpha",
        message_text=current_question,
        sender_key="onebot_v11:group:770362397:2206406352",
        sender_label="New+7 (2206406352)",
        trigger_reason="mention_bot",
        reply_target=target,
    )

    assert robot_service._enqueue_pending_chat_followup(
        robot=robot,
        conversation_key="group:770362397",
    )
    assert len(queued_jobs) == 1
    message = queued_jobs[0]["composed_message"]
    assert "[Recent QQ live context; latest" in message
    assert fact in message
    assert confirmation in message
    assert message.count(current_question) == 1


def test_pending_qq_batch_keeps_senders_and_isolates_conversations(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    queued_jobs: list[Any] = []

    monkeypatch.setattr(
        robot_service,
        "_dispatch_pending_batch_to_session_queue",
        lambda **kwargs: queued_jobs.append(kwargs) or True,
    )
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    monkeypatch.setattr(robot_conversation_memory, "read_recent", lambda *_args, **_kwargs: "")

    def target(group_id: str, user_id: str, display_name: str) -> RobotReplyTarget:
        return RobotReplyTarget(
            target_type="group",
            target_id=group_id,
            metadata={
                "conversation": {"type": "group", "id": group_id},
                "sender": {"user_id": user_id, "display_name": display_name},
            },
        )

    for conversation_key, group_id, user_id, display_name, text in (
        ("group:g1", "g1", "u1", "Alice", "A 的问题"),
        ("group:g1", "g1", "u2", "Bob", "B 的问题"),
        ("group:g2", "g2", "u3", "Carol", "另一个群的问题"),
    ):
        robot_service._record_pending_chat_input(
            robot=robot,
            conversation_key=conversation_key,
            item_id=item.id,
            route_key="alpha",
            message_text=text,
            sender_key=f"onebot_v11:{conversation_key}:{user_id}",
            sender_label=f"{display_name} ({user_id})",
            trigger_reason="mention_bot",
            reply_target=target(group_id, user_id, display_name),
        )

    assert robot_service._enqueue_pending_chat_followup(
        robot=robot,
        conversation_key="group:g1",
    )
    assert len(queued_jobs) == 1
    message = queued_jobs[0]["composed_message"]
    assert "source=QQ; conversation=group:g1" in message
    assert "sender=Alice (u1)" in message
    assert "sender_key=onebot_v11:group:g1:u1" in message
    assert "sender=Bob (u2)" in message
    assert "sender_key=onebot_v11:group:g1:u2" in message
    assert "A 的问题" in message
    assert "B 的问题" in message
    assert queued_jobs[0]["followup_reply_target"].metadata["allow_multiple_reply_messages"] is True
    assert "Carol" not in message
    assert "另一个群的问题" not in message

    other_pending, _ = robot_service._pending_chat_snapshot_locked(
        robot.id,
        "group:g2",
    )
    assert len(other_pending) == 1
    assert other_pending[0]["sender_label"] == "Carol (u3)"
    robot_service._drain_pending_chat_inputs(robot.id, "group:g2")


def test_pending_qq_batch_merges_same_sender_into_one_reply_intent(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    queued_jobs: list[Any] = []
    target = RobotReplyTarget(
        target_type="group",
        target_id="g1",
        metadata={
            "conversation": {"type": "group", "id": "g1"},
            "sender": {"user_id": "u1", "display_name": "Alice"},
        },
    )

    monkeypatch.setattr(
        robot_service,
        "_dispatch_pending_batch_to_session_queue",
        lambda **kwargs: queued_jobs.append(kwargs) or True,
    )
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    monkeypatch.setattr(robot_conversation_memory, "read_recent", lambda *_args, **_kwargs: "")

    for text in ("Are you there?", "Reply now"):
        robot_service._record_pending_chat_input(
            robot=robot,
            conversation_key="group:g1",
            item_id=item.id,
            route_key="alpha",
            message_text=text,
            sender_key="onebot_v11:group:g1:u1",
            sender_label="Alice (u1)",
            trigger_reason="mention_bot",
            reply_target=target,
        )

    assert robot_service._enqueue_pending_chat_followup(
        robot=robot,
        conversation_key="group:g1",
    )
    assert len(queued_jobs) == 1
    job = queued_jobs[0]
    assert job["followup_reply_target"].metadata["allow_multiple_reply_messages"] is False
    # 内容不同的消息都要保留且被覆盖，回复引导是按语义组织（同义融合、
    # 不同题目各一个气泡）。
    assert "Are you there?" in job["composed_message"]
    assert "Reply now" in job["composed_message"]
    assert "语义" in job["composed_message"]


def test_plain_task_control_message_dispatches_immediately_while_processing(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []
    monkeypatch.setattr(
        robot_service,
        "_enqueue_chat_job",
        lambda job: queued_jobs.append(job) or True,
    )
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "安装 Temurin Java 17",
            sender_key="onebot_v11:group:g1:u1",
            target={"id": "g1"},
            sender={"user_id": "u1", "display_name": "Alice"},
            mentioned_bot=True,
        ),
    )
    assert first.reason == "queued"

    change = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "先别下，换个国内镜像",
            sender_key="onebot_v11:group:g1:u1",
            target={"id": "g1"},
            sender={"user_id": "u1", "display_name": "Alice"},
        ),
    )

    assert change.reason == "queued"
    assert len(queued_jobs) == 2
    assert queued_jobs[1].direct_reply_trigger is False
    assert queued_jobs[1].reply_context_active is True
    snapshots = robot_service.conversation_controller_snapshots(
        {robot.id},
        item_ids={item.id},
    )
    assert snapshots[0]["pending_count"] == 0


def test_direct_wakeup_pending_messages_continue_after_first_job_sends_no_reply(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", lambda job: queued_jobs.append(job) or True)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "第一个唤醒",
            sender_key="onebot_v11:group:g1:u1",
            target={"id": "g1"},
            sender={"user_id": "u1", "display_name": "Alice"},
            mentioned_bot=True,
        ),
    )
    assert first.ignored is False
    first_job = queued_jobs[-1]
    _simulate_worker_pickup(monkeypatch, first_job)

    active_plain = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "旁边人闲聊一句",
            sender_key="onebot_v11:group:g1:u2",
            target={"id": "g1"},
            sender={"user_id": "u2", "display_name": "Bob"},
        ),
    )
    second_wakeup = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "第二个人也叫你",
            sender_key="onebot_v11:group:g1:u3",
            target={"id": "g1"},
            sender={"user_id": "u3", "display_name": "Carol"},
            mentioned_bot=True,
        ),
    )
    third_wakeup = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "第三个人继续叫你",
            sender_key="onebot_v11:group:g1:u4",
            target={"id": "g1"},
            sender={"user_id": "u4", "display_name": "Dave"},
            mentioned_bot=True,
        ),
    )

    # Every follow-up (plain chat or fresh @mention) merges into the pending
    # queue while a turn is in flight; the turn-end drain starts one follow-up.
    assert active_plain.reason == "queued_pending"
    assert second_wakeup.reason == "queued_pending"
    assert third_wakeup.reason == "queued_pending"
    assert len(queued_jobs) == 1
    snapshots = robot_service.conversation_controller_snapshots(
        {robot.id},
        item_ids={item.id},
    )
    assert snapshots[0]["pending_count"] == 3
    assert snapshots[0]["pending_messages"][0]["message_preview"] == "旁边人闲聊一句"
def test_pure_qq_image_message_is_ignored_before_agent_dispatch(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", lambda job: queued_jobs.append(job) or True)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "[CQ:image,summary=&#91;animation&#93;,file=emoji.jpg,url=https://example.test/a.jpg]",
            target={"id": "g-image-only"},
            sender={"user_id": "u1", "display_name": "Alice"},
            mentioned_bot=True,
        ),
    )

    assert response.success is True
    assert response.ignored is True
    assert response.reason == "media_message_ignored"
    assert queued_jobs == []


def test_qq_image_segments_are_removed_from_agent_and_pending_messages(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []
    dispatched_batches: list[Any] = []
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", lambda job: queued_jobs.append(job) or True)
    monkeypatch.setattr(
        robot_service,
        "_dispatch_pending_batch_to_session_queue",
        lambda **kwargs: dispatched_batches.append(kwargs) or True,
    )
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "[CQ:at,qq=10001] first [CQ:image,file=first.jpg,url=https://example.test/first.jpg] question",
            sender_key="onebot_v11:group:g-image-mixed:u1",
            target={"id": "g-image-mixed"},
            sender={"user_id": "u1", "display_name": "Alice"},
            mentioned_bot=True,
            bot_self_ids=["10001"],
        ),
    )
    assert first.ignored is False
    assert len(queued_jobs) == 1
    first_job = queued_jobs[-1]
    _simulate_worker_pickup(monkeypatch, first_job)
    assert "[CQ:image" not in first_job.message
    assert first_job.message.endswith(
        "[Current QQ message]\n[CQ:at,qq=10001] first question"
    )

    image_only = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "[CQ:image,summary=&#91;animation&#93;,file=emoji.jpg,url=https://example.test/emoji.jpg]",
            sender_key="onebot_v11:group:g-image-mixed:u2",
            target={"id": "g-image-mixed"},
            sender={"user_id": "u2", "display_name": "Bob"},
        ),
    )
    assert image_only.ignored is True
    assert image_only.reason == "media_message_ignored"
    assert len(queued_jobs) == 1

    mixed_pending = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "look [CQ:image,file=pending.jpg,url=https://example.test/pending.jpg] now",
            sender_key="onebot_v11:group:g-image-mixed:u3",
            target={"id": "g-image-mixed"},
            sender={"user_id": "u3", "display_name": "Carol"},
        ),
    )
    assert mixed_pending.reason == "queued_pending"

    robot_service._apply_reply_context_result(
        robot,
        first_job.conversation_key,
        robot_message_sent=True,
        reply_target=first_job.reply_target,
        conversation_generation=first_job.conversation_generation,
    )
    assert robot_service._enqueue_pending_chat_followup(
        robot=robot,
        conversation_key=first_job.conversation_key,
    ) is True

    assert len(queued_jobs) == 1
    assert len(dispatched_batches) == 1
    pending_message = dispatched_batches[-1]["composed_message"]
    assert "[CQ:image" not in pending_message
    assert (
        "1. sender=Carol (u3); sender_key=onebot_v11:group:g-image-mixed:u3; "
        "trigger=active_chat_window: look now"
    ) in pending_message


def test_pending_chat_queue_keeps_latest_five_messages(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []
    dispatched_batches: list[Any] = []
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", lambda job: queued_jobs.append(job) or True)
    monkeypatch.setattr(
        robot_service,
        "_dispatch_pending_batch_to_session_queue",
        lambda **kwargs: dispatched_batches.append(kwargs) or True,
    )
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")
    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message(
            "正在处理的第一条",
            sender_key="onebot_v11:group:g1:u0",
            target={"id": "g1"},
            sender={"user_id": "u0", "display_name": "User0"},
            mentioned_bot=True,
        ),
    )
    assert first.ignored is False
    first_job = queued_jobs[-1]
    _simulate_worker_pickup(monkeypatch, first_job)

    for index in range(1, 8):
        response = robot_service.handle_inbound_message(
            db,
            robot,
            _message(
                f"pending {index}",
                sender_key=f"onebot_v11:group:g1:u{index}",
                target={"id": "g1"},
                sender={"user_id": f"u{index}", "display_name": f"User{index}"},
            ),
        )
        assert response.reason == "queued_pending"

    robot_service._apply_reply_context_result(
        robot,
        first_job.conversation_key,
        robot_message_sent=True,
        reply_target=first_job.reply_target,
        conversation_generation=first_job.conversation_generation,
    )
    assert robot_service._enqueue_pending_chat_followup(
        robot=robot,
        conversation_key=first_job.conversation_key,
    ) is True

    assert len(queued_jobs) == 1
    assert len(dispatched_batches) == 1
    message = dispatched_batches[-1]["composed_message"]
    assert "pending 1" not in message
    assert "pending 2" not in message
    for index in range(3, 8):
        assert f"pending {index}" in message
    assert message.count("trigger=active_chat_window: pending") == 5
def test_processing_controller_timeout_sleeps_group_and_blocks_plain_message(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["group", "mention"],
            "reply_context_window_seconds": 10,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    now = robot_service._now()
    current_time = {"value": now}
    monkeypatch.setattr(robot_service, "_now", lambda: current_time["value"])
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("wake and then stall", target={"id": "g1"}, mentioned_bot=True),
    )
    assert response.ignored is False
    job = queued_jobs[-1]

    processing_snapshot = robot_service.conversation_controller_snapshots({robot.id})[0]
    assert processing_snapshot["status"] == "processing"
    assert processing_snapshot["processing_expires_at"] is not None
    assert processing_snapshot["processing_seconds_remaining"] == 120

    current_time["value"] = now + timedelta(seconds=121)
    plain = robot_service.handle_inbound_message(
        db,
        robot,
        _message("ordinary group chatter", target={"id": "g1"}),
    )
    assert plain.success is True
    assert plain.ignored is True
    assert plain.reason == "conversation_sleeping"
    assert not robot_service.conversation_controller_allows_reply(
        robot.id,
        job.conversation_key,
        job.conversation_generation,
        requires_awake=True,
    )

    sleeping_snapshot = robot_service.conversation_controller_snapshots({robot.id})[0]
    assert sleeping_snapshot["status"] == "sleeping"
    assert sleeping_snapshot["awake"] is False
    assert sleeping_snapshot["sleeping"] is True


def test_agent_no_reply_clears_controller_as_sleeping_gate(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["group", "mention"],
            "reply_context_window_seconds": 10,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("wake but no reply", target={"id": "g1"}, mentioned_bot=True),
    )
    assert response.ignored is False
    job = queued_jobs[-1]

    robot_service._apply_reply_context_result(
        robot,
        job.conversation_key,
        robot_message_sent=False,
        reply_target=job.reply_target,
        conversation_generation=job.conversation_generation,
    )

    plain = robot_service.handle_inbound_message(
        db,
        robot,
        _message("ordinary group chatter", target={"id": "g1"}),
    )
    assert plain.success is True
    assert plain.ignored is True
    assert plain.reason == "conversation_sleeping"
    assert len(queued_jobs) == 1


def test_reply_context_expiry_sleeps_group_conversation_until_direct_wakeup(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["group", "mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    now = robot_service._now()
    current_time = {"value": now}
    monkeypatch.setattr(robot_service, "_now", lambda: current_time["value"])
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello mention", target={"id": "g1"}, mentioned_bot=True),
    )
    assert first.ignored is False
    robot_service._apply_reply_context_result(
        robot,
        queued_jobs[-1].conversation_key,
        robot_message_sent=True,
        reply_target=queued_jobs[-1].reply_target,
        conversation_generation=queued_jobs[-1].conversation_generation,
    )

    current_time["value"] = now + timedelta(seconds=30)
    stale_plain = robot_service.handle_inbound_message(
        db,
        robot,
        _message("plain after idle", target={"id": "g1"}),
    )
    assert stale_plain.success is True
    assert stale_plain.ignored is True
    assert stale_plain.reason == "conversation_sleeping"
    assert len(queued_jobs) == 1

    wake = robot_service.handle_inbound_message(
        db,
        robot,
        _message("wake again", target={"id": "g1"}, mentioned_bot=True),
    )
    assert wake.ignored is False
    assert len(queued_jobs) == 2
    assert queued_jobs[-1].conversation_generation > queued_jobs[0].conversation_generation


def test_active_chat_window_no_reply_sleeps_group_controller_until_direct_wakeup(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["group", "mention"],
            "reply_context_window_seconds": 10,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello mention", target={"id": "g1"}, mentioned_bot=True),
    )
    assert first.ignored is False
    first_job = queued_jobs[-1]
    robot_service._apply_reply_context_result(
        robot,
        first_job.conversation_key,
        robot_message_sent=True,
        reply_target=first_job.reply_target,
        conversation_generation=first_job.conversation_generation,
    )

    active_plain = robot_service.handle_inbound_message(
        db,
        robot,
        _message("ordinary group chatter", target={"id": "g1"}),
    )
    assert active_plain.ignored is False
    assert len(queued_jobs) == 2
    active_job = queued_jobs[-1]
    assert active_job.reply_context_active is True
    assert active_job.direct_reply_trigger is False

    robot_service._apply_reply_context_result(
        robot,
        active_job.conversation_key,
        robot_message_sent=False,
        reply_target=active_job.reply_target,
        conversation_generation=active_job.conversation_generation,
        sleep_when_no_reply=True,
    )

    sleeping_plain = robot_service.handle_inbound_message(
        db,
        robot,
        _message("more ordinary group chatter", target={"id": "g1"}),
    )
    assert sleeping_plain.success is True
    assert sleeping_plain.ignored is True
    assert sleeping_plain.reason == "conversation_sleeping"
    assert len(queued_jobs) == 2
    assert not robot_service.conversation_controller_allows_reply(
        robot.id,
        active_job.conversation_key,
        active_job.conversation_generation,
        requires_awake=True,
    )

    wake = robot_service.handle_inbound_message(
        db,
        robot,
        _message("wake again", target={"id": "g1"}, mentioned_bot=True),
    )
    assert wake.ignored is False
    assert len(queued_jobs) == 3
    assert queued_jobs[-1].conversation_generation > active_job.conversation_generation


def test_conversation_controller_snapshot_reports_awake_then_sleeping(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 10,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    now = robot_service._now()
    current_time = {"value": now}
    monkeypatch.setattr(robot_service, "_now", lambda: current_time["value"])
    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello mention", target={"id": "g1"}, mentioned_bot=True),
    )
    assert response.ignored is False
    job = queued_jobs[-1]
    robot_service._apply_reply_context_result(
        robot,
        job.conversation_key,
        robot_message_sent=True,
        reply_target=job.reply_target,
        conversation_generation=job.conversation_generation,
    )

    awake_snapshot = robot_service.conversation_controller_snapshots({robot.id})[0]
    assert awake_snapshot["robot_id"] == str(robot.id)
    assert awake_snapshot["conversation_key"] == "group:g1"
    assert awake_snapshot["conversation_type"] == "group"
    assert awake_snapshot["conversation_id"] == "g1"
    assert awake_snapshot["item_id"] == str(item.id)
    assert awake_snapshot["status"] == "awake"
    assert awake_snapshot["awake"] is True
    assert awake_snapshot["sleeping"] is False
    assert awake_snapshot["seconds_remaining"] == 10
    assert robot_service.conversation_controller_snapshots(
        {robot.id},
        item_ids={create_random_item(db).id},
    ) == []

    current_time["value"] = now + timedelta(seconds=11)
    sleeping_snapshot = robot_service.conversation_controller_snapshots({robot.id})[0]
    assert sleeping_snapshot["status"] == "sleeping"
    assert sleeping_snapshot["awake"] is False
    assert sleeping_snapshot["sleeping"] is True
    assert sleeping_snapshot["expires_at"] is None
    assert sleeping_snapshot["seconds_remaining"] == 0
    assert sleeping_snapshot["generation"] > job.conversation_generation


def test_sleep_command_blocks_group_messages_until_direct_wakeup(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["group", "mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)
    monkeypatch.setattr(robot_service, "_conversation_impression_card", lambda **_: "")

    first = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello mention", target={"id": "g1"}, mentioned_bot=True),
    )
    assert first.ignored is False
    robot_service._apply_reply_context_result(
        robot,
        queued_jobs[-1].conversation_key,
        robot_message_sent=True,
        reply_target=queued_jobs[-1].reply_target,
        conversation_generation=queued_jobs[-1].conversation_generation,
    )

    sleep = robot_service.handle_inbound_message(
        db,
        robot,
        _message("\u4f11\u7720\u53bb", target={"id": "g1"}, replied_to_bot=True),
    )
    assert sleep.success is True
    assert sleep.ignored is True
    assert sleep.reason == "sleep_command"
    assert len(queued_jobs) == 1

    sleeping_plain = robot_service.handle_inbound_message(
        db,
        robot,
        _message("\u4f46\u662f\u4e00\u76f4\u6da8\u600e\u4e48\u4e70\uff1f", target={"id": "g1"}),
    )
    assert sleeping_plain.success is True
    assert sleeping_plain.ignored is True
    assert sleeping_plain.reason == "conversation_sleeping"
    assert len(queued_jobs) == 1

    wake = robot_service.handle_inbound_message(
        db,
        robot,
        _message("\u9192\u9192", target={"id": "g1"}, mentioned_bot=True),
    )
    assert wake.ignored is False
    assert len(queued_jobs) == 2
    assert queued_jobs[-1].conversation_generation > queued_jobs[0].conversation_generation

def test_reply_to_bot_only_keeps_context_active_after_agent_sends_qq_message(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {
            "reply_message_types": ["mention"],
            "reply_context_window_seconds": 15,
        },
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    queued_jobs: list[Any] = []

    def fake_enqueue_chat_job(job) -> bool:
        queued_jobs.append(job)
        return True

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", fake_enqueue_chat_job)

    replied = robot_service.handle_inbound_message(
        db,
        robot,
        _message("reply text", target={"id": "g1"}, replied_to_bot=True),
    )
    assert replied.ignored is False
    robot_service._apply_reply_context_result(
        robot,
        queued_jobs[0].conversation_key,
        robot_message_sent=False,
        reply_target=queued_jobs[0].reply_target,
    )

    no_tool_follow_up = robot_service.handle_inbound_message(
        db,
        robot,
        _message("plain after reply without tool", target={"id": "g1"}),
    )
    assert no_tool_follow_up.ignored is True
    assert no_tool_follow_up.reason == "conversation_sleeping"

    replied_again = robot_service.handle_inbound_message(
        db,
        robot,
        _message("reply text again", target={"id": "g1"}, replied_to_bot=True),
    )
    assert replied_again.ignored is False
    robot_service._apply_reply_context_result(
        robot,
        queued_jobs[-1].conversation_key,
        robot_message_sent=True,
        reply_target=queued_jobs[-1].reply_target,
    )

    follow_up = robot_service.handle_inbound_message(
        db,
        robot,
        _message("plain after reply", target={"id": "g1"}),
    )
    assert follow_up.ignored is False
    assert [job.message for job in queued_jobs] == [
        "reply text",
        "reply text again",
        "plain after reply",
    ]


def test_term_agent_response_is_not_auto_chunked_for_bridge(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    robot.config = {
        "credentials": dict(robot.config.get("credentials", {})),
        "options": {"max_message_length": 1000},
    }
    db.add(robot)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=False,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("/term alpha status?"),
    )

    assert response.success is True
    assert captured["job"].message == "status?"
    assert response.reply_chunks == []


def test_robot_message_is_ignored_when_backend_queue_is_full(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    monkeypatch.setattr(robot_service, "_enqueue_chat_job", lambda _job: False)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("hello", mentioned_bot=True),
    )

    assert response.success is True
    assert response.ignored is True
    assert response.reason == "dispatch_queue_full"
    assert response.item_id == str(item.id)
    assert response.route_key == "alpha"
    assert response.reply_chunks == []



def test_robot_impression_card_is_not_truncated_by_count(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    base_metadata = {
        "robot_id": str(robot.id),
        "robot_conversation_key": "group:g1",
        "robot_memory_schema_version": 3,
        "memory_scope": "conversation",
    }
    facts = [
        {
            "id": f"fact-{index}",
            "content": f"群事实编号 {index} 的内容",
            "metadata": {**base_metadata, "memory_type": "fact"},
        }
        for index in range(9)
    ]
    preferences = [
        {
            "id": f"pref-{index}",
            "content": f"稳定偏好编号 {index} 的内容",
            "metadata": {**base_metadata, "memory_type": "preference"},
        }
        for index in range(3)
    ]
    monkeypatch.setattr(
        vector_store,
        "get_all_memories",
        lambda *_args, **_kwargs: [*facts, *preferences],
    )
    monkeypatch.setattr(vector_store, "search_memories", lambda **_kwargs: [])

    card = robot_service._conversation_impression_card(
        item_id=item.id,
        robot=robot,
        conversation_key="group:g1",
        sender_key="onebot_v11:user:u1",
        query="",
    )

    for index in range(9):
        assert f"群事实编号 {index} 的内容" in card
    for index in range(3):
        assert f"稳定偏好编号 {index} 的内容" in card


def test_failed_job_drains_pending_chat_inputs_into_followup(
    db: Session,
    monkeypatch,
) -> None:
    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    target = RobotReplyTarget(
        target_type="group",
        target_id="g1",
        metadata={"target": {"id": "g1"}},
    )
    from datetime import datetime, timezone

    from app.plugins.robot.service import QueuedRobotChatJob

    job = QueuedRobotChatJob(
        job_id="job-fail-1",
        robot_id=robot.id,
        robot_owner_id=robot.owner_id,
        item_id=item.id,
        route_key="group:g1",
        message="[Current QQ message]\n装java",
        sender_key="onebot_v11:group:g1:u1",
        reply_target=target,
        conversation_key="group:g1",
        enqueued_at=datetime.now(timezone.utc),
        direct_reply_trigger=True,
        message_text="装java",
    )

    robot_service._record_pending_chat_input(
        robot=robot,
        conversation_key="group:g1",
        item_id=item.id,
        route_key="group:g1",
        message_text="装完叫我",
        sender_key="onebot_v11:group:g1:u1",
        sender_label="FLY",
        trigger_reason="mentioned",
        reply_target=target,
    )
    from app.services.agent.input_merge_buffer import input_merge_buffer

    scope_item, scope_key = robot_service._pending_chat_scope(robot.id, "group:g1")
    assert input_merge_buffer.count(scope_item, scope_key) > 0

    monkeypatch.setattr(
        robot_service,
        "conversation_controller_allows_reply",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        robot_service,
        "_prepare_queued_chat_message",
        lambda **kwargs: "msg",
    )

    async def failing_chat(**kwargs):
        raise RuntimeError("llm exploded")

    monkeypatch.setattr(robot_service, "_chat_with_item", failing_chat)
    monkeypatch.setattr(
        robot_service,
        "_record_and_send_job_error",
        lambda *args, **kwargs: None,
    )
    enqueued: list[object] = []
    monkeypatch.setattr(
        robot_service,
        "_dispatch_pending_batch_to_session_queue",
        lambda **kwargs: enqueued.append(kwargs) or True,
    )

    robot_service._process_chat_job(job)

    assert input_merge_buffer.count(scope_item, scope_key) == 0
    assert enqueued, "failed job should drain pending inputs into a followup batch"


def test_empty_direct_reply_is_logged_without_corrective_retry(
    db: Session,
    monkeypatch,
) -> None:
    from datetime import datetime, timezone

    from app.plugins.robot.service import QueuedRobotChatJob

    item = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.commit()

    job = QueuedRobotChatJob(
        job_id="job-retry-1",
        robot_id=robot.id,
        robot_owner_id=robot.owner_id,
        item_id=item.id,
        route_key="group:g1",
        message="[Current QQ message]\n你好",
        sender_key="onebot_v11:group:g1:u1",
        reply_target=RobotReplyTarget(
            target_type="group",
            target_id="g1",
            metadata={"target": {"id": "g1"}},
        ),
        conversation_key="group:g1",
        enqueued_at=datetime.now(timezone.utc),
        direct_reply_trigger=True,
        message_text="你好",
    )

    monkeypatch.setattr(
        robot_service,
        "conversation_controller_allows_reply",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        robot_service,
        "_prepare_queued_chat_message",
        lambda **kwargs: "msg",
    )
    monkeypatch.setattr(
        robot_service,
        "_apply_reply_context_result",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        robot_service,
        "_enqueue_pending_chat_followup",
        lambda **kwargs: False,
    )

    calls: list[str] = []

    async def fake_chat(**kwargs):
        calls.append(str(kwargs["message"]))
        return ChatResponseResult(content="", robot_message_sent=False)

    monkeypatch.setattr(robot_service, "_chat_with_item", fake_chat)
    delivered: list[str] = []
    monkeypatch.setattr(
        robot_service,
        "_send_visible_agent_response",
        lambda _job, text: delivered.append(text) or True,
    )

    robot_service._process_chat_job(job)

    assert len(calls) == 1, "empty direct reply must not trigger a corrective retry"
    assert delivered == []


def test_internal_retry_prefix_counts_as_internal_callback() -> None:
    from app.api.routes.chat import (
        INTERNAL_AGENT_RETRY_PREFIX,
        _is_internal_agent_callback,
    )

    message = INTERNAL_AGENT_RETRY_PREFIX + "\n（对方在直接问你：你好。）"
    assert _is_internal_agent_callback(message, "qq") is True
    assert _is_internal_agent_callback(message, "web") is False
    assert _is_internal_agent_callback("你好", "qq") is False


def test_pending_batch_template_is_sanitized_as_internal_trace() -> None:
    from app.plugins.robot.internal_trace import (
        is_robot_internal_trace_text,
        sanitize_robot_visible_text,
    )

    batch = (
        "[Pending QQ messages; merge same-sender follow-ups and answer current "
        "unresolved intents]\n"
        "These messages arrived while the bot was already thinking. Treat them "
        "as current live QQ messages, not old log history.\n"
        "source=QQ; conversation=private:2537134688\n"
        "1. sender=FLY (2537134688); sender_key=onebot_v11:private:2537134688; "
        "trigger=mention_bot: 怎么不回了\n"
        "Messages from the same sender are one evolving intent: use earlier "
        "lines only as context and answer that sender once based on the latest "
        "unresolved request."
    )

    assert is_robot_internal_trace_text(batch) is True
    assert sanitize_robot_visible_text(batch) == ""


def test_sanitize_strips_batch_lines_but_keeps_real_reply() -> None:
    from app.plugins.robot.internal_trace import sanitize_robot_visible_text

    mixed = (
        "在呢在呢，刚才卡了一下\n"
        "[Pending QQ messages; merge same-sender follow-ups]\n"
        "source=QQ; conversation=private:2537134688\n"
        "1. sender=FLY (2537134688); sender_key=onebot_v11:private:2537134688; trigger=mention_bot: 嗯\n"
        "你刚才说啥来着"
    )

    cleaned = sanitize_robot_visible_text(mixed)
    assert "在呢在呢，刚才卡了一下" in cleaned
    assert "你刚才说啥来着" in cleaned
    assert "Pending QQ messages" not in cleaned
    assert "source=QQ" not in cleaned
    assert "sender_key" not in cleaned


def test_normal_reply_is_not_internal_trace() -> None:
    from app.plugins.robot.internal_trace import (
        is_robot_internal_trace_text,
        sanitize_robot_visible_text,
    )

    assert is_robot_internal_trace_text("你好呀，我在") is False
    assert sanitize_robot_visible_text("你好呀，我在") == "你好呀，我在"


def test_reap_stuck_dispatch_jobs_force_fails_and_recovers(db: Session, monkeypatch) -> None:
    import time as time_module

    from app.core.config import settings as app_settings

    item = create_random_item(db)
    robot = create_random_robot(db)
    service = RobotService()
    monkeypatch.setattr(
        app_settings,
        "ROBOT_BACKEND_DISPATCH_HARD_TIMEOUT_SECONDS",
        0.1,
    )
    job = robot_service_module_queued_job(robot, item, "stuck-1")
    with service._lock:
        service._active_dispatch_jobs["stuck-1"] = (
            time_module.monotonic() - 10,
            job,
        )
    errors: list[str] = []
    monkeypatch.setattr(
        service,
        "_record_job_timeout_silent",
        lambda _job, detail: errors.append(detail),
    )

    reaped = service.reap_stuck_dispatch_jobs()

    assert reaped == 1
    assert errors and "hard timeout" in errors[0]
    assert "stuck-1" not in service._active_dispatch_jobs


def robot_service_module_queued_job(robot, item, job_id):
    from datetime import datetime, timezone

    from app.plugins.robot.service import QueuedRobotChatJob

    return QueuedRobotChatJob(
        job_id=job_id,
        robot_id=robot.id,
        robot_owner_id=robot.owner_id,
        item_id=item.id,
        route_key="group:g1",
        message="[Current QQ message]\n你好",
        sender_key="onebot_v11:group:g1:u1",
        reply_target=RobotReplyTarget(
            target_type="group",
            target_id="g1",
            metadata={},
        ),
        conversation_key="group:g1",
        enqueued_at=datetime.now(timezone.utc),
        direct_reply_trigger=True,
        message_text="你好",
    )


def test_reap_leaves_fresh_dispatch_jobs_alone(db: Session) -> None:
    import time as time_module

    item = create_random_item(db)
    robot = create_random_robot(db)
    service = RobotService()
    job = robot_service_module_queued_job(robot, item, "fresh-1")
    with service._lock:
        service._active_dispatch_jobs["fresh-1"] = (time_module.monotonic(), job)

    assert service.reap_stuck_dispatch_jobs() == 0
    assert "fresh-1" in service._active_dispatch_jobs


def test_reap_stuck_dispatch_jobs_starts_replacement_worker(
    db: Session, monkeypatch
) -> None:
    import time as time_module

    from app.core.config import settings as app_settings

    item = create_random_item(db)
    robot = create_random_robot(db)
    service = RobotService()
    monkeypatch.setattr(
        app_settings,
        "ROBOT_BACKEND_DISPATCH_HARD_TIMEOUT_SECONDS",
        0.1,
    )
    service._ensure_dispatch_workers()
    initial_workers = service._dispatch_worker_count
    job = robot_service_module_queued_job(robot, item, "stuck-replacement")
    with service._lock:
        service._active_dispatch_jobs["stuck-replacement"] = (
            time_module.monotonic() - 10,
            job,
        )
    monkeypatch.setattr(
        service,
        "_record_and_send_job_error",
        lambda *args, **kwargs: None,
    )

    reaped = service.reap_stuck_dispatch_jobs()

    assert reaped == 1
    assert service._dispatch_worker_count == initial_workers + 1


def test_record_and_send_job_error_reaches_qq_before_state_clear(
    db: Session, monkeypatch
) -> None:
    from dataclasses import replace as dataclass_replace

    from app.plugins.robot import bridge_client

    item = create_random_item(db)
    robot = create_random_robot(db)
    item.owner_id = robot.owner_id
    db.add(item)
    db.commit()
    service = RobotService()
    generation = service._begin_reply_context_dispatch(robot, "group:g1")
    job = robot_service_module_queued_job(robot, item, "error-1")
    job = dataclass_replace(
        job,
        conversation_generation=generation,
        reply_requires_awake=True,
    )
    sent: list[str] = []
    monkeypatch.setattr(
        bridge_client.robot_bridge_client,
        "send_message",
        lambda _robot_id, _target, message: sent.append(message),
    )
    monkeypatch.setattr(
        service,
        "_enqueue_pending_chat_followup",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        service,
        "_remember_assistant_conversation_memory",
        lambda *args, **kwargs: None,
    )

    service._record_and_send_job_error(job, "任务处理超时，已强制中止。请重新发一次。")

    # 内部失败文本不直发群：QQ 零发送，失败详情派生为纠正轮输入，由 agent
    # 自己用大白话告诉用户（或判断为内部噪音静默）。
    assert sent == []
    key = service._conversation_controller_key(robot.id, "group:g1")
    controller = service._conversation_controllers[key]
    assert controller.processing is False
    assert controller.sleeping is True
    pending_texts = service._pending_chat_message_texts(robot.id, "group:g1")
    assert pending_texts and "超时" in pending_texts[0]
    assert "上一轮处理失败" in pending_texts[0]


def test_think_blocks_are_stripped_from_visible_text() -> None:
    from app.plugins.robot.internal_trace import (
        is_robot_internal_trace_text,
        sanitize_robot_visible_text,
    )

    assert (
        sanitize_robot_visible_text("<think>用户问我装没装好，查一下记忆</think>装好啦，openjdk 21")
        == "装好啦，openjdk 21"
    )
    assert sanitize_robot_visible_text("</think>") == ""
    assert sanitize_robot_visible_text("结果\n</think>\n") == "结果"
    assert is_robot_internal_trace_text("</think>") is True
    assert is_robot_internal_trace_text("<think>想了一下</think>") is True
    assert is_robot_internal_trace_text("在呢") is False

def test_term_command_without_text_switches_default_terminal(
    db: Session,
    monkeypatch,
) -> None:
    item_a = create_random_item(db)
    item_b = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item_a.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item_b.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="beta",
            is_default_target=False,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    switch = robot_service.handle_inbound_message(
        db,
        robot,
        _message("/term beta"),
    )
    assert switch.success is True
    assert switch.item_id == str(item_b.id)
    assert switch.route_key == "beta"
    assert any("已切换" in chunk for chunk in switch.reply_chunks)
    assert "job" not in captured

    follow_up = robot_service.handle_inbound_message(
        db,
        robot,
        _message("现在默认是哪个终端？", mentioned_bot=True),
    )
    assert follow_up.success is True
    assert follow_up.item_id == str(item_b.id)
    assert captured["job"].message == "现在默认是哪个终端？"

def test_term_list_shows_bound_terminals_and_switch_hint(
    db: Session,
    monkeypatch,
) -> None:
    item_a = create_random_item(db)
    item_b = create_random_item(db)
    robot = create_random_robot(db)
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item_a.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="alpha",
            is_default_target=True,
        )
    )
    db.add(
        RobotItem(
            robot_id=robot.id,
            item_id=item_b.id,
            allow_chat=True,
            receive_filtered_output=False,
            chat_alias="beta",
            is_default_target=False,
        )
    )
    db.commit()

    captured = _capture_queued_chat(monkeypatch)

    response = robot_service.handle_inbound_message(
        db,
        robot,
        _message("/term list"),
    )
    assert response.success is True
    assert response.item_id == str(item_a.id)
    listing = "\n".join(response.reply_chunks)
    assert "alpha" in listing
    assert "beta" in listing
    assert "默认" in listing
    assert "/term <别名>" in listing
    assert "job" not in captured

    # After switching the default to beta, /term list marks beta as current.
    robot_service.handle_inbound_message(db, robot, _message("/term beta"))
    follow = robot_service.handle_inbound_message(db, robot, _message("/term ls"))
    assert follow.item_id == str(item_b.id)
    assert "当前" in "\n".join(follow.reply_chunks)
