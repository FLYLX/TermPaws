"""sweep 救捞停放的后台 job 结果标记：从死 store 转成可处理的追问输入。"""

import uuid
from datetime import datetime

from sqlmodel import Session

from app.plugins.robot.contracts import RobotReplyTarget
from app.plugins.robot.service import (
    PendingRobotTaskReply,
    robot_service,
)
from tests.utils.item import create_random_item  # noqa: E402
from tests.utils.robot import create_random_robot  # noqa: E402


def test_sweep_rescues_parked_task_replies(db: Session, monkeypatch) -> None:
    robot = create_random_robot(db)
    robot.is_enabled = True
    db.add(robot)
    db.commit()
    conversation_key = "group:sweep-rescue"
    reply_target = RobotReplyTarget(
        target_type="group",
        target_id="sweep-rescue",
        conversation_key=conversation_key,
    )
    key = robot_service._pending_chat_key(robot.id, conversation_key)

    parked = PendingRobotTaskReply(
        pending_id="parked-1",
        item_id=uuid.uuid4(),
        route_key="background_job",
        command="sleep 999 && echo done",
        sender_key="onebot_v11:group:sweep-rescue:42",
        sender_label="tester",
        trigger_reason="background_job",
        reply_target=reply_target,
        enqueued_at=datetime.now(),
    )
    with robot_service._lock:
        robot_service._pending_task_replies[key] = [parked]

    kicked: list[str] = []
    monkeypatch.setattr(
        robot_service,
        "_enqueue_pending_chat_followup",
        lambda *, robot, conversation_key, direct_wakeup_only=False: (
            kicked.append(conversation_key) is None or True
        ),
    )
    monkeypatch.setattr(
        robot_service,
        "conversation_has_active_dispatch",
        lambda robot_id, conversation_key: False,
    )

    stats = robot_service.sweep_pending_conversations()

    assert kicked == [conversation_key]
    assert stats["pending_turns_kicked"] == 1
    # 停放标记已清除
    with robot_service._lock:
        assert not robot_service._pending_task_replies.get(key)
    # 且已转成 merge buffer 里的追问输入（正常 followup 会处理它）
    from app.services.agent.input_merge_buffer import input_merge_buffer

    scope_item, scope_key = robot_service._pending_chat_scope(
        robot.id, conversation_key
    )
    entries = input_merge_buffer.peek(scope_item, scope_key)
    assert len(entries) == 1
    assert "sleep 999" in entries[0].content
    assert "list_jobs" in entries[0].content
