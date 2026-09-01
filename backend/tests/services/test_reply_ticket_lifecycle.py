from datetime import datetime, timedelta
from types import SimpleNamespace

from app.plugins.robot.api import (
    _merge_active_reply_tickets_into_controller_snapshots,
)
from app.plugins.robot.contracts import RobotReplyTarget
from app.plugins.robot.mcp.context import (
    RobotMCPContext,
    register_robot_mcp_context,
    unregister_robot_mcp_context,
)
from app.services.agent.reply_ticket import ReplyTicketManager


def test_qq_reply_ticket_snapshot_keeps_request_and_route_until_delivery() -> None:
    target = RobotReplyTarget(
        target_type="group",
        target_id="770362397",
        metadata={
            "conversation": {"type": "group", "id": "770362397"},
            "sender": {
                "user_id": "2537134688",
                "display_name": "FLY",
            },
            "message": {
                "raw_message": "[CQ:at,qq=2900669542] install temurin 17"
            },
        },
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-1",
            sender_key="onebot_v11:group:770362397:2537134688",
            reply_target=target,
            conversation_key="group:770362397",
            conversation_generation=3,
        )
    )
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="robot-1",
            robot_context_token=token,
            reply_ticket_id="",
        )
    )
    manager = ReplyTicketManager()

    try:
        ticket = manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="fallback prompt",
            source_type="qq",
        )
        manager.mark_task_plan(ticket.ticket_id, "plan-1")
        manager.mark_completed(ticket.ticket_id)

        snapshot = manager.snapshot("item-1")
        assert len(snapshot) == 1
        assert snapshot[0]["status"] == "completed"
        assert snapshot[0]["request_message"] == "install temurin 17"
        assert snapshot[0]["sender_label"] == "FLY (2537134688)"
        assert snapshot[0]["conversation_key"] == "group:770362397"

        manager.mark_delivered(ticket.ticket_id)
        assert manager.snapshot("item-1")[0]["status"] == "delivered"
    finally:
        unregister_robot_mcp_context(token)


def test_controller_pending_list_removes_ticket_only_after_delivered(
    monkeypatch,
) -> None:
    base_ticket = {
        "ticket_id": "ticket-1",
        "item_id": "item-1",
        "source_type": "qq",
        "source_label": "QQ group:770362397",
        "status": "completed",
        "task_request_id": "plan-1",
        "request_message": "start the server",
        "command": "bash run.sh",
        "created_at": (datetime.now() - timedelta(seconds=10)).isoformat(),
        "updated_at": datetime.now().isoformat(),
        "delivery_error": "",
        "robot_id": "robot-1",
        "sender_key": "sender-1",
        "sender_label": "FLY (2537134688)",
        "conversation_key": "group:770362397",
        "conversation_generation": 3,
        "workflow": {
            "workflow_id": "workflow-java",
            "objective": "install Java 17",
            "status": "active",
            "current_step": "switch package source",
            "latest_progress": "default source did not contain the package",
            "blocker": "",
            "steps": [
                {
                    "step_id": "step-source",
                    "title": "switch package source",
                    "status": "running",
                    "attempts": 1,
                    "recovery": True,
                }
            ],
        },
    }

    monkeypatch.setattr(
        "app.plugins.robot.api.reply_ticket_manager.snapshot",
        lambda _item_id: [dict(base_ticket)],
    )
    pending_snapshots: dict[str, list[dict[str, object]]] = {}
    _merge_active_reply_tickets_into_controller_snapshots(
        pending_snapshots,
        item_id="item-1",
        robot_ids={"robot-1"},
    )

    pending = pending_snapshots["robot-1"][0]["pending_messages"]
    assert len(pending) == 1
    assert pending[0]["reply_ticket_id"] == "ticket-1"
    assert pending[0]["reply_ticket_status"] == "completed"
    assert pending[0]["task_request_id"] == "plan-1"
    assert pending[0]["workflow_objective"] == "install Java 17"
    assert pending[0]["workflow_current_step"] == "switch package source"
    assert pending[0]["workflow_steps"][0]["recovery"] is True

    monkeypatch.setattr(
        "app.plugins.robot.api.reply_ticket_manager.snapshot",
        lambda _item_id: [{**base_ticket, "status": "delivered"}],
    )
    delivered_snapshots: dict[str, list[dict[str, object]]] = {}
    _merge_active_reply_tickets_into_controller_snapshots(
        delivered_snapshots,
        item_id="item-1",
        robot_ids={"robot-1"},
    )

    assert delivered_snapshots == {}


def test_reply_ticket_pruning_drops_stale_tickets_regardless_of_status() -> None:
    manager = ReplyTicketManager()
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="",
            robot_context_token="",
            reply_ticket_id="",
        )
    )
    stale_pending = manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="stale pending request",
        source_type="web",
    )
    stale_pending.updated_at -= timedelta(hours=7)

    manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="new request",
        source_type="web",
    )
    # A ticket untouched for longer than the TTL will never be delivered and
    # would otherwise fake a permanent "processing" row in the UI.
    assert manager.get(stale_pending.ticket_id) is None

    fresh_pending = manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="fresh pending request",
        source_type="web",
    )
    manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="another request",
        source_type="web",
    )
    assert manager.get(fresh_pending.ticket_id) is fresh_pending


def test_deliver_skips_qq_resend_after_external_report_sent(monkeypatch) -> None:
    target = RobotReplyTarget(
        target_type="group",
        target_id="770362397",
        metadata={
            "conversation": {"type": "group", "id": "770362397"},
            "sender": {"user_id": "2537134688", "display_name": "FLY"},
        },
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-1",
            sender_key="onebot_v11:group:770362397:2537134688",
            reply_target=target,
            conversation_key="group:770362397",
            conversation_generation=3,
        )
    )
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="robot-1",
            robot_context_token=token,
            reply_ticket_id="",
        )
    )
    manager = ReplyTicketManager()
    try:
        ticket = manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="speedtest",
            source_type="qq",
        )
        manager.mark_external_report_sent(ticket.ticket_id)

        sent: list[tuple] = []
        monkeypatch.setattr(
            "app.plugins.robot.bridge_client.robot_bridge_client.send_message",
            lambda *args: sent.append(args),
        )

        # The result already reached QQ via the send tool, so this bookkeeping
        # delivery (workflow closure after a watchdog resume) must finalize
        # the ticket silently instead of re-sending a duplicate report.
        assert manager.deliver(ticket.ticket_id, "收尾旁白") is True
        assert sent == []
        assert manager.get(ticket.ticket_id).status == "delivered"
    finally:
        unregister_robot_mcp_context(token)


def _qq_ticket_manager(conversation_key: str = "group:770362397"):
    target = RobotReplyTarget(
        target_type="group",
        target_id="770362397",
        metadata={
            "conversation": {"type": "group", "id": "770362397"},
            "sender": {"user_id": "2537134688", "display_name": "FLY"},
            "message": {"raw_message": "[CQ:at,qq=2900669542] 5*5=?"},
        },
    )
    token = register_robot_mcp_context(
        RobotMCPContext(
            robot_id="robot-1",
            sender_key="onebot_v11:group:770362397:2537134688",
            reply_target=target,
            conversation_key=conversation_key,
            conversation_generation=3,
        )
    )
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="robot-1",
            robot_context_token=token,
            reply_ticket_id="",
        )
    )
    return token, agent, ReplyTicketManager()


def test_new_qq_ticket_supersedes_stale_undelivered_ticket() -> None:
    token, agent, manager = _qq_ticket_manager()
    try:
        first = manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="5*5=?",
            source_type="qq",
        )
        second = manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="5*5=?",
            source_type="qq",
        )
        # A retry/newer turn for the same conversation must close the older
        # undelivered ticket instead of leaving a stuck duplicate.
        assert manager.get(first.ticket_id).status == "failed"
        assert (
            manager.get(first.ticket_id).delivery_error
            == "superseded by newer QQ message"
        )
        assert manager.get(second.ticket_id).status == "pending"
    finally:
        unregister_robot_mcp_context(token)


def test_qq_ticket_with_task_plan_not_superseded() -> None:
    token, agent, manager = _qq_ticket_manager()
    try:
        first = manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="run a long task and tell me",
            source_type="qq",
        )
        manager.mark_task_plan(first.ticket_id, "plan-1")
        manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="are you there",
            source_type="qq",
        )
        # Tickets backing a background task must survive later chatter.
        assert manager.get(first.ticket_id).status == "running"
    finally:
        unregister_robot_mcp_context(token)


def test_qq_ticket_with_sent_report_finalized_on_supersede() -> None:
    token, agent, manager = _qq_ticket_manager()
    try:
        first = manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="5*5=?",
            source_type="qq",
        )
        manager.mark_external_report_sent(first.ticket_id)
        manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="6*6=?",
            source_type="qq",
        )
        # The answer already reached QQ; close as delivered, not failed.
        assert manager.get(first.ticket_id).status == "delivered"
    finally:
        unregister_robot_mcp_context(token)


def test_watchdog_closes_stale_orphan_qq_ticket(monkeypatch) -> None:
    from datetime import timezone

    from app.services.agent import task_watchdog

    token, agent, manager = _qq_ticket_manager()
    monkeypatch.setattr(
        "app.services.agent.reply_ticket.reply_ticket_manager",
        manager,
    )
    try:
        stale = manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="5*5=?",
            source_type="qq",
        )
        fresh = manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="6*6=?",
            source_type="qq",
        )
        # The supersede path already closed the first ticket; reopen it to
        # simulate a ticket whose turn never delivered anything.
        stale.status = "running"
        stale.delivery_error = ""
        stale.updated_at -= timedelta(minutes=11)

        stats = {"orphan_tickets_removed": 0, "stale_tickets_closed": 0}
        task_watchdog._remove_orphan_tickets(datetime.now(timezone.utc), stats)

        assert stats["stale_tickets_closed"] == 1
        closed = manager.get(stale.ticket_id)
        assert closed.status == "failed"
        # Ticket datetimes must stay naive local time, or every naive
        # comparison against them after a restart-restore raises TypeError.
        assert closed.updated_at.tzinfo is None
        assert manager.get(fresh.ticket_id).status == "pending"
    finally:
        unregister_robot_mcp_context(token)


def test_watchdog_never_touches_ticket_with_open_plan(monkeypatch) -> None:
    """计划还有未完成步骤的票据：watchdog 既不陈旧关闭也不 TTL 删除，
    哪怕 updated_at 早已超过阈值（长工具调用期间没有 plan 更新）。"""
    from datetime import timezone

    from app.services.agent import task_watchdog
    from app.services.agent.reply_ticket import TICKET_TTL

    token, agent, manager = _qq_ticket_manager()
    monkeypatch.setattr(
        "app.services.agent.reply_ticket.reply_ticket_manager",
        manager,
    )
    try:
        ticket = manager.create_for_agent(
            agent,
            item_id="item-1",
            handler_id="handler-1",
            message="install jdk",
            source_type="qq",
        )
        manager.update_ticket_plan(
            ticket.ticket_id, [{"step": "install", "status": "in_progress"}]
        )
        # 超过 10 分钟陈旧阈值：不应被关闭。
        ticket.updated_at -= timedelta(minutes=11)
        stats = {"orphan_tickets_removed": 0, "stale_tickets_closed": 0}
        task_watchdog._remove_orphan_tickets(datetime.now(timezone.utc), stats)
        assert stats == {"orphan_tickets_removed": 0, "stale_tickets_closed": 0}
        assert manager.get(ticket.ticket_id).status == "pending"

        # 超过 6h TTL：也不应被删除。
        ticket.updated_at -= TICKET_TTL + timedelta(hours=1)
        task_watchdog._remove_orphan_tickets(datetime.now(timezone.utc), stats)
        assert stats == {"orphan_tickets_removed": 0, "stale_tickets_closed": 0}
        assert manager.get(ticket.ticket_id) is not None

        # 计划跑完（plan 清空）后恢复普通票据语义：超时即删。
        manager.update_ticket_plan(ticket.ticket_id, [])
        ticket.updated_at -= TICKET_TTL + timedelta(hours=1)
        task_watchdog._remove_orphan_tickets(datetime.now(timezone.utc), stats)
        assert stats["orphan_tickets_removed"] == 1
        assert manager.get(ticket.ticket_id) is None
    finally:
        unregister_robot_mcp_context(token)


def test_touch_refreshes_ticket_activity() -> None:
    manager = ReplyTicketManager()
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="",
            robot_context_token="",
            reply_ticket_id="",
        )
    )
    ticket = manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="request",
        source_type="web",
    )
    ticket.updated_at -= timedelta(minutes=5)
    before = ticket.updated_at
    manager.touch(ticket.ticket_id)
    assert ticket.updated_at > before
    assert (datetime.now() - ticket.updated_at).total_seconds() < 5
    # 缺失/空票据安静跳过。
    manager.touch("missing-ticket")
    manager.touch("")


def test_supersede_plan_closes_old_plan_by_prefix() -> None:
    manager = ReplyTicketManager()
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="",
            robot_context_token="",
            reply_ticket_id="",
        )
    )
    old = manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="install java",
        source_type="web",
    )
    manager.update_ticket_plan(
        old.ticket_id,
        [
            {"step": "apt update", "status": "completed"},
            {"step": "install jdk", "status": "in_progress"},
        ],
    )
    # 完整 id 关闭
    finalized = manager.supersede_plan(old.ticket_id, "item-1")
    assert finalized is not None
    assert finalized[0]["status"] == "completed"
    assert finalized[1]["status"] == "cancelled"
    assert old.plan == []

    # 前缀匹配 + 幂等
    manager.update_ticket_plan(old.ticket_id, [{"step": "retry", "status": "pending"}])
    finalized = manager.supersede_plan(old.ticket_id[:8], "item-1")
    assert finalized is not None and finalized[0]["status"] == "cancelled"
    assert manager.supersede_plan(old.ticket_id[:8], "item-1") is None
    assert manager.supersede_plan("nonexistent", "item-1") is None
    # 其他 item 的同前缀票据不被误伤
    assert manager.supersede_plan(old.ticket_id, "item-other") is None


def test_cancel_plan_terminates_plan_permanently() -> None:
    """硬性停止后计划必须真的结束：取消后 resume sweep 不再选中它。"""
    from app.services.agent.reply_ticket import plan_has_open_steps

    manager = ReplyTicketManager()
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="",
            robot_context_token="",
            reply_ticket_id="",
        )
    )
    ticket = manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="task",
        source_type="web",
    )
    manager.update_ticket_plan(ticket.ticket_id, [{"step": "s", "status": "in_progress"}])
    assert plan_has_open_steps(ticket.plan)

    finalized = manager.cancel_plan(ticket.ticket_id)
    assert finalized is not None
    assert finalized[0]["status"] == "cancelled"
    assert ticket.plan == []
    assert not plan_has_open_steps(ticket.plan)
    # 幂等 + 不存在票据
    assert manager.cancel_plan(ticket.ticket_id) is None
    assert manager.cancel_plan("missing") is None


def test_prune_tolerates_aware_restored_timestamps() -> None:
    from datetime import timezone

    from app.services.agent.reply_ticket import ticket_from_payload, ticket_to_payload

    manager = ReplyTicketManager()
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="",
            robot_context_token="",
            reply_ticket_id="",
        )
    )
    ticket = manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="old request",
        source_type="web",
    )
    # Simulate a ticket whose timestamps were persisted as aware datetimes
    # (e.g. written by a buggy code path before a restart).
    ticket.updated_at = datetime.now(timezone.utc)
    restored = ticket_from_payload(ticket_to_payload(ticket))
    assert restored.updated_at.tzinfo is None
    manager._tickets[restored.ticket_id] = restored

    # create_for_agent runs _prune_locked with a naive now; it must not raise
    # "can't compare offset-naive and offset-aware datetimes".
    manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="new request",
        source_type="web",
    )


def test_ticket_plan_serialization_roundtrip_and_legacy_payload() -> None:
    from app.services.agent.reply_ticket import ticket_from_payload, ticket_to_payload

    manager = ReplyTicketManager()
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="",
            robot_context_token="",
            reply_ticket_id="",
        )
    )
    ticket = manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="request with plan",
        source_type="web",
    )
    ticket.plan = [
        {"step": "download", "status": "completed"},
        {"step": "install", "status": "in_progress"},
    ]

    restored = ticket_from_payload(ticket_to_payload(ticket))
    assert restored.plan == ticket.plan

    # Legacy payloads persisted before the plan field existed must load
    # with an empty plan instead of failing.
    legacy_payload = ticket_to_payload(ticket)
    legacy_payload.pop("plan", None)
    legacy = ticket_from_payload(legacy_payload)
    assert legacy.plan == []


def test_update_ticket_plan_and_latest_plan_for_item() -> None:
    manager = ReplyTicketManager()
    agent = SimpleNamespace(
        _context=SimpleNamespace(
            robot_id="",
            robot_context_token="",
            reply_ticket_id="",
        )
    )
    ticket = manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="first request",
        source_type="web",
    )
    plan = [{"step": "step one", "status": "in_progress"}]

    updated = manager.update_ticket_plan(ticket.ticket_id, plan)
    assert updated is ticket
    assert ticket.plan == plan
    assert manager.update_ticket_plan("missing-ticket", plan) is None

    found_ticket, found_plan = manager.latest_plan_for_item("item-1")
    assert found_ticket is ticket
    assert found_plan == plan
    # Mutating the returned plan must not leak into the ticket.
    found_plan.append({"step": "bogus", "status": "pending"})
    assert ticket.plan == plan

    # 新票据没有计划时，旧票据的计划仍是 item 的当前计划——
    # plan 跟着创建它的票据走，直到计划跑完。
    newer = manager.create_for_agent(
        agent,
        item_id="item-1",
        handler_id="handler-1",
        message="second request",
        source_type="web",
    )
    assert newer.plan == []
    found_ticket, found_plan = manager.latest_plan_for_item("item-1")
    assert found_ticket is ticket
    assert found_plan == plan

    # 票据 status 不影响计划可见性：delivered/failed 只代表发过回复，
    # 不代表任务结束。
    newer.status = "failed"
    ticket.status = "delivered"
    found_ticket, found_plan = manager.latest_plan_for_item("item-1")
    assert found_ticket is ticket
    assert found_plan == plan

    # 计划清空（全部终态）后才不再暴露。
    manager.update_ticket_plan(ticket.ticket_id, [])
    assert manager.latest_plan_for_item("item-1") == (None, [])
    assert manager.latest_plan_for_item("item-unknown") == (None, [])
