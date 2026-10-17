"""Watchdog that keeps agent task state tidy.

Closes stale orphan reply tickets, removes tickets past their TTL, dedupes
near-duplicate long-term memories, reaps stuck dispatch jobs, and sweeps
pending inputs so queued messages always start a turn.

All work is best-effort: individual failures are logged and never stop the
scan.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

# How often near-duplicate long-term memories are auto-merged per item.
MEMORY_DEDUP_INTERVAL_SECONDS = 1800.0
# Rule-based cluster merge threshold (no LLM involved).
MEMORY_DEDUP_SIMILARITY_THRESHOLD = 0.85
# Orphan reply tickets (no workflow, no task plan) stuck in a non-final status
# longer than this are closed instead of lingering until the 6h ticket TTL.
# Comfortably above ROBOT_BACKEND_JOB_TIMEOUT_SECONDS (240s), so a live turn
# always finishes first; if such a turn still delivers later, deliver() simply
# re-opens and finalizes the ticket.
ORPHAN_TICKET_STALE = timedelta(minutes=10)

_memory_dedup_last_run: dict[str, float] = {}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        # Naive datetimes in tickets are local time (datetime.now()).
        return value.astimezone(timezone.utc)
    return value


def _stale_timeout_seconds(status: str) -> float:
    if status == "waiting_job":
        return float(settings.AGENT_WATCHDOG_WAITING_JOB_STALE_SECONDS)
    if status == "blocked":
        # Legitimate external waits (awaiting a user/event) get a longer leash.
        return float(settings.AGENT_WATCHDOG_BLOCKED_STALE_SECONDS)
    return float(settings.AGENT_WATCHDOG_STALE_SECONDS)


def _close_stale_orphan_ticket(ticket: Any, stats: dict[str, int]) -> None:
    from app.services.agent.reply_ticket import _persist_ticket, reply_ticket_manager

    with reply_ticket_manager._lock:
        current = reply_ticket_manager._tickets.get(ticket.ticket_id)
        if current is None or current.status in {"delivered", "failed"}:
            return
        # ReplyTicket datetimes are naive local time (datetime.now()); writing
        # the watchdog's aware UTC now would poison the ticket and crash every
        # naive comparison after a restart-restore.
        closed_at = datetime.now()
        if current.external_report_sent:
            # The report already reached the source; finalize bookkeeping.
            current.status = "delivered"
            current.delivered_at = closed_at
        else:
            current.status = "failed"
            current.delivery_error = "stale pending reply ticket closed by watchdog"
            # QQ 票据有实际执行痕迹（ledger/命令）却从没回过来源：立"未送达"
            # 标记，让 prompt 固定注入提醒模型判断是否补发（纯静默闲聊票据
            # 没有执行痕迹，不在这里产生假欠账）。
            if (
                getattr(current, "is_qq", False)
                and (current.action_ledger or current.command)
            ):
                current.report_undelivered = True
                current.action_ledger = (
                    current.action_ledger
                    + ["票据超时关闭时仍未向来源汇报（待补发判断）"]
                )[-12:]
        current.updated_at = closed_at
        _persist_ticket(current)
    logger.info(
        "[TaskWatchdog] Closed stale orphan reply ticket=%s source=%s new_status=%s",
        ticket.ticket_id,
        ticket.source_type,
        current.status,
    )
    stats["stale_tickets_closed"] += 1


def _remove_orphan_tickets(now: datetime, stats: dict[str, int]) -> None:
    from app.services.agent.reply_ticket import (
        TICKET_TTL,
        plan_has_open_steps,
        reply_ticket_manager,
    )

    inactive_cutoff = now - TICKET_TTL
    stale_cutoff = now - ORPHAN_TICKET_STALE
    with reply_ticket_manager._lock:
        tickets = list(reply_ticket_manager._tickets.values())
    for ticket in tickets:
        # 计划还有未完成步骤的票据是活任务：长工具调用期间 updated_at
        # 由轮次 touch 刷新，但即便失手也绝不由 watchdog 关闭/删除——
        # 计划生命周期只由 plan 步骤自身决定（跑到 plan 为空为止）。
        if plan_has_open_steps(getattr(ticket, "plan", None)):
            continue
        updated_at = _as_utc(ticket.updated_at)
        if updated_at >= inactive_cutoff:
            if (
                ticket.status not in {"delivered", "failed"}
                and not ticket.task_request_id
                and updated_at < stale_cutoff
            ):
                _close_stale_orphan_ticket(ticket, stats)
            continue
        with reply_ticket_manager._lock:
            reply_ticket_manager._tickets.pop(ticket.ticket_id, None)
            reply_ticket_manager._remove_aliases_for_locked(ticket.ticket_id)
        from app.services.agent.reply_ticket import (
            _delete_persisted_tickets,
            _persist_aliases,
        )

        _delete_persisted_tickets([ticket.ticket_id])
        _persist_aliases(reply_ticket_manager._ticket_aliases)
        stats["orphan_tickets_removed"] += 1


def _list_all_item_ids() -> list[str]:
    from sqlmodel import Session, select

    from app.core.db import engine
    from app.models import Item

    with Session(engine) as db:
        return [str(item.id) for item in db.exec(select(Item)).all()]


def _dedupe_memory_clusters(now: datetime, stats: dict[str, int]) -> None:
    """Auto-merge near-duplicate long-term memories per handler (rule-based, no LLM).

    Runs at most once per MEMORY_DEDUP_INTERVAL_SECONDS per handler. Keeps the
    first memory of each near-duplicate cluster and deletes the rest, so
    recalled context does not fill up with repeated facts.
    """
    from app.services.agent.memory.scope import resolve_handler_id
    from app.services.agent.memory.vector_store import vector_store

    now_ts = now.timestamp()
    item_ids = _list_all_item_ids()
    handler_ids = {resolve_handler_id(item_id) for item_id in item_ids}
    handler_ids.discard("")
    known = handler_ids
    for stale_key in set(_memory_dedup_last_run) - known:
        _memory_dedup_last_run.pop(stale_key, None)
    for handler_id in handler_ids:
        last_run = _memory_dedup_last_run.get(handler_id, 0.0)
        if now_ts - last_run < MEMORY_DEDUP_INTERVAL_SECONDS:
            continue
        _memory_dedup_last_run[handler_id] = now_ts
        try:
            removed = vector_store.deduplicate_memories(
                handler_id, threshold=MEMORY_DEDUP_SIMILARITY_THRESHOLD
            )
            superseded = vector_store.supersede_by_memory_key(handler_id)
            removed += superseded
        except Exception as exc:
            logger.info(
                "[TaskWatchdog] Memory dedup failed for handler=%s: %s", handler_id, exc
            )
            continue
        if removed:
            stats["memories_deduplicated"] += removed
            logger.info(
                "[TaskWatchdog] Merged %s near-duplicate/keyed memories for handler=%s",
                removed,
                handler_id,
            )


def run_once(now: datetime | None = None) -> dict[str, int]:
    now = now or _utcnow()
    stats = {
        "orphan_tickets_removed": 0,
        "stale_tickets_closed": 0,
        "dispatch_reaped": 0,
        "memories_deduplicated": 0,
    }
    try:
        _dedupe_memory_clusters(now, stats)
    except Exception:
        logger.exception("[TaskWatchdog] Memory dedup pass failed")
    try:
        from app.services.agent.integrations.registry import get_agent_integrations

        for integration in get_agent_integrations():
            if hasattr(integration, "reap_stuck_dispatch_jobs"):
                stats["dispatch_reaped"] = integration.reap_stuck_dispatch_jobs()
    except Exception:
        logger.exception("[TaskWatchdog] Dispatch reaper failed")
    try:
        from app.services.agent.integrations.registry import get_agent_integrations

        for integration in get_agent_integrations():
            if hasattr(integration, "sweep_pending_inputs"):
                swept = integration.sweep_pending_inputs() or {}
                stats["pending_turns_kicked"] = (
                    stats.get("pending_turns_kicked", 0)
                    + int(swept.get("pending_turns_kicked") or 0)
                )
                stats["buffered_results_flushed"] = (
                    stats.get("buffered_results_flushed", 0)
                    + int(swept.get("buffered_results_flushed") or 0)
                )
    except Exception:
        logger.exception("[TaskWatchdog] Pending input sweep failed")
    try:
        _remove_orphan_tickets(now, stats)
    except Exception:
        logger.exception("[TaskWatchdog] Failed to remove orphan tickets")
    if stats["orphan_tickets_removed"] or stats["stale_tickets_closed"] or stats["memories_deduplicated"]:
        logger.info("[TaskWatchdog] Pass finished: %s", stats)
    return stats


class AgentTaskWatchdog:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop,
                name="TermPaws-agent-task-watchdog",
                daemon=True,
            )
            self._thread.start()
        logger.info(
            "[TaskWatchdog] Started (interval=%ss, idle_resume=%ss, stale=%ss, blocked_stale=%ss, waiting_job_stale=%ss)",
            settings.AGENT_WATCHDOG_INTERVAL_SECONDS,
            settings.AGENT_WATCHDOG_IDLE_RESUME_SECONDS,
            settings.AGENT_WATCHDOG_STALE_SECONDS,
            settings.AGENT_WATCHDOG_BLOCKED_STALE_SECONDS,
            settings.AGENT_WATCHDOG_WAITING_JOB_STALE_SECONDS,
        )

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=5)

    def _loop(self) -> None:
        interval = max(5.0, float(settings.AGENT_WATCHDOG_INTERVAL_SECONDS))
        while not self._stop_event.wait(interval):
            try:
                run_once()
            except Exception:
                logger.exception("[TaskWatchdog] Pass failed")


agent_task_watchdog = AgentTaskWatchdog()
