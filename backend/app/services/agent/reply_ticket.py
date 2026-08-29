from __future__ import annotations

import logging
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_QQ = "qq"
SOURCE_WEB = "web"
SOURCE_TERMINAL = "terminal"

TICKET_TTL = timedelta(hours=6)
# How long a ticket's plan stays visible to the /plan API after the ticket
# was last touched (plan updates and turn activity refresh it).
PLAN_LOOKUP_FRESH_SECONDS = 30 * 60
QQ_TICKET_SUPERSEDE_REASON = "superseded by newer QQ message"
CURRENT_QQ_MESSAGE_MARKER = "[Current QQ message]"
CQ_CODE_RE = re.compile(r"\[CQ:[^\]]+\]", re.IGNORECASE)

TICKET_ALIASES_KV_KEY = "reply_ticket_aliases"


def plan_has_open_steps(plan: Any) -> bool:
    """计划的唯一事实源：还有 pending/in_progress 步骤 = 任务没跑完。

    展示、轮内续跑、resume sweep、watchdog 保护全部只看这一个条件，
    不再耦合票据 status——delivered 只代表"发过一条回复"，不代表任务结束。
    """
    if not isinstance(plan, list) or not plan:
        return False
    return any(
        isinstance(step, dict)
        and str(step.get("status") or "").strip().lower() in {"pending", "in_progress"}
        for step in plan
    )


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif not value:
        return None
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        # Ticket datetimes are naive local time (datetime.now()) everywhere in
        # this module; an aware value (e.g. persisted by an older writer) must
        # be normalized or every naive comparison against it raises TypeError.
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


EXPLICIT_QQ_TARGET_RE = re.compile(
    r"(私聊|私|群聊|群组|群|group|private)\s*(?:聊|组)?\s*[:：]?\s*(\d{5,12})",
    re.IGNORECASE,
)


def extract_explicit_qq_targets(message: str) -> list[dict[str, str]]:
    """Extract explicitly named QQ targets from a user message, e.g.
    "发到私聊 2537134688 和群 770362397". Only long numeric ids count, so
    arithmetic like "1+1" can never be misread as a target."""
    targets: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in EXPLICIT_QQ_TARGET_RE.finditer(str(message or "")):
        keyword = match.group(1).lower()
        target_type = "private" if keyword in {"私聊", "私", "private"} else "group"
        target_id = match.group(2)
        key = f"{target_type}:{target_id}"
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            {"target_type": target_type, "target_id": target_id, "conversation": key}
        )
    return targets


def robot_send_has_explicit_destination(tool_args: dict[str, Any]) -> bool:
    """Whether an mcp_robot_send_message call carries an explicit destination.

    QQ-originated turns reply to the current conversation with no target args
    (the robot MCP server locks those turns to that conversation). Every other
    source has no "current QQ conversation", so the call must name where to go:
    target_type+target_id, a targets array, or a reply_to/conversation
    reference that the robot MCP server resolves against visible context
    targets. Anything else is a bare send with no destination.
    """
    if not isinstance(tool_args, dict):
        return False
    if str(tool_args.get("target_id") or "").strip():
        return True
    if str(tool_args.get("reply_to") or "").strip():
        return True
    if str(tool_args.get("conversation") or "").strip():
        return True
    targets = tool_args.get("targets")
    if isinstance(targets, list):
        return any(
            isinstance(target, dict) and str(target.get("target_id") or "").strip()
            for target in targets
        )
    return False


def ticket_to_payload(ticket: ReplyTicket) -> dict[str, Any]:
    payload = asdict(ticket)
    for key in ("created_at", "updated_at", "delivered_at"):
        value = payload.get(key)
        payload[key] = value.isoformat() if isinstance(value, datetime) else None
    return payload


def ticket_from_payload(payload: dict[str, Any]) -> ReplyTicket:
    allowed = {field_info.name for field_info in dataclass_fields(ReplyTicket)}
    kwargs = {key: value for key, value in dict(payload or {}).items() if key in allowed}
    kwargs["created_at"] = _parse_dt(kwargs.get("created_at")) or datetime.now()
    kwargs["updated_at"] = _parse_dt(kwargs.get("updated_at")) or datetime.now()
    kwargs["delivered_at"] = _parse_dt(kwargs.get("delivered_at"))
    kwargs["reply_target"] = dict(kwargs.get("reply_target") or {})
    return ReplyTicket(**kwargs)


def _persist_ticket(ticket: ReplyTicket | None) -> None:
    if ticket is None:
        return
    try:
        from app.services.agent import state_store

        state_store.save_ticket(ticket_to_payload(ticket))
    except Exception as exc:
        logger.debug("[ReplyTicket] Persist ticket failed: %s", exc)


def _delete_persisted_tickets(ticket_ids: list[str]) -> None:
    if not ticket_ids:
        return
    try:
        from app.services.agent import state_store

        for ticket_id in ticket_ids:
            state_store.delete_ticket(ticket_id)
    except Exception as exc:
        logger.debug("[ReplyTicket] Delete persisted tickets failed: %s", exc)


def _persist_aliases(aliases: dict[str, str]) -> None:
    try:
        from app.services.agent import state_store

        state_store.save_kv(TICKET_ALIASES_KV_KEY, {"aliases": dict(aliases)})
    except Exception as exc:
        logger.debug("[ReplyTicket] Persist ticket aliases failed: %s", exc)


def restore_tickets_from_store() -> int:
    try:
        from app.services.agent import state_store
    except Exception:
        return 0
    restored = 0
    with reply_ticket_manager._lock:
        for payload in state_store.load_tickets():
            try:
                ticket = ticket_from_payload(payload)
            except Exception:
                continue
            reply_ticket_manager._tickets[ticket.ticket_id] = ticket
            restored += 1
        aliases = state_store.load_kv(TICKET_ALIASES_KV_KEY).get("aliases")
        if isinstance(aliases, dict):
            reply_ticket_manager._ticket_aliases.update(
                {str(key): str(value) for key, value in aliases.items()}
            )
        # Drop tickets that went stale while the process was down; otherwise
        # they would be restored forever and fake "processing" in the UI.
        reply_ticket_manager._prune_locked(datetime.now())
    return restored


@dataclass
class ReplyTicket:
    ticket_id: str
    item_id: str
    handler_id: str
    source_type: str
    source_label: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    task_request_id: str = ""
    request_message: str = ""
    command: str = ""
    delivered_at: datetime | None = None
    delivery_error: str = ""
    robot_id: str = ""
    sender_key: str = ""
    sender_label: str = ""
    conversation_key: str = ""
    conversation_generation: int = 0
    reply_target: dict[str, Any] = field(default_factory=dict)
    scheduled_task_id: str = ""
    scheduled_execution_id: str = ""
    terminal_target: str = ""
    external_report_sent: bool = False
    # Explicit QQ targets extracted from the user's message (e.g. "发到私聊A
    # 和群B"). The ticket carries them so the agent can send to each one by
    # name instead of rediscovering targets itself.
    extra_targets: list[dict[str, str]] = field(default_factory=list)
    # Codex-style plan scratchpad ({step, status} entries) owned by this
    # ticket. Identity, lifecycle and cleanup all reuse the ticket's existing
    # mechanisms; old persisted payloads without this field default to [].
    plan: list[dict] = field(default_factory=list)
    # run_job 启动的 daemon 后台任务 id 列表（同一票据可并行多个 job）。
    # 结果由 backend 轮询线程送回，但进程重启后线程消亡——持久化这些 id
    # 让 resume sweep 能重新接管：向 daemon 拉取已完成的结果并注入。
    pending_job_ids: list[str] = field(default_factory=list)

    @property
    def is_qq(self) -> bool:
        return self.source_type == SOURCE_QQ and bool(self.robot_id and self.reply_target)


class ReplyTicketManager:
    def __init__(self) -> None:
        self._tickets: dict[str, ReplyTicket] = {}
        self._ticket_aliases: dict[str, str] = {}
        self._lock = threading.RLock()

    def _resolve_ticket_id_locked(self, ticket_id: str) -> str:
        current = str(ticket_id or "")
        visited: set[str] = set()
        while current in self._ticket_aliases and current not in visited:
            visited.add(current)
            current = self._ticket_aliases[current]
        return current

    def resolve_ticket_id(self, ticket_id: str) -> str:
        with self._lock:
            return self._resolve_ticket_id_locked(ticket_id)

    def _remove_aliases_for_locked(self, ticket_id: str) -> None:
        resolved = self._resolve_ticket_id_locked(ticket_id)
        stale_aliases = [
            alias
            for alias, target in self._ticket_aliases.items()
            if alias == resolved or self._resolve_ticket_id_locked(target) == resolved
        ]
        for alias in stale_aliases:
            self._ticket_aliases.pop(alias, None)

    def _prune_locked(self, now: datetime) -> None:
        cutoff = now - TICKET_TTL
        # Prune every ticket untouched for longer than the TTL, regardless of
        # status: a ticket stuck in pending/running/failed that long will
        # never be delivered (in-memory agent state cannot survive that), and
        # keeping it only fakes a permanent "processing" row in the UI.
        stale_ids = [
            ticket_id
            for ticket_id, ticket in self._tickets.items()
            if ticket.updated_at < cutoff
        ]
        for ticket_id in stale_ids:
            self._tickets.pop(ticket_id, None)
            self._remove_aliases_for_locked(ticket_id)
        _delete_persisted_tickets(stale_ids)
        _persist_aliases(self._ticket_aliases)

    def reset(self) -> None:
        with self._lock:
            _delete_persisted_tickets(list(self._tickets.keys()))
            self._tickets.clear()
            self._ticket_aliases.clear()
            _persist_aliases(self._ticket_aliases)

    @staticmethod
    def _robot_request_message(
        reply_target: Any,
        fallback: str,
        request_message: str = "",
    ) -> str:
        # Structured propagation from the robot dispatch queue carries the
        # real user request; fall back to bridge metadata, then raw text.
        text = str(request_message or "").strip()
        if not text:
            metadata = getattr(reply_target, "metadata", None)
            metadata = metadata if isinstance(metadata, dict) else {}
            native_message = metadata.get("message")
            native_message = native_message if isinstance(native_message, dict) else {}
            text = str(
                native_message.get("raw_message")
                or native_message.get("plain_text")
                or ""
            ).strip()
        if not text:
            raw_fallback = str(fallback or "")
            if CURRENT_QQ_MESSAGE_MARKER in raw_fallback:
                text = raw_fallback.rsplit(CURRENT_QQ_MESSAGE_MARKER, 1)[-1].strip()
            else:
                text = raw_fallback.strip()
        text = CQ_CODE_RE.sub(" ", text)
        return re.sub(r"\s+", " ", text).strip()[:500]

    @staticmethod
    def _robot_sender_label(reply_target: Any, sender_key: str) -> str:
        metadata = getattr(reply_target, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        sender = metadata.get("sender")
        sender = sender if isinstance(sender, dict) else {}
        sender_id = str(sender.get("user_id") or "").strip()
        display_name = str(
            sender.get("display_name")
            or sender.get("card")
            or sender.get("nickname")
            or sender_id
            or sender_key
            or ""
        ).strip()
        if display_name and sender_id and display_name != sender_id:
            return f"{display_name} ({sender_id})"
        return display_name or sender_key

    def _supersede_stale_qq_tickets_locked(self, new_ticket: ReplyTicket, now: datetime) -> None:
        """Close older undelivered QQ tickets for the same conversation.

        Every QQ turn creates a fresh ticket. Turns that finish without a QQ
        delivery (empty reply, corrective retry, superseded intent) would
        otherwise leave their ticket pending until the TTL prune, surfacing as
        stuck duplicate entries in the conversation controller snapshot.
        Tickets still backing a background task (task_request_id / workflow)
        or currently mid-delivery are left untouched; tickets whose report
        already reached QQ are finalized as delivered instead of failed.
        """
        if new_ticket.source_type != SOURCE_QQ or not new_ticket.conversation_key:
            return
        for ticket in list(self._tickets.values()):
            if (
                ticket.ticket_id == new_ticket.ticket_id
                or ticket.source_type != SOURCE_QQ
                or ticket.robot_id != new_ticket.robot_id
                or ticket.conversation_key != new_ticket.conversation_key
                or ticket.status in {"delivered", "failed", "sending"}
                or ticket.task_request_id
            ):
                continue
            # A ticket with a plan is a real in-flight task (parallel dispatch
            # from the same conversation); only idle follow-up turns get
            # superseded.
            if ticket.plan:
                continue
            if ticket.external_report_sent:
                ticket.status = "delivered"
                ticket.delivered_at = now
            else:
                ticket.status = "failed"
                ticket.delivery_error = QQ_TICKET_SUPERSEDE_REASON
            ticket.updated_at = now
            _persist_ticket(ticket)
            logger.info(
                "[ReplyTicket] Superseded stale QQ ticket=%s conversation=%s new_status=%s",
                ticket.ticket_id,
                ticket.conversation_key,
                ticket.status,
            )

    def create_for_agent(
        self,
        agent: Any,
        *,
        item_id: str,
        handler_id: str,
        message: str = "",
        source_type: str | None = None,
    ) -> ReplyTicket:
        now = datetime.now()
        forced_source_type = str(source_type or "").strip().lower()
        context = getattr(agent, "_context", None)
        robot_id = str(getattr(context, "robot_id", "") or "").strip()
        ticket = ReplyTicket(
            ticket_id=uuid.uuid4().hex,
            item_id=str(item_id),
            handler_id=str(handler_id),
            source_type=SOURCE_WEB,
            source_label="TermPaws web chat",
            request_message=str(message or "").strip()[:500],
        )
        ticket.extra_targets = extract_explicit_qq_targets(message)

        if forced_source_type == SOURCE_WEB:
            robot_id = ""

        if robot_id:
            try:
                from app.services.agent.integrations.hooks import (
                    get_integration_context_by_token,
                )

                robot_context = get_integration_context_by_token(
                    str(getattr(context, "robot_context_token", "") or "").strip()
                )
            except Exception:
                robot_context = None

            reply_target = getattr(robot_context, "reply_target", None)
            sender_key = str(getattr(robot_context, "sender_key", "") or "").strip()
            conversation_key = str(
                getattr(robot_context, "conversation_key", "") or ""
            ).strip()
            if reply_target is not None and hasattr(reply_target, "model_dump"):
                ticket.source_type = SOURCE_QQ
                ticket.source_label = f"QQ {conversation_key or 'current conversation'}"
                ticket.robot_id = robot_id
                ticket.sender_key = sender_key
                ticket.sender_label = self._robot_sender_label(
                    reply_target,
                    sender_key,
                )
                ticket.conversation_key = conversation_key
                ticket.conversation_generation = int(
                    getattr(robot_context, "conversation_generation", 0) or 0
                )
                ticket.reply_target = reply_target.model_dump(mode="json")
                ticket.request_message = self._robot_request_message(
                    reply_target,
                    message,
                    str(getattr(robot_context, "request_message", "") or ""),
                )

        if forced_source_type == SOURCE_QQ and ticket.source_type != SOURCE_QQ:
            logger.warning(
                "[ReplyTicket] Requested QQ source but no active robot context was available: item=%s handler=%s",
                item_id,
                handler_id,
            )

        with self._lock:
            self._prune_locked(now)
            self._supersede_stale_qq_tickets_locked(ticket, now)
            self._tickets[ticket.ticket_id] = ticket
        self.attach_to_agent(agent, ticket.ticket_id)
        _persist_ticket(ticket)
        return ticket

    def create_for_scheduled_task(
        self,
        agent: Any,
        *,
        item_id: str,
        handler_id: str,
        message: str,
        scheduled_task_id: str,
        scheduled_execution_id: str = "",
        request_message: str = "",
    ) -> ReplyTicket:
        ticket = self.create_for_agent(
            agent,
            item_id=item_id,
            handler_id=handler_id,
            message=message,
            source_type=SOURCE_WEB,
        )
        with self._lock:
            ticket.source_label = "TermPaws scheduled task"
            ticket.request_message = str(request_message or message or "").strip()[:4000]
            ticket.scheduled_task_id = str(scheduled_task_id or "").strip()
            ticket.scheduled_execution_id = str(
                scheduled_execution_id or ""
            ).strip()
            ticket.updated_at = datetime.now()
        _persist_ticket(ticket)
        return ticket

    @staticmethod
    def _sanitize_delivery_text(content: Any) -> str:
        try:
            from app.services.agent.integrations.hooks import (
                sanitize_integration_visible_text,
            )

            return sanitize_integration_visible_text(content).strip()
        except Exception:
            return str(content or "").strip()

    def create_for_terminal(        self,
        agent: Any,
        *,
        item_id: str,
        handler_id: str,
        player: str,
        message: str,
    ) -> ReplyTicket:
        ticket = ReplyTicket(
            ticket_id=uuid.uuid4().hex,
            item_id=str(item_id),
            handler_id=str(handler_id),
            source_type=SOURCE_TERMINAL,
            source_label=f"Minecraft player {player}",
            sender_key=str(player),
            sender_label=str(player),
            request_message=str(message or "")[:500],
            terminal_target=str(player),
        )
        now = datetime.now()
        with self._lock:
            self._prune_locked(now)
            self._tickets[ticket.ticket_id] = ticket
        self.attach_to_agent(agent, ticket.ticket_id)
        _persist_ticket(ticket)
        return ticket

    def attach_to_agent(self, agent: Any, ticket_id: str) -> None:
        context = getattr(agent, "_context", None)
        if context is not None:
            context.reply_ticket_id = self.resolve_ticket_id(ticket_id)

    def detach_from_agent(self, agent: Any, ticket_id: str) -> None:
        context = getattr(agent, "_context", None)
        if context is None:
            return
        resolved_ticket_id = self.resolve_ticket_id(ticket_id)
        if str(getattr(context, "reply_ticket_id", "") or "") in {
            str(ticket_id or ""),
            resolved_ticket_id,
        }:
            context.reply_ticket_id = ""

    def get(self, ticket_id: str) -> ReplyTicket | None:
        if not ticket_id:
            return None
        with self._lock:
            return self._tickets.get(self._resolve_ticket_id_locked(ticket_id))

    def update_ticket_plan(
        self, ticket_id: str, plan_list: list[dict]
    ) -> ReplyTicket | None:
        """Write the agent plan scratchpad onto a ticket and persist it."""
        if not ticket_id:
            return None
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return None
            ticket.plan = [dict(entry) for entry in plan_list or []]
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)
            return ticket

    def add_pending_job(self, ticket_id: str, job_id: str) -> None:
        if not ticket_id or not job_id:
            return
        with self._lock:
            ticket = self._tickets.get(self._resolve_ticket_id_locked(ticket_id))
            if ticket is None:
                return
            if str(job_id) not in ticket.pending_job_ids:
                ticket.pending_job_ids.append(str(job_id))
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def take_pending_job(self, ticket_id: str, job_id: str) -> bool:
        """从票据摘除该 job_id 并返回 True；不存在则 False。

        结果投递的 exactly-once 闸门：轮询线程和 resume sweep 都可能拿到
        已完成结果，谁摘成功谁负责投递，另一方直接跳过。
        """
        if not ticket_id or not job_id:
            return False
        with self._lock:
            ticket = self._tickets.get(self._resolve_ticket_id_locked(ticket_id))
            if ticket is None or str(job_id) not in ticket.pending_job_ids:
                return False
            ticket.pending_job_ids.remove(str(job_id))
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)
            return True

    def touch(self, ticket_id: str) -> None:
        """轮次活性信号：每轮迭代调用一次，刷新 updated_at。

        长工具调用（安装/编译等 run_job）期间没有任何 plan 更新，不刷新
        就会被 watchdog 的 10 分钟陈旧关闭器误判死亡。只更新内存，
        持久化由后续的 plan/status 写入顺带完成，避免每轮写盘。
        """
        if not ticket_id:
            return
        with self._lock:
            ticket = self._tickets.get(self._resolve_ticket_id_locked(ticket_id))
            if ticket is not None:
                ticket.updated_at = datetime.now()

    def latest_plan_for_item(
        self, item_id: str
    ) -> tuple[ReplyTicket | None, list[dict]]:
        """Plan of the item's most recently updated ticket that still has one.

        规则只有一条：计划还有未完成步骤就返回它。票据 status 不参与判断
        （delivered 只代表发过回复，不代表任务结束）；唯一的时间窗口是
        PLAN_LOOKUP_FRESH_SECONDS 遗弃保护，运行中的轮次会通过 touch 持续
        刷新 updated_at，不会误伤。
        """
        with self._lock:
            now = datetime.now()
            candidates = []
            for ticket in self._tickets.values():
                if ticket.item_id != str(item_id):
                    continue
                if not plan_has_open_steps(ticket.plan):
                    continue
                if (now - ticket.updated_at).total_seconds() > PLAN_LOOKUP_FRESH_SECONDS:
                    continue
                candidates.append(ticket)
        if not candidates:
            return None, []
        ticket = max(candidates, key=lambda entry: entry.updated_at)
        return ticket, list(ticket.plan)

    def active_plans_for_item(self, item_id: str) -> list[dict[str, Any]]:
        """该 item 所有未完成计划的票据（多任务展示用），按更新时间倒序。
        有未完成步骤就展示，步骤全部终态（plan 会被清空）即消失；
        票据 status 不参与过滤。PLAN_LOOKUP_FRESH_SECONDS 仅作遗弃保护。"""
        with self._lock:
            now = datetime.now()
            entries = []
            for ticket in self._tickets.values():
                if ticket.item_id != str(item_id):
                    continue
                if not plan_has_open_steps(ticket.plan):
                    continue
                idle_seconds = (now - ticket.updated_at).total_seconds()
                if idle_seconds > PLAN_LOOKUP_FRESH_SECONDS:
                    continue
                done = sum(
                    1
                    for step in ticket.plan
                    if str(step.get("status")) in {"completed", "cancelled"}
                )
                entries.append(
                    {
                        "ticket_id": ticket.ticket_id,
                        "request": ticket.request_message or ticket.command or "",
                        "status": ticket.status,
                        "updated_at": ticket.updated_at.isoformat(),
                        "done": done,
                        "total": len(ticket.plan),
                        "plan": list(ticket.plan),
                    }
                )
            entries.sort(key=lambda entry: entry["updated_at"], reverse=True)
            return entries

    def mark_task_plan(self, ticket_id: str, task_request_id: str) -> None:
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return
            ticket.task_request_id = str(task_request_id or "")
            ticket.status = "running"
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def mark_command(self, ticket_id: str, command: str) -> None:
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return
            ticket.command = str(command or "")
            ticket.status = "running"
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def mark_completed(self, ticket_id: str) -> None:
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return
            if ticket.status != "delivered":
                ticket.status = "completed"
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def mark_failed(self, ticket_id: str, error: str = "") -> None:
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return
            ticket.status = "failed"
            ticket.delivery_error = str(error or "")
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def mark_external_report_sent(self, ticket_id: str) -> None:
        """Record that a result for this ticket already reached its QQ source.

        Set when ``mcp_robot_send_message`` succeeded during a turn attached
        to this ticket while the workflow was not yet finalizable. Later
        bookkeeping deliveries (e.g. workflow closure after a watchdog
        resume) must not re-send a duplicate report to QQ.
        """
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket or ticket.external_report_sent:
                return
            ticket.external_report_sent = True
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def mark_delivered(self, ticket_id: str) -> bool:
        ticket_id = self.resolve_ticket_id(ticket_id)

        with self._lock:
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return False
            now = datetime.now()
            ticket.status = "delivered"
            ticket.delivered_at = now
            ticket.updated_at = now
            ticket.delivery_error = ""
            _persist_ticket(ticket)
        return True

    def build_prompt(self, ticket_id: str) -> str:
        ticket = self.get(ticket_id)
        if not ticket:
            return ""
        if ticket.scheduled_task_id:
            return (
                "Authoritative scheduled task ticket:\n"
                f"- ticket_id: {ticket.ticket_id}\n"
                f"- schedule_id: {ticket.scheduled_task_id}\n"
                f"- execution_id: {ticket.scheduled_execution_id or 'unknown'}\n"
                "- source: TermPaws scheduled task\n"
                "- treat terminal/background-job feedback as the continuation of this "
                "scheduled execution.\n"
                "- on failure, keep the task for transient failures. Delete it with "
                "mcp_local_delete_scheduled_task only if it is obsolete, invalid, unsafe, "
                "or permanently unable to succeed.\n"
                "- any visible report must be labelled as a scheduled task result.\n"
            )
        from app.services.agent.integrations.hooks import build_integration_ticket_prompt

        integration_prompt = build_integration_ticket_prompt(ticket)
        if integration_prompt:
            return integration_prompt
        if ticket.source_type == SOURCE_WEB:
            targets_hint = ""
            if ticket.extra_targets:
                listed = ", ".join(
                    f"{t.get('target_type')}:{t.get('target_id')}"
                    for t in ticket.extra_targets
                )
                targets_hint = (
                    f"- QQ TARGETS for this turn: {listed}. "
                    "\u7528\u6237\u660e\u786e\u8981\u6c42\u7684 QQ \u76ee\u6807\uff0c"
                    "\u9010\u4e2a\u8c03\u7528 mcp_robot_send_message\uff08\u663e\u5f0f target_type/target_id\uff09\u53d1\u9001\uff0c"
                    "\u6bcf\u4e2a\u76ee\u6807\u90fd\u8981\u53d1\u5230\uff1b\u5168\u90e8\u53d1\u5b8c\u540e\u518d\u5728\u8fd9\u91cc\u6c47\u62a5\u3002\n"
                )
            return (
                "Authoritative reply ticket:\n"
                f"- ticket_id: {ticket.ticket_id}\n"
                "- source: TermPaws web chat\n"
                f"{targets_hint}"
                "- REPLY ROUTING: this conversation originated from the web chat. "
                "Reply directly in this web response.\n"
                "- If the user asks you to send a message to QQ or elsewhere, perform "
                "that action, then report the outcome here in web chat.\n"
            )
        return (
            "Authoritative reply ticket:\n"
            f"- ticket_id: {ticket.ticket_id}\n"
            f"- source: {ticket.source_type}\n"
            "- REPLY ROUTING: reply back to the same source this conversation "
            "originated from.\n"
        )

    def deliver(self, ticket_id: str, content: str) -> bool:
        text = self._sanitize_delivery_text(content)
        if not ticket_id or not text:
            return False
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket or ticket.status == "delivered":
                return False
            if (
                ticket.external_report_sent
                and ticket.source_type == SOURCE_QQ
                and ticket.status != "failed"
            ):
                # The result already reached QQ via the send tool in an
                # earlier turn. This call is bookkeeping (e.g. workflow
                # closure after a watchdog resume); finalize silently instead
                # of sending a duplicate report. Failure reports (status
                # "failed") always go through.
                return self.mark_delivered(ticket_id)
            ticket.status = "sending"
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

        try:
            self._deliver_ticket_content(ticket, text)
        except Exception as exc:
            logger.warning(
                "[ReplyTicket] Failed to deliver ticket=%s source=%s: %s",
                ticket_id,
                ticket.source_label,
                exc,
            )
            with self._lock:
                current = self._tickets.get(ticket_id)
                if current:
                    current.status = "failed"
                    current.delivery_error = str(exc)
                    current.updated_at = datetime.now()
                _persist_ticket(current)
            return False

        self.mark_delivered(ticket_id)
        return True

    def _deliver_ticket_content(self, ticket: ReplyTicket, text: str) -> None:
        from app.services.agent.integrations.hooks import (
            deliver_integration_ticket,
            integration_conversation_memory_append,
        )

        if deliver_integration_ticket(ticket, text):
            integration_conversation_memory_append(ticket, text)
            return

        if ticket.source_type == SOURCE_WEB:
            from app.services.agent.history.chat import append_chat_message
            from app.services.agent.stream_manager import stream_manager

            event = append_chat_message(
                ticket.item_id,
                role="assistant",
                content=text,
                message_type="agent_response",
                extra={"pending_reply_id": ticket.ticket_id},
            )
            stream_manager.broadcast_chat_event(ticket.item_id, event)
            return

        raise ValueError(f"unsupported destination type: {ticket.source_type}")

    def snapshot(self, item_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            self._prune_locked(datetime.now())
            tickets = list(self._tickets.values())
        if item_id:
            tickets = [ticket for ticket in tickets if ticket.item_id == str(item_id)]
        tickets.sort(key=lambda ticket: ticket.updated_at, reverse=True)
        snapshots: list[dict[str, Any]] = []
        for ticket in tickets:
            snapshots.append(
                {
                    "ticket_id": ticket.ticket_id,
                    "item_id": ticket.item_id,
                    "handler_id": ticket.handler_id,
                    "source_type": ticket.source_type,
                    "source_label": ticket.source_label,
                    "status": ticket.status,
                    "task_request_id": ticket.task_request_id,
                    "request_message": ticket.request_message,
                    "command": ticket.command,
                    "created_at": ticket.created_at.isoformat(),
                    "updated_at": ticket.updated_at.isoformat(),
                    "delivered_at": ticket.delivered_at.isoformat()
                    if ticket.delivered_at
                    else None,
                    "delivery_error": ticket.delivery_error,
                    "scheduled_task_id": ticket.scheduled_task_id,
                    "scheduled_execution_id": ticket.scheduled_execution_id,
                    "pending_reply_status": "",
                    "pending_reply_requester": "",
                    "pending_reply_plan": [],
                    "pending_reply_awaiting_kind": "",
                    "pending_reply_awaiting_key": "",
                    "robot_id": ticket.robot_id,
                    "sender_key": ticket.sender_key,
                    "sender_label": ticket.sender_label,
                    "conversation_key": ticket.conversation_key,
                    "conversation_generation": ticket.conversation_generation,
                    "workflow": None,
                }
            )
        return snapshots


reply_ticket_manager = ReplyTicketManager()
