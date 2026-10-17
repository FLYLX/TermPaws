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

# 首轮宽限期：票据创建到第一笔 update_plan 之间可能有很长一段工具执行
# （复杂任务几分钟都正常）。窗口内的票据绝不被同会话新消息顶掉。
QQ_TICKET_FIRST_TURN_GRACE_SECONDS = 240.0
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
    kwargs["inherited_debts"] = [
        dict(entry) for entry in (kwargs.get("inherited_debts") or []) if isinstance(entry, dict)
    ]
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
    # 用户取消的墓碑：取消后轮次可能还在半空（LLM 思考/工具执行中），
    # 它的迟到 update_plan 会把计划重新建起来——sweep 续跑 → 任务复活。
    # 立了碑的票据拒绝一切计划写入，取消就是终态。
    plan_cancelled: bool = False
    # run_job 启动的 daemon 后台任务 id 列表（同一票据可并行多个 job）。
    # 结果由 backend 轮询线程送回，但进程重启后线程消亡——持久化这些 id
    # 让 resume sweep 能重新接管：向 daemon 拉取已完成的结果并注入。
    pending_job_ids: list[str] = field(default_factory=list)
    # 执行证据账本：关键动作的极简一行式记录（job 结果、QQ 送达状态）。
    # 收尾审计时回显给模型自查——证据链不依赖会滚动的对话窗口。
    action_ledger: list[str] = field(default_factory=list)
    # 最终汇报是否卡在通道故障里（未送达）。发送成功清除；票据被
    # supersede 时也清除（新轮次接管汇报义务）。
    report_undelivered: bool = False
    # 接管来的汇报义务：本票据 supersede 了别的 QQ 票据时，把对方
    # 尚未回复的来源路由+原请求搬到这里（结构跟随任务，不随旧票据消失）。
    # 每条 {target_type, target_id, sender_label, request, from_ticket_id}；
    # 向对应 target 发送成功后由 mark_debt_fulfilled 核销。
    inherited_debts: list[dict] = field(default_factory=list)
    # 内容完全相同的活跃票据 id（同会话同文本在飞来）。只作信息提示，
    # 是否按重复处理由模型判断。
    duplicate_of: str = ""

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
            # 长任务保护：计划未完成或还有后台 job 在等的票据不是"卡死",
            # 只是活得久——删除会让 6h+ 的任务在任何新消息到来时凭空消失。
            and not plan_has_open_steps(ticket.plan)
            and not ticket.pending_job_ids
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
            # 年轻票据可能正处在"已创建、还没写下第一笔计划"的首轮窗口里
            # （任务越复杂这个窗口越长）。直接关掉会把正在跑的轮次的回复
            # 路由埋掉——跳过它，让它自己的轮次收尾。
            age_seconds = (now - ticket.created_at).total_seconds()
            if age_seconds < QQ_TICKET_FIRST_TURN_GRACE_SECONDS:
                continue
            # 有执行痕迹（ledger/命令/后台 job）的老票据：任务跑过却没汇报，
            # 回复义务转移给新票据，由模型判断补回还是说明。
            has_traces = bool(
                ticket.action_ledger or ticket.command or ticket.pending_job_ids
            )
            if ticket.external_report_sent:
                ticket.status = "delivered"
                ticket.delivered_at = now
            elif has_traces:
                ticket.status = "failed"
                ticket.delivery_error = QQ_TICKET_SUPERSEDE_REASON
                if isinstance(ticket.reply_target, dict) and ticket.reply_target.get("target_id"):
                    new_ticket.inherited_debts.append({
                        "target_type": str(ticket.reply_target.get("target_type") or ""),
                        "target_id": str(ticket.reply_target.get("target_id") or ""),
                        "sender_label": str(ticket.sender_label or ""),
                        "request": str(ticket.request_message or "")[:200],
                        "from_ticket_id": ticket.ticket_id,
                    })
                    new_ticket.updated_at = now
                    _persist_ticket(new_ticket)
            else:
                ticket.status = "failed"
                ticket.delivery_error = QQ_TICKET_SUPERSEDE_REASON
            # 新轮次接管汇报义务，旧票据不再算"未送达"
            ticket.report_undelivered = False
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

    def create_detached(
        self,
        *,
        item_id: str,
        handler_id: str,
        message: str = "",
        alias_of: str = "",
    ) -> ReplyTicket:
        """Agent-free ticket for flows that outlive their original ticket
        (plans, job chains). When alias_of is given, lookups of the dead
        ticket id resolve to the new one, so in-flight tool args keep working."""
        now = datetime.now()
        ticket = ReplyTicket(
            ticket_id=uuid.uuid4().hex,
            item_id=str(item_id),
            handler_id=str(handler_id),
            source_type=SOURCE_WEB,
            source_label="ticket renewal",
            request_message=str(message or "").strip()[:500],
        )
        with self._lock:
            self._prune_locked(now)
            self._tickets[ticket.ticket_id] = ticket
            if alias_of:
                self._ticket_aliases[str(alias_of)] = ticket.ticket_id
                _persist_aliases(self._ticket_aliases)
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
        self, ticket_id: str, plan_list: list[dict], *, force: bool = False
    ) -> ReplyTicket | None:
        """Write the agent plan scratchpad onto a ticket and persist it."""
        if not ticket_id:
            return None
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return None
            if ticket.plan_cancelled and not force:
                return None
            ticket.plan = [dict(entry) for entry in plan_list or []]
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)
            return ticket

    def cancel_ticket_plan(self, ticket_id: str) -> None:
        """取消计划的唯一入口：立碑 + 清空 + 落盘，一次完成。"""
        if not ticket_id:
            return
        with self._lock:
            ticket = self._tickets.get(self._resolve_ticket_id_locked(ticket_id))
            if ticket is None:
                return
            ticket.plan_cancelled = True
            ticket.plan = []
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def cancel_plan(self, ticket_id: str, *, explanation: str = "stopped") -> list[dict] | None:
        """终止票据上的计划：步骤标记取消、清除、持久化、广播。

        轮次被硬性停止（循环/预算耗尽）时使用——消息告诉用户"任务已结束"
        就必须真的结束，否则计划保持未完成，resume sweep 会反复复活它。
        """
        return self.supersede_plan(ticket_id, "", explanation=explanation)

    def supersede_plan(
        self,
        ticket_ref: str,
        item_id: str,
        *,
        explanation: str = "superseded",
        inherit_to_ticket_id: str = "",
    ) -> list[dict] | None:
        """关闭同一任务旧票据上的计划（agent 接管任务时调用）。

        支持完整 id 或唯一前缀（提示词里只暴露前 8 位）。步骤标记
        cancelled 后清除并持久化+广播——旧计划立即从条带消失，
        不会被 resume sweep 续跑。无匹配返回 None。

        inherit_to_ticket_id 给出时，若旧票据是还没回复来源的 QQ 票据，
        把它的回复义务（路由+原请求）转移到接管票据的 inherited_debts，
        义务跟着任务走，不随旧票据消失。
        """
        ref = str(ticket_ref or "").strip()
        if not ref:
            return None
        with self._lock:
            ticket = self._tickets.get(self._resolve_ticket_id_locked(ref))
            if ticket is None and item_id:
                matches = [
                    t
                    for t in self._tickets.values()
                    if t.item_id == str(item_id)
                    and t.ticket_id.startswith(ref)
                    and t.plan
                ]
                ticket = matches[0] if len(matches) == 1 else None
            if ticket is None or not ticket.plan:
                return None
            finalized = [
                {
                    **step,
                    "status": (
                        step.get("status")
                        if str(step.get("status")) in {"completed", "cancelled"}
                        else "cancelled"
                    ),
                }
                for step in ticket.plan
            ]
            ticket.plan = []
            ticket.updated_at = datetime.now()

            inherit = self._tickets.get(
                self._resolve_ticket_id_locked(inherit_to_ticket_id)
            ) if inherit_to_ticket_id else None
            if (
                inherit is not None
                and inherit.ticket_id != ticket.ticket_id
                and ticket.is_qq
                and not ticket.external_report_sent
                and ticket.status not in {"delivered", "sending"}
                and isinstance(ticket.reply_target, dict)
                and ticket.reply_target.get("target_id")
            ):
                debt = {
                    "target_type": str(ticket.reply_target.get("target_type") or ""),
                    "target_id": str(ticket.reply_target.get("target_id") or ""),
                    "sender_label": str(ticket.sender_label or ""),
                    "request": str(ticket.request_message or "")[:200],
                    "from_ticket_id": ticket.ticket_id,
                }
                known = {
                    (str(d.get("target_type")), str(d.get("target_id")), str(d.get("from_ticket_id")))
                    for d in inherit.inherited_debts
                }
                key = (debt["target_type"], debt["target_id"], debt["from_ticket_id"])
                if key not in known:
                    inherit.inherited_debts.append(debt)
                    inherit.action_ledger = (
                        inherit.action_ledger
                        + [f"接管汇报义务：QQ {debt['target_type']}:{debt['target_id']}（票据 {ticket.ticket_id[:8]}）"]
                    )[-12:]
                    inherit.updated_at = datetime.now()
                    _persist_ticket(inherit)
                    ticket.action_ledger = (
                        ticket.action_ledger
                        + [f"汇报义务已移交给票据 {inherit.ticket_id[:8]}"]
                    )[-12:]
                    _persist_ticket(ticket)
        try:
            from app.services.agent.stream_manager import stream_manager

            stream_manager.broadcast_chat_event(
                ticket.item_id,
                {
                    "type": "plan_updated",
                    "item_id": ticket.item_id,
                    "plan": finalized,
                    "explanation": explanation,
                },
            )
        except Exception:
            pass
        return finalized

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

    def record_action(self, ticket_id: str, line: str, *, cap: int = 12) -> None:
        """追加一条极简执行证据（一行），超出上限丢最旧的。"""
        line = str(line or "").strip()
        if not line:
            return
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return
            ticket.action_ledger = (ticket.action_ledger + [line])[-cap:]
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def mark_report_undelivered(self, ticket_id: str, error: str = "") -> None:
        """QQ 汇报发送失败（通道故障）：立起"未送达"标记，直到补发成功。"""
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket:
                return
            ticket.report_undelivered = True
            detail = str(error or "").strip()[:60]
            line = f"QQ 汇报发送失败{('：' + detail) if detail else ''}（未送达）"
            ticket.action_ledger = (ticket.action_ledger + [line])[-12:]
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def mark_report_delivered(self, ticket_id: str) -> None:
        """QQ 发送成功：清除"未送达"标记。"""
        with self._lock:
            ticket_id = self._resolve_ticket_id_locked(ticket_id)
            ticket = self._tickets.get(ticket_id)
            if not ticket or not ticket.report_undelivered:
                return
            ticket.report_undelivered = False
            ticket.action_ledger = (ticket.action_ledger + ["QQ 汇报已送达"])[-12:]
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def undelivered_reports_for_item(self, item_id: str) -> list[dict[str, Any]]:
        """该 item 里汇报卡在通道故障中的票据（供 prompt 固定注入）。"""
        with self._lock:
            return [
                {"ticket_id": ticket.ticket_id, "request": ticket.request_message}
                for ticket in self._tickets.values()
                if ticket.item_id == str(item_id) and ticket.report_undelivered
            ]

    def find_active_duplicate(
        self,
        *,
        robot_id: str,
        conversation_key: str,
        request_message: str,
        exclude_ticket_id: str = "",
    ) -> ReplyTicket | None:
        """同会话里内容完全相同且尚未收尾的 QQ 票据（供重复请求提示）。

        只做精确文本相等判断并返回事实，是否按重复处理由模型决定。
        """
        text = str(request_message or "").strip()
        if not (robot_id and conversation_key and text):
            return None
        with self._lock:
            candidates = [
                t
                for t in self._tickets.values()
                if t.source_type == SOURCE_QQ
                and t.robot_id == str(robot_id)
                and t.conversation_key == str(conversation_key)
                and t.ticket_id != exclude_ticket_id
                and t.status not in {"delivered", "failed"}
                and str(t.request_message or "").strip() == text
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda t: t.created_at)
            return candidates[0]

    def mark_duplicate_of(self, ticket_id: str, other_ticket_id: str) -> None:
        with self._lock:
            ticket = self._tickets.get(self._resolve_ticket_id_locked(ticket_id))
            other = self._tickets.get(self._resolve_ticket_id_locked(other_ticket_id))
            if ticket is None or other is None or ticket.ticket_id == other.ticket_id:
                return
            ticket.duplicate_of = other.ticket_id
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def active_sibling_summary(
        self,
        *,
        robot_id: str,
        conversation_key: str,
        exclude_ticket_id: str = "",
    ) -> list[dict[str, Any]]:
        """同会话里仍在飞的其它 QQ 票据摘要（供票据 prompt 注入）。

        同一件事被拆成多张票据（重发、乱码重发、补充追问）时，各票据互相
        看不见就会重复执行/重复汇报。这里只提供事实清单，合并、接管还是
        各回各的，由模型判断。
        """
        if not (robot_id and conversation_key):
            return []
        with self._lock:
            siblings = []
            for t in self._tickets.values():
                if (
                    t.source_type != SOURCE_QQ
                    or t.robot_id != str(robot_id)
                    or t.conversation_key != str(conversation_key)
                    or t.ticket_id == exclude_ticket_id
                    or t.status in {"delivered", "failed", "sending"}
                ):
                    continue
                open_steps = [
                    str(step.get("step") or "")[:40]
                    for step in t.plan
                    if str(step.get("status") or "") not in {"completed", "cancelled"}
                ]
                siblings.append({
                    "ticket_id": t.ticket_id[:8],
                    "request": str(t.request_message or "")[:60],
                    "open_steps": open_steps,
                    "reported": bool(t.external_report_sent),
                })
            siblings.sort(key=lambda s: s["ticket_id"])
            return siblings

    def mark_debt_fulfilled(
        self, ticket_id: str, *, target_type: str, target_id: str
    ) -> None:
        """向某 target 发送成功后，核销本票据上对应该来源的接管债务。"""
        with self._lock:
            ticket = self._tickets.get(self._resolve_ticket_id_locked(ticket_id))
            if ticket is None or not ticket.inherited_debts:
                return
            remaining = [
                d
                for d in ticket.inherited_debts
                if not (
                    str(d.get("target_type") or "") == str(target_type)
                    and str(d.get("target_id") or "") == str(target_id)
                )
            ]
            if len(remaining) == len(ticket.inherited_debts):
                return
            ticket.inherited_debts = remaining
            ticket.action_ledger = (
                ticket.action_ledger
                + [f"接管汇报已送达：QQ {target_type}:{target_id}"]
            )[-12:]
            ticket.updated_at = datetime.now()
            _persist_ticket(ticket)

    def pending_report_debts(self, item_id: str) -> list[dict[str, Any]]:
        """该 item 当前所有未了结的 QQ 汇报义务（供 prompt 注入，模型自行判断）。

        覆盖三种来源：
        - undelivered：汇报发送失败（通道故障），等待补发；
        - inherited：本票据接管了别的票据的任务，对方来源还没收到结果；
        - open_task：QQ 票据任务在跑且尚未向来源汇报（合并轮次里提醒各回各家）。
        """
        debts: list[dict[str, Any]] = []
        with self._lock:
            tickets = [
                t for t in self._tickets.values() if t.item_id == str(item_id)
            ]
            for ticket in tickets:
                if ticket.report_undelivered and ticket.is_qq:
                    debts.append({
                        "kind": "undelivered",
                        "target_type": str(ticket.reply_target.get("target_type") or ""),
                        "target_id": str(ticket.reply_target.get("target_id") or ""),
                        "sender_label": str(ticket.sender_label or ""),
                        "request": str(ticket.request_message or "")[:120],
                        "ticket_id": ticket.ticket_id[:8],
                    })
                for debt in ticket.inherited_debts:
                    debts.append({
                        "kind": "inherited",
                        "target_type": str(debt.get("target_type") or ""),
                        "target_id": str(debt.get("target_id") or ""),
                        "sender_label": str(debt.get("sender_label") or ""),
                        "request": str(debt.get("request") or "")[:120],
                        "ticket_id": str(debt.get("from_ticket_id") or "")[:8],
                    })
                if (
                    ticket.is_qq
                    and plan_has_open_steps(ticket.plan)
                    and not ticket.external_report_sent
                    and ticket.status not in {"delivered", "failed", "sending"}
                ):
                    debts.append({
                        "kind": "open_task",
                        "target_type": str(ticket.reply_target.get("target_type") or ""),
                        "target_id": str(ticket.reply_target.get("target_id") or ""),
                        "sender_label": str(ticket.sender_label or ""),
                        "request": str(ticket.request_message or "")[:120],
                        "ticket_id": ticket.ticket_id[:8],
                    })
        seen: set[tuple[str, str, str]] = set()
        unique: list[dict[str, Any]] = []
        for entry in debts:
            key = (entry["kind"], entry["target_type"], entry["target_id"])
            if entry["kind"] != "open_task" and key in seen:
                continue
            seen.add(key)
            unique.append(entry)
        return unique


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
            # 真实的投递失败（非 supersede/看门狗收尾）意味着汇报没送到用户——
            # 立起未送达标记，让固定注入持续提醒补发，直到成功。
            error_text = str(error or "")
            if error_text and error_text != QQ_TICKET_SUPERSEDE_REASON and "watchdog" not in error_text:
                ticket.report_undelivered = True
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
