import { useQuery, useQueryClient } from "@tanstack/react-query"
import { Link } from "@tanstack/react-router"
import { Activity, Brain, ChevronDown, ChevronUp, Download, ExternalLink, ImagePlus, Loader2, Lock, Send, Square, Terminal, Trash2, X } from "lucide-react"
import {
  useEffect,
  useEffectEvent,
  useLayoutEffect,
  memo,
  useMemo,
  useRef,
  useState,
} from "react"

import type { ItemHandlerPublic } from "@/client"
import { ItemHandlerAssociationsService } from "@/client"
import { OpenAPI } from "@/client/core/OpenAPI"
import {
  advanceChatHistoryOffset,
  buildStableChatMessageRows,
  getChatMessageKey,
} from "@/components/Items/chatHistory"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"

type TimelineRole = "user" | "assistant" | "terminal"
type TerminalSource = "filtered_output" | "raw_feedback"
type AgentStatusKind =
  | "replying"
  | "thinking"
  | "tool"
  | "terminal_wait"
  | "terminal_job"
  | "terminal_analysis"

interface AgentStatusState {
  text: string
  source?: TerminalSource | null
  status?: string | null
  toolName?: string | null
  kind?: AgentStatusKind
  autoClearMs?: number
}

interface ChatMessage {
  role: TimelineRole
  content: string
  type?: string
  timestamp?: string
  tool_name?: string
  localEcho?: boolean
}

interface ChatPanelProps {
  itemId: string
}

const STATUS_ONLY_TYPES = new Set([
  "agent_status",
  "agent_action",
  "agent_tool_result",
  "session_summary",
])

const COMPLETION_TYPES = new Set([
  "agent_response",
  "chat_assistant",
  "agent_warning",
  "agent_error",
])

const LIVE_REPLY_STATUS_TIMEOUT_MS = 90_000
const ACTIVE_AGENT_STATUS_TIMEOUT_MS = 180_000
const CHAT_HISTORY_PAGE_SIZE = 20
const INTERNAL_QQ_BACKGROUND_JOB_PREFIX =
  "[Background terminal job result for this QQ conversation]"
const CHAT_LIVE_MESSAGE_LIMIT = 300
const CHAT_REQUEST_HISTORY_LIMIT = 40
const CHAT_MESSAGE_CONTENT_MAX_CHARS = 16_000
const AGENT_EVENT_RECONNECT_INITIAL_MS = 2_000
const AGENT_EVENT_RECONNECT_MAX_MS = 30_000
const AGENT_EVENT_ERROR_LOG_INTERVAL_MS = 30_000
const CHAT_DEBUG = import.meta.env.DEV
const TERMINAL_STATUS_DONE_STATES = new Set([
  "idle",
  "done",
  "complete",
  "completed",
  "aborted",
  "error",
  "interrupted",
])
const TERMINAL_COMMAND_TOOL_NAMES = new Set([
  "mcp_local_execute_command",
  "mcp_local_interrupt_command",
])
const TERMINAL_JOB_TOOL_NAMES = new Set(["mcp_local_run_job"])

const ROBOT_HEADER_RE = /^\[Robot message;\s*([^\]]+)\]/
const ROBOT_CURRENT_MESSAGE_RE = /\[Current QQ message\]\r?\n([\s\S]*)$/
const ROBOT_PENDING_LINE_RE =
  /^(\d+)\.\s+sender=([^;]+?)(?:;\s+sender_key=[^;]+)?;\s+trigger=([^:]+):\s*(.*)$/
const CQ_REPLY_RE = /\[CQ:reply,id=([^\]]+)\]/g
const CQ_AT_RE = /\[CQ:at,qq=([^,\]]+)(?:,[^\]]*)?\]/g

type ChatSessionPage = {
  messages: ChatMessage[]
  total?: number
  offset?: number
  limit?: number | null
  has_more?: boolean
}

async function getChatSession(
  itemId: string,
  options: { limit?: number; offset?: number } = {},
): Promise<ChatSessionPage> {
  const token = localStorage.getItem("access_token") || ""
  const params = new URLSearchParams()
  if (typeof options.limit === "number") {
    params.set("limit", String(options.limit))
  }
  if (typeof options.offset === "number") {
    params.set("offset", String(options.offset))
  }
  const query = params.toString()
  const response = await fetch(
    `${OpenAPI.BASE}/api/v1/memory/${itemId}/session${query ? `?${query}` : ""}`,
    {
      headers: { Authorization: `Bearer ${token}` },
    },
  )

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`HTTP ${response.status} ${errorText}`)
  }

  return response.json()
}

async function clearChatSession(itemId: string): Promise<void> {
  const token = localStorage.getItem("access_token") || ""
  const response = await fetch(
    `${OpenAPI.BASE}/api/v1/memory/${itemId}/session`,
    {
      method: "DELETE",
      headers: { Authorization: `Bearer ${token}` },
    },
  )

  if (!response.ok) {
    const errorText = await response.text()
    throw new Error(`HTTP ${response.status} ${errorText}`)
  }
}

function normalizeMessage(message: {
  role?: unknown
  content?: unknown
  type?: unknown
  timestamp?: unknown
  tool_name?: unknown
}): ChatMessage | null {
  if (typeof message.content !== "string") {
    return null
  }

  const type = typeof message.type === "string" ? message.type : undefined
  const timestamp =
    typeof message.timestamp === "string" ? message.timestamp : undefined
  const toolName =
    typeof message.tool_name === "string" ? message.tool_name : undefined

  if (
    message.role === "user" ||
    message.role === "assistant" ||
    message.role === "terminal"
  ) {
    return {
      role: message.role,
      content: message.content,
      type,
      timestamp,
      tool_name: toolName,
    }
  }

  if (type === "terminal_output") {
    return {
      role: "terminal",
      content: message.content,
      type,
      timestamp,
      tool_name: toolName,
    }
  }

  if (type === "chat_user" || type === "qq_user" || type === "scheduled_task") {
    return {
      role: "user",
      content: message.content,
      type,
      timestamp,
      tool_name: toolName,
    }
  }

  return {
    role: "assistant",
    content: message.content,
    type,
    timestamp,
    tool_name: toolName,
  }
}

function isInternalAgentCallbackMessage(message: ChatMessage): boolean {
  return message.content
    .trimStart()
    .startsWith(INTERNAL_QQ_BACKGROUND_JOB_PREFIX)
}

function shouldRenderMessage(message: ChatMessage): boolean {
  return (
    !STATUS_ONLY_TYPES.has(message.type ?? "") &&
    !isInternalAgentCallbackMessage(message)
  )
}

function compactLongChatContent(content: string): string {
  if (content.length <= CHAT_MESSAGE_CONTENT_MAX_CHARS) {
    return content
  }

  const edgeLength = Math.floor((CHAT_MESSAGE_CONTENT_MAX_CHARS - 96) / 2)
  return [
    content.slice(0, edgeLength),
    `\n...[trimmed ${content.length - edgeLength * 2} chars for browser performance]...\n`,
    content.slice(-edgeLength),
  ].join("")
}

function compactChatMessageContent(content: string): string {
  if (content.length <= CHAT_MESSAGE_CONTENT_MAX_CHARS) {
    return content
  }

  if (content.includes("[Robot message;")) {
    const firstLineEnd = content.indexOf("\n")
    const header = firstLineEnd >= 0 ? content.slice(0, firstLineEnd) : ""
    const currentMarker = "[Current QQ message]"
    const currentIndex = content.lastIndexOf(currentMarker)
    if (currentIndex >= 0) {
      return compactLongChatContent(
        [header, content.slice(currentIndex)].filter(Boolean).join("\n"),
      )
    }
  }

  return compactLongChatContent(content)
}

function prepareChatMessageForState(message: ChatMessage): ChatMessage {
  const content = compactChatMessageContent(message.content)
  if (content === message.content) {
    return message
  }
  return { ...message, content }
}

function limitLiveChatMessages(messages: ChatMessage[]): ChatMessage[] {
  if (messages.length <= CHAT_LIVE_MESSAGE_LIMIT) {
    return messages
  }
  return messages.slice(-CHAT_LIVE_MESSAGE_LIMIT)
}

function normalizeRenderableMessages(
  messages: ChatSessionPage["messages"] | undefined,
): ChatMessage[] {
  return (messages ?? [])
    .map((message) => normalizeMessage(message))
    .filter(
      (message): message is ChatMessage =>
        message !== null && shouldRenderMessage(message),
    )
    .map((message) => prepareChatMessageForState(message))
}

function getVisibleChatContent(message: ChatMessage): string {
  if (message.role !== "user" || !message.content.includes("[Robot message;")) {
    return message.content
  }

  const currentMatch = message.content.match(ROBOT_CURRENT_MESSAGE_RE)
  if (!currentMatch) {
    return message.content
  }

  const current = currentMatch[1].trim()
  if (!current.startsWith("[Pending QQ messages;")) {
    return current || message.content
  }

  const pendingLines = current
    .split(/\r?\n/)
    .map((line) => {
      const match = line.match(ROBOT_PENDING_LINE_RE)
      if (!match) {
        return null
      }
      return `${match[1]}. ${match[2]}: ${match[4]}`
    })
    .filter((line): line is string => Boolean(line))

  return pendingLines.length > 0 ? pendingLines.join("\n") : current
}

type RobotPendingMessage = {
  index: string
  sender: string
  trigger: string
  content: string
}

type RobotMessageDisplay = {
  body: string
  conversationType: string | null
  conversationId: string | null
  trigger: string | null
  senderName: string | null
  senderId: string | null
  mentions: string[]
  replyIds: string[]
  pendingMessages: RobotPendingMessage[]
}

function parseRobotMetadata(content: string): Record<string, string> | null {
  const match = content.match(ROBOT_HEADER_RE)
  if (!match) {
    return null
  }

  return match[1].split(";").reduce<Record<string, string>>((metadata, part) => {
    const [rawKey, ...valueParts] = part.split("=")
    const key = rawKey.trim()
    const value = valueParts.join("=").trim()
    if (key && value) {
      metadata[key] = value
    }
    return metadata
  }, {})
}

function parseConversation(value: string | undefined): {
  type: string | null
  id: string | null
} {
  if (!value) {
    return { type: null, id: null }
  }

  const [type, ...idParts] = value.split(":")
  return {
    type: type || null,
    id: idParts.join(":") || value,
  }
}

function parseSender(value: string | undefined): {
  name: string | null
  id: string | null
} {
  if (!value) {
    return { name: null, id: null }
  }

  const match = value.match(/^(.*?)\s*\(([^()]+)\)$/)
  if (!match) {
    return { name: value.trim() || null, id: null }
  }

  return {
    name: match[1].trim() || null,
    id: match[2].trim() || null,
  }
}

function uniqueValues(values: string[]): string[] {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean)))
}

function decodeRobotText(value: string): string {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&#91;/g, "[")
    .replace(/&#93;/g, "]")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
}

function cleanRobotQqContent(value: string): string {
  return decodeRobotText(value)
    .replace(CQ_REPLY_RE, "")
    .replace(CQ_AT_RE, "@$1 ")
    .replace(/\[CQ:image[^\]]*\]/g, "[图片]")
    .replace(/\[CQ:record[^\]]*\]/g, "[语音]")
    .replace(/\[CQ:video[^\]]*\]/g, "[视频]")
    .replace(/\[CQ:face[^\]]*\]/g, "[表情]")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

function parseRobotPendingMessages(current: string): RobotPendingMessage[] {
  if (!current.startsWith("[Pending QQ messages;")) {
    return []
  }

  return current
    .split(/\r?\n/)
    .map((line) => {
      const match = line.match(ROBOT_PENDING_LINE_RE)
      if (!match) {
        return null
      }
      return {
        index: match[1],
        sender: match[2].trim(),
        trigger: match[3].trim(),
        content: cleanRobotQqContent(match[4]),
      }
    })
    .filter((line): line is RobotPendingMessage => Boolean(line))
}

function getRobotMessageDisplay(message: ChatMessage): RobotMessageDisplay | null {
  if (message.role !== "user") {
    return null
  }
  if (message.type === "qq_user") {
    const extra = (message as unknown as Record<string, unknown>).extra as Record<string, unknown> | undefined
    return {
      body: message.content,
      conversationType: "private",
      conversationId: String(extra?.sender_key ?? ""),
      trigger: "mention_bot",
      senderName: String(extra?.sender_label ?? "QQ"),
      senderId: String(extra?.sender_key ?? ""),
      mentions: [],
      replyIds: [],
      pendingMessages: [],
    }
  }
  if (!message.content.includes("[Robot message;")) {
    return null
  }

  const metadata = parseRobotMetadata(message.content)
  const currentMatch = message.content.match(ROBOT_CURRENT_MESSAGE_RE)
  if (!metadata || !currentMatch) {
    return null
  }

  const current = currentMatch[1].trim()
  const pendingMessages = parseRobotPendingMessages(current)
  const { type, id } = parseConversation(metadata.conversation)
  const sender = parseSender(metadata.sender)
  const body = pendingMessages.length > 0 ? "" : cleanRobotQqContent(current)
  const mentionsFromHeader = (metadata.mentions ?? "")
    .split(",")
    .map((mention) => mention.trim())
    .filter(Boolean)
  const mentionsFromBody = Array.from(current.matchAll(CQ_AT_RE)).map(
    (match) => match[1],
  )

  return {
    body: body || current,
    conversationType: type,
    conversationId: id,
    trigger: metadata.trigger ?? null,
    senderName: sender.name,
    senderId: sender.id,
    mentions: uniqueValues([...mentionsFromHeader, ...mentionsFromBody]),
    replyIds: uniqueValues(
      Array.from(current.matchAll(CQ_REPLY_RE)).map((match) => match[1]),
    ),
    pendingMessages,
  }
}

function getConversationTypeLabel(type: string | null): string {
  if (type === "group") {
    return "群号"
  }

  if (type === "private") {
    return "私聊"
  }

  if (type === "channel") {
    return "频道"
  }

  return "对话"
}

function getTriggerLabel(trigger: string | null): string | null {
  if (!trigger) {
    return null
  }

  const labels: Record<string, string> = {
    active_chat_window: "已唤醒",
    mention_bot: "@机器人",
    pending_queue: "排队消息",
    plain: "普通消息",
    private: "私聊",
    reply_to_bot: "回复机器人",
  }

  return labels[trigger] ?? trigger
}

function formatChatTimestamp(timestamp: string | undefined): string | null {
  if (!timestamp) {
    return null
  }

  const date = new Date(timestamp)
  if (Number.isNaN(date.getTime())) {
    return timestamp
  }

  return date.toLocaleString()
}

function RobotMetaPill({
  label,
  value,
}: {
  label: string
  value: string | null | undefined
}) {
  if (!value) {
    return null
  }

  return (
    <span className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-muted/60 px-1.5 py-0.5 text-[10px] leading-4 text-foreground">
      <span className="shrink-0 text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate font-medium">{value}</span>
    </span>
  )
}

function RobotMessageCard({
  display,
  timestamp,
}: {
  display: RobotMessageDisplay
  timestamp?: string
}) {
  const triggerLabel = getTriggerLabel(display.trigger) ?? "消息"
  const conversationLabel = display.conversationId
    ? `${getConversationTypeLabel(display.conversationType)} ${display.conversationId}`
    : getConversationTypeLabel(display.conversationType)
  const isPending =
    display.trigger === "pending_queue" || display.pendingMessages.length > 0
  const accentBorder = isPending ? "border-l-amber-500" : "border-l-sky-500"
  const accentText = isPending
    ? "text-amber-600 dark:text-amber-400"
    : "text-sky-600 dark:text-sky-400"
  const accentPill = isPending
    ? "bg-amber-500/15 text-amber-700 dark:text-amber-300"
    : "bg-sky-500/10 text-sky-700 dark:text-sky-300"

  return (
    <div
      className={`w-full overflow-hidden rounded-lg border border-border ${accentBorder} bg-card text-card-foreground shadow-sm`}
    >
      <div className="border-b border-border bg-muted/50 px-2.5 py-2">
        <details className="group">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-2 [&::-webkit-details-marker]:hidden">
            <div className="flex min-w-0 items-center gap-2 text-[11px] font-semibold">
              <span className={`shrink-0 ${accentText}`}>QQ → Agent</span>
              <span
                className={`shrink-0 rounded-md px-1.5 py-0.5 ${accentPill}`}
              >
                {triggerLabel}
              </span>
              <span className="min-w-0 truncate text-muted-foreground">
                {conversationLabel}
                {display.senderName ? ` · ${display.senderName}` : ""}
              </span>
            </div>
            <ChevronDown className="size-3.5 shrink-0 text-muted-foreground transition-transform group-open:rotate-180" />
          </summary>
          <div className="mt-2 flex flex-wrap gap-1.5">
            <RobotMetaPill label="时间" value={formatChatTimestamp(timestamp)} />
            <RobotMetaPill
              label={getConversationTypeLabel(display.conversationType)}
              value={display.conversationId}
            />
            <RobotMetaPill label="触发" value={triggerLabel} />
            <RobotMetaPill label="发送者" value={display.senderName} />
            <RobotMetaPill label="QQ" value={display.senderId} />
            <RobotMetaPill label="回复ID" value={display.replyIds.join(", ")} />
            <RobotMetaPill label="@" value={display.mentions.join(", ")} />
          </div>
        </details>
      </div>
      <div className="px-3 py-2 text-sm leading-6">
        {display.pendingMessages.length > 0 ? (
          <div className="divide-y divide-border">
            {display.pendingMessages.map((pending) => (
              <div
                key={`${pending.index}-${pending.sender}`}
                className="py-1.5 first:pt-0 last:pb-0"
              >
                <div className="mb-0.5 flex min-w-0 items-center gap-2 text-[11px] text-muted-foreground">
                  <span className="shrink-0 font-mono">#{pending.index}</span>
                  <span className="min-w-0 truncate font-medium text-foreground">
                    {pending.sender}
                  </span>
                  <span className="shrink-0">
                    {getTriggerLabel(pending.trigger) ?? pending.trigger}
                  </span>
                </div>
                <div className="whitespace-pre-wrap break-words">
                  {pending.content ? (
                    <ChatRichContent content={pending.content} />
                  ) : (
                    "[空消息]"
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="whitespace-pre-wrap break-words">
            {display.body ? (
              <ChatRichContent content={display.body} />
            ) : (
              "[空消息]"
            )}
          </div>
        )}
      </div>
    </div>
  )
}
function getAgentStatusKind(options: {
  status?: string | null
  source?: TerminalSource | null
  toolName?: string | null
  text?: string
}): AgentStatusKind {
  const toolName = options.toolName ?? ""
  const status = options.status ?? ""
  const text = options.text ?? ""

  if (TERMINAL_JOB_TOOL_NAMES.has(toolName)) {
    return "terminal_job"
  }

  if (
    status === "waiting_terminal" ||
    TERMINAL_COMMAND_TOOL_NAMES.has(toolName) ||
    (options.source === "raw_feedback" && text.includes("\u547d\u4ee4\u5df2\u53d1\u9001"))
  ) {
    return "terminal_wait"
  }

  if (options.source === "raw_feedback") {
    return "terminal_analysis"
  }

  if (status === "thinking") {
    return "thinking"
  }

  return "replying"
}

function isTerminalBusyStatus(status: AgentStatusState | null): boolean {
  return (
    status?.kind === "terminal_wait" ||
    status?.kind === "terminal_job" ||
    status?.kind === "terminal_analysis"
  )
}

function getAgentStatusPresentation(status: AgentStatusState): {
  title: string
  description: string
  icon: "activity" | "download" | "lock" | "terminal"
  className: string
  iconClassName: string
  progressClassName: string
} {
  if (status.kind === "terminal_job") {
    return {
      title: "\u540e\u53f0\u4efb\u52a1\u8fd0\u884c\u4e2d",
      description: "\u4e0b\u8f7d\u3001\u5b89\u88c5\u6216\u6784\u5efa\u5df2\u5728\u72ec\u7acb\u4efb\u52a1\u91cc\u6267\u884c\uff0c\u5b8c\u6210\u540e\u4e00\u6b21\u6027\u628a\u7ed3\u679c\u4ea4\u7ed9 Agent\u3002",
      icon: "download",
      className:
        "border-sky-300 bg-sky-50 text-sky-950 dark:border-sky-500/40 dark:bg-sky-950/30 dark:text-sky-100",
      iconClassName: "bg-sky-500/15 text-sky-700 dark:text-sky-300",
      progressClassName: "bg-sky-500",
    }
  }

  if (status.kind === "terminal_wait") {
    return {
      title: "\u7ec8\u7aef\u6267\u884c\u4e2d",
      description: "\u5f53\u524d\u662f\u4e0d\u53ef\u4ea4\u4e92\u547d\u4ee4\u53cd\u9988\u7a97\u53e3\uff0c\u7ec8\u7aef\u8f93\u5165\u6682\u65f6\u9501\u5b9a\uff0c\u5148\u7b49\u7ed3\u679c\u56de\u6765\u518d\u7ee7\u7eed\u53d1 shell \u547d\u4ee4\u3002",
      icon: "lock",
      className:
        "border-amber-300 bg-amber-50 text-amber-950 dark:border-amber-500/40 dark:bg-amber-950/30 dark:text-amber-100",
      iconClassName: "bg-amber-500/15 text-amber-700 dark:text-amber-300",
      progressClassName: "bg-amber-500",
    }
  }

  if (status.kind === "terminal_analysis") {
    return {
      title: "\u5206\u6790\u7ec8\u7aef\u53cd\u9988",
      description: "\u6b63\u5728\u8bfb\u53d6\u547d\u4ee4\u8f93\u51fa\u5e76\u5224\u65ad\u662f\u5426\u5b8c\u6210\uff0c\u671f\u95f4\u666e\u901a\u804a\u5929\u53ef\u7ee7\u7eed\uff0c\u7ec8\u7aef\u547d\u4ee4\u4f1a\u53d7\u4fdd\u62a4\u3002",
      icon: "terminal",
      className:
        "border-cyan-300 bg-cyan-50 text-cyan-950 dark:border-cyan-500/40 dark:bg-cyan-950/30 dark:text-cyan-100",
      iconClassName: "bg-cyan-500/15 text-cyan-700 dark:text-cyan-300",
      progressClassName: "bg-cyan-500",
    }
  }

  if (status.kind === "tool") {
    return {
      title: "\u5de5\u5177\u8c03\u7528\u4e2d",
      description: "Agent \u6b63\u5728\u8c03\u7528\u5de5\u5177\u5e76\u6574\u7406\u8fd4\u56de\u7ed3\u679c\u3002",
      icon: "activity",
      className: "border-border bg-muted/60 text-foreground",
      iconClassName: "bg-muted text-muted-foreground",
      progressClassName: "bg-primary",
    }
  }

  return {
    title: status.kind === "thinking" ? "\u601d\u8003\u4e2d" : "\u56de\u590d\u4e2d",
    description: "Agent \u6b63\u5728\u5904\u7406\u5f53\u524d\u8bf7\u6c42\u3002",
    icon: "activity",
    className: "border-border bg-muted/60 text-foreground",
    iconClassName: "bg-muted text-muted-foreground",
    progressClassName: "bg-primary",
  }
}

function AgentStatusIcon({ icon }: { icon: "activity" | "download" | "lock" | "terminal" }) {
  if (icon === "download") {
    return <Download className="size-3.5" />
  }
  if (icon === "lock") {
    return <Lock className="size-3.5" />
  }
  if (icon === "terminal") {
    return <Terminal className="size-3.5" />
  }
  return <Activity className="size-3.5" />
}

function AgentStatusCard({
  status,
  canAbort,
  onAbort,
}: {
  status: AgentStatusState
  canAbort: boolean
  onAbort: () => void
}) {
  const presentation = getAgentStatusPresentation(status)
  const sourceLabel = getTerminalSourceLabel(status.source)
  const terminalBusy = isTerminalBusyStatus(status)

  return (
    <div className={`mb-2 rounded-md border px-3 py-2 text-xs ${presentation.className}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 flex-1 gap-2">
          <div className={`mt-0.5 flex size-6 shrink-0 items-center justify-center rounded-md ${presentation.iconClassName}`}>
            <AgentStatusIcon icon={presentation.icon} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="font-medium">{presentation.title}</span>
              {sourceLabel && (
                <span
                  className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${getTerminalSourceBadgeClass(status.source)}`}
                >
                  {sourceLabel}
                </span>
              )}
              {status.toolName && (
                <span className="max-w-full truncate rounded-full border border-current/15 px-1.5 py-0.5 font-mono text-[10px] opacity-80">
                  {status.toolName}
                </span>
              )}
            </div>
            <div className="mt-1 break-words text-[11px] opacity-90">
              {status.text}
            </div>
            <div className="mt-0.5 text-[11px] opacity-70">
              {presentation.description}
            </div>
            {terminalBusy && (
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-current/10">
                <div className={`h-full w-2/5 animate-pulse rounded-full ${presentation.progressClassName}`} />
              </div>
            )}
          </div>
        </div>
        {canAbort && (
          <Button
            size="sm"
            variant="destructive"
            onClick={onAbort}
            className="h-7 shrink-0 px-2 text-[11px]"
            title="\u4e2d\u65ad\u672c\u6b21\u4f1a\u8bdd"
          >
            <Square className="mr-1 size-3" />
            {"\u4e2d\u65ad"}
          </Button>
        )}
      </div>
    </div>
  )
}

function getTransientStatus(message: ChatMessage): AgentStatusState | null {
  const toolName = message.tool_name ?? null

  if (message.type === "agent_thinking") {
    return { text: "\u601d\u8003\u4e2d", kind: "thinking" }
  }

  if (message.type === "agent_action") {
    if (toolName && TERMINAL_JOB_TOOL_NAMES.has(toolName)) {
      return {
        text: "\u540e\u53f0\u4efb\u52a1\u5df2\u542f\u52a8\uff0c\u7b49\u5f85\u5b8c\u6210\u7ed3\u679c",
        source: "raw_feedback",
        toolName,
        kind: "terminal_job",
      }
    }

    if (toolName && TERMINAL_COMMAND_TOOL_NAMES.has(toolName)) {
      return {
        text: "\u547d\u4ee4\u6b63\u5728\u53d1\u9001\u5230\u4ea4\u4e92\u7ec8\u7aef",
        source: "raw_feedback",
        toolName,
        kind: "terminal_wait",
      }
    }

    return {
      text: toolName ? `\u8c03\u7528\u5de5\u5177\u4e2d\uff1a${toolName}` : "\u8c03\u7528\u5de5\u5177\u4e2d",
      toolName,
      kind: "tool",
    }
  }

  if (message.type === "agent_tool_result") {
    if (toolName && TERMINAL_JOB_TOOL_NAMES.has(toolName)) {
      return {
        text: "\u540e\u53f0\u4efb\u52a1\u5df2\u8fd4\u56de\uff0c\u6574\u7406\u7ed3\u679c\u4e2d",
        source: "raw_feedback",
        toolName,
        kind: "terminal_analysis",
      }
    }

    return {
      text: toolName ? `\u5de5\u5177\u5df2\u8fd4\u56de\uff0c\u6574\u7406\u7ed3\u679c\u4e2d\uff1a${toolName}` : "\u5de5\u5177\u5df2\u8fd4\u56de\uff0c\u6574\u7406\u7ed3\u679c\u4e2d",
      toolName,
      kind: "tool",
    }
  }

  return null
}

function getTerminalSource(source: unknown): TerminalSource | null {
  if (source === "filtered_output" || source === "raw_feedback") {
    return source
  }

  return null
}

function getTerminalSourceLabel(
  source: TerminalSource | null | undefined,
): string | null {
  if (source === "filtered_output") {
    return "过滤输出"
  }

  if (source === "raw_feedback") {
    return "命令反馈"
  }

  return null
}

function getTerminalSourceBadgeClass(
  source: TerminalSource | null | undefined,
): string {
  if (source === "raw_feedback") {
    return "border border-amber-200 bg-amber-50 text-amber-700"
  }

  return "border border-sky-200 bg-sky-50 text-sky-700"
}

function extractSsePayloads(buffer: string): {
  payloads: string[]
  rest: string
} {
  const chunks = buffer.split("\n\n")
  return {
    payloads: chunks.slice(0, -1),
    rest: chunks[chunks.length - 1] ?? "",
  }
}

function readSseData(eventChunk: string): string | null {
  const dataLines = eventChunk
    .split("\n")
    .filter((line) => line.startsWith("data: "))
    .map((line) => line.slice(6))

  if (dataLines.length === 0) {
    return null
  }

  return dataLines.join("\n")
}

function isTerminalStreamEvent(event: unknown): boolean {
  if (!event || typeof event !== "object") {
    return false
  }

  const data = event as { done?: unknown; type?: unknown }
  return data.done === true || data.type === "aborted" || data.type === "error"
}

function buildRequestHistory(
  messages: ChatMessage[],
): Array<{ role: "user" | "assistant"; content: string }> {
  return messages.slice(-CHAT_REQUEST_HISTORY_LIMIT).reduce<
    Array<{ role: "user" | "assistant"; content: string }>
  >((history, message) => {
    if (
      message.localEcho ||
      message.type === "agent_thinking" ||
      STATUS_ONLY_TYPES.has(message.type ?? "")
    ) {
      return history
    }

    if (message.role === "user") {
      history.push({ role: "user", content: message.content })
      return history
    }

    if (message.role === "terminal") {
      history.push({
        role: "user",
        content: `终端过滤输出:\n${message.content}`,
      })
      return history
    }

    if (
      message.type &&
      !COMPLETION_TYPES.has(message.type) &&
      message.type !== "chat_assistant"
    ) {
      return history
    }

    history.push({ role: "assistant", content: message.content })
    return history
  }, [])
}

function getMessageLabel(message: ChatMessage): string | null {
  if (message.type === "scheduled_task") {
    return "定时任务"
  }

  if (message.role === "terminal") {
    return "term"
  }

  if (message.type === "agent_warning") {
    return "警告"
  }

  if (message.type === "agent_error") {
    return "错误"
  }

  if (message.type === "agent_qq_reply") {
    return "QQ"
  }

  return null
}

function getMessageClasses(message: ChatMessage): string {
  if (message.type === "scheduled_task") {
    return "border border-amber-500/40 bg-amber-500/10 text-foreground"
  }

  if (message.role === "user") {
    return "bg-primary text-primary-foreground"
  }

  if (message.role === "terminal") {
    return "border border-slate-700 bg-slate-950 font-mono text-xs text-emerald-300"
  }

  if (message.type === "agent_error") {
    return "border border-red-300 bg-red-50 text-red-900"
  }

  if (message.type === "agent_warning") {
    return "border border-amber-300 bg-amber-50 text-amber-950"
  }

  if (message.type === "agent_qq_reply") {
    return "border border-sky-500/30 bg-sky-500/10 text-sky-950 dark:text-sky-100"
  }

  return "border bg-muted"
}

const ThinkingBlock = memo(function ThinkingBlock({
  message,
}: {
  message: ChatMessage
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div className="flex justify-start">
      <div className="w-full max-w-[90%]">
        <button
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] text-muted-foreground transition-colors hover:bg-muted/70 hover:text-foreground"
        >
          <Brain className="h-3 w-3" />
          <span>思考过程</span>
          {expanded ? (
            <ChevronUp className="h-3 w-3" />
          ) : (
            <ChevronDown className="h-3 w-3" />
          )}
        </button>
        <div
          className={`grid transition-all duration-200 ease-out ${
            expanded ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
          }`}
        >
          <div className="overflow-hidden">
            <div className="mt-1 whitespace-pre-wrap break-words rounded-lg border border-dashed border-muted-foreground/25 bg-muted/40 px-3 py-2 text-xs leading-relaxed text-muted-foreground">
              {message.content}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
})

const CHAT_IMAGE_PATTERN = /!\[([^\]]*)\]\(((?:https?:\/\/|\/)[^)\s]+)\)/g

function ChatRichContent({ content }: { content: string }) {
  const parts: Array<{ kind: "text" | "image"; value: string; alt?: string }> =
    []
  let lastIndex = 0
  for (const match of content.matchAll(CHAT_IMAGE_PATTERN)) {
    const index = match.index ?? 0
    if (index > lastIndex) {
      parts.push({ kind: "text", value: content.slice(lastIndex, index) })
    }
    parts.push({ kind: "image", value: match[2], alt: match[1] })
    lastIndex = index + match[0].length
  }
  if (lastIndex < content.length) {
    parts.push({ kind: "text", value: content.slice(lastIndex) })
  }
  if (parts.every((part) => part.kind === "text")) {
    return <>{content}</>
  }
  return (
    <>
      {parts.map((part, index) =>
        part.kind === "image" ? (
          <img
            key={index}
            src={part.value}
            alt={part.alt || "图片"}
            loading="lazy"
            className="my-1 block max-w-56 rounded-md border bg-background"
          />
        ) : (
          <span key={index}>{part.value}</span>
        ),
      )}
    </>
  )
}

const TASK_STEP_STATUS_ICON: Record<string, string> = {
  in_progress: "▶",
  pending: "○",
  completed: "✓",
  cancelled: "✕",
}

function ChatTaskStrip({ itemId }: { itemId: string }) {
  const queryClient = useQueryClient()
  const cancelPlan = async (ticketId: string) => {
    const token = localStorage.getItem("access_token") || ""
    try {
      await fetch(
        `${OpenAPI.BASE}/api/v1/items/${itemId}/plan/${ticketId}/cancel`,
        { method: "POST", headers: { Authorization: `Bearer ${token}` } },
      )
    } finally {
      await queryClient.invalidateQueries({ queryKey: ["item-plan", itemId] })
    }
  }

  const planQuery = useQuery({
    queryKey: ["item-plan", itemId],
    queryFn: async () => {
      const token = localStorage.getItem("access_token") || ""
      const response = await fetch(
        `${OpenAPI.BASE}/api/v1/items/${itemId}/plan`,
        { headers: { Authorization: `Bearer ${token}` } },
      )
      if (!response.ok) {
        return { plans: [] }
      }
      return response.json()
    },
    refetchInterval: 5000,
  })
  const jobsQuery = useQuery({
    queryKey: ["item-jobs", itemId],
    queryFn: async () => {
      const token = localStorage.getItem("access_token") || ""
      const response = await fetch(
        `${OpenAPI.BASE}/api/v1/items/${itemId}/jobs`,
        { headers: { Authorization: `Bearer ${token}` } },
      )
      if (!response.ok) {
        return { jobs: [] }
      }
      return response.json()
    },
    refetchInterval: 5000,
  })

  const tasks: Array<Record<string, unknown>> = Array.isArray(
    planQuery.data?.plans,
  )
    ? planQuery.data.plans
    : []
  const jobs: Array<Record<string, unknown>> = Array.isArray(
    jobsQuery.data?.jobs,
  )
    ? jobsQuery.data.jobs
    : []
  const showJobsBlock = jobs.length > 0 || tasks.length > 0
  if (tasks.length === 0 && jobs.length === 0) {
    return null
  }

  return (
    <div className="mt-2 space-y-1 text-xs">
      {tasks.map((task) => {
        const steps: Array<Record<string, unknown>> = Array.isArray(task.plan)
          ? (task.plan as Array<Record<string, unknown>>)
          : []
        const done = Number(task.done ?? 0)
        const total = Number(task.total ?? steps.length)
        const title =
          String(task.request ?? "").trim() ||
          `任务 ${String(task.ticket_id ?? "").slice(0, 8)}`
        return (
          <details
            key={String(task.ticket_id)}
            className="group rounded-lg border bg-muted/20"
          >
            <summary className="flex cursor-pointer list-none items-center gap-2 px-2.5 py-1.5 [&::-webkit-details-marker]:hidden">
              <ChevronDown className="size-3.5 shrink-0 text-muted-foreground transition-transform group-open:rotate-180" />
              <span className="min-w-0 flex-1 truncate font-medium">
                {title}
              </span>
              <span className="shrink-0 text-muted-foreground">
                {done}/{total}
              </span>
              <button
                type="button"
                title="取消该任务"
                className="flex size-4 shrink-0 items-center justify-center rounded-full border bg-background text-muted-foreground hover:text-red-600"
                onClick={(event) => {
                  event.preventDefault()
                  event.stopPropagation()
                  void cancelPlan(String(task.ticket_id))
                }}
              >
                <X className="size-3" />
              </button>
            </summary>
            <div className="space-y-0.5 border-t px-2.5 py-1.5">
              {steps.map((step, index) => {
                const status = String(step.status ?? "pending")
                const stepDone =
                  status === "completed" || status === "cancelled"
                return (
                  <div
                    key={index}
                    className={`flex items-center gap-1.5 ${stepDone ? "text-muted-foreground line-through" : ""}`}
                  >
                    <span className="shrink-0">
                      {TASK_STEP_STATUS_ICON[status] ?? "○"}
                    </span>
                    <span className="break-all">
                      {String(step.step ?? "")}
                    </span>
                  </div>
                )
              })}
            </div>
          </details>
        )
      })}
      {showJobsBlock ? (
        <div className="rounded-lg border bg-muted/20 px-2.5 py-1.5">
          <div className="mb-1 font-semibold text-muted-foreground">
            后台任务
          </div>
          {jobs.length === 0 ? (
            <div className="text-muted-foreground">暂无后台任务</div>
          ) : (
            <div className="space-y-1">
              {jobs.map((job, index) => {
            const command = String(job.command ?? "")
            const elapsed = Math.round(Number(job.elapsed_seconds ?? 0))
            const tailLines = String(job.output_tail ?? "")
              .split("\n")
              .filter((line) => line.trim())
            const lastOutput = tailLines[tailLines.length - 1] ?? ""
            return (
              <details
                key={String(job.job_id ?? index)}
                className="group rounded-lg border bg-muted/20"
              >
                <summary className="flex cursor-pointer list-none items-center gap-1.5 px-2.5 py-1.5 [&::-webkit-details-marker]:hidden">
                  <ChevronDown className="size-3.5 shrink-0 text-muted-foreground transition-transform group-open:rotate-180" />
                  <Loader2 className="size-3 shrink-0 animate-spin" />
                  <span
                    className="max-w-64 truncate font-medium"
                    title={command}
                  >
                    {command}
                  </span>
                  <span className="shrink-0 text-muted-foreground">
                    {elapsed}s
                  </span>
                  {lastOutput ? (
                    <span className="min-w-0 flex-1 truncate text-muted-foreground">
                      · {lastOutput}
                    </span>
                  ) : null}
                </summary>
                {tailLines.length > 0 ? (
                  <div className="max-h-40 space-y-0.5 overflow-y-auto whitespace-pre-wrap break-all border-t px-2.5 py-1.5 font-mono text-[11px] text-muted-foreground">
                    {tailLines.map((line, lineIndex) => (
                      <div key={lineIndex}>{line}</div>
                    ))}
                  </div>
                ) : (
                  <div className="border-t px-2.5 py-1.5 text-muted-foreground">
                    暂无输出
                  </div>
                )}
              </details>
            )
          })}
            </div>
          )}
        </div>
      ) : null}
    </div>
  )
}

const ChatMessageRow = memo(function ChatMessageRow({
  message,
}: {
  message: ChatMessage
}) {
  if (message.type === "agent_thinking") {
    return <ThinkingBlock message={message} />
  }

  const label = getMessageLabel(message)
  const robotDisplay = getRobotMessageDisplay(message)
  const visibleContent = robotDisplay ? "" : getVisibleChatContent(message)
  const alignment = message.role === "user" ? "justify-end" : "justify-start"

  return (
    <div className={`flex ${alignment}`}>
      {robotDisplay ? (
        <div className="w-full max-w-[94%]">
          <RobotMessageCard
            display={robotDisplay}
            timestamp={message.timestamp}
          />
        </div>
      ) : (
        <div
          className={`max-w-[90%] rounded-lg px-3 py-2 text-sm ${getMessageClasses(message)}`}
        >
          {label && (
            <div className="mb-1 border-b border-current/15 pb-1 text-[11px] uppercase tracking-wide opacity-70">
              {label}
            </div>
          )}
          <div
            className={`whitespace-pre-wrap break-words ${
              message.role === "terminal" ? "font-mono" : "font-sans"
            }`}
          >
            {typeof visibleContent === "string" ? (
              <ChatRichContent content={visibleContent} />
            ) : (
              visibleContent
            )}
          </div>
        </div>
      )}
    </div>
  )
})

export function ChatPanel({ itemId }: ChatPanelProps) {
  const queryClient = useQueryClient()
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const inputHistoryRef = useRef<string[]>([])
  const historyCursorRef = useRef(-1)
  const historyDraftRef = useRef("")

  const navigateInputHistory = (direction: 1 | -1) => {
    const history = inputHistoryRef.current
    if (history.length === 0) {
      return
    }
    const cursor = historyCursorRef.current
    if (direction === -1) {
      // 向上（更早）
      const next = Math.min(cursor + 1, history.length - 1)
      if (cursor === -1) {
        historyDraftRef.current = input
      }
      historyCursorRef.current = next
      setInput(history[history.length - 1 - next])
    } else {
      // 向下（更近/草稿）
      if (cursor === -1) {
        return
      }
      const next = cursor - 1
      historyCursorRef.current = next
      setInput(next === -1 ? historyDraftRef.current : history[history.length - 1 - next])
    }
  }
  const [pendingImages, setPendingImages] = useState<string[]>([])
  const imageInputRef = useRef<HTMLInputElement | null>(null)

  const MAX_CHAT_IMAGES = 4

  const handleImageFiles = (files: FileList | null) => {
    if (!files || files.length === 0) {
      return
    }
    const selected = Array.from(files)
      .filter((file) => file.type.startsWith("image/"))
      .slice(0, MAX_CHAT_IMAGES - pendingImages.length)
    for (const file of selected) {
      const reader = new FileReader()
      reader.onload = () => {
        const result = typeof reader.result === "string" ? reader.result : ""
        if (result.startsWith("data:image/")) {
          setPendingImages((prev) =>
            prev.length >= MAX_CHAT_IMAGES ? prev : [...prev, result],
          )
        }
      }
      reader.readAsDataURL(file)
    }
  }
  const [isLoading, setIsLoading] = useState(false)
  const [handler, setHandler] = useState<ItemHandlerPublic | null>(null)
  const [historyError, setHistoryError] = useState<string | null>(null)
  const [liveError, setLiveError] = useState<string | null>(null)
  const [agentStatus, setAgentStatus] = useState<AgentStatusState | null>(null)
  const [turnCount, setTurnCount] = useState(0)
  const turnActiveRef = useRef(false)
  const [historyHasMore, setHistoryHasMore] = useState(false)
  const [historyTotal, setHistoryTotal] = useState(0)
  const [isLoadingHistory, setIsLoadingHistory] = useState(false)

  const scrollRef = useRef<HTMLDivElement>(null)
  const scrollContentRef = useRef<HTMLDivElement>(null)
  const messageCountRef = useRef(0)
  const initialScrollPendingRef = useRef(true)
  const shouldStickToBottomRef = useRef(true)
  const pendingHistoryAnchorRef = useRef<{
    scrollHeight: number
    scrollTop: number
  } | null>(null)
  const loadedPersistedMessageCountRef = useRef(0)
  const streamReaderRef =
    useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null)
  const streamAbortRef = useRef<AbortController | null>(null)
  const canAbortSession =
    Boolean(handler) && (Boolean(agentStatus) || isLoading)
  messageCountRef.current = messages.length

  useEffect(() => {
    if (!agentStatus?.autoClearMs) {
      return
    }

    const timer = window.setTimeout(() => {
      setAgentStatus((current) => (current === agentStatus ? null : current))
    }, agentStatus.autoClearMs)

    return () => window.clearTimeout(timer)
  }, [agentStatus])

  const appendLiveMessage = (message: ChatMessage) => {
    const nextMessage = prepareChatMessageForState(message)
    setMessages((prev) => {
      for (let index = prev.length - 1; index >= 0; index -= 1) {
        const current = prev[index]
        if (
          current.localEcho &&
          current.role === nextMessage.role &&
          (nextMessage.type === "chat_user" || nextMessage.type === "qq_user") &&
          current.content === nextMessage.content
        ) {
          const next = [...prev]
          next[index] = nextMessage
          if (!nextMessage.localEcho) {
            loadedPersistedMessageCountRef.current += 1
          }
          return limitLiveChatMessages(next)
        }
      }

      const lastMessage = prev[prev.length - 1]
      if (
        lastMessage &&
        lastMessage.role === nextMessage.role &&
        lastMessage.type === nextMessage.type &&
        lastMessage.timestamp === nextMessage.timestamp &&
        lastMessage.content === nextMessage.content
      ) {
        return prev
      }

      if (!nextMessage.localEcho) {
        loadedPersistedMessageCountRef.current += 1
      }
      return limitLiveChatMessages([...prev, nextMessage])
    })
  }

  const handleIncomingEvent = useEffectEvent((event: unknown) => {
    if (!event || typeof event !== "object") {
      return
    }

    const data = event as {
      done?: unknown
      type?: unknown
      status?: unknown
      content?: unknown
      terminal_source?: unknown
      tool_name?: unknown
      plan?: unknown
    }

    if (data.done === true) {
      turnActiveRef.current = false
      setAgentStatus(null)
      return
    }

    if (data.type === "aborted" || data.type === "error") {
      turnActiveRef.current = false
      setAgentStatus(null)
      return
    }

    if (data.type === "plan_updated") {
      queryClient.invalidateQueries({ queryKey: ["items", "plan", itemId] })
      queryClient.invalidateQueries({
        queryKey: ["items", "token-by-task", itemId],
      })
      return
    }

    if (data.type === "agent_status") {
      const status = typeof data.status === "string" ? data.status : ""
      const content = typeof data.content === "string" ? data.content.trim() : ""
      if (!content || TERMINAL_STATUS_DONE_STATES.has(status)) {
        turnActiveRef.current = false
        setAgentStatus(null)
        return
      }
      if (status === "running" && !turnActiveRef.current) {
        // Each running cycle after a done/idle boundary is one agent-driven
        // turn (continuation, job callback, stall resume). User inputs reset
        // the counter via the chat_user branch instead of counting here.
        turnActiveRef.current = true
        setTurnCount((current) => current + 1)
      }

      const terminalSource = getTerminalSource(data.terminal_source)
      const toolName =
        typeof data.tool_name === "string" ? data.tool_name : null
      setAgentStatus({
        text: content,
        status,
        source: terminalSource,
        toolName,
        kind: getAgentStatusKind({
          status,
          source: terminalSource,
          toolName,
          text: content,
        }),
        autoClearMs: ACTIVE_AGENT_STATUS_TIMEOUT_MS,
      })
      return
    }

    if (data.type === "qq_message_processed") {
      getChatSession(itemId, { limit: CHAT_HISTORY_PAGE_SIZE, offset: 0 })
        .then((session) => {
          const normalizedMessages = normalizeRenderableMessages(
            session.messages,
          )
          setMessages(normalizedMessages)
          loadedPersistedMessageCountRef.current =
            session.messages?.length ?? 0
          setHistoryTotal(session.total ?? normalizedMessages.length)
        })
        .catch(() => {})
      return
    }

    if (data.type === "plan_updated") {
      // Plan updates feed the PlanPanel query cache; they are not chat
      // messages, so this branch must return before normalizeMessage runs.
      const rawPlan = Array.isArray(data.plan) ? data.plan : []
      const plan = rawPlan
        .filter(
          (step): step is { step: string; status: string } =>
            typeof step === "object" &&
            step !== null &&
            typeof (step as { step?: unknown }).step === "string" &&
            typeof (step as { status?: unknown }).status === "string",
        )
        .map((step) => ({
          step: step.step,
          status: step.status as "pending" | "in_progress" | "completed",
        }))
      queryClient.setQueryData(["items", "plan", itemId], {
        item_id: itemId,
        plan,
        updated: true,
      })
      return
    }

    const normalizedMessage = normalizeMessage(event as Record<string, unknown>)
    if (!normalizedMessage) {
      return
    }
    if (isInternalAgentCallbackMessage(normalizedMessage)) {
      return
    }

    const transientStatus = getTransientStatus(normalizedMessage)
    if (transientStatus) {
      setAgentStatus({
        ...transientStatus,
        autoClearMs: ACTIVE_AGENT_STATUS_TIMEOUT_MS,
      })
      if (normalizedMessage.type === "agent_thinking") {
        if (shouldRenderMessage(normalizedMessage)) {
          appendLiveMessage(normalizedMessage)
        }
        return
      }
      return
    }
    if (normalizedMessage.type === "chat_user" || normalizedMessage.type === "qq_user") {
      setTurnCount(1)
      turnActiveRef.current = true
      setAgentStatus({
        text: "\u56de\u590d\u4e2d",
        kind: "replying",
        autoClearMs: LIVE_REPLY_STATUS_TIMEOUT_MS,
      })
    } else if (
      COMPLETION_TYPES.has(normalizedMessage.type ?? "") ||
      normalizedMessage.role === "assistant"
    ) {
      setAgentStatus(null)
    }

    if (!shouldRenderMessage(normalizedMessage)) {
      return
    }

    appendLiveMessage(normalizedMessage)
  })

  useEffect(() => {
    let cancelled = false

    const fetchData = async () => {
      initialScrollPendingRef.current = true
      shouldStickToBottomRef.current = true
      pendingHistoryAnchorRef.current = null
      setHandler(null)
      setMessages([])
      setHistoryError(null)
      setLiveError(null)
      setIsLoading(false)
      setAgentStatus(null)
      setTurnCount(0)
      turnActiveRef.current = false
      setHistoryHasMore(false)
      setHistoryTotal(0)
      setIsLoadingHistory(false)
      loadedPersistedMessageCountRef.current = 0

      const [handlersResult, sessionResult] = await Promise.allSettled([
        ItemHandlerAssociationsService.getHandlersForItem({ itemId }),
        getChatSession(itemId, { limit: CHAT_HISTORY_PAGE_SIZE, offset: 0 }),
      ])

      if (cancelled) {
        return
      }

      if (handlersResult.status === "fulfilled") {
        const handlers = handlersResult.value
        if (handlers && handlers.length > 0) {
          setHandler(handlers[0] as ItemHandlerPublic)
        }
      } else {
        console.error("[Chat] Failed to fetch handler:", handlersResult.reason)
      }

      if (sessionResult.status === "fulfilled") {
        const normalizedMessages = normalizeRenderableMessages(
          sessionResult.value.messages,
        )
        setMessages(normalizedMessages)
        loadedPersistedMessageCountRef.current =
          sessionResult.value.messages?.length ?? 0
        setHistoryTotal(sessionResult.value.total ?? normalizedMessages.length)
        setHistoryHasMore(Boolean(sessionResult.value.has_more))
      } else {
        const errorMessage =
          sessionResult.reason instanceof Error
            ? sessionResult.reason.message
            : String(sessionResult.reason)
        console.error(
          "[Chat] Failed to load session history:",
          sessionResult.reason,
        )
        setHistoryError(`无法读取 SQLite 会话历史: ${errorMessage}`)
      }
    }

    void fetchData()

    return () => {
      cancelled = true
    }
  }, [itemId])

  const loadOlderHistory = async () => {
    if (isLoadingHistory || !historyHasMore) {
      return
    }

    const scrollElement = scrollRef.current
    const previousScrollHeight = scrollElement?.scrollHeight ?? 0
    const previousScrollTop = scrollElement?.scrollTop ?? 0
    const loadedPersistedMessages = loadedPersistedMessageCountRef.current

    setIsLoadingHistory(true)
    try {
      const page = await getChatSession(itemId, {
        limit: CHAT_HISTORY_PAGE_SIZE,
        offset: loadedPersistedMessages,
      })
      const olderMessages = normalizeRenderableMessages(page.messages)
      loadedPersistedMessageCountRef.current = advanceChatHistoryOffset(
        loadedPersistedMessageCountRef.current,
        page.messages,
      )
      setHistoryTotal(page.total ?? historyTotal)
      setHistoryHasMore(Boolean(page.has_more))
      pendingHistoryAnchorRef.current = {
        scrollHeight: previousScrollHeight,
        scrollTop: previousScrollTop,
      }
      setMessages((current) => {
        const existingKeys = new Set(current.map((message) => getChatMessageKey(message)))
        const uniqueOlderMessages = olderMessages.filter(
          (message) => !existingKeys.has(getChatMessageKey(message)),
        )
        if (uniqueOlderMessages.length === 0) {
          pendingHistoryAnchorRef.current = null
          return current
        }
        return [...uniqueOlderMessages, ...current]
      })
    } catch (error) {
      console.error("[Chat] Failed to load older session history:", error)
      setHistoryError(
        `加载更早历史失败: ${error instanceof Error ? error.message : String(error)}`,
      )
    } finally {
      setIsLoadingHistory(false)
    }
  }
  useEffect(() => {
    if (!handler) {
      return
    }

    const token = localStorage.getItem("access_token") || ""
    const url = `${OpenAPI.BASE}/api/v1/chat/${itemId}/agent-events`
    let disposed = false
    let reconnectTimer: number | null = null
    let controller: AbortController | null = null
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null
    let reconnectAttempt = 0
    let lastErrorLogAt = 0

    const scheduleReconnect = () => {
      if (disposed || reconnectTimer !== null) {
        return
      }
      const delay = Math.min(
        AGENT_EVENT_RECONNECT_MAX_MS,
        AGENT_EVENT_RECONNECT_INITIAL_MS * 2 ** reconnectAttempt,
      )
      reconnectAttempt += 1
      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null
        void connect()
      }, delay)
    }

    const connect = async () => {
      if (disposed) {
        return
      }

      controller = new AbortController()

      try {
        const response = await fetch(url, {
          headers: {
            Authorization: `Bearer ${token}`,
            Accept: "text/event-stream",
          },
          signal: controller.signal,
        })

        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(`HTTP ${response.status} ${errorText}`)
        }

        reader = response.body?.getReader() ?? null
        if (!reader) {
          throw new Error("No SSE reader available")
        }

        reconnectAttempt = 0
        setLiveError(null)

        const decoder = new TextDecoder()
        let buffer = ""

        while (!disposed) {
          const { done, value } = await reader.read()
          if (done) {
            break
          }

          buffer += decoder.decode(value, { stream: true })
          const { payloads, rest } = extractSsePayloads(buffer)
          buffer = rest

          for (const payload of payloads) {
            const rawEvent = readSseData(payload)
            if (!rawEvent) {
              continue
            }

            try {
              handleIncomingEvent(JSON.parse(rawEvent))
            } catch (error) {
              console.error(
                "[Chat] Failed to parse agent event payload:",
                error,
              )
            }
          }
        }
      } catch (error) {
        if (
          !disposed &&
          !(error instanceof DOMException && error.name === "AbortError")
        ) {
          const message = error instanceof Error ? error.message : String(error)
          if (CHAT_DEBUG) {
            const now = Date.now()
            if (now - lastErrorLogAt >= AGENT_EVENT_ERROR_LOG_INTERVAL_MS) {
              lastErrorLogAt = now
              console.warn("[Chat] Agent event stream error:", error)
            }
          }
          const nextError = `实时事件连接异常: ${message}`
          setLiveError((current) => (current === nextError ? current : nextError))
        }
      } finally {
        if (reader) {
          reader.cancel().catch(() => undefined)
          reader = null
        }

        scheduleReconnect()
      }
    }

    void connect()

    return () => {
      disposed = true
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer)
      }
      controller?.abort()
      if (reader) {
        reader.cancel().catch(() => undefined)
      }
    }
  }, [handler, itemId])

  useLayoutEffect(() => {
    const scrollElement = scrollRef.current
    if (!scrollElement) {
      return
    }

    const historyAnchor = pendingHistoryAnchorRef.current
    if (historyAnchor) {
      pendingHistoryAnchorRef.current = null
      scrollElement.scrollTop =
        scrollElement.scrollHeight -
        historyAnchor.scrollHeight +
        historyAnchor.scrollTop
      shouldStickToBottomRef.current =
        scrollElement.scrollHeight -
          scrollElement.scrollTop -
          scrollElement.clientHeight <
        80
      return
    }

    if (messages.length === 0) {
      return
    }
    if (
      initialScrollPendingRef.current ||
      shouldStickToBottomRef.current
    ) {
      scrollElement.scrollTop = scrollElement.scrollHeight
      initialScrollPendingRef.current = false
      shouldStickToBottomRef.current = true
    }
  }, [messages])

  useEffect(() => {
    const contentElement = scrollContentRef.current
    if (!contentElement || typeof ResizeObserver === "undefined") {
      return
    }

    const observer = new ResizeObserver(() => {
      const scrollElement = scrollRef.current
      if (
        !scrollElement ||
        pendingHistoryAnchorRef.current ||
        (!initialScrollPendingRef.current &&
          !shouldStickToBottomRef.current)
      ) {
        return
      }
      scrollElement.scrollTop = scrollElement.scrollHeight
      if (messageCountRef.current > 0) {
        initialScrollPendingRef.current = false
      }
    })
    observer.observe(contentElement)
    return () => observer.disconnect()
  }, [itemId])

  const abortChat = async () => {
    const token = localStorage.getItem("access_token") || ""
    setAgentStatus({ text: "中断中" })

    try {
      streamAbortRef.current?.abort()
      streamAbortRef.current = null

      if (streamReaderRef.current) {
        await streamReaderRef.current.cancel()
        streamReaderRef.current = null
      }

      await fetch(`${OpenAPI.BASE}/api/v1/chat/${itemId}/abort`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      })
    } catch (error) {
      console.error("[Chat] Failed to abort:", error)
      setAgentStatus(null)
    } finally {
      setAgentStatus(null)
      setIsLoading(false)
    }
  }

  const sendMessage = async () => {
    const messageText = input.trim()
    const images = pendingImages
    if ((!messageText && images.length === 0) || !handler) {
      return
    }

    const history = buildRequestHistory(messages)
    setMessages((prev) =>
      limitLiveChatMessages([
        ...prev,
        prepareChatMessageForState({
          role: "user",
          content:
            images.length > 0
              ? `${messageText || ""}[图片×${images.length}]`
              : messageText,
          type: "chat_user",
          localEcho: true,
        }),
      ]),
    )
    setInput("")
    setPendingImages([])
    if (messageText.trim()) {
      inputHistoryRef.current = [
        ...inputHistoryRef.current.slice(-99),
        messageText,
      ]
    }
    historyCursorRef.current = -1
    historyDraftRef.current = ""
    setAgentStatus({
        text: "\u56de\u590d\u4e2d",
        kind: "replying",
        autoClearMs: LIVE_REPLY_STATUS_TIMEOUT_MS,
      })
    setIsLoading(true)

    const token = localStorage.getItem("access_token") || ""
    const controller = new AbortController()
    streamAbortRef.current = controller

    try {
      const response = await fetch(
        `${OpenAPI.BASE}/api/v1/chat/${itemId}/stream`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          signal: controller.signal,
          body: JSON.stringify({
            message: messageText,
            history,
            images,
          }),
        },
      )

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`HTTP ${response.status} ${errorText}`)
      }

      const reader = response.body?.getReader() ?? null
      streamReaderRef.current = reader

      if (!reader) {
        throw new Error("No response reader available")
      }

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const { payloads, rest } = extractSsePayloads(buffer)
        buffer = rest

        let shouldStop = false

        for (const payload of payloads) {
          const rawEvent = readSseData(payload)
          if (!rawEvent) {
            continue
          }

          try {
            const data = JSON.parse(rawEvent)
            handleIncomingEvent(data)
            if (isTerminalStreamEvent(data)) {
              shouldStop = true
            }
          } catch (error) {
            console.error("[Chat] Failed to parse stream payload:", error)
          }
        }

        if (shouldStop) {
          break
        }
      }

      const trailingEvent = readSseData(buffer)
      if (trailingEvent) {
        try {
          handleIncomingEvent(JSON.parse(trailingEvent))
        } catch (error) {
          console.error("[Chat] Failed to parse trailing stream payload:", error)
        }
      }
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        const errorMessage =
          error instanceof Error ? error.message : String(error)
        console.error("[Chat] Failed to send message:", error)
        setAgentStatus(null)
        appendLiveMessage({
          role: "assistant",
          content: `请求失败: ${errorMessage}`,
          type: "agent_error",
        })
      }
    } finally {
      streamReaderRef.current = null
      streamAbortRef.current = null
      setAgentStatus(null)
      setIsLoading(false)
    }
  }

  const visibleAgentStatus =
    agentStatus ?? (isLoading ? { text: "\u56de\u590d\u4e2d", kind: "replying" as const } : null)
  const hasConnectionIssue = historyError || liveError
  const terminalBusy = isTerminalBusyStatus(visibleAgentStatus)
  const inputPlaceholder = terminalBusy
    ? "\u7ec8\u7aef\u4efb\u52a1\u8fd0\u884c\u4e2d\uff0c\u6d88\u606f\u5c06\u6392\u961f\u5408\u5e76\u5904\u7406"
    : handler
      ? "\u8f93\u5165\u6d88\u606f..."
      : "\u8bf7\u5148\u5173\u8054 ItemHandler"
  const stableMessageRows = useMemo(
    () => buildStableChatMessageRows(messages),
    [messages],
  )

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center justify-between border-b bg-background px-4 py-3">
        <div className="flex items-center gap-2">
          {handler ? (
            <Link
              to="/item-handlers/$itemHandlerId"
              params={{ itemHandlerId: handler.id }}
              className="flex items-center gap-2 text-sm font-medium transition-colors hover:text-primary"
            >
              <span>{handler.name}</span>
              <ExternalLink className="size-3" />
            </Link>
          ) : (
            <span className="text-sm text-muted-foreground">
              未关联 ItemHandler
            </span>
          )}
        </div>

        {messages.length > 0 && (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="ghost" size="icon" title="清空会话时间线">
                <Trash2 className="size-4" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>清空会话时间线</AlertDialogTitle>
                <AlertDialogDescription>
                  这会删除当前 item 在 SQLite 里的聊天时间线历史，操作不可撤销。
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>取消</AlertDialogCancel>
                <AlertDialogAction
                  onClick={async () => {
                    try {
                      await clearChatSession(itemId)
                      setMessages([])
                      loadedPersistedMessageCountRef.current = 0
                      setHistoryError(null)
                      setAgentStatus(null)
                      setTurnCount(0)
                      turnActiveRef.current = false
                      setHistoryHasMore(false)
                      setHistoryTotal(0)
                    } catch (error) {
                      console.error(
                        "[Chat] Failed to clear chat session:",
                        error,
                      )
                      setHistoryError(
                        `清空会话历史失败: ${error instanceof Error ? error.message : String(error)}`,
                      )
                    }
                  }}
                >
                  确认清空
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>

      {hasConnectionIssue && (
        <div className="shrink-0 border-b border-amber-200 bg-amber-50 px-4 py-2 text-xs text-amber-950">
          {historyError && <div>{historyError}</div>}
          {liveError && <div>{liveError}</div>}
        </div>
      )}

      <div
        className="min-h-0 flex-1 overflow-y-auto"
        ref={scrollRef}
        data-testid="chat-history-scroll"
        onScroll={(event) => {
          const element = event.currentTarget
          shouldStickToBottomRef.current =
            element.scrollHeight - element.scrollTop - element.clientHeight < 80
        }}
      >
        <div className="space-y-3 p-3" ref={scrollContentRef}>
          {messages.length === 0 && (
            <div className="py-8 text-center text-sm text-muted-foreground">
              {handler
                ? "开始与 Agent 对话，或等待终端过滤事件写入时间线。"
                : "关联 ItemHandler 后可启用对话与实时事件。"}
            </div>
          )}

          {messages.length > 0 && historyHasMore && (
            <div className="flex justify-center">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={() => void loadOlderHistory()}
                disabled={isLoadingHistory}
                data-testid="chat-load-older"
                className="h-7 gap-1.5 px-2 text-xs text-muted-foreground"
                title="向上加载更早的 20 条历史"
              >
                {isLoadingHistory ? (
                  <Loader2 className="size-3.5 animate-spin" />
                ) : (
                  <ChevronUp className="size-3.5" />
                )}
                <span>
                  {isLoadingHistory ? "加载中" : "加载更早 20 条"}
                  {historyTotal > 0 ? ` · ${messages.length}/${historyTotal}` : ""}
                </span>
              </Button>
            </div>
          )}

          {stableMessageRows.map(({ key, message }) => (
            <ChatMessageRow key={key} message={message} />
          ))}
        </div>
      </div>

      <div className="shrink-0 border-t bg-background px-3 py-3">
        <div className="flex items-center gap-2">
          {visibleAgentStatus && (
            <AgentStatusCard
              status={visibleAgentStatus}
              canAbort={canAbortSession}
              onAbort={() => void abortChat()}
            />
          )}
          {turnCount > 0 ? (
            <Badge
              variant="outline"
              className="shrink-0 border-cyan-500/40 bg-cyan-500/10 text-[10px] text-cyan-700 dark:text-cyan-300"
              title="你这条输入之后 agent 已跑的轮次（含自动续跑/后台回调；内部纠偏不计）"
            >
              第 {turnCount} 轮
            </Badge>
          ) : null}
        </div>

        <div className="flex gap-2">
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            multiple
            className="hidden"
            onChange={(event) => {
              handleImageFiles(event.target.files)
              event.target.value = ""
            }}
          />
          <Button
            type="button"
            size="icon"
            variant="outline"
            title="上传图片（多模态模型可看图）"
            onClick={() => imageInputRef.current?.click()}
            disabled={!handler || pendingImages.length >= MAX_CHAT_IMAGES}
          >
            <ImagePlus className="size-4" />
          </Button>
          <Input
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault()
                void sendMessage()
              } else if (event.key === "ArrowUp") {
                event.preventDefault()
                navigateInputHistory(-1)
              } else if (event.key === "ArrowDown") {
                event.preventDefault()
                navigateInputHistory(1)
              }
            }}
            placeholder={inputPlaceholder}
            disabled={!handler}
            className="flex-1"
          />
          <Button
            size="icon"
            variant="outline"
            title="中断当前任务（等同 Ctrl+C）"
            onClick={() => void abortChat()}
            disabled={!canAbortSession}
            className="text-red-600 hover:bg-red-50 disabled:opacity-40"
          >
            <Square className="size-4" />
          </Button>
          <Button
            size="icon"
            onClick={() => void sendMessage()}
            disabled={(!input.trim() && pendingImages.length === 0) || !handler}
          >
            <Send className="size-4" />
          </Button>
        </div>
        {pendingImages.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-2">
            {pendingImages.map((dataUrl, index) => (
              <div
                key={`${index}-${dataUrl.length}`}
                className="relative rounded-lg border bg-muted/20 p-1"
              >
                <img
                  src={dataUrl}
                  alt={`pending-${index}`}
                  className="h-14 w-14 rounded-md object-cover"
                />
                <button
                  type="button"
                  className="absolute -right-2 -top-2 flex size-5 items-center justify-center rounded-full border bg-background text-muted-foreground shadow-sm hover:text-foreground"
                  onClick={() =>
                    setPendingImages((prev) =>
                      prev.filter((_, i) => i !== index),
                    )
                  }
                >
                  <X className="size-3" />
                </button>
              </div>
            ))}
          </div>
        ) : null}
        <ChatTaskStrip itemId={itemId} />
      </div>
    </div>
  )
}
