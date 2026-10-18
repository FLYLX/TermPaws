import { OpenAPI } from "@/client"
import { apiRequest } from "@/lib/api-request"

export type RobotReplyMessageType =
  | "private"
  | "group"
  | "channel"
  | "command"
  | "mention"

export type RobotMentionMatchMode = "bot" | "any"

export type RobotConfig = {
  credentials?: Record<string, unknown>
  options?: Record<string, unknown> & {
    reply_message_types?: RobotReplyMessageType[]
    mention_match_mode?: RobotMentionMatchMode
    reply_context_window_seconds?: number
  }
}

export type RobotRecord = {
  id: string
  name: string
  platform: string
  protocol: string
  provider: string
  app_id: string | null
  app_secret: string | null
  bot_token: string | null
  use_websocket: boolean
  is_enabled: boolean
  config: RobotConfig | null
  owner_id: string
  created_at?: string | null
  updated_at?: string | null
}

export type RobotListResponse = {
  data: RobotRecord[]
  count: number
}

export type CreateRobotBindingPayload = {
  item_id: string
  allow_chat?: boolean
  receive_filtered_output?: boolean
  chat_alias?: string | null
  is_default_target?: boolean
}

export type UpdateRobotBindingPayload = {
  allow_chat?: boolean
  receive_filtered_output?: boolean
  chat_alias?: string | null
  is_default_target?: boolean
}

export type RobotBindingRecord = {
  robot_id: string
  item_id: string
  item_title: string
  allow_chat: boolean
  receive_filtered_output: boolean
  chat_alias: string | null
  route_key: string
  is_default_target: boolean
}

export type RobotConversationControllerStatus = {
  robot_id: string
  conversation_key: string
  conversation_type: string
  conversation_id: string
  item_id: string | null
  status: "awake" | "sleeping" | "processing"
  awake: boolean
  sleeping: boolean
  processing?: boolean
  generation: number
  expires_at: string | null
  processing_expires_at?: string | null
  processing_seconds_remaining?: number
  updated_at: string
  seconds_remaining: number
  pending_count?: number
  pending_messages?: RobotPendingReplyMessage[]
}

export type RobotPendingReplyMessage = {
  index: number
  item_id: string
  route_key: string
  sender_key: string
  sender_label: string
  trigger_reason: string
  message_preview: string
  enqueued_at: string
  direct_wakeup: boolean
  pending_reply_id?: string
  reply_ticket_id?: string
  reply_ticket_status?:
    | "pending"
    | "running"
    | "completed"
    | "sending"
    | "failed"
  command_preview?: string
  task_request_id?: string
  delivery_error?: string
  workflow_id?: string
  workflow_objective?: string
  workflow_status?: string
  workflow_current_step?: string
  workflow_latest_progress?: string
  workflow_blocker?: string
  workflow_steps?: Array<{
    step_id: string
    title: string
    status: string
    attempts: number
    note?: string
    evidence?: string
    last_error?: string
    recovery?: boolean
  }>
}

export type ItemRobotControllerStatusRecord = {
  robot_id: string
  robot_name: string
  is_enabled: boolean
  allow_chat: boolean
  route_key: string
  reply_context_window_seconds: number
  conversation_controllers: RobotConversationControllerStatus[]
}

export type ItemRobotControllerStatusResponse = {
  item_id: string
  robots: ItemRobotControllerStatusRecord[]
  count: number
}

export type RobotPlatformField = {
  key: string
  label: string
  required: boolean
  secret: boolean
}

export type RobotPlatformRecord = {
  id: string
  label: string
  provider: string
  description: string
  fields: RobotPlatformField[]
}

export type CreateRobotPayload = {
  name: string
  platform?: string
  protocol?: string
  provider?: string
  app_id?: string | null
  app_secret?: string | null
  bot_token?: string | null
  use_websocket?: boolean
  is_enabled?: boolean
  config?: RobotConfig
}

export type UpdateRobotPayload = Partial<CreateRobotPayload>

export function getRobotsQueryKey() {
  return ["robots"] as const
}

export function getRobotPlatformsQueryKey() {
  return ["robot-platforms"] as const
}

export function getRobotQueryKey(robotId: string) {
  return ["robot", robotId] as const
}

export function getRobotBindingsQueryKey(robotId: string) {
  return ["robot-bindings", robotId] as const
}

export function getItemRobotControllerStatusQueryKey(itemId: string) {
  return ["item-robot-controller-status", itemId] as const
}

export async function listRobots() {
  return apiRequest<RobotListResponse>("/api/v1/robots/")
}

export type RobotStickerTag = {
  tag: string
  files: string[]
}

export type RobotStickersResponse = {
  tags: RobotStickerTag[]
}

export function getRobotStickersQueryKey(robotId: string) {
  return ["robot-stickers", robotId] as const
}

export function robotStickerFileUrl(tag: string, name: string) {
  return `${OpenAPI.BASE}/api/v1/robots/stickers/file/${encodeURIComponent(tag)}/${encodeURIComponent(name)}`
}

export async function listRobotStickers() {
  return apiRequest<RobotStickersResponse>("/api/v1/robots/stickers")
}

export async function deleteRobotSticker(tag: string, name: string) {
  return apiRequest<{ message: string }>(
    `/api/v1/robots/stickers/${encodeURIComponent(tag)}/${encodeURIComponent(name)}`,
    { method: "DELETE" },
  )
}

export async function readRobot(robotId: string) {
  return apiRequest<RobotRecord>(`/api/v1/robots/${robotId}`)
}

export async function listRobotPlatforms() {
  return apiRequest<RobotPlatformRecord[]>("/api/v1/robots/platforms")
}

export async function getItemRobotControllerStatus(itemId: string) {
  return apiRequest<ItemRobotControllerStatusResponse>(
    `/api/v1/robots/items/${itemId}/conversation-controllers`,
  )
}

export async function createRobot(payload: CreateRobotPayload) {
  return apiRequest<RobotRecord>("/api/v1/robots/", {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

export async function deleteRobot(robotId: string) {
  return apiRequest<{ message: string }>(`/api/v1/robots/${robotId}`, {
    method: "DELETE",
  })
}

export async function updateRobot(
  robotId: string,
  payload: UpdateRobotPayload,
) {
  return apiRequest<RobotRecord>(`/api/v1/robots/${robotId}`, {
    method: "PUT",
    body: JSON.stringify(payload),
  })
}

export async function listRobotBindings(robotId: string) {
  return apiRequest<RobotBindingRecord[]>(`/api/v1/robots/${robotId}/items`)
}

export async function createRobotBinding(
  robotId: string,
  payload: CreateRobotBindingPayload,
) {
  return apiRequest<RobotBindingRecord>(`/api/v1/robots/${robotId}/items`, {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

export async function updateRobotBinding(
  robotId: string,
  itemId: string,
  payload: UpdateRobotBindingPayload,
) {
  return apiRequest<RobotBindingRecord>(
    `/api/v1/robots/${robotId}/items/${itemId}`,
    {
      method: "PUT",
      body: JSON.stringify(payload),
    },
  )
}

export async function deleteRobotBinding(robotId: string, itemId: string) {
  return apiRequest<{ message: string }>(
    `/api/v1/robots/${robotId}/items/${itemId}`,
    {
      method: "DELETE",
    },
  )
}

export type BridgeHealthResponse = {
  loaded_robot_count: number
  connected_bot_count: number
  onebot_client_count?: number
  onebot_clients?: Array<Record<string, unknown>>
  onebot_reverse_ws_url?: string | null
  platforms: string[]
  connected_identities: Array<string | number>
  robots: Record<string, { identity: string | number; connected: boolean }>
  error?: string
  connected?: boolean
}

export type RobotConnectionStatus = {
  robot_id: string
  identity?: string
  connected: boolean
  platform?: string
  reason?: string
  error?: string
}

export function getBridgeHealthQueryKey() {
  return ["robot-bridge-health"] as const
}

export function getRobotConnectionQueryKey(robotId: string) {
  return ["robot-connection", robotId] as const
}

export async function getBridgeHealth() {
  return apiRequest<BridgeHealthResponse>("/api/v1/robots/bridge/health")
}

export async function getRobotConnectionStatus(robotId: string) {
  return apiRequest<RobotConnectionStatus>(
    `/api/v1/robots/${robotId}/connection`,
  )
}

export type RobotDiagnoseResult = {
  robot_id: string
  robot_name: string
  robot_enabled: boolean
  platform: string
  checked_at?: string
  overall_status: "ok" | "degraded"
  chain: {
    robot_config: {
      status: string
      app_id: string | null
      provider: string
    }
    qq_to_bridge: {
      status: string
      connected: boolean
      identity?: string | number | null
      backend_reachable?: boolean
      error?: string
      stale_error?: string | null
      stale_error_at?: string | null
      bridge_checked_at?: string | null
      live?: boolean
    }
    napcat_socket: {
      status: string
      connected: boolean
      server_url?: string | number | null
      reverse_ws_url?: string | number | null
      ws_url?: string | number | null
      last_event?: string | null
      last_event_at?: string | null
      self_id?: string | number | null
      last_message_seen?: boolean
      last_socket_receive_seen?: boolean
      error?: string
    }
    items: Array<{
      item_id: string
      item_title: string
      allow_chat: boolean
      receive_filtered_output?: boolean
      is_default_target?: boolean
      route_key?: string
      daemon: {
        status: string
        online: boolean
      }
      agent_route?: {
        status: string
        chat_enabled: boolean
        route_key: string
        default_target: boolean
        filtered_output_enabled: boolean
      }
    }>
  }
}

export function getRobotDiagnoseQueryKey(robotId: string) {
  return ["robot-diagnose", robotId] as const
}

export async function diagnoseRobotChain(robotId: string) {
  return apiRequest<RobotDiagnoseResult>(`/api/v1/robots/${robotId}/diagnose`)
}

export type RobotDebugEvent = {
  timestamp: string
  direction: string
  event: string
  status: string
  message?: string | null
  payload: Record<string, unknown>
}

export type RobotDebugInfo = {
  robot: {
    id: string
    name: string
    platform: string
    provider: string
    is_enabled: boolean
    app_id: string | null
  }
  bridge: {
    url: string
    napcat_ws_url?: string | null
    onebot_reverse_ws_url?: string | null
    status: string
    loaded_robot_count: number
    connected_bot_count: number
    onebot_client_count?: number
    onebot_clients?: Array<Record<string, unknown>>
    connected: boolean
    identity?: string | null
    checked_at?: string | null
    cooldown_remaining_seconds?: number
    bot?: {
      self_id?: string | null
      adapter?: string | null
      class?: string | null
      bot_info?: Record<string, unknown>
    } | null
    bots?: Array<{
      self_id?: string | null
      adapter?: string | null
      class?: string | null
      bot_info?: Record<string, unknown>
    }>
    onebot_socket?: Record<string, unknown>
    last_platform_event_at?: string | null
    last_message_event_at?: string | null
    error?: string | null
  }
  diagnostics?: {
    event_count: number
    last_event_at?: string | null
    qq_event_hint?: string | null
  }
  events: RobotDebugEvent[]
}

export function getRobotDebugQueryKey(robotId: string) {
  return ["robot-debug", robotId] as const
}

export async function getRobotDebug(robotId: string) {
  return apiRequest<RobotDebugInfo>(`/api/v1/robots/${robotId}/debug`)
}

export async function reloadRobotBridge(robotId: string) {
  return apiRequest<{ success: boolean; message?: string; error?: string }>(
    `/api/v1/robots/${robotId}/reload`,
    { method: "POST" },
  )
}

export async function sendRobotDebugMessage(robotId: string, text: string) {
  return apiRequest<{
    success: boolean
    target_type?: string
    target_id?: string
  }>(`/api/v1/robots/${robotId}/debug/send`, {
    method: "POST",
    body: JSON.stringify({ text }),
  })
}

export type RobotManualMessageTargetType = "group" | "private"

export type RobotConversationMemoryEntry = {
  robot_id: string
  conversation_key: string
  path: string
  exists: boolean
  size_bytes: number
  updated_at: string | null
  filename: string
}

export type RobotConversationMemoryListResponse = {
  data: RobotConversationMemoryEntry[]
  count: number
}

export type RobotConversationMemoryReadResponse = {
  memory: string
  info: {
    robot_id: string
    conversation_key: string
    path: string
    exists: boolean
    size_bytes: number
    updated_at: string | null
  }
}

export async function sendRobotManualMessage(
  robotId: string,
  payload: {
    target_type: RobotManualMessageTargetType
    target_id: string
    text: string
  },
) {
  return apiRequest<{
    success: boolean
    target_type?: string
    target_id?: string
  }>(`/api/v1/robots/${robotId}/messages/send`, {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

function buildRobotAuthHeaders() {
  const token = localStorage.getItem("access_token") || ""
  return new Headers({ Authorization: `Bearer ${token}` })
}

function encodeConversationKey(conversationKey: string) {
  return encodeURIComponent(conversationKey)
}

export function getRobotConversationMemoryQueryKey(robotId: string) {
  return ["robot-conversation-memory", robotId] as const
}

export async function listRobotConversationMemory(robotId: string) {
  return apiRequest<RobotConversationMemoryListResponse>(
    `/api/v1/robots/${robotId}/conversation-memory`,
  )
}

export async function readRobotConversationMemory(
  robotId: string,
  conversationKey: string,
) {
  return apiRequest<RobotConversationMemoryReadResponse>(
    `/api/v1/robots/${robotId}/conversation-memory/${encodeConversationKey(conversationKey)}`,
  )
}

export async function importRobotConversationMemory(
  robotId: string,
  conversationKey: string,
  payload: { content: string; append?: boolean },
) {
  return apiRequest<{
    success: boolean
    info: RobotConversationMemoryReadResponse["info"]
  }>(
    `/api/v1/robots/${robotId}/conversation-memory/${encodeConversationKey(conversationKey)}/import`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  )
}

export async function downloadRobotConversationMemory(
  robotId: string,
  entry: RobotConversationMemoryEntry,
) {
  const response = await fetch(
    `${OpenAPI.BASE}/api/v1/robots/${robotId}/conversation-memory/${encodeConversationKey(entry.conversation_key)}/export`,
    {
      headers: buildRobotAuthHeaders(),
    },
  )

  if (!response.ok) {
    let detail = "Request failed"
    try {
      const payload = await response.json()
      detail = payload.detail || detail
    } catch {
      detail = response.statusText || detail
    }
    throw new Error(detail)
  }

  const blob = await response.blob()
  const url = window.URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = entry.filename || `${entry.conversation_key}.log`
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => {
    window.URL.revokeObjectURL(url)
  }, 0)
}
