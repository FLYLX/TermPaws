import { useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute, redirect, useNavigate } from "@tanstack/react-router"
import {
  Bot,
  Brain,
  FileText,
  Filter,
  Layers,
  Loader2,
  MessageSquare,
  Server,
  Settings,
  Stethoscope,
  Terminal,
  Trash2,
  X,
  Zap,
} from "lucide-react"
import {
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import { createPortal } from "react-dom"

import {
  ApiError,
  ItemHandlerAssociationsService,
  ItemHandlersService,
  ItemsService,
  McpService,
  SkillsService,
} from "@/client"
import AddItemHandler from "@/components/ItemHandlers/AddItemHandler"
import AddItemToHandler from "@/components/ItemHandlers/AddItemToHandler"
import { ChatLogPanel } from "@/components/ItemHandlers/ChatLogPanel"
import { HandlerSettingsPanel } from "@/components/ItemHandlers/HandlerSettingsPanel"
import { MCPSelector as McpSelector } from "@/components/ItemHandlers/McpSelector"
import { SkillSelector } from "@/components/ItemHandlers/SkillSelector"
import { TerminalWsPanel } from "@/components/ItemHandlers/TerminalWsPanel"
import AddItem from "@/components/Items/AddItem"
import { ChatPanel } from "@/components/Items/ChatPanel"
import { ItemConfigPanel } from "@/components/Items/ItemConfigPanel"
import { ItemFilesPanel } from "@/components/Items/ItemFilesPanel"
import { ItemFiltersPanel } from "@/components/Items/ItemFiltersPanel"
import ItemHandlersList from "@/components/Items/ItemHandlersList"
import { RobotConversationDebugPanel } from "@/components/Items/RobotConversationDebugPanel"
import { TerminalOutputPanel } from "@/components/Items/TerminalOutputPanel"
import { TokenUsagePanel } from "@/components/Items/TokenUsagePanel"
import { KnowledgeBindingSelector } from "@/components/Knowledge/KnowledgeBindingSelector"
import { useI18n } from "@/components/locale-provider"
import { MemoryManager } from "@/components/memory-manager"
import {
  createRobotBinding,
  deleteRobot,
  deleteRobotBinding,
  diagnoseRobotChain,
  getRobotPlatformsQueryKey,
  getRobotsQueryKey,
  listRobotBindings,
  listRobotPlatforms,
  listRobots,
  updateRobotBinding,
} from "@/components/Robots/api"
import { RobotDetail } from "@/components/Robots/RobotDetail"
import { CreateRobotDialog } from "@/components/Robots/RobotManager"
import { ScheduledTasksManager } from "@/components/scheduled-tasks-manager"
import { isLoggedIn } from "@/hooks/useAuth"
import useCustomToast from "@/hooks/useCustomToast"
import { getPluginsQueryOptions, isPluginEnabled } from "@/lib/plugins-api"
import { extractErrorMessage } from "@/utils"

export const Route = createFileRoute("/_layout/item-handlers/$itemHandlerId")({
  component: TerminalDispatcherPage,
  beforeLoad: async () => {
    if (!isLoggedIn()) {
      throw redirect({ to: "/login" })
    }
  },
  head: () => ({
    meta: [{ title: "终端调度器 - TermPaws" }],
  }),
})

const DISPATCHER_BOARD_CSS = `
.dispatcher-board { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #1f1f1f; background: linear-gradient(135deg, #f5f1e8 0%, #f2f4ef 55%, #e9efec 100%); }
.dispatcher-board::before { content: ""; position: absolute; inset: 0; pointer-events: none; background-image: linear-gradient(to right, rgba(58,58,58,0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(58,58,58,0.05) 1px, transparent 1px); background-size: 28px 28px; }
.dispatcher-board .text-muted-foreground { color: #565654; }
.dispatcher-board .text-foreground { color: #1f1f1f; }
.sketch-box { background: rgba(255,255,255,.62); backdrop-filter: blur(3px) saturate(1.05); -webkit-backdrop-filter: blur(3px) saturate(1.05); border: 2px solid #3a3a3a; box-shadow: 3px 4px 0 rgba(0,0,0,.10); color: #1f1f1f; }
.glass-box { background: rgba(255,255,255,.42); backdrop-filter: blur(3px) saturate(1.05); -webkit-backdrop-filter: blur(3px) saturate(1.05); border: 2px solid rgba(58,58,58,.75); box-shadow: 3px 4px 0 rgba(0,0,0,.07); color: #1f1f1f; border-radius: 14px; }
.glass-box.sketch-hover:hover { background: rgba(255,255,255,.62); }
.sketch-a { border-radius: 255px 15px 225px 15px / 15px 225px 15px 255px; }
.sketch-b { border-radius: 15px 225px 15px 255px / 255px 15px 225px 15px; }
.sketch-c { border-radius: 225px 12px 255px 18px / 18px 255px 12px 225px; }
.sketch-hover { transition: transform .15s ease, box-shadow .15s ease; cursor: pointer; }
.sketch-hover:hover { transform: translate(-1px, -2px) rotate(-.3deg); box-shadow: 4px 6px 0 rgba(0,0,0,.14); background: #fbfbfa; }
.sketch-active { outline: 3px solid #6b7280; outline-offset: 2px; }
.sketch-tray { background: rgba(244,244,243,.5); backdrop-filter: blur(3px); border: 2px dashed #6b6b69; border-radius: 18px; }
.sketch-tray-hint { background: #e5e5e3; border-color: #525250; outline: 3px dashed #6b7280; outline-offset: 2px; }
.sketch-drop-target { outline: 3px dashed #4b5563; outline-offset: 3px; }
.board-wires { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 20; }
.board-wires .wire-visible { stroke: #3a3a3a; stroke-width: 2; fill: none; stroke-linecap: round; pointer-events: none; }
.board-wires .wire-bot { stroke: #6baed6; }
.board-wires .wire-hit { stroke: rgba(0,0,0,0); stroke-width: 18; fill: none; stroke-linecap: round; pointer-events: stroke; cursor: pointer; }
.board-wires g:hover .wire-visible { stroke: #b91c1c; stroke-width: 3; }
.board-wires .wire-dash { stroke-dasharray: 8 5; opacity: .45; }
.board-wires .wire-preview { stroke: #6b7280; stroke-width: 3; stroke-dasharray: 9 5; opacity: .95; }
.dispatcher-board .board-grid, .dispatcher-board .board-grid section, .dispatcher-board .board-grid aside { pointer-events: none; }
.dispatcher-board .board-grid button, .dispatcher-board .board-grid a, .dispatcher-board .board-grid .wire-handle, .dispatcher-board .board-grid .item-del, .dispatcher-board .board-grid select, .dispatcher-board .board-grid input, .dispatcher-board .board-grid textarea, .dispatcher-board .board-grid .pe-auto, .dispatcher-board .board-grid .pe-auto * { pointer-events: auto; }
.wire-handle { position: absolute; right: -7px; top: 50%; width: 13px; height: 13px; margin-top: -6px; border-radius: 9999px; background: #fff; border: 2px solid #3a3a3a; cursor: crosshair; z-index: 30; }
.wire-handle:hover { background: #e5e5e3; border-color: #1f1f1f; }
.board-card { animation: card-in .32s cubic-bezier(.2,.8,.3,1) backwards; }
.board-card:nth-child(3n+1) { animation-delay: .04s; }
.board-card:nth-child(3n+2) { animation-delay: .09s; }
.board-card:nth-child(3n) { animation-delay: .14s; }
.wire-preview { animation: dash-flow .9s linear infinite; }
.main-terminal-block { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 4px; min-height: 130px; width: 100%; cursor: pointer; }
.terminal-titlebar { display: flex; align-items: center; gap: 6px; background: #efece3; color: #3a3a3a; padding: 6px 10px; border-bottom: 2px solid #3a3a3a; }
.terminal-titlebar .dot { width: 9px; height: 9px; border-radius: 50%; background: #4a5563; border: 1.5px solid #161616; }
.drawer-backdrop { position: fixed; inset: 0; z-index: 40; background: rgba(0,0,0,.28); animation: drawer-fade .15s ease-out; }
.terminal-detail-panel { position: fixed; right: 0; top: 0; bottom: 0; z-index: 45; width: min(720px, 94vw); background: #fff; border-left: 2px solid #3a3a3a; box-shadow: -6px 0 0 rgba(0,0,0,.10); display: flex; flex-direction: column; animation: drawer-in-right .2s ease-out; }
.terminal-detail-panel .terminal-frame { border: none; border-radius: 0; box-shadow: none; flex: 1; min-height: 0; display: flex; flex-direction: column; }
.terminal-detail-panel .terminal-body { flex: 1; min-height: 0; background: #fff; }
.terminal-frame { border: 3px solid #3a3a3a; box-shadow: 3px 4px 0 rgba(0,0,0,.14); border-radius: 18px 60px 18px 60px / 60px 18px 60px 18px; overflow: hidden; background: #fff; }
@keyframes drawer-in-right { from { transform: translateX(100%); } to { transform: translateX(0); } }
@keyframes drawer-fade { from { opacity: 0; } to { opacity: 1; } }
.bot-detail-panel { position: fixed; left: 0; right: 0; bottom: 0; top: 0; z-index: 60; background: linear-gradient(135deg, #f5f1e8 0%, #f2f4ef 55%, #e9efec 100%); color: #1f1f1f; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; display: flex; flex-direction: column; animation: bot-panel-up .25s cubic-bezier(.2,.8,.3,1); }
.bot-detail-panel::before { content: ""; position: absolute; inset: 0; pointer-events: none; background-image: linear-gradient(to right, rgba(58,58,58,0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(58,58,58,0.05) 1px, transparent 1px); background-size: 28px 28px; }
.bot-detail-panel > * { position: relative; z-index: 1; }
@keyframes bot-panel-up { from { transform: translateY(60px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
.zoom-backdrop { position: fixed; inset: 0; z-index: 50; display: flex; align-items: center; justify-content: center; padding: 2rem; background: rgba(255,255,255,.72); backdrop-filter: blur(2px); }
.zoom-backdrop-feature { z-index: 70; }
.zoom-card { width: min(980px, 94vw); max-height: 84vh; overflow: auto; animation: dispatcher-zoom .16s ease-out; background: #fff; color: #1f1f1f; display: flex; flex-direction: column; }
.feature-zoom-body { flex: 1; min-height: 0; }
.feature-zoom-fill { height: 62vh; }
.sketch-content { color: #1f1f1f; }
.sketch-content .rounded-lg, .sketch-content .rounded-md { border-color: #3a3a3a; border-width: 2px; }
.sketch-content .text-muted-foreground { color: #565654; }
.sketch-content button { font-weight: 600; }
@keyframes dispatcher-zoom { from { transform: scale(.85); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.confirm-backdrop { position: fixed; inset: 0; z-index: 80; display: flex; align-items: center; justify-content: center; padding: 2rem; background: rgba(0,0,0,.3); animation: drawer-fade .15s ease-out; }
.confirm-card { width: min(400px, 92vw); background: #fff; padding: 20px; animation: dispatcher-zoom .16s ease-out; }
.terminal-edge-panel { border-left: none; border-radius: 0 18px 18px 0 / 18px 18px 18px 0; animation: edge-slide-in .3s cubic-bezier(.2,.8,.3,1); }
@keyframes edge-slide-in { from { transform: translateX(-100%); } to { transform: translateX(0); } }
.terminal-edge-panel-right { border-right: none; border-radius: 18px 0 0 18px / 18px 18px 0 18px; animation: edge-slide-in-right .3s cubic-bezier(.2,.8,.3,1); }
@keyframes edge-slide-in-right { from { transform: translateX(100%); } to { transform: translateX(0); } }
.terminal-fullscreen { position: fixed; inset: 0; z-index: 55; background: #f4f4f3; display: flex; flex-direction: column; animation: terminal-slide-in .25s cubic-bezier(.2,.8,.3,1); }
@keyframes terminal-slide-in { from { transform: translateX(60px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
.terminal-fullscreen-header { display: flex; align-items: center; gap: 10px; padding: 10px 14px; border-bottom: 2px solid #3a3a3a; background: #fff; }
.terminal-fullscreen-body { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0,1fr) auto; gap: 12px; padding: 14px; overflow: hidden; }
.terminal-fullscreen-body.terminal-body-3col { grid-template-columns: minmax(0,1fr) minmax(0,1fr) auto; }
.terminal-titlebar-light { background: rgba(255,255,255,.75); backdrop-filter: blur(4px); color: #3a3a3a; }
.terminal-fullscreen-body .terminal-frame { flex: 1; min-height: 0; display: flex; flex-direction: column; height: 100%; }
.terminal-fullscreen-body .terminal-body { flex: 1; min-height: 0; background: #fff; }
.feature-icon-rail { display: flex; flex-direction: column; gap: 6px; width: 44px; overflow: hidden; overflow-y: auto; padding: 4px; border: 2px solid #3a3a3a; border-radius: 14px; background: #fff; box-shadow: 2px 3px 0 rgba(0,0,0,.10); transition: width .22s ease; scrollbar-width: none; }
.feature-icon-rail::-webkit-scrollbar { display: none; }
.feature-icon-rail:hover { width: 168px; }
.feature-icon-btn { display: flex; align-items: center; gap: 8px; width: 100%; padding: 7px 8px; border-radius: 10px; color: #565654; white-space: nowrap; cursor: pointer; flex-shrink: 0; }
.feature-icon-btn:hover { background: #f0f0ef; color: #1f1f1f; }
.feature-icon-btn.feature-icon-active { background: #3a3a3a; color: #fff; }
.feature-icon-btn svg { flex-shrink: 0; }
.feature-icon-label { font-size: 11px; font-weight: 700; opacity: 0; transition: opacity .15s ease; }
.feature-icon-rail:hover .feature-icon-label { opacity: 1; }
.bot-tag { position: relative; margin-top: 16px; transform-origin: bottom center; transform: rotate(var(--tilt, 0deg)); border: 2px solid #3a3a3a; background: #fff; color: #242424; padding: 8px 12px 16px; font-weight: 700; font-size: 12px; display: flex; flex-direction: column; align-items: center; gap: 3px; box-shadow: 3px -4px 0 rgba(0,0,0,.10); border-radius: 18px 18px 10px 10px; cursor: pointer; transition: transform .18s ease, box-shadow .18s ease, background .18s ease; }
.bot-tag .tag-hole { position: absolute; bottom: 4px; top: auto; left: 50%; margin-left: -5px; width: 10px; height: 10px; border: 2px solid #3a3a3a; border-radius: 9999px; background: #f4f4f3; }
.bot-tag:hover { transform: rotate(0deg) translateY(-4px); box-shadow: 4px -6px 0 rgba(0,0,0,.14); background: #fbfbfa; }
.bot-tag .item-del { top: auto; right: auto; bottom: -16px; left: 50%; transform: translateX(-50%); }
.bot-tag-tool { display: inline-flex; align-items: center; justify-content: center; }
.bot-tag-tool .btn-add-compact { min-width: 0; height: auto; padding: 0; }
.bot-diagnose-tab { position: absolute; right: -26px; top: 50%; transform: translateY(-50%); display: inline-flex; align-items: center; justify-content: center; width: 24px; height: 26px; border: 2px solid #dc2626; border-left: none; background: #fff; color: #dc2626; border-radius: 0 9px 9px 0; cursor: pointer; z-index: 5; }
.bot-diagnose-tab:hover { background: #fef2f2; color: #b91c1c; }
.scroll-arrow { display: flex; align-items: center; justify-content: center; padding: 0 2px; border: none; background: transparent; color: #1f1f1f; font-size: 20px; font-weight: 900; line-height: 1.1; cursor: pointer; flex-shrink: 0; text-shadow: 1px 1px 0 rgba(0,0,0,.15); }
.scroll-arrow:hover { color: #6b7280; transform: scale(1.15); }
.scroll-col { scrollbar-width: none; }
.scroll-col::-webkit-scrollbar { display: none; }
.host-shade-0 { background: rgba(255,255,255,.45); backdrop-filter: blur(2px); }
.host-shade-1 { background: rgba(240,240,239,.45); backdrop-filter: blur(2px); }
.host-shade-2 { background: rgba(227,227,225,.45); backdrop-filter: blur(2px); }
.host-shade-3 { background: rgba(214,214,211,.45); backdrop-filter: blur(2px); }
.host-shade-4 { background: rgba(201,201,198,.45); backdrop-filter: blur(2px); }
.btn-add-compact { display: inline-flex; align-items: center; justify-content: center; gap: 0; border: 2px solid #16a34a !important; background: #fff !important; color: #16a34a !important; font-weight: 900; font-size: 15px; line-height: 1; min-width: 26px; height: 26px; padding: 0; border-radius: 9999px !important; cursor: pointer; box-shadow: 1px 2px 0 rgba(0,0,0,.12); }
.btn-add-compact:hover { background: #f0faf4 !important; color: #15803d !important; }
.btn-add-compact svg { margin-right: 0 !important; width: 14px; height: 14px; color: #16a34a !important; stroke-width: 3; }
.edge-add-wrap { display: flex; justify-content: center; padding: 2px 0 8px; }
.edge-add-frame { display: inline-flex; align-items: center; justify-content: center; padding: 5px; border: 2px solid #3a3a3a; background: rgba(255,255,255,.62); backdrop-filter: blur(3px); box-shadow: 3px 4px 0 rgba(0,0,0,.10); transition: transform .15s ease, box-shadow .15s ease; }
.edge-add-frame:hover { transform: translate(-1px, -2px) rotate(-.5deg); box-shadow: 4px 6px 0 rgba(0,0,0,.14); }
.edge-add-frame-left { border-radius: 18px 60px 18px 60px / 60px 18px 60px 18px; }
.edge-add-frame-right { border-radius: 60px 18px 60px 18px / 18px 60px 18px 60px; }
.item-del { position: absolute; top: 50%; right: 6px; transform: translateY(-50%); display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; border: none; background: transparent; color: #dc2626; cursor: pointer; z-index: 6; }
.item-del:hover { background: transparent; color: #b91c1c; }
.item-del svg { width: 18px; height: 18px; stroke-width: 3; }
.dispatcher-box-body { padding-right: 30px; }
.edge-left-box { border-left: none !important; border-top-left-radius: 0 !important; border-bottom-left-radius: 0 !important; }
.edge-right-box { border-right: none !important; border-top-right-radius: 0 !important; border-bottom-right-radius: 0 !important; }
.terminal-running { border-color: #16a34a !important; }
.terminal-stopped { border-color: #dc2626 !important; }
`

function ScrollColumn({
  children,
  className,
  arrows = true,
}: {
  children: ReactNode
  className?: string
  arrows?: boolean
}) {
  const ref = useRef<HTMLDivElement | null>(null)
  const scroll = (dir: number) => {
    ref.current?.scrollBy({ top: dir * 180, behavior: "smooth" })
  }
  return (
    <div className={`flex min-h-0 flex-1 flex-col gap-1 ${className ?? ""}`}>
      {arrows ? (
        <button
          type="button"
          className="scroll-arrow"
          onClick={() => scroll(-1)}
          aria-label="向上滚动"
        >
          ▲
        </button>
      ) : null}
      <div ref={ref} className="scroll-col min-h-0 flex-1 overflow-y-auto">
        {children}
      </div>
      {arrows ? (
        <button
          type="button"
          className="scroll-arrow"
          onClick={() => scroll(1)}
          aria-label="向下滚动"
        >
          ▼
        </button>
      ) : null}
    </div>
  )
}

function TerminalMemoryPanel({ connectedItems }: { connectedItems: any[] }) {
  const [selectedItemId, setSelectedItemId] = useState("")
  const effectiveItemId = connectedItems.some(
    (item) => String(item.id) === selectedItemId,
  )
    ? selectedItemId
    : String(connectedItems[0]?.id ?? "")

  return (
    <div className="space-y-3">
      {connectedItems.length > 1 ? (
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">
            入口终端（记忆池为调度器共享，选哪个都一样）：
          </span>
          <select
            className="rounded-md border bg-background px-2 py-1"
            value={effectiveItemId}
            onChange={(event) => setSelectedItemId(event.target.value)}
          >
            {connectedItems.map((item) => (
              <option key={item.id} value={String(item.id)}>
                {item.title || item.id}
              </option>
            ))}
          </select>
        </div>
      ) : null}
      {effectiveItemId ? <MemoryManager itemId={effectiveItemId} /> : null}
    </div>
  )
}

export function TerminalDispatcherBoard({
  itemHandlerId,
  itemHandler,
}: {
  itemHandlerId: string
  itemHandler: any
}) {
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const { t } = useI18n()
  const { showSuccessToast, showErrorToast } = useCustomToast()

  const { data: allHandlersData } = useQuery({
    queryKey: ["itemHandlers"],
    queryFn: () =>
      ItemHandlersService.readItemHandlers({ skip: 0, limit: 100 }),
  })
  const allHandlers = useMemo<any[]>(() => {
    if (Array.isArray(allHandlersData)) {
      return allHandlersData
    }
    return (allHandlersData as any)?.data ?? []
  }, [allHandlersData])

  const { data: connectedItems } = useQuery({
    queryFn: () =>
      ItemHandlerAssociationsService.getItemsForHandler({ itemHandlerId }),
    queryKey: ["itemHandler-items", itemHandlerId],
    enabled: Boolean(itemHandlerId),
  })
  const connectedItemsList = useMemo<any[]>(
    () => (connectedItems as any[]) || [],
    [connectedItems],
  )
  const connectedItemIds = useMemo(
    () => new Set(connectedItemsList.map((item: any) => String(item.id))),
    [connectedItemsList],
  )
  const { data: allItemsData } = useQuery({
    queryKey: ["items"],
    queryFn: () => ItemsService.readItems(),
  })

  const [selectedTerminalId, setSelectedTerminalId] = useState("")
  const [mainView, setMainView] = useState<string>("chat")
  const [expandedCard, setExpandedCard] = useState<string | null>(null)
  const [terminalPanelOpen, setTerminalPanelOpen] = useState(false)
  const [botDetailId, setBotDetailId] = useState<string | null>(null)
  const [confirmState, setConfirmState] = useState<{
    message: string
    action: () => void
  } | null>(null)

  const askConfirm = useCallback((message: string, action: () => void) => {
    setConfirmState({ message, action })
  }, [])

  const { data: skillsData, isLoading: skillsLoading } = useQuery({
    queryKey: ["skills"],
    queryFn: () => SkillsService.listSkills({}),
    enabled: expandedCard === "skills",
  })
  const { data: mcpData, isLoading: mcpLoading } = useQuery({
    queryKey: ["mcp-servers"],
    queryFn: () => McpService.listMcpServers(),
    enabled: expandedCard === "mcp",
  })
  const skills = skillsData?.data || []
  const enabledSkills: string[] = itemHandler?.enabled_skills ?? []
  const mcpServers = mcpData?.data || []
  const enabledMcpServers: string[] = itemHandler?.enabled_mcp_servers ?? []
  const enabledKnowledgeFiles: string[] =
    itemHandler?.enabled_knowledge_files ?? []

  useEffect(() => {
    if (!expandedCard && !terminalPanelOpen && !botDetailId) {
      return
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (botDetailId) {
          setBotDetailId(null)
        } else {
          setExpandedCard(null)
          setTerminalPanelOpen(false)
        }
      }
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [expandedCard, terminalPanelOpen, botDetailId])

  const handleSkillToggle = async (skillId: string, enable: boolean) => {
    if (!itemHandlerId) {
      showErrorToast("先创建一个调度器")
      return
    }
    try {
      const newSkills = enable
        ? [...enabledSkills, skillId]
        : enabledSkills.filter((id) => id !== skillId)
      await ItemHandlersService.updateItemHandler({
        id: itemHandlerId,
        requestBody: { enabled_skills: newSkills },
      })
      showSuccessToast("技能配置已更新")
      queryClient.invalidateQueries({
        queryKey: ["itemHandler", itemHandlerId],
      })
    } catch (error) {
      showErrorToast(
        error instanceof ApiError
          ? extractErrorMessage(error)
          : "技能配置更新失败",
      )
    }
  }

  const handleMcpServerToggle = async (serverName: string, enable: boolean) => {
    if (!itemHandlerId) {
      showErrorToast("先创建一个调度器")
      return
    }
    try {
      const newServers = enable
        ? [...enabledMcpServers, serverName]
        : enabledMcpServers.filter((name) => name !== serverName)
      await ItemHandlersService.updateItemHandler({
        id: itemHandlerId,
        requestBody: { enabled_mcp_servers: newServers },
      })
      showSuccessToast("MCP 配置已更新")
      queryClient.invalidateQueries({
        queryKey: ["itemHandler", itemHandlerId],
      })
    } catch (error) {
      showErrorToast(
        error instanceof ApiError
          ? extractErrorMessage(error)
          : "MCP 配置更新失败",
      )
    }
  }

  const handleKnowledgeToggle = async (filePath: string, enable: boolean) => {
    if (!itemHandlerId) {
      showErrorToast("先创建一个调度器")
      return
    }
    try {
      const newFiles = enable
        ? [...enabledKnowledgeFiles, filePath]
        : enabledKnowledgeFiles.filter((path) => path !== filePath)
      await ItemHandlersService.updateItemHandler({
        id: itemHandlerId,
        requestBody: { enabled_knowledge_files: newFiles } as any,
      })
      showSuccessToast("知识库配置已更新")
      queryClient.invalidateQueries({
        queryKey: ["itemHandler", itemHandlerId],
      })
    } catch (error) {
      showErrorToast(
        error instanceof ApiError
          ? extractErrorMessage(error)
          : "知识库配置更新失败",
      )
    }
  }

  const handleDisconnect = async (itemId: string) => {
    try {
      await ItemHandlerAssociationsService.removeItemFromHandler({
        itemHandlerId,
        itemId,
      })
      showSuccessToast(t("itemHandlers.detail.itemDisconnected"))
      queryClient.invalidateQueries({
        queryKey: ["itemHandler-items", itemHandlerId],
      })
      queryClient.invalidateQueries({
        queryKey: ["itemHandler", itemHandlerId],
      })
      queryClient.invalidateQueries({ queryKey: ["itemHandlers"] })
      queryClient.invalidateQueries({ queryKey: ["items"] })
    } catch (_error) {
      showErrorToast(t("itemHandlers.detail.itemDisconnectFailed"))
    }
  }

  const deleteDispatcherById = (handler: any) => {
    askConfirm(`确认删除调度器「${handler.name}」？`, async () => {
      try {
        await ItemHandlersService.deleteItemHandler({ id: String(handler.id) })
        showSuccessToast("调度器已删除")
        queryClient.invalidateQueries({ queryKey: ["itemHandlers"] })
        if (String(handler.id) === String(itemHandlerId)) {
          void navigate({ to: "/item-handlers" })
        }
      } catch (error) {
        showErrorToast(
          error instanceof Error ? error.message : "删除调度器失败",
        )
      }
    })
  }

  const allItemsList = ((allItemsData as any)?.data || []) as any[]
  const unboundItems = allItemsList.filter(
    (item: any) => !connectedItemIds.has(String(item.id)),
  )

  const handleConnect = async (itemId: string) => {
    try {
      await ItemHandlerAssociationsService.addItemToHandler({
        requestBody: { item_handler_id: itemHandlerId, item_id: itemId },
      })
      showSuccessToast(t("itemHandlers.detail.itemConnected"))
      queryClient.invalidateQueries({
        queryKey: ["itemHandler-items", itemHandlerId],
      })
      queryClient.invalidateQueries({
        queryKey: ["itemHandler", itemHandlerId],
      })
      queryClient.invalidateQueries({ queryKey: ["itemHandlers"] })
      queryClient.invalidateQueries({ queryKey: ["items"] })
    } catch (_error) {
      showErrorToast(t("itemHandlers.detail.itemConnectFailed"))
    }
  }

  const pluginsQuery = useQuery({
    ...getPluginsQueryOptions(),
    enabled: Boolean(itemHandlerId),
  })
  const robotPluginEnabled = pluginsQuery.data
    ? isPluginEnabled(pluginsQuery.data, "TermPaws.robot")
    : true
  const terminalWsPluginEnabled = pluginsQuery.data
    ? isPluginEnabled(pluginsQuery.data, "TermPaws.terminal_ws")
    : true

  const robotsQuery = useQuery({
    queryKey: getRobotsQueryKey(),
    queryFn: () => listRobots(),
    enabled: robotPluginEnabled,
  })
  const dispatcherRobots = robotsQuery.data?.data ?? []
  const platformsQuery = useQuery({
    queryKey: getRobotPlatformsQueryKey(),
    queryFn: () => listRobotPlatforms(),
    enabled: robotPluginEnabled,
  })
  const robotPlatforms = Array.isArray(platformsQuery.data)
    ? platformsQuery.data
    : ((platformsQuery.data as any)?.data ?? [])
  const [diagnoseOpen, setDiagnoseOpen] = useState(false)
  const [diagnoseResults, setDiagnoseResults] = useState<any[]>([])
  const [isDiagnosing, setIsDiagnosing] = useState(false)

  const runDiagnose = async () => {
    setDiagnoseOpen(true)
    setIsDiagnosing(true)
    try {
      const results = await Promise.all(
        dispatcherRobots.map(async (robot: any) => {
          try {
            return await diagnoseRobotChain(String(robot.id))
          } catch (error) {
            return {
              robot_name: robot.name,
              overall_status: "error",
              error: error instanceof Error ? error.message : String(error),
            }
          }
        }),
      )
      setDiagnoseResults(results)
    } finally {
      setIsDiagnosing(false)
    }
  }

  const deleteRobotById = (robot: any) => {
    askConfirm(`确认删除机器人「${robot.name}」？`, async () => {
      try {
        await deleteRobot(String(robot.id))
        showSuccessToast("机器人已删除")
        invalidateBots()
      } catch (error) {
        showErrorToast(
          error instanceof Error ? error.message : "删除机器人失败",
        )
      }
    })
  }
  const botBindingsQuery = useQuery({
    queryKey: [
      "dispatcher-bot-bindings",
      itemHandlerId,
      dispatcherRobots.map((robot: any) => robot.id).join(","),
    ],
    queryFn: async () =>
      Promise.all(
        dispatcherRobots.map(async (robot: any) => ({
          robot,
          bindings: ((await listRobotBindings(robot.id)) as any[]).filter(
            (binding) => connectedItemIds.has(String(binding.item_id)),
          ),
        })),
      ),
    enabled: dispatcherRobots.length > 0,
  })
  const boundBots = useMemo(
    () =>
      (botBindingsQuery.data ?? []).filter(
        (entry: any) => entry.bindings.length > 0,
      ),
    [botBindingsQuery.data],
  )
  const unboundBots = useMemo(
    () =>
      (botBindingsQuery.data ?? []).filter(
        (entry: any) => entry.bindings.length === 0,
      ),
    [botBindingsQuery.data],
  )

  const [panelSettingsOpen, setPanelSettingsOpen] = useState(false)

  type WireDrag = {
    mode: "bind-item" | "bind-bot" | "wire-from-dispatcher"
    id: string
    from: { x: number; y: number }
    to: { x: number; y: number }
  }
  const [wireDrag, setWireDrag] = useState<WireDrag | null>(null)
  const [dropHint, setDropHint] = useState<"dispatcher" | "tray" | null>(null)
  const wireDragRef = useRef<WireDrag | null>(null)
  const dispatcherNodeRef = useRef<HTMLDivElement | null>(null)

  const hitRect = (point: { x: number; y: number }, rect: DOMRect) =>
    point.x >= rect.left &&
    point.x <= rect.right &&
    point.y >= rect.top &&
    point.y <= rect.bottom

  const startWireDrag = (
    event: React.PointerEvent,
    mode: WireDrag["mode"],
    id: string,
    anchorKey: string,
  ) => {
    const board = boardRef.current
    if (!board) {
      return
    }
    event.preventDefault()
    const boardRect = board.getBoundingClientRect()
    const anchorEl = anchorRefs.current.get(anchorKey)
    const anchorRect = anchorEl?.getBoundingClientRect()
    const from = {
      x: (anchorRect ? anchorRect.right : event.clientX) - boardRect.left,
      y:
        (anchorRect
          ? (anchorRect.top + anchorRect.bottom) / 2
          : event.clientY) - boardRect.top,
    }
    const initial: WireDrag = {
      mode,
      id,
      from,
      to: {
        x: event.clientX - boardRect.left,
        y: event.clientY - boardRect.top,
      },
    }
    wireDragRef.current = initial
    setWireDrag(initial)

    const onMove = (ev: PointerEvent) => {
      const current = wireDragRef.current
      if (!current) {
        return
      }
      const to = {
        x: ev.clientX - boardRect.left,
        y: ev.clientY - boardRect.top,
      }
      let hint: "dispatcher" | "tray" | null = null
      const dispatcherRect =
        dispatcherNodeRef.current?.getBoundingClientRect() ?? null
      if (current.mode === "bind-item") {
        if (
          dispatcherRect &&
          hitRect({ x: ev.clientX, y: ev.clientY }, dispatcherRect)
        ) {
          hint = "dispatcher"
        }
      }
      wireDragRef.current = { ...current, to }
      setWireDrag(wireDragRef.current)
      setDropHint(hint)
    }
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", onMove)
      window.removeEventListener("pointerup", onUp)
      const ended = wireDragRef.current
      wireDragRef.current = null
      setWireDrag(null)
      setDropHint(null)
      if (!ended) {
        return
      }
      const point = { x: ev.clientX, y: ev.clientY }
      const dispatcherRect = dispatcherNodeRef.current?.getBoundingClientRect()
      const overDispatcher = dispatcherRect
        ? hitRect(point, dispatcherRect)
        : false

      if (ended.mode === "bind-item" && overDispatcher) {
        void handleConnect(ended.id)
        return
      }
      if (ended.mode === "bind-item" && !overDispatcher) {
        showErrorToast("请把线拖到左侧带 ⚙ 的当前调度器卡片上")
        return
      }
      if (ended.mode === "wire-from-dispatcher") {
        // drop onto one of the unbound terminal boxes
        for (const item of unboundItems) {
          const el = anchorRefs.current.get(`item-${item.id}`)
          if (el && hitRect(point, el.getBoundingClientRect())) {
            void handleConnect(String(item.id))
            return
          }
        }
        showErrorToast("请把线拖到右侧待接入的终端卡片上")
        return
      }
      if (ended.mode === "bind-bot") {
        // drop onto one of the bound terminal boxes
        for (const item of connectedItemsList) {
          const el = anchorRefs.current.get(`item-${item.id}`)
          if (el && hitRect(point, el.getBoundingClientRect())) {
            void bindBotToTerminal(ended.id, String(item.id))
            return
          }
        }
        // dropped onto a terminal that is not connected to this dispatcher
        for (const item of unboundItems) {
          const el = anchorRefs.current.get(`item-${item.id}`)
          if (el && hitRect(point, el.getBoundingClientRect())) {
            showErrorToast("只能连接到已接入调度器的终端")
            return
          }
        }
      }
    }
    window.addEventListener("pointermove", onMove)
    window.addEventListener("pointerup", onUp)
  }

  const invalidateBots = () => {
    queryClient.invalidateQueries({ queryKey: ["dispatcher-bot-bindings"] })
    queryClient.invalidateQueries({ queryKey: ["robot-bindings"] })
    queryClient.invalidateQueries({ queryKey: getRobotsQueryKey() })
  }

  const bindBotToTerminal = async (robotId: string, itemId: string) => {
    try {
      // 一个 bot 只保留一个主连接终端：拖到哪个终端就切到哪个
      const existing = (await listRobotBindings(String(robotId))) as any[]
      const alreadyBound = existing.some(
        (binding) => String(binding.item_id) === String(itemId),
      )
      for (const binding of existing) {
        if (String(binding.item_id) !== String(itemId)) {
          await deleteRobotBinding(String(robotId), String(binding.item_id))
        }
      }
      if (!alreadyBound) {
        await createRobotBinding(String(robotId), {
          item_id: itemId,
          allow_chat: true,
          is_default_target: true,
        })
      } else {
        await updateRobotBinding(String(robotId), String(itemId), {
          is_default_target: true,
        })
      }
      showSuccessToast("机器人主连接终端已切换")
      invalidateBots()
    } catch (error) {
      const message = error instanceof Error ? error.message : ""
      showErrorToast(message || "机器人连接失败")
    }
  }

  const unbindBotFromDispatcher = (robotId: string) => {
    const entry = boundBots.find((bot: any) => bot.robot.id === robotId)
    if (!entry) {
      return
    }
    const doUnbind = async () => {
      try {
        for (const binding of entry.bindings) {
          await deleteRobotBinding(String(robotId), String(binding.item_id))
        }
        showSuccessToast("机器人已断开")
        invalidateBots()
      } catch (_error) {
        showErrorToast("机器人断开失败")
      }
    }
    if (entry.bindings.length > 1) {
      askConfirm(
        `该机器人连接了本调度器的 ${entry.bindings.length} 个终端，全部断开？`,
        () => void doUnbind(),
      )
    } else {
      void doUnbind()
    }
  }

  // Keep the terminals in their original (stable) order — do NOT group by
  // bound status, otherwise rows jump around on connect/disconnect and the
  // wire endpoints no longer line up with the row you clicked.
  const terminalItems = useMemo(() => allItemsList, [allItemsList])
  // Assign a gray shade per daemon host so terminals on different hosts are
  // visually distinguishable.
  const hostShadeClass = useMemo(() => {
    const hosts = Array.from(
      new Set(
        terminalItems.map((item: any) =>
          String(item.socket_host || item.daemon_url || "unknown"),
        ),
      ),
    ).sort()
    const shadeOf = new Map<string, string>()
    hosts.forEach((host, index) => {
      shadeOf.set(host, `host-shade-${index % 5}`)
    })
    return (item: any) =>
      shadeOf.get(String(item.socket_host || item.daemon_url || "unknown")) ||
      "host-shade-0"
  }, [terminalItems])
  // Group terminals by daemon host so same-host items stay together.
  const terminalGroups = useMemo(() => {
    const groups = new Map<string, any[]>()
    for (const item of terminalItems) {
      const host = String(item.socket_host || item.daemon_url || "unknown")
      if (!groups.has(host)) {
        groups.set(host, [])
      }
      groups.get(host)!.push(item)
    }
    return Array.from(groups.entries()).map(([host, items]) => ({
      host,
      items,
      shade: hostShadeClass(items[0]),
    }))
  }, [terminalItems, hostShadeClass])
  const primaryTerminalId = terminalItems.some(
    (item: any) => String(item.id) === selectedTerminalId,
  )
    ? selectedTerminalId
    : String(terminalItems[0]?.id ?? "")
  const primaryTerminalTitle =
    terminalItems.find((item: any) => String(item.id) === primaryTerminalId)
      ?.title ?? ""
  const primaryTerminalStatus =
    terminalItems.find((item: any) => String(item.id) === primaryTerminalId)
      ?.status ?? ""
  const terminalFeatures = [
    {
      key: "chat",
      title: "Web Chat",
      icon: <MessageSquare className="size-4" />,
    },
    { key: "output", title: "终端输出", icon: <Terminal className="size-4" /> },
    ...(terminalWsPluginEnabled
      ? [
          {
            key: "ws",
            title: "WebSocket Server",
            icon: <Server className="size-4" />,
          },
        ]
      : []),
    ...(robotPluginEnabled
      ? [
          {
            key: "qq",
            title: "QQ 对话调试",
            icon: <MessageSquare className="size-4" />,
          },
        ]
      : []),
    { key: "files", title: "文件", icon: <FileText className="size-4" /> },
    { key: "filters", title: "过滤器", icon: <Filter className="size-4" /> },
    { key: "config", title: "配置", icon: <Settings className="size-4" /> },
    { key: "handlers", title: "调度器", icon: <Layers className="size-4" /> },
    { key: "tasks", title: "定时任务", icon: <Zap className="size-4" /> },
    { key: "token", title: "Token 统计", icon: <Brain className="size-4" /> },
  ]
  useEffect(() => {
    if (!terminalWsPluginEnabled && mainView === "ws") {
      setMainView("chat")
    }
    if (!robotPluginEnabled && mainView === "qq") {
      setMainView("chat")
    }
  }, [terminalWsPluginEnabled, robotPluginEnabled, mainView])

  const [itemAction, setItemAction] = useState<
    "start" | "stop" | "restart" | null
  >(null)
  const refreshTerminalData = useCallback(() => {
    queryClient.invalidateQueries({
      queryKey: ["itemHandler-items", itemHandlerId],
    })
    queryClient.invalidateQueries({ queryKey: ["items"] })
  }, [queryClient, itemHandlerId])
  const handleItemAction = useCallback(
    async (action: "start" | "stop" | "restart") => {
      if (!primaryTerminalId || itemAction) {
        return
      }
      setItemAction(action)
      try {
        const service =
          action === "start"
            ? ItemsService.startItem
            : action === "stop"
              ? ItemsService.stopItem
              : ItemsService.restartItem
        const result = (await service({ id: primaryTerminalId })) as any
        showSuccessToast(
          result?.message ||
            (action === "start"
              ? "终端已启动"
              : action === "stop"
                ? "终端已停止"
                : "终端已重启"),
        )
      } catch (error) {
        console.error(`Failed to ${action} item:`, error)
        showErrorToast(
          action === "start"
            ? "启动失败"
            : action === "stop"
              ? "停止失败"
              : "重启失败",
        )
      } finally {
        setItemAction(null)
        refreshTerminalData()
      }
    },
    [
      primaryTerminalId,
      itemAction,
      refreshTerminalData,
      showSuccessToast,
      showErrorToast,
    ],
  )

  const boardRef = useRef<HTMLDivElement | null>(null)
  const anchorRefs = useRef(new Map<string, Element>())
  const [wirePaths, setWirePaths] = useState<
    { key: string; d: string; kind: "item" | "bot"; id: string }[]
  >([])

  // All wires share the same dispatcher origin, so their hit areas overlap.
  // Pick the wire whose path is geometrically nearest to the click point
  // instead of whichever overlapping path happens to be on top.
  const pickWireAtPoint = useCallback(
    (clientX: number, clientY: number) => {
      const board = boardRef.current
      if (!board) {
        return null
      }
      const rect = board.getBoundingClientRect()
      const px = clientX - rect.left
      const py = clientY - rect.top
      let best: { key: string; kind: string; id: string; dist: number } | null =
        null
      for (const wire of wirePaths) {
        const pathEl = document.createElementNS(
          "http://www.w3.org/2000/svg",
          "path",
        )
        pathEl.setAttribute("d", wire.d)
        const len = pathEl.getTotalLength()
        const steps = 28
        for (let i = 0; i <= steps; i++) {
          const pt = pathEl.getPointAtLength((len * i) / steps)
          const dist = Math.hypot(pt.x - px, pt.y - py)
          if (!best || dist < best.dist) {
            best = { key: wire.key, kind: wire.kind, id: wire.id, dist }
          }
        }
      }
      return best && best.dist <= 24 ? best : null
    },
    [wirePaths],
  )

  const setAnchor = useCallback(
    (key: string) => (el: Element | null) => {
      if (el) {
        anchorRefs.current.set(key, el)
      } else {
        anchorRefs.current.delete(key)
      }
    },
    [],
  )

  const recomputeWires = useCallback(() => {
    const board = boardRef.current
    if (!board) {
      return
    }
    const boardRect = board.getBoundingClientRect()
    const anchorPoint = (key: string, side: "left" | "right" | "top") => {
      const el = anchorRefs.current.get(key)
      if (!el) {
        return null
      }
      const rect = el.getBoundingClientRect()
      const x =
        side === "left"
          ? rect.left
          : side === "right"
            ? rect.right
            : (rect.left + rect.right) / 2
      const y = side === "top" ? rect.top : (rect.top + rect.bottom) / 2
      return { x: x - boardRect.left, y: y - boardRect.top }
    }
    const horizontalCurve = (
      a: { x: number; y: number },
      b: { x: number; y: number },
    ) => {
      const midX = (a.x + b.x) / 2
      return `M ${a.x} ${a.y} C ${midX} ${a.y}, ${midX} ${b.y}, ${b.x} ${b.y}`
    }
    // Inverse-function bend: leaves the bot going straight up, then flattens
    // and plugs into the item's front face horizontally.
    const inverseCurve = (
      from: { x: number; y: number },
      to: { x: number; y: number },
    ) => {
      const rise = Math.max(70, (from.y - to.y) * 0.5)
      const approach = Math.min(120, Math.max(50, (to.x - from.x) * 0.4))
      const cp1 = { x: from.x, y: from.y - rise }
      const cp2 = { x: to.x - approach, y: to.y }
      return `M ${from.x} ${from.y} C ${cp1.x} ${cp1.y}, ${cp2.x} ${cp2.y}, ${to.x} ${to.y}`
    }
    const next: { key: string; d: string; kind: "item" | "bot"; id: string }[] =
      []
    for (const item of connectedItemsList) {
      const from = anchorPoint("dispatcher-port", "right")
      const to = anchorPoint(`item-port-${item.id}`, "left")
      if (from && to) {
        next.push({
          key: `wire-item-${item.id}`,
          d: horizontalCurve(from, to),
          kind: "item",
          id: String(item.id),
        })
      }
    }
    for (const entry of boundBots) {
      const from = anchorPoint(`bot-${entry.robot.id}`, "top")
      const targetItemId = String(entry.bindings[0]?.item_id ?? "")
      const to = targetItemId
        ? anchorPoint(`item-port-${targetItemId}`, "left")
        : null
      if (from && to) {
        next.push({
          key: `wire-bot-${entry.robot.id}`,
          d: inverseCurve(from, to),
          kind: "bot",
          id: String(entry.robot.id),
        })
      }
    }
    setWirePaths(next)
  }, [connectedItemsList, boundBots])

  useLayoutEffect(() => {
    recomputeWires()
    window.addEventListener("resize", recomputeWires)
    const observer = new ResizeObserver(() => recomputeWires())
    if (boardRef.current) {
      observer.observe(boardRef.current)
    }
    // re-measure after slide-in animations / fonts settle so wire
    // endpoints land on the real anchor positions
    const raf = requestAnimationFrame(() => recomputeWires())
    const timers = [80, 250, 450].map((ms) => setTimeout(recomputeWires, ms))
    return () => {
      window.removeEventListener("resize", recomputeWires)
      observer.disconnect()
      cancelAnimationFrame(raf)
      timers.forEach(clearTimeout)
    }
  }, [recomputeWires])

  const loadingHint = (text: string) => (
    <div className="flex items-center justify-center gap-2 py-10 text-sm text-muted-foreground">
      <Loader2 className="size-4 animate-spin" />
      {text}
    </div>
  )

  const boardCards: {
    key: string
    title: string
    icon: ReactNode
    summary: string
    content: ReactNode
  }[] = [
    {
      key: "bindings",
      title: "终端绑定",
      icon: <Terminal className="size-4" />,
      summary: `${connectedItemsList.length} 个终端`,
      content: (
        <div className="space-y-2">
          {connectedItemsList.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              还没有绑定任何终端，点击下方按钮添加。
            </p>
          ) : (
            connectedItemsList.map((item: any) => (
              <div
                key={item.id}
                className="flex items-center justify-between gap-2 rounded-lg border px-2 py-1.5 text-sm"
              >
                <span className="min-w-0 truncate">
                  {item.title || item.id}
                </span>
                <button
                  type="button"
                  onClick={() => void handleDisconnect(String(item.id))}
                  className="rounded border px-2 py-0.5 text-xs hover:bg-muted"
                >
                  解绑
                </button>
              </div>
            ))
          )}
          <AddItemToHandler itemHandlerId={itemHandlerId} />
        </div>
      ),
    },
    {
      key: "skills",
      title: "技能",
      icon: <Zap className="size-4" />,
      summary: `${enabledSkills.length} 项启用`,
      content: skillsLoading ? (
        loadingHint("正在加载技能...")
      ) : (
        <SkillSelector
          allSkills={skills}
          enabledSkills={enabledSkills}
          onSkillToggle={handleSkillToggle}
        />
      ),
    },
    {
      key: "mcp",
      title: "MCP",
      icon: <Server className="size-4" />,
      summary: `${enabledMcpServers.length} 个服务`,
      content: mcpLoading ? (
        loadingHint("正在加载 MCP...")
      ) : (
        <McpSelector
          allServers={mcpServers}
          enabledServers={enabledMcpServers}
          onServerToggle={handleMcpServerToggle}
        />
      ),
    },
    {
      key: "knowledge",
      title: "知识库",
      icon: <FileText className="size-4" />,
      summary: `${enabledKnowledgeFiles.length} 个文件`,
      content: (
        <KnowledgeBindingSelector
          itemHandlerId={itemHandlerId}
          enabledKnowledgeFiles={enabledKnowledgeFiles}
          onKnowledgeToggle={handleKnowledgeToggle}
        />
      ),
    },
    {
      key: "memory",
      title: "长期记忆",
      icon: <Brain className="size-4" />,
      summary: "共享记忆池",
      content:
        connectedItemsList.length === 0 ? (
          <p className="py-6 text-center text-sm text-muted-foreground">
            还没有绑定任何终端，绑定后即可管理共享记忆。
          </p>
        ) : (
          <TerminalMemoryPanel connectedItems={connectedItemsList} />
        ),
    },
    {
      key: "chatLogs",
      title: "聊天记录",
      icon: <MessageSquare className="size-4" />,
      summary: `${boundBots.length} 个机器人`,
      content: (
        <ChatLogPanel
          itemHandlerId={itemHandlerId}
          connectedItems={connectedItemsList}
        />
      ),
    },
    {
      key: "config",
      title: "模型配置",
      icon: <Settings className="size-4" />,
      summary: String(itemHandler?.model || ""),
      content: itemHandler ? (
        <HandlerSettingsPanel itemHandler={itemHandler} />
      ) : (
        <p className="py-6 text-center text-sm text-muted-foreground">
          先在左侧创建一个调度器。
        </p>
      ),
    },
  ]
  const expandedCardData =
    boardCards.find((card) => card.key === expandedCard) ?? null

  return (
    <div
      ref={boardRef}
      className="dispatcher-board relative flex min-h-0 w-full flex-1 flex-col"
    >
      <style>{DISPATCHER_BOARD_CSS}</style>
      <svg className="board-wires" aria-hidden>
        {wirePaths.map((wire) => (
          <g key={wire.key}>
            <path
              d={wire.d}
              className={`wire-visible ${wire.kind === "bot" ? "wire-bot" : ""}`}
            />
            <path
              d={wire.d}
              className="wire-hit"
              onClick={(event) => {
                const picked = pickWireAtPoint(event.clientX, event.clientY)
                if (!picked) {
                  return
                }
                if (picked.kind === "item") {
                  const title =
                    terminalItems.find((it: any) => String(it.id) === picked.id)
                      ?.title || picked.id
                  askConfirm(`断开终端「${title}」与调度器的连接？`, () =>
                    handleDisconnect(picked.id),
                  )
                } else {
                  const robot = boundBots.find(
                    (b: any) => String(b.robot.id) === picked.id,
                  )?.robot
                  askConfirm(
                    `断开机器人「${robot?.name || picked.id}」的连接？`,
                    () => unbindBotFromDispatcher(picked.id),
                  )
                }
              }}
            />
          </g>
        ))}
        {wireDrag ? (
          <path
            className="wire-preview"
            d={`M ${wireDrag.from.x} ${wireDrag.from.y} L ${wireDrag.to.x} ${wireDrag.to.y}`}
          />
        ) : null}
      </svg>

      <div className="board-grid relative z-30 grid min-h-0 flex-1 gap-4 lg:grid-cols-[230px_minmax(0,1fr)_230px]">
        <aside className="terminal-edge-panel relative z-30 flex min-h-0 flex-col self-stretch py-3 pl-0 pr-1">
          <div className="edge-add-wrap">
            <span className="edge-add-frame edge-add-frame-left">
              <AddItemHandler
                triggerVariant="outline"
                triggerClassName="btn-add-compact"
              />
            </span>
          </div>
          <ScrollColumn>
            <div className="space-y-2">
              {allHandlers.length === 0 ? (
                <div className="sketch-box sketch-b px-3 py-4 text-center text-xs">
                  还没有调度器
                </div>
              ) : (
                allHandlers.map((handler: any, index: number) => {
                  const isActive = String(handler.id) === String(itemHandlerId)
                  const boxClass = `sketch-box edge-left-box px-3 py-2.5 text-left text-sm w-full ${
                    index % 2 ? "sketch-b" : "sketch-c"
                  } ${isActive ? "sketch-active" : "sketch-hover"}`
                  if (isActive) {
                    return (
                      <div
                        key={handler.id}
                        ref={(el) => {
                          dispatcherNodeRef.current = el
                          setAnchor("dispatcher")(el)
                        }}
                        title="当前调度器"
                        className={`${boxClass} relative ${
                          dropHint === "dispatcher" ? "sketch-drop-target" : ""
                        }`}
                      >
                        <div className="dispatcher-box-body">
                          <div className="font-black">⚙ {handler.name}</div>
                          <div className="text-[10px] text-muted-foreground">
                            dispatcher · {handler.model || "未设置模型"}
                          </div>
                        </div>
                        <div
                          ref={setAnchor("dispatcher-bot") as any}
                          className="mx-auto h-0 w-0"
                        />
                        <span
                          className="item-del"
                          title="删除该调度器"
                          onClick={(event) => {
                            event.stopPropagation()
                            deleteDispatcherById(handler)
                          }}
                        >
                          <Trash2 className="size-3" />
                        </span>
                        <span
                          ref={setAnchor("dispatcher-port") as any}
                          className="wire-handle"
                          title="按住拉线到右侧待接入终端进行连接"
                          onPointerDown={(event) => {
                            event.stopPropagation()
                            startWireDrag(
                              event,
                              "wire-from-dispatcher",
                              "",
                              "dispatcher-port",
                            )
                          }}
                        />
                      </div>
                    )
                  }
                  return (
                    <button
                      key={handler.id}
                      type="button"
                      onClick={() =>
                        void navigate({
                          to: "/item-handlers/$itemHandlerId",
                          params: { itemHandlerId: String(handler.id) },
                        })
                      }
                      title="切换到这个调度器"
                      className={`${boxClass} relative`}
                    >
                      <div className="dispatcher-box-body">
                        <div className="font-bold">⚙ {handler.name}</div>
                        <div className="text-[10px] text-muted-foreground">
                          {handler.model || "未设置模型"}
                        </div>
                      </div>
                      <span
                        className="item-del"
                        title="删除该调度器"
                        onClick={(event) => {
                          event.stopPropagation()
                          deleteDispatcherById(handler)
                        }}
                      >
                        <Trash2 className="size-3" />
                      </span>
                    </button>
                  )
                })
              )}
            </div>
          </ScrollColumn>
        </aside>

        <section className="relative z-30 flex min-h-0 flex-col justify-center pb-10 pt-28">
          <ScrollColumn className="scroll-col-pad" arrows={false}>
            <div className="mx-auto grid w-full max-w-md auto-rows-[64px] grid-cols-4 grid-flow-dense gap-2">
              {boardCards.map((card) => (
                <button
                  key={card.key}
                  type="button"
                  onClick={() => setExpandedCard(card.key)}
                  className="glass-box sketch-hover board-card col-span-2 flex flex-col justify-center px-3 py-2 text-left"
                >
                  <div className="flex items-center gap-1.5 text-[13px] font-semibold">
                    {card.icon}
                    {card.title}
                  </div>
                  <div className="mt-1 truncate text-[11px] text-muted-foreground">
                    {card.summary}
                  </div>
                </button>
              ))}
            </div>
          </ScrollColumn>
        </section>

        <section className="terminal-edge-panel-right relative z-30 flex min-h-0 flex-col space-y-2 pr-0">
          <div className="edge-add-wrap">
            <span className="edge-add-frame edge-add-frame-right">
              <AddItem
                items={terminalItems}
                triggerVariant="outline"
                triggerClassName="btn-add-compact"
              />
            </span>
          </div>
          <ScrollColumn>
            <div className="space-y-3 pr-0">
              {terminalGroups.length === 0 ? (
                <div className="sketch-box sketch-b px-3 py-4 text-center text-xs">
                  还没有终端，先去项目页创建
                </div>
              ) : (
                terminalGroups.map((group) => (
                  <div
                    key={group.host}
                    className={`space-y-1 rounded-lg p-1.5 ${group.shade}`}
                  >
                    <div
                      className="truncate px-1 text-[10px] font-bold text-muted-foreground"
                      title={group.host}
                    >
                      {group.host}
                    </div>
                    {group.items.map((item: any, index: number) => {
                      const bound = connectedItemIds.has(String(item.id))
                      return (
                        <button
                          key={item.id}
                          ref={setAnchor(`item-${item.id}`) as any}
                          type="button"
                          onClick={() => {
                            setSelectedTerminalId(String(item.id))
                            setTerminalPanelOpen(true)
                          }}
                          title={
                            bound
                              ? "点击打开交互终端 · 点连线断开"
                              : "点击打开详情 · 拉圆圈到调度器接入"
                          }
                          className={`relative block w-full px-3 py-2 text-left text-sm edge-right-box ${
                            item.status === "running"
                              ? "terminal-running"
                              : "terminal-stopped"
                          } ${
                            bound
                              ? `sketch-box sketch-hover ${
                                  index % 2 ? "sketch-b" : "sketch-c"
                                } ${
                                  String(item.id) === primaryTerminalId
                                    ? "sketch-active"
                                    : ""
                                }`
                              : "sketch-tray sketch-hover cursor-crosshair border-dashed"
                          }`}
                        >
                          <div className="font-bold">
                            {item.title || item.id}
                          </div>
                          <div className="text-[10px] text-muted-foreground">
                            {bound ? "已接入" : "待接入"}
                          </div>
                          <span
                            className="item-del"
                            title={bound ? "断开该终端" : "删除该终端"}
                            onClick={(event) => {
                              event.stopPropagation()
                              if (bound) {
                                askConfirm(
                                  `断开终端「${item.title || item.id}」与调度器的连接？`,
                                  () => void handleDisconnect(String(item.id)),
                                )
                              }
                            }}
                          >
                            <Trash2 className="size-3" />
                          </span>
                          <span
                            ref={setAnchor(`item-port-${item.id}`) as any}
                            className="wire-handle"
                            style={{ left: "-7px", right: "auto" }}
                            title={bound ? "连接点" : "按住拉线到调度器接入"}
                            onPointerDown={
                              bound
                                ? undefined
                                : (event) => {
                                    event.stopPropagation()
                                    startWireDrag(
                                      event,
                                      "bind-item",
                                      String(item.id),
                                      `item-port-${item.id}`,
                                    )
                                  }
                            }
                          />
                        </button>
                      )
                    })}
                  </div>
                ))
              )}
            </div>
          </ScrollColumn>
        </section>
      </div>

      {robotPluginEnabled ? (
        <footer className="relative z-10 mt-4 flex flex-shrink-0 items-end justify-center gap-4 px-4 pb-6 pt-3">
          <span className="bot-tag bot-tag-tool">
            <CreateRobotDialog
              platforms={robotPlatforms}
              iconOnly
              triggerClassName="btn-add-compact"
            />
          </span>
          {boundBots.length === 0 && unboundBots.length === 0 ? (
            <div className="sketch-box sketch-c inline-block px-3 py-2 text-xs">
              还没有机器人
            </div>
          ) : null}
          {boundBots.map((entry: any, index: number) => (
            <div
              key={entry.robot.id}
              ref={setAnchor(`bot-${entry.robot.id}`) as any}
              title="点击查看机器人详情"
              className="bot-tag"
              onClick={() => setBotDetailId(String(entry.robot.id))}
              style={{ "--tilt": index % 2 ? "1.6deg" : "-1.8deg" } as any}
            >
              <span
                className="item-del"
                title="删除该机器人"
                onClick={(event) => {
                  event.stopPropagation()
                  deleteRobotById(entry.robot)
                }}
              >
                <Trash2 className="size-3" />
              </span>
              <span
                className="bot-diagnose-tab"
                title="诊断机器人链路"
                onClick={(event) => {
                  event.stopPropagation()
                  void runDiagnose()
                }}
              >
                <Stethoscope className="size-3.5" />
              </span>
              <div className="flex items-center gap-1.5 font-bold">
                <Bot className="size-4" />
                {entry.robot.name}
              </div>
              <div className="text-[10px]">
                {entry.robot.platform} →{" "}
                {entry.bindings
                  .map(
                    (binding: any) => binding.route_key || binding.item_title,
                  )
                  .join(", ")}
              </div>
            </div>
          ))}
          {unboundBots.map((entry: any, index: number) => (
            <div
              key={entry.robot.id}
              ref={setAnchor(`bot-${entry.robot.id}`) as any}
              title="点击查看机器人详情"
              className="bot-tag relative border-dashed"
              onClick={() => setBotDetailId(String(entry.robot.id))}
              style={{ "--tilt": index % 2 ? "-1.4deg" : "1.7deg" } as any}
            >
              <span
                className="item-del"
                title="删除该机器人"
                onClick={(event) => {
                  event.stopPropagation()
                  deleteRobotById(entry.robot)
                }}
              >
                <Trash2 className="size-3" />
              </span>
              <span
                className="bot-diagnose-tab"
                title="诊断机器人链路"
                onClick={(event) => {
                  event.stopPropagation()
                  void runDiagnose()
                }}
              >
                <Stethoscope className="size-3.5" />
              </span>
              <span className="flex items-center gap-1.5 font-bold">
                <Bot className="size-4" />
                {entry.robot.name}
              </span>
              <span className="text-[10px]">{entry.robot.platform}</span>
              <span
                className="wire-handle"
                style={{
                  top: "-7px",
                  left: "50%",
                  marginLeft: "-6px",
                  right: "auto",
                  marginTop: "0",
                }}
                onPointerDown={(event) => {
                  event.stopPropagation()
                  startWireDrag(
                    event,
                    "bind-bot",
                    String(entry.robot.id),
                    `bot-${entry.robot.id}`,
                  )
                }}
              />
            </div>
          ))}
        </footer>
      ) : null}

      {diagnoseOpen ? (
        <div className="zoom-backdrop" onClick={() => setDiagnoseOpen(false)}>
          <div
            className="zoom-card sketch-box sketch-a"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="sticky top-0 z-10 flex items-center justify-between border-b bg-white/95 px-4 py-2 backdrop-blur">
              <div className="flex items-center gap-2 text-sm font-bold">
                <Stethoscope className="size-4" />
                机器人链路诊断
              </div>
              <button
                type="button"
                onClick={() => setDiagnoseOpen(false)}
                className="rounded p-1 hover:bg-muted"
                aria-label="关闭"
              >
                <X className="size-4" />
              </button>
            </div>
            <div className="space-y-3 p-4 text-sm">
              {isDiagnosing ? (
                <div className="flex items-center gap-2 py-6 text-muted-foreground">
                  <Loader2 className="size-4 animate-spin" /> 正在诊断…
                </div>
              ) : diagnoseResults.length === 0 ? (
                <div className="py-6 text-center text-muted-foreground">
                  没有可诊断的机器人
                </div>
              ) : (
                diagnoseResults.map((result: any, index: number) => (
                  <div key={index} className="rounded-lg border p-3">
                    <div className="flex items-center justify-between">
                      <span className="font-bold">
                        {result.robot_name || result.robot_id}
                      </span>
                      <span
                        className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                          result.overall_status === "ok"
                            ? "bg-emerald-100 text-emerald-700"
                            : "bg-amber-100 text-amber-700"
                        }`}
                      >
                        {result.overall_status === "ok" ? "正常" : "异常"}
                      </span>
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {result.platform}
                      {result.chain
                        ? ` · 配置:${result.chain.robot_config?.status ?? "-"} · 桥接:${result.chain.qq_to_bridge?.status ?? "-"} · 后端:${result.chain.bridge_to_backend?.status ?? "-"}`
                        : ""}
                      {result.error ? ` · ${result.error}` : ""}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      ) : null}

      {confirmState ? (
        <div className="confirm-backdrop" onClick={() => setConfirmState(null)}>
          <div
            className="confirm-card sketch-box sketch-a"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="text-sm font-bold">{confirmState.message}</div>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                className="sketch-mini-box px-3 py-1"
                onClick={() => setConfirmState(null)}
              >
                取消
              </button>
              <button
                type="button"
                className="sketch-mini-box px-3 py-1 hover:bg-red-50"
                onClick={() => {
                  const action = confirmState.action
                  setConfirmState(null)
                  action()
                }}
              >
                确认断开
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {botDetailId
        ? createPortal(
            <div className="bot-detail-panel">
              <div className="flex items-center gap-2 border-b-2 border-[#3a3a3a] bg-white px-3 py-2">
                <button
                  type="button"
                  onClick={() => setBotDetailId(null)}
                  className="sketch-box sketch-c flex items-center gap-1 px-2 py-1 text-xs font-bold hover:bg-stone-100"
                >
                  ← 返回
                </button>
                <div className="flex items-center gap-2 text-sm font-black">
                  <Bot className="size-4" />
                  机器人详情
                </div>
              </div>
              <div className="hang-page min-h-0 flex-1 overflow-y-auto">
                <RobotDetail robotId={botDetailId} />
              </div>
            </div>,
            document.body,
          )
        : null}

      {terminalPanelOpen
        ? createPortal(
            <div className="terminal-fullscreen">
              <div className="terminal-fullscreen-header">
                <button
                  type="button"
                  onClick={() => {
                    setTerminalPanelOpen(false)
                  }}
                  className="sketch-box sketch-c flex items-center gap-1 px-2 py-1 text-xs font-bold hover:bg-stone-100"
                >
                  ← 返回
                </button>
                <div className="flex items-center gap-2 text-sm font-black">
                  <Terminal className="size-4" />
                  主交互终端
                </div>
                <span className="text-xs text-muted-foreground">
                  {primaryTerminalTitle || "未绑定终端"}
                </span>
                {primaryTerminalStatus ? (
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                      primaryTerminalStatus === "running"
                        ? "bg-emerald-100 text-emerald-700"
                        : primaryTerminalStatus === "error"
                          ? "bg-red-100 text-red-700"
                          : "bg-stone-200 text-stone-600"
                    }`}
                  >
                    {primaryTerminalStatus === "running"
                      ? "运行中"
                      : primaryTerminalStatus === "starting"
                        ? "启动中"
                        : primaryTerminalStatus === "stopping"
                          ? "停止中"
                          : primaryTerminalStatus === "stopped"
                            ? "已停止"
                            : primaryTerminalStatus === "error"
                              ? "错误"
                              : primaryTerminalStatus}
                  </span>
                ) : null}
                {primaryTerminalId ? (
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      disabled={itemAction !== null}
                      onClick={() => void handleItemAction("start")}
                      className="sketch-box sketch-c flex items-center gap-1 px-2 py-1 text-xs font-bold hover:bg-emerald-50 disabled:opacity-50"
                    >
                      {itemAction === "start" ? (
                        <Loader2 className="size-3 animate-spin" />
                      ) : null}
                      启动
                    </button>
                    <button
                      type="button"
                      disabled={itemAction !== null}
                      onClick={() => void handleItemAction("stop")}
                      className="sketch-box sketch-c flex items-center gap-1 px-2 py-1 text-xs font-bold hover:bg-amber-50 disabled:opacity-50"
                    >
                      {itemAction === "stop" ? (
                        <Loader2 className="size-3 animate-spin" />
                      ) : null}
                      停止
                    </button>
                    <button
                      type="button"
                      disabled={itemAction !== null}
                      onClick={() => void handleItemAction("restart")}
                      className="sketch-box sketch-c flex items-center gap-1 px-2 py-1 text-xs font-bold hover:bg-sky-50 disabled:opacity-50"
                    >
                      {itemAction === "restart" ? (
                        <Loader2 className="size-3 animate-spin" />
                      ) : null}
                      重启
                    </button>
                  </div>
                ) : null}
                <div className="flex-1" />
                <button
                  type="button"
                  onClick={() => setPanelSettingsOpen((v) => !v)}
                  className="text-xs font-bold hover:underline"
                >
                  {panelSettingsOpen ? "▾" : "▸"} 终端功能设置
                </button>
              </div>
              {panelSettingsOpen ? (
                <div className="border-b-2 border-[#3a3a3a] bg-white px-3 py-2">
                  <div className="sketch-box sketch-b mt-1 space-y-2 p-3 text-sm">
                    <div className="font-bold">
                      当前终端：{primaryTerminalTitle || "未绑定"}
                      <span className="ml-1 text-[10px] font-normal">
                        {primaryTerminalId
                          ? `#${primaryTerminalId.slice(0, 8)}`
                          : ""}
                      </span>
                    </div>
                    {connectedItemsList.length > 1 ? (
                      <label className="flex items-center gap-2 text-xs">
                        切换主终端：
                        <select
                          className="rounded border border-[#3a3a3a] bg-white px-2 py-1"
                          value={primaryTerminalId}
                          onChange={(event) =>
                            setSelectedTerminalId(event.target.value)
                          }
                        >
                          {connectedItemsList.map((item: any) => (
                            <option key={item.id} value={String(item.id)}>
                              {item.title || item.id}
                            </option>
                          ))}
                        </select>
                      </label>
                    ) : null}
                    <div className="flex flex-wrap gap-2 text-xs font-bold">
                      {primaryTerminalId ? (
                        <button
                          type="button"
                          className="sketch-box sketch-c px-2 py-1 hover:bg-red-50"
                          onClick={() => {
                            askConfirm(
                              `从调度器断开终端「${primaryTerminalTitle}」？`,
                              () => {
                                setTerminalPanelOpen(false)
                                void handleDisconnect(primaryTerminalId)
                              },
                            )
                          }}
                        >
                          ✂ 解绑此终端
                        </button>
                      ) : (
                        <span className="font-normal">
                          先在左侧接入一个终端
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ) : null}
              <div className="terminal-fullscreen-body terminal-body-3col">
                <div className="terminal-frame">
                  <div className="terminal-titlebar">
                    <span className="dot" />
                    <span className="dot" />
                    <span className="dot" />
                    <span className="ml-2 text-[11px]">
                      {primaryTerminalTitle || "未绑定终端"}
                    </span>
                    <span className="ml-auto pr-1 text-[10px] font-bold text-stone-400">
                      终端输出
                    </span>
                  </div>
                  <div className="terminal-body" style={{ minHeight: 0 }}>
                    {primaryTerminalId ? (
                      <TerminalOutputPanel
                        key={primaryTerminalId}
                        itemId={primaryTerminalId}
                        running={primaryTerminalStatus === "running"}
                      />
                    ) : (
                      <div className="p-6 text-center text-xs">
                        先在左侧接入一个终端
                      </div>
                    )}
                  </div>
                </div>
                <div className="terminal-frame">
                  <div className="terminal-titlebar terminal-titlebar-light">
                    <span className="text-[11px] font-bold">
                      {terminalFeatures.find((f) => f.key === mainView)
                        ?.title ?? ""}
                    </span>
                  </div>
                  <div className="terminal-body" style={{ minHeight: 0 }}>
                    {primaryTerminalId ? (
                      mainView === "chat" ? (
                        <ChatPanel itemId={primaryTerminalId} />
                      ) : (
                        <div
                          className={
                            mainView === "files"
                              ? "h-full min-h-0 bg-white/70 p-3 backdrop-blur-sm"
                              : "h-full min-h-0 overflow-y-auto bg-white/70 p-3 backdrop-blur-sm"
                          }
                        >
                          {mainView === "ws" ? (
                            <TerminalWsPanel
                              itemId={primaryTerminalId}
                              itemTitle={primaryTerminalTitle || "终端"}
                            />
                          ) : mainView === "qq" ? (
                            <RobotConversationDebugPanel
                              itemId={primaryTerminalId}
                            />
                          ) : mainView === "files" ? (
                            <ItemFilesPanel itemId={primaryTerminalId} />
                          ) : mainView === "filters" ? (
                            <ItemFiltersPanel itemId={primaryTerminalId} />
                          ) : mainView === "config" ? (
                            <ItemConfigPanel itemId={primaryTerminalId} />
                          ) : mainView === "handlers" ? (
                            <ItemHandlersList itemId={primaryTerminalId} />
                          ) : mainView === "tasks" ? (
                            <ScheduledTasksManager itemId={primaryTerminalId} />
                          ) : mainView === "token" ? (
                            <TokenUsagePanel itemId={primaryTerminalId} />
                          ) : null}
                        </div>
                      )
                    ) : (
                      <div className="p-6 text-center text-xs">
                        先在左侧接入一个终端
                      </div>
                    )}
                  </div>
                </div>
                <div className="feature-icon-rail">
                  {terminalFeatures
                    .filter((feature) => feature.key !== "output")
                    .map((feature) => (
                      <button
                        key={feature.key}
                        type="button"
                        onClick={() => setMainView(feature.key)}
                        title={feature.title}
                        className={`feature-icon-btn ${
                          mainView === feature.key ? "feature-icon-active" : ""
                        }`}
                      >
                        {feature.icon}
                        <span className="feature-icon-label">
                          {feature.title}
                        </span>
                      </button>
                    ))}
                </div>
              </div>
            </div>,
            document.body,
          )
        : null}

      {expandedCardData ? (
        <div className="zoom-backdrop" onClick={() => setExpandedCard(null)}>
          <div
            className="zoom-card sketch-box sketch-a"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="sticky top-0 z-10 flex items-center justify-between border-b bg-white/95 px-4 py-2 backdrop-blur">
              <div className="flex items-center gap-2 text-sm font-bold">
                {expandedCardData.icon}
                {expandedCardData.title}
              </div>
              <button
                type="button"
                onClick={() => setExpandedCard(null)}
                className="rounded p-1 hover:bg-muted"
                aria-label="关闭"
              >
                <X className="size-4" />
              </button>
            </div>
            <div className="p-4 sketch-content">{expandedCardData.content}</div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

function TerminalDispatcherPage() {
  const { itemHandlerId } = Route.useParams()
  const { data: itemHandler, isLoading } = useQuery({
    queryFn: () => ItemHandlersService.readItemHandler({ id: itemHandlerId }),
    queryKey: ["itemHandler", itemHandlerId],
  })

  if (isLoading || !itemHandler) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="size-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  return (
    <div
      className="full-bleed flex flex-col overflow-hidden text-zinc-900"
      style={{
        height: "calc(100svh - 79px)",
      }}
    >
      <TerminalDispatcherBoard
        key={itemHandlerId}
        itemHandlerId={itemHandlerId}
        itemHandler={itemHandler}
      />
    </div>
  )
}
export default TerminalDispatcherPage
