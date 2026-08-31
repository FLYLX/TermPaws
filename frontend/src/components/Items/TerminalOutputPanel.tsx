import { Loader2, Send } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import {
  type TerminalOutput,
  useTerminalConnection,
} from "@/hooks/useTerminalConnection"

function getEntryText(entry: TerminalOutput): string {
  if (typeof entry.stdout === "string") return entry.stdout
  if (typeof entry.stderr === "string") return entry.stderr
  if (typeof entry.stdin === "string") return entry.stdin
  return ""
}

export function TerminalOutputPanel({
  itemId,
  running = true,
}: {
  itemId: string
  running?: boolean
}) {
  const [command, setCommand] = useState("")
  const outputRef = useRef<HTMLDivElement | null>(null)
  const stickToBottomRef = useRef(true)
  const { isConnected, isConnecting, error, output, sendCommand, reconnect } =
    useTerminalConnection({ itemId, enabled: running })

  useEffect(() => {
    const el = outputRef.current
    if (el && stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight
    }
  }, [])

  const handleSend = () => {
    const trimmed = command.trim()
    if (!trimmed || !isConnected) return
    sendCommand(trimmed)
    setCommand("")
  }

  if (!running) {
    return (
      <div className="flex h-full min-h-0 flex-col items-center justify-center gap-2 bg-[#121212]">
        <span className="text-xs text-slate-400">
          终端未运行，点顶部「启动」后再查看输出
        </span>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#121212]">
      <div
        ref={outputRef}
        onScroll={(event) => {
          const el = event.currentTarget
          stickToBottomRef.current =
            el.scrollHeight - el.scrollTop - el.clientHeight < 64
        }}
        className="min-h-0 flex-1 overflow-auto px-4 py-3 font-mono text-[13px] leading-[1.45]"
      >
        {isConnecting ? (
          <div className="flex h-full flex-col items-center justify-center gap-3">
            <Loader2 className="size-6 animate-spin text-blue-400" />
            <span className="text-xs text-slate-400">正在连接终端...</span>
          </div>
        ) : error ? (
          <div className="flex h-full flex-col items-center justify-center gap-3">
            <span className="text-xs text-red-300">连接失败：{error}</span>
            <button
              type="button"
              onClick={() => reconnect(true)}
              className="rounded border border-zinc-700 px-3 py-1 text-xs text-slate-200 hover:bg-zinc-800"
            >
              重连
            </button>
          </div>
        ) : output.length === 0 ? (
          <div className="text-lime-400">终端已连接，等待输出...</div>
        ) : (
          output.map((entry, index) => {
            const text = getEntryText(entry)
            if (!text) return null
            const isError = Boolean(entry.stderr)
            const isInput = Boolean(entry.stdin)
            return (
              <span
                key={`term-${index}`}
                className={`whitespace-pre-wrap break-all ${
                  isError
                    ? "text-red-300"
                    : isInput
                      ? "text-green-400"
                      : "text-blue-300"
                }`}
              >
                {text}
              </span>
            )
          })
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2 border-t border-zinc-800 bg-zinc-950 px-3 py-2">
        <input
          value={command}
          onChange={(event) => setCommand(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault()
              handleSend()
            }
          }}
          placeholder={
            isConnected ? "输入命令，回车发送..." : "终端未连接，无法发送命令"
          }
          disabled={!isConnected}
          className="code-editor h-8 flex-1 rounded border border-zinc-700 bg-zinc-900/80 px-3 font-mono text-xs text-slate-100 outline-none placeholder:text-slate-500 disabled:opacity-50"
        />
        <button
          type="button"
          onClick={handleSend}
          disabled={!isConnected || !command.trim()}
          className="flex h-8 w-8 items-center justify-center rounded border border-zinc-700 text-slate-200 hover:bg-zinc-800 disabled:opacity-40"
          title="发送命令"
        >
          <Send className="size-3.5" />
        </button>
      </div>
    </div>
  )
}
