import { useCallback, useEffect, useRef, useState } from "react"
import { io, type Socket } from "socket.io-client"
import { ItemsService } from "@/client"

export interface TerminalOutput {
  stdout?: string
  stderr?: string
  stdin?: string
  source?: string
  status?: string
  job_id?: string
  exit_code?: number
  timed_out?: boolean
  cancelled?: boolean
  exit?: boolean
}

export interface TerminalCompletion {
  value: string
  cursor: number
  candidates: string[]
}

interface RoomInfo {
  permanent_count: number
  temporary_count: number
}

interface TerminalConnectedData {
  item_uuid: string
  subscriber_type: string
  room_info: RoomInfo
  user_uuid?: string
}

interface UseTerminalConnectionOptions {
  itemId: string
  enabled?: boolean
  onConnected?: (data: TerminalConnectedData) => void
  onDisconnected?: () => void
  onError?: (error: string) => void
}

interface UseTerminalConnectionReturn {
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  output: TerminalOutput[]
  clearOutput: () => void
  sendCommand: (command: string) => void
  completeCommand: (
    command: string,
    cursor: number,
  ) => Promise<TerminalCompletion | null>
  sendCtrlC: () => void
  reconnect: (force?: boolean) => void
  disconnect: () => void
}

const ESC = String.fromCharCode(27)
const BEL = String.fromCharCode(7)
const ANSI_ESCAPE_RE = new RegExp(
  `${ESC}(?:\\[[0-?]*[ -/]*[@-~]|\\][^${BEL}]*(?:${BEL}|${ESC}\\\\)|[PX^_].*?${ESC}\\\\|[@-Z\\\\-_=>])`,
  "g",
)
const PROMPT_BEFORE_TIMESTAMP_RE = /^(?:\s*[>#]\s*)+(?=\[\d{2}:\d{2}:\d{2}\])/gm
const TERMINAL_DEBUG = import.meta.env.DEV
const TERMINAL_OUTPUT_MAX_ENTRIES = 1200
const TERMINAL_OUTPUT_MAX_CHARS = 240_000
const TERMINAL_PENDING_OUTPUT_MAX_ENTRIES = 4000
const TERMINAL_STREAM_FLUSH_MS = 160

function stripTerminalControlChars(text: string): string {
  let result = ""
  for (const char of text) {
    const code = char.charCodeAt(0)
    if (
      code <= 7 ||
      code === 11 ||
      code === 12 ||
      (code >= 14 && code <= 31) ||
      code === 127
    ) {
      continue
    }
    result += char
  }
  return result
}

function sanitizeTerminalText(value: unknown): string {
  let text = String(value ?? "")
  if (!text) {
    return ""
  }
  if (text.length > TERMINAL_OUTPUT_MAX_CHARS) {
    text = text.slice(-TERMINAL_OUTPUT_MAX_CHARS)
  }
  return stripTerminalControlChars(text.replace(ANSI_ESCAPE_RE, "")).replace(
    PROMPT_BEFORE_TIMESTAMP_RE,
    "",
  )
}

function sanitizeTerminalOutput(entry: TerminalOutput): TerminalOutput {
  return {
    ...entry,
    stdout:
      typeof entry.stdout === "string"
        ? sanitizeTerminalText(entry.stdout)
        : entry.stdout,
    stderr:
      typeof entry.stderr === "string"
        ? sanitizeTerminalText(entry.stderr)
        : entry.stderr,
    stdin:
      typeof entry.stdin === "string"
        ? sanitizeTerminalText(entry.stdin)
        : entry.stdin,
  }
}

function getOutputText(entry: TerminalOutput): {
  key: "stdout" | "stderr" | "stdin"
  text: string
} | null {
  if (typeof entry.stdout === "string") {
    return { key: "stdout", text: entry.stdout }
  }
  if (typeof entry.stderr === "string") {
    return { key: "stderr", text: entry.stderr }
  }
  if (typeof entry.stdin === "string") {
    return { key: "stdin", text: entry.stdin }
  }
  return null
}

function replaceOutputText(
  entry: TerminalOutput,
  key: "stdout" | "stderr" | "stdin",
  text: string,
): TerminalOutput {
  return { ...entry, [key]: text }
}

function trimCurrentTerminalLine(entries: TerminalOutput[]): void {
  while (entries.length > 0) {
    const lastIndex = entries.length - 1
    const current = getOutputText(entries[lastIndex])
    if (!current) {
      entries.pop()
      continue
    }

    const newlineIndex = current.text.lastIndexOf("\n")
    if (newlineIndex >= 0) {
      const kept = current.text.slice(0, newlineIndex + 1)
      if (kept) {
        entries[lastIndex] = replaceOutputText(
          entries[lastIndex],
          current.key,
          kept,
        )
      } else {
        entries.pop()
      }
      return
    }

    entries.pop()
  }
}

function removeLastTerminalChar(entries: TerminalOutput[]): void {
  while (entries.length > 0) {
    const lastIndex = entries.length - 1
    const current = getOutputText(entries[lastIndex])
    if (!current) {
      entries.pop()
      continue
    }

    if (current.text.length <= 1) {
      entries.pop()
      return
    }

    entries[lastIndex] = replaceOutputText(
      entries[lastIndex],
      current.key,
      current.text.slice(0, -1),
    )
    return
  }
}

function appendTerminalText(
  entries: TerminalOutput[],
  key: "stdout" | "stderr" | "stdin",
  text: string,
  metadata: TerminalOutput,
): void {
  let buffer = ""

  const flush = () => {
    if (!buffer) {
      return
    }
    entries.push({ ...metadata, [key]: buffer })
    buffer = ""
  }

  for (const char of text) {
    if (char === "\r") {
      flush()
      trimCurrentTerminalLine(entries)
      continue
    }

    if (char === "\b") {
      flush()
      removeLastTerminalChar(entries)
      continue
    }

    buffer += char
  }

  flush()
}

function appendTerminalOutput(
  previous: TerminalOutput[],
  incoming: TerminalOutput,
): TerminalOutput[] {
  const sanitizedIncoming = sanitizeTerminalOutput(incoming)
  const textFields = [
    ["stdin", sanitizedIncoming.stdin],
    ["stdout", sanitizedIncoming.stdout],
    ["stderr", sanitizedIncoming.stderr],
  ] as const
  const hasControl = textFields.some(
    ([, value]) =>
      typeof value === "string" &&
      (value.includes("\r") || value.includes("\b")),
  )

  if (!hasControl) {
    return [...previous, sanitizedIncoming]
  }

  const next = [...previous]
  const metadata = {
    source: sanitizedIncoming.source,
    status: sanitizedIncoming.status,
    job_id: sanitizedIncoming.job_id,
    exit_code: sanitizedIncoming.exit_code,
    timed_out: sanitizedIncoming.timed_out,
    cancelled: sanitizedIncoming.cancelled,
  }

  for (const [key, value] of textFields) {
    if (typeof value === "string" && value) {
      appendTerminalText(next, key, value, metadata)
    }
  }

  return next
}

function limitTerminalOutput(entries: TerminalOutput[]): TerminalOutput[] {
  if (entries.length <= TERMINAL_OUTPUT_MAX_ENTRIES) {
    let totalChars = 0
    for (const entry of entries) {
      totalChars += getOutputText(entry)?.text.length ?? 0
      if (totalChars > TERMINAL_OUTPUT_MAX_CHARS) {
        break
      }
    }
    if (totalChars <= TERMINAL_OUTPUT_MAX_CHARS) {
      return entries
    }
  }

  const kept: TerminalOutput[] = []
  let totalChars = 0

  for (let index = entries.length - 1; index >= 0; index -= 1) {
    if (kept.length >= TERMINAL_OUTPUT_MAX_ENTRIES) {
      break
    }

    const entry = entries[index]
    const current = getOutputText(entry)
    const textLength = current?.text.length ?? 0
    if (
      kept.length > 0 &&
      textLength > 0 &&
      totalChars + textLength > TERMINAL_OUTPUT_MAX_CHARS
    ) {
      break
    }

    kept.push(entry)
    totalChars += textLength
  }

  return kept.reverse()
}

export function useTerminalConnection({
  itemId,
  enabled = true,
  onConnected,
  onDisconnected,
  onError,
}: UseTerminalConnectionOptions): UseTerminalConnectionReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [output, setOutput] = useState<TerminalOutput[]>([])

  const socketRef = useRef<Socket | null>(null)
  const connectingRef = useRef(false)
  const mountedRef = useRef(true)
  const connectionIdRef = useRef(0)
  const pendingOutputRef = useRef<TerminalOutput[]>([])
  const outputFlushTimerRef = useRef<number | null>(null)

  const onConnectedRef = useRef(onConnected)
  const onDisconnectedRef = useRef(onDisconnected)
  const onErrorRef = useRef(onError)

  useEffect(() => {
    onConnectedRef.current = onConnected
    onDisconnectedRef.current = onDisconnected
    onErrorRef.current = onError
  }, [onConnected, onDisconnected, onError])

  const clearOutput = useCallback(() => {
    pendingOutputRef.current = []
    setOutput([])
  }, [])

  const flushPendingOutput = useCallback(() => {
    outputFlushTimerRef.current = null
    const pending = pendingOutputRef.current
    if (pending.length === 0) {
      return
    }

    pendingOutputRef.current = []
    setOutput((previous) => {
      let next = previous
      for (const entry of pending) {
        next = appendTerminalOutput(next, entry)
      }
      return limitTerminalOutput(next)
    })
  }, [])

  const enqueueOutput = useCallback(
    (entry: TerminalOutput) => {
      pendingOutputRef.current.push(entry)
      if (pendingOutputRef.current.length > TERMINAL_PENDING_OUTPUT_MAX_ENTRIES) {
        pendingOutputRef.current = pendingOutputRef.current.slice(
          -TERMINAL_PENDING_OUTPUT_MAX_ENTRIES,
        )
      }
      if (outputFlushTimerRef.current !== null) {
        return
      }
      outputFlushTimerRef.current = window.setTimeout(
        flushPendingOutput,
        TERMINAL_STREAM_FLUSH_MS,
      )
    },
    [flushPendingOutput],
  )

  const cancelPendingOutputFlush = useCallback(() => {
    if (outputFlushTimerRef.current !== null) {
      window.clearTimeout(outputFlushTimerRef.current)
      outputFlushTimerRef.current = null
    }
    pendingOutputRef.current = []
  }, [])

  const disposeSocket = useCallback((socket: Socket | null) => {
    if (!socket) {
      return
    }
    socket.removeAllListeners()
    socket.disconnect()
    if (socketRef.current === socket) {
      socketRef.current = null
    }
  }, [])

  const disconnect = useCallback(() => {
    const socket = socketRef.current
    disposeSocket(socket)
    cancelPendingOutputFlush()
    setIsConnected(false)
    setIsConnecting(false)
    connectingRef.current = false
  }, [cancelPendingOutputFlush, disposeSocket])

  const doConnect = useCallback(async (force = false) => {
    if ((!force && !enabled) || !itemId) {
      return
    }

    if (connectingRef.current) {
      return
    }

    const existingSocket = socketRef.current
    if (existingSocket) {
      if (existingSocket.connected) {
        return
      }
      disposeSocket(existingSocket)
    }

    const currentConnectionId = ++connectionIdRef.current
    connectingRef.current = true
    setIsConnecting(true)
    setError(null)

    try {
      const outputData = (await ItemsService.getItemOutput({ id: itemId })) as {
        success: boolean
        output: string | null
      }
      if (outputData.success && outputData.output) {
        const lines = sanitizeTerminalText(outputData.output).split("\n")
        const outputLines = lines
          .filter((line) => line.trim())
          .map((line) => ({ stdout: `${line}\n` }))
        setOutput(limitTerminalOutput(outputLines))
      }

      if (
        !mountedRef.current ||
        currentConnectionId !== connectionIdRef.current
      ) {
        return
      }

      const tokenData = await ItemsService.getTerminalToken({ id: itemId })

      if (
        !mountedRef.current ||
        currentConnectionId !== connectionIdRef.current
      ) {
        return
      }

      if (!tokenData.success) {
        throw new Error("Failed to get terminal token")
      }

      const { temp_token, ws_url, item_uuid } = tokenData as {
        temp_token: string
        ws_url: string
        item_uuid: string
      }

      const socket = io(ws_url, {
        transports: ["websocket"],
        auth: {
          temp_token,
          item_uuid,
        },
        reconnection: false,
      })

      if (
        !mountedRef.current ||
        currentConnectionId !== connectionIdRef.current
      ) {
        socket.removeAllListeners()
        socket.disconnect()
        return
      }

      socketRef.current = socket

      socket.on("connect", () => {
        if (TERMINAL_DEBUG) {
          console.debug("[Terminal] Socket connected")
        }
      })

      socket.on("terminal_connected", (data: TerminalConnectedData) => {
        if (
          !mountedRef.current ||
          currentConnectionId !== connectionIdRef.current
        ) {
          return
        }
        if (TERMINAL_DEBUG) {
          console.debug("[Terminal] Terminal connected:", data)
        }
        setIsConnected(true)
        setIsConnecting(false)
        connectingRef.current = false
        onConnectedRef.current?.(data)
      })

      socket.on("auth_error", (data: { message: string }) => {
        if (
          !mountedRef.current ||
          currentConnectionId !== connectionIdRef.current
        ) {
          return
        }
        if (TERMINAL_DEBUG) {
          console.warn("[Terminal] Auth error:", data.message)
        }
        setError(data.message)
        setIsConnecting(false)
        connectingRef.current = false
        onErrorRef.current?.(data.message)
        disposeSocket(socket)
      })

      socket.on("stream", (data: TerminalOutput) => {
        if (
          !mountedRef.current ||
          currentConnectionId !== connectionIdRef.current
        ) {
          return
        }
        if (data.exit) {
          const code = data.exit_code
          const label =
            code === 0 || code == null
              ? "[Process exited]"
              : `[Process exited with code ${code}]`
          enqueueOutput({ stdout: label + "\r\n", exit: true, exit_code: code ?? undefined })
          return
        }
        enqueueOutput(data)
      })

      socket.on("disconnect", (reason) => {
        if (TERMINAL_DEBUG) {
          console.debug("[Terminal] Disconnected:", reason)
        }
        if (socketRef.current === socket) {
          setIsConnected(false)
          setIsConnecting(false)
          connectingRef.current = false
          socketRef.current = null
          onDisconnectedRef.current?.()
        }
      })

      socket.on("connect_error", (err) => {
        if (
          !mountedRef.current ||
          currentConnectionId !== connectionIdRef.current
        ) {
          return
        }
        if (TERMINAL_DEBUG) {
          console.warn("[Terminal] Connect error:", err.message)
        }
        setError(err.message)
        setIsConnecting(false)
        connectingRef.current = false
        onErrorRef.current?.(err.message)
        disposeSocket(socket)
      })
    } catch (err) {
      if (
        !mountedRef.current ||
        currentConnectionId !== connectionIdRef.current
      ) {
        return
      }
      if (TERMINAL_DEBUG) {
        console.warn("[Terminal] Connection error:", err)
      }
      const errorMessage =
        err instanceof Error ? err.message : "Connection failed"
      setError(errorMessage)
      setIsConnecting(false)
      connectingRef.current = false
      onErrorRef.current?.(errorMessage)
    }
  }, [disposeSocket, enabled, enqueueOutput, itemId])

  const reconnect = useCallback((force = false) => {
    disconnect()
    connectionIdRef.current++
    setTimeout(() => {
      if (mountedRef.current) {
        doConnect(force)
      }
    }, 300)
  }, [doConnect, disconnect])

  const sendCommand = useCallback(
    (command: string) => {
      if (socketRef.current && isConnected) {
        socketRef.current.emit("terminal/write", { command })
      }
    },
    [isConnected],
  )

  const completeCommand = useCallback(
    (command: string, cursor: number) =>
      new Promise<TerminalCompletion | null>((resolve) => {
        const socket = socketRef.current
        if (!socket || !isConnected) {
          resolve(null)
          return
        }

        let settled = false
        const timeoutId = window.setTimeout(() => {
          if (!settled) {
            settled = true
            resolve(null)
          }
        }, 1500)

        socket.emit(
          "terminal/complete",
          { command, cursor },
          (result: (TerminalCompletion & { success?: boolean }) | undefined) => {
            if (settled) {
              return
            }
            settled = true
            window.clearTimeout(timeoutId)
            if (!result?.success) {
              resolve(null)
              return
            }
            resolve({
              value: result.value,
              cursor: result.cursor,
              candidates: result.candidates || [],
            })
          },
        )
      }),
    [isConnected],
  )

  const sendCtrlC = useCallback(() => {
    if (socketRef.current && isConnected) {
      socketRef.current.emit("terminal/write", { command: "\x03" })
    }
  }, [isConnected])

  useEffect(() => {
    mountedRef.current = true

    if (enabled && itemId) {
      doConnect()
    }

    return () => {
      mountedRef.current = false
      connectionIdRef.current++
      const socket = socketRef.current
      disposeSocket(socket)
      cancelPendingOutputFlush()
      connectingRef.current = false
    }
  }, [
    cancelPendingOutputFlush,
    disposeSocket,
    enabled,
    itemId,
    doConnect,
  ])

  return {
    isConnected,
    isConnecting,
    error,
    output,
    clearOutput,
    sendCommand,
    completeCommand,
    sendCtrlC,
    reconnect,
    disconnect,
  }
}
