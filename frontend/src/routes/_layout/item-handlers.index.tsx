import { createFileRoute, useNavigate } from "@tanstack/react-router"
import { useEffect } from "react"

import { ItemHandlersService } from "@/client"

import { TerminalDispatcherBoard } from "./item-handlers.$itemHandlerId"

export const Route = createFileRoute("/_layout/item-handlers/")({
  component: ItemHandlersIndex,
})

function ItemHandlersIndex() {
  const navigate = useNavigate()

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const handlers = (await ItemHandlersService.readItemHandlers({
          skip: 0,
          limit: 1,
        })) as any
        const first = Array.isArray(handlers)
          ? handlers[0]
          : (handlers?.data ?? [])[0]
        if (!cancelled && first?.id) {
          await navigate({
            to: "/item-handlers/$itemHandlerId",
            params: { itemHandlerId: String(first.id) },
            replace: true,
          })
        }
      } catch {
        // stay on empty-state board
      }
    })()
    return () => {
      cancelled = true
    }
  }, [navigate])

  return (
    <div
      className="full-bleed flex flex-col overflow-hidden text-zinc-900"
      style={{
        height: "calc(100svh - 79px)",
      }}
    >
      <TerminalDispatcherBoard itemHandlerId="" itemHandler={null} />
    </div>
  )
}
