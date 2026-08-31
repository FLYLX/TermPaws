import { useQuery, useQueryClient } from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import {
  ArrowLeft,
  Bot,
  CheckCircle2,
  Plug,
  Power,
  RefreshCw,
  Search,
  Server,
  Terminal,
} from "lucide-react"
import { useMemo, useState } from "react"

import PendingItems from "@/components/Pending/PendingItems"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import useAuth from "@/hooks/useAuth"
import useCustomToast from "@/hooks/useCustomToast"
import {
  getPluginsQueryOptions,
  type PluginItem,
  type PluginListResponse,
  requestPluginApi,
} from "@/lib/plugins-api"

function pluginIcon(plugin: PluginItem) {
  if (plugin.plugin_id.includes("robot")) {
    return Bot
  }
  if (plugin.category === "terminal") {
    return Terminal
  }
  if (plugin.capabilities.includes("api")) {
    return Server
  }
  return Plug
}

function PluginsPage() {
  const { user: currentUser } = useAuth()
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const queryClient = useQueryClient()
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedPluginId, setSelectedPluginId] = useState<string | null>(null)
  const [actionPluginId, setActionPluginId] = useState<string | null>(null)
  const [isReloading, setIsReloading] = useState(false)

  const { data, isLoading } = useQuery(getPluginsQueryOptions())
  const plugins = data?.data || []

  const filteredPlugins = useMemo(
    () =>
      plugins.filter((plugin) => {
        const query = searchQuery.trim().toLowerCase()
        if (!query) {
          return true
        }
        return (
          plugin.name.toLowerCase().includes(query) ||
          plugin.plugin_id.toLowerCase().includes(query) ||
          plugin.category.toLowerCase().includes(query) ||
          plugin.description.toLowerCase().includes(query)
        )
      }),
    [plugins, searchQuery],
  )

  const selectedPlugin =
    plugins.find((plugin) => plugin.plugin_id === selectedPluginId) ||
    filteredPlugins[0] ||
    null

  const reloadPlugins = async () => {
    if (!currentUser?.is_superuser) {
      return
    }
    setIsReloading(true)
    try {
      await requestPluginApi<PluginListResponse>("/reload", { method: "POST" })
      await queryClient.invalidateQueries({ queryKey: ["plugins"] })
      showSuccessToast("Plugins reloaded")
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : "Reload failed")
    } finally {
      setIsReloading(false)
    }
  }

  const togglePlugin = async (plugin: PluginItem) => {
    if (!currentUser?.is_superuser || !plugin.configurable) {
      return
    }
    setActionPluginId(plugin.plugin_id)
    try {
      await requestPluginApi<PluginItem>(
        `/${encodeURIComponent(plugin.plugin_id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({ enabled: !plugin.enabled }),
        },
      )
      await queryClient.invalidateQueries({ queryKey: ["plugins"] })
      showSuccessToast(plugin.enabled ? "Plugin disabled" : "Plugin enabled")
    } catch (error) {
      showErrorToast(error instanceof Error ? error.message : "Update failed")
    } finally {
      setActionPluginId(null)
    }
  }

  if (isLoading) {
    return <PendingItems />
  }

  return (
    <div className="flex h-[calc(100svh-180px)] min-h-[420px] flex-col overflow-hidden rounded-lg border md:flex-row">
      <div
        className={`w-full flex-col border-b bg-muted/30 md:flex md:w-80 md:shrink-0 md:border-b-0 md:border-r ${
          selectedPlugin ? "hidden" : "flex"
        }`}
      >
        <div className="shrink-0 space-y-2 border-b p-3">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Plugin Market</h2>
            {currentUser?.is_superuser && (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => void reloadPlugins()}
                disabled={isReloading}
                title="Reload plugins"
              >
                <RefreshCw
                  className={`size-4 ${isReloading ? "animate-spin" : ""}`}
                />
              </Button>
            )}
          </div>
          <div className="relative">
            <Search className="absolute left-2 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search plugins..."
              value={searchQuery}
              onChange={(event) => setSearchQuery(event.target.value)}
              className="h-8 pl-8"
            />
          </div>
        </div>

        <ScrollArea className="min-h-0 flex-1">
          {filteredPlugins.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted-foreground">
              No matching plugins
            </div>
          ) : (
            <div className="p-1">
              {filteredPlugins.map((plugin) => {
                const Icon = pluginIcon(plugin)
                return (
                  <button
                    key={plugin.plugin_id}
                    type="button"
                    onClick={() => setSelectedPluginId(plugin.plugin_id)}
                    className={`w-full rounded-md border p-3 text-left transition-colors ${
                      selectedPlugin?.plugin_id === plugin.plugin_id
                        ? "border-primary/20 bg-primary/10"
                        : "border-transparent hover:bg-muted"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <Icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate font-medium">
                            {plugin.name}
                          </span>
                          <Badge
                            variant={plugin.enabled ? "default" : "secondary"}
                            className="shrink-0 text-xs"
                          >
                            {plugin.enabled ? "Enabled" : "Disabled"}
                          </Badge>
                        </div>
                        <div className="truncate text-xs text-muted-foreground">
                          {plugin.plugin_id}
                        </div>
                        <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
                          {plugin.description}
                        </p>
                      </div>
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </ScrollArea>
      </div>

      <div
        className={`min-h-0 min-w-0 flex-1 ${selectedPlugin ? "" : "hidden md:block"}`}
      >
        {selectedPlugin ? (
          <PluginDetail
            plugin={selectedPlugin}
            canManage={Boolean(currentUser?.is_superuser)}
            isBusy={actionPluginId === selectedPlugin.plugin_id}
            onToggle={() => void togglePlugin(selectedPlugin)}
            onBack={() => setSelectedPluginId(null)}
          />
        ) : (
          <div className="flex h-full flex-col items-center justify-center text-muted-foreground">
            <Plug className="mb-4 size-16 opacity-20" />
            <p className="text-lg font-medium">No plugin selected</p>
          </div>
        )}
      </div>
    </div>
  )
}

function PluginDetail({
  plugin,
  canManage,
  isBusy,
  onToggle,
  onBack,
}: {
  plugin: PluginItem
  canManage: boolean
  isBusy: boolean
  onToggle: () => void
  onBack: () => void
}) {
  const Icon = pluginIcon(plugin)
  return (
    <div className="flex h-full flex-col">
      <div className="flex shrink-0 flex-col gap-3 border-b bg-muted/30 p-4 md:flex-row md:items-start md:justify-between">
        <div className="flex min-w-0 items-start gap-2">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="shrink-0 md:hidden"
            onClick={onBack}
            title="返回列表"
          >
            <ArrowLeft className="size-4" />
          </Button>
          <div className="flex min-w-0 items-start gap-3">
            <div className="rounded-md border bg-background p-2">
              <Icon className="size-5 text-muted-foreground" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-xl font-semibold">{plugin.name}</h1>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                <code className="rounded bg-muted px-1.5 py-0.5">
                  {plugin.plugin_id}
                </code>
                <Badge variant="outline">{plugin.category}</Badge>
                <Badge variant="secondary">{plugin.version}</Badge>
                {plugin.builtin && <Badge variant="secondary">Built-in</Badge>}
              </div>
            </div>
          </div>
        </div>
        <Button
          type="button"
          variant={plugin.enabled ? "outline" : "default"}
          onClick={onToggle}
          disabled={!canManage || !plugin.configurable || isBusy}
          className="w-fit"
        >
          {isBusy ? (
            <RefreshCw className="size-4 animate-spin" />
          ) : plugin.enabled ? (
            <Power className="size-4" />
          ) : (
            <CheckCircle2 className="size-4" />
          )}
          {plugin.enabled ? "Disable" : "Enable"}
        </Button>
      </div>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        <section className="space-y-2">
          <h2 className="text-sm font-semibold">Overview</h2>
          <p className="max-w-3xl text-sm text-muted-foreground">
            {plugin.description || "No description"}
          </p>
        </section>

        <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          <InfoTile
            label="Current State"
            value={plugin.enabled ? "Enabled" : "Disabled"}
          />
          <InfoTile
            label="Default State"
            value={plugin.default_enabled ? "Enabled" : "Disabled"}
          />
          <InfoTile
            label="Source"
            value={plugin.builtin ? "Built-in" : "Local"}
          />
          <InfoTile
            label="Configurable"
            value={plugin.configurable ? "Yes" : "No"}
          />
        </section>

        <section className="space-y-2">
          <h2 className="text-sm font-semibold">Capabilities</h2>
          <div className="flex flex-wrap gap-2">
            {plugin.capabilities.length === 0 ? (
              <Badge variant="secondary">none</Badge>
            ) : (
              plugin.capabilities.map((capability) => (
                <Badge key={capability} variant="secondary">
                  {capability}
                </Badge>
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  )
}

function InfoTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border bg-muted/20 p-3">
      <div className="text-xs uppercase text-muted-foreground">{label}</div>
      <div className="mt-1 font-medium">{value}</div>
    </div>
  )
}

export const Route = createFileRoute("/_layout/plugins")({
  component: PluginsPage,
  head: () => ({
    meta: [
      {
        title: "Plugins - TermPaws",
      },
    ],
  }),
})
