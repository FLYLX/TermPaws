import { zodResolver } from "@hookform/resolvers/zod"
import {
  useMutation,
  useQueryClient,
  useSuspenseQuery,
} from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import {
  ArrowLeft,
  Pencil,
  Plus,
  RefreshCw,
  Search,
  Server,
  Trash2,
} from "lucide-react"
import { Suspense, useState } from "react"
import { useForm } from "react-hook-form"
import { z } from "zod"
import { McpService } from "@/client"
import PendingItems from "@/components/Pending/PendingItems"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import useAuth from "@/hooks/useAuth"
import useCustomToast from "@/hooks/useCustomToast"

function getMCPServersQueryOptions() {
  return {
    queryFn: () => McpService.listMcpServers(),
    queryKey: ["mcp-servers"],
  }
}

const serverFormSchema = z.object({
  name: z.string().min(1, "Name is required"),
  command: z.string().min(1, "Command is required"),
  args: z.string().optional(),
  env: z.string().optional(),
  enabled: z.boolean(),
  description: z.string().optional(),
})

type ServerFormData = z.infer<typeof serverFormSchema>

function MCPServersPage() {
  const { data: servers } = useSuspenseQuery(getMCPServersQueryOptions())
  const { user: currentUser } = useAuth()
  const [selectedServerName, setSelectedServerName] = useState<string | null>(
    null,
  )
  const [searchQuery, setSearchQuery] = useState("")
  const queryClient = useQueryClient()

  const selectedServer = servers.data.find((s) => s.name === selectedServerName)

  const filteredServers = servers.data.filter(
    (s) =>
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.description?.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const handleReload = async () => {
    if (currentUser?.is_superuser) {
      await McpService.reloadMcpServers()
      queryClient.invalidateQueries({ queryKey: ["mcp-servers"] })
    }
  }

  return (
    <div className="flex h-[calc(100svh-180px)] min-h-[400px] flex-col border rounded-lg overflow-hidden md:flex-row">
      <div
        className={`w-full flex-col border-b bg-muted/30 shrink-0 md:flex md:w-72 md:border-b-0 md:border-r ${
          selectedServer ? "hidden" : "flex"
        }`}
      >
        <div className="p-3 border-b space-y-2 shrink-0">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">MCP Servers</h2>
            <div className="flex items-center gap-1">
              {currentUser?.is_superuser && (
                <>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleReload}
                    title="Reload"
                  >
                    <RefreshCw className="h-4 w-4" />
                  </Button>
                  <AddMCPServer />
                </>
              )}
            </div>
          </div>
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search servers..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 h-8"
            />
          </div>
        </div>
        <ScrollArea className="flex-1 min-h-0">
          {filteredServers.length === 0 ? (
            <div className="p-4 text-center text-muted-foreground text-sm">
              {servers.data.length === 0
                ? "No MCP servers found"
                : "No matching servers"}
            </div>
          ) : (
            <div className="p-1">
              {filteredServers.map((server) => (
                <button
                  key={server.name}
                  onClick={() => setSelectedServerName(server.name)}
                  className={`w-full text-left p-3 rounded-md transition-colors ${
                    selectedServerName === server.name
                      ? "bg-primary/10 border border-primary/20"
                      : "hover:bg-muted border border-transparent"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="font-medium truncate">{server.name}</div>
                    </div>
                    <Badge
                      variant={server.enabled ? "default" : "secondary"}
                      className="text-xs shrink-0"
                    >
                      {server.enabled ? "Enabled" : "Disabled"}
                    </Badge>
                  </div>
                  {server.description && (
                    <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                      {server.description}
                    </p>
                  )}
                </button>
              ))}
            </div>
          )}
        </ScrollArea>
      </div>

      <div
        className={`flex-1 flex-col min-w-0 min-h-0 ${
          selectedServer ? "flex" : "hidden md:flex"
        }`}
      >
        {selectedServer ? (
          <>
            <div className="shrink-0 border-b p-2 md:hidden">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedServerName(null)}
              >
                <ArrowLeft className="mr-1 h-4 w-4" /> 返回列表
              </Button>
            </div>
            <MCPServerEditor server={selectedServer} />
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground">
            <Server className="h-16 w-16 mb-4 opacity-20" />
            <p className="text-lg font-medium">No server selected</p>
            <p className="text-sm">
              Select a server from the list to view and edit
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

function MCPServerEditor({
  server,
}: {
  server: {
    name: string
    command: string
    args: string[]
    env: Record<string, string>
    enabled: boolean
    description: string
  }
}) {
  const [editDialog, setEditDialog] = useState(false)
  const [deleteDialog, setDeleteDialog] = useState(false)
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const queryClient = useQueryClient()

  const deleteMutation = useMutation({
    mutationFn: () => McpService.deleteMcpServer({ serverName: server.name }),
    onSuccess: () => {
      showSuccessToast("Server deleted successfully")
      queryClient.invalidateQueries({ queryKey: ["mcp-servers"] })
    },
    onError: (error: any) => {
      showErrorToast(error?.message || "Failed to delete server")
    },
  })

  return (
    <>
      <div className="border-b p-3 flex items-center justify-between bg-muted/30 shrink-0">
        <div className="flex items-center gap-3">
          <div>
            <h3 className="font-semibold">{server.name}</h3>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <code className="bg-muted px-1.5 py-0.5 rounded">
                {server.command}
              </code>
              <Badge
                variant={server.enabled ? "default" : "secondary"}
                className="text-xs"
              >
                {server.enabled ? "Enabled" : "Disabled"}
              </Badge>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setEditDialog(true)}
          >
            <Pencil className="h-4 w-4 mr-1" />
            Edit
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setDeleteDialog(true)}
          >
            <Trash2 className="h-4 w-4 mr-1" />
            Delete
          </Button>
        </div>
      </div>

      <ScrollArea className="flex-1 min-h-0">
        <div className="p-4 space-y-4">
          <div>
            <h4 className="text-sm font-semibold mb-1">Description</h4>
            <p className="text-sm text-muted-foreground">
              {server.description || "No description"}
            </p>
          </div>

          <div>
            <h4 className="text-sm font-semibold mb-1">Command</h4>
            <code className="block text-sm bg-muted p-2 rounded">
              {server.command}
            </code>
          </div>

          <div>
            <h4 className="text-sm font-semibold mb-1">Arguments</h4>
            {server.args.length > 0 ? (
              <code className="block text-sm bg-muted p-2 rounded">
                {server.args.join(" ")}
              </code>
            ) : (
              <p className="text-sm text-muted-foreground">No arguments</p>
            )}
          </div>

          <div>
            <h4 className="text-sm font-semibold mb-1">
              Environment Variables
            </h4>
            {Object.keys(server.env).length > 0 ? (
              <pre className="text-sm bg-muted p-2 rounded overflow-auto">
                {JSON.stringify(server.env, null, 2)}
              </pre>
            ) : (
              <p className="text-sm text-muted-foreground">
                No environment variables
              </p>
            )}
          </div>

          <div>
            <h4 className="text-sm font-semibold mb-1">Status</h4>
            <Badge variant={server.enabled ? "default" : "secondary"}>
              {server.enabled ? "Enabled" : "Disabled"}
            </Badge>
          </div>
        </div>
      </ScrollArea>

      <EditMCPServerDialog
        server={server}
        open={editDialog}
        onOpenChange={setEditDialog}
      />

      <AlertDialog open={deleteDialog} onOpenChange={setDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete MCP Server</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{server.name}"? This action
              cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteMutation.mutate()}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}

function AddMCPServer() {
  const [open, setOpen] = useState(false)
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const queryClient = useQueryClient()

  const form = useForm<ServerFormData>({
    resolver: zodResolver(serverFormSchema),
    defaultValues: {
      name: "",
      command: "",
      args: "",
      env: "",
      enabled: true,
      description: "",
    },
  })

  const createMutation = useMutation({
    mutationFn: (data: ServerFormData) =>
      McpService.createMcpServer({
        requestBody: {
          name: data.name,
          command: data.command,
          args: data.args ? data.args.split(" ").filter(Boolean) : [],
          env: data.env ? JSON.parse(data.env) : {},
          enabled: data.enabled,
          description: data.description || "",
        },
      }),
    onSuccess: () => {
      showSuccessToast("Server created successfully")
      setOpen(false)
      form.reset()
      queryClient.invalidateQueries({ queryKey: ["mcp-servers"] })
    },
    onError: (error: any) => {
      showErrorToast(error?.message || "Failed to create server")
    },
  })

  const onSubmit = (data: ServerFormData) => {
    createMutation.mutate(data)
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setOpen(true)}
        title="Add Server"
      >
        <Plus className="h-4 w-4" />
      </Button>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Add MCP Server</DialogTitle>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input placeholder="my-mcp-server" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="command"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Command</FormLabel>
                  <FormControl>
                    <Input placeholder="npx" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="args"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Arguments</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="-y @modelcontextprotocol/server-filesystem /path"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>Space-separated arguments</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="env"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Environment Variables (JSON)</FormLabel>
                  <FormControl>
                    <Textarea placeholder='{"API_KEY": "xxx"}' {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description</FormLabel>
                  <FormControl>
                    <Input placeholder="Server description" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="enabled"
              render={({ field }) => (
                <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                  <FormControl>
                    <Checkbox
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                  <div className="space-y-1 leading-none">
                    <FormLabel>Enabled</FormLabel>
                  </div>
                </FormItem>
              )}
            />
            <DialogFooter>
              <Button type="submit" disabled={createMutation.isPending}>
                {createMutation.isPending ? "Creating..." : "Create"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}

function EditMCPServerDialog({
  server,
  open,
  onOpenChange,
}: {
  server: {
    name: string
    command: string
    args: string[]
    env: Record<string, string>
    enabled: boolean
    description: string
  }
  open: boolean
  onOpenChange: (open: boolean) => void
}) {
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const queryClient = useQueryClient()

  const form = useForm<ServerFormData>({
    resolver: zodResolver(serverFormSchema),
    defaultValues: {
      name: server.name,
      command: server.command,
      args: server.args.join(" "),
      env:
        Object.keys(server.env).length > 0
          ? JSON.stringify(server.env, null, 2)
          : "",
      enabled: server.enabled,
      description: server.description || "",
    },
  })

  const updateMutation = useMutation({
    mutationFn: (data: ServerFormData) =>
      McpService.updateMcpServer({
        serverName: server.name,
        requestBody: {
          command: data.command,
          args: data.args ? data.args.split(" ").filter(Boolean) : [],
          env: data.env ? JSON.parse(data.env) : {},
          enabled: data.enabled,
          description: data.description,
        },
      }),
    onSuccess: () => {
      showSuccessToast("Server updated successfully")
      onOpenChange(false)
      queryClient.invalidateQueries({ queryKey: ["mcp-servers"] })
    },
    onError: (error: any) => {
      showErrorToast(error?.message || "Failed to update server")
    },
  })

  const onSubmit = (data: ServerFormData) => {
    updateMutation.mutate(data)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit MCP Server</DialogTitle>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Name</FormLabel>
                  <FormControl>
                    <Input {...field} disabled />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="command"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Command</FormLabel>
                  <FormControl>
                    <Input placeholder="npx" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="args"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Arguments</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="-y @modelcontextprotocol/server-filesystem /path"
                      {...field}
                    />
                  </FormControl>
                  <FormDescription>Space-separated arguments</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="env"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Environment Variables (JSON)</FormLabel>
                  <FormControl>
                    <Textarea placeholder='{"API_KEY": "xxx"}' {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="description"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Description</FormLabel>
                  <FormControl>
                    <Input placeholder="Server description" {...field} />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="enabled"
              render={({ field }) => (
                <FormItem className="flex flex-row items-start space-x-3 space-y-0">
                  <FormControl>
                    <Checkbox
                      checked={field.value}
                      onCheckedChange={field.onChange}
                    />
                  </FormControl>
                  <div className="space-y-1 leading-none">
                    <FormLabel>Enabled</FormLabel>
                  </div>
                </FormItem>
              )}
            />
            <DialogFooter>
              <Button type="submit" disabled={updateMutation.isPending}>
                {updateMutation.isPending ? "Saving..." : "Save"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
}

export const Route = createFileRoute("/_layout/mcp-servers")({
  component: () => (
    <Suspense fallback={<PendingItems />}>
      <MCPServersPage />
    </Suspense>
  ),
})
