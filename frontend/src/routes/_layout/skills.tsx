import {
  useQuery,
  useQueryClient,
  useSuspenseQuery,
} from "@tanstack/react-query"
import { createFileRoute } from "@tanstack/react-router"
import {
  ArrowLeft,
  Code2,
  FileCode,
  FilePlus,
  RefreshCw,
  Search,
  Trash2,
  X,
} from "lucide-react"
import { Suspense, useState } from "react"
import { type SkillListItem, SkillsService } from "@/client"
import PendingItems from "@/components/Pending/PendingItems"
import AddSkill from "@/components/Skills/AddSkill"
import DeleteSkill from "@/components/Skills/DeleteSkill"
import EditSkill from "@/components/Skills/EditSkill"
import GenerateSkillDialog from "@/components/Skills/GenerateSkillDialog"
import UploadSkillZip from "@/components/Skills/UploadSkillZip"
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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import useAuth from "@/hooks/useAuth"
import useCustomToast from "@/hooks/useCustomToast"

function getSkillsQueryOptions() {
  return {
    queryFn: () => SkillsService.listSkills({}),
    queryKey: ["skills"],
  }
}

interface SkillFile {
  path: string
  name: string
  size: number
  is_binary: boolean
}

function SkillsPage() {
  const { data: skills } = useSuspenseQuery(getSkillsQueryOptions())
  const { user: currentUser } = useAuth()
  const [selectedSkillId, setSelectedSkillId] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const queryClient = useQueryClient()

  const selectedSkill = skills.data.find((s) => s.skill_id === selectedSkillId)

  const filteredSkills = skills.data.filter(
    (s) =>
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.skill_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      s.category.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  const handleReload = async () => {
    if (currentUser?.is_superuser) {
      await SkillsService.reloadSkills()
      queryClient.invalidateQueries({ queryKey: ["skills"] })
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col border rounded-lg overflow-hidden bg-card/60 backdrop-blur-sm md:flex-row">
      <div
        className={`w-full flex-col border-b bg-muted/30 shrink-0 md:flex md:w-60 md:border-b-0 md:border-r ${
          selectedSkill ? "hidden" : "flex"
        }`}
      >
        <div className="p-3 border-b space-y-2 shrink-0">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold">Skills</h2>
            <div className="flex items-center gap-1">
              {currentUser?.is_superuser && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleReload}
                  title="Reload Skills"
                >
                  <RefreshCw className="h-4 w-4" />
                </Button>
              )}
              <GenerateSkillDialog onCreated={setSelectedSkillId} />
              <AddSkill />
              <UploadSkillZip />
            </div>
          </div>
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search skills..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 h-8"
            />
          </div>
        </div>
        <ScrollArea className="flex-1 min-h-0">
          {filteredSkills.length === 0 ? (
            <div className="p-4 text-center text-muted-foreground text-sm">
              {skills.data.length === 0
                ? "No skills found"
                : "No matching skills"}
            </div>
          ) : (
            <div className="p-1">
              {filteredSkills.map((skill) => (
                <button
                  key={skill.skill_id}
                  onClick={() => setSelectedSkillId(skill.skill_id)}
                  className={`w-full text-left p-3 rounded-md transition-colors ${
                    selectedSkillId === skill.skill_id
                      ? "bg-primary/10 border border-primary/20"
                      : "hover:bg-muted border border-transparent"
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 flex-1">
                      <div className="font-medium truncate">{skill.name}</div>
                      <div className="text-xs text-muted-foreground truncate">
                        {skill.skill_id}
                      </div>
                    </div>
                    <Badge variant="secondary" className="text-xs shrink-0">
                      {skill.category}
                    </Badge>
                  </div>
                  {skill.description && (
                    <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                      {skill.description}
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
          selectedSkill ? "flex" : "hidden md:flex"
        }`}
      >
        {selectedSkill ? (
          <>
            <div className="shrink-0 border-b p-2 md:hidden">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedSkillId(null)}
              >
                <ArrowLeft className="mr-1 h-4 w-4" /> 返回列表
              </Button>
            </div>
            <SkillEditor skill={selectedSkill} />
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground">
            <Code2 className="h-16 w-16 mb-4 opacity-20" />
            <p className="text-lg font-medium">No skill selected</p>
            <p className="text-sm">
              Select a skill from the list to view and edit
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

function SkillEditor({ skill }: { skill: SkillListItem }) {
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [fileContent, setFileContent] = useState<string>("")
  const [isSaving, setIsSaving] = useState(false)
  const [newFileDialog, setNewFileDialog] = useState(false)
  const [newFilePath, setNewFilePath] = useState("")
  const [deleteFileDialog, setDeleteFileDialog] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<"info" | "content" | "files">(
    "files",
  )
  const { showSuccessToast, showErrorToast } = useCustomToast()
  const queryClient = useQueryClient()

  const { data: detail, isLoading } = useQuery({
    queryKey: ["skill", skill.skill_id],
    queryFn: () => SkillsService.getSkill({ skillId: skill.skill_id }),
  })

  const { data: filesData, refetch: refetchFiles } = useQuery({
    queryKey: ["skill-files", skill.skill_id],
    queryFn: async () => {
      const result = await SkillsService.listSkillFiles({
        skillId: skill.skill_id,
      })
      return result as { files: SkillFile[] }
    },
  })

  const files = filesData?.files || []

  const handleFileClick = async (filePath: string) => {
    try {
      const result = await SkillsService.getSkillFile({
        skillId: skill.skill_id,
        filePath: filePath,
      })
      if (
        typeof result === "object" &&
        result !== null &&
        "content" in result
      ) {
        setFileContent(result.content as string)
        setSelectedFile(filePath)
      }
    } catch (error) {
      console.error("Failed to load file:", error)
      showErrorToast("Failed to load file")
    }
  }

  const handleSaveFile = async () => {
    if (!selectedFile) return
    setIsSaving(true)
    try {
      await SkillsService.updateSkillFile({
        skillId: skill.skill_id,
        filePath: selectedFile,
        requestBody: { content: fileContent },
      })
      showSuccessToast("File saved successfully")
      queryClient.invalidateQueries({ queryKey: ["skill", skill.skill_id] })
    } catch (error) {
      console.error("Failed to save file:", error)
      showErrorToast("Failed to save file")
    } finally {
      setIsSaving(false)
    }
  }

  const handleCreateFile = async () => {
    if (!newFilePath.trim()) {
      showErrorToast("File path is required")
      return
    }

    try {
      await SkillsService.createSkillFile({
        skillId: skill.skill_id,
        filePath: newFilePath,
        requestBody: { content: "" },
      })
      showSuccessToast("File created successfully")
      setNewFileDialog(false)
      setNewFilePath("")
      refetchFiles()
      setSelectedFile(newFilePath)
      setFileContent("")
    } catch (error) {
      console.error("Failed to create file:", error)
      showErrorToast("Failed to create file")
    }
  }

  const handleDeleteFile = async () => {
    if (!deleteFileDialog) return

    try {
      await SkillsService.deleteSkillFile({
        skillId: skill.skill_id,
        filePath: deleteFileDialog,
      })
      showSuccessToast("File deleted successfully")
      setDeleteFileDialog(null)
      if (selectedFile === deleteFileDialog) {
        setSelectedFile(null)
        setFileContent("")
      }
      refetchFiles()
    } catch (error) {
      console.error("Failed to delete file:", error)
      showErrorToast("Failed to delete file")
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const getFileIcon = (file: SkillFile) => {
    if (file.is_binary) {
      return <FileCode className="h-4 w-4 text-orange-500" />
    }
    if (file.name.endsWith(".md")) {
      return <FileCode className="h-4 w-4 text-blue-500" />
    }
    if (file.name.endsWith(".py") || file.name.endsWith(".sh")) {
      return <FileCode className="h-4 w-4 text-green-500" />
    }
    if (file.name.endsWith(".json")) {
      return <FileCode className="h-4 w-4 text-yellow-500" />
    }
    return <FileCode className="h-4 w-4" />
  }

  return (
    <>
      <div className="border-b p-3 flex items-center justify-between bg-muted/30 shrink-0">
        <div className="flex items-center gap-3">
          <div>
            <h3 className="font-semibold">{skill.name}</h3>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <code className="bg-muted px-1.5 py-0.5 rounded">
                {skill.skill_id}
              </code>
              <Badge variant="secondary" className="text-xs">
                {skill.category}
              </Badge>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <EditSkill skill={skill} />
          <DeleteSkill skill={skill} />
        </div>
      </div>

      <div className="flex-1 flex min-h-0 overflow-hidden">
        <div className="w-56 border-r flex flex-col bg-muted/20 shrink-0">
          <div className="flex border-b shrink-0">
            <button
              onClick={() => setActiveTab("files")}
              className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
                activeTab === "files"
                  ? "bg-background border-b-2 border-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Files
            </button>
            <button
              onClick={() => setActiveTab("info")}
              className={`flex-1 px-3 py-2 text-xs font-medium transition-colors ${
                activeTab === "info"
                  ? "bg-background border-b-2 border-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Info
            </button>
          </div>

          {activeTab === "files" && (
            <>
              <div className="flex items-center justify-between p-2 border-b shrink-0">
                <span className="text-xs font-medium text-muted-foreground">
                  {files.length} files
                </span>
                <div className="flex gap-1">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => setNewFileDialog(true)}
                    title="Create new file"
                  >
                    <FilePlus className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={() => refetchFiles()}
                    title="Refresh"
                  >
                    <RefreshCw className="h-3 w-3" />
                  </Button>
                </div>
              </div>
              <ScrollArea className="flex-1 min-h-0">
                {files.length === 0 ? (
                  <div className="p-4 text-center text-xs text-muted-foreground">
                    No files found
                  </div>
                ) : (
                  <div className="p-1">
                    {files.map((file) => (
                      <div
                        key={file.path}
                        className={`group flex items-center justify-between p-2 rounded cursor-pointer transition-colors ${
                          selectedFile === file.path
                            ? "bg-primary/10"
                            : "hover:bg-muted"
                        }`}
                        onClick={() =>
                          !file.is_binary && handleFileClick(file.path)
                        }
                      >
                        <div className="flex items-center gap-2 min-w-0 flex-1">
                          {getFileIcon(file)}
                          <div className="min-w-0 flex-1">
                            <div className="text-sm truncate">{file.path}</div>
                            <div className="text-xs text-muted-foreground">
                              {formatFileSize(file.size)}
                              {file.is_binary && " (binary)"}
                            </div>
                          </div>
                        </div>
                        {!file.is_binary && file.path !== "SKILL.md" && (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 opacity-0 group-hover:opacity-100"
                            onClick={(e) => {
                              e.stopPropagation()
                              setDeleteFileDialog(file.path)
                            }}
                          >
                            <Trash2 className="h-3 w-3 text-destructive" />
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </>
          )}

          {activeTab === "info" && isLoading ? (
            <div className="flex items-center justify-center py-8">
              <RefreshCw className="h-4 w-4 animate-spin" />
            </div>
          ) : activeTab === "info" && detail ? (
            <ScrollArea className="flex-1 min-h-0">
              <div className="p-3 space-y-4">
                <div>
                  <h4 className="text-xs font-semibold mb-1">Description</h4>
                  <p className="text-xs text-muted-foreground">
                    {detail.description || "No description"}
                  </p>
                </div>
                <div>
                  <h4 className="text-xs font-semibold mb-1">Trigger</h4>
                  <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-32">
                    {JSON.stringify(detail.trigger, null, 2)}
                  </pre>
                </div>
                <div>
                  <h4 className="text-xs font-semibold mb-1">Action</h4>
                  <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-32">
                    {JSON.stringify(detail.action, null, 2)}
                  </pre>
                </div>
                <div>
                  <h4 className="text-xs font-semibold mb-1">Safety</h4>
                  <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-32">
                    {JSON.stringify(detail.safety, null, 2)}
                  </pre>
                </div>
                <div className="flex gap-4">
                  <div>
                    <h4 className="text-xs font-semibold mb-1">Scripts</h4>
                    <div className="flex gap-1 flex-wrap">
                      {detail.scripts.length > 0 ? (
                        detail.scripts.map((s) => (
                          <Badge key={s} variant="outline" className="text-xs">
                            {s}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-xs text-muted-foreground">
                          None
                        </span>
                      )}
                    </div>
                  </div>
                  <div>
                    <h4 className="text-xs font-semibold mb-1">Templates</h4>
                    <div className="flex gap-1 flex-wrap">
                      {detail.templates.length > 0 ? (
                        detail.templates.map((t) => (
                          <Badge key={t} variant="outline" className="text-xs">
                            {t}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-xs text-muted-foreground">
                          None
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </ScrollArea>
          ) : null}
        </div>

        <div className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden">
          {selectedFile ? (
            <>
              <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/20 shrink-0">
                <div className="flex items-center gap-2">
                  <FileCode className="h-4 w-4 text-muted-foreground" />
                  <span className="text-sm font-medium">{selectedFile}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    onClick={handleSaveFile}
                    disabled={isSaving}
                  >
                    {isSaving ? (
                      <>
                        <RefreshCw className="h-3 w-3 mr-1 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      "Save"
                    )}
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7"
                    onClick={() => {
                      setSelectedFile(null)
                      setFileContent("")
                    }}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </div>
              <div className="flex-1 p-2 min-h-0">
                <Textarea
                  className="h-full w-full font-mono text-sm resize-none border-0 focus-visible:ring-0 bg-muted/30"
                  value={fileContent}
                  onChange={(e) => setFileContent(e.target.value)}
                />
              </div>
            </>
          ) : detail ? (
            <div className="flex-1 flex flex-col min-h-0">
              <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/20 shrink-0">
                <div className="flex items-center gap-2">
                  <FileCode className="h-4 w-4 text-blue-500" />
                  <span className="text-sm font-medium">SKILL.md</span>
                </div>
              </div>
              <ScrollArea className="flex-1 min-h-0">
                <pre className="p-4 text-sm whitespace-pre-wrap">
                  {detail.content}
                </pre>
              </ScrollArea>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-muted-foreground">
              <div className="text-center">
                <FileCode className="h-12 w-12 mx-auto mb-2 opacity-20" />
                <p className="text-sm">Select a file to edit</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <Dialog open={newFileDialog} onOpenChange={setNewFileDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create New File</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium">File Path</label>
              <p className="text-xs text-muted-foreground mb-2">
                e.g., scripts/my_script.sh, templates/output.txt
              </p>
              <Input
                value={newFilePath}
                onChange={(e) => setNewFilePath(e.target.value)}
                placeholder="scripts/new_script.sh"
              />
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setNewFileDialog(false)}>
                Cancel
              </Button>
              <Button onClick={handleCreateFile}>Create</Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={!!deleteFileDialog}
        onOpenChange={() => setDeleteFileDialog(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete File</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{deleteFileDialog}"? This action
              cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteFile}
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

export const Route = createFileRoute("/_layout/skills")({
  component: Skills,
  head: () => ({
    meta: [
      {
        title: "Skills - TermPaws",
      },
    ],
  }),
})

function Skills() {
  return (
    <div className="flex h-[calc(100svh-115px)] flex-col">
      <Suspense fallback={<PendingItems />}>
        <SkillsPage />
      </Suspense>
    </div>
  )
}
