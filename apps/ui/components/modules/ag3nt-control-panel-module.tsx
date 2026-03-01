"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import {
  Activity,
  Brain,
  Clock,
  FileText,
  FolderOpen,
  Hash,
  Key,
  Laptop,
  Radio,
  RefreshCw,
  Server,
  ShieldCheck,
  Signal,
  Terminal,
  Zap,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
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
import { ModuleContainer, EmptyModuleState, ErrorModuleState, LoadingModuleState } from "./module-container"
import { useAgentConnection } from "@/hooks/use-agent-connection"
import { useChat } from "@/providers/chat-provider"
import { AVAILABLE_MODELS } from "@/components/features/chat/model-selector"
import type { ModuleConfig, ModuleInstanceProps } from "@/types/modules"

type ViewId =
  | "dashboard"
  | "nodes"
  | "logs"
  | "workspace"
  | "memory"
  | "sessions"

export const ag3ntControlPanelModuleConfig: ModuleConfig = {
  metadata: {
    id: "ag3nt-control-panel",
    displayName: "AG3NT Control Panel",
    description: "Gateway dashboard, nodes, logs, workspace, and memory",
    icon: "Activity",
    category: "utility",
    version: "1.0.0",
  },
  hasHeader: true,
  initialState: { isLoading: true, error: null, data: {} },
  agentConfig: {
    enabled: true,
    supportedCommands: ["refresh", "setView"],
    emittedEvents: ["view-changed", "refreshed"],
    contextDescription: "AG3NT Gateway control panel (status, nodes, logs, workspace, memory)",
  },
}

function ControlPanelSection({
  icon: Icon,
  label,
  iconColor = "text-text-muted",
  children,
  action,
}: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  iconColor?: string
  children: React.ReactNode
  action?: React.ReactNode
}) {
  return (
    <div>
      <div className="flex items-center gap-2 py-2.5">
        <Icon className={cn("w-3.5 h-3.5", iconColor)} />
        <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">
          {label}
        </span>
        <div className="flex-1 border-b border-border/30 ml-2" />
        {action && <div className="shrink-0">{action}</div>}
      </div>
      <div className="mt-1">{children}</div>
    </div>
  )
}

const VIEWS: Array<{ id: ViewId; label: string; icon: any }> = [
  { id: "dashboard", label: "Dashboard", icon: Activity },
  { id: "nodes", label: "Nodes", icon: Laptop },
  { id: "logs", label: "Logs", icon: Terminal },
  { id: "workspace", label: "Workspace", icon: FolderOpen },
  { id: "memory", label: "Memory", icon: FileText },
  { id: "sessions", label: "Sessions", icon: ShieldCheck },
]

const DEFAULT_GATEWAY_URL = process.env.NEXT_PUBLIC_AG3NT_GATEWAY_URL || "http://127.0.0.1:18789"

const LOG_LEVEL_COLORS: Record<string, { stripe: string; badge: string }> = {
  debug: { stripe: "bg-zinc-500", badge: "text-text-muted" },
  info: { stripe: "bg-emerald-400", badge: "text-emerald-400" },
  warn: { stripe: "bg-amber-400", badge: "text-amber-400" },
  error: { stripe: "bg-red-400", badge: "text-red-400" },
}

function toWsUrl(httpUrl: string): string {
  return httpUrl.replace(/^http/i, "ws").replace(/\/+$/, "") + "/ws?debug=true"
}

async function fetchGatewayJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`/api/ag3nt/gateway/${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    cache: "no-store",
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) {
    const msg = (data && (data.error || data.detail)) || res.statusText || "Gateway request failed"
    throw new Error(String(msg))
  }
  return data as T
}

type GatewayLog = {
  id?: string
  timestamp?: string
  level?: "debug" | "info" | "warn" | "error"
  source?: string
  message?: string
  type?: string
}

export function Ag3ntControlPanelModule({ instanceId, agentEnabled = true, className }: ModuleInstanceProps) {
  const [view, setView] = useState<ViewId>("dashboard")
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)

  const { updateContext, sendEvent, onCommand } = useAgentConnection({
    instanceId: instanceId || `ag3nt-control-panel-${Date.now()}`,
    moduleType: "ag3nt-control-panel",
    autoRegister: agentEnabled,
    initialContext: { view, state: { isLoading: true, error: null } },
  })

  const { setSelectedModel: setChatModel } = useChat()

  // ---------------------------------------------------------------------------
  // Shared helpers
  // ---------------------------------------------------------------------------

  const setViewSafe = useCallback(
    (next: ViewId) => {
      setView(next)
      if (agentEnabled) {
        updateContext({ view: next, lastUpdated: Date.now() } as any)
        sendEvent("view-changed", { view: next })
      }
    },
    [agentEnabled, sendEvent, updateContext]
  )

  const refreshCurrent = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      await load(view)
      if (agentEnabled) sendEvent("refreshed", { view })
    } catch (e: any) {
      setError(e?.message || "Failed to refresh")
    } finally {
      setIsLoading(false)
    }
  }, [agentEnabled, sendEvent, view])

  // Agent commands
  useEffect(() => {
    const sub1 = onCommand("refresh", async () => refreshCurrent())
    const sub2 = onCommand("setView", async (params: { view?: string }) => {
      const v = String(params?.view || "")
      if (VIEWS.some((x) => x.id === v)) setViewSafe(v as ViewId)
    })
    return () => {
      sub1.unsubscribe()
      sub2.unsubscribe()
    }
  }, [onCommand, refreshCurrent, setViewSafe])

  // ---------------------------------------------------------------------------
  // View state
  // ---------------------------------------------------------------------------

  const [dashboard, setDashboard] = useState<any>(null)
  const [modelOptions, setModelOptions] = useState<Record<string, any>>({})
  const [modelProvider, setModelProvider] = useState<string>("")
  const [modelName, setModelName] = useState<string>("")
  const [modelSaving, setModelSaving] = useState(false)
  const [agentWorker, setAgentWorker] = useState<any>(null)
  const [agentWorkerLoading, setAgentWorkerLoading] = useState(false)
  const [nodes, setNodes] = useState<any>(null)
  const [sessions, setSessions] = useState<any>(null)
  const [approveSessionId, setApproveSessionId] = useState<string>("")
  const [approveSessionCode, setApproveSessionCode] = useState<string>("")

  const [logs, setLogs] = useState<GatewayLog[]>([])
  const [logLevel, setLogLevel] = useState<"debug" | "info" | "warn" | "error">("info")
  const [logsLive, setLogsLive] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  const [workspaceTree, setWorkspaceTree] = useState<any>(null)
  const [workspaceSelected, setWorkspaceSelected] = useState<string | null>(null)
  const [workspaceContent, setWorkspaceContent] = useState<string>("")

  const [memoryFiles, setMemoryFiles] = useState<any[]>([])
  const [memorySelected, setMemorySelected] = useState<string | null>(null)
  const [memoryContent, setMemoryContent] = useState<string>("")
  const [memorySaving, setMemorySaving] = useState(false)

  const [confirmDialog, setConfirmDialog] = useState<{
    open: boolean
    title: string
    description: string
    onConfirm: () => void
  }>({ open: false, title: "", description: "", onConfirm: () => {} })

  const showConfirm = useCallback((title: string, description: string, onConfirm: () => void) => {
    setConfirmDialog({ open: true, title, description, onConfirm })
  }, [])

  const load = useCallback(async (v: ViewId) => {
    if (agentEnabled) {
      updateContext({ state: { isLoading: true, error: null } } as any)
    }

    if (v === "dashboard") {
      const [healthRes, statusRes, modelRes, agentRes] = await Promise.allSettled([
        fetchGatewayJson<any>("health"),
        fetchGatewayJson<any>("status"),
        fetchGatewayJson<any>("model/config"),
        fetchGatewayJson<any>("agent/health"),
      ])

      const health = healthRes.status === "fulfilled" ? healthRes.value : {}
      const status = statusRes.status === "fulfilled" ? statusRes.value : {}
      setDashboard({ health, status })

      if (modelRes.status === "fulfilled") {
        const provider = String(modelRes.value?.provider || "")
        const model = String(modelRes.value?.model || "")
        const options = (modelRes.value?.options && typeof modelRes.value.options === "object") ? modelRes.value.options : {}

        setModelOptions(options)
        setModelProvider(provider)

        const models = Array.isArray(options?.[provider]?.models) ? options[provider].models : []
        const hasModel = models.some((m: any) => String(m?.id) === model)
        setModelName(hasModel ? model : String(models?.[0]?.id || model || ""))
      } else {
        setModelOptions({})
        setModelProvider("")
        setModelName("")
      }

      if (agentRes.status === "fulfilled") {
        setAgentWorker(agentRes.value)
      } else {
        setAgentWorker(null)
      }
      return
    }

    if (v === "nodes") {
      const [nodesRes, approvedRes, pairingRes] = await Promise.all([
        fetchGatewayJson<any>("nodes"),
        fetchGatewayJson<any>("nodes/approved"),
        fetchGatewayJson<any>("nodes/pairing/active"),
      ])
      setNodes({ nodesRes, approvedRes, pairingRes })
      return
    }

    if (v === "logs") {
      const recent = await fetchGatewayJson<any>(`logs/recent?count=200&level=${logLevel}`)
      setLogs(Array.isArray(recent?.logs) ? recent.logs : [])
      return
    }

    if (v === "workspace") {
      const tree = await fetchGatewayJson<any>("workspace/files")
      setWorkspaceTree(tree)
      return
    }

    if (v === "memory") {
      const files = await fetchGatewayJson<any>("memory/files")
      setMemoryFiles(Array.isArray(files?.files) ? files.files : [])
      return
    }

    if (v === "sessions") {
      const [all, pending] = await Promise.all([
        fetchGatewayJson<any>("sessions"),
        fetchGatewayJson<any>("sessions/pending"),
      ])
      setSessions({ all, pending })
      return
    }
  }, [agentEnabled, logLevel, updateContext])

  useEffect(() => {
    setIsLoading(true)
    setError(null)
    load(view)
      .catch((e: any) => setError(e?.message || "Failed to load"))
      .finally(() => setIsLoading(false))
  }, [load, view])

  useEffect(() => {
    if (agentEnabled) {
      updateContext({ view, state: { isLoading, error } } as any)
    }
  }, [agentEnabled, error, isLoading, updateContext, view])

  // ---------------------------------------------------------------------------
  // Logs live stream
  // ---------------------------------------------------------------------------

  useEffect(() => {
    if (!logsLive || view !== "logs") return

    const wsUrl = toWsUrl(DEFAULT_GATEWAY_URL)
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(String(ev.data))
        if (msg?.type === "log") {
          setLogs((prev) => [msg as GatewayLog, ...prev].slice(0, 500))
        }
      } catch {
        // ignore
      }
    }

    ws.onerror = () => {
      // keep existing logs; surface error via badge
    }

    return () => {
      try {
        ws.close()
      } catch {
        // ignore
      }
      wsRef.current = null
    }
  }, [logsLive, view])

  // ---------------------------------------------------------------------------
  // Actions (Nodes / Scheduler / Skills / Memory / Logs)
  // ---------------------------------------------------------------------------

  const launchTui = useCallback(async () => {
    setActionMessage(null)
    try {
      await fetchGatewayJson<any>("tui/launch", { method: "POST" })
      setActionMessage("TUI launched")
    } catch (e: any) {
      setActionMessage(e?.message ? `TUI launch failed: ${e.message}` : "TUI launch failed")
    }
  }, [])

  const restartAgentWorker = useCallback(async () => {
    setActionMessage(null)
    try {
      await fetchGatewayJson<any>("agent/restart", { method: "POST" })
      setActionMessage("Agent restart initiated")
    } catch (e: any) {
      setActionMessage(e?.message ? `Restart failed: ${e.message}` : "Restart failed")
    }
  }, [])

  const checkAgentStatus = useCallback(async () => {
    setAgentWorkerLoading(true)
    try {
      const health = await fetchGatewayJson<any>("agent/health")
      setAgentWorker(health)
    } catch (e: any) {
      setActionMessage(e?.message ? `Agent status check failed: ${e.message}` : "Agent status check failed")
    } finally {
      setAgentWorkerLoading(false)
    }
  }, [])

  const saveModelConfig = useCallback(async () => {
    if (!modelProvider || !modelName) return
    setModelSaving(true)
    setActionMessage(null)
    try {
      await fetchGatewayJson<any>("model/config", {
        method: "POST",
        body: JSON.stringify({ provider: modelProvider, model: modelName }),
      })
      setActionMessage(`Model saved: ${modelProvider}/${modelName}`)

      // Sync to chat bar if the saved model exists in AVAILABLE_MODELS
      const chatModelExists = AVAILABLE_MODELS.some((m) => m.id === modelName)
      if (chatModelExists) {
        setChatModel(modelName)
      }

      await refreshCurrent()
    } catch (e: any) {
      setActionMessage(e?.message ? `Model save failed: ${e.message}` : "Model save failed")
    } finally {
      setModelSaving(false)
    }
  }, [modelName, modelProvider, refreshCurrent, setChatModel])

  const generatePairingCode = useCallback(async () => {
    await fetchGatewayJson<any>("nodes/pairing/generate", { method: "POST" })
    await refreshCurrent()
  }, [refreshCurrent])

  const revokeNodeApproval = useCallback(
    async (nodeId: string) => {
      await fetchGatewayJson<any>(`nodes/${encodeURIComponent(nodeId)}/approval`, { method: "DELETE" })
      await refreshCurrent()
    },
    [refreshCurrent]
  )

  const clearLogs = useCallback(async () => {
    await fetchGatewayJson<any>("logs/clear", { method: "POST" })
    await refreshCurrent()
  }, [refreshCurrent])

  const openWorkspaceFile = useCallback(async (filePath: string) => {
    setWorkspaceSelected(filePath)
    const res = await fetchGatewayJson<any>(`workspace/file?path=${encodeURIComponent(filePath)}`)
    setWorkspaceContent(String(res?.content || ""))
  }, [])

  const openMemoryFile = useCallback(async (filePath: string) => {
    setMemorySelected(filePath)
    const res = await fetchGatewayJson<any>(`memory/file?path=${encodeURIComponent(filePath)}`)
    setMemoryContent(String(res?.content || ""))
  }, [])

  const saveMemoryFile = useCallback(async () => {
    if (!memorySelected) return
    setMemorySaving(true)
    try {
      await fetchGatewayJson<any>("memory/file", {
        method: "POST",
        body: JSON.stringify({ path: memorySelected, content: memoryContent }),
      })
    } finally {
      setMemorySaving(false)
    }
  }, [memoryContent, memorySelected])

  const approveSession = useCallback(
    async (sessionId: string, code?: string) => {
      await fetchGatewayJson<any>(`sessions/${encodeURIComponent(sessionId)}/approve`, {
        method: "POST",
        body: JSON.stringify(code ? { code } : {}),
      })
      await refreshCurrent()
    },
    [refreshCurrent]
  )

  const deleteSession = useCallback(
    async (sessionId: string) => {
      setActionMessage(null)
      try {
        await fetchGatewayJson<any>(`sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" })
        setActionMessage(`Session deleted: ${sessionId}`)
        await refreshCurrent()
      } catch (e: any) {
        setActionMessage(e?.message ? `Delete failed: ${e.message}` : "Delete failed")
      }
    },
    [refreshCurrent]
  )

  const clearAllSessions = useCallback(async () => {
    setActionMessage(null)
    try {
      const res = await fetchGatewayJson<any>("sessions/clear", { method: "POST" })
      const cleared = res?.cleared
      setActionMessage(typeof cleared === "number" ? `Cleared ${cleared} sessions` : "Sessions cleared")
      await refreshCurrent()
    } catch (e: any) {
      setActionMessage(e?.message ? `Clear failed: ${e.message}` : "Clear failed")
    }
  }, [refreshCurrent])

  // ---------------------------------------------------------------------------
  // View renderers
  // ---------------------------------------------------------------------------

  const viewTabs = (
    <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border bg-surface">
      <div className="flex items-center gap-0.5 rounded-lg bg-surface-elevated p-1">
        {VIEWS.map((v) => {
          const Icon = v.icon
          const active = v.id === view
          return (
            <button
              key={v.id}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all duration-150",
                active
                  ? "bg-surface text-text-primary shadow-sm"
                  : "text-text-muted hover:text-text-secondary"
              )}
              onClick={() => setViewSafe(v.id)}
              title={v.label}
            >
              <Icon className="h-3.5 w-3.5" />
              {v.label}
            </button>
          )
        })}
      </div>
      <div className="ml-auto flex items-center gap-2">
        {view === "dashboard" && (
          <>
            <Button variant="outline" size="sm" className="h-7 text-xs" onClick={launchTui}>
              Launch TUI
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              onClick={() => showConfirm(
                "Restart Agent Worker",
                "This may open a new terminal window. Are you sure you want to restart?",
                restartAgentWorker
              )}
            >
              Restart Agent
            </Button>
          </>
        )}
        <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={refreshCurrent} title="Refresh">
          <RefreshCw className={cn("h-3.5 w-3.5", isLoading && "animate-spin")} />
        </Button>
      </div>
    </div>
  )

  const renderDashboard = () => {
    if (!dashboard) return <EmptyModuleState icon={Signal} title="No data" description="Gateway status not loaded" />
    const health = dashboard.health || {}
    const status = dashboard.status || {}
    const channels = Array.isArray(health.channels) ? health.channels : []
    const providerEntries = Object.entries(modelOptions || {})
    const providerLabel =
      modelProvider && modelOptions?.[modelProvider]?.name
        ? String(modelOptions[modelProvider].name)
        : modelProvider
    const models = Array.isArray(modelOptions?.[modelProvider]?.models) ? modelOptions[modelProvider].models : []
    const modelLabel = modelName
      ? String(models.find((m: any) => String(m?.id) === modelName)?.name || modelName)
      : ""
    const agentStatus = String(agentWorker?.status || "")

    return (
      <div className="p-4 space-y-5 overflow-y-auto">
        {actionMessage && (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-surface-elevated border border-border text-sm text-text-secondary">
            <Zap className="h-3.5 w-3.5 text-status-info shrink-0" />
            {actionMessage}
          </div>
        )}

        <ControlPanelSection icon={Signal} label="Gateway Status" iconColor="text-status-success">
          <div className="flex items-center gap-3">
            <Badge variant="outline" className={cn("gap-2", health.ok ? "text-status-success border-status-success/30" : "text-text-muted")}>
              <span className={cn("w-1.5 h-1.5 rounded-full", health.ok ? "bg-status-success" : "bg-zinc-500")} />
              {health.ok ? "Online" : "Unknown"}
            </Badge>
            <span className="text-xs text-text-muted">{String(health.name || "ag3nt-gateway")}</span>
          </div>
        </ControlPanelSection>

        <ControlPanelSection icon={Hash} label="Overview" iconColor="text-status-info">
          <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center gap-3 p-3 rounded-xl border border-border bg-surface-elevated">
              <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-surface">
                <ShieldCheck className="w-4 h-4 text-blue-400" />
              </div>
              <div>
                <div className="text-xs text-text-muted">Sessions</div>
                <div className="text-lg font-semibold text-text-primary tabular-nums">{String(status.sessions ?? health.sessions ?? 0)}</div>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-xl border border-border bg-surface-elevated">
              <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-surface">
                <Clock className="w-4 h-4 text-amber-400" />
              </div>
              <div>
                <div className="text-xs text-text-muted">Scheduler Jobs</div>
                <div className="text-lg font-semibold text-text-primary tabular-nums">
                  {String(status.scheduler?.jobCount ?? health.scheduler?.jobCount ?? 0)}
                </div>
              </div>
            </div>
          </div>
        </ControlPanelSection>

        <ControlPanelSection icon={Brain} label="Model Configuration" iconColor="text-purple-400">
          <div className="p-3 rounded-xl border border-border bg-surface-elevated space-y-3">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm text-text-primary">Active Model</span>
              <Badge variant="outline">{providerLabel && modelLabel ? `${providerLabel} / ${modelLabel}` : "Not configured"}</Badge>
            </div>
            {providerEntries.length === 0 ? (
              <div className="text-xs text-text-muted">Model config unavailable</div>
            ) : (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <select
                    value={modelProvider}
                    onChange={(e) => {
                      const nextProvider = e.target.value
                      const nextModels = Array.isArray(modelOptions?.[nextProvider]?.models)
                        ? modelOptions[nextProvider].models
                        : []
                      const stillValid = nextModels.some((m: any) => String(m?.id) === modelName)
                      setModelProvider(nextProvider)
                      setModelName(stillValid ? modelName : String(nextModels?.[0]?.id || ""))
                    }}
                    className="h-8 rounded-lg border border-border bg-surface px-3 text-xs text-text-secondary focus:outline-none focus:ring-1 focus:ring-status-info/30"
                  >
                    {providerEntries.map(([key, val]: any) => (
                      <option key={String(key)} value={String(key)}>
                        {String(val?.name || key)}
                      </option>
                    ))}
                  </select>
                  <select
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                    className="h-8 rounded-lg border border-border bg-surface px-3 text-xs text-text-secondary focus:outline-none focus:ring-1 focus:ring-status-info/30"
                  >
                    {models.map((m: any) => (
                      <option key={String(m?.id)} value={String(m?.id)}>
                        {String(m?.name || m?.id)}
                      </option>
                    ))}
                  </select>
                </div>
                <Button size="sm" className="h-7 text-xs" onClick={saveModelConfig} disabled={!modelProvider || !modelName || modelSaving}>
                  {modelSaving ? "Saving..." : "Save Model"}
                </Button>
              </div>
            )}
          </div>
        </ControlPanelSection>

        <ControlPanelSection icon={Server} label="Agent Worker" iconColor="text-emerald-400">
          <div className="p-3 rounded-xl border border-border bg-surface-elevated space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm text-text-primary">Status</span>
              <Badge variant="outline" className={cn(
                agentStatus === "running" ? "text-status-success border-status-success/30" :
                agentStatus === "error" ? "text-status-error border-status-error/30" :
                "text-text-muted"
              )}>
                {agentStatus || "unknown"}
              </Badge>
            </div>
            {agentWorker?.message && (
              <div className="text-xs text-text-muted">{String(agentWorker.message)}</div>
            )}
            <div className="flex items-center gap-2 pt-1">
              <Button size="sm" variant="outline" className="h-7 text-xs" onClick={checkAgentStatus} disabled={agentWorkerLoading}>
                {agentWorkerLoading ? "Checking..." : "Check Status"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                className="h-7 text-xs"
                onClick={() => showConfirm(
                  "Restart Agent Worker",
                  "This may open a new terminal window. Are you sure you want to restart?",
                  restartAgentWorker
                )}
              >
                Restart
              </Button>
            </div>
          </div>
        </ControlPanelSection>

        <ControlPanelSection icon={Radio} label="Channels" iconColor="text-cyan-400">
          {channels.length === 0 ? (
            <div className="text-xs text-text-muted py-2">No channels reported</div>
          ) : (
            <div className="space-y-1.5">
              {channels.map((c: any) => (
                <div key={String(c.id)} className="flex items-center justify-between p-2.5 rounded-lg border border-border bg-surface-elevated">
                  <span className="text-sm text-text-primary">{String(c.id)}</span>
                  <span className="flex items-center gap-1.5">
                    <span className={cn("w-1.5 h-1.5 rounded-full", c.connected ? "bg-status-success" : "bg-status-error")} />
                    <span className={cn("text-[10px]", c.connected ? "text-status-success" : "text-status-error")}>
                      {c.connected ? "connected" : "disconnected"}
                    </span>
                  </span>
                </div>
              ))}
            </div>
          )}
        </ControlPanelSection>
      </div>
    )
  }

  const renderNodes = () => {
    const pairingCode = nodes?.pairingRes?.code ?? null
    const nodeList = Array.isArray(nodes?.nodesRes?.nodes) ? nodes.nodesRes.nodes : []
    const approved = Array.isArray(nodes?.approvedRes?.nodes) ? nodes.approvedRes.nodes : []

    return (
      <div className="p-4 space-y-5 overflow-y-auto">
        <ControlPanelSection icon={Hash} label="Pairing" iconColor="text-amber-400" action={
          <Button size="sm" className="h-7 text-xs" onClick={generatePairingCode}>Generate Code</Button>
        }>
          <div className="p-3 rounded-xl border border-border bg-surface-elevated">
            <div className="flex items-center gap-2">
              <span className="text-xs text-text-muted">Active Code:</span>
              {pairingCode ? (
                <code className="text-sm font-mono text-text-primary bg-surface px-2 py-0.5 rounded">{String(pairingCode)}</code>
              ) : (
                <span className="text-xs text-text-muted">None active</span>
              )}
            </div>
          </div>
        </ControlPanelSection>

        <ControlPanelSection icon={Laptop} label="Connected Nodes" iconColor="text-status-info">
          {nodeList.length === 0 ? (
            <div className="text-xs text-text-muted py-2">No nodes connected</div>
          ) : (
            <div className="space-y-1.5">
              {nodeList.map((n: any) => (
                <div key={String(n.id)} className="p-3 rounded-xl border border-border bg-surface-elevated">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium text-text-primary">{String(n.name || n.id)}</span>
                    <Badge variant="outline">{String(n.status || "unknown")}</Badge>
                  </div>
                  {Array.isArray(n.capabilities) && n.capabilities.length > 0 && (
                    <div className="flex items-center gap-1 mt-2 flex-wrap">
                      {n.capabilities.slice(0, 8).map((cap: string) => (
                        <span key={cap} className="text-[10px] px-2 py-0.5 rounded-full bg-surface border border-border text-text-muted">
                          {String(cap)}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </ControlPanelSection>

        <ControlPanelSection icon={ShieldCheck} label="Approved Nodes" iconColor="text-emerald-400">
          {approved.length === 0 ? (
            <div className="text-xs text-text-muted py-2">No approved nodes</div>
          ) : (
            <div className="space-y-1.5">
              {approved.map((n: any) => (
                <div key={String(n.nodeId)} className="flex items-center justify-between p-3 rounded-xl border border-border bg-surface-elevated">
                  <span className="text-sm text-text-primary">{String(n.name || n.nodeId)}</span>
                  <Button variant="destructive" size="sm" className="h-7 text-xs" onClick={() => revokeNodeApproval(String(n.nodeId))}>
                    Revoke
                  </Button>
                </div>
              ))}
            </div>
          )}
        </ControlPanelSection>
      </div>
    )
  }

  const renderLogs = () => {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border bg-surface-elevated shrink-0">
          <select
            value={logLevel}
            onChange={(e) => setLogLevel(e.target.value as any)}
            className="h-8 rounded-lg border border-border bg-surface px-3 text-xs text-text-secondary focus:outline-none focus:ring-1 focus:ring-status-info/30"
          >
            <option value="debug">Debug+</option>
            <option value="info">Info+</option>
            <option value="warn">Warn+</option>
            <option value="error">Error only</option>
          </select>

          <div className="flex items-center gap-2 text-xs text-text-muted">
            <Switch checked={logsLive} onCheckedChange={setLogsLive} />
            <span>Live</span>
            {logsLive && <span className="w-1.5 h-1.5 rounded-full bg-status-success animate-pulse" />}
          </div>

          <div className="ml-auto flex items-center gap-2">
            <span className="text-xs text-text-muted tabular-nums">{logs.length} entries</span>
            <Button variant="outline" size="sm" className="h-7 text-xs" onClick={clearLogs}>
              Clear
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          {logs.length === 0 ? (
            <EmptyModuleState icon={Terminal} title="No logs" description="No recent logs available" />
          ) : (
            <div className="space-y-1">
              {logs.slice(0, 200).map((l, idx) => {
                const level = String(l.level || "info")
                const colors = LOG_LEVEL_COLORS[level] || LOG_LEVEL_COLORS.info
                return (
                  <div
                    key={l.id || `${idx}`}
                    className="flex items-stretch gap-0 rounded-lg border border-border bg-surface-elevated overflow-hidden"
                  >
                    <div className={cn("w-0.5 shrink-0", colors.stripe)} />
                    <div className="flex-1 px-3 py-2 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-[10px] text-text-muted truncate font-mono">
                          {String(l.timestamp || "")} · {String(l.source || "Gateway")}
                        </span>
                        <span className={cn("text-[10px] font-medium uppercase shrink-0", colors.badge)}>
                          {level}
                        </span>
                      </div>
                      <div className="mt-0.5 text-xs text-text-primary font-mono whitespace-pre-wrap break-all">
                        {String(l.message || "")}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    )
  }

  function renderTree(nodes: any[]) {
    return (
      <div className="space-y-0.5">
        {nodes.map((n) => {
          const isDir = n.type === "directory"
          return (
            <div key={String(n.path)}>
              <button
                className={cn(
                  "w-full text-left flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs transition-colors",
                  workspaceSelected === n.path
                    ? "bg-surface-elevated text-text-primary"
                    : "text-text-secondary hover:bg-interactive-hover hover:text-text-primary"
                )}
                onClick={() => {
                  if (!isDir) openWorkspaceFile(String(n.path))
                }}
              >
                {isDir ? (
                  <FolderOpen className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                ) : (
                  <FileText className="w-3.5 h-3.5 text-text-muted shrink-0" />
                )}
                <span className="truncate">{String(n.name)}</span>
              </button>
              {isDir && Array.isArray(n.children) && n.children.length > 0 && (
                <div className="pl-4">{renderTree(n.children)}</div>
              )}
            </div>
          )
        })}
      </div>
    )
  }

  const renderWorkspace = () => {
    const files = Array.isArray(workspaceTree?.files) ? workspaceTree.files : []

    return (
      <div className="flex h-full overflow-hidden">
        <div className="w-64 shrink-0 border-r border-border overflow-y-auto p-3 bg-surface">
          <div className="flex items-center gap-2 px-2.5 pb-2 mb-2 border-b border-border/30">
            <FolderOpen className="w-3.5 h-3.5 text-text-muted" />
            <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">Files</span>
          </div>
          {files.length === 0 ? (
            <div className="text-xs text-text-muted px-2.5 py-4">Workspace is empty</div>
          ) : (
            renderTree(files)
          )}
        </div>
        <div className="flex-1 overflow-hidden flex flex-col">
          {!workspaceSelected ? (
            <EmptyModuleState icon={FolderOpen} title="Select a file" description="Choose a workspace file to view" />
          ) : (
            <>
              <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-surface-elevated shrink-0">
                <FileText className="w-3.5 h-3.5 text-text-muted shrink-0" />
                <span className="text-xs text-text-muted font-mono truncate">{workspaceSelected}</span>
              </div>
              <div className="flex-1 overflow-auto p-4">
                <pre className="text-xs text-text-primary font-mono whitespace-pre-wrap">{workspaceContent}</pre>
              </div>
            </>
          )}
        </div>
      </div>
    )
  }

  const renderMemory = () => {
    return (
      <div className="flex h-full overflow-hidden">
        <div className="w-64 shrink-0 border-r border-border overflow-y-auto p-3 bg-surface">
          <div className="flex items-center gap-2 px-2.5 pb-2 mb-2 border-b border-border/30">
            <FileText className="w-3.5 h-3.5 text-text-muted" />
            <span className="text-xs font-semibold uppercase tracking-wider text-text-muted">Memory Files</span>
          </div>
          {memoryFiles.length === 0 ? (
            <div className="text-xs text-text-muted px-2.5 py-4">No memory files</div>
          ) : (
            <div className="space-y-0.5">
              {memoryFiles.map((f) => (
                <button
                  key={String(f.path)}
                  className={cn(
                    "w-full text-left flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs transition-colors",
                    memorySelected === f.path
                      ? "bg-surface-elevated text-text-primary"
                      : "text-text-secondary hover:bg-interactive-hover hover:text-text-primary"
                  )}
                  onClick={() => openMemoryFile(String(f.path))}
                >
                  <FileText className="w-3.5 h-3.5 text-text-muted shrink-0" />
                  <span className="truncate">{String(f.name || f.path)}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <div className="flex-1 overflow-hidden flex flex-col">
          {!memorySelected ? (
            <EmptyModuleState icon={FileText} title="Select a file" description="Choose a memory file to view or edit" />
          ) : (
            <>
              <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-surface-elevated shrink-0">
                <FileText className="w-3.5 h-3.5 text-text-muted shrink-0" />
                <span className="text-xs text-text-muted font-mono truncate flex-1">{memorySelected}</span>
                <Button size="sm" className="h-7 text-xs" onClick={saveMemoryFile} disabled={memorySaving}>
                  {memorySaving ? "Saving..." : "Save"}
                </Button>
              </div>
              <div className="flex-1 overflow-hidden p-3">
                <Textarea
                  value={memoryContent}
                  onChange={(e) => setMemoryContent(e.target.value)}
                  className="h-full font-mono text-xs resize-none bg-surface border-border"
                />
              </div>
            </>
          )}
        </div>
      </div>
    )
  }

  const renderSessions = () => {
    const all = Array.isArray(sessions?.all?.sessions) ? sessions.all.sessions : []
    const pending = Array.isArray(sessions?.pending?.sessions) ? sessions.pending.sessions : []

    return (
      <div className="p-4 space-y-5 overflow-y-auto">
        <ControlPanelSection icon={Clock} label="Pending Sessions" iconColor="text-amber-400">
          {pending.length === 0 ? (
            <div className="text-xs text-text-muted py-2">No pending sessions</div>
          ) : (
            <div className="space-y-1.5">
              {pending.map((s: any) => (
                <div key={String(s.id)} className="flex items-center justify-between p-3 rounded-xl border border-border bg-surface-elevated">
                  <span className="text-sm font-mono text-text-primary">{String(s.id)}</span>
                  <Badge variant="outline" className="text-amber-400 border-amber-400/30">
                    code: {String(s.pairingCode || "\u2014")}
                  </Badge>
                </div>
              ))}
            </div>
          )}
        </ControlPanelSection>

        <ControlPanelSection icon={Key} label="Approve Session" iconColor="text-purple-400">
          <div className="flex items-center gap-2 p-3 rounded-xl border border-border bg-surface-elevated">
            <Input
              value={approveSessionId}
              onChange={(e) => setApproveSessionId(e.target.value)}
              placeholder="Session ID"
              className="max-w-[200px] h-8 text-xs bg-surface border-border"
            />
            <Input
              value={approveSessionCode}
              onChange={(e) => setApproveSessionCode(e.target.value)}
              placeholder="Code (optional)"
              className="max-w-[150px] h-8 text-xs bg-surface border-border"
            />
            <Button
              size="sm"
              className="h-8 text-xs"
              onClick={() =>
                approveSessionId &&
                approveSession(approveSessionId, approveSessionCode || undefined)
              }
            >
              Approve
            </Button>
          </div>
        </ControlPanelSection>

        <ControlPanelSection
          icon={ShieldCheck}
          label="Active Sessions"
          iconColor="text-emerald-400"
          action={
            all.length > 0 ? (
              <Button
                size="sm"
                variant="destructive"
                className="h-7 text-xs"
                onClick={() => showConfirm(
                  "Clear All Sessions",
                  "This will remove all active sessions. This cannot be undone.",
                  clearAllSessions
                )}
              >
                Clear All
              </Button>
            ) : undefined
          }
        >
          {all.length === 0 ? (
            <div className="text-xs text-text-muted py-2">No active sessions</div>
          ) : (
            <div className="space-y-1.5">
              {all.map((s: any) => (
                <div key={String(s.id)} className="p-3 rounded-xl border border-border bg-surface-elevated">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-mono text-text-primary">{String(s.id)}</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className={cn(
                        s.paired ? "text-status-success border-status-success/30" : "text-text-muted"
                      )}>
                        {s.paired ? "paired" : "unpaired"}
                      </Badge>
                      <Button
                        variant="destructive"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => showConfirm(
                          "Delete Session",
                          `Delete session "${s.id}"? This cannot be undone.`,
                          () => deleteSession(String(s.id))
                        )}
                      >
                        Delete
                      </Button>
                    </div>
                  </div>
                  {(s.channelType || s.userId) && (
                    <div className="text-[10px] text-text-muted mt-1.5">
                      {[s.channelType, s.userId].filter(Boolean).map(String).join(" \u00b7 ")}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </ControlPanelSection>
      </div>
    )
  }

  const body = (() => {
    if (isLoading) return <LoadingModuleState message="Loading AG3NT Gateway…" />
    if (error) return <ErrorModuleState error={error} onRetry={refreshCurrent} />

    switch (view) {
      case "dashboard":
        return renderDashboard()
      case "nodes":
        return renderNodes()
      case "logs":
        return renderLogs()
      case "workspace":
        return renderWorkspace()
      case "memory":
        return renderMemory()
      case "sessions":
        return renderSessions()
      default:
        return null
    }
  })()

  return (
    <ModuleContainer config={ag3ntControlPanelModuleConfig} className={cn("flex flex-col h-full", className)} showHeader={false}>
      {viewTabs}
      <div className="flex-1 overflow-hidden bg-surface-secondary">{body}</div>
      <AlertDialog open={confirmDialog.open} onOpenChange={(open) => !open && setConfirmDialog(prev => ({ ...prev, open: false }))}>
        <AlertDialogContent className="bg-surface border-border">
          <AlertDialogHeader>
            <AlertDialogTitle className="text-text-primary">{confirmDialog.title}</AlertDialogTitle>
            <AlertDialogDescription className="text-text-muted">{confirmDialog.description}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel className="border-border text-text-secondary hover:bg-interactive-hover hover:text-text-primary">Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                confirmDialog.onConfirm()
                setConfirmDialog(prev => ({ ...prev, open: false }))
              }}
            >
              Confirm
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </ModuleContainer>
  )
}

export default Ag3ntControlPanelModule
