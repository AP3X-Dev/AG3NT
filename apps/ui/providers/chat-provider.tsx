"use client";

import {
  createContext,
  useContext,
  useMemo,
  useReducer,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type {
  Message,
  ChatSession,
  CLIContent,
  ApprovalRequest,
  FileAttachment,
  ThreadInfo,
} from "@/types/types";
import { getLanguageFromPath } from "@/lib/cli/file-operations";
import {
  emitBrowserSession,
  isBrowserSessionTool,
} from "@/lib/browser-session-events";
import { getTabContextManager } from "@/lib/tab-context-manager";
import { buildAgentContext } from "@/lib/agent-context";
import { AVAILABLE_MODELS } from "@/components/features/chat/model-selector";

// Debug logging — only enabled in development to reduce console noise in production
const debugLog: (...args: unknown[]) => void =
  process.env.NODE_ENV === "development"
    ? (...args: unknown[]) => { console.log(...args); }
    : () => {};

// Default model - Kimi K2.5 via OpenRouter (works without Anthropic credits)
const DEFAULT_MODEL = "moonshotai/kimi-k2.5";

// Valid model IDs for validation
const VALID_MODEL_IDS: Set<string> = new Set(AVAILABLE_MODELS.map(m => m.id));

/**
 * Derive the gateway provider key from a chat-bar model ID.
 * - IDs with "/" are OpenRouter models (e.g. "anthropic/claude-sonnet-4.5")
 * - "kimi-for-coding" → "kimi" (direct Kimi API)
 * - "claude-opus-4-6" → "anthropic" (direct Anthropic API)
 * - Fallback → "openrouter"
 */
function deriveProvider(modelId: string): string {
  if (modelId.includes("/")) return "openrouter";
  if (modelId === "kimi-for-coding") return "kimi";
  if (modelId.startsWith("claude-")) return "anthropic";
  return "openrouter";
}

/**
 * Format a user-friendly status message for tool execution.
 * For subagent tasks, shows the agent type (e.g., "Deep Research Agent working...")
 * For regular tools, shows a user-friendly message.
 */
function formatStatusMessage(
  toolName: string,
  args?: Record<string, any>,
): string {
  if (toolName === "task") {
    if (args?.subagent_type) {
      const subagentType = String(args.subagent_type);
      // Format: "deep-research" -> "Deep Research Agent"
      const formatted = subagentType
        .split(/[_-]/)
        .map(
          (word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase(),
        )
        .join(" ");
      return `${formatted} Agent working...`;
    }
    // No subagent_type but we have a description
    if (args?.description) {
      const desc = String(args.description);
      const preview = desc.length > 40 ? desc.slice(0, 40) + "..." : desc;
      return `Agent working: ${preview}`;
    }
    return "Subagent working...";
  }

  // User-friendly names for common tools
  const friendlyNames: Record<string, string> = {
    web_search: "Searching the web...",
    read_file: "Reading file...",
    write_file: "Writing file...",
    edit_file: "Editing file...",
    list_directory: "Listing directory...",
    run_shell_command: "Running command...",
    shell: "Running command...",
    execute: "Executing code...",
    deep_research: "Deep Research Agent working...",
  };

  if (friendlyNames[toolName]) {
    return friendlyNames[toolName];
  }

  // Fallback: format the tool name nicely
  const formatted = toolName
    .split(/[_-]/)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
  return `${formatted}...`;
}

// Chat Context Types
interface ChatState {
  currentSession: ChatSession | null;
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  // Agent session state
  threadId: string | null;
  assistantId: string;
  autoApprove: boolean;
  // Streaming status
  status: string | null;
  statusMessage: string | null;
  // Track the current streaming assistant message ID
  streamingMessageId: string | null;
  // Selected model
  selectedModel: string;
  // Plan mode
  planMode: boolean;
  // Thread history
  threads: ThreadInfo[];
  threadsLoading: boolean;
  showThreadHistory: boolean;
  // Plan execution state
  planStatus: "idle" | "planning" | "reviewing" | "executing" | "complete";
  planProgress: {
    currentStep: number;
    totalSteps: number;
    description: string;
  } | null;
}

interface ChatContextType extends ChatState {
  sendMessage: (
    content: string,
    attachments?: FileAttachment[],
  ) => Promise<void>;
  decideApproval: (
    interruptId: string,
    decision: "approve" | "reject" | "auto_approve_all",
  ) => Promise<void>;
  clearMessages: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setAutoApprove: (enabled: boolean) => void;
  setSelectedModel: (model: string) => void;
  setPlanMode: (enabled: boolean) => void;
  stopAgent: () => void;
  clearCachesAndRestart: () => Promise<any>;
  status: string | null;
  statusMessage: string | null;
  // Thread management
  loadThreads: () => Promise<void>;
  selectThread: (threadId: string) => Promise<void>;
  deleteThread: (threadId: string) => Promise<void>;
  createNewThread: () => void;
  toggleThreadHistory: () => void;
  // Plan approval
  decidePlanApproval: (
    decision: "approve" | "reject",
    reason?: string,
  ) => Promise<void>;
}

// Actions
type ChatAction =
  | { type: "SET_MESSAGES"; payload: Message[] }
  | { type: "ADD_MESSAGE"; payload: Message }
  | {
      type: "UPDATE_MESSAGE";
      payload: { id: string; updates: Partial<Message> };
    }
  | { type: "APPEND_TO_MESSAGE"; payload: { id: string; text: string } }
  | {
      type: "APPEND_CLI_CONTENT";
      payload: { id: string; cliContent: CLIContent };
    }
  | {
      type: "UPDATE_CLI_CONTENT";
      payload: {
        messageId: string;
        toolCallId: string;
        updates: Partial<CLIContent>;
      };
    }
  | { type: "REMOVE_MESSAGE"; payload: string }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "CLEAR_MESSAGES" }
  | { type: "SET_SESSION"; payload: ChatSession }
  | { type: "SET_THREAD_ID"; payload: string | null }
  | { type: "SET_AUTO_APPROVE"; payload: boolean }
  | { type: "SET_ASSISTANT_ID"; payload: string }
  | {
      type: "SET_STATUS";
      payload: { status: string | null; message: string | null };
    }
  | { type: "SET_STREAMING_MESSAGE_ID"; payload: string | null }
  | { type: "SET_SELECTED_MODEL"; payload: string }
  | { type: "SET_PLAN_MODE"; payload: boolean }
  | { type: "SET_THREADS"; payload: ThreadInfo[] }
  | { type: "SET_THREADS_LOADING"; payload: boolean }
  | { type: "SET_SHOW_THREAD_HISTORY"; payload: boolean }
  | { type: "REMOVE_THREAD"; payload: string }
  | {
      type: "SET_PLAN_STATUS";
      payload: "idle" | "planning" | "reviewing" | "executing" | "complete";
    }
  | {
      type: "SET_PLAN_PROGRESS";
      payload: {
        currentStep: number;
        totalSteps: number;
        description: string;
      } | null;
    };

// Initial state
const initialState: ChatState = {
  currentSession: null,
  messages: [],
  isLoading: false,
  error: null,
  threadId: null,
  assistantId: "agent",
  autoApprove: false,
  status: null,
  statusMessage: null,
  streamingMessageId: null,
  selectedModel: DEFAULT_MODEL,
  planMode: false,
  threads: [],
  threadsLoading: false,
  showThreadHistory: false,
  planStatus: "idle",
  planProgress: null,
};

// Reducer
function chatReducer(state: ChatState, action: ChatAction): ChatState {
  switch (action.type) {
    case "SET_MESSAGES":
      return { ...state, messages: action.payload };
    case "ADD_MESSAGE":
      return { ...state, messages: [...state.messages, action.payload] };
    case "UPDATE_MESSAGE":
      return {
        ...state,
        messages: state.messages.map((msg) =>
          msg.id === action.payload.id
            ? { ...msg, ...action.payload.updates }
            : msg,
        ),
      };
    case "APPEND_TO_MESSAGE":
      return {
        ...state,
        messages: state.messages.map((msg) => {
          if (msg.id !== action.payload.id) return msg;
          // Append text to the last text part in cliContent, or create one
          const cliContent = msg.cliContent || [];
          const lastPart = cliContent[cliContent.length - 1];
          if (lastPart && lastPart.type === "text") {
            // Append to existing text part
            return {
              ...msg,
              content: msg.content + action.payload.text,
              cliContent: [
                ...cliContent.slice(0, -1),
                {
                  ...lastPart,
                  content: lastPart.content + action.payload.text,
                },
              ],
            };
          } else {
            // Create new text part
            return {
              ...msg,
              content: msg.content + action.payload.text,
              cliContent: [
                ...cliContent,
                { type: "text" as const, content: action.payload.text },
              ],
            };
          }
        }),
      };
    case "APPEND_CLI_CONTENT":
      return {
        ...state,
        messages: state.messages.map((msg) =>
          msg.id === action.payload.id
            ? {
                ...msg,
                cliContent: [
                  ...(msg.cliContent || []),
                  action.payload.cliContent,
                ],
              }
            : msg,
        ),
      };
    case "UPDATE_CLI_CONTENT":
      return {
        ...state,
        messages: state.messages.map((msg) => {
          if (msg.id !== action.payload.messageId) return msg;
          const cliContent = (msg.cliContent || []).map((cli) => {
            // Match by title field which stores the toolCallId
            if (
              cli.type === "tool-call" &&
              cli.title === action.payload.toolCallId
            ) {
              return { ...cli, ...action.payload.updates };
            }
            return cli;
          });
          return { ...msg, cliContent };
        }),
      };
    case "REMOVE_MESSAGE":
      return {
        ...state,
        messages: state.messages.filter((msg) => msg.id !== action.payload),
      };
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "CLEAR_MESSAGES":
      return {
        ...state,
        messages: [],
        error: null,
        status: null,
        statusMessage: null,
      };
    case "SET_SESSION":
      return {
        ...state,
        currentSession: action.payload,
        messages: action.payload.messages,
      };
    case "SET_THREAD_ID":
      return { ...state, threadId: action.payload };
    case "SET_AUTO_APPROVE":
      return { ...state, autoApprove: action.payload };
    case "SET_ASSISTANT_ID":
      return { ...state, assistantId: action.payload };
    case "SET_STATUS":
      return {
        ...state,
        status: action.payload.status,
        statusMessage: action.payload.message,
      };
    case "SET_STREAMING_MESSAGE_ID":
      return { ...state, streamingMessageId: action.payload };
    case "SET_SELECTED_MODEL":
      return { ...state, selectedModel: action.payload };
    case "SET_PLAN_MODE":
      return { ...state, planMode: action.payload };
    case "SET_THREADS":
      return { ...state, threads: action.payload };
    case "SET_THREADS_LOADING":
      return { ...state, threadsLoading: action.payload };
    case "SET_SHOW_THREAD_HISTORY":
      return { ...state, showThreadHistory: action.payload };
    case "REMOVE_THREAD":
      return {
        ...state,
        threads: state.threads.filter((t) => t.threadId !== action.payload),
      };
    case "SET_PLAN_STATUS":
      return { ...state, planStatus: action.payload };
    case "SET_PLAN_PROGRESS":
      return { ...state, planProgress: action.payload };
    default:
      return state;
  }
}

// Context
const ChatContext = createContext<ChatContextType | undefined>(undefined);

// Provider
interface ChatProviderProps {
  children: ReactNode;
}

export function ChatProvider({ children }: ChatProviderProps) {
  const [state, dispatch] = useReducer(chatReducer, initialState);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Abort in-flight requests on unmount to prevent stale state updates
  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
    };
  }, []);

  // Ref to always have current messages (avoids stale closure issues)
  const messagesRef = useRef(state.messages);
  messagesRef.current = state.messages;

  // Persist thread id so reloads keep the same LangGraph thread.
  useEffect(() => {
    try {
      const saved = localStorage.getItem("ap3x.threadId");
      if (saved) dispatch({ type: "SET_THREAD_ID", payload: saved });
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      if (state.threadId) localStorage.setItem("ap3x.threadId", state.threadId);
    } catch {
      // ignore
    }
  }, [state.threadId]);

  // Persist autoApprove preference
  useEffect(() => {
    try {
      const saved = localStorage.getItem("ap3x.autoApprove");
      if (saved === "true")
        dispatch({ type: "SET_AUTO_APPROVE", payload: true });
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(
        "ap3x.autoApprove",
        state.autoApprove ? "true" : "false",
      );
    } catch {
      // ignore
    }
  }, [state.autoApprove]);

  // Persist selected model & sync with gateway on mount
  useEffect(() => {
    // First apply localStorage immediately (avoids flash of default model)
    try {
      const saved = localStorage.getItem("ap3x.selectedModel");
      if (saved && VALID_MODEL_IDS.has(saved)) {
        dispatch({ type: "SET_SELECTED_MODEL", payload: saved });
      } else if (saved) {
        console.warn(`[Chat] Invalid cached model "${saved}", using default`);
        localStorage.removeItem("ap3x.selectedModel");
      }
    } catch {
      // ignore
    }

    // Then fetch gateway config — gateway is source of truth when reachable
    fetch("/api/ag3nt/gateway/model/config", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data) => {
        const gatewayModel = data?.model as string | undefined;
        if (gatewayModel && VALID_MODEL_IDS.has(gatewayModel)) {
          debugLog("[Chat] Gateway model config:", gatewayModel);
          dispatch({ type: "SET_SELECTED_MODEL", payload: gatewayModel });
        } else {
          debugLog("[Chat] Gateway model not in AVAILABLE_MODELS, keeping local:", gatewayModel);
        }
      })
      .catch(() => {
        debugLog("[Chat] Gateway unreachable, using localStorage/default");
      });
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("ap3x.selectedModel", state.selectedModel);
    } catch {
      // ignore
    }
  }, [state.selectedModel]);

  const applyAgentResult = useCallback(
    (result: any) => {
      if (result?.thread_id && result.thread_id !== state.threadId) {
        dispatch({ type: "SET_THREAD_ID", payload: result.thread_id });
      }
      if (typeof result?.auto_approve === "boolean") {
        dispatch({ type: "SET_AUTO_APPROVE", payload: result.auto_approve });
      }

      // Tool events
      const events: any[] = Array.isArray(result?.events) ? result.events : [];
      for (const ev of events) {
        if (ev?.type !== "tool_result") continue;
        const toolName: string = ev.tool_name || "tool";
        const status: string = ev.status || "success";
        const output: string = ev.read_output || ev.output || "";
        const args: Record<string, any> = ev.args || {};

        const cli: CLIContent[] = [];

        if (toolName === "shell" || toolName === "execute") {
          const command = args.command || args.cmd || "(command)";
          const exitMatch = String(output).match(/Exit code:\s*([0-9]+)/i);
          const exitCode = exitMatch
            ? Number(exitMatch[1])
            : status === "success"
              ? 0
              : 1;
          cli.push({
            type: "command-output",
            command,
            content: String(output),
            exitCode,
          });
        } else if (toolName === "read_file") {
          const p = args.file_path || args.path || ev.path || "(file)";
          cli.push({
            type: "file-content",
            path: String(p),
            content: String(output),
            language: getLanguageFromPath(String(p)),
          });
        } else if (toolName === "write_file" || toolName === "edit_file") {
          if (ev.diff) {
            cli.push({
              type: "file-diff",
              title: ev.path || toolName,
              content: String(ev.diff),
            });
          }
          if (status !== "success") {
            cli.push({
              type: "error",
              title: `${toolName} failed`,
              content: String(ev.error || output),
            });
          } else {
            // Show successful write/edit as a tool call
            cli.push({
              type: "tool-call",
              toolName,
              args,
              content: String(output),
              status: "success" as const,
            });
          }
        } else {
          // All other tools use the new tool-call display
          cli.push({
            type: "tool-call",
            toolName,
            args,
            content: String(output),
            status:
              status === "success" ? ("success" as const) : ("error" as const),
            error:
              status !== "success" ? String(ev.error || output) : undefined,
          });
        }

        if (cli.length > 0) {
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: `tool-${Date.now()}-${Math.random().toString(16).slice(2)}`,
              role: "system",
              content: "",
              timestamp: new Date(),
              cliContent: cli,
            },
          });
        }
      }

      // Assistant message (may be empty if waiting for approval)
      if (result?.assistant) {
        dispatch({
          type: "ADD_MESSAGE",
          payload: {
            id: `asst-${Date.now()}-${Math.random().toString(16).slice(2)}`,
            role: "assistant",
            content: String(result.assistant),
            timestamp: new Date(),
          },
        });
      }

      // Approval required
      if (result?.approval_required && Array.isArray(result?.approvals)) {
        for (const ap of result.approvals) {
          const approvalRequest: ApprovalRequest = {
            threadId: result.thread_id,
            assistantId: state.assistantId,
            interruptId: ap.interrupt_id,
            actionRequests: Array.isArray(ap.action_requests)
              ? ap.action_requests
              : [],
          };
          dispatch({
            type: "ADD_MESSAGE",
            payload: {
              id: `approval-${Date.now()}-${Math.random().toString(16).slice(2)}`,
              role: "system",
              content: "",
              timestamp: new Date(),
              approvalRequest,
            },
          });
        }
      }
    },
    [state.threadId, state.assistantId],
  );

  const sendMessage = useCallback(
    async (content: string, attachments?: FileAttachment[]): Promise<void> => {
      // Allow sending with just attachments (no text)
      if (!content.trim() && (!attachments || attachments.length === 0)) return;

      // Abort any existing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content,
        timestamp: new Date(),
        attachments:
          attachments && attachments.length > 0 ? attachments : undefined,
      };

      dispatch({ type: "ADD_MESSAGE", payload: userMessage });
      dispatch({ type: "SET_LOADING", payload: true });
      dispatch({ type: "SET_ERROR", payload: null });
      dispatch({
        type: "SET_STATUS",
        payload: { status: "thinking", message: "Agent is thinking..." },
      });

      let streamingMsgId: string | null = null;

      try {
        debugLog(
          "[Chat] Starting stream for message:",
          content,
          "with model:",
          state.selectedModel,
          "attachments:",
          attachments?.length || 0,
        );
        const manager = getTabContextManager();
        const assembled = manager.assembleContext({
          includeActiveTabFull: true,
          includeBackgroundSummaries: true,
          // Only include pinned background tabs.
          minPriority: "high",
          maxBackgroundTabs: 20,
          includeModuleContext: true,
        });
        const uiContext = buildAgentContext(assembled, 6000);

        const response = await fetch("/api/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            threadId: state.threadId,
            assistantId: state.assistantId,
            message: content,
            autoApprove: state.autoApprove,
            model: state.selectedModel,
            planMode: state.planMode,
            attachments:
              attachments && attachments.length > 0 ? attachments : undefined,
            uiContext,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          const errorText = await response.text();
          console.error("[Chat] Stream failed:", response.status, errorText);
          throw new Error("Failed to start stream");
        }
        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        debugLog("[Chat] Stream started, reading events...");

        const decoder = new TextDecoder();
        let buffer = "";
        let currentEvent = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            debugLog("[Chat] Stream ended (done=true)");
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7);
            } else if (line.startsWith("data: ") && currentEvent) {
              try {
                const data = JSON.parse(line.slice(6));
                debugLog("[Chat] Received event:", currentEvent, data);
                // Process each event type
                if (currentEvent === "status") {
                  // Only show status if we're NOT currently streaming text
                  if (!streamingMsgId) {
                    dispatch({
                      type: "SET_STATUS",
                      payload: { status: data.status, message: data.message },
                    });
                  }
                } else if (currentEvent === "thread_id" && data.threadId) {
                  dispatch({ type: "SET_THREAD_ID", payload: data.threadId });
                } else if (currentEvent === "text_delta") {
                  // Only clear status if there are no pending tool calls
                  if (streamingMsgId) {
                    const currentMsg = messagesRef.current.find(
                      (m) => m.id === streamingMsgId,
                    );
                    const pendingTools =
                      currentMsg?.cliContent?.filter(
                        (c) => c.status === "pending",
                      ) || [];
                    if (pendingTools.length === 0) {
                      dispatch({
                        type: "SET_STATUS",
                        payload: { status: null, message: null },
                      });
                    }
                    dispatch({
                      type: "APPEND_TO_MESSAGE",
                      payload: { id: streamingMsgId, text: data.text },
                    });
                  } else {
                    // No streaming message yet, clear status
                    dispatch({
                      type: "SET_STATUS",
                      payload: { status: null, message: null },
                    });
                    streamingMsgId = `asst-${Date.now()}-${Math.random().toString(16).slice(2)}`;
                    // Initialize with cliContent containing the first text part
                    dispatch({
                      type: "ADD_MESSAGE",
                      payload: {
                        id: streamingMsgId,
                        role: "assistant",
                        content: data.text,
                        timestamp: new Date(),
                        cliContent: [
                          { type: "text" as const, content: data.text },
                        ],
                      },
                    });
                  }
                } else if (currentEvent === "reasoning_delta") {
                  // Only show reasoning/thinking if we're NOT currently streaming text
                  if (!streamingMsgId) {
                    dispatch({
                      type: "SET_STATUS",
                      payload: {
                        status: "thinking",
                        message:
                          data.text.slice(0, 100) +
                          (data.text.length > 100 ? "..." : ""),
                      },
                    });
                  }
                } else if (currentEvent === "tool_call") {
                  debugLog(
                    "[Chat] tool_call event:",
                    data.toolCallId,
                    data.toolName,
                  );
                  // Show executing status so user knows agent is working
                  dispatch({
                    type: "SET_STATUS",
                    payload: {
                      status: "executing",
                      message: `Executing ${data.toolName}...`,
                    },
                  });

                  // Auto-open browser tab when agent creates a browser session
                  if (isBrowserSessionTool(data.toolName)) {
                    debugLog(
                      "[Chat] Browser session tool detected, emitting event for auto-open",
                    );
                    emitBrowserSession({
                      toolName: data.toolName,
                      sessionId: data.args?.sessionId as string | undefined,
                      initialUrl: (data.args?.startUrl || data.args?.url) as
                        | string
                        | undefined,
                    });
                  }

                  // Append tool call to the current streaming message (interleaved order)
                  if (streamingMsgId) {
                    debugLog(
                      "[Chat] Appending tool call to existing message:",
                      streamingMsgId,
                    );
                    dispatch({
                      type: "APPEND_CLI_CONTENT",
                      payload: {
                        id: streamingMsgId,
                        cliContent: {
                          type: "tool-call",
                          toolName: data.toolName,
                          args: data.args,
                          content: "",
                          status: "pending" as const,
                          title: data.toolCallId,
                        },
                      },
                    });
                  } else {
                    // Create new assistant message if none exists
                    streamingMsgId = `asst-${Date.now()}-${Math.random().toString(16).slice(2)}`;
                    debugLog(
                      "[Chat] Creating new message for tool call:",
                      streamingMsgId,
                    );
                    dispatch({
                      type: "ADD_MESSAGE",
                      payload: {
                        id: streamingMsgId,
                        role: "assistant",
                        content: "",
                        timestamp: new Date(),
                        cliContent: [
                          {
                            type: "tool-call",
                            toolName: data.toolName,
                            args: data.args,
                            content: "",
                            status: "pending" as const,
                            title: data.toolCallId,
                          },
                        ],
                      },
                    });
                  }
                } else if (currentEvent === "tool_result") {
                  debugLog(
                    "[Chat] tool_result event:",
                    data.toolCallId,
                    data.status,
                  );
                  const output = data.output || "";

                  // Check if this is a browser session result - extract wsPath and emit event
                  const isBrowserLiveResult =
                    data.toolName &&
                    (data.toolName === "browser_live_start" ||
                      data.toolName === "browser-live-start" ||
                      data.toolName === "open_live_browser") && // Custom tool (non-MCP)
                    data.status === "success";
                  if (isBrowserLiveResult) {
                    debugLog(
                      "[Chat] browser_live_start detected, raw output:",
                      output,
                    );
                    try {
                      // Try to parse JSON output for wsPath
                      // Output format could be:
                      // 1. Direct JSON: {"ok":true,"wsPath":"..."}
                      // 2. LangChain content block: {"type":"text","text":"{\"ok\":true,...}"}
                      // 3. Stringified content block array: [{"type":"text","text":"..."}]
                      let parsed =
                        typeof output === "string"
                          ? JSON.parse(output)
                          : output;
                      debugLog(
                        "[Chat] browser_live_start parsed step 1:",
                        parsed,
                      );

                      // Handle LangChain content block format - extract nested text and parse it
                      if (
                        parsed &&
                        parsed.type === "text" &&
                        typeof parsed.text === "string"
                      ) {
                        debugLog(
                          "[Chat] browser_live_start detected LangChain content block, extracting text",
                        );
                        try {
                          parsed = JSON.parse(parsed.text);
                          debugLog(
                            "[Chat] browser_live_start parsed step 2:",
                            parsed,
                          );
                        } catch (e) {
                          debugLog(
                            "[Chat] browser_live_start failed to parse nested text:",
                            e,
                          );
                        }
                      }
                      // Handle array of content blocks
                      if (
                        Array.isArray(parsed) &&
                        parsed.length > 0 &&
                        parsed[0]?.type === "text"
                      ) {
                        debugLog(
                          "[Chat] browser_live_start detected array of content blocks",
                        );
                        try {
                          parsed = JSON.parse(parsed[0].text);
                          debugLog(
                            "[Chat] browser_live_start parsed step 2 (array):",
                            parsed,
                          );
                        } catch (e) {
                          debugLog(
                            "[Chat] browser_live_start failed to parse array text:",
                            e,
                          );
                        }
                      }

                      debugLog(
                        "[Chat] browser_live_start final parsed object:",
                        parsed,
                        "wsPath:",
                        parsed?.wsPath,
                      );
                      if (parsed && parsed.wsPath) {
                        debugLog(
                          "[Chat] browser_live_start result, emitting wsPath event:",
                          parsed.wsPath,
                        );
                        emitBrowserSession({
                          toolName: data.toolName,
                          sessionId: parsed.sessionId,
                          wsPath: parsed.wsPath,
                          initialUrl: parsed.url,
                        });
                      } else {
                        debugLog(
                          "[Chat] browser_live_start: wsPath not found in parsed object",
                        );
                      }
                    } catch (e) {
                      debugLog(
                        "[Chat] browser_live_start parse error, trying regex fallback:",
                        e,
                      );
                      // Output might not be JSON, check for websocket path pattern
                      // Pattern matches wsPath format: /v1/agent-browser/sessions/.../live/ws
                      const wsPathMatch = String(output).match(
                        /\/v1\/agent-browser\/sessions\/[^"'\s]+\/live\/ws|ws[s]?:\/\/[^\s"]+|\/ws\/[^\s"]+/,
                      );
                      if (wsPathMatch) {
                        debugLog(
                          "[Chat] browser_live_start result, extracted wsPath via regex:",
                          wsPathMatch[0],
                        );
                        emitBrowserSession({
                          toolName: data.toolName,
                          wsPath: wsPathMatch[0],
                        });
                      } else {
                        debugLog(
                          "[Chat] browser_live_start: regex also failed to find wsPath",
                        );
                      }
                    }
                  }

                  // Check if this is a browser navigation result - emit URL change event
                  const isBrowserNavigateResult =
                    data.toolName &&
                    (data.toolName === "browser_navigate" ||
                      data.toolName === "browser-navigate") &&
                    data.status === "success";
                  if (isBrowserNavigateResult) {
                    debugLog(
                      "[Chat] browser_navigate detected, output:",
                      output,
                    );
                    try {
                      let parsed =
                        typeof output === "string"
                          ? JSON.parse(output)
                          : output;
                      // Handle LangChain content block format
                      if (
                        parsed &&
                        parsed.type === "text" &&
                        typeof parsed.text === "string"
                      ) {
                        parsed = JSON.parse(parsed.text);
                      }
                      if (
                        Array.isArray(parsed) &&
                        parsed.length > 0 &&
                        parsed[0]?.type === "text"
                      ) {
                        parsed = JSON.parse(parsed[0].text);
                      }
                      if (parsed && parsed.url) {
                        debugLog(
                          "[Chat] browser_navigate emitting URL update:",
                          parsed.url,
                        );
                        emitBrowserSession({
                          toolName: data.toolName,
                          sessionId: parsed.sessionId,
                          navigatedUrl: parsed.url,
                          eventType: "navigate",
                        });
                      }
                    } catch (e) {
                      debugLog("[Chat] browser_navigate parse error:", e);
                    }
                  }

                  // Check if this is an image generation result
                  const imageMatch =
                    output.match(/Image generated and saved to:\s*(.+)$/i) ||
                    output.match(/Image saved:\s*(.+)$/i) ||
                    output.match(/Edited image saved to:\s*(.+)$/i);
                  // Update the tool call in the streaming message's cliContent
                  if (streamingMsgId) {
                    const currentMsg = messagesRef.current.find(
                      (m) => m.id === streamingMsgId,
                    );
                    const currentCli = currentMsg?.cliContent || [];
                    debugLog(
                      "[Chat] Current cliContent:",
                      currentCli.map((c) => ({
                        type: c.type,
                        title: c.title,
                        status: c.status,
                      })),
                    );
                    debugLog(
                      "[Chat] Looking for toolCallId:",
                      data.toolCallId,
                    );

                    const updatedCli = currentCli.map((cli) => {
                      if (
                        cli.type === "tool-call" &&
                        cli.title === data.toolCallId
                      ) {
                        debugLog(
                          "[Chat] FOUND matching tool call, updating to success",
                        );
                        if (
                          imageMatch &&
                          data.toolName &&
                          (data.toolName.includes("image") ||
                            data.toolName === "generate_image" ||
                            data.toolName === "edit_image")
                        ) {
                          return {
                            ...cli,
                            type: "image" as const,
                            content: output,
                            status: "success" as const,
                            imagePath: imageMatch[1].trim(),
                          };
                        }
                        return {
                          ...cli,
                          content: output,
                          status:
                            data.status === "success"
                              ? ("success" as const)
                              : ("error" as const),
                          error: data.error,
                        };
                      }
                      return cli;
                    });

                    debugLog(
                      "[Chat] Updated cliContent:",
                      updatedCli.map((c) => ({
                        type: c.type,
                        title: c.title,
                        status: c.status,
                      })),
                    );

                    dispatch({
                      type: "UPDATE_MESSAGE",
                      payload: {
                        id: streamingMsgId,
                        updates: { cliContent: updatedCli },
                      },
                    });

                    // Check if there are still pending tool calls - keep showing working status
                    const stillPending = updatedCli.filter(
                      (cli) => cli.status === "pending",
                    );
                    if (stillPending.length > 0) {
                      const pendingTool = stillPending[0];
                      const statusMsg = formatStatusMessage(
                        pendingTool.toolName || "tool",
                        pendingTool.args,
                      );
                      dispatch({
                        type: "SET_STATUS",
                        payload: { status: "executing", message: statusMsg },
                      });
                    } else {
                      // All tools done, show thinking status while agent processes results
                      dispatch({
                        type: "SET_STATUS",
                        payload: {
                          status: "thinking",
                          message: "Agent is processing...",
                        },
                      });
                    }
                  }
                } else if (
                  currentEvent === "approval_required" &&
                  Array.isArray(data.approvals)
                ) {
                  for (const ap of data.approvals) {
                    dispatch({
                      type: "ADD_MESSAGE",
                      payload: {
                        id: `approval-${Date.now()}-${Math.random().toString(16).slice(2)}`,
                        role: "system",
                        content: "",
                        timestamp: new Date(),
                        approvalRequest: {
                          threadId: state.threadId || "",
                          assistantId: state.assistantId,
                          interruptId: ap.interrupt_id,
                          actionRequests: ap.action_requests || [],
                        },
                      },
                    });
                  }
                } else if (currentEvent === "tool_progress") {
                  // Progress updates from long-running tools (e.g. parallel_tasks)
                  const msg = data.message || `Executing ${data.toolName || "tool"}...`;
                  debugLog("[Chat] tool_progress:", msg);
                  dispatch({
                    type: "SET_STATUS",
                    payload: {
                      status: "executing",
                      message: msg,
                    },
                  });

                // ── Plan execution SSE events ──
                } else if (currentEvent === "plan_created") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "planning" });

                } else if (currentEvent === "plan_approval_requested") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "reviewing" });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-approval-${Date.now()}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planApproval: {
                        planPath: data.plan_path || data.planPath || "",
                        planContent: data.plan_content || data.planContent || "",
                        summary: data.summary || "",
                        stepCount: data.step_count || data.stepCount || 0,
                        checkpointCount:
                          data.checkpoint_count || data.checkpointCount || 0,
                      },
                    },
                  });

                } else if (currentEvent === "plan_approved") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "executing" });

                } else if (currentEvent === "plan_rejected") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "idle" });

                } else if (currentEvent === "plan_step_started") {
                  dispatch({
                    type: "SET_PLAN_PROGRESS",
                    payload: {
                      currentStep: data.step_number || data.stepNumber || 0,
                      totalSteps: data.total_steps || data.totalSteps || 0,
                      description: data.description || "",
                    },
                  });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-step-${data.step_number || data.stepNumber}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planProgress: {
                        currentStep: data.step_number || data.stepNumber || 0,
                        totalSteps: data.total_steps || data.totalSteps || 0,
                        description: data.description || "",
                        status: "running" as const,
                        steps: [],
                      },
                    },
                  });

                } else if (currentEvent === "plan_step_completed") {
                  const stepNum = data.step_number || data.stepNumber || 0;
                  const existingMsgId = `plan-step-${stepNum}`;
                  const existingMsg = messagesRef.current.find(
                    (m) => m.id === existingMsgId,
                  );
                  if (existingMsg) {
                    dispatch({
                      type: "UPDATE_MESSAGE",
                      payload: {
                        id: existingMsgId,
                        updates: {
                          planProgress: {
                            currentStep: stepNum,
                            totalSteps:
                              data.total_steps || data.totalSteps || 0,
                            description: data.description || "",
                            status: data.success
                              ? ("completed" as const)
                              : ("failed" as const),
                            steps: [],
                          },
                        },
                      },
                    });
                  }

                } else if (currentEvent === "plan_checkpoint") {
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-checkpoint-${Date.now()}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planCheckpoint: {
                        stepNumber: data.step_number || data.stepNumber || 0,
                        passed: data.validation_result?.passed ?? true,
                        summary: data.validation_result?.summary || "",
                      },
                    },
                  });

                } else if (currentEvent === "plan_execution_completed") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "complete" });
                  dispatch({ type: "SET_PLAN_PROGRESS", payload: null });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-summary-${Date.now()}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planSummary: {
                        success: data.success ?? true,
                        stepsCompleted:
                          data.steps_completed || data.stepsCompleted || 0,
                        totalSteps: data.total_steps || data.totalSteps || 0,
                      },
                    },
                  });

                } else if (currentEvent === "done") {
                  debugLog("[Chat] Received done event");
                  dispatch({
                    type: "SET_STATUS",
                    payload: { status: null, message: null },
                  });
                } else if (currentEvent === "error") {
                  console.error("[Chat] Received error event:", data.message);
                  dispatch({
                    type: "SET_ERROR",
                    payload: data.message || "Stream error",
                  });
                }
              } catch (parseError) {
                console.warn(
                  "[Chat] Failed to parse event data:",
                  line,
                  parseError,
                );
              }
              currentEvent = "";
            }
          }
        }
      } catch (error: any) {
        // Don't show error if aborted by user
        if (error.name !== "AbortError") {
          console.error("[Chat] Stream error:", error);
          dispatch({
            type: "SET_ERROR",
            payload: error.message || "Failed to send message",
          });
        } else {
          debugLog("[Chat] Stream aborted by user");
        }
      } finally {
        debugLog(
          "[Chat] Stream cleanup - setting loading=false, clearing status",
        );
        abortControllerRef.current = null;
        dispatch({ type: "SET_LOADING", payload: false });
        dispatch({
          type: "SET_STATUS",
          payload: { status: null, message: null },
        });
      }
    },
    [state.threadId, state.assistantId, state.autoApprove, state.selectedModel, state.planMode],
  );

  const decideApproval = useCallback(
    async (
      interruptId: string,
      decision: "approve" | "reject" | "auto_approve_all",
    ): Promise<void> => {
      if (!state.threadId) {
        throw new Error("No active threadId for approval");
      }

      // Find and remove the approval message immediately
      // Use ref to get current messages (avoids stale closure)
      const currentMessages = messagesRef.current;
      const approvalMsg = currentMessages.find(
        (m) => m.approvalRequest?.interruptId === interruptId,
      );

      if (approvalMsg) {
        dispatch({ type: "REMOVE_MESSAGE", payload: approvalMsg.id });
      }

      dispatch({ type: "SET_LOADING", payload: true });
      dispatch({ type: "SET_ERROR", payload: null });
      dispatch({
        type: "SET_STATUS",
        payload: { status: "thinking", message: "Agent is working..." },
      });

      let streamingMsgId: string | null = null;

      try {
        const response = await fetch("/api/chat/resume-stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            threadId: state.threadId,
            assistantId: state.assistantId,
            interruptId,
            decision,
          }),
        });

        if (!response.ok) throw new Error("Failed to send approval decision");
        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";
        let currentEvent = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7);
            } else if (line.startsWith("data: ") && currentEvent) {
              try {
                const data = JSON.parse(line.slice(6));
                if (currentEvent === "status") {
                  // Only show status if we're NOT currently streaming text
                  if (!streamingMsgId) {
                    dispatch({
                      type: "SET_STATUS",
                      payload: { status: data.status, message: data.message },
                    });
                  }
                } else if (currentEvent === "text_delta") {
                  // Clear status when text starts streaming
                  dispatch({
                    type: "SET_STATUS",
                    payload: { status: null, message: null },
                  });

                  if (streamingMsgId) {
                    dispatch({
                      type: "APPEND_TO_MESSAGE",
                      payload: { id: streamingMsgId, text: data.text },
                    });
                  } else {
                    streamingMsgId = `asst-${Date.now()}-${Math.random().toString(16).slice(2)}`;
                    dispatch({
                      type: "ADD_MESSAGE",
                      payload: {
                        id: streamingMsgId,
                        role: "assistant",
                        content: data.text,
                        timestamp: new Date(),
                      },
                    });
                  }
                } else if (currentEvent === "tool_call") {
                  // Show executing status so user knows agent is working
                  const statusMsg = formatStatusMessage(
                    data.toolName,
                    data.args,
                  );
                  dispatch({
                    type: "SET_STATUS",
                    payload: { status: "executing", message: statusMsg },
                  });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `tool-${data.toolCallId}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      cliContent: [
                        {
                          type: "tool-call",
                          toolName: data.toolName,
                          args: data.args,
                          content: "",
                          status: "pending" as const,
                        },
                      ],
                    },
                  });
                } else if (currentEvent === "tool_result") {
                  const output = data.output || "";
                  // Check if this is an image generation result
                  const imageMatch =
                    output.match(/Image generated and saved to:\s*(.+)$/i) ||
                    output.match(/Image saved:\s*(.+)$/i) ||
                    output.match(/Edited image saved to:\s*(.+)$/i);
                  if (
                    imageMatch &&
                    data.toolName &&
                    (data.toolName.includes("image") ||
                      data.toolName === "generate_image" ||
                      data.toolName === "edit_image")
                  ) {
                    const imagePath = imageMatch[1].trim();
                    const cli: CLIContent[] = [
                      {
                        type: "image",
                        toolName: data.toolName,
                        args: data.args || {},
                        content: output,
                        status: "success" as const,
                        imagePath,
                      },
                    ];
                    dispatch({
                      type: "UPDATE_MESSAGE",
                      payload: {
                        id: `tool-${data.toolCallId}`,
                        updates: { cliContent: cli },
                      },
                    });
                  } else {
                    const cli: CLIContent[] = [
                      {
                        type: "tool-call",
                        toolName: data.toolName,
                        args: data.args || {},
                        content: output,
                        status:
                          data.status === "success"
                            ? ("success" as const)
                            : ("error" as const),
                        error: data.error,
                      },
                    ];
                    dispatch({
                      type: "UPDATE_MESSAGE",
                      payload: {
                        id: `tool-${data.toolCallId}`,
                        updates: { cliContent: cli },
                      },
                    });
                  }
                } else if (
                  currentEvent === "approval_required" &&
                  Array.isArray(data.approvals)
                ) {
                  for (const ap of data.approvals) {
                    dispatch({
                      type: "ADD_MESSAGE",
                      payload: {
                        id: `approval-${Date.now()}-${Math.random().toString(16).slice(2)}`,
                        role: "system",
                        content: "",
                        timestamp: new Date(),
                        approvalRequest: {
                          threadId: state.threadId || "",
                          assistantId: state.assistantId,
                          interruptId: ap.interrupt_id,
                          actionRequests: ap.action_requests || [],
                        },
                      },
                    });
                  }
                } else if (currentEvent === "tool_progress") {
                  const msg = data.message || `Executing ${data.toolName || "tool"}...`;
                  dispatch({
                    type: "SET_STATUS",
                    payload: { status: "executing", message: msg },
                  });

                // ── Plan execution SSE events (decideApproval) ──
                } else if (currentEvent === "plan_created") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "planning" });

                } else if (currentEvent === "plan_approval_requested") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "reviewing" });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-approval-${Date.now()}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planApproval: {
                        planPath: data.plan_path || data.planPath || "",
                        planContent:
                          data.plan_content || data.planContent || "",
                        summary: data.summary || "",
                        stepCount: data.step_count || data.stepCount || 0,
                        checkpointCount:
                          data.checkpoint_count || data.checkpointCount || 0,
                      },
                    },
                  });

                } else if (currentEvent === "plan_approved") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "executing" });

                } else if (currentEvent === "plan_rejected") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "idle" });

                } else if (currentEvent === "plan_step_started") {
                  dispatch({
                    type: "SET_PLAN_PROGRESS",
                    payload: {
                      currentStep:
                        data.step_number || data.stepNumber || 0,
                      totalSteps: data.total_steps || data.totalSteps || 0,
                      description: data.description || "",
                    },
                  });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-step-${data.step_number || data.stepNumber}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planProgress: {
                        currentStep:
                          data.step_number || data.stepNumber || 0,
                        totalSteps:
                          data.total_steps || data.totalSteps || 0,
                        description: data.description || "",
                        status: "running" as const,
                        steps: [],
                      },
                    },
                  });

                } else if (currentEvent === "plan_step_completed") {
                  const stepNum =
                    data.step_number || data.stepNumber || 0;
                  const existingMsgId = `plan-step-${stepNum}`;
                  const existingMsg = messagesRef.current.find(
                    (m) => m.id === existingMsgId,
                  );
                  if (existingMsg) {
                    dispatch({
                      type: "UPDATE_MESSAGE",
                      payload: {
                        id: existingMsgId,
                        updates: {
                          planProgress: {
                            currentStep: stepNum,
                            totalSteps:
                              data.total_steps || data.totalSteps || 0,
                            description: data.description || "",
                            status: data.success
                              ? ("completed" as const)
                              : ("failed" as const),
                            steps: [],
                          },
                        },
                      },
                    });
                  }

                } else if (currentEvent === "plan_checkpoint") {
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-checkpoint-${Date.now()}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planCheckpoint: {
                        stepNumber:
                          data.step_number || data.stepNumber || 0,
                        passed: data.validation_result?.passed ?? true,
                        summary: data.validation_result?.summary || "",
                      },
                    },
                  });

                } else if (currentEvent === "plan_execution_completed") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "complete" });
                  dispatch({ type: "SET_PLAN_PROGRESS", payload: null });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-summary-${Date.now()}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planSummary: {
                        success: data.success ?? true,
                        stepsCompleted:
                          data.steps_completed || data.stepsCompleted || 0,
                        totalSteps:
                          data.total_steps || data.totalSteps || 0,
                      },
                    },
                  });

                } else if (currentEvent === "done") {
                  dispatch({
                    type: "SET_STATUS",
                    payload: { status: null, message: null },
                  });
                } else if (currentEvent === "error") {
                  dispatch({
                    type: "SET_ERROR",
                    payload: data.message || "Stream error",
                  });
                }
              } catch {
                /* ignore */
              }
              currentEvent = "";
            }
          }
        }
      } catch (error: any) {
        dispatch({
          type: "SET_ERROR",
          payload: error.message || "Failed to process approval",
        });
      } finally {
        dispatch({ type: "SET_LOADING", payload: false });
        dispatch({
          type: "SET_STATUS",
          payload: { status: null, message: null },
        });
      }
    },
    [state.threadId, state.assistantId],
  );

  const decidePlanApproval = useCallback(
    async (decision: "approve" | "reject", reason?: string) => {
      if (!state.threadId) return;

      dispatch({ type: "SET_LOADING", payload: true });

      try {
        const response = await fetch("/api/chat/resume-stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            threadId: state.threadId,
            assistantId: state.assistantId,
            planDecision: decision,
            planRejectReason: reason || "",
          }),
        });

        if (!response.ok || !response.body) {
          dispatch({ type: "SET_LOADING", payload: false });
          return;
        }

        // Process the SSE response using the same pattern as decideApproval
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let currentEvent = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              currentEvent = line.slice(7);
            } else if (line.startsWith("data: ") && currentEvent) {
              try {
                const data = JSON.parse(line.slice(6));

                if (currentEvent === "text_delta" && data.text) {
                  // Handle streaming text from the agent after plan decision
                  const streamingMsgId = `plan-execution-text-${state.threadId}`;
                  const existing = messagesRef.current.find(
                    (m) => m.id === streamingMsgId,
                  );
                  if (existing) {
                    dispatch({
                      type: "APPEND_TO_MESSAGE",
                      payload: {
                        id: streamingMsgId,
                        text: data.text,
                      },
                    });
                  } else {
                    dispatch({
                      type: "ADD_MESSAGE",
                      payload: {
                        id: streamingMsgId,
                        role: "assistant",
                        content: data.text,
                        timestamp: new Date(),
                      },
                    });
                  }
                } else if (currentEvent === "plan_approved") {
                  dispatch({
                    type: "SET_PLAN_STATUS",
                    payload: "executing",
                  });
                } else if (currentEvent === "plan_rejected") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "idle" });
                } else if (currentEvent === "plan_step_started") {
                  dispatch({
                    type: "SET_PLAN_PROGRESS",
                    payload: {
                      currentStep: data.step_number || 0,
                      totalSteps: data.total_steps || 0,
                      description: data.description || "",
                    },
                  });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-step-${data.step_number}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planProgress: {
                        currentStep: data.step_number || 0,
                        totalSteps: data.total_steps || 0,
                        description: data.description || "",
                        status: "running" as const,
                        steps: [],
                      },
                    },
                  });
                } else if (currentEvent === "plan_step_completed") {
                  const stepNum = data.step_number || 0;
                  dispatch({
                    type: "UPDATE_MESSAGE",
                    payload: {
                      id: `plan-step-${stepNum}`,
                      updates: {
                        planProgress: {
                          currentStep: stepNum,
                          totalSteps: data.total_steps || 0,
                          description: data.description || "",
                          status: data.success
                            ? ("completed" as const)
                            : ("failed" as const),
                          steps: [],
                        },
                      },
                    },
                  });
                } else if (currentEvent === "plan_checkpoint") {
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-checkpoint-${Date.now()}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planCheckpoint: {
                        stepNumber: data.step_number || 0,
                        passed: data.validation_result?.passed ?? true,
                        summary: data.validation_result?.summary || "",
                      },
                    },
                  });
                } else if (currentEvent === "plan_execution_completed") {
                  dispatch({ type: "SET_PLAN_STATUS", payload: "complete" });
                  dispatch({ type: "SET_PLAN_PROGRESS", payload: null });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `plan-summary-${Date.now()}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      planSummary: {
                        success: data.success ?? true,
                        stepsCompleted: data.steps_completed || 0,
                        totalSteps: data.total_steps || 0,
                      },
                    },
                  });
                } else if (currentEvent === "tool_call") {
                  const statusMsg = formatStatusMessage(
                    data.toolName,
                    data.args,
                  );
                  dispatch({
                    type: "SET_STATUS",
                    payload: { status: "executing", message: statusMsg },
                  });
                  dispatch({
                    type: "ADD_MESSAGE",
                    payload: {
                      id: `tool-${data.toolCallId}`,
                      role: "system",
                      content: "",
                      timestamp: new Date(),
                      cliContent: [
                        {
                          type: "tool-call",
                          toolName: data.toolName,
                          args: data.args,
                          content: "",
                          status: "pending" as const,
                        },
                      ],
                    },
                  });
                } else if (currentEvent === "tool_result") {
                  const output = data.output || "";
                  const cli: CLIContent[] = [
                    {
                      type: "tool-call",
                      toolName: data.toolName,
                      args: data.args || {},
                      content: output,
                      status:
                        data.status === "success"
                          ? ("success" as const)
                          : ("error" as const),
                      error: data.error,
                    },
                  ];
                  dispatch({
                    type: "UPDATE_MESSAGE",
                    payload: {
                      id: `tool-${data.toolCallId}`,
                      updates: { cliContent: cli },
                    },
                  });
                } else if (currentEvent === "done") {
                  dispatch({ type: "SET_LOADING", payload: false });
                }
              } catch {
                // ignore JSON parse errors
              }
              currentEvent = "";
            }
          }
        }
      } catch (error) {
        console.error("[Chat] Plan approval error:", error);
      } finally {
        dispatch({ type: "SET_LOADING", payload: false });
      }
    },
    [state.threadId, state.assistantId],
  );

  const clearMessages = useCallback(() => {
    dispatch({ type: "CLEAR_MESSAGES" });
    dispatch({ type: "SET_THREAD_ID", payload: null });
    try {
      localStorage.removeItem("ap3x.threadId");
    } catch {
      // ignore
    }
  }, []);

  const setLoading = useCallback((loading: boolean) => {
    dispatch({ type: "SET_LOADING", payload: loading });
  }, []);

  const setError = useCallback((error: string | null) => {
    dispatch({ type: "SET_ERROR", payload: error });
  }, []);

  const setAutoApprove = useCallback((enabled: boolean) => {
    dispatch({ type: "SET_AUTO_APPROVE", payload: enabled });
  }, []);

  const setSelectedModel = useCallback(
    (model: string) => {
      // Clear thread when changing models to avoid context pollution
      if (model !== state.selectedModel) {
        debugLog(
          "[Chat] Model changed from",
          state.selectedModel,
          "to",
          model,
          "- clearing thread",
        );
        dispatch({ type: "SET_THREAD_ID", payload: null });
        dispatch({ type: "CLEAR_MESSAGES" });
        try {
          localStorage.removeItem("ap3x.threadId");
        } catch {
          /* ignore */
        }
      }
      dispatch({ type: "SET_SELECTED_MODEL", payload: model });

      // Fire-and-forget: sync model selection to the gateway
      const provider = deriveProvider(model);
      fetch("/api/ag3nt/gateway/model/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, model }),
      }).catch(() => {
        // Gateway may be unreachable — silently ignore
      });
    },
    [state.selectedModel],
  );

  const setPlanMode = useCallback((enabled: boolean) => {
    dispatch({ type: "SET_PLAN_MODE", payload: enabled });
  }, []);

  const stopAgent = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    dispatch({ type: "SET_LOADING", payload: false });
    dispatch({ type: "SET_STATUS", payload: { status: null, message: null } });
  }, []);

  const clearCachesAndRestart = useCallback(async () => {
    try {
      debugLog("[Chat] Clearing caches and restarting daemon...");
      const response = await fetch("/api/chat/clear-caches", {
        method: "POST",
      });
      const result = await response.json();
      debugLog("[Chat] Clear caches result:", result);

      // Clear local state
      dispatch({ type: "CLEAR_MESSAGES" });
      dispatch({ type: "SET_THREAD_ID", payload: null });

      // Clear localStorage thread
      try {
        localStorage.removeItem("ap3x.threadId");
      } catch {
        /* ignore */
      }

      return result;
    } catch (error) {
      console.error("[Chat] Failed to clear caches:", error);
      throw error;
    }
  }, []);

  // Thread management functions
  const loadThreads = useCallback(async () => {
    dispatch({ type: "SET_THREADS_LOADING", payload: true });
    try {
      const response = await fetch("/api/threads?limit=50");
      const data = await response.json();
      if (data.threads) {
        // Map daemon response to ThreadInfo format
        const threads: ThreadInfo[] = data.threads.map((t: any) => ({
          threadId: t.thread_id,
          agentName: t.agent_name,
          updatedAt: t.updated_at,
          preview: t.preview,
        }));
        dispatch({ type: "SET_THREADS", payload: threads });
      }
    } catch (error) {
      console.error("[Chat] Failed to load threads:", error);
    } finally {
      dispatch({ type: "SET_THREADS_LOADING", payload: false });
    }
  }, []);

  const selectThread = useCallback(async (threadId: string) => {
    debugLog("[Chat] Selecting thread:", threadId);
    // Clear current messages
    dispatch({ type: "CLEAR_MESSAGES" });
    // Set the new thread ID
    dispatch({ type: "SET_THREAD_ID", payload: threadId });
    // Close thread history view
    dispatch({ type: "SET_SHOW_THREAD_HISTORY", payload: false });
    // Persist to localStorage
    try {
      localStorage.setItem("ap3x.threadId", threadId);
    } catch {
      /* ignore */
    }
  }, []);

  const deleteThread = useCallback(
    async (threadId: string) => {
      try {
        const response = await fetch(`/api/threads?id=${threadId}`, {
          method: "DELETE",
        });
        const result = await response.json();
        if (result.deleted) {
          dispatch({ type: "REMOVE_THREAD", payload: threadId });
          // If we deleted the current thread, clear it
          if (state.threadId === threadId) {
            dispatch({ type: "SET_THREAD_ID", payload: null });
            dispatch({ type: "CLEAR_MESSAGES" });
            try {
              localStorage.removeItem("ap3x.threadId");
            } catch {
              /* ignore */
            }
          }
        }
      } catch (error) {
        console.error("[Chat] Failed to delete thread:", error);
      }
    },
    [state.threadId],
  );

  const createNewThread = useCallback(() => {
    dispatch({ type: "CLEAR_MESSAGES" });
    dispatch({ type: "SET_THREAD_ID", payload: null });
    dispatch({ type: "SET_SHOW_THREAD_HISTORY", payload: false });
    try {
      localStorage.removeItem("ap3x.threadId");
    } catch {
      /* ignore */
    }
  }, []);

  const toggleThreadHistory = useCallback(() => {
    const newValue = !state.showThreadHistory;
    dispatch({ type: "SET_SHOW_THREAD_HISTORY", payload: newValue });
    // Load threads when opening the history view
    if (newValue) {
      loadThreads();
    }
  }, [state.showThreadHistory, loadThreads]);

  const contextValue: ChatContextType = useMemo(() => ({
    ...state,
    sendMessage,
    decideApproval,
    decidePlanApproval,
    clearMessages,
    setLoading,
    setError,
    setAutoApprove,
    setSelectedModel,
    setPlanMode,
    stopAgent,
    clearCachesAndRestart,
    loadThreads,
    selectThread,
    deleteThread,
    createNewThread,
    toggleThreadHistory,
  }), [
    state,
    sendMessage,
    decideApproval,
    decidePlanApproval,
    clearMessages,
    setLoading,
    setError,
    setAutoApprove,
    setSelectedModel,
    setPlanMode,
    stopAgent,
    clearCachesAndRestart,
    loadThreads,
    selectThread,
    deleteThread,
    createNewThread,
    toggleThreadHistory,
  ]);

  return (
    <ChatContext.Provider value={contextValue}>{children}</ChatContext.Provider>
  );
}

// Hook to use the chat context
export function useChat() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error("useChat must be used within a ChatProvider");
  }
  return context;
}
