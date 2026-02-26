// File attachment for messages
export interface FileAttachment {
  id: string;
  name: string;
  type: string; // MIME type
  size: number;
  // For images/small files, store as data URL
  dataUrl?: string;
  // For larger files, store content as base64
  content?: string;
  // Preview URL for display
  previewUrl?: string;
}

// Chat and Message types
export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: Date;
  // File attachments
  attachments?: FileAttachment[];
  // CLI integration fields
  cliContent?: CLIContent[];
  fileMentions?: FileMentionData[];
  // Agent approval (human-in-the-loop)
  approvalRequest?: ApprovalRequest;
  // Plan execution fields
  planApproval?: {
    planPath: string;
    planContent: string;
    summary: string;
    stepCount: number;
    checkpointCount: number;
  };
  planProgress?: {
    currentStep: number;
    totalSteps: number;
    description: string;
    status: "running" | "completed" | "failed";
    steps: Array<{ number: number; description: string; status: string }>;
  };
  planCheckpoint?: {
    stepNumber: number;
    passed: boolean;
    summary: string;
  };
  planSummary?: {
    success: boolean;
    stepsCompleted: number;
    totalSteps: number;
  };
}

// CLI content types for messages
// "text" type allows interleaving text with tool calls in chronological order
export type CLIContentType =
  | "text"
  | "file-content"
  | "command-output"
  | "file-diff"
  | "tool-output"
  | "tool-call"
  | "error"
  | "image";

export interface CLIContent {
  type: CLIContentType;
  title?: string;
  content: string;
  language?: string;
  exitCode?: number;
  path?: string;
  command?: string;
  executionTime?: number;
  // Tool call specific fields
  toolName?: string;
  args?: Record<string, any>;
  status?: "pending" | "success" | "error";
  error?: string;
  // Image specific fields
  imagePath?: string;
}

export interface FileMentionData {
  path: string;
  exists: boolean;
  content?: string;
  language?: string;
}

export interface ApprovalActionRequest {
  id?: string;
  name: string;
  args?: Record<string, any>;
  description?: string;
}

export interface ApprovalRequest {
  threadId: string;
  assistantId: string;
  interruptId: string;
  actionRequests: ApprovalActionRequest[];
}

export interface ChatSession {
  id: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

// Thread/Session information for thread history
export interface ThreadInfo {
  threadId: string;
  agentName: string | null;
  updatedAt: string | null;
  preview: string | null;
}

// Thread history grouped by time period
export interface ThreadHistoryGroup {
  label: string; // e.g., "Today", "Last 7 days", "Last 30 days"
  threads: ThreadInfo[];
}

// UI Component types
export interface ResizableLayoutProps {
  defaultSidebarWidth?: number;
  minSidebarWidth?: number;
  maxSidebarWidth?: number;
}

export interface PreviewPanelProps {
  previewUrl: string | null;
  isLoading?: boolean;
}

export interface SidebarProps {
  setPreviewUrl: (url: string | null) => void;
  className?: string;
}

// Tab types for the main preview area
export interface Tab {
  id: string;
  title: string;
  url: string | null;
  isLoading?: boolean;
  error?: string | null;
  /** Module type for this tab (default: 'browser') */
  moduleType?: string;
  /** Module instance ID (for agent communication) */
  moduleInstanceId?: string;
  /** Module-specific data */
  moduleData?: Record<string, unknown>;
  /** Icon for the tab */
  icon?: string;
  /** Whether the tab can be closed */
  closable?: boolean;
  /** Whether this tab is pinned into agent context */
  pinned?: boolean;
}

export interface TabsState {
  tabs: Tab[];
  activeTabId: string | null;
}

// App Configuration types
export interface AppConfig {
  features: {
    cliIntegration: boolean;
    fileOperations: boolean;
    shellCommands: boolean;
  };
}
