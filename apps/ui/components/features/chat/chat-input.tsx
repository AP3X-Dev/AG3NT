"use client"

import type { FormEvent, KeyboardEvent, ChangeEvent } from "react"
import { useRef, useEffect, useState, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ArrowUp, Paperclip, Sparkles, Terminal, FolderOpen, Square, Zap, X, FileText, Image as ImageIcon, ClipboardList } from "lucide-react"
import { cn } from "@/lib/utils"
import { AutocompleteMenu } from "./autocomplete-menu"
import { InputModeIndicator } from "./input-mode-indicator"
import { ModelSelector } from "./model-selector"
import { useFileAutocomplete } from "@/hooks/use-file-autocomplete"
import { useCommandHistory } from "@/hooks/use-command-history"
import { parseFileMentions } from "@/lib/cli/autocomplete"
import { isBashCommand } from "@/lib/cli/shell"
import type { InputMode, FileMention } from "@/types/cli"
import type { FileAttachment } from "@/types/types"

interface ChatInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (e: FormEvent<HTMLFormElement>) => void
  isLoading?: boolean
  placeholder?: string
  className?: string
  autoApprove?: boolean
  onAutoApproveChange?: (enabled: boolean) => void
  onStop?: () => void
  selectedModel?: string
  onModelChange?: (model: string) => void
  // Plan mode
  planMode?: boolean
  onPlanModeChange?: (enabled: boolean) => void
  // Plan execution status
  planStatus?: 'idle' | 'planning' | 'reviewing' | 'executing' | 'complete'
  planProgress?: { currentStep: number; totalSteps: number; description: string } | null
  // File attachments
  attachments?: FileAttachment[]
  onAttachmentsChange?: (attachments: FileAttachment[]) => void
}

// Max file size: 10MB
const MAX_FILE_SIZE = 10 * 1024 * 1024
// Accepted file types
const ACCEPTED_FILE_TYPES = [
  'image/png', 'image/jpeg', 'image/gif', 'image/webp', 'image/svg+xml',
  'text/plain', 'text/csv', 'text/markdown',
  'application/json', 'application/pdf',
  'application/javascript', 'text/javascript',
  'text/html', 'text/css',
]

export function ChatInput({
  value,
  onChange,
  onSubmit,
  isLoading = false,
  placeholder = "Instruct your AG3NT",
  className,
  autoApprove = false,
  onAutoApproveChange,
  onStop,
  selectedModel,
  onModelChange,
  planMode = false,
  onPlanModeChange,
  planStatus,
  planProgress,
  attachments = [],
  onAttachmentsChange,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const lineHeight = 24
  const maxLines = 10
  const defaultLines = 2
  const minHeight = lineHeight * defaultLines
  const maxHeight = lineHeight * maxLines

  // Input mode detection
  const [inputMode, setInputMode] = useState<InputMode>("chat")
  const [fileMentions, setFileMentions] = useState<FileMention[]>([])

  // Autocomplete
  const autocomplete = useFileAutocomplete({ debounceMs: 150, maxItems: 8 })

  // Command history
  const history = useCommandHistory({ maxEntries: 100 })

  // Handle file selection
  const handleFileSelect = useCallback(async (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0 || !onAttachmentsChange) return

    const newAttachments: FileAttachment[] = []

    for (const file of Array.from(files)) {
      // Check file size
      if (file.size > MAX_FILE_SIZE) {
        console.warn(`File ${file.name} is too large (max 10MB)`)
        continue
      }

      // Check file type
      if (!ACCEPTED_FILE_TYPES.includes(file.type) && !file.type.startsWith('text/')) {
        console.warn(`File type ${file.type} not supported`)
        continue
      }

      try {
        const attachment = await readFileAsAttachment(file)
        newAttachments.push(attachment)
      } catch (error) {
        console.error(`Failed to read file ${file.name}:`, error)
      }
    }

    if (newAttachments.length > 0) {
      onAttachmentsChange([...attachments, ...newAttachments])
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [attachments, onAttachmentsChange])

  // Remove attachment
  const handleRemoveAttachment = useCallback((id: string) => {
    if (onAttachmentsChange) {
      onAttachmentsChange(attachments.filter(a => a.id !== id))
    }
  }, [attachments, onAttachmentsChange])

  // Open file picker
  const handleAttachClick = useCallback(() => {
    fileInputRef.current?.click()
  }, [])

  // Update input mode based on value
  useEffect(() => {
    if (isBashCommand(value)) {
      setInputMode("bash-command")
    } else {
      setInputMode("chat")
    }

    // Parse file mentions
    const mentions = parseFileMentions(value)
    setFileMentions(mentions.map(m => ({
      ...m,
      exists: undefined,
      content: undefined,
    })))
  }, [value])

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = "auto"
      const newHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight)
      textarea.style.height = `${newHeight}px`
    }
  }, [value, minHeight, maxHeight])

  // Handle input change with autocomplete
  const handleChange = useCallback((newValue: string) => {
    onChange(newValue)
    const cursorPos = textareaRef.current?.selectionStart ?? newValue.length
    autocomplete.updateQuery(newValue, cursorPos)
  }, [onChange, autocomplete])

  // Handle autocomplete selection
  const handleAutocompleteSelect = useCallback((index: number) => {
    const result = autocomplete.selectItem(index)
    if (result) {
      onChange(result.text)
      // Set cursor position after React updates
      setTimeout(() => {
        textareaRef.current?.setSelectionRange(result.newCursorPosition, result.newCursorPosition)
        textareaRef.current?.focus()
      }, 0)
    }
  }, [autocomplete, onChange])

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Handle autocomplete navigation
    if (autocomplete.isOpen) {
      if (e.key === "ArrowDown") {
        e.preventDefault()
        autocomplete.selectNext()
        return
      }
      if (e.key === "ArrowUp") {
        e.preventDefault()
        autocomplete.selectPrevious()
        return
      }
      if (e.key === "Tab" || (e.key === "Enter" && autocomplete.items.length > 0)) {
        e.preventDefault()
        handleAutocompleteSelect(autocomplete.selectedIndex)
        return
      }
      if (e.key === "Escape") {
        e.preventDefault()
        autocomplete.close()
        return
      }
    }

    // Handle history navigation (when autocomplete is closed)
    if (!autocomplete.isOpen) {
      if (e.key === "ArrowUp" && !e.shiftKey) {
        const cursorAtStart = textareaRef.current?.selectionStart === 0
        if (cursorAtStart || !value.includes('\n')) {
          const prev = history.navigateUp(value)
          if (prev !== null) {
            e.preventDefault()
            onChange(prev)
          }
        }
        return
      }
      if (e.key === "ArrowDown" && !e.shiftKey) {
        const prev = history.navigateDown()
        if (prev !== null) {
          e.preventDefault()
          onChange(prev)
        }
        return
      }
    }

    // Submit on Enter (without Shift for multi-line)
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      if (value.trim()) {
        // Add to history
        history.addEntry(value, inputMode)
        history.reset()
        onSubmit(e as any)
      }
    }
  }, [autocomplete, history, value, inputMode, onChange, onSubmit, handleAutocompleteSelect])

  return (
    <form
      data-testid="chat-input-form"
      onSubmit={onSubmit}
      className={cn(
        "relative flex flex-col border border-border-default rounded-xl bg-surface-input/80 backdrop-blur-sm",
        "transition-all duration-200 hover:border-border-strong chat-input-focus",
        className
      )}
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={ACCEPTED_FILE_TYPES.join(',')}
        onChange={handleFileSelect}
        className="hidden"
      />

      {/* Autocomplete Menu */}
      <AutocompleteMenu
        items={autocomplete.items}
        selectedIndex={autocomplete.selectedIndex}
        isLoading={autocomplete.isLoading}
        type={autocomplete.activeType}
        onSelect={handleAutocompleteSelect}
      />

      {/* File Attachments Preview */}
      {attachments.length > 0 && (
        <div className="px-3 pt-2 flex flex-wrap gap-2">
          {attachments.map((attachment) => (
            <FileAttachmentPreview
              key={attachment.id}
              attachment={attachment}
              onRemove={() => handleRemoveAttachment(attachment.id)}
            />
          ))}
        </div>
      )}

      {/* Mode Indicator */}
      {(inputMode !== "chat" || fileMentions.length > 0) && (
        <div className="px-3 pt-2">
          <InputModeIndicator
            mode={inputMode}
            hasFileMentions={fileMentions.length > 0}
          />
        </div>
      )}

      <Textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => handleChange(e.target.value)}
        placeholder={placeholder}
        className={cn(
          "resize-none border-0 bg-transparent focus-visible:ring-0",
          "focus-visible:ring-offset-0 focus-visible:border-0 focus:outline-none",
          "shadow-none overflow-y-auto font-mono text-[17px] leading-relaxed",
          "placeholder:text-text-muted/60",
          inputMode === "bash-command" && "text-pink-400"
        )}
        style={{ minHeight: `${minHeight}px`, maxHeight: `${maxHeight}px` }}
        onKeyDown={handleKeyDown}
        disabled={isLoading}
      />
      <div className="flex items-center justify-between gap-1 px-2 pb-2">
        {/* Left side - Auto toggle and Model selector */}
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <button
            data-testid="auto-approve-toggle"
            type="button"
            onClick={() => onAutoApproveChange?.(!autoApprove)}
            className={cn(
              "flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium transition-all duration-150 shrink-0 active:scale-[0.97]",
              autoApprove
                ? "bg-amber-500/10 border border-amber-500/20 text-amber-400 hover-underglow-amber"
                : "bg-interactive-hover border border-border-default text-text-muted hover:text-text-secondary hover-underglow"
            )}
          >
            <Zap className={cn("h-3.5 w-3.5", autoApprove && "fill-current")} />
            <div className={cn(
              "w-6 h-3.5 rounded-full transition-all duration-200 relative",
              autoApprove ? "bg-amber-500" : "bg-interactive-hover"
            )}>
              <div className={cn(
                "absolute top-0.5 w-2.5 h-2.5 rounded-full bg-white transition-all duration-200",
                autoApprove ? "translate-x-3" : "translate-x-0.5"
              )} />
            </div>
          </button>

          {/* Plan Mode toggle */}
          <button
            type="button"
            onClick={() => onPlanModeChange?.(!planMode)}
            title={planMode ? `Plan mode ON — ${planStatus || 'idle'}` : "Plan mode OFF"}
            className={cn(
              "flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium transition-all duration-150 shrink-0 active:scale-[0.97]",
              planMode
                ? planStatus === "executing"
                  ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400"
                  : planStatus === "reviewing"
                    ? "bg-amber-500/10 border border-amber-500/20 text-amber-400"
                    : "bg-blue-500/10 border border-blue-500/20 text-blue-400 hover-underglow-blue"
                : "bg-interactive-hover border border-border-default text-text-muted hover:text-text-secondary hover-underglow"
            )}
          >
            <ClipboardList className={cn("h-3.5 w-3.5", planMode && "fill-current")} />
            <span className="hidden sm:inline">
              {planMode
                ? planStatus === "planning" ? "Planning..."
                  : planStatus === "reviewing" ? "Review"
                  : planStatus === "executing" ? `Step ${planProgress?.currentStep || '?'}/${planProgress?.totalSteps || '?'}`
                  : planStatus === "complete" ? "Done"
                  : "Plan"
                : "Plan"
              }
            </span>
            {planMode && (planStatus === "planning" || planStatus === "executing") && (
              <span className={cn(
                "w-1.5 h-1.5 rounded-full animate-pulse",
                planStatus === "executing" ? "bg-emerald-400" : "bg-blue-400"
              )} />
            )}
          </button>

          {/* Model Selector - truncated inline */}
          {selectedModel && onModelChange && (
            <ModelSelector
              selectedModel={selectedModel}
              onModelChange={onModelChange}
              disabled={isLoading}
              compact
            />
          )}
        </div>

        {/* Right side - action buttons */}
        <div className="flex items-center gap-0.5 shrink-0">
          {/* File mention hint */}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Browse files"
            className="h-7 w-7 text-text-muted hover:text-text-secondary hover:bg-interactive-hover transition-all duration-150 active:scale-[0.95]"
            disabled={isLoading}
          >
            <FolderOpen className="h-4 w-4" />
          </Button>

          {/* Bash hint */}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Terminal mode"
            className={cn(
              "h-7 w-7 hover:bg-interactive-hover transition-all duration-150 active:scale-[0.95]",
              inputMode === "bash-command" ? "text-pink-400" : "text-text-muted hover:text-text-secondary"
            )}
            disabled={isLoading}
          >
            <Terminal className="h-4 w-4" />
          </Button>

          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="AI features"
            className="h-7 w-7 text-text-muted hover:text-text-secondary hover:bg-interactive-hover transition-all duration-150 active:scale-[0.95]"
            disabled={isLoading}
          >
            <Sparkles className="h-4 w-4" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className={cn(
              "h-7 w-7 hover:bg-interactive-hover transition-all duration-150 active:scale-[0.95]",
              attachments.length > 0
                ? "text-blue-400 hover:text-blue-300"
                : "text-text-muted hover:text-text-secondary"
            )}
            disabled={isLoading}
            onClick={handleAttachClick}
            title="Attach files (images, text, code)"
            aria-label="Attach files"
          >
            <Paperclip className="h-4 w-4" />
            {attachments.length > 0 && (
              <span className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-blue-500 text-[10px] font-bold text-white flex items-center justify-center">
                {attachments.length}
              </span>
            )}
          </Button>
          {isLoading ? (
            <Button
              type="button"
              onClick={onStop}
              size="icon"
              aria-label="Stop generation"
              className="h-7 w-7 bg-red-500/15 border border-red-500/25 text-red-400 hover:bg-red-500/20 hover:shadow-[0_0_10px_rgba(239,68,68,0.2)] transition-all duration-150 active:scale-[0.95]"
            >
              <Square className="h-3.5 w-3.5 fill-current" />
            </Button>
          ) : (
            <Button
              type="submit"
              disabled={!value.trim() && attachments.length === 0}
              size="icon"
              aria-label="Send message"
              className={cn(
                "h-7 w-7 transition-all duration-150 active:scale-[0.95]",
                (value.trim() || attachments.length > 0)
                  ? "bg-emerald-500/15 border border-emerald-500/25 text-emerald-400 hover:bg-emerald-500/20 hover:shadow-[0_0_10px_rgba(52,211,153,0.2)]"
                  : "bg-interactive-hover border border-border-default text-text-muted"
              )}
            >
              <ArrowUp className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </form>
  )
}

// ============================================================================
// Helper Components & Functions
// ============================================================================

interface FileAttachmentPreviewProps {
  attachment: FileAttachment
  onRemove: () => void
}

function FileAttachmentPreview({ attachment, onRemove }: FileAttachmentPreviewProps) {
  const isImage = attachment.type.startsWith('image/')

  return (
    <div className="relative group flex items-center gap-2 px-2 py-1.5 bg-interactive-hover border border-border-default rounded-md max-w-[200px]">
      {/* Preview */}
      {isImage && attachment.dataUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={attachment.dataUrl}
          alt={attachment.name}
          className="h-8 w-8 object-cover rounded"
        />
      ) : (
        <div className="h-8 w-8 flex items-center justify-center bg-interactive-hover rounded">
          <FileText className="h-4 w-4 text-text-muted" />
        </div>
      )}

      {/* Name */}
      <div className="flex-1 min-w-0">
        <div className="text-[11px] text-text-primary truncate">{attachment.name}</div>
        <div className="text-[10px] text-text-muted">{formatFileSize(attachment.size)}</div>
      </div>

      {/* Remove button */}
      <button
        type="button"
        onClick={onRemove}
        className="absolute -top-1.5 -right-1.5 h-4 w-4 rounded-full bg-red-500 text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity hover:bg-red-600"
      >
        <X className="h-2.5 w-2.5" />
      </button>
    </div>
  )
}

/**
 * Read a File object and convert to FileAttachment
 */
async function readFileAsAttachment(file: File): Promise<FileAttachment> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = (e) => {
      const result = e.target?.result as string
      const isImage = file.type.startsWith('image/')

      resolve({
        id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        name: file.name,
        type: file.type,
        size: file.size,
        dataUrl: isImage ? result : undefined,
        content: isImage ? result.split(',')[1] : btoa(result), // base64 encode
        previewUrl: isImage ? result : undefined,
      })
    }

    reader.onerror = () => reject(new Error('Failed to read file'))

    // Read as data URL for images, as text for other files
    if (file.type.startsWith('image/')) {
      reader.readAsDataURL(file)
    } else {
      reader.readAsText(file)
    }
  })
}

/**
 * Format file size in human readable format
 */
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
