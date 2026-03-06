"use client"

import { useState, useEffect, useRef, type FormEvent, useCallback } from "react"
import { ChatMessage } from "./chat-message"
import { ChatInput } from "./chat-input"
import { AgentStatus } from "./agent-status"
import { ThreadHeader } from "./thread-header"
import { ThreadHistory } from "./thread-history"
import { cn } from "@/lib/utils"
import { useChat } from "@/providers"
import { useTasks } from "@/providers/tasks-provider"
import { TaskListView } from "@/components/features/tasks/task-list-view"
import type { FileAttachment } from "@/types/types"

export function ChatSidebar({ className }: { className?: string }) {
  const {
    messages,
    isLoading,
    sendMessage,
    autoApprove,
    setAutoApprove,
    status,
    statusMessage,
    stopAgent,
    selectedModel,
    setSelectedModel,
    planMode,
    setPlanMode,
    planStatus,
    planProgress,
    threadId,
    clearMessages,
    showThreadHistory,
    createNewThread
  } = useChat()
  const { isVisible: tasksVisible } = useTasks()
  const [input, setInput] = useState("")
  const [attachments, setAttachments] = useState<FileAttachment[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined)

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if ((!input.trim() && attachments.length === 0) || isLoading) return

    const currentInput = input.trim()
    const currentAttachments = [...attachments]
    setInput("") // Clear input immediately
    setAttachments([]) // Clear attachments

    try {
      await sendMessage(currentInput, currentAttachments)
    } catch (error) {
      console.error("Error sending message:", error)
    }
  }

  // Debounced auto-scroll to bottom when new messages arrive or status changes
  const scrollToBottom = useCallback(() => {
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current)
    }
    scrollTimeoutRef.current = setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }, [])

  useEffect(() => {
    scrollToBottom()
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current)
      }
    }
  }, [messages, status, scrollToBottom])

  // Show thread history or chat view
  if (showThreadHistory) {
    return (
      <aside className={cn("flex h-full w-full flex-col bg-background", className)}>
        <ThreadHistory />
      </aside>
    )
  }

  return (
    <aside data-testid="chat-sidebar" className={cn("flex h-full w-full flex-col bg-background", className)}>
      {/* Thread Header */}
      <ThreadHeader threadId={threadId} onNewThread={createNewThread} />

      {/* Messages Container with Centered Column */}
      <div className="thin-scrollbar flex-1 overflow-auto overscroll-y-auto pt-0 flex flex-col" role="log" aria-live="polite" aria-label="Chat messages">
        {/* Show TaskListView when tasks are visible */}
        {tasksVisible ? (
          <TaskListView />
        ) : (
          <>
            {/* Empty State - Centered in middle of chat */}
            {messages.length === 0 && !status && (
              <div data-testid="chat-empty-state" className="flex-1 flex items-center justify-center text-text-muted">
                <div className="text-center space-y-2">
                  <p className="text-lg font-medium">Welcome to AG3NT</p>
                  <p className="text-sm text-text-secondary">The agent will use tools (files/shell/web) when needed. You can reference paths like @src/foo.ts.</p>
                </div>
              </div>
            )}

            {/* Messages - only render container when there are messages */}
            {(messages.length > 0 || status) && (
              <div className="mx-auto w-full max-w-[670px] px-3">
                {/* Messages */}
                {messages.map((message, index) => (
                  <div key={message.id} className="w-full" style={{ opacity: 1, transform: 'none' }}>
                    <ChatMessage message={message} />
                  </div>
                ))}

                {/* Agent Status Indicator */}
                {status && (
                  <div className="w-full py-1">
                    <AgentStatus status={status} message={statusMessage} />
                  </div>
                )}

                {/* Bottom fade gradient */}
                <div className="from-background pointer-events-none sticky bottom-0 z-20 h-2 bg-gradient-to-t to-transparent" />

                <div ref={messagesEndRef} />
              </div>
            )}
          </>
        )}
      </div>

      {/* Input Container */}
      <div className="p-3 pt-0">
        <div className="mx-auto w-full max-w-[670px]">
          <ChatInput
            value={input}
            onChange={setInput}
            onSubmit={handleSubmit}
            isLoading={isLoading}
            placeholder="Plan, search, build anything..."
            autoApprove={autoApprove}
            onAutoApproveChange={setAutoApprove}
            planMode={planMode}
            onPlanModeChange={setPlanMode}
            planStatus={planStatus}
            planProgress={planProgress}
            onStop={stopAgent}
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            attachments={attachments}
            onAttachmentsChange={setAttachments}
          />
        </div>
      </div>
    </aside>
  )
}
