"use client"

import { memo, useMemo } from 'react'
import { Terminal, FileDiff, Image as ImageIcon, ExternalLink, FileText, Paperclip } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Message, FileAttachment, CLIContent } from '@/types'
import { FilePreview } from './file-preview'
import { CommandOutput } from './command-output'
import { ToolCallDisplay } from './tool-call-display'
import { ToolCallGroup } from './tool-call-group'
import { CodeBlock } from './code-block'
import { PlanApprovalCard } from './plan-approval-card'
import { PlanStepProgress } from './plan-step-progress'
import { PlanCheckpointResult } from './plan-checkpoint-result'
import { PlanExecutionSummary } from './plan-execution-summary'
import { useChat } from '@/providers/chat-provider'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// ============================================================================
// Tool Call Grouping Logic
// ============================================================================

type ContentGroup =
  | { type: 'single'; item: CLIContent; index: number }
  | { type: 'collapsed'; items: CLIContent[]; startIndex: number }

function isToolLike(item: CLIContent): boolean {
  return (
    item.type === 'tool-call' ||
    item.type === 'tool-output' ||
    item.type === 'command-output'
  )
}

function isPending(item: CLIContent): boolean {
  return isToolLike(item) && item.status === 'pending'
}

function groupConsecutiveToolCalls(
  contents: CLIContent[],
  minGroupSize: number = 3
): ContentGroup[] {
  const groups: ContentGroup[] = []
  let runBuffer: { item: CLIContent; index: number }[] = []
  let toolCountInRun = 0

  function flushRun() {
    if (runBuffer.length === 0) return

    if (toolCountInRun >= minGroupSize) {
      // Emit as a collapsed group
      groups.push({
        type: 'collapsed',
        items: runBuffer.map((b) => b.item),
        startIndex: runBuffer[0].index,
      })
    } else {
      // Emit each item individually
      for (const entry of runBuffer) {
        groups.push({ type: 'single', item: entry.item, index: entry.index })
      }
    }

    runBuffer = []
    toolCountInRun = 0
  }

  for (let i = 0; i < contents.length; i++) {
    const item = contents[i]

    // Pending tool calls should never be grouped — they're actively running
    if (isPending(item)) {
      flushRun()
      groups.push({ type: 'single', item, index: i })
      continue
    }

    if (isToolLike(item)) {
      // Tool-like item always extends the current run
      runBuffer.push({ item, index: i })
      toolCountInRun++
      continue
    }

    if (item.type === 'text') {
      // Text extends the run only if there's already at least one tool in it
      if (toolCountInRun > 0) {
        runBuffer.push({ item, index: i })
        continue
      }
      // Otherwise this text is standalone — flush any pending run and emit individually
      flushRun()
      groups.push({ type: 'single', item, index: i })
      continue
    }

    // Non-tool, non-text items (file-content, file-diff, image, error) break the run
    flushRun()
    groups.push({ type: 'single', item, index: i })
  }

  // Flush any remaining run
  flushRun()

  return groups
}

interface ChatMessageProps {
  message: Message
  className?: string
  onOpenFile?: (path: string) => void
}

// Filter out XML function call tags and internal markers from assistant content
// Only removes complete XML blocks - not aggressive with fragments
function filterXMLTags(content: string): string {
  let filtered = content

  // Remove complete antml: function call blocks
  filtered = filtered.replace(/<function_calls>[\s\S]*?<\/antml:function_calls>/gi, '')
  filtered = filtered.replace(/<function_calls>[\s\S]*?<\/antml:function_calls>/gi, '')
  filtered = filtered.replace(/<function_calls>[\s\S]*?<\/function_calls>/gi, '')

  // Remove standalone antml tags if they appear
  filtered = filtered.replace(/<\/?antml:function_calls>/gi, '')
  filtered = filtered.replace(/<\/?antml:invoke[^>]*>/gi, '')
  filtered = filtered.replace(/<\/?antml:parameter[^>]*>/gi, '')

  // Remove tool_calls_end tags and similar internal markers
  filtered = filtered.replace(/<\/?tool_calls_end>/gi, '')
  filtered = filtered.replace(/<\/?tool_calls>/gi, '')

  // Remove internal tool execution metadata lines (e.g., "Researcher\ndescription: ...")
  // This catches leaked internal tool descriptions
  filtered = filtered.replace(/^[A-Za-z_]+\s*\ndescription:\s*[^\n]*\.{3,}$/gm, '')

  // Remove "Internet Search\nquery: ..." type patterns
  filtered = filtered.replace(/^(Internet Search|Researcher|Web Search)\s*\nquery:\s*[^\n]*$/gm, '')

  // Clean up excessive whitespace
  return filtered.replace(/\n{3,}/g, '\n\n').trim()
}

// Markdown renderer component with custom styling
function MarkdownContent({ content }: { content: string }) {
  // Filter XML tags before rendering — memoize to avoid expensive regex on every render
  const cleanContent = useMemo(() => filterXMLTags(content), [content])
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      className="prose prose-invert max-w-none"
      components={{
        // Headings
        h1: ({ children }) => (
          <h1 className="text-2xl font-semibold mt-6 mb-3 flex items-center gap-2">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-xl font-semibold mt-5 mb-2 flex items-center gap-2">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-lg font-semibold mt-4 mb-2 flex items-center gap-2">
            {children}
          </h3>
        ),
        h4: ({ children }) => (
          <h4 className="text-base font-semibold mt-3 mb-1.5 flex items-center gap-2">
            {children}
          </h4>
        ),
        // Paragraphs
        p: ({ children }) => (
          <p className="mb-3 leading-relaxed">{children}</p>
        ),
        // Lists
        ul: ({ children }) => (
          <ul className="list-disc list-inside mb-3 space-y-1 ml-2">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal list-inside mb-3 space-y-1 ml-2">{children}</ol>
        ),
        li: ({ children }) => (
          <li className="leading-relaxed">{children}</li>
        ),
        // Inline code
        code: ({ inline, className, children, ...props }: any) => {
          if (inline) {
            return (
              <code
                className="bg-[#1a1a1a] px-1.5 py-0.5 rounded text-[13px] font-mono text-[#a5d6ff]"
                {...props}
              >
                {children}
              </code>
            )
          }
          // Block code is handled by pre
          return <code {...props}>{children}</code>
        },
        // Code blocks with syntax highlighting
        pre: ({ children, ...props }: any) => {
          // Extract code content and language from code child
          const codeChild = (children as any)?.props
          const className = codeChild?.className || ''
          const match = /language-(\w+)/.exec(className)
          const language = match ? match[1] : ''

          // Get the actual code text
          const codeContent = String(codeChild?.children || children || '')

          return (
            <CodeBlock
              code={codeContent}
              language={language}
            />
          )
        },
        // Blockquotes
        blockquote: ({ children }) => (
          <blockquote className="border-l-4 border-blue-500 pl-4 py-2 mb-3 italic text-text-muted">
            {children}
          </blockquote>
        ),
        // Links
        a: ({ href, children }) => (
          <a
            href={href}
            className="text-blue-400 hover:text-blue-300 underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            {children}
          </a>
        ),
        // Images
        img: ({ src, alt }) => (
          <div className="my-3">
            <img
              src={src}
              alt={alt}
              className="max-w-full rounded-lg shadow-lg"
              style={{ maxHeight: '400px' }}
            />
          </div>
        ),
        // Horizontal rule
        hr: () => <hr className="my-4 border-border" />,
        // Strong/Bold
        strong: ({ children }) => (
          <strong className="font-semibold text-text-primary">{children}</strong>
        ),
        // Emphasis/Italic
        em: ({ children }) => (
          <em className="italic">{children}</em>
        ),
        // Tables
        table: ({ children }) => (
          <div className="overflow-x-auto mb-3">
            <table className="min-w-full border border-border rounded-lg">
              {children}
            </table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-surface-secondary">{children}</thead>
        ),
        tbody: ({ children }) => <tbody>{children}</tbody>,
        tr: ({ children }) => (
          <tr className="border-b border-border">{children}</tr>
        ),
        th: ({ children }) => (
          <th className="px-4 py-2 text-left font-semibold">{children}</th>
        ),
        td: ({ children }) => (
          <td className="px-4 py-2">{children}</td>
        ),
        // Task lists (GFM)
        input: ({ checked, ...props }: any) => (
          <input
            type="checkbox"
            checked={checked}
            disabled
            className="mr-2 align-middle"
            {...props}
          />
        ),
      }}
    >
      {cleanContent}
    </ReactMarkdown>
  )
}

// Main ChatMessage component - memoized for performance
const ChatMessageComponent = ({ message, className, onOpenFile }: ChatMessageProps) => {
  const { decideApproval, isLoading } = useChat()
  const isUser = message.role === 'user'
  const hasCLIContent = message.cliContent && message.cliContent.length > 0
  const hasFileMentions = message.fileMentions && message.fileMentions.length > 0
  const hasImage = message.content?.includes('![') && message.content?.includes('](')
  const hasApproval = !!message.approvalRequest

  // Group consecutive tool calls for collapsed rendering
  const contentGroups = useMemo(() => {
    if (!message.cliContent) return []
    return groupConsecutiveToolCalls(message.cliContent, 3)
  }, [message.cliContent])

  // Format timestamp for user messages
  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    })
  }

  return (
    <div className={cn('py-1', isUser ? 'flex flex-col items-end' : '', className)}>
      {/* Timestamp for user messages */}
      {isUser && message.timestamp && (
        <div className="text-[11px] text-text-muted mb-1 px-1">
          {formatTimestamp(message.timestamp)}
        </div>
      )}

      <div
        className={cn(
          isUser
            ? 'max-w-[85%] rounded-lg bg-[#1e4976] text-white px-3 py-2 text-[13px]'
            : 'w-full text-text-primary text-[13px] leading-relaxed',
        )}
      >
        {/* User messages: render content directly */}
        {isUser && message.content && (
          <div className={cn(
            (hasCLIContent || hasFileMentions || message.attachments?.length) && "px-1 pb-2"
          )}>
            <p className="whitespace-pre-wrap">{filterXMLTags(message.content)}</p>
          </div>
        )}

        {/* User message attachments */}
        {isUser && message.attachments && message.attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {message.attachments.map((attachment) => (
              <MessageAttachment key={attachment.id} attachment={attachment} />
            ))}
          </div>
        )}

        {/* Assistant messages without cliContent: render content with markdown */}
        {!isUser && !hasCLIContent && message.content && (
          <div>
            <MarkdownContent content={message.content} />
          </div>
        )}

        {/* File mentions */}
        {hasFileMentions && (
          <div className="space-y-2 mt-2">
            {message.fileMentions!.map((file, index) => (
              <FilePreview
                key={`${file.path}-${index}`}
                path={file.path}
                maxLines={8}
                onOpenFile={onOpenFile}
              />
            ))}
          </div>
        )}

        {/* CLI content: render in chronological order (interleaved text + tools) */}
        {/* Groups of 3+ consecutive tool calls are collapsed into a summary bar */}
        {hasCLIContent && (
          <div>
            {contentGroups.map((group, groupIdx) => {
              // Collapsed group of tool calls
              if (group.type === 'collapsed') {
                return (
                  <div key={`group-${group.startIndex}`} className={groupIdx > 0 ? 'mt-2' : ''}>
                    <ToolCallGroup items={group.items} onOpenFile={onOpenFile} />
                  </div>
                )
              }

              // Single item — render exactly as before
              const content = group.item
              const index = group.index
              const isToolCall = content.type === 'tool-output' || content.type === 'tool-call'
              const spacingClass = groupIdx > 0 ? (isToolCall ? 'mt-2' : 'mt-3') : ''

              // Text content - render with markdown (interleaved with tools)
              if (content.type === 'text') {
                return (
                  <div key={`text-${index}`} className={spacingClass}>
                    <MarkdownContent content={content.content} />
                  </div>
                )
              }

              if (content.type === 'tool-output' || content.type === 'tool-call') {
                // Use unique key based on title (toolCallId) for smooth transitions
                const toolKey = content.title || content.toolName || `tool-${index}`
                return (
                  <div key={toolKey} className={spacingClass}>
                    <ToolCallDisplay
                      toolName={content.toolName || content.title || 'tool'}
                      args={content.args}
                      output={content.content}
                      status={content.status}
                      error={content.error}
                    />
                  </div>
                )
              }

              if (content.type === 'command-output') {
                return (
                  <div key={`cmd-${index}`} className={spacingClass}>
                    <CommandOutput
                      command={content.command || content.title || 'command'}
                      output={content.content}
                      exitCode={content.exitCode ?? 0}
                      executionTime={content.executionTime}
                    />
                  </div>
                )
              }

              if (content.type === 'file-content') {
                return (
                  <div key={`file-${index}`} className={spacingClass}>
                    <FilePreview
                      path={content.path || content.title || 'file'}
                      initialContent={content.content}
                      initialLanguage={content.language}
                      maxLines={10}
                      onOpenFile={onOpenFile}
                    />
                  </div>
                )
              }

              if (content.type === 'file-diff') {
                return (
                  <div
                    key={`diff-${index}`}
                    className={cn("rounded-lg border border-[#2a2a2a] bg-[#1a1a1a] p-3", spacingClass)}
                  >
                    <div className="flex items-center gap-2 text-text-primary text-sm font-medium mb-2">
                      <FileDiff className="h-4 w-4 text-text-secondary" />
                      {content.title || content.path || 'File changes'}
                    </div>
                    <pre className="text-sm font-mono overflow-x-auto whitespace-pre-wrap bg-[#0f0f0f] border border-[#2a2a2a] rounded p-2">
                      {content.content.split('\n').map((line, i) => {
                        const isAdd = line.startsWith('+') && !line.startsWith('+++')
                        const isRemove = line.startsWith('-') && !line.startsWith('---')
                        return (
                          <div
                            key={i}
                            className={cn(
                              isAdd && 'text-[#69db7c] bg-[#1f3d1f]',
                              isRemove && 'text-[#ff6b6b] bg-[#3d1f1f]',
                              !isAdd && !isRemove && 'text-text-muted'
                            )}
                          >
                            {line}
                          </div>
                        )
                      })}
                    </pre>
                  </div>
                )
              }

              if (content.type === 'image') {
                // Render generated image inline
                const imagePath = content.imagePath || ''
                // Create a file:// URL or use API route to serve the image
                const imageUrl = imagePath ? `/api/file?path=${encodeURIComponent(imagePath)}` : ''
                return (
                  <div
                    key={`image-${index}`}
                    className={cn("rounded-md border border-[#2a2a2a] bg-[#1a1a1a] p-3", spacingClass)}
                  >
                    <div className="flex items-center gap-2 text-gray-300 text-sm font-medium mb-2">
                      <ImageIcon className="h-4 w-4" />
                      <span>Generated Image</span>
                      {imagePath && (
                        <a
                          href={imageUrl}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ml-auto flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300"
                        >
                          <ExternalLink className="h-3 w-3" />
                          Open
                        </a>
                      )}
                    </div>
                    {imageUrl && (
                      <img
                        src={imageUrl}
                        alt="Generated image"
                        className="max-w-full rounded-lg shadow-lg max-h-[400px] object-contain"
                        onError={(e) => {
                          // Hide broken image and show path instead
                          (e.target as HTMLImageElement).style.display = 'none'
                        }}
                      />
                    )}
                    <div className="mt-2 text-xs text-gray-500 truncate" title={imagePath}>
                      {imagePath}
                    </div>
                  </div>
                )
              }

              if (content.type === 'error') {
                return (
                  <div
                    key={`error-${index}`}
                    className={cn("rounded-md border border-[#3a2a2a] bg-[#1E1E1E] p-3", spacingClass)}
                  >
                    <div className="flex items-center gap-2 text-red-400 text-sm font-medium mb-1">
                      <Terminal className="h-4 w-4" />
                      {content.title || 'Error'}
                    </div>
                    <pre className="text-sm font-mono text-red-300 whitespace-pre-wrap">
                      {content.content}
                    </pre>
                  </div>
                )
              }

              // Default: render as code block
              return (
                <div key={`content-${index}`} className={spacingClass}>
                  <pre className="rounded-md bg-surface-secondary p-3 text-sm font-mono overflow-x-auto">
                    <code>{content.content}</code>
                  </pre>
                </div>
              )
            })}
          </div>
        )}

        {/* Plan execution components */}
        {message.planApproval && (
          <PlanApprovalCard
            planPath={message.planApproval.planPath}
            planContent={message.planApproval.planContent}
            summary={message.planApproval.summary}
            stepCount={message.planApproval.stepCount}
            checkpointCount={message.planApproval.checkpointCount}
          />
        )}
        {message.planProgress && (
          <PlanStepProgress
            currentStep={message.planProgress.currentStep}
            totalSteps={message.planProgress.totalSteps}
            description={message.planProgress.description}
            status={message.planProgress.status}
          />
        )}
        {message.planCheckpoint && (
          <PlanCheckpointResult
            stepNumber={message.planCheckpoint.stepNumber}
            passed={message.planCheckpoint.passed}
            summary={message.planCheckpoint.summary}
          />
        )}
        {message.planSummary && (
          <PlanExecutionSummary
            success={message.planSummary.success}
            stepsCompleted={message.planSummary.stepsCompleted}
            totalSteps={message.planSummary.totalSteps}
          />
        )}

        {/* Approval request (human-in-the-loop) - Augment style inline */}
        {hasApproval && message.approvalRequest && (
          <div className="space-y-0.5">
            {/* Tool call rows with waiting text */}
            {message.approvalRequest.actionRequests.map((ar, idx) => (
              <ToolCallDisplay
                key={`approval-${ar.name}-${idx}`}
                toolName={ar.name}
                args={ar.args}
                status="pending"
                waitingForApproval={true}
                onApprove={() => decideApproval(message.approvalRequest!.interruptId, 'approve')}
                onSkip={() => decideApproval(message.approvalRequest!.interruptId, 'reject')}
              />
            ))}

            {/* Waiting indicator below last tool - only shown once for all tools */}
            <div className="flex items-center gap-2 py-1.5 text-[13px] text-text-muted">
              <div className="flex gap-1">
                <div className="w-1 h-4 bg-[#333333] rounded-sm animate-pulse" style={{ animationDelay: '0ms' }} />
                <div className="w-1 h-4 bg-[#333333] rounded-sm animate-pulse" style={{ animationDelay: '150ms' }} />
              </div>
              <span>Waiting for user input...</span>
            </div>
          </div>
        )}

      </div>
    </div>
  )
}

// ============================================================================
// Message Attachment Display Component
// ============================================================================

interface MessageAttachmentProps {
  attachment: FileAttachment
}

function MessageAttachment({ attachment }: MessageAttachmentProps) {
  const isImage = attachment.type.startsWith('image/')

  if (isImage && attachment.dataUrl) {
    return (
      <div className="relative group">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={attachment.dataUrl}
          alt={attachment.name}
          className="max-w-[200px] max-h-[150px] rounded-md border border-[#353535] object-cover cursor-pointer hover:opacity-90 transition-opacity"
          onClick={() => window.open(attachment.dataUrl, '_blank')}
        />
        <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-[10px] text-white px-1.5 py-0.5 rounded-b-md truncate">
          {attachment.name}
        </div>
      </div>
    )
  }

  // Non-image file
  return (
    <div className="flex items-center gap-2 px-2 py-1.5 bg-[#1a1a1a] border border-[#353535] rounded-md max-w-[180px]">
      <FileText className="h-4 w-4 text-text-muted flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <div className="text-[11px] text-text-primary truncate">{attachment.name}</div>
        <div className="text-[10px] text-text-muted">{formatFileSize(attachment.size)}</div>
      </div>
    </div>
  )
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

// Export memoized version for performance
export const ChatMessage = memo(ChatMessageComponent, (prevProps, nextProps) => {
  // Only re-render if message content or ID changes
  return (
    prevProps.message.id === nextProps.message.id &&
    prevProps.message.content === nextProps.message.content &&
    prevProps.message.cliContent === nextProps.message.cliContent &&
    prevProps.message.approvalRequest === nextProps.message.approvalRequest &&
    prevProps.message.planApproval === nextProps.message.planApproval &&
    prevProps.message.planProgress === nextProps.message.planProgress &&
    prevProps.message.planCheckpoint === nextProps.message.planCheckpoint &&
    prevProps.message.planSummary === nextProps.message.planSummary
  )
})
