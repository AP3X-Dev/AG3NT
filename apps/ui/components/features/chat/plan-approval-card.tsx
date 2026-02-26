"use client"

import { useState } from "react"
import { ClipboardList, ChevronDown, ChevronRight, Check, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { useChat } from "@/providers/chat-provider"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

interface PlanApprovalCardProps {
  planPath: string
  planContent: string
  summary: string
  stepCount: number
  checkpointCount: number
}

type ApprovalState = "pending" | "accepted" | "rejected"

export function PlanApprovalCard({
  planPath,
  planContent,
  summary,
  stepCount,
  checkpointCount,
}: PlanApprovalCardProps) {
  const { decidePlanApproval } = useChat()
  const [approvalState, setApprovalState] = useState<ApprovalState>("pending")
  const [expanded, setExpanded] = useState(false)
  const [showRejectInput, setShowRejectInput] = useState(false)
  const [rejectReason, setRejectReason] = useState("")

  const handleAccept = async () => {
    setApprovalState("accepted")
    await decidePlanApproval("approve")
  }

  const handleReject = async () => {
    setApprovalState("rejected")
    await decidePlanApproval("reject", rejectReason || undefined)
  }

  return (
    <div
      className={cn(
        "rounded-lg border p-4 my-2 transition-colors",
        approvalState === "accepted" &&
          "border-emerald-500/30 bg-emerald-500/5",
        approvalState === "rejected" &&
          "border-red-500/30 bg-red-500/5",
        approvalState === "pending" &&
          "border-blue-500/20 bg-blue-500/5",
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-2">
        <ClipboardList className="h-5 w-5 text-blue-400 shrink-0" />
        <span className="font-semibold text-text-primary text-sm">
          Plan Ready for Review
        </span>
        <span className="ml-auto flex items-center gap-1.5">
          <span className="rounded-full bg-blue-500/10 border border-blue-500/20 px-2 py-0.5 text-xs text-blue-400">
            {stepCount} {stepCount === 1 ? "step" : "steps"}
          </span>
          <span className="rounded-full bg-purple-500/10 border border-purple-500/20 px-2 py-0.5 text-xs text-purple-400">
            {checkpointCount} {checkpointCount === 1 ? "checkpoint" : "checkpoints"}
          </span>
        </span>
      </div>

      {/* Filename */}
      <p className="text-xs text-text-muted mb-2 font-mono truncate">
        {planPath}
      </p>

      {/* Summary */}
      <p className="text-sm text-text-secondary mb-3">{summary}</p>

      {/* Expandable plan content */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-xs text-text-muted hover:text-text-secondary transition-colors mb-2"
      >
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5" />
        )}
        {expanded ? "Hide plan details" : "Show plan details"}
      </button>

      {expanded && (
        <div className="rounded-md border border-border bg-surface p-3 mb-3 text-sm prose prose-invert prose-sm max-w-none overflow-auto max-h-96">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {planContent}
          </ReactMarkdown>
        </div>
      )}

      {/* Actions */}
      {approvalState === "pending" && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <Button
              variant="success"
              size="sm"
              onClick={handleAccept}
            >
              <Check className="h-4 w-4 mr-1" />
              Accept Plan
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowRejectInput(!showRejectInput)}
            >
              <X className="h-4 w-4 mr-1" />
              Reject
            </Button>
          </div>

          {showRejectInput && (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                placeholder="Reason (optional)"
                className="flex-1 rounded-md border border-border bg-surface px-3 py-1.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:ring-1 focus:ring-blue-500/40"
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleReject()
                }}
              />
              <Button variant="danger" size="sm" onClick={handleReject}>
                Confirm Reject
              </Button>
            </div>
          )}
        </div>
      )}

      {approvalState === "accepted" && (
        <div className="flex items-center gap-2 text-sm text-emerald-400">
          <Check className="h-4 w-4" />
          Plan approved
        </div>
      )}

      {approvalState === "rejected" && (
        <div className="flex items-center gap-2 text-sm text-red-400">
          <X className="h-4 w-4" />
          Plan rejected{rejectReason ? `: ${rejectReason}` : ""}
        </div>
      )}
    </div>
  )
}
