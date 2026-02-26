"use client"

import { CheckCircle2, XCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface PlanExecutionSummaryProps {
  success: boolean
  stepsCompleted: number
  totalSteps: number
}

export function PlanExecutionSummary({
  success,
  stepsCompleted,
  totalSteps,
}: PlanExecutionSummaryProps) {
  return (
    <div
      className={cn(
        "flex items-center gap-2.5 rounded-md border px-3 py-2 my-1",
        success
          ? "border-emerald-500/20 bg-emerald-500/5"
          : "border-red-500/20 bg-red-500/5",
      )}
    >
      {success ? (
        <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
      ) : (
        <XCircle className="h-4 w-4 shrink-0 text-red-400" />
      )}

      <div className="min-w-0">
        <span
          className={cn(
            "text-sm font-medium",
            success ? "text-emerald-400" : "text-red-400",
          )}
        >
          {success ? "Plan Execution Complete" : "Plan Execution Failed"}
        </span>
        <p className="text-xs text-text-muted">
          {stepsCompleted}/{totalSteps} steps completed
        </p>
      </div>
    </div>
  )
}
