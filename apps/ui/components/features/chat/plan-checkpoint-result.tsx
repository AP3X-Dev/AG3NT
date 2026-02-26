"use client"

import { ShieldCheck, ShieldAlert } from "lucide-react"
import { cn } from "@/lib/utils"

interface PlanCheckpointResultProps {
  stepNumber: number
  passed: boolean
  summary?: string
}

export function PlanCheckpointResult({
  stepNumber,
  passed,
  summary,
}: PlanCheckpointResultProps) {
  return (
    <div
      className={cn(
        "flex items-start gap-2.5 rounded-md border px-3 py-2 my-1",
        passed
          ? "border-emerald-500/20 bg-emerald-500/5"
          : "border-amber-500/20 bg-amber-500/5",
      )}
    >
      {passed ? (
        <ShieldCheck className="h-4 w-4 shrink-0 text-emerald-400 mt-0.5" />
      ) : (
        <ShieldAlert className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
      )}

      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm text-text-secondary">
            Checkpoint after Step {stepNumber}
          </span>
          <span
            className={cn(
              "rounded-full px-2 py-0.5 text-xs font-medium",
              passed
                ? "bg-emerald-500/10 border border-emerald-500/20 text-emerald-400"
                : "bg-amber-500/10 border border-amber-500/20 text-amber-400",
            )}
          >
            {passed ? "PASS" : "FAIL"}
          </span>
        </div>

        {summary && (
          <p className="mt-1 text-xs text-text-muted">{summary}</p>
        )}
      </div>
    </div>
  )
}
