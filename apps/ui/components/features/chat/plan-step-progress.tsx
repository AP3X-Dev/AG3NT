"use client"

import { Loader2, CheckCircle2, XCircle } from "lucide-react"
import { cn } from "@/lib/utils"

interface PlanStepProgressProps {
  currentStep: number
  totalSteps: number
  description: string
  status: "running" | "completed" | "failed"
}

const statusConfig = {
  running: {
    icon: Loader2,
    iconClass: "animate-spin text-blue-400",
    textClass: "text-blue-400",
    barClass: "bg-blue-500",
    bgClass: "border-blue-500/20 bg-blue-500/5",
  },
  completed: {
    icon: CheckCircle2,
    iconClass: "text-emerald-400",
    textClass: "text-emerald-400",
    barClass: "bg-emerald-500",
    bgClass: "border-emerald-500/20 bg-emerald-500/5",
  },
  failed: {
    icon: XCircle,
    iconClass: "text-red-400",
    textClass: "text-red-400",
    barClass: "bg-red-500",
    bgClass: "border-red-500/20 bg-red-500/5",
  },
}

export function PlanStepProgress({
  currentStep,
  totalSteps,
  description,
  status,
}: PlanStepProgressProps) {
  const config = statusConfig[status]
  const Icon = config.icon
  const percentage = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0

  return (
    <div
      className={cn(
        "flex items-center gap-2.5 rounded-md border px-3 py-2 my-1",
        config.bgClass,
      )}
    >
      <Icon className={cn("h-4 w-4 shrink-0", config.iconClass)} />

      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-1.5">
          <span className={cn("text-sm font-medium", config.textClass)}>
            Step {currentStep}/{totalSteps}
          </span>
          <span className="text-sm text-text-secondary truncate">
            {description}
          </span>
        </div>

        {/* Progress bar */}
        <div className="mt-1 h-1 w-full rounded-full bg-white/5">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-300",
              config.barClass,
            )}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    </div>
  )
}
