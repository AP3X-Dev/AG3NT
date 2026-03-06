"use client"

import type { LucideIcon } from "lucide-react"
import { ChevronRight, Eye } from "lucide-react"
import { cn } from "@/lib/utils"

interface ModuleCardProps {
  id: string
  title: string
  description?: string
  icon: LucideIcon
  /** Full Tailwind bg class for accent stripe, e.g. "bg-blue-400" */
  accentColorClass: string
  /** Full Tailwind text class for icon, e.g. "text-blue-400" */
  iconColorClass: string
  isSelected?: boolean
  viewMode: "list" | "grid"
  onSelect: () => void
  status?: {
    label: string
    color: "emerald" | "gray" | "red" | "amber" | "blue"
  }
  chips?: Array<{ label: string; className?: string }>
  action?: React.ReactNode
}

const STATUS_DOT_COLORS = {
  emerald: "bg-emerald-400",
  gray: "bg-zinc-500",
  red: "bg-red-400",
  amber: "bg-amber-400",
  blue: "bg-blue-400",
} as const

const STATUS_TEXT_COLORS = {
  emerald: "text-emerald-400",
  gray: "text-text-muted",
  red: "text-red-400",
  amber: "text-amber-400",
  blue: "text-blue-400",
} as const

const ACCENT_DOT_COLORS: Record<string, string> = {
  "bg-blue-400": "bg-blue-400",
  "bg-purple-400": "bg-purple-400",
  "bg-emerald-400": "bg-emerald-400",
  "bg-orange-400": "bg-orange-400",
  "bg-cyan-400": "bg-cyan-400",
  "bg-amber-400": "bg-amber-400",
  "bg-zinc-400": "bg-zinc-400",
  "bg-teal-400": "bg-teal-400",
  "bg-pink-400": "bg-pink-400",
  "bg-zinc-500": "bg-zinc-500",
  "bg-red-400": "bg-red-400",
}

export function ModuleCard({
  title,
  description,
  icon: Icon,
  accentColorClass,
  iconColorClass,
  isSelected,
  viewMode,
  onSelect,
  status,
  chips,
  action,
}: ModuleCardProps) {
  const dotColor = ACCENT_DOT_COLORS[accentColorClass] || "bg-zinc-500"

  if (viewMode === "list") {
    return (
      <button
        onClick={onSelect}
        className={cn(
          "group w-full flex items-center gap-3 rounded-xl border p-3 text-left transition-all duration-150",
          "bg-surface-elevated border-border hover:ring-1 hover:ring-white/5",
          isSelected && "ring-1 ring-status-info/30 border-status-info/40"
        )}
      >
        <div className={cn("w-0.5 self-stretch rounded-full", accentColorClass)} />

        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-surface shrink-0">
          <Icon className={cn("w-4 h-4", iconColorClass)} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-text-primary truncate">{title}</span>
            {status && (
              <span className="flex items-center gap-1.5 shrink-0">
                <span className={cn("w-1.5 h-1.5 rounded-full", STATUS_DOT_COLORS[status.color])} />
                <span className={cn("text-[10px]", STATUS_TEXT_COLORS[status.color])}>{status.label}</span>
              </span>
            )}
          </div>
          {description && (
            <p className="text-xs text-text-muted truncate mt-0.5">{description}</p>
          )}
        </div>

        {chips && chips.length > 0 && (
          <div className="hidden sm:flex items-center gap-1 shrink-0">
            {chips.slice(0, 2).map((chip) => (
              <span key={chip.label} className={cn("text-[10px] px-2 py-0.5 rounded-full bg-surface border border-border text-text-muted", chip.className)}>
                {chip.label}
              </span>
            ))}
          </div>
        )}

        <div className="flex items-center gap-1 shrink-0" onClick={(e) => e.stopPropagation()}>
          {action}
        </div>
        <ChevronRight className="w-3.5 h-3.5 text-text-muted opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
      </button>
    )
  }

  // Grid view — inspired by dispatch card aesthetic
  return (
    <div
      onClick={onSelect}
      className={cn(
        "group relative flex flex-col min-w-0 overflow-hidden rounded-2xl border cursor-pointer transition-all duration-150",
        "bg-surface-elevated border-border hover:bg-interactive-hover hover:border-interactive-border hover:-translate-y-0.5",
        isSelected && "ring-1 ring-status-info/30 border-status-info/40"
      )}
    >
      {/* Top status bar */}
      <div className="flex items-center gap-2 px-4 pt-3.5 pb-1.5 min-w-0">
        {/* Accent dots */}
        <div className="flex items-center gap-1">
          <span className={cn("w-[7px] h-[7px] rounded-full", dotColor)} />
          <span className={cn("w-[7px] h-[7px] rounded-full opacity-60", dotColor)} />
          <span className={cn("w-[7px] h-[7px] rounded-full opacity-30", dotColor)} />
        </div>

        {/* Title */}
        <span className="text-sm font-semibold text-text-primary truncate flex-1 ml-1" title={title}>
          {title}
        </span>

        {/* Status indicator */}
        {status && (
          <span className="flex items-center gap-1.5 shrink-0">
            <span className={cn("w-1.5 h-1.5 rounded-full", STATUS_DOT_COLORS[status.color])} />
            <span className={cn("text-[10px] font-medium", STATUS_TEXT_COLORS[status.color])}>
              {status.label}
            </span>
          </span>
        )}
      </div>

      {/* Description body */}
      <div className="px-4 pb-2.5 pt-1">
        {description && (
          <p className="text-xs text-text-muted leading-relaxed line-clamp-2">
            {description}
          </p>
        )}
      </div>

      {/* Bottom action row */}
      <div className="flex items-center gap-1.5 px-4 pb-3 pt-1 min-w-0">
        <button
          className="text-[11px] font-medium px-3 py-1.5 rounded-full bg-surface border border-border text-text-secondary hover:bg-interactive-hover hover:text-text-primary transition-colors shrink-0"
          onClick={(e) => {
            e.stopPropagation()
            onSelect()
          }}
        >
          Details
        </button>

        {/* Chip badges — overflow hidden so they collapse on small cards */}
        {chips && chips.length > 0 && (
          <div className="flex items-center gap-1 ml-auto min-w-0 overflow-hidden">
            {chips.slice(0, 2).map((chip) => (
              <span
                key={chip.label}
                className={cn(
                  "text-[10px] px-2 py-1 rounded-full bg-surface border border-border text-text-muted truncate shrink-0",
                  chip.className
                )}
              >
                {chip.label}
              </span>
            ))}
          </div>
        )}

        {/* Custom action (delete, toggle, etc.) */}
        {action && (
          <div
            className="ml-auto flex items-center shrink-0"
            onClick={(e) => e.stopPropagation()}
          >
            {action}
          </div>
        )}
      </div>
    </div>
  )
}
