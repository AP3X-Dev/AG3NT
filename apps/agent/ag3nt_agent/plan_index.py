"""Cross-session plan index for searching and resuming plans."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

PLANS_DIR = Path.home() / ".ag3nt" / "plans"


@dataclass
class PlanSummary:
    """Summary of a plan for listing/search."""
    path: Path
    title: str
    status: str  # "active", "completed", "failed", "unknown"
    created: datetime | None
    task_count: int
    completed_count: int

    @property
    def plan_id(self) -> str:
        return self.path.stem


class PlanIndex:
    """Index of all plans across sessions."""

    def __init__(self, plans_dir: Path | None = None):
        self._plans_dir = plans_dir or PLANS_DIR

    def list_plans(self, status: str | None = None, limit: int = 10) -> list[PlanSummary]:
        """List plans by recency, optionally filtered by status."""
        if not self._plans_dir.exists():
            return []

        plans = []
        for path in sorted(self._plans_dir.glob("*.md"), reverse=True):
            summary = self._parse_plan(path)
            if status and status != "all" and summary.status != status:
                continue
            plans.append(summary)
            if len(plans) >= limit:
                break
        return plans

    def search_plans(self, query: str) -> list[PlanSummary]:
        """Search plans by keyword in title/content."""
        if not self._plans_dir.exists():
            return []

        query_lower = query.lower()
        results = []
        for path in self._plans_dir.glob("*.md"):
            try:
                content = path.read_text(encoding="utf-8").lower()
                if query_lower in content:
                    results.append(self._parse_plan(path))
            except Exception:
                continue
        return sorted(results, key=lambda p: p.path.name, reverse=True)

    def get_active_plans(self) -> list[PlanSummary]:
        """Get plans that are IN_PROGRESS / active."""
        return self.list_plans(status="active", limit=50)

    def get_plan_content(self, plan_id: str) -> str | None:
        """Read full plan content by plan_id (filename stem)."""
        path = self._plans_dir / f"{plan_id}.md"
        if not path.exists():
            # Try fuzzy match
            for p in self._plans_dir.glob(f"*{plan_id}*.md"):
                path = p
                break
            else:
                return None
        return path.read_text(encoding="utf-8")

    def _parse_plan(self, path: Path) -> PlanSummary:
        """Parse a plan file into a PlanSummary."""
        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return PlanSummary(path=path, title=path.stem, status="unknown",
                             created=None, task_count=0, completed_count=0)

        # Extract title from first heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem

        # Extract status
        # Handle both plain "Status: X" and markdown "**Status:** X" formats
        status = "unknown"
        status_match = re.search(r"\*{0,2}(?:status|Status)\*{0,2}[:\s*]+(\w+)", content)
        if status_match:
            raw = status_match.group(1).lower()
            if raw in ("completed", "done", "finished", "ready"):
                status = "completed"
            elif raw in ("failed", "error", "aborted"):
                status = "failed"
            elif raw in ("active", "in_progress", "inprogress", "started", "in"):
                status = "active"
        else:
            # No explicit status -- if file has unchecked tasks, it's active
            has_unchecked = bool(re.search(r"- \[ \]", content))
            has_checked = bool(re.search(r"- \[x\]", content, re.IGNORECASE))
            if has_unchecked:
                status = "active"
            elif has_checked:
                status = "completed"

        # Extract creation date from filename (YYYY-MM-DD or YYYYMMDD prefix)
        created = None
        date_match = re.match(r"(\d{4}-\d{2}-\d{2})", path.stem)
        if date_match:
            try:
                created = datetime.strptime(date_match.group(1), "%Y-%m-%d")
            except ValueError:
                pass
        else:
            date_match = re.match(r"(\d{4})(\d{2})(\d{2})", path.stem)
            if date_match:
                try:
                    created = datetime.strptime(
                        f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}",
                        "%Y-%m-%d",
                    )
                except ValueError:
                    pass

        # Count tasks (markdown checkboxes)
        tasks = re.findall(r"- \[[ x]\]", content, re.IGNORECASE)
        completed = re.findall(r"- \[x\]", content, re.IGNORECASE)

        return PlanSummary(
            path=path, title=title, status=status,
            created=created, task_count=len(tasks),
            completed_count=len(completed),
        )


# Singleton
_plan_index: PlanIndex | None = None


def get_plan_index() -> PlanIndex:
    global _plan_index
    if _plan_index is None:
        _plan_index = PlanIndex()
    return _plan_index
