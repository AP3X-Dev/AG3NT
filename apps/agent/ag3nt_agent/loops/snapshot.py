"""SnapshotAutoTrigger — auto git snapshots for safety.

Creates lightweight git tags before risky operations.
Max 20 snapshots with rotation (oldest deleted first).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from ag3nt_agent.autonomous.event_bus import Event, EventBus, EventPriority

logger = logging.getLogger("ag3nt.loops.snapshot")


class SnapshotReason(str, Enum):
    PRE_BATCH_EDIT = "pre-batch-edit"
    KNOWN_GOOD = "known-good"
    PRE_DESTRUCTIVE = "pre-destructive"
    PRE_DELEGATION = "pre-delegation"
    PRE_RISKY = "pre-risky"
    MANUAL = "manual"


@dataclass
class SnapshotConfig:
    enabled: bool = True
    max_snapshots: int = 20
    tag_prefix: str = "ag3nt-snapshot"


class SnapshotAutoTrigger:
    """Auto-creates git tag snapshots before risky operations."""

    def __init__(self, bus: EventBus, config: SnapshotConfig | None = None) -> None:
        self._bus = bus
        self._config = config or SnapshotConfig()
        self._sub_ids: list[str] = []

    async def start(self) -> None:
        """Subscribe to events that should trigger snapshots."""
        if not self._config.enabled:
            return
        # Subscribe to batch edit and destructive events
        for event_type in ("tool.called",):
            self._sub_ids.append(
                self._bus.subscribe(self._on_tool_event, event_types={event_type})
            )
        logger.info("SnapshotAutoTrigger started")

    async def stop(self) -> None:
        for sub_id in self._sub_ids:
            self._bus.unsubscribe(sub_id)
        self._sub_ids.clear()

    async def _on_tool_event(self, event: Event) -> None:
        """Check if a tool call warrants a snapshot."""
        tool_name = event.payload.get("tool_name", "")
        destructive = {"delete_file", "exec_command", "git_reset", "apply_patch"}
        batch = {"multi_edit", "batch"}
        if tool_name in destructive:
            await self.create_snapshot(SnapshotReason.PRE_DESTRUCTIVE)
        elif tool_name in batch:
            await self.create_snapshot(SnapshotReason.PRE_BATCH_EDIT)

    async def create_snapshot(self, reason: SnapshotReason, label: str = "") -> str | None:
        """Create a git tag snapshot. Returns tag name or None."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tag_name = f"{self._config.tag_prefix}-{reason.value}-{timestamp}"
        if label:
            tag_name += f"-{label}"

        try:
            result = subprocess.run(
                ["git", "tag", tag_name],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                logger.warning("Snapshot tag failed: %s", result.stderr.strip())
                return None

            logger.info("Snapshot created: %s", tag_name)

            # Emit event
            self._bus.emit_sync(Event(
                event_type="snapshot.created",
                source="snapshot:auto",
                payload={"tag": tag_name, "reason": reason.value},
                priority=EventPriority.LOW,
            ))

            # Rotate if needed
            await self._rotate_snapshots()
            return tag_name

        except FileNotFoundError:
            logger.debug("git not available, snapshot skipped")
            return None
        except subprocess.TimeoutExpired:
            logger.warning("git tag timed out")
            return None
        except Exception as exc:
            logger.warning("Snapshot error: %s", exc)
            return None

    async def _rotate_snapshots(self) -> None:
        """Delete oldest snapshots if over max."""
        try:
            result = subprocess.run(
                ["git", "tag", "-l", f"{self._config.tag_prefix}-*"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return
            tags = sorted(result.stdout.strip().split("\n"))
            tags = [t for t in tags if t]  # filter empty
            while len(tags) > self._config.max_snapshots:
                oldest = tags.pop(0)
                subprocess.run(
                    ["git", "tag", "-d", oldest],
                    capture_output=True, timeout=10,
                )
                logger.debug("Rotated snapshot: %s", oldest)
        except Exception as exc:
            logger.debug("Snapshot rotation error: %s", exc)

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": self._config.enabled,
            "max_snapshots": self._config.max_snapshots,
        }
