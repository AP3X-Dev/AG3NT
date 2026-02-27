"""
Lane-Aware Priority Queue for AG3NT Autonomous System

Provides 4 priority lanes for scheduling work items so that system-critical
tasks preempt user and background work, preventing state races.

Lanes (lower number = higher priority):
    SYSTEM(0)     — internal bookkeeping, state transitions, shutdown
    SUBAGENT(1)   — delegated sub-agent results / requests
    USER(2)       — direct user actions and commands
    BACKGROUND(3) — telemetry, periodic sweeps, non-urgent housekeeping

The queue is thread-safe and async-friendly: callers may ``put`` from any
thread and ``await get()`` from an asyncio task.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("ag3nt.loops.lane_queue")


# ------------------------------------------------------------------
# Lane definitions
# ------------------------------------------------------------------

class Lane(IntEnum):
    """Priority lanes — lower value = higher priority."""

    SYSTEM = 0
    SUBAGENT = 1
    USER = 2
    BACKGROUND = 3


# ------------------------------------------------------------------
# Queue item
# ------------------------------------------------------------------

@dataclass(order=False)
class LaneItem:
    """
    A single item in the lane-aware queue.

    Ordering is determined by ``(lane, seq)`` so that items in a
    higher-priority lane always sort before lower-priority ones,
    and within the same lane items are FIFO by insertion order.
    """

    lane: Lane
    payload: Any
    item_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: float = field(default_factory=time.monotonic)

    # Internal sequence number assigned by the queue on insertion.
    # Not meant to be set by callers.
    _seq: int = field(default=0, repr=False, compare=False)

    # -- heap ordering helpers --

    def _sort_key(self) -> tuple[int, int]:
        return (int(self.lane), self._seq)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LaneItem):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, LaneItem):
            return NotImplemented
        return self._sort_key() <= other._sort_key()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, LaneItem):
            return NotImplemented
        return self._sort_key() > other._sort_key()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, LaneItem):
            return NotImplemented
        return self._sort_key() >= other._sort_key()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LaneItem):
            return NotImplemented
        return self._sort_key() == other._sort_key()

    def __hash__(self) -> int:
        return hash(self.item_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging / persistence."""
        return {
            "item_id": self.item_id,
            "lane": self.lane.name,
            "created_at": self.created_at,
            "payload": str(self.payload),
        }


# ------------------------------------------------------------------
# LaneAwareQueue
# ------------------------------------------------------------------

class LaneAwareQueue:
    """
    Thread-safe, async-compatible priority queue with 4 lanes.

    Usage::

        q = LaneAwareQueue()

        # Put items (sync, safe from any thread)
        q.put(Lane.USER, {"action": "run_tool", "name": "exec"})
        q.put(Lane.SYSTEM, "flush_state")

        # Get items (async — returns highest-priority item first)
        item = await q.get()          # will be the SYSTEM item
        item = await q.get()          # will be the USER item

        # Optional timeout
        item = await q.get(timeout=5.0)  # raises asyncio.TimeoutError
    """

    def __init__(self, max_size: int = 0) -> None:
        """
        Args:
            max_size: Maximum number of items across all lanes.
                      0 means unbounded.
        """
        self._max_size = max_size

        # Heap storage — list of LaneItem
        self._heap: list[LaneItem] = []

        # Monotonically increasing sequence counter for FIFO within a lane
        self._seq: int = 0

        # Per-lane counters for metrics / introspection
        self._lane_counts: dict[Lane, int] = {lane: 0 for lane in Lane}

        # Threading lock protects _heap, _seq, _lane_counts
        self._lock = threading.Lock()

        # Asyncio event to wake up blocked ``get()`` callers
        # Lazily initialised on first async access so the queue can be
        # created before an event loop exists.
        self._notify: Optional[asyncio.Event] = None

        # Total items ever enqueued (monotonic, never decremented)
        self._total_enqueued: int = 0

        logger.debug("LaneAwareQueue created (max_size=%s)", max_size)

    # ------------------------------------------------------------------
    # Lazy async wiring
    # ------------------------------------------------------------------

    def _ensure_notify(self) -> asyncio.Event:
        """Return (and lazily create) the asyncio.Event used for signalling."""
        if self._notify is None:
            self._notify = asyncio.Event()
        return self._notify

    # ------------------------------------------------------------------
    # Put (synchronous, thread-safe)
    # ------------------------------------------------------------------

    def put(self, lane: Lane, payload: Any, item_id: Optional[str] = None) -> LaneItem:
        """
        Enqueue *payload* into *lane*.

        This method is synchronous and safe to call from any thread.

        Args:
            lane: The priority lane.
            payload: Arbitrary data to enqueue.
            item_id: Optional explicit item id. Auto-generated if omitted.

        Returns:
            The ``LaneItem`` that was enqueued.

        Raises:
            QueueFull: If ``max_size`` is set and the queue is at capacity.
        """
        item = LaneItem(lane=lane, payload=payload)
        if item_id is not None:
            item.item_id = item_id

        with self._lock:
            if self._max_size > 0 and len(self._heap) >= self._max_size:
                raise QueueFull(
                    f"LaneAwareQueue is full ({self._max_size} items)"
                )

            self._seq += 1
            item._seq = self._seq
            heapq.heappush(self._heap, item)
            self._lane_counts[lane] += 1
            self._total_enqueued += 1

        # Wake any blocked async consumer.  This is safe to call from a
        # non-async thread — ``asyncio.Event.set()`` is thread-safe in
        # CPython when used with the GIL.  We intentionally do this
        # outside the lock to avoid holding it while signalling.
        notify = self._notify
        if notify is not None:
            notify.set()

        logger.debug(
            "Enqueued item %s in lane %s (queue_size=%d)",
            item.item_id,
            lane.name,
            self.qsize(),
        )
        return item

    # ------------------------------------------------------------------
    # Get (async)
    # ------------------------------------------------------------------

    async def get(self, timeout: Optional[float] = None) -> LaneItem:
        """
        Dequeue and return the highest-priority item.

        Blocks until an item is available or *timeout* seconds elapse.

        Args:
            timeout: Maximum seconds to wait.  ``None`` means wait forever.

        Returns:
            The highest-priority ``LaneItem``.

        Raises:
            asyncio.TimeoutError: If *timeout* expires before an item
                arrives.
        """
        notify = self._ensure_notify()

        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            # Try to pop under lock
            with self._lock:
                if self._heap:
                    item = heapq.heappop(self._heap)
                    self._lane_counts[item.lane] -= 1

                    # Clear the event if the heap is now empty so that the
                    # next get() will block.
                    if not self._heap:
                        notify.clear()

                    logger.debug(
                        "Dequeued item %s from lane %s (queue_size=%d)",
                        item.item_id,
                        item.lane.name,
                        len(self._heap),
                    )
                    return item

            # Heap was empty — wait for a signal from put().
            notify.clear()

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise asyncio.TimeoutError(
                        "LaneAwareQueue.get() timed out"
                    )
                try:
                    await asyncio.wait_for(notify.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise asyncio.TimeoutError(
                        "LaneAwareQueue.get() timed out"
                    )
            else:
                await notify.wait()

    # ------------------------------------------------------------------
    # Non-blocking get
    # ------------------------------------------------------------------

    def get_nowait(self) -> LaneItem:
        """
        Return the highest-priority item without blocking.

        Raises:
            QueueEmpty: If the queue has no items.
        """
        with self._lock:
            if not self._heap:
                raise QueueEmpty("LaneAwareQueue is empty")
            item = heapq.heappop(self._heap)
            self._lane_counts[item.lane] -= 1

            notify = self._notify
            if notify is not None and not self._heap:
                notify.clear()

            return item

    # ------------------------------------------------------------------
    # Peek
    # ------------------------------------------------------------------

    def peek(self) -> Optional[LaneItem]:
        """Return the next item without removing it, or ``None``."""
        with self._lock:
            if self._heap:
                return self._heap[0]
            return None

    # ------------------------------------------------------------------
    # Drain helpers
    # ------------------------------------------------------------------

    def drain(self, lane: Optional[Lane] = None) -> list[LaneItem]:
        """
        Remove and return all items, optionally filtered to a single *lane*.

        Returns items sorted by priority (highest first).
        """
        with self._lock:
            if lane is None:
                items = sorted(self._heap)
                self._heap.clear()
                for l in Lane:
                    self._lane_counts[l] = 0
            else:
                kept: list[LaneItem] = []
                removed: list[LaneItem] = []
                for it in self._heap:
                    if it.lane == lane:
                        removed.append(it)
                    else:
                        kept.append(it)
                heapq.heapify(kept)
                self._heap = kept
                self._lane_counts[lane] = 0
                items = sorted(removed)

            notify = self._notify
            if notify is not None and not self._heap:
                notify.clear()

        return items

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self, item_id: str) -> bool:
        """
        Remove a specific item by its ``item_id``.

        Returns ``True`` if the item was found and removed.
        """
        with self._lock:
            for i, it in enumerate(self._heap):
                if it.item_id == item_id:
                    self._lane_counts[it.lane] -= 1
                    # Remove and re-heapify
                    self._heap[i] = self._heap[-1]
                    self._heap.pop()
                    if self._heap:
                        heapq.heapify(self._heap)

                    notify = self._notify
                    if notify is not None and not self._heap:
                        notify.clear()

                    logger.debug("Cancelled item %s", item_id)
                    return True
        return False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def qsize(self) -> int:
        """Return the current number of items in the queue."""
        with self._lock:
            return len(self._heap)

    def empty(self) -> bool:
        """Return ``True`` if the queue is empty."""
        return self.qsize() == 0

    def lane_sizes(self) -> dict[str, int]:
        """Return per-lane item counts."""
        with self._lock:
            return {lane.name: self._lane_counts[lane] for lane in Lane}

    def get_metrics(self) -> dict[str, Any]:
        """Return queue metrics for monitoring / dashboards."""
        with self._lock:
            return {
                "queue_size": len(self._heap),
                "max_size": self._max_size,
                "total_enqueued": self._total_enqueued,
                "lanes": {
                    lane.name: self._lane_counts[lane] for lane in Lane
                },
            }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.qsize()

    def __bool__(self) -> bool:
        return not self.empty()

    def __repr__(self) -> str:
        return (
            f"LaneAwareQueue(size={self.qsize()}, "
            f"lanes={self.lane_sizes()})"
        )


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class QueueFull(Exception):
    """Raised when the queue is at capacity."""


class QueueEmpty(Exception):
    """Raised when ``get_nowait()`` is called on an empty queue."""
