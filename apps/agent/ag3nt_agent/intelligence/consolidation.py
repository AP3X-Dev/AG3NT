"""
ConsolidationLoop — End-of-Session Memory Extraction.

Analyses session messages, segments them by topic, and extracts
durable memory fragments (preferences, decisions, corrections, facts).
Deduplicates against existing memories before persisting.

Subscribes to session lifecycle events on the EventBus.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from ag3nt_agent.autonomous.event_bus import Event, EventBus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MemorySegment:
    """A single extracted memory from session messages."""

    content: str
    category: str       # "preference" | "decision" | "correction" | "fact"
    confidence: float   # 0.0 .. 1.0


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# Each pattern maps to (category, compiled regex).
# The regex should capture the relevant sentence / clause.
_EXTRACTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("preference", re.compile(
        r"(?:^|[.!?]\s+)([^.!?]*?\b(?:prefer|always|never)\b[^.!?]*)",
        re.IGNORECASE,
    )),
    ("decision", re.compile(
        r"(?:^|[.!?]\s+)([^.!?]*?\b(?:decided|chose|went\s+with)\b[^.!?]*)",
        re.IGNORECASE,
    )),
    ("correction", re.compile(
        r"(?:^|[.!?]\s+)([^.!?]*?\b(?:actually|instead|not\s+\w+\s+but)\b[^.!?]*)",
        re.IGNORECASE,
    )),
    ("fact", re.compile(
        r"(?:^|[.!?]\s+)([^.!?]*?\b(?:remember|note\s+that|important:?)\b[^.!?]*)",
        re.IGNORECASE,
    )),
]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set[str]:
    """Return the set of lowercased words in *text*."""
    return set(re.findall(r"\w+", text.lower()))


def _word_overlap(a: str, b: str) -> float:
    """Compute Jaccard-style word overlap between two strings.

    Returns a float in ``[0.0, 1.0]``.
    """
    words_a = _word_set(a)
    words_b = _word_set(b)
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# ConsolidationLoop
# ---------------------------------------------------------------------------

class ConsolidationLoop:
    """End-of-session memory extraction loop.

    Subscribes to ``session.ended`` events on the EventBus.  When a
    session concludes, the loop consolidates the session messages into
    durable :class:`MemorySegment` objects.

    Usage::

        bus = EventBus()
        loop = ConsolidationLoop(bus)
        await loop.start()
        # ... at session end, messages are consolidated ...
        await loop.stop()
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._subscription_ids: list[str] = []
        self._running = False

        # Stats
        self._sessions_processed = 0
        self._memories_extracted = 0
        self._memories_deduped = 0

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to session lifecycle events."""
        if self._running:
            return

        self._running = True
        sub_id = self._bus.subscribe(
            handler=self._on_session_ended,
            event_types={"session.ended"},
        )
        self._subscription_ids.append(sub_id)

        logger.info("ConsolidationLoop started")

    async def stop(self) -> None:
        """Unsubscribe and shut down."""
        self._running = False
        for sub_id in self._subscription_ids:
            self._bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
        logger.info("ConsolidationLoop stopped")

    # -- event handler -------------------------------------------------------

    async def _on_session_ended(self, event: Event) -> None:
        """Handle session.ended — consolidate messages into memories."""
        payload = event.payload or {}
        messages = payload.get("messages", [])
        existing = payload.get("existing_memories", [])

        if not messages:
            return

        try:
            memories = await self.consolidate(messages)

            # Dedup against existing
            if existing:
                existing_segments = [
                    MemorySegment(content=m, category="unknown", confidence=0.0)
                    for m in existing
                ]
                memories = self._dedup_against_existing(memories, existing_segments)

            self._sessions_processed += 1
            self._memories_extracted += len(memories)

            if memories:
                logger.info(
                    "ConsolidationLoop: extracted %d memories from session",
                    len(memories),
                )

        except Exception:
            logger.exception("Error consolidating session messages")

    # -- main consolidation --------------------------------------------------

    async def consolidate(self, messages: list[str]) -> list[MemorySegment]:
        """Main consolidation pipeline.

        1. Segment messages by topic.
        2. Extract memories from each segment.
        3. Return the combined list.
        """
        segments = self._segment_by_topic(messages)
        all_memories: list[MemorySegment] = []

        for segment in segments:
            memories = self._extract_memories(segment)
            all_memories.extend(memories)

        return all_memories

    # -- topical segmentation ------------------------------------------------

    def _segment_by_topic(self, messages: list[str]) -> list[list[str]]:
        """Split messages into topical segments.

        Uses a 15% word-overlap boundary: when two consecutive messages
        share fewer than 15% of their words, a new segment begins.
        """
        if not messages:
            return []

        segments: list[list[str]] = [[messages[0]]]

        for i in range(1, len(messages)):
            overlap = _word_overlap(messages[i - 1], messages[i])
            if overlap < 0.15:
                # Topic boundary — start new segment
                segments.append([messages[i]])
            else:
                segments[-1].append(messages[i])

        return segments

    # -- memory extraction ---------------------------------------------------

    def _extract_memories(self, segment: list[str]) -> list[MemorySegment]:
        """Extract memories from a topical segment.

        Scans each message for preference, decision, correction, and fact
        patterns.  Each match becomes a :class:`MemorySegment`.
        """
        memories: list[MemorySegment] = []
        seen_contents: set[str] = set()

        for message in segment:
            for category, pattern in _EXTRACTION_PATTERNS:
                for match in pattern.finditer(message):
                    content = match.group(1).strip()
                    if not content or len(content) < 10:
                        continue

                    # Skip duplicates within the same extraction run
                    normalised = content.lower()
                    if normalised in seen_contents:
                        continue
                    seen_contents.add(normalised)

                    # Confidence heuristic: longer, more specific = higher
                    word_count = len(content.split())
                    confidence = min(1.0, 0.5 + (word_count / 40.0))

                    memories.append(MemorySegment(
                        content=content,
                        category=category,
                        confidence=round(confidence, 2),
                    ))

        return memories

    # -- deduplication -------------------------------------------------------

    def _dedup_against_existing(
        self,
        memories: list[MemorySegment],
        existing: list[MemorySegment],
    ) -> list[MemorySegment]:
        """Remove memories that overlap >85% with existing ones.

        Uses word-level Jaccard similarity to detect near-duplicates.
        """
        if not existing:
            return memories

        unique: list[MemorySegment] = []
        for mem in memories:
            is_dup = False
            for ex in existing:
                if _word_overlap(mem.content, ex.content) > 0.85:
                    is_dup = True
                    self._memories_deduped += 1
                    break
            if not is_dup:
                unique.append(mem)

        return unique

    # -- status --------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current status and statistics."""
        return {
            "running": self._running,
            "subscriptions": len(self._subscription_ids),
            "sessions_processed": self._sessions_processed,
            "memories_extracted": self._memories_extracted,
            "memories_deduped": self._memories_deduped,
        }
