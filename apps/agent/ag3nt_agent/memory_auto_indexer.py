"""Automatic memory indexing from conversations.

Extracts insights (preferences, decisions, facts) from conversation
messages at configurable intervals and appends them to ~/.ag3nt/MEMORY.md.

This is fire-and-forget: failures are logged but never crash the agent.

Usage:
    from ag3nt_agent.memory_auto_indexer import MemoryAutoIndexer

    indexer = MemoryAutoIndexer(flush_interval=5)
    indexed = indexer.maybe_index(messages, session_id="abc123")
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

MEMORY_FILE = Path.home() / ".ag3nt" / "MEMORY.md"

# ---------------------------------------------------------------------------
# Regex patterns for indexable content
# ---------------------------------------------------------------------------

_PREFERENCE_PATTERNS = [
    re.compile(
        r"(?:I |i )(?:prefer|like|want|always use|never use)\b.*", re.IGNORECASE
    ),
    re.compile(r"(?:please |)(?:always|never|don't)\b.*", re.IGNORECASE),
]

_DECISION_PATTERNS = [
    re.compile(
        r"(?:I've |I have |We )(?:decided|chosen|picked|selected)\b.*", re.IGNORECASE
    ),
    re.compile(
        r"(?:Let's |We should |We'll )(?:use|go with|switch to)\b.*", re.IGNORECASE
    ),
]

_FACT_PATTERNS = [
    re.compile(r"(?:Important|Note|Remember|FYI)[:\s].*", re.IGNORECASE),
    re.compile(
        r"(?:The |Our )(?:API|database|server|service|endpoint)\b.*(?:is|are|uses)\b.*",
        re.IGNORECASE,
    ),
]

# Minimum / maximum length for a valid insight sentence
_MIN_INSIGHT_LEN = 10
_MAX_INSIGHT_LEN = 200

# Sentence-splitting regex (split on period, exclamation, question mark)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def extract_insights(messages: list) -> list[str]:
    """Extract indexable insights from messages.

    Splits each message's content into sentences and checks each against
    preference, decision, and fact patterns.  Returns matching sentences
    whose length falls within [_MIN_INSIGHT_LEN, _MAX_INSIGHT_LEN].

    Args:
        messages: List of LangChain message objects (HumanMessage / AIMessage).

    Returns:
        De-duplicated list of insight strings.
    """
    all_patterns = _PREFERENCE_PATTERNS + _DECISION_PATTERNS + _FACT_PATTERNS
    insights: list[str] = []
    seen: set[str] = set()

    for msg in messages:
        content = getattr(msg, "content", None)
        if not content or not isinstance(content, str):
            continue

        sentences = _split_sentences(content)
        for sentence in sentences:
            if len(sentence) < _MIN_INSIGHT_LEN or len(sentence) > _MAX_INSIGHT_LEN:
                continue
            for pattern in all_patterns:
                if pattern.search(sentence):
                    normalised = sentence.strip()
                    if normalised not in seen:
                        seen.add(normalised)
                        insights.append(normalised)
                    break  # one match per sentence is enough

    return insights


class MemoryAutoIndexer:
    """Periodically extracts insights from conversation and appends to MEMORY.md.

    Args:
        flush_interval: Number of turns between indexing passes (default 5).
    """

    def __init__(self, flush_interval: int = 5) -> None:
        self._flush_interval = flush_interval
        self._turn_count = 0

    # -- public API ----------------------------------------------------------

    def maybe_index(self, messages: list, session_id: str) -> bool:
        """Increment turn counter and, at the configured interval, index.

        Args:
            messages: Full (or recent) conversation messages list.
            session_id: Current session identifier (written to the log entry).

        Returns:
            ``True`` if indexing was performed on this call, ``False`` otherwise.
        """
        self._turn_count += 1
        if self._turn_count % self._flush_interval != 0:
            return False

        try:
            # Only look at the most recent messages (2x interval window)
            recent = messages[-(self._flush_interval * 2) :]
            insights = extract_insights(recent)
            if insights:
                self._append_to_memory(insights, session_id)
                logger.info(
                    "Auto-indexed %d insights for session %s",
                    len(insights),
                    session_id,
                )
            else:
                logger.debug("No insights found at turn %d", self._turn_count)
        except Exception:
            # Fire-and-forget: never crash the agent
            logger.exception("Memory auto-indexing failed (turn %d)", self._turn_count)

        return True

    # -- internals -----------------------------------------------------------

    def _append_to_memory(self, insights: list[str], session_id: str) -> None:
        """Append extracted insights to MEMORY.md.

        Creates the file and parent directories if they do not exist.
        """
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"\n## Auto-indexed ({timestamp}, session: {session_id})\n"]
        for insight in insights:
            lines.append(f"- {insight}\n")

        with open(MEMORY_FILE, "a", encoding="utf-8") as fh:
            fh.writelines(lines)

        logger.debug("Appended %d insights to %s", len(insights), MEMORY_FILE)
