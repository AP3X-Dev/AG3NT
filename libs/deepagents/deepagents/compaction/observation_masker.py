"""ObservationMasker for replacing large tool outputs with placeholders.

This module handles the automatic masking of large tool outputs:
1. Tool outputs exceeding a size threshold are persisted as artifacts
2. The conversational record stores a compact placeholder
3. Recent outputs are kept unmasked for short-term grounding
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deepagents.compaction.models import EvidenceRecord, MaskedObservationPlaceholder

if TYPE_CHECKING:
    from deepagents.compaction.artifact_store import ArtifactStore
    from deepagents.compaction.config import CompactionConfig

logger = logging.getLogger(__name__)


@dataclass
class UnmaskedObservation:
    """A recent tool output that hasn't been masked yet."""

    tool_call_id: str
    tool_name: str
    content: str
    timestamp: float


@dataclass
class ObservationMasker:
    """Handles masking of large tool outputs.

    Tool outputs exceeding the configured threshold are persisted as artifacts
    and replaced with compact placeholders. A configurable number of recent
    outputs are kept unmasked for short-term grounding.

    Args:
        store: ArtifactStore for persisting large outputs.
        config: Compaction configuration.
    """

    store: ArtifactStore
    config: CompactionConfig
    _recent_unmasked: deque[UnmaskedObservation] = field(default_factory=deque)
    _masked_placeholders: list[MaskedObservationPlaceholder] = field(default_factory=list)
    _evidence_ledger: list[EvidenceRecord] = field(default_factory=list)
    _masked_count: int = 0

    def __post_init__(self) -> None:
        """Initialize with proper max length for recent observations."""
        self._recent_unmasked = deque(maxlen=self.config.keep_last_unmasked_tool_outputs)

    def should_mask(self, content: str) -> bool:
        """Check if content should be masked based on size threshold.

        Args:
            content: The tool output content.

        Returns:
            True if the content exceeds the masking threshold.
        """
        return len(content) > self.config.mask_tool_output_if_chars_gt

    def _extract_highlights(self, content: str, max_highlights: int = 5) -> list[str]:
        """Extract key highlights from content for the placeholder.

        Uses simple heuristics to find important sentences or lines.
        """
        highlights = []

        # Split into lines and filter
        lines = [line.strip() for line in content.split("\n") if line.strip()]

        # Look for lines that seem important (start with keywords, contain key info)
        important_patterns = [
            r"^(summary|conclusion|result|finding|key|important|note):",
            r"^(error|warning|success):",
            r"^\d+\.\s+",  # Numbered lists
            r"^[-*]\s+",  # Bullet points
        ]

        for line in lines[:50]:  # Only check first 50 lines
            if len(highlights) >= max_highlights:
                break
            for pattern in important_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Truncate long lines
                    highlight = line[:200] + "..." if len(line) > 200 else line
                    highlights.append(highlight)
                    break

        # If we didn't find enough, take first few non-empty lines
        if len(highlights) < 2:
            for line in lines[:5]:
                if line and line not in highlights:
                    highlight = line[:200] + "..." if len(line) > 200 else line
                    highlights.append(highlight)
                    if len(highlights) >= max_highlights:
                        break

        return highlights

    def _generate_digest(self, content: str, tool_name: str) -> str:
        """Generate a short digest summarizing the content."""
        # Simple digest: first 100 chars + size info
        preview = content[:100].replace("\n", " ").strip()
        if len(content) > 100:
            preview += "..."
        return f"{tool_name} output ({len(content):,} chars): {preview}"

    def _detect_url(self, content: str, tool_name: str) -> str | None:
        """Try to detect a source URL from the content or tool name."""
        # Common patterns for URLs in tool outputs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        matches = re.findall(url_pattern, content[:2000])
        return matches[0] if matches else None

    def mask_observation(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
        *,
        source_url: str | None = None,
        title: str | None = None,
    ) -> MaskedObservationPlaceholder | str:
        """Process a tool output and mask if necessary.

        Args:
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            content: The tool output content.
            source_url: Optional URL if this is from a web fetch.
            title: Optional title for the content.

        Returns:
            MaskedObservationPlaceholder if masked, or original content if not.
        """
        import time

        if not self.should_mask(content):
            # Keep in recent unmasked queue
            self._recent_unmasked.append(
                UnmaskedObservation(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    content=content,
                    timestamp=time.time(),
                )
            )
            return content

        # Content exceeds threshold - persist and mask
        detected_url = source_url or self._detect_url(content, tool_name)

        # Determine content type
        content_type = "text/plain"
        if "<html" in content.lower()[:500] or "<!doctype html" in content.lower()[:500]:
            content_type = "text/html"
        elif content.strip().startswith("{") or content.strip().startswith("["):
            content_type = "application/json"

        # Store the artifact
        artifact_id, artifact_path = self.store.write_artifact(
            content,
            tool_name=tool_name,
            source_url=detected_url,
            content_type=content_type,
            title=title,
        )

        # Extract highlights and generate digest
        highlights = self._extract_highlights(content)
        digest = self._generate_digest(content, tool_name)

        # Create placeholder
        placeholder = MaskedObservationPlaceholder(
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            digest=digest,
            artifact_id=artifact_id,
            artifact_path=artifact_path,
            highlights=highlights,
            size_bytes=len(content.encode("utf-8")),
        )

        self._masked_placeholders.append(placeholder)
        self._masked_count += 1

        # If this is from a web source, add to evidence ledger
        if detected_url:
            evidence = EvidenceRecord(
                url=detected_url,
                title=title,
                artifact_id=artifact_id,
                notes=f"Fetched by {tool_name}",
            )
            self._evidence_ledger.append(evidence)

        logger.info(f"Masked {tool_name} output: {len(content):,} chars -> artifact {artifact_id}")
        return placeholder

    def get_placeholder_text(self, placeholder: MaskedObservationPlaceholder) -> str:
        """Get the placeholder text for a masked observation."""
        return placeholder.to_placeholder_text()

    def get_recent_unmasked(self) -> list[UnmaskedObservation]:
        """Get the list of recent unmasked observations."""
        return list(self._recent_unmasked)

    def get_masked_placeholders(self) -> list[MaskedObservationPlaceholder]:
        """Get all masked observation placeholders."""
        return self._masked_placeholders.copy()

    def get_evidence_ledger(self) -> list[EvidenceRecord]:
        """Get the evidence ledger with all tracked sources."""
        return self._evidence_ledger.copy()

    def add_evidence(self, evidence: EvidenceRecord) -> None:
        """Add an evidence record to the ledger."""
        self._evidence_ledger.append(evidence)

    def get_masked_count(self) -> int:
        """Get the total count of masked observations."""
        return self._masked_count

    def clear_old_placeholders(self, keep_last: int = 20) -> int:
        """Clear old placeholders, keeping the most recent ones.

        Args:
            keep_last: Number of recent placeholders to keep.

        Returns:
            Number of placeholders removed.
        """
        if len(self._masked_placeholders) <= keep_last:
            return 0
        removed = len(self._masked_placeholders) - keep_last
        self._masked_placeholders = self._masked_placeholders[-keep_last:]
        return removed
