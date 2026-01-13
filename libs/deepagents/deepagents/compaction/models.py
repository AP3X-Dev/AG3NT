"""Data models for the Context Window Compaction System.

All models use Pydantic for validation and serialization.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Get current UTC time in a timezone-aware manner."""
    return datetime.now(timezone.utc)


class Confidence(str, Enum):
    """Confidence level for findings."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ArtifactMeta(BaseModel):
    """Metadata for a stored artifact.

    Artifacts are stored objects produced by tools or external sources,
    such as HTML, PDF, extracted text, screenshots, JSON, logs.
    """

    artifact_id: str = Field(..., description="Unique identifier for the artifact")
    created_at: datetime = Field(default_factory=_utcnow)
    tool_name: str = Field(..., description="Name of the tool that produced this artifact")
    source_url: str | None = Field(None, description="URL if fetched from web")
    content_type: str = Field("text/plain", description="MIME type of the content")
    content_hash: str = Field(..., description="SHA256 hash of the content")
    stored_raw_path: str = Field(..., description="Path to raw artifact file")
    stored_clean_path: str | None = Field(None, description="Path to cleaned/processed version")
    size_bytes: int = Field(..., description="Size of the artifact in bytes")
    publish_date: datetime | None = Field(None, description="Publication date if known")
    title: str | None = Field(None, description="Title or summary of the artifact")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


class EvidenceRecord(BaseModel):
    """Record of an evidence source.

    Tracks external sources (pages, PDFs) that have been fetched and stored.
    """

    url: str = Field(..., description="URL of the source")
    title: str | None = Field(None, description="Title of the source")
    publish_date: datetime | None = Field(None, description="Publication date if known")
    fetched_at: datetime = Field(default_factory=_utcnow)
    artifact_id: str = Field(..., description="ID of the stored artifact")
    notes: str = Field("", description="Short notes about the source")
    quotes: list[str] = Field(default_factory=list, description="Key quotes extracted")


class Finding(BaseModel):
    """A claim or finding with confidence and evidence pointers."""

    claim: str = Field(..., description="The factual claim or finding")
    confidence: Confidence = Field(Confidence.MEDIUM, description="Confidence level")
    evidence_artifact_ids: list[str] = Field(
        default_factory=list,
        description="IDs of artifacts supporting this claim"
    )
    notes: str | None = Field(None, description="Additional notes or caveats")


class MaskedObservationPlaceholder(BaseModel):
    """Compact placeholder for a masked tool output.

    Replaces large tool outputs in conversation memory while preserving
    access to the full content via artifact pointer.
    """

    tool_name: str = Field(..., description="Name of the tool that produced the output")
    tool_call_id: str = Field(..., description="ID of the original tool call")
    digest: str = Field(..., description="Short summary of the content")
    artifact_id: str = Field(..., description="ID of the stored artifact")
    artifact_path: str = Field(..., description="Path to the stored artifact")
    highlights: list[str] = Field(
        default_factory=list,
        description="Key extracted highlights from the content"
    )
    size_bytes: int = Field(..., description="Original size of the content")
    created_at: datetime = Field(default_factory=_utcnow)

    def to_placeholder_text(self) -> str:
        """Generate the placeholder text for insertion into conversation."""
        highlights_str = ""
        if self.highlights:
            highlights_str = "\nKey points:\n" + "\n".join(f"- {h}" for h in self.highlights[:5])

        return (
            f"[MASKED OUTPUT: {self.tool_name}]\n"
            f"Digest: {self.digest}\n"
            f"Artifact: {self.artifact_id}{highlights_str}\n"
            f"Use retrieve_snippets(artifact_id='{self.artifact_id}', query='...') for details."
        )


class ReasoningState(BaseModel):
    """Structured summary of the agent's reasoning progress.

    Created periodically to compress intermediate reasoning while
    preserving key facts and evidence pointers.
    """

    executive_summary: str = Field(..., description="High-level summary of progress")
    confirmed_facts: list[str] = Field(
        default_factory=list,
        description="Facts confirmed with evidence"
    )
    hypotheses: list[str] = Field(
        default_factory=list,
        description="Working hypotheses not yet confirmed"
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions still to be answered"
    )
    visited_sources: list[str] = Field(
        default_factory=list,
        description="URLs or artifact IDs of sources visited"
    )
    next_actions: list[str] = Field(
        default_factory=list,
        description="Planned next actions"
    )
    created_at: datetime = Field(default_factory=_utcnow)
    step_number: int = Field(0, description="Agent step when this was created")


class ResearchBundle(BaseModel):
    """Compact research results returned by a research subagent.

    Contains distilled findings with evidence pointers for verification.
    """

    executive_summary: str = Field(..., description="High-level summary of findings")
    findings: list[Finding] = Field(default_factory=list, description="Key findings")
    evidence: list[EvidenceRecord] = Field(
        default_factory=list,
        description="Evidence sources consulted"
    )
    extracted_data_json: dict[str, Any] | None = Field(
        None,
        description="Structured data extracted during research"
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Questions that couldn't be answered"
    )
    created_at: datetime = Field(default_factory=_utcnow)

