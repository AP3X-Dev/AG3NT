"""ArtifactStore for persisting and retrieving artifacts.

Artifacts are stored objects produced by tools or external sources,
such as HTML, PDF, extracted text, screenshots, JSON, logs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.compaction.models import ArtifactMeta

if TYPE_CHECKING:
    from deepagents.compaction.config import CompactionConfig

logger = logging.getLogger(__name__)

# Patterns for secret redaction
SECRET_PATTERNS = [
    (re.compile(r'(api[_-]?key|apikey)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE), r'\1="[REDACTED]"'),
    (re.compile(r'(password|passwd|pwd)["\s:=]+["\']?([^\s"\']{8,})["\']?', re.IGNORECASE), r'\1="[REDACTED]"'),
    (re.compile(r'(secret|token)["\s:=]+["\']?([a-zA-Z0-9_\-]{20,})["\']?', re.IGNORECASE), r'\1="[REDACTED]"'),
    (re.compile(r'(bearer|authorization)["\s:]+["\']?([a-zA-Z0-9_\-\.]{20,})["\']?', re.IGNORECASE), r'\1="[REDACTED]"'),
    (re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE), "[REDACTED_API_KEY]"),
    (re.compile(r"(ghp_[a-zA-Z0-9]{36})", re.IGNORECASE), "[REDACTED_GITHUB_TOKEN]"),
]


class ArtifactStore:
    """Store for persisting and retrieving artifacts.

    Artifacts are stored in a filesystem-based backend with a JSONL metadata ledger.
    Each artifact gets a unique ID, content hash, and associated metadata.

    Args:
        config: Compaction configuration with workspace settings.
    """

    def __init__(self, config: CompactionConfig) -> None:
        self.config = config
        self._artifacts_dir = config.get_artifacts_dir()
        self._metadata_path = config.get_metadata_path()
        self._metadata_cache: dict[str, ArtifactMeta] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load existing metadata from the ledger file."""
        if self._metadata_path.exists():
            with open(self._metadata_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            meta = ArtifactMeta.model_validate(data)
                            self._metadata_cache[meta.artifact_id] = meta
                        except Exception as e:
                            logger.warning(f"Failed to parse metadata line: {e}")

    def _append_metadata(self, meta: ArtifactMeta) -> None:
        """Append metadata to the ledger file."""
        with open(self._metadata_path, "a", encoding="utf-8") as f:
            f.write(meta.model_dump_json() + "\n")
        self._metadata_cache[meta.artifact_id] = meta

    def _compute_hash(self, content: str | bytes) -> str:
        """Compute SHA256 hash of content."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _redact_secrets(self, content: str) -> str:
        """Redact obvious secrets from content."""
        if not self.config.redact_secrets:
            return content
        for pattern, replacement in SECRET_PATTERNS:
            content = pattern.sub(replacement, content)
        return content

    def _generate_artifact_id(self) -> str:
        """Generate a unique artifact ID."""
        return f"art_{uuid.uuid4().hex[:12]}"

    def write_artifact(
        self,
        content: str | bytes,
        *,
        tool_name: str,
        source_url: str | None = None,
        content_type: str = "text/plain",
        title: str | None = None,
        tags: list[str] | None = None,
        publish_date: datetime | None = None,
    ) -> tuple[str, str]:
        """Write an artifact to storage.

        Args:
            content: The content to store (text or bytes).
            tool_name: Name of the tool that produced this artifact.
            source_url: URL if fetched from web.
            content_type: MIME type of the content.
            title: Title or summary of the artifact.
            tags: Tags for categorization.
            publish_date: Publication date if known.

        Returns:
            Tuple of (artifact_id, stored_path).
        """
        artifact_id = self._generate_artifact_id()

        # Handle text content
        if isinstance(content, str):
            content = self._redact_secrets(content)
            content_bytes = content.encode("utf-8")
        else:
            content_bytes = content

        content_hash = self._compute_hash(content_bytes)

        # Determine file extension based on content type
        ext_map = {
            "text/html": ".html",
            "text/plain": ".txt",
            "application/json": ".json",
            "application/pdf": ".pdf",
            "image/png": ".png",
            "image/jpeg": ".jpg",
        }
        ext = ext_map.get(content_type, ".bin")

        # Write to file
        filename = f"{artifact_id}{ext}"
        stored_path = self._artifacts_dir / filename
        with open(stored_path, "wb") as f:
            f.write(content_bytes)

        # Create and store metadata
        meta = ArtifactMeta(
            artifact_id=artifact_id,
            tool_name=tool_name,
            source_url=source_url,
            content_type=content_type,
            content_hash=content_hash,
            stored_raw_path=str(stored_path),
            size_bytes=len(content_bytes),
            title=title,
            tags=tags or [],
            publish_date=publish_date,
        )
        self._append_metadata(meta)

        logger.debug(f"Stored artifact {artifact_id}: {len(content_bytes)} bytes")
        return artifact_id, str(stored_path)

    def read_artifact(self, artifact_id: str) -> str | bytes | None:
        """Read an artifact by ID.

        Args:
            artifact_id: The artifact ID to read.

        Returns:
            The artifact content, or None if not found.
        """
        meta = self._metadata_cache.get(artifact_id)
        if meta is None:
            logger.warning(f"Artifact not found: {artifact_id}")
            return None

        path = Path(meta.stored_raw_path)
        if not path.exists():
            logger.warning(f"Artifact file not found: {path}")
            return None

        # Determine if text or binary based on content type
        if meta.content_type.startswith("text/") or meta.content_type == "application/json":
            with open(path, encoding="utf-8") as f:
                return f.read()
        else:
            with open(path, "rb") as f:
                return f.read()

    def read_artifact_by_path(self, path: str) -> str | bytes | None:
        """Read an artifact by file path.

        Args:
            path: The file path of the artifact.

        Returns:
            The artifact content, or None if not found.
        """
        # Find artifact by path
        for meta in self._metadata_cache.values():
            if meta.stored_raw_path == path:
                return self.read_artifact(meta.artifact_id)

        # Try direct file read if path exists
        file_path = Path(path)
        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                with open(file_path, "rb") as f:
                    return f.read()
        return None

    def get_metadata(self, artifact_id: str) -> ArtifactMeta | None:
        """Get metadata for an artifact.

        Args:
            artifact_id: The artifact ID.

        Returns:
            The artifact metadata, or None if not found.
        """
        return self._metadata_cache.get(artifact_id)

    def list_artifacts(
        self,
        *,
        tool_name: str | None = None,
        tags: list[str] | None = None,
        source_url_contains: str | None = None,
        limit: int = 100,
    ) -> list[ArtifactMeta]:
        """List artifacts matching filters.

        Args:
            tool_name: Filter by tool name.
            tags: Filter by tags (any match).
            source_url_contains: Filter by URL substring.
            limit: Maximum number of results.

        Returns:
            List of matching artifact metadata.
        """
        results = []
        for meta in self._metadata_cache.values():
            if tool_name and meta.tool_name != tool_name:
                continue
            if tags and not any(t in meta.tags for t in tags):
                continue
            if source_url_contains and (not meta.source_url or source_url_contains not in meta.source_url):
                continue
            results.append(meta)
            if len(results) >= limit:
                break
        return results

    def get_artifact_count(self) -> int:
        """Get the total number of stored artifacts."""
        return len(self._metadata_cache)

    def get_total_bytes(self) -> int:
        """Get the total bytes stored across all artifacts."""
        return sum(m.size_bytes for m in self._metadata_cache.values())
