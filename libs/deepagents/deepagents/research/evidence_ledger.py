"""Evidence Ledger for tracking research sources and citations.

The EvidenceLedger provides persistent storage and retrieval of evidence
records, serving as the source of truth for all citations in research bundles.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents.compaction.models import EvidenceRecord

if TYPE_CHECKING:
    from deepagents.research.config import ResearchConfig

logger = logging.getLogger(__name__)


class EvidenceLedger:
    """Persistent storage for evidence records.

    The EvidenceLedger stores all evidence gathered during research,
    providing the source of truth for citations. Records are persisted
    to a JSONL file and loaded on initialization.

    Args:
        session_dir: Directory for the research session.
        config: Research configuration (optional).
    """

    def __init__(
        self,
        session_dir: Path,
        config: ResearchConfig | None = None,
    ) -> None:
        self.session_dir = session_dir
        self.config = config
        self._ledger_path = session_dir / "evidence_ledger.jsonl"
        self._records: dict[str, EvidenceRecord] = {}
        self._load()

    def _load(self) -> None:
        """Load existing records from the ledger file."""
        if self._ledger_path.exists():
            with open(self._ledger_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            record = EvidenceRecord.model_validate(data)
                            self._records[record.artifact_id] = record
                        except Exception as e:
                            logger.warning(f"Failed to parse evidence record: {e}")

    def _append(self, record: EvidenceRecord) -> None:
        """Append a record to the ledger file."""
        with open(self._ledger_path, "a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
        self._records[record.artifact_id] = record

    def add_record(
        self,
        *,
        url: str,
        artifact_id: str,
        title: str | None = None,
        publish_date: datetime | None = None,
        notes: str = "",
        quotes: list[str] | None = None,
    ) -> EvidenceRecord:
        """Add a new evidence record.

        Args:
            url: URL of the source.
            artifact_id: ID of the stored artifact.
            title: Title of the source.
            publish_date: Publication date if known.
            notes: Short notes about the source.
            quotes: Key quotes extracted from the source.

        Returns:
            The created EvidenceRecord.
        """
        record = EvidenceRecord(
            url=url,
            artifact_id=artifact_id,
            title=title,
            publish_date=publish_date,
            notes=notes,
            quotes=quotes or [],
        )
        self._append(record)
        logger.debug(f"Added evidence record: {url} -> {artifact_id}")
        return record

    def get_by_artifact_id(self, artifact_id: str) -> EvidenceRecord | None:
        """Get a record by artifact ID."""
        return self._records.get(artifact_id)

    def get_by_url(self, url: str) -> EvidenceRecord | None:
        """Get a record by URL."""
        for record in self._records.values():
            if record.url == url:
                return record
        return None

    def list_records(
        self,
        *,
        domain_contains: str | None = None,
        has_publish_date: bool | None = None,
        limit: int = 100,
    ) -> list[EvidenceRecord]:
        """List evidence records with optional filters.

        Args:
            domain_contains: Filter by domain substring.
            has_publish_date: Filter by presence of publish date.
            limit: Maximum number of records to return.

        Returns:
            List of matching EvidenceRecord objects.
        """
        results = []
        for record in self._records.values():
            if domain_contains and domain_contains not in record.url:
                continue
            if has_publish_date is True and record.publish_date is None:
                continue
            if has_publish_date is False and record.publish_date is not None:
                continue
            results.append(record)
            if len(results) >= limit:
                break
        return results

    def get_all(self) -> list[EvidenceRecord]:
        """Get all evidence records."""
        return list(self._records.values())

    def count(self) -> int:
        """Get the number of evidence records."""
        return len(self._records)

    def get_unique_domains(self) -> set[str]:
        """Get the set of unique domains in the ledger."""
        from urllib.parse import urlparse

        domains = set()
        for record in self._records.values():
            try:
                parsed = urlparse(record.url)
                if parsed.netloc:
                    domains.add(parsed.netloc.lower())
            except Exception:
                pass
        return domains
