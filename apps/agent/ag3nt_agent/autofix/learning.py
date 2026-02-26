"""Fix history persistence for the AutoFix pipeline.

Stores fix outcomes at ``~/.ag3nt/autofix_history.json`` and provides
success-rate queries used by :class:`ConfidenceScorer` and skip-decision
logic.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from ag3nt_agent.autofix.context import FixContext

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".ag3nt" / "autofix_history.json"
_MAX_RECORDS = 500
_SKIP_THRESHOLD = 0.80   # >=80 % failure rate
_SKIP_MIN_ATTEMPTS = 5   # require at least this many before skipping


def _error_hash_from_ctx(ctx: FixContext) -> str:
    """Compute a stable hash for the errors in a FixContext."""
    msgs = sorted(
        str(e.get("message", e.get("error", ""))) for e in ctx.errors
    )
    raw = f"{ctx.category}:{'|'.join(msgs)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class FixHistory:
    """Persistent record of AutoFix outcomes.

    Parameters
    ----------
    path:
        Location of the JSON backing store.  Defaults to
        ``~/.ag3nt/autofix_history.json``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_PATH
        self._records: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, ctx: FixContext) -> None:
        """Save the outcome of a pipeline run."""
        entry: dict[str, Any] = {
            "id": ctx.id,
            "category": ctx.category,
            "severity": ctx.severity,
            "outcome": ctx.outcome,
            "confidence": ctx.confidence,
            "duration_ms": ctx.fix_duration_ms,
            "error_hash": _error_hash_from_ctx(ctx),
            "timestamp": time.time(),
        }
        self._records.append(entry)

        # Trim oldest entries.
        if len(self._records) > _MAX_RECORDS:
            self._records = self._records[-_MAX_RECORDS:]

        self._save()
        logger.debug(
            "FixHistory: recorded %s outcome=%s category=%s",
            ctx.id,
            ctx.outcome,
            ctx.category,
        )

    def get_success_rate(self, category: str) -> float:
        """Return the success rate (0.0-1.0) for a given error *category*.

        Returns ``-1.0`` when there are no records for the category (so the
        caller can distinguish "no data" from "0 % success").
        """
        matching = [r for r in self._records if r.get("category") == category]
        if not matching:
            return -1.0
        successes = sum(1 for r in matching if r.get("outcome") == "success")
        return successes / len(matching)

    def should_skip(self, error_hash: str) -> bool:
        """Return *True* if the error should be skipped.

        An error is skipped when it has been attempted at least
        ``_SKIP_MIN_ATTEMPTS`` times and the failure rate is at or above
        ``_SKIP_THRESHOLD``.
        """
        matching = [r for r in self._records if r.get("error_hash") == error_hash]
        if len(matching) < _SKIP_MIN_ATTEMPTS:
            return False
        failures = sum(
            1 for r in matching if r.get("outcome") not in ("success",)
        )
        rate = failures / len(matching)
        return rate >= _SKIP_THRESHOLD

    def get_records(self, category: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent records, optionally filtered by *category*."""
        recs = self._records
        if category:
            recs = [r for r in recs if r.get("category") == category]
        return list(recs[-limit:])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load history from disk."""
        if self._path.is_file():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._records = data if isinstance(data, list) else []
                logger.debug(
                    "FixHistory: loaded %d records from %s",
                    len(self._records),
                    self._path,
                )
            except Exception as exc:
                logger.warning("FixHistory: failed to load %s: %s", self._path, exc)
                self._records = []
        else:
            self._records = []

    def _save(self) -> None:
        """Persist history to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._records, default=str, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("FixHistory: failed to save %s: %s", self._path, exc)
