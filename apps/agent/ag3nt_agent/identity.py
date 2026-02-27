"""Identity file loader for AG3NT — layered identity system.

Reads IDENTITY.md, SOUL.md, USER.md, AGENTS.md from a base directory
and builds a layered system prompt following ONI's priority model.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# File mapping: key -> filename
_IDENTITY_FILES = {
    "identity": "IDENTITY.md",
    "soul": "SOUL.md",
    "user_context": "USER.md",
    "agents": "AGENTS.md",
}

# Header pattern to strip: lines like "# IDENTITY.md — AG3NT" or "# SOUL.md"
_HEADER_RE = re.compile(
    r"^#\s+(?:IDENTITY|SOUL|USER|AGENTS)\.md(?:\s*[—\-–].*)?$",
    re.MULTILINE,
)

_SOUL_GUIDANCE = (
    "The following describes your personality and communication style. "
    "Embody these traits naturally in all interactions."
)


class IdentityLoader:
    """Load and compose identity files into a system prompt."""

    def __init__(self, base_dir: str | Path = "~/.ag3nt") -> None:
        self._base = Path(base_dir).expanduser()

    def load(self) -> dict[str, str]:
        """Read each identity file that exists."""
        result: dict[str, str] = {}
        for key, filename in _IDENTITY_FILES.items():
            path = self._base / filename
            if path.is_file():
                try:
                    result[key] = path.read_text(encoding="utf-8")
                except OSError:
                    logger.debug("Failed to read %s", path)
        return result

    def build_system_prompt(self, *, minimal: bool = False) -> str:
        """Build a layered system prompt from identity files."""
        if minimal:
            return "You are AG3NT."

        data = self.load()
        if not data:
            return ""

        parts: list[str] = []

        # Priority 1: IDENTITY.md (complete override)
        if "identity" in data:
            parts.append(self._strip_header(data["identity"]))
        # Priority 2: SOUL.md alone (if no IDENTITY.md)
        elif "soul" in data:
            parts.append(_SOUL_GUIDANCE)
            parts.append(self._strip_header(data["soul"]))

        # Layer SOUL.md on top of IDENTITY.md when both exist
        if "identity" in data and "soul" in data:
            parts.append(_SOUL_GUIDANCE)
            parts.append(self._strip_header(data["soul"]))

        # User context section
        if "user_context" in data:
            parts.append(f"## User Context\n{self._strip_header(data['user_context'])}")

        # Behavior guidelines section
        if "agents" in data:
            parts.append(f"## Behavior Guidelines\n{self._strip_header(data['agents'])}")

        return "\n\n".join(parts)

    @staticmethod
    def _strip_header(text: str) -> str:
        """Remove file metadata headers like '# IDENTITY.md — AG3NT'."""
        stripped = _HEADER_RE.sub("", text).strip()
        return stripped
