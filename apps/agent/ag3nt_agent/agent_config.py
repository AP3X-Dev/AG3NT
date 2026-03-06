"""Centralized agent configuration for AG3NT.

Reads from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path


# Shell execution
SHELL_TIMEOUT: float = float(os.environ.get("AG3NT_SHELL_TIMEOUT", "60.0"))
MAX_OUTPUT_BYTES: int = int(os.environ.get("AG3NT_MAX_OUTPUT_BYTES", "100000"))

# Gateway connection
GATEWAY_URL: str = os.environ.get("AG3NT_GATEWAY_URL", "http://127.0.0.1:18789")

# Workspace
WORKSPACE_PATH: Path = Path.home() / ".ag3nt" / "workspace"
USER_DATA_PATH: Path = Path.home() / ".ag3nt"

# Process management
PROCESS_MAX_AGE: float = float(os.environ.get("AG3NT_PROCESS_MAX_AGE", "3600"))

# File watcher
FILE_WATCHER_DEBOUNCE: float = float(
    os.environ.get("AG3NT_FILE_WATCHER_DEBOUNCE", "0.1")
)

# Smart output truncation
TRUNCATION_MAX_LINES: int = int(os.environ.get("AG3NT_TRUNCATION_MAX_LINES", "2000"))
TRUNCATION_MAX_BYTES: int = int(os.environ.get("AG3NT_TRUNCATION_MAX_BYTES", str(50 * 1024)))
TRUNCATION_DIR: Path = Path.home() / ".ag3nt" / "tool_output"

# Gateway authentication
GATEWAY_TOKEN: str = os.environ.get("AG3NT_GATEWAY_TOKEN", "")

# Agent guard
MAX_AGENT_STEPS: int = int(os.environ.get("AG3NT_MAX_STEPS", "75"))

# YOLO mode — full autonomous operation, no approval gates
YOLO_MODE: bool = os.environ.get("AG3NT_YOLO_MODE", "false").lower() == "true"

# Cost guardrails
MAX_SESSION_COST_USD: float = float(os.environ.get("AG3NT_MAX_SESSION_COST", "5.00"))
