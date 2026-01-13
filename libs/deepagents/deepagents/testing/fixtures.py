"""Test Fixtures - Recorded agent interactions for replay testing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class FixtureMessage:
    """A recorded message in a fixture."""

    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FixtureMessage:
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            tool_calls=data.get("tool_calls", []),
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass
class FixtureResult:
    """Result of running a fixture."""

    fixture_name: str
    passed: bool
    expected_output: str | None
    actual_output: str | None
    error: str | None = None
    duration_ms: float = 0.0
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fixture_name": self.fixture_name,
            "passed": self.passed,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "token_count": self.token_count,
        }


@dataclass
class AgentFixture:
    """A recorded agent interaction for replay testing.

    Fixtures capture:
    - Input messages
    - Expected tool calls
    - Expected final output
    - Metadata (model, timestamp, etc.)
    """

    name: str
    description: str
    messages: list[FixtureMessage]
    expected_output: str | None = None
    expected_tool_calls: list[str] = field(default_factory=list)
    model: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "messages": [m.to_dict() for m in self.messages],
            "expected_output": self.expected_output,
            "expected_tool_calls": self.expected_tool_calls,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentFixture:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            messages=[FixtureMessage.from_dict(m) for m in data["messages"]],
            expected_output=data.get("expected_output"),
            expected_tool_calls=data.get("expected_tool_calls", []),
            model=data.get("model"),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path | str) -> None:
        """Save fixture to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved fixture to {path}")

    @classmethod
    def load(cls, path: Path | str) -> AgentFixture:
        """Load fixture from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_input_messages(self) -> list[FixtureMessage]:
        """Get only user input messages."""
        return [m for m in self.messages if m.role == "user"]
