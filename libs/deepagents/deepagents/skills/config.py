"""Configuration for the Skills Toolkit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class SkillsConfig:
    """Configuration for the skills toolkit.

    Attributes:
        skills_dirs: List of directories to scan for skills.
            Later directories have higher priority (override earlier).
        max_skills_list: Maximum number of skills to return in list_skills.
        max_skill_body_chars: Maximum characters to load from skill body.
        enable_builder: Whether to enable the build_skill tool.
        builder_output_dir: Directory where builder creates new skills.
        enforce_tool_allowlist: Whether to enforce tool restrictions from skills.
        allow_deprecated_skills: Whether to allow loading deprecated skills.
        dev_mode: If True, allows supervisor override of tool restrictions.
        workspace_dir: Directory for ledger and metrics output.
        budget_default_tokens: Default token budget when loading skills.
    """

    # Skill discovery
    skills_dirs: list[str] = field(default_factory=lambda: ["./skills/"])
    max_skills_list: int = 50
    max_skill_body_chars: int = 12000

    # Builder configuration
    enable_builder: bool = True
    builder_output_dir: str = "./skills/"

    # Security and enforcement
    enforce_tool_allowlist: bool = True
    allow_deprecated_skills: bool = False
    dev_mode: bool = False

    # Workspace for output
    workspace_dir: Path | None = None

    # Token budgets
    budget_default_tokens: int = 3000
    budget_examples_tokens: int = 1000

    # Metrics
    enable_metrics: bool = True
    metrics_file: str = "skills_metrics.jsonl"

    def get_workspace_dir(self) -> Path:
        """Get the workspace directory, creating if needed."""
        if self.workspace_dir is None:
            import tempfile
            self.workspace_dir = Path(tempfile.mkdtemp(prefix="deepagents_skills_"))
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        return self.workspace_dir

    def get_ledger_path(self) -> Path:
        """Get path to the skill usage ledger."""
        return self.get_workspace_dir() / "skill_usage_ledger.jsonl"

    def get_metrics_path(self) -> Path:
        """Get path to the metrics file."""
        return self.get_workspace_dir() / self.metrics_file

    def get_registry_index_path(self) -> Path:
        """Get path to the registry index artifact."""
        return self.get_workspace_dir() / "skills_registry_index.json"

    def resolve_skills_dirs(self) -> list[Path]:
        """Resolve all skill directories to absolute paths."""
        resolved = []
        for dir_str in self.skills_dirs:
            path = Path(dir_str).expanduser()
            if not path.is_absolute():
                path = Path.cwd() / path
            resolved.append(path.resolve())
        return resolved

