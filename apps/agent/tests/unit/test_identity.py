"""Tests for IdentityLoader."""
from pathlib import Path

import pytest

from ag3nt_agent.identity import IdentityLoader


@pytest.fixture()
def identity_dir(tmp_path):
    """Create a temporary identity directory."""
    return tmp_path


class TestIdentityLoaderLoad:
    def test_load_returns_empty_when_no_files(self, identity_dir):
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.load()
        assert result == {}

    def test_load_reads_identity_file(self, identity_dir):
        (identity_dir / "IDENTITY.md").write_text("I am AG3NT.", encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.load()
        assert "identity" in result
        assert "I am AG3NT." in result["identity"]

    def test_load_reads_all_four_files(self, identity_dir):
        (identity_dir / "IDENTITY.md").write_text("identity content", encoding="utf-8")
        (identity_dir / "SOUL.md").write_text("soul content", encoding="utf-8")
        (identity_dir / "USER.md").write_text("user content", encoding="utf-8")
        (identity_dir / "AGENTS.md").write_text("agents content", encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.load()
        assert len(result) == 4
        assert result["identity"] == "identity content"
        assert result["soul"] == "soul content"
        assert result["user_context"] == "user content"
        assert result["agents"] == "agents content"

    def test_load_skips_missing_files(self, identity_dir):
        (identity_dir / "SOUL.md").write_text("soul only", encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.load()
        assert "soul" in result
        assert "identity" not in result


class TestIdentityLoaderBuildPrompt:
    def test_minimal_mode_returns_agent_name(self, identity_dir):
        (identity_dir / "IDENTITY.md").write_text("full identity", encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.build_system_prompt(minimal=True)
        assert "AG3NT" in result
        assert "full identity" not in result

    def test_identity_takes_priority(self, identity_dir):
        (identity_dir / "IDENTITY.md").write_text("I am the agent.", encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.build_system_prompt()
        assert "I am the agent." in result

    def test_soul_layers_on_identity(self, identity_dir):
        (identity_dir / "IDENTITY.md").write_text("Base identity.", encoding="utf-8")
        (identity_dir / "SOUL.md").write_text("Personality traits.", encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.build_system_prompt()
        assert "Base identity." in result
        assert "Personality traits." in result

    def test_user_context_section(self, identity_dir):
        (identity_dir / "IDENTITY.md").write_text("Identity.", encoding="utf-8")
        (identity_dir / "USER.md").write_text("User prefs.", encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.build_system_prompt()
        assert "## User Context" in result
        assert "User prefs." in result

    def test_agents_behavior_section(self, identity_dir):
        (identity_dir / "IDENTITY.md").write_text("Identity.", encoding="utf-8")
        (identity_dir / "AGENTS.md").write_text("Behavior rules.", encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.build_system_prompt()
        assert "## Behavior Guidelines" in result
        assert "Behavior rules." in result

    def test_no_files_returns_empty(self, identity_dir):
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.build_system_prompt()
        assert result == ""

    def test_strips_file_header(self, identity_dir):
        content = "# IDENTITY.md — AG3NT\n\nActual content here."
        (identity_dir / "IDENTITY.md").write_text(content, encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.build_system_prompt()
        assert "# IDENTITY.md" not in result
        assert "Actual content here." in result

    def test_soul_alone_without_identity(self, identity_dir):
        (identity_dir / "SOUL.md").write_text("Soul content.", encoding="utf-8")
        loader = IdentityLoader(base_dir=identity_dir)
        result = loader.build_system_prompt()
        assert "Soul content." in result
