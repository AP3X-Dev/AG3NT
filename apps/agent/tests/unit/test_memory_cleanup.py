"""Unit tests for memory_cleanup module."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ag3nt_agent.memory_cleanup import DEFAULT_TTL_DAYS, MemoryCleanup


# ------------------------------------------------------------------
# __init__ defaults
# ------------------------------------------------------------------


class TestInit:
    def test_defaults(self, tmp_path: Path):
        mc = MemoryCleanup(memory_dir=tmp_path, blueprints_dir=tmp_path)
        assert mc.ttl_days == DEFAULT_TTL_DAYS
        assert mc._memory_dir == tmp_path
        assert mc._blueprints_dir == tmp_path

    def test_custom_ttl_days(self, tmp_path: Path):
        mc = MemoryCleanup(memory_dir=tmp_path, blueprints_dir=tmp_path, ttl_days=30)
        assert mc.ttl_days == 30

    def test_env_var_ttl(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AG3NT_MEMORY_TTL_DAYS", "45")
        mc = MemoryCleanup(memory_dir=tmp_path, blueprints_dir=tmp_path)
        assert mc.ttl_days == 45

    def test_explicit_ttl_overrides_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AG3NT_MEMORY_TTL_DAYS", "45")
        mc = MemoryCleanup(memory_dir=tmp_path, blueprints_dir=tmp_path, ttl_days=10)
        assert mc.ttl_days == 10


# ------------------------------------------------------------------
# cleanup_memory
# ------------------------------------------------------------------


class TestCleanupMemory:
    def test_nonexistent_dir_returns_zero(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist"
        mc = MemoryCleanup(memory_dir=missing, blueprints_dir=tmp_path)
        assert mc.cleanup_memory() == 0

    def test_removes_old_dated_files(self, tmp_path: Path):
        """A file named 2025-01-01.md should be removed with TTL 90 days
        (we are well past 90 days from 2025-01-01)."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        old_file = mem_dir / "2025-01-01.md"
        old_file.write_text("old memory")

        mc = MemoryCleanup(memory_dir=mem_dir, blueprints_dir=tmp_path, ttl_days=90)
        removed = mc.cleanup_memory()

        assert removed == 1
        assert not old_file.exists()

    def test_removes_old_dated_file_with_topic_suffix(self, tmp_path: Path):
        """A file named 2025-01-01-topic.md should also be removed."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        old_file = mem_dir / "2025-01-01-project-notes.md"
        old_file.write_text("old memory")

        mc = MemoryCleanup(memory_dir=mem_dir, blueprints_dir=tmp_path, ttl_days=90)
        removed = mc.cleanup_memory()

        assert removed == 1
        assert not old_file.exists()

    def test_keeps_recent_dated_files(self, tmp_path: Path):
        """A file dated today should be kept."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        recent_file = mem_dir / f"{today}.md"
        recent_file.write_text("fresh memory")

        mc = MemoryCleanup(memory_dir=mem_dir, blueprints_dir=tmp_path, ttl_days=90)
        removed = mc.cleanup_memory()

        assert removed == 0
        assert recent_file.exists()

    def test_mtime_fallback_removes_old_nondated_file(self, tmp_path: Path):
        """A non-dated file with old mtime should be removed."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        nondated = mem_dir / "random-notes.md"
        nondated.write_text("some notes")
        # Set mtime to 200 days ago
        old_ts = time.time() - (200 * 86400)
        os.utime(nondated, (old_ts, old_ts))

        mc = MemoryCleanup(memory_dir=mem_dir, blueprints_dir=tmp_path, ttl_days=90)
        removed = mc.cleanup_memory()

        assert removed == 1
        assert not nondated.exists()

    def test_mtime_fallback_keeps_recent_nondated_file(self, tmp_path: Path):
        """A non-dated file with recent mtime should be kept."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        nondated = mem_dir / "random-notes.md"
        nondated.write_text("some notes")
        # mtime is just-now by default, well within 90-day TTL

        mc = MemoryCleanup(memory_dir=mem_dir, blueprints_dir=tmp_path, ttl_days=90)
        removed = mc.cleanup_memory()

        assert removed == 0
        assert nondated.exists()

    def test_mixed_files(self, tmp_path: Path):
        """Mix of old dated, recent dated, old non-dated, recent non-dated."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()

        # Old dated -- should be removed
        (mem_dir / "2024-06-15.md").write_text("old")
        # Recent dated -- should be kept
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        (mem_dir / f"{today}-notes.md").write_text("recent")
        # Old non-dated -- should be removed
        old_nd = mem_dir / "misc.md"
        old_nd.write_text("old misc")
        os.utime(old_nd, (time.time() - 200 * 86400,) * 2)
        # Recent non-dated -- should be kept
        (mem_dir / "ideas.md").write_text("fresh ideas")

        mc = MemoryCleanup(memory_dir=mem_dir, blueprints_dir=tmp_path, ttl_days=90)
        removed = mc.cleanup_memory()

        assert removed == 2
        assert not (mem_dir / "2024-06-15.md").exists()
        assert not old_nd.exists()
        assert (mem_dir / f"{today}-notes.md").exists()
        assert (mem_dir / "ideas.md").exists()

    def test_ignores_non_md_files(self, tmp_path: Path):
        """Only .md files should be considered."""
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        txt_file = mem_dir / "2024-01-01.txt"
        txt_file.write_text("not markdown")

        mc = MemoryCleanup(memory_dir=mem_dir, blueprints_dir=tmp_path, ttl_days=90)
        removed = mc.cleanup_memory()

        assert removed == 0
        assert txt_file.exists()


# ------------------------------------------------------------------
# cleanup_blueprints
# ------------------------------------------------------------------


class TestCleanupBlueprints:
    def test_nonexistent_dir_returns_zero(self, tmp_path: Path):
        missing = tmp_path / "no_blueprints"
        mc = MemoryCleanup(memory_dir=tmp_path, blueprints_dir=missing)
        assert mc.cleanup_blueprints() == 0

    def test_removes_old_blueprints(self, tmp_path: Path):
        bp_dir = tmp_path / "blueprints"
        bp_dir.mkdir()
        old_bp = bp_dir / "plan-abc.json"
        old_bp.write_text("{}")
        old_ts = time.time() - (200 * 86400)
        os.utime(old_bp, (old_ts, old_ts))

        mc = MemoryCleanup(memory_dir=tmp_path, blueprints_dir=bp_dir, ttl_days=90)
        removed = mc.cleanup_blueprints()

        assert removed == 1
        assert not old_bp.exists()

    def test_keeps_recent_blueprints(self, tmp_path: Path):
        bp_dir = tmp_path / "blueprints"
        bp_dir.mkdir()
        recent_bp = bp_dir / "plan-xyz.json"
        recent_bp.write_text("{}")
        # mtime is now -- well within TTL

        mc = MemoryCleanup(memory_dir=tmp_path, blueprints_dir=bp_dir, ttl_days=90)
        removed = mc.cleanup_blueprints()

        assert removed == 0
        assert recent_bp.exists()

    def test_ignores_non_json_files(self, tmp_path: Path):
        bp_dir = tmp_path / "blueprints"
        bp_dir.mkdir()
        txt_file = bp_dir / "notes.txt"
        txt_file.write_text("not json")
        old_ts = time.time() - (200 * 86400)
        os.utime(txt_file, (old_ts, old_ts))

        mc = MemoryCleanup(memory_dir=tmp_path, blueprints_dir=bp_dir, ttl_days=90)
        removed = mc.cleanup_blueprints()

        assert removed == 0
        assert txt_file.exists()

    def test_multiple_old_blueprints(self, tmp_path: Path):
        bp_dir = tmp_path / "blueprints"
        bp_dir.mkdir()
        old_ts = time.time() - (200 * 86400)
        for name in ["a.json", "b.json", "c.json"]:
            f = bp_dir / name
            f.write_text("{}")
            os.utime(f, (old_ts, old_ts))
        # One recent
        (bp_dir / "d.json").write_text("{}")

        mc = MemoryCleanup(memory_dir=tmp_path, blueprints_dir=bp_dir, ttl_days=90)
        removed = mc.cleanup_blueprints()

        assert removed == 3
        assert (bp_dir / "d.json").exists()


# ------------------------------------------------------------------
# cleanup_all
# ------------------------------------------------------------------


class TestCleanupAll:
    def test_returns_combined_results(self, tmp_path: Path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        bp_dir = tmp_path / "blueprints"
        bp_dir.mkdir()

        # Old memory file
        (mem_dir / "2024-01-01.md").write_text("old")
        # Old blueprint
        old_bp = bp_dir / "old.json"
        old_bp.write_text("{}")
        old_ts = time.time() - (200 * 86400)
        os.utime(old_bp, (old_ts, old_ts))
        # Recent files that should be kept
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        (mem_dir / f"{today}.md").write_text("fresh")
        (bp_dir / "new.json").write_text("{}")

        mc = MemoryCleanup(memory_dir=mem_dir, blueprints_dir=bp_dir, ttl_days=90)
        result = mc.cleanup_all()

        assert result == {"memory": 1, "blueprints": 1}

    def test_empty_dirs(self, tmp_path: Path):
        mem_dir = tmp_path / "memory"
        mem_dir.mkdir()
        bp_dir = tmp_path / "blueprints"
        bp_dir.mkdir()

        mc = MemoryCleanup(memory_dir=mem_dir, blueprints_dir=bp_dir, ttl_days=90)
        result = mc.cleanup_all()

        assert result == {"memory": 0, "blueprints": 0}

    def test_nonexistent_dirs(self, tmp_path: Path):
        mc = MemoryCleanup(
            memory_dir=tmp_path / "no_mem",
            blueprints_dir=tmp_path / "no_bp",
            ttl_days=90,
        )
        result = mc.cleanup_all()
        assert result == {"memory": 0, "blueprints": 0}


# ------------------------------------------------------------------
# _parse_date_from_filename
# ------------------------------------------------------------------


class TestParseDateFromFilename:
    def test_simple_date(self):
        result = MemoryCleanup._parse_date_from_filename("2026-03-01")
        assert result == datetime(2026, 3, 1, tzinfo=timezone.utc)

    def test_date_with_suffix(self):
        result = MemoryCleanup._parse_date_from_filename("2026-03-01-project-notes")
        assert result == datetime(2026, 3, 1, tzinfo=timezone.utc)

    def test_no_date(self):
        assert MemoryCleanup._parse_date_from_filename("random-notes") is None

    def test_partial_date(self):
        assert MemoryCleanup._parse_date_from_filename("2026-03") is None

    def test_invalid_date_values(self):
        # Month 13 is invalid
        assert MemoryCleanup._parse_date_from_filename("2026-13-01") is None

    def test_empty_string(self):
        assert MemoryCleanup._parse_date_from_filename("") is None

    def test_non_date_digits(self):
        assert MemoryCleanup._parse_date_from_filename("12345678") is None

    def test_date_feb_29_leap_year(self):
        result = MemoryCleanup._parse_date_from_filename("2024-02-29")
        assert result == datetime(2024, 2, 29, tzinfo=timezone.utc)

    def test_date_feb_29_non_leap_year(self):
        # 2025 is not a leap year
        assert MemoryCleanup._parse_date_from_filename("2025-02-29") is None
