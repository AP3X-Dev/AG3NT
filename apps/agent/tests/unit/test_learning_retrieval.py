"""Tests for ag3nt_agent.learning_retrieval."""
from __future__ import annotations

import json

from ag3nt_agent.learning_retrieval import (
    _keyword_search_blueprints,
    _keyword_search_plans,
    format_learnings_for_prompt,
    retrieve_learnings,
)


class TestKeywordSearchBlueprints:
    def test_empty_dir(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig = lr.BLUEPRINTS_DIR
        lr.BLUEPRINTS_DIR = tmp_path / "empty"
        try:
            assert _keyword_search_blueprints("auth") == []
        finally:
            lr.BLUEPRINTS_DIR = orig

    def test_finds_matching_blueprint(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig = lr.BLUEPRINTS_DIR
        lr.BLUEPRINTS_DIR = tmp_path

        bp = {
            "goal": "add user authentication",
            "learnings": ["Use bcrypt for password hashing", "JWT tokens expire after 1h"],
            "status": "completed",
        }
        (tmp_path / "bp1.json").write_text(json.dumps(bp), encoding="utf-8")
        try:
            results = _keyword_search_blueprints("authentication")
            assert len(results) >= 1
            assert "bcrypt" in results[0] or "JWT" in results[0]
        finally:
            lr.BLUEPRINTS_DIR = orig

    def test_no_match(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig = lr.BLUEPRINTS_DIR
        lr.BLUEPRINTS_DIR = tmp_path

        bp = {"goal": "deploy kubernetes cluster", "learnings": ["Use helm"], "status": "done"}
        (tmp_path / "bp1.json").write_text(json.dumps(bp), encoding="utf-8")
        try:
            results = _keyword_search_blueprints("authentication login")
            assert results == []
        finally:
            lr.BLUEPRINTS_DIR = orig

    def test_handles_bad_json(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig = lr.BLUEPRINTS_DIR
        lr.BLUEPRINTS_DIR = tmp_path
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
        try:
            results = _keyword_search_blueprints("anything")
            assert results == []
        finally:
            lr.BLUEPRINTS_DIR = orig


class TestKeywordSearchPlans:
    def test_empty_dir(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig = lr.PLANS_DIR
        lr.PLANS_DIR = tmp_path / "empty"
        try:
            assert _keyword_search_plans("auth") == []
        finally:
            lr.PLANS_DIR = orig

    def test_finds_matching_plan(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig = lr.PLANS_DIR
        lr.PLANS_DIR = tmp_path

        content = "# Add Authentication\n\n**Status:** completed\n\n- [x] Task 1\n- [x] Task 2\n"
        (tmp_path / "2026-03-01-auth.md").write_text(content, encoding="utf-8")
        try:
            results = _keyword_search_plans("authentication")
            assert len(results) == 1
            assert "Authentication" in results[0]
            assert "2/2" in results[0]
        finally:
            lr.PLANS_DIR = orig

    def test_no_match(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig = lr.PLANS_DIR
        lr.PLANS_DIR = tmp_path

        content = "# Deploy Pipeline\n\n- [ ] Task 1\n"
        (tmp_path / "2026-03-01-deploy.md").write_text(content, encoding="utf-8")
        try:
            results = _keyword_search_plans("authentication")
            assert results == []
        finally:
            lr.PLANS_DIR = orig


class TestRetrieveLearnings:
    def test_empty_when_no_data(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig_bp = lr.BLUEPRINTS_DIR
        orig_plans = lr.PLANS_DIR
        lr.BLUEPRINTS_DIR = tmp_path / "bp"
        lr.PLANS_DIR = tmp_path / "plans"
        try:
            results = retrieve_learnings("anything")
            assert results == []
        finally:
            lr.BLUEPRINTS_DIR = orig_bp
            lr.PLANS_DIR = orig_plans

    def test_combines_sources(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig_bp = lr.BLUEPRINTS_DIR
        orig_plans = lr.PLANS_DIR

        bp_dir = tmp_path / "bp"
        bp_dir.mkdir()
        plans_dir = tmp_path / "plans"
        plans_dir.mkdir()

        lr.BLUEPRINTS_DIR = bp_dir
        lr.PLANS_DIR = plans_dir

        bp = {"goal": "build api", "learnings": ["Use FastAPI"], "status": "completed"}
        (bp_dir / "bp1.json").write_text(json.dumps(bp), encoding="utf-8")
        (plans_dir / "2026-01-01-api.md").write_text(
            "# Build API\n\n- [x] Done\n", encoding="utf-8"
        )
        try:
            results = retrieve_learnings("build api")
            assert len(results) >= 1
        finally:
            lr.BLUEPRINTS_DIR = orig_bp
            lr.PLANS_DIR = orig_plans

    def test_deduplicates(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig_bp = lr.BLUEPRINTS_DIR
        orig_plans = lr.PLANS_DIR
        lr.BLUEPRINTS_DIR = tmp_path / "bp"
        lr.PLANS_DIR = tmp_path / "plans"
        lr.BLUEPRINTS_DIR.mkdir()
        lr.PLANS_DIR.mkdir()

        # Two blueprints with identical learning
        for i in range(2):
            bp = {"goal": "test dedup", "learnings": ["Same learning"], "status": "ok"}
            (lr.BLUEPRINTS_DIR / f"bp{i}.json").write_text(
                json.dumps(bp), encoding="utf-8"
            )
        try:
            results = retrieve_learnings("test dedup")
            assert results.count("[ok] test dedup: Same learning") == 1
        finally:
            lr.BLUEPRINTS_DIR = orig_bp
            lr.PLANS_DIR = orig_plans

    def test_respects_limit(self, tmp_path):
        import ag3nt_agent.learning_retrieval as lr
        orig_bp = lr.BLUEPRINTS_DIR
        lr.BLUEPRINTS_DIR = tmp_path
        lr.PLANS_DIR = tmp_path / "empty"

        for i in range(10):
            bp = {"goal": f"task {i}", "learnings": [f"learning {i}"], "status": "ok"}
            (tmp_path / f"bp{i}.json").write_text(json.dumps(bp), encoding="utf-8")
        try:
            results = retrieve_learnings("task", limit=3)
            assert len(results) <= 3
        finally:
            lr.BLUEPRINTS_DIR = orig_bp


class TestFormatLearnings:
    def test_empty_list(self):
        assert format_learnings_for_prompt([]) == ""

    def test_formats_learnings(self):
        result = format_learnings_for_prompt(["Use bcrypt", "JWT expires in 1h"])
        assert "## Past Learnings" in result
        assert "- Use bcrypt" in result
        assert "- JWT expires in 1h" in result

    def test_includes_header(self):
        result = format_learnings_for_prompt(["Something"])
        assert "past experience" in result.lower()
