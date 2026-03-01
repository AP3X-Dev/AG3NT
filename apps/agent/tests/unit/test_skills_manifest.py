"""Tests for skills manifest injection in system prompt."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock the langchain / langgraph / deepagents module tree.
# Uses the same pattern as test_prompt_modes.py.
# ---------------------------------------------------------------------------
_mock_modules: dict[str, ModuleType] = {}


def _ensure_mock_module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    if name in _mock_modules:
        return _mock_modules[name]
    mod = ModuleType(name)
    _mock_modules[name] = mod
    sys.modules[name] = mod
    return mod


_lc = _ensure_mock_module("langchain")
_lc_agents = _ensure_mock_module("langchain.agents")
_lc_agents_mw = _ensure_mock_module("langchain.agents.middleware")
_lc_agents_mw_types = _ensure_mock_module("langchain.agents.middleware.types")

if not hasattr(_lc_agents_mw_types, "AgentMiddleware") or not isinstance(
    _lc_agents_mw_types.AgentMiddleware, type
):
    _lc_agents_mw_types.AgentMiddleware = type(
        "AgentMiddleware", (), {"__class_getitem__": classmethod(lambda cls, *a: cls)}
    )
if not hasattr(_lc_agents_mw_types, "AgentState"):
    _lc_agents_mw_types.AgentState = MagicMock()
if not hasattr(_lc_agents_mw_types, "ModelRequest"):
    _lc_agents_mw_types.ModelRequest = MagicMock()
if not hasattr(_lc_agents_mw_types, "ModelResponse"):
    _lc_agents_mw_types.ModelResponse = MagicMock()
if not hasattr(_lc_agents_mw_types, "ModelCallResult"):
    _lc_agents_mw_types.ModelCallResult = MagicMock()

_lc_core = _ensure_mock_module("langchain_core")
_lc_core_msgs = _ensure_mock_module("langchain_core.messages")
if not hasattr(_lc_core_msgs, "SystemMessage"):
    _lc_core_msgs.SystemMessage = MagicMock()

_da = _ensure_mock_module("deepagents")
_da_mw = _ensure_mock_module("deepagents.middleware")
_da_mw_utils = _ensure_mock_module("deepagents.middleware._utils")
if not hasattr(_da_mw_utils, "append_to_system_message"):
    _da_mw_utils.append_to_system_message = MagicMock()

from ag3nt_agent.turn_context_middleware import TurnContextMiddleware, PromptMode


class TestSkillsManifest:
    def test_manifest_includes_skill_names_and_descriptions(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        skills_metadata = {
            "git-commit": {"name": "git-commit", "description": "Commit changes to git"},
            "web-search": {"name": "web-search", "description": "Search the web"},
        }
        manifest = mw._build_skills_manifest(skills_metadata)
        assert "git-commit" in manifest
        assert "Commit changes to git" in manifest
        assert "web-search" in manifest

    def test_manifest_is_compact(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        skills = {
            f"skill-{i}": {"name": f"skill-{i}", "description": f"Description {i}"}
            for i in range(20)
        }
        manifest = mw._build_skills_manifest(skills)
        assert len(manifest) < 2000

    def test_manifest_empty_when_no_skills(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        manifest = mw._build_skills_manifest({})
        assert manifest == ""

    def test_minimal_mode_skips_manifest(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.MINIMAL)
        manifest = mw._build_skills_manifest(
            {"s": {"name": "s", "description": "d"}}
        )
        assert manifest == ""

    def test_none_mode_skips_manifest(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.NONE)
        manifest = mw._build_skills_manifest(
            {"s": {"name": "s", "description": "d"}}
        )
        assert manifest == ""

    def test_manifest_header_format(self):
        mw = TurnContextMiddleware(prompt_mode=PromptMode.FULL)
        manifest = mw._build_skills_manifest(
            {"test": {"name": "test", "description": "A test skill"}}
        )
        assert manifest.startswith("## Available Skills")
        assert "- **test**: A test skill" in manifest
