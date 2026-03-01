# apps/agent/tests/unit/test_thinking_fallback.py
"""Tests for thinking-level escalation in model fallback chain."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
class TestThinkingLevel:
    """Tests for the ThinkingLevel enum."""

    def test_thinking_level_enum_values(self):
        from ag3nt_agent.model_fallback import ThinkingLevel
        assert ThinkingLevel.OFF == "off"
        assert ThinkingLevel.LOW == "low"
        assert ThinkingLevel.MEDIUM == "medium"
        assert ThinkingLevel.HIGH == "high"

    def test_provider_state_has_thinking_level(self):
        from ag3nt_agent.model_fallback import ProviderState, ThinkingLevel
        state = ProviderState()
        assert state.thinking_level == ThinkingLevel.OFF

    def test_thinking_level_next(self):
        from ag3nt_agent.model_fallback import ThinkingLevel
        assert ThinkingLevel.OFF.next() == ThinkingLevel.LOW
        assert ThinkingLevel.LOW.next() == ThinkingLevel.MEDIUM
        assert ThinkingLevel.MEDIUM.next() == ThinkingLevel.HIGH
        assert ThinkingLevel.HIGH.next() is None


@pytest.mark.unit
class TestEscalateThinking:
    """Tests for the escalate_thinking method on ModelFallbackChain."""

    def test_escalate_from_off_to_low(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain, ThinkingLevel
        chain = ModelFallbackChain(
            providers=[{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}]
        )
        result = chain.escalate_thinking(0)
        assert result is True
        assert chain._states[0].thinking_level == ThinkingLevel.LOW

    def test_escalate_through_all_levels(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain, ThinkingLevel
        chain = ModelFallbackChain(
            providers=[{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}]
        )
        # OFF -> LOW
        assert chain.escalate_thinking(0) is True
        # LOW -> MEDIUM
        assert chain.escalate_thinking(0) is True
        # MEDIUM -> HIGH
        assert chain.escalate_thinking(0) is True
        # HIGH -> None (can't escalate further)
        assert chain.escalate_thinking(0) is False
        assert chain._states[0].thinking_level == ThinkingLevel.HIGH

    def test_escalate_resets_on_success(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain, ThinkingLevel
        chain = ModelFallbackChain(
            providers=[{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}]
        )
        chain.escalate_thinking(0)
        assert chain._states[0].thinking_level == ThinkingLevel.LOW
        chain.mark_success(0)
        assert chain._states[0].thinking_level == ThinkingLevel.OFF


@pytest.mark.unit
class TestRunWithFallbackThinking:
    """Tests for thinking-level escalation within run_with_fallback."""

    def test_context_overflow_triggers_escalation(self):
        from ag3nt_agent.model_fallback import (
            ModelFallbackChain, run_with_fallback, ThinkingLevel,
        )
        chain = ModelFallbackChain(
            providers=[{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}],
            cooldown_seconds=0.1,
        )
        call_count = 0

        def action(model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("context window overflow")
            return "success"

        mock_model = MagicMock()
        with patch(
            "ag3nt_agent.model_fallback._create_model_for_provider",
            return_value=mock_model,
        ):
            result = run_with_fallback(chain, action, max_attempts=4)
        assert result == "success"
        assert call_count == 2

    def test_timeout_triggers_escalation(self):
        from ag3nt_agent.model_fallback import (
            ModelFallbackChain, run_with_fallback, ThinkingLevel,
        )
        chain = ModelFallbackChain(
            providers=[{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}],
            cooldown_seconds=0.1,
        )
        call_count = 0

        def action(model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("request timed out")
            return "success"

        mock_model = MagicMock()
        with patch(
            "ag3nt_agent.model_fallback._create_model_for_provider",
            return_value=mock_model,
        ):
            result = run_with_fallback(chain, action, max_attempts=4)
        assert result == "success"
        assert call_count == 2
