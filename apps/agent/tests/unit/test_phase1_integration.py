# apps/agent/tests/unit/test_phase1_integration.py
import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _clean_env():
    keys = ["AG3NT_MODEL_PROVIDER", "AG3NT_MODEL_NAME",
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.mark.unit
class TestPhase1AgentIntegration:
    def test_fallback_chain_created_and_primary_used(self):
        """Fallback chain creates correctly and serves primary model."""
        from ag3nt_agent.model_fallback import ModelFallbackChain

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-test-ant",
            "OPENAI_API_KEY": "sk-test-oai",
        }):
            chain = ModelFallbackChain.from_env()
            assert len(chain.providers) >= 2
            assert chain.providers[0]["provider"] == "anthropic"

    def test_fallback_recovers_from_rate_limit(self):
        """When primary hits rate limit, fallback chain switches to next."""
        from ag3nt_agent.model_fallback import ModelFallbackChain, run_with_fallback

        chain = ModelFallbackChain(
            providers=[
                {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
                {"provider": "openai", "model": "gpt-4o"},
            ],
            cooldown_seconds=0.1,
        )
        call_log = []

        def action(model):
            call_log.append(model)
            if len(call_log) == 1:
                raise Exception("429 rate limit exceeded")
            return "recovered"

        mock1, mock2 = MagicMock(), MagicMock()
        with patch(
            "ag3nt_agent.model_fallback._create_model_for_provider",
            side_effect=[mock1, mock2],
        ):
            result = run_with_fallback(chain, action)

        assert result == "recovered"
        assert len(call_log) == 2

    def test_error_classification_comprehensive(self):
        """Error classifier handles all expected error types."""
        from ag3nt_agent.model_fallback import classify_error

        cases = [
            ("rate limit exceeded", "rate_limit"),
            ("429 Too Many Requests", "rate_limit"),
            ("authentication failed", "auth"),
            ("401 Unauthorized", "auth"),
            ("invalid api key", "auth"),
            ("context window exceeded", "context_overflow"),
            ("billing quota reached", "billing"),
            ("request timed out", "timeout"),
            ("random error XYZ", "unknown"),
        ]
        for msg, expected in cases:
            assert classify_error(Exception(msg)) == expected, f"Failed for: {msg}"
