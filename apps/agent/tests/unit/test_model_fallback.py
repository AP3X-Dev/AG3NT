# apps/agent/tests/unit/test_model_fallback.py
import os
import time
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _clean_env():
    """Remove model-related env vars to avoid interference."""
    keys = [
        "AG3NT_MODEL_PROVIDER", "AG3NT_MODEL_NAME",
        "AG3NT_MODEL_FALLBACKS",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "OPENROUTER_API_KEY", "GOOGLE_API_KEY",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.mark.unit
class TestModelFallbackChain:
    def test_create_with_single_provider(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain
        chain = ModelFallbackChain(
            providers=[{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}]
        )
        assert len(chain.providers) == 1
        assert chain.providers[0]["provider"] == "anthropic"

    def test_create_with_multiple_providers(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain
        chain = ModelFallbackChain(providers=[
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "openrouter", "model": "moonshotai/kimi-k2.5"},
        ])
        assert len(chain.providers) == 3

    def test_get_model_returns_primary(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain
        mock_model = MagicMock()
        chain = ModelFallbackChain(
            providers=[{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}]
        )
        with patch("ag3nt_agent.model_fallback._create_model_for_provider", return_value=mock_model):
            model = chain.get_model()
        assert model is mock_model

    def test_mark_failure_adds_cooldown(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain
        chain = ModelFallbackChain(providers=[
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
            {"provider": "openai", "model": "gpt-4o"},
        ])
        chain.mark_failure(0, "rate_limit")
        assert chain.is_in_cooldown(0)
        assert not chain.is_in_cooldown(1)

    def test_get_next_available_skips_cooldown(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain
        chain = ModelFallbackChain(providers=[
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
            {"provider": "openai", "model": "gpt-4o"},
        ])
        chain.mark_failure(0, "rate_limit")
        idx = chain.get_next_available(start=0)
        assert idx == 1

    def test_all_in_cooldown_returns_none(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain
        chain = ModelFallbackChain(providers=[
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        ])
        chain.mark_failure(0, "rate_limit")
        idx = chain.get_next_available(start=0)
        assert idx is None

    def test_cooldown_expires(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain
        chain = ModelFallbackChain(
            providers=[{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}],
            cooldown_seconds=0.1,
        )
        # mark_failure uses max(cooldown_seconds, backoff_schedule[0]) = max(0.1, 30) = 30s
        # So we mock time.time to simulate the passage of time
        base_time = time.time()
        with patch("ag3nt_agent.model_fallback.time") as mock_time:
            mock_time.time.return_value = base_time
            chain.mark_failure(0, "rate_limit")
            assert chain.is_in_cooldown(0)
            # Advance past the 30s cooldown
            mock_time.time.return_value = base_time + 31
            assert not chain.is_in_cooldown(0)

    def test_mark_success_resets_errors(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain
        chain = ModelFallbackChain(providers=[
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        ])
        chain.mark_failure(0, "rate_limit")
        chain.mark_failure(0, "rate_limit")
        chain.mark_success(0)
        assert chain.get_consecutive_errors(0) == 0
        assert not chain.is_in_cooldown(0)

    def test_classify_error(self):
        from ag3nt_agent.model_fallback import classify_error
        assert classify_error(Exception("rate limit exceeded")) == "rate_limit"
        assert classify_error(Exception("authentication failed")) == "auth"
        assert classify_error(Exception("context window exceeded")) == "context_overflow"
        assert classify_error(Exception("something weird")) == "unknown"

    def test_from_env_creates_chain(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-test",
            "OPENAI_API_KEY": "sk-test2",
        }):
            chain = ModelFallbackChain.from_env()
            assert len(chain.providers) >= 2

    def test_run_with_fallback_success_on_retry(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain, run_with_fallback
        chain = ModelFallbackChain(
            providers=[
                {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
                {"provider": "openai", "model": "gpt-4o"},
            ],
            cooldown_seconds=0.1,
        )
        call_count = 0
        def action(model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("rate limit exceeded")
            return "success"
        mock_m1, mock_m2 = MagicMock(), MagicMock()
        with patch("ag3nt_agent.model_fallback._create_model_for_provider", side_effect=[mock_m1, mock_m2]):
            result = run_with_fallback(chain, action)
        assert result == "success"
        assert call_count == 2

    def test_run_with_fallback_all_fail(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain, run_with_fallback
        chain = ModelFallbackChain(
            providers=[{"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}],
            cooldown_seconds=0.1,
        )
        def action(model):
            raise Exception("total failure")
        with patch("ag3nt_agent.model_fallback._create_model_for_provider", return_value=MagicMock()):
            with pytest.raises(Exception, match="total failure"):
                run_with_fallback(chain, action)
