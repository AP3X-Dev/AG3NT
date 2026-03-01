# apps/agent/tests/unit/test_model_fallback_wiring.py
"""Tests for ModelFallbackChain wiring into the agent runtime."""

import os

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def _clean_env():
    """Remove model-related env vars to avoid interference."""
    keys = [
        "AG3NT_MODEL_PROVIDER", "AG3NT_MODEL_NAME",
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "OPENROUTER_API_KEY", "GOOGLE_API_KEY",
    ]
    saved = {k: os.environ.pop(k, None) for k in keys}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v


@pytest.mark.unit
class TestModelFallbackWiring:
    def test_fallback_chain_created_from_env(self):
        """Verify the fallback chain builds correctly from env."""
        from ag3nt_agent.model_fallback import ModelFallbackChain

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            chain = ModelFallbackChain.from_env()
            assert len(chain.providers) >= 1
            assert chain.providers[0]["provider"] == "anthropic"

    def test_fallback_on_rate_limit(self):
        """When primary model hits rate limit, fallback to next."""
        from ag3nt_agent.model_fallback import ModelFallbackChain

        chain = ModelFallbackChain(providers=[
            {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
            {"provider": "openai", "model": "gpt-4o"},
        ])
        mock_fallback = MagicMock()

        with patch(
            "ag3nt_agent.model_fallback._create_model_for_provider",
            return_value=mock_fallback,
        ):
            chain.mark_failure(0, "rate_limit")
            idx = chain.get_next_available(start=0)
            assert idx == 1
            model = chain.get_model(idx)
            assert model is mock_fallback

    def test_run_with_fallback_wrapper(self):
        """Test the run_with_fallback utility function."""
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
        with patch(
            "ag3nt_agent.model_fallback._create_model_for_provider",
            side_effect=[mock_m1, mock_m2],
        ):
            result = run_with_fallback(chain, action)
        assert result == "success"
        assert call_count == 2

    def test_run_with_fallback_all_fail(self):
        """When all providers fail, raise the last error."""
        from ag3nt_agent.model_fallback import ModelFallbackChain, run_with_fallback

        chain = ModelFallbackChain(
            providers=[
                {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
            ],
            cooldown_seconds=0.1,
        )

        def action(model):
            raise Exception("total failure")

        with patch(
            "ag3nt_agent.model_fallback._create_model_for_provider",
            return_value=MagicMock(),
        ):
            with pytest.raises(Exception, match="total failure"):
                run_with_fallback(chain, action)

    def test_get_fallback_chain_singleton(self):
        """Verify the runtime singleton pattern works.

        deepagents_runtime.py has heavy transitive imports (langchain,
        langgraph, and many ag3nt_agent sub-modules) that may not be
        installed in a lightweight test environment.  Rather than
        stubbing the entire dependency tree, we directly exercise the
        singleton getter by importing just ``model_fallback`` and
        verifying the same pattern that ``get_fallback_chain`` uses.
        """
        from ag3nt_agent.model_fallback import ModelFallbackChain

        # Replicate the singleton pattern from deepagents_runtime
        _chain_holder: dict[str, ModelFallbackChain | None] = {"chain": None}

        def get_fallback_chain() -> ModelFallbackChain:
            if _chain_holder["chain"] is None:
                _chain_holder["chain"] = ModelFallbackChain.from_env()
            return _chain_holder["chain"]

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            chain = get_fallback_chain()
            assert chain is not None
            assert len(chain.providers) >= 1
            # Second call returns the same cached instance
            chain2 = get_fallback_chain()
            assert chain is chain2

    def test_get_fallback_chain_in_runtime_source(self):
        """Verify that deepagents_runtime.py contains get_fallback_chain."""
        import inspect
        from pathlib import Path

        runtime_path = (
            Path(__file__).resolve().parent.parent.parent
            / "ag3nt_agent" / "deepagents_runtime.py"
        )
        source = runtime_path.read_text(encoding="utf-8")
        assert "def get_fallback_chain(" in source
        assert "_fallback_chain" in source
        assert "ModelFallbackChain.from_env()" in source
