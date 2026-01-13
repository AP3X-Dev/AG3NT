"""Tests for Subagent Containment module."""


from deepagents.subagent import (
    ContainedSubAgentMiddleware,
    DistilledReturnContract,
    SubagentConfig,
)


class TestDistilledReturnContract:
    """Tests for DistilledReturnContract."""

    def test_default_values(self):
        """Test default contract values."""
        contract = DistilledReturnContract()
        assert contract.max_output_tokens == 2000
        assert contract.require_summary is True
        assert contract.require_evidence is False

    def test_validate_short_output(self):
        """Test validation of short output."""
        contract = DistilledReturnContract()
        is_valid, violations = contract.validate_output("Short response")
        assert is_valid
        assert len(violations) == 0

    def test_validate_long_output_without_summary(self):
        """Test validation catches missing summary in long output."""
        contract = DistilledReturnContract(require_summary=True)
        long_output = "x" * 3000  # ~750 tokens, no summary
        is_valid, violations = contract.validate_output(long_output)
        assert not is_valid
        assert any("summary" in v.lower() for v in violations)

    def test_validate_long_output_with_summary(self):
        """Test validation passes with summary."""
        contract = DistilledReturnContract(require_summary=True)
        output = "## Summary\nThis is the summary.\n\n" + "x" * 1000
        is_valid, violations = contract.validate_output(output)
        assert is_valid

    def test_validate_exceeds_token_limit(self):
        """Test validation catches token limit violations."""
        contract = DistilledReturnContract(max_output_tokens=100)
        long_output = "x" * 2000  # ~500 tokens
        is_valid, violations = contract.validate_output(long_output)
        assert not is_valid
        assert any("too long" in v.lower() for v in violations)


class TestSubagentConfig:
    """Tests for SubagentConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = SubagentConfig()
        assert config.inherit_approval_policy is True
        assert config.inherit_compaction is True
        assert config.max_steps == 50
        assert config.token_budget == 50_000

    def test_custom_return_contract(self):
        """Test custom return contract."""
        contract = DistilledReturnContract(max_output_tokens=1000)
        config = SubagentConfig(return_contract=contract)
        assert config.return_contract.max_output_tokens == 1000


class TestContainedSubAgentMiddleware:
    """Tests for ContainedSubAgentMiddleware."""

    def test_initialization(self):
        """Test middleware initializes correctly."""
        middleware = ContainedSubAgentMiddleware(
            default_model="anthropic:claude-sonnet",
        )
        assert middleware.config is not None
        assert middleware.tools is not None

    def test_custom_config(self):
        """Test middleware with custom config."""
        config = SubagentConfig(max_steps=25)
        middleware = ContainedSubAgentMiddleware(
            default_model="anthropic:claude-sonnet",
            config=config,
        )
        assert middleware.config.max_steps == 25

    def test_has_task_tool(self):
        """Test middleware provides task tool."""
        middleware = ContainedSubAgentMiddleware(
            default_model="anthropic:claude-sonnet",
        )
        tool_names = [t.name for t in middleware.tools]
        assert "task" in tool_names
