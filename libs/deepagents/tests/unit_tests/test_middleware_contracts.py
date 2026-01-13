"""Tests for middleware contracts and validation."""

import pytest

from deepagents.middleware.contracts import (
    MIDDLEWARE_CONTRACTS,
    MiddlewareContract,
    MiddlewarePhase,
    PromptBudget,
    validate_middleware_stack,
)


class MockMiddleware:
    """Mock middleware for testing."""


class MemoryMiddleware(MockMiddleware):
    """Mock MemoryMiddleware."""


class SkillsMiddleware(MockMiddleware):
    """Mock SkillsMiddleware."""


class FilesystemMiddleware(MockMiddleware):
    """Mock FilesystemMiddleware."""


class SubAgentMiddleware(MockMiddleware):
    """Mock SubAgentMiddleware."""


class CompactionMiddleware(MockMiddleware):
    """Mock CompactionMiddleware."""


class SummarizationMiddleware(MockMiddleware):
    """Mock SummarizationMiddleware."""


class HumanInTheLoopMiddleware(MockMiddleware):
    """Mock HumanInTheLoopMiddleware."""


class UnknownMiddleware(MockMiddleware):
    """Unknown middleware not in registry."""


class TestMiddlewarePhase:
    """Tests for MiddlewarePhase enum."""

    def test_phase_ordering(self):
        """Phases should be ordered correctly."""
        assert MiddlewarePhase.CONTEXT_LOADING < MiddlewarePhase.TOOL_REGISTRATION
        assert MiddlewarePhase.TOOL_REGISTRATION < MiddlewarePhase.ORCHESTRATION
        assert MiddlewarePhase.ORCHESTRATION < MiddlewarePhase.CONTEXT_MANAGEMENT
        assert MiddlewarePhase.CONTEXT_MANAGEMENT < MiddlewarePhase.MODEL_OPTIMIZATION
        assert MiddlewarePhase.MODEL_OPTIMIZATION < MiddlewarePhase.APPROVAL_GATING


class TestMiddlewareContract:
    """Tests for MiddlewareContract dataclass."""

    def test_contract_creation(self):
        """Should create contract with defaults."""
        contract = MiddlewareContract(
            name="TestMiddleware",
            phase=MiddlewarePhase.TOOL_REGISTRATION,
        )
        assert contract.name == "TestMiddleware"
        assert contract.phase == MiddlewarePhase.TOOL_REGISTRATION
        assert contract.injects_prompt is False
        assert contract.registers_tools is False
        assert contract.tool_names == []

    def test_contract_with_budget(self):
        """Should create contract with prompt budget."""
        budget = PromptBudget(max_tokens=1000, priority=80)
        contract = MiddlewareContract(
            name="TestMiddleware",
            phase=MiddlewarePhase.CONTEXT_LOADING,
            injects_prompt=True,
            prompt_budget=budget,
        )
        assert contract.prompt_budget is not None
        assert contract.prompt_budget.max_tokens == 1000
        assert contract.prompt_budget.priority == 80


class TestMiddlewareRegistry:
    """Tests for the middleware contracts registry."""

    def test_known_middleware_registered(self):
        """All known middleware should be in registry."""
        expected = [
            "MemoryMiddleware",
            "SkillsMiddleware",
            "FilesystemMiddleware",
            "SubAgentMiddleware",
            "CompactionMiddleware",
            "HumanInTheLoopMiddleware",
        ]
        for name in expected:
            assert name in MIDDLEWARE_CONTRACTS, f"{name} not in registry"

    def test_no_duplicate_tools(self):
        """No two middleware should register the same tool unless they conflict.

        Conflicting middleware (e.g., CompactionMiddleware and ContextEngineeringMiddleware)
        are designed to never be used together, so they can share tool names.
        """
        all_tools: dict[str, str] = {}
        for name, contract in MIDDLEWARE_CONTRACTS.items():
            for tool in contract.tool_names:
                if tool in all_tools:
                    # Check if these middleware are marked as conflicting
                    other_name = all_tools[tool]
                    other_contract = MIDDLEWARE_CONTRACTS[other_name]
                    conflicts = contract.conflicts_with or []
                    other_conflicts = other_contract.conflicts_with or []

                    # It's OK if they conflict with each other
                    if name in other_conflicts or other_name in conflicts:
                        continue

                    pytest.fail(f"Tool '{tool}' registered by both '{other_name}' and '{name}'")
                all_tools[tool] = name


class TestValidateMiddlewareStack:
    """Tests for validate_middleware_stack function."""

    def test_valid_stack(self):
        """Valid stack should pass validation."""
        stack = [
            MemoryMiddleware(),
            SkillsMiddleware(),
            FilesystemMiddleware(),
            SubAgentMiddleware(),
            CompactionMiddleware(),
            HumanInTheLoopMiddleware(),
        ]
        result = validate_middleware_stack(stack)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_out_of_order_stack(self):
        """Out of order stack should fail validation."""
        stack = [
            HumanInTheLoopMiddleware(),  # Should be last
            MemoryMiddleware(),
        ]
        result = validate_middleware_stack(stack)
        assert result.valid is False
        assert any("out of order" in e for e in result.errors)

    def test_missing_dependency(self):
        """Missing dependency should fail validation."""
        stack = [
            SubAgentMiddleware(),  # Requires FilesystemMiddleware
        ]
        result = validate_middleware_stack(stack)
        assert result.valid is False
        assert any("requires" in e for e in result.errors)

    def test_unknown_middleware_warning(self):
        """Unknown middleware should generate warning."""
        stack = [UnknownMiddleware()]
        result = validate_middleware_stack(stack)
        assert result.valid is True  # Warnings don't fail
        assert any("Unknown" in w for w in result.warnings)

    def test_budget_calculation(self):
        """Should calculate total prompt budget."""
        stack = [
            MemoryMiddleware(),
            SkillsMiddleware(),
        ]
        result = validate_middleware_stack(stack)
        # MemoryMiddleware: 2000, SkillsMiddleware: 1000
        assert result.total_prompt_budget == 3000

    def test_budget_warning(self):
        """Should warn when over budget."""
        stack = [
            MemoryMiddleware(),
            SkillsMiddleware(),
            FilesystemMiddleware(),
            CompactionMiddleware(),
        ]
        result = validate_middleware_stack(stack, max_prompt_budget=1000)
        assert any("exceeds" in w for w in result.warnings)
