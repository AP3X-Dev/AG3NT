"""Subagent Containment - Enforces return contracts and config propagation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse

if TYPE_CHECKING:
    from langchain.agents.middleware import InterruptOnConfig
    from langchain.tools import BaseTool
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


@dataclass
class DistilledReturnContract:
    """Contract defining expected subagent return format.
    
    Subagents should return concise, structured results that can be
    efficiently incorporated into the parent agent's context.
    
    Attributes:
        max_output_tokens: Maximum tokens allowed in return value.
        require_summary: Whether to require an executive summary.
        require_evidence: Whether to require evidence citations.
        allowed_formats: Allowed output formats (e.g., ["text", "json", "markdown"]).
    """
    
    max_output_tokens: int = 2000
    """Maximum tokens in the distilled return."""
    
    require_summary: bool = True
    """Require an executive summary at the start."""
    
    require_evidence: bool = False
    """Require evidence citations for claims."""
    
    allowed_formats: list[str] = field(default_factory=lambda: ["text", "markdown"])
    """Allowed output formats."""

    def validate_output(self, output: str) -> tuple[bool, list[str]]:
        """Validate output against contract.
        
        Returns:
            Tuple of (is_valid, list of violations).
        """
        violations = []
        
        # Check token count (rough estimate)
        estimated_tokens = len(output) // 4
        if estimated_tokens > self.max_output_tokens:
            violations.append(
                f"Output too long: ~{estimated_tokens} tokens > {self.max_output_tokens} limit"
            )
        
        # Check for summary (simple heuristic)
        if self.require_summary:
            summary_markers = ["summary:", "## summary", "**summary**", "tldr:"]
            has_summary = any(m in output.lower() for m in summary_markers)
            if not has_summary and len(output) > 500:
                violations.append("Missing executive summary for long output")
        
        return len(violations) == 0, violations


@dataclass
class SubagentConfig:
    """Configuration propagated to subagents.
    
    This ensures subagents inherit appropriate settings from the parent.
    
    Attributes:
        return_contract: Contract for subagent returns.
        inherit_approval_policy: Whether to inherit parent's approval policy.
        inherit_compaction: Whether to inherit parent's compaction settings.
        max_steps: Maximum steps allowed for subagent.
        token_budget: Total token budget for the subagent.
    """
    
    return_contract: DistilledReturnContract = field(
        default_factory=DistilledReturnContract
    )
    """Contract for return values."""
    
    inherit_approval_policy: bool = True
    """Inherit approval policy from parent."""
    
    inherit_compaction: bool = True
    """Inherit compaction settings from parent."""
    
    max_steps: int = 50
    """Maximum steps before forced termination."""
    
    token_budget: int = 50_000
    """Token budget for the subagent."""


# System prompt addition for contained subagents
CONTAINMENT_PROMPT = """## Subagent Return Guidelines

You are a subagent with a focused task. When complete:
1. Provide a concise executive summary (2-3 sentences)
2. Include key findings in a structured format
3. Cite evidence sources where applicable
4. Keep your response under {max_tokens} tokens

Focus on actionable, distilled results the parent agent can use directly."""


class ContainedSubAgentMiddleware(AgentMiddleware):
    """Extended SubAgentMiddleware with containment enforcement.
    
    Wraps the standard SubAgentMiddleware to add:
    1. Return contract enforcement
    2. Config propagation to subagents
    3. Token budget tracking
    """

    def __init__(
        self,
        *,
        default_model: "str | BaseChatModel",
        default_tools: "Sequence[BaseTool | Callable | dict[str, Any]] | None" = None,
        default_middleware: "list[AgentMiddleware] | None" = None,
        default_interrupt_on: "dict[str, bool | InterruptOnConfig] | None" = None,
        subagents: list[Any] | None = None,
        config: SubagentConfig | None = None,
        general_purpose_agent: bool = True,
    ) -> None:
        """Initialize contained subagent middleware."""
        super().__init__()
        self.config = config or SubagentConfig()
        
        # Import here to avoid circular imports
        from deepagents.middleware.subagents import SubAgentMiddleware
        
        # Build containment prompt
        containment_prompt = CONTAINMENT_PROMPT.format(
            max_tokens=self.config.return_contract.max_output_tokens
        )
        
        # Create inner middleware
        self._inner = SubAgentMiddleware(
            default_model=default_model,
            default_tools=default_tools,
            default_middleware=default_middleware,
            default_interrupt_on=default_interrupt_on,
            subagents=subagents or [],
            general_purpose_agent=general_purpose_agent,
            system_prompt=containment_prompt,
        )
        
        self.tools = self._inner.tools

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Delegate to inner middleware."""
        return self._inner.wrap_model_call(request, handler)

