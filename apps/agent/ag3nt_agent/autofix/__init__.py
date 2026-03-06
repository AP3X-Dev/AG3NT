"""AutoFix — self-healing error pipeline for AG3NT.

7-stage pipeline: Detect -> Classify -> Enrich -> Score -> Fix -> Verify -> Learn
"""

from ag3nt_agent.autofix.engine import AutofixEngine, AutofixConfig  # noqa: F401
from ag3nt_agent.autofix.circuit_breaker import CircuitBreaker  # noqa: F401
from ag3nt_agent.autofix.context import FixContext  # noqa: F401
from ag3nt_agent.autofix.pipeline import Pipeline, PipelineStage  # noqa: F401
