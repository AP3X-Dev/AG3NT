"""Research Orchestrator for coordinating research tasks.

The ResearchOrchestrator is responsible for:
- Converting a user goal into a ResearchBrief
- Planning queries and deciding breadth vs depth
- Routing each step to Reader Mode or Browser Mode
- Enforcing budgets, stop conditions, and quality gates
- Producing the final ResearchBundle
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from deepagents.compaction.models import (
    Confidence,
    EvidenceRecord,
    Finding,
    ReasoningState,
    ResearchBundle,
)
from deepagents.research.config import ResearchConfig
from deepagents.research.models import (
    ResearchBrief,
    ResearchMode,
    SourceQueueItem,
    SourceReasonCode,
    SourceStatus,
)

if TYPE_CHECKING:
    from deepagents.research.session import ResearchSession

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StopCondition:
    """Represents a stop condition for research."""
    
    def __init__(self, reason: str, details: str = ""):
        self.reason = reason
        self.details = details
    
    def __str__(self) -> str:
        return f"{self.reason}: {self.details}" if self.details else self.reason


class ResearchOrchestrator:
    """Orchestrator for research sessions.
    
    The orchestrator coordinates the entire research workflow:
    1. Parse and validate the research brief
    2. Generate search queries via source collector
    3. Process sources via reader or browser mode
    4. Extract findings and build evidence links
    5. Review quality and trigger follow-ups if needed
    6. Produce final ResearchBundle
    
    Args:
        session: The ResearchSession to operate on.
        config: Optional research config override.
    """
    
    def __init__(
        self,
        session: ResearchSession,
        config: ResearchConfig | None = None,
    ) -> None:
        self.session = session
        self.config = config or session.config
        
        # Lazy imports to avoid circular dependencies
        self._source_collector = None
        self._page_reader = None
        self._browser_operator = None
        self._distiller = None
        self._reviewer = None
    
    def create_brief(
        self,
        goal: str,
        *,
        constraints: dict[str, Any] | None = None,
        required_outputs: list[str] | None = None,
        mode_preference: ResearchMode = ResearchMode.BROWSER_ALLOWED,
        max_sources: int | None = None,
        max_steps: int | None = None,
    ) -> ResearchBrief:
        """Create a ResearchBrief from a goal.
        
        Args:
            goal: The research goal or question.
            constraints: Optional constraints dict.
            required_outputs: Items that must be in the bundle.
            mode_preference: Browser mode preference.
            max_sources: Override for max sources.
            max_steps: Override for max steps.
            
        Returns:
            A configured ResearchBrief.
        """
        brief = ResearchBrief(
            goal=goal,
            constraints=constraints or {},
            required_outputs=required_outputs or [],
            mode_preference=mode_preference,
            max_sources=max_sources or self.config.max_sources,
            max_steps=max_steps or self.config.max_steps,
            bundle_token_budget=self.config.bundle_token_budget,
        )
        
        # Apply recency constraint if provided
        if constraints and "recency" in constraints:
            recency = constraints["recency"]
            if isinstance(recency, str) and recency.startswith("last_"):
                days = int(recency.split("_")[1].replace("days", ""))
                brief.recency_days = days
        
        return brief
    
    def check_stop_conditions(self) -> StopCondition | None:
        """Check if any stop conditions are met.
        
        Returns:
            StopCondition if should stop, None otherwise.
        """
        brief = self.session.get_brief()
        if not brief:
            return StopCondition("no_brief", "No research brief set")
        
        # Check step limit
        if self.session.current_step >= brief.max_steps:
            return StopCondition(
                "step_limit",
                f"Reached max steps: {brief.max_steps}"
            )
        
        # Check source limit
        processed = len([
            s for s in self.session.get_source_queue()
            if s.status in (SourceStatus.READ, SourceStatus.BROWSED)
        ])
        if processed >= brief.max_sources:
            return StopCondition(
                "source_limit",
                f"Processed max sources: {brief.max_sources}"
            )
        
        # Check if all sources are processed
        pending = self.session.get_pending_sources()
        if not pending and processed > 0:
            return StopCondition(
                "queue_empty",
                "All sources processed"
            )
        
        return None

    def decide_mode(self, source: SourceQueueItem) -> ResearchMode:
        """Decide whether to use reader or browser mode for a source.

        Args:
            source: The source queue item.

        Returns:
            The recommended ResearchMode.
        """
        brief = self.session.get_brief()

        # Respect brief mode preference
        if brief and brief.mode_preference == ResearchMode.READER_ONLY:
            return ResearchMode.READER_ONLY
        if brief and brief.mode_preference == ResearchMode.BROWSER_REQUIRED:
            return ResearchMode.BROWSER_REQUIRED

        # Check if source has failed reader mode enough times
        if source.retry_count >= self.config.reader_fail_escalation_count:
            return ResearchMode.BROWSER_REQUIRED

        # Check if explicitly marked as needing browser
        if source.status == SourceStatus.BROWSER_NEEDED:
            return ResearchMode.BROWSER_REQUIRED

        # Default to reader mode first
        return ResearchMode.BROWSER_ALLOWED

    def update_reasoning_state(self) -> ReasoningState:
        """Create and store a reasoning state snapshot.

        Returns:
            The new ReasoningState.
        """
        evidence = self.session.evidence_ledger.get_all()
        sources = self.session.get_source_queue()
        brief = self.session.get_brief()

        # Build confirmed facts from processed sources
        confirmed_facts = []
        for ev in evidence[:5]:
            if ev.notes:
                confirmed_facts.append(ev.notes)

        # Identify open questions
        open_questions = []
        if brief and brief.required_outputs:
            # Check which required outputs we haven't found
            for req in brief.required_outputs:
                if not any(req.lower() in (ev.notes or "").lower() for ev in evidence):
                    open_questions.append(f"Still need to find: {req}")

        # Count by status
        read_count = sum(1 for s in sources if s.status == SourceStatus.READ)
        pending_count = sum(1 for s in sources if s.status == SourceStatus.QUEUED)

        state = ReasoningState(
            executive_summary=f"Step {self.session.current_step}: Read {read_count} sources, {pending_count} pending",
            confirmed_facts=confirmed_facts,
            hypotheses=[],
            open_questions=open_questions,
            visited_sources=[ev.url for ev in evidence],
            next_actions=[f"Process next source" for _ in range(min(3, pending_count))],
            step_number=self.session.current_step,
        )

        self.session.add_reasoning_state(state)
        return state

    def build_result_bundle(
        self,
        findings: list[Finding] | None = None,
    ) -> ResearchBundle:
        """Build the final ResearchBundle.

        Args:
            findings: Optional list of findings (extracted if not provided).

        Returns:
            The completed ResearchBundle.
        """
        evidence = self.session.evidence_ledger.get_all()
        brief = self.session.get_brief()

        # Use provided findings or create from evidence
        if findings is None:
            findings = self._extract_findings_from_evidence(evidence)

        # Build executive summary
        summary_parts = []
        if brief:
            summary_parts.append(f"Research on: {brief.goal[:100]}")
        summary_parts.append(f"Consulted {len(evidence)} sources from {len(self.session.evidence_ledger.get_unique_domains())} domains.")
        if findings:
            high_conf = sum(1 for f in findings if f.confidence == Confidence.HIGH)
            summary_parts.append(f"Found {len(findings)} findings ({high_conf} high confidence).")

        # Get open questions from latest reasoning state
        latest_state = self.session.get_latest_reasoning_state()
        open_questions = latest_state.open_questions if latest_state else []

        bundle = ResearchBundle(
            executive_summary=" ".join(summary_parts),
            findings=findings,
            evidence=evidence,
            open_questions=open_questions,
        )

        return bundle

    def _extract_findings_from_evidence(
        self,
        evidence: list[EvidenceRecord],
    ) -> list[Finding]:
        """Extract findings from evidence records.

        Args:
            evidence: List of evidence records.

        Returns:
            List of extracted findings.
        """
        findings = []

        for ev in evidence:
            # Create a finding from notes and quotes
            if ev.notes:
                findings.append(Finding(
                    claim=ev.notes[:300],
                    confidence=Confidence.MEDIUM,
                    evidence_artifact_ids=[ev.artifact_id],
                ))

            for quote in ev.quotes[:2]:
                if len(quote) > 30:
                    findings.append(Finding(
                        claim=f'According to source: "{quote[:200]}"',
                        confidence=Confidence.HIGH,
                        evidence_artifact_ids=[ev.artifact_id],
                    ))

        return findings[:20]  # Limit findings

    async def run_step(self) -> bool:
        """Run a single research step.

        Returns:
            True if should continue, False if should stop.
        """
        # Check stop conditions
        stop = self.check_stop_conditions()
        if stop:
            logger.info(f"Stopping research: {stop}")
            return False

        # Increment step
        self.session.increment_step()

        # Get next source to process
        pending = self.session.get_pending_sources()
        if not pending:
            logger.info("No pending sources")
            return False

        source = pending[0]
        mode = self.decide_mode(source)

        logger.info(f"Step {self.session.current_step}: Processing {source.url} in {mode} mode")

        # Process source (will be implemented in Phase 2)
        # For now, mark as processed
        try:
            # This will be replaced with actual processing
            success = await self._process_source(source, mode)
            if not success and mode != ResearchMode.BROWSER_REQUIRED:
                source.needs_browser()
                self.session.record_browser_escalation()
        except Exception as e:
            source.mark_error(str(e))
            self.session.record_error(str(e))
            logger.error(f"Error processing source: {e}")

        # Update metrics periodically
        if self.session.current_step % 5 == 0:
            self.session.update_artifact_metrics()
            self.update_reasoning_state()

        return True

    async def _process_source(
        self,
        source: SourceQueueItem,
        mode: ResearchMode,
    ) -> bool:
        """Process a single source.

        Args:
            source: The source to process.
            mode: The processing mode.

        Returns:
            True if successful, False otherwise.
        """
        # Placeholder - will be implemented in Phase 2
        # For now, just mark as read with a dummy artifact
        artifact_id, _ = self.session.artifact_store.write_artifact(
            f"Placeholder content for {source.url}",
            tool_name="page_reader",
            source_url=source.url,
            title=source.title,
        )

        source.mark_read(artifact_id)

        # Add evidence record
        self.session.evidence_ledger.add_record(
            url=source.url,
            artifact_id=artifact_id,
            title=source.title,
            notes=source.snippet,
        )

        return True

    async def research(
        self,
        goal: str,
        *,
        constraints: dict[str, Any] | None = None,
        required_outputs: list[str] | None = None,
    ) -> ResearchBundle:
        """Run a complete research session.

        Args:
            goal: The research goal or question.
            constraints: Optional constraints.
            required_outputs: Items that must be in the bundle.

        Returns:
            The final ResearchBundle.
        """
        # Create and set brief
        brief = self.create_brief(
            goal,
            constraints=constraints,
            required_outputs=required_outputs,
        )
        self.session.set_brief(brief)
        self.session.set_status("running")

        logger.info(f"Starting research: {goal[:100]}")

        # Run steps until done
        while await self.run_step():
            pass

        # Build and store result
        bundle = self.build_result_bundle()
        self.session.set_result(bundle)

        logger.info(f"Research complete: {len(bundle.findings)} findings, {len(bundle.evidence)} sources")

        return bundle
