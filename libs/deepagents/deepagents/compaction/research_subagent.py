"""ResearchSubagent wrapper for compacted research tasks.

This module provides a wrapper for running research subagents that:
1. Automatically use the compaction system
2. Return structured ResearchBundle results
3. Include evidence pointers for verification
4. Support optional reviewer logic
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from deepagents.compaction.models import (
    Confidence,
    EvidenceRecord,
    Finding,
    ResearchBundle,
)

if TYPE_CHECKING:
    from deepagents.compaction.middleware import CompactionMiddleware

logger = logging.getLogger(__name__)


class ResearchSubagentRunner:
    """Runner for research subagents with compaction support.

    Wraps research tasks to automatically use the compaction system
    and return structured ResearchBundle results.

    Args:
        middleware: CompactionMiddleware instance for artifact management.
        max_steps: Maximum steps for the research subagent.
        require_evidence: Whether to require evidence for findings.
    """

    def __init__(
        self,
        middleware: CompactionMiddleware,
        *,
        max_steps: int = 20,
        require_evidence: bool = True,
    ) -> None:
        self.middleware = middleware
        self.max_steps = max_steps
        self.require_evidence = require_evidence

    def _extract_findings_from_messages(
        self,
        messages: list[BaseMessage],
    ) -> list[Finding]:
        """Extract findings from research messages.

        Args:
            messages: The conversation messages.

        Returns:
            List of extracted findings.
        """
        findings: list[Finding] = []

        # Get evidence ledger for linking
        evidence = self.middleware.masker.get_evidence_ledger()
        artifact_ids = [e.artifact_id for e in evidence]

        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue

            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            # Look for finding patterns
            # Pattern: "Finding: <claim>" or "Conclusion: <claim>"
            import re
            patterns = [
                r"(?:finding|conclusion|result|discovered)[:\s]+(.+?)(?:\.|$)",
                r"(?:confirmed|verified|established)[:\s]+(.+?)(?:\.|$)",
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content, re.I | re.MULTILINE)
                for match in matches[:3]:
                    claim = match.strip()
                    if len(claim) > 20:
                        # Determine confidence based on language
                        confidence = Confidence.MEDIUM
                        if any(w in content.lower() for w in ["confirmed", "verified", "certain"]):
                            confidence = Confidence.HIGH
                        elif any(w in content.lower() for w in ["possibly", "might", "unclear"]):
                            confidence = Confidence.LOW

                        findings.append(Finding(
                            claim=claim[:300],
                            confidence=confidence,
                            evidence_artifact_ids=artifact_ids[:3],  # Link to recent artifacts
                        ))

        return findings[:10]  # Limit findings

    def _generate_executive_summary(
        self,
        task: str,
        findings: list[Finding],
        evidence: list[EvidenceRecord],
    ) -> str:
        """Generate an executive summary of the research.

        Args:
            task: The original research task.
            findings: Extracted findings.
            evidence: Evidence records.

        Returns:
            Executive summary string.
        """
        parts = [f"Research task: {task[:200]}"]

        if findings:
            high_conf = sum(1 for f in findings if f.confidence == Confidence.HIGH)
            parts.append(f"Found {len(findings)} findings ({high_conf} high confidence).")

        if evidence:
            parts.append(f"Consulted {len(evidence)} sources.")

        return " ".join(parts)

    def create_bundle_from_messages(
        self,
        task: str,
        messages: list[BaseMessage],
        extracted_data: dict[str, Any] | None = None,
    ) -> ResearchBundle:
        """Create a ResearchBundle from research messages.

        Args:
            task: The original research task.
            messages: The conversation messages from research.
            extracted_data: Optional structured data extracted.

        Returns:
            ResearchBundle with findings and evidence.
        """
        # Extract findings
        findings = self._extract_findings_from_messages(messages)

        # Get evidence from masker
        evidence = self.middleware.masker.get_evidence_ledger()

        # Generate summary
        executive_summary = self._generate_executive_summary(task, findings, evidence)

        # Identify open questions
        open_questions = self.middleware.summarizer.get_latest_summary()
        questions = open_questions.open_questions if open_questions else []

        return ResearchBundle(
            executive_summary=executive_summary,
            findings=findings,
            evidence=evidence,
            extracted_data_json=extracted_data,
            open_questions=questions,
        )

    def review_bundle(
        self,
        bundle: ResearchBundle,
        *,
        min_findings: int = 1,
        min_evidence: int = 1,
        require_high_confidence: bool = False,
    ) -> tuple[bool, list[str]]:
        """Review a research bundle for quality.

        Args:
            bundle: The ResearchBundle to review.
            min_findings: Minimum number of findings required.
            min_evidence: Minimum number of evidence sources required.
            require_high_confidence: Whether to require at least one high-confidence finding.

        Returns:
            Tuple of (passed, issues) where issues is a list of problems found.
        """
        issues: list[str] = []

        # Check findings count
        if len(bundle.findings) < min_findings:
            issues.append(f"Insufficient findings: {len(bundle.findings)} < {min_findings}")

        # Check evidence count
        if len(bundle.evidence) < min_evidence:
            issues.append(f"Insufficient evidence: {len(bundle.evidence)} < {min_evidence}")

        # Check for high confidence findings
        if require_high_confidence:
            high_conf = sum(1 for f in bundle.findings if f.confidence == Confidence.HIGH)
            if high_conf == 0:
                issues.append("No high-confidence findings")

        # Check evidence linkage
        if self.require_evidence:
            unlinked = sum(1 for f in bundle.findings if not f.evidence_artifact_ids)
            if unlinked > 0:
                issues.append(f"{unlinked} findings without evidence links")

        passed = len(issues) == 0
        return passed, issues

    def format_bundle_for_response(
        self,
        bundle: ResearchBundle,
        *,
        include_evidence_details: bool = True,
        max_findings: int = 10,
    ) -> str:
        """Format a ResearchBundle for inclusion in agent response.

        Args:
            bundle: The ResearchBundle to format.
            include_evidence_details: Whether to include evidence details.
            max_findings: Maximum findings to include.

        Returns:
            Formatted string representation.
        """
        lines = [
            "## Research Results",
            "",
            f"**Summary:** {bundle.executive_summary}",
            "",
        ]

        if bundle.findings:
            lines.append("### Findings")
            for i, finding in enumerate(bundle.findings[:max_findings], 1):
                conf_emoji = {"high": "✓", "medium": "○", "low": "?"}
                emoji = conf_emoji.get(finding.confidence.value, "○")
                lines.append(f"{i}. [{emoji}] {finding.claim}")
                if finding.notes:
                    lines.append(f"   *Note: {finding.notes}*")
            lines.append("")

        if include_evidence_details and bundle.evidence:
            lines.append("### Sources Consulted")
            for ev in bundle.evidence[:5]:
                title = ev.title or ev.url
                lines.append(f"- [{title}]({ev.url}) (artifact: {ev.artifact_id})")
            if len(bundle.evidence) > 5:
                lines.append(f"- ... and {len(bundle.evidence) - 5} more sources")
            lines.append("")

        if bundle.open_questions:
            lines.append("### Open Questions")
            for q in bundle.open_questions[:3]:
                lines.append(f"- {q}")
            lines.append("")

        if bundle.extracted_data_json:
            lines.append("### Extracted Data")
            lines.append("```json")
            import json
            lines.append(json.dumps(bundle.extracted_data_json, indent=2)[:500])
            lines.append("```")

        return "\n".join(lines)
