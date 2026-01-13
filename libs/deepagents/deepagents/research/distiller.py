"""Distiller for extracting and synthesizing research findings.

The Distiller is responsible for:
- Extracting key facts and claims from source content
- Synthesizing findings across multiple sources
- Producing structured Finding objects with evidence links
- Enforcing output token budgets
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from deepagents.compaction.models import Confidence, Finding
from deepagents.research.config import ResearchConfig
from deepagents.research.page_reader import PageContent

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class ExtractionResult:
    """Result of extracting information from a source."""

    source_url: str
    artifact_id: str

    # Extracted content
    key_facts: list[str] = field(default_factory=list)
    quotes: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)

    # Metadata
    topic_relevance: float = 0.5
    information_density: float = 0.5

    # Errors
    error: str | None = None


class Extractor:
    """Extracts structured information from page content.

    The Extractor uses pattern matching and heuristics to identify:
    - Key facts and claims
    - Important quotes
    - Named entities (people, organizations, products)
    - Numerical data and statistics

    Args:
        config: Research configuration.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config

    def extract(
        self,
        content: PageContent,
        artifact_id: str,
        goal: str | None = None,
    ) -> ExtractionResult:
        """Extract information from page content.

        Args:
            content: The page content to extract from.
            artifact_id: ID of the stored artifact.
            goal: Optional research goal for relevance scoring.

        Returns:
            ExtractionResult with extracted information.
        """
        if content.error:
            return ExtractionResult(
                source_url=content.url,
                artifact_id=artifact_id,
                error=content.error,
            )

        text = content.content

        # Extract key facts (sentences with factual indicators)
        key_facts = self._extract_key_facts(text)

        # Extract quotes (text in quotation marks)
        quotes = self._extract_quotes(text)

        # Extract entities
        entities = self._extract_entities(text)

        # Calculate relevance if goal provided
        topic_relevance = 0.5
        if goal:
            topic_relevance = self._calculate_relevance(text, goal)

        # Calculate information density
        information_density = self._calculate_density(text, key_facts, quotes)

        return ExtractionResult(
            source_url=content.url,
            artifact_id=artifact_id,
            key_facts=key_facts[:10],  # Limit to top 10
            quotes=quotes[:5],  # Limit to top 5
            entities=entities[:20],  # Limit to top 20
            topic_relevance=topic_relevance,
            information_density=information_density,
        )

    def _extract_key_facts(self, text: str) -> list[str]:
        """Extract key factual statements."""
        facts = []

        # Split into sentences
        sentences = re.split(r"[.!?]+", text)

        # Factual indicators
        indicators = [
            r"\b\d+%",  # Percentages
            r"\$[\d,]+",  # Dollar amounts
            r"\b\d{4}\b",  # Years
            r"\b(according to|research shows|studies indicate|data shows)\b",
            r"\b(announced|released|launched|introduced)\b",
            r"\b(increased|decreased|grew|declined)\b",
        ]

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 300:
                continue

            for pattern in indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    facts.append(sentence)
                    break

        return facts

    def _extract_quotes(self, text: str) -> list[str]:
        """Extract quoted text."""
        quotes = []

        # Match text in quotes
        patterns = [
            r'"([^"]{20,200})"',
            r"'([^']{20,200})'",
            r'"([^"]{20,200})"',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            quotes.extend(matches)

        return quotes

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities (simplified)."""
        entities = []

        # Look for capitalized phrases (simple NER)
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
        matches = re.findall(pattern, text)

        # Deduplicate
        seen = set()
        for match in matches:
            if match not in seen and len(match) > 3:
                entities.append(match)
                seen.add(match)

        return entities

    def _calculate_relevance(self, text: str, goal: str) -> float:
        """Calculate topic relevance score."""
        text_lower = text.lower()
        goal_lower = goal.lower()

        # Extract keywords from goal
        goal_words = set(re.findall(r"\b\w{4,}\b", goal_lower))

        # Count matches
        matches = sum(1 for word in goal_words if word in text_lower)

        if not goal_words:
            return 0.5

        return min(1.0, matches / len(goal_words))

    def _calculate_density(
        self,
        text: str,
        facts: list[str],
        quotes: list[str],
    ) -> float:
        """Calculate information density score."""
        if not text:
            return 0.0

        word_count = len(text.split())
        if word_count == 0:
            return 0.0

        # Score based on facts and quotes per 100 words
        info_items = len(facts) + len(quotes)
        density = (info_items / word_count) * 100

        return min(1.0, density / 5)  # Normalize to 0-1


class Distiller:
    """Distills findings from extracted information.

    The Distiller synthesizes information across sources to produce
    structured Finding objects with confidence levels and evidence links.

    Args:
        config: Research configuration.
    """

    def __init__(self, config: ResearchConfig) -> None:
        self.config = config
        self.extractor = Extractor(config)

    def distill(
        self,
        extractions: list[ExtractionResult],
        goal: str,
        token_budget: int | None = None,
    ) -> list[Finding]:
        """Distill findings from extractions.

        Args:
            extractions: List of extraction results.
            goal: The research goal.
            token_budget: Maximum tokens for output.

        Returns:
            List of Finding objects.
        """
        token_budget = token_budget or self.config.bundle_token_budget
        findings: list[Finding] = []

        # Group facts by similarity
        fact_groups = self._group_similar_facts(extractions)

        # Create findings from groups
        for group in fact_groups:
            finding = self._create_finding(group, goal)
            if finding:
                findings.append(finding)

        # Add quote-based findings
        for extraction in extractions:
            for quote in extraction.quotes[:2]:
                findings.append(
                    Finding(
                        claim=f'Source states: "{quote[:150]}"',
                        confidence=Confidence.HIGH,
                        evidence_artifact_ids=[extraction.artifact_id],
                    )
                )

        # Sort by confidence and relevance
        findings.sort(
            key=lambda f: ({"high": 3, "medium": 2, "low": 1}.get(f.confidence.value, 0),),
            reverse=True,
        )

        # Enforce token budget (rough estimate: 1 token ≈ 4 chars)
        total_chars = 0
        max_chars = token_budget * 4
        filtered_findings = []

        for finding in findings:
            finding_chars = len(finding.claim) + 50  # Overhead
            if total_chars + finding_chars <= max_chars:
                filtered_findings.append(finding)
                total_chars += finding_chars

        return filtered_findings

    def _group_similar_facts(
        self,
        extractions: list[ExtractionResult],
    ) -> list[list[tuple[str, str]]]:
        """Group similar facts across sources.

        Returns list of groups, where each group is a list of (fact, artifact_id) tuples.
        """
        all_facts: list[tuple[str, str]] = []

        for extraction in extractions:
            for fact in extraction.key_facts:
                all_facts.append((fact, extraction.artifact_id))

        # Simple grouping by keyword overlap
        groups: list[list[tuple[str, str]]] = []
        used = set()

        for i, (fact1, aid1) in enumerate(all_facts):
            if i in used:
                continue

            group = [(fact1, aid1)]
            used.add(i)

            words1 = set(fact1.lower().split())

            for j, (fact2, aid2) in enumerate(all_facts[i + 1 :], i + 1):
                if j in used:
                    continue

                words2 = set(fact2.lower().split())
                overlap = len(words1 & words2) / max(len(words1), len(words2), 1)

                if overlap > 0.3:
                    group.append((fact2, aid2))
                    used.add(j)

            groups.append(group)

        return groups

    def _create_finding(
        self,
        group: list[tuple[str, str]],
        goal: str,
    ) -> Finding | None:
        """Create a finding from a group of similar facts."""
        if not group:
            return None

        # Use the first fact as the claim
        claim = group[0][0]

        # Collect all artifact IDs
        artifact_ids = list({aid for _, aid in group})

        # Determine confidence based on corroboration
        if len(artifact_ids) >= 3:
            confidence = Confidence.HIGH
        elif len(artifact_ids) >= 2:
            confidence = Confidence.MEDIUM
        else:
            confidence = Confidence.LOW

        return Finding(
            claim=claim,
            confidence=confidence,
            evidence_artifact_ids=artifact_ids,
        )
