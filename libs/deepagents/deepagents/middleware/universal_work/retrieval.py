"""Retrieval and reranking system for Universal Work System.

Provides:
- Hybrid retrieval (keyword + embeddings) for candidate generation
- Reranking pipelines for duplicates, related items, assignees, priority
- Configurable retrieval backends
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deepagents.middleware.universal_work.models import (
    LinkType,
    SuggestionType,
    TriageSuggestion,
    TriageSuggestionBundle,
    WorkItem,
    WorkItemStatus,
)

if TYPE_CHECKING:
    from deepagents.middleware.universal_work.storage import WorkStorageProtocol

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """A candidate item from retrieval."""
    item: WorkItem
    score: float
    match_type: str  # "keyword", "embedding", "metadata"
    matched_text: str | None = None


class RetrievalBackend(ABC):
    """Abstract retrieval backend."""

    @abstractmethod
    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[RetrievalCandidate]:
        """Search for candidates matching query."""
        ...

    @abstractmethod
    def index_item(self, item: WorkItem) -> None:
        """Add or update an item in the index."""
        ...


class SimpleKeywordRetrieval(RetrievalBackend):
    """Simple keyword-based retrieval using in-memory search.
    
    Uses basic text matching for v1. Can be replaced with:
    - Elasticsearch for production keyword search
    - Vector DB (Pinecone, Weaviate, etc.) for embeddings
    """

    def __init__(self, storage: WorkStorageProtocol):
        self.storage = storage
        self._index: dict[str, set[str]] = {}  # word -> set of item IDs

    def _tokenize(self, text: str) -> set[str]:
        """Simple tokenization."""
        words = re.findall(r'\w+', text.lower())
        return set(words)

    def index_item(self, item: WorkItem) -> None:
        """Index a WorkItem for keyword search."""
        text = f"{item.title} {item.body} {' '.join(item.labels)}"
        tokens = self._tokenize(text)
        
        for token in tokens:
            if token not in self._index:
                self._index[token] = set()
            self._index[token].add(item.id)

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[RetrievalCandidate]:
        """Search for items matching query keywords."""
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Score by token overlap
        item_scores: dict[str, float] = {}
        for token in query_tokens:
            if token in self._index:
                for item_id in self._index[token]:
                    item_scores[item_id] = item_scores.get(item_id, 0) + 1
        
        # Normalize scores
        max_score = len(query_tokens)
        for item_id in item_scores:
            item_scores[item_id] /= max_score
        
        # Sort by score and get top items
        sorted_ids = sorted(item_scores.keys(), key=lambda x: item_scores[x], reverse=True)
        
        candidates = []
        for item_id in sorted_ids[:limit]:
            item = self.storage.get_work_item(item_id)
            if item is None:
                continue
            
            # Apply filters
            if filters:
                if filters.get("status") and item.status != filters["status"]:
                    continue
                if filters.get("domain") and item.domain != filters["domain"]:
                    continue
                if filters.get("exclude_id") and item.id == filters["exclude_id"]:
                    continue
            
            candidates.append(RetrievalCandidate(
                item=item,
                score=item_scores[item_id],
                match_type="keyword",
                matched_text=f"Matched {int(item_scores[item_id] * max_score)}/{max_score} keywords",
            ))
        
        return candidates

    def rebuild_index(self) -> None:
        """Rebuild the entire index from storage."""
        self._index.clear()
        for item in self.storage.list_work_items(limit=10000):
            self.index_item(item)


class Reranker(ABC):
    """Abstract reranker for specific suggestion types."""

    @abstractmethod
    def rerank(
        self,
        query_item: WorkItem,
        candidates: list[RetrievalCandidate],
        limit: int = 3,
    ) -> list[TriageSuggestion]:
        """Rerank candidates and produce suggestions."""
        ...


class DuplicateReranker(Reranker):
    """Reranker for duplicate detection.

    Uses title/body similarity to identify potential duplicates.
    """

    def _compute_similarity(self, item1: WorkItem, item2: WorkItem) -> float:
        """Compute text similarity between two items."""
        # Simple Jaccard similarity on tokens
        text1 = f"{item1.title} {item1.body}".lower()
        text2 = f"{item2.title} {item2.body}".lower()

        tokens1 = set(re.findall(r'\w+', text1))
        tokens2 = set(re.findall(r'\w+', text2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def rerank(
        self,
        query_item: WorkItem,
        candidates: list[RetrievalCandidate],
        limit: int = 3,
    ) -> list[TriageSuggestion]:
        """Rerank candidates for duplicate detection."""
        scored = []
        for candidate in candidates:
            if candidate.item.id == query_item.id:
                continue

            similarity = self._compute_similarity(query_item, candidate.item)

            # Only consider as duplicate if similarity > 0.5
            if similarity > 0.5:
                scored.append((candidate, similarity))

        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        suggestions = []
        for candidate, similarity in scored[:limit]:
            reasons = []
            if similarity > 0.8:
                reasons.append("Very high text similarity")
            elif similarity > 0.6:
                reasons.append("High text similarity")
            else:
                reasons.append("Moderate text similarity")

            if candidate.item.domain == query_item.domain:
                reasons.append("Same domain")

            if set(candidate.item.labels) & set(query_item.labels):
                reasons.append("Shared labels")

            suggestions.append(TriageSuggestion(
                suggestion_type=SuggestionType.DUPLICATE,
                suggested_value=candidate.item.id,
                confidence=similarity,
                reasons=reasons[:3],
                evidence=[candidate.item.id],
            ))

        return suggestions


class RelatedReranker(Reranker):
    """Reranker for finding related items.

    Uses broader matching criteria than duplicate detection.
    """

    def rerank(
        self,
        query_item: WorkItem,
        candidates: list[RetrievalCandidate],
        limit: int = 5,
    ) -> list[TriageSuggestion]:
        """Rerank candidates for related item detection."""
        scored = []
        for candidate in candidates:
            if candidate.item.id == query_item.id:
                continue

            # Score based on multiple factors
            score = candidate.score * 0.5  # Base retrieval score
            reasons = []

            # Domain match
            if candidate.item.domain == query_item.domain:
                score += 0.2
                reasons.append("Same domain")

            # Label overlap
            common_labels = set(candidate.item.labels) & set(query_item.labels)
            if common_labels:
                score += 0.1 * len(common_labels)
                reasons.append(f"Shared labels: {', '.join(list(common_labels)[:2])}")

            # Recency bonus for open items
            if candidate.item.status in [WorkItemStatus.IN_PROGRESS, WorkItemStatus.ACCEPTED]:
                score += 0.1
                reasons.append("Currently active")

            if not reasons:
                reasons.append("Text similarity")

            scored.append((candidate, min(score, 1.0), reasons))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        suggestions = []
        for candidate, score, reasons in scored[:limit]:
            suggestions.append(TriageSuggestion(
                suggestion_type=SuggestionType.RELATED,
                suggested_value=candidate.item.id,
                confidence=score,
                reasons=reasons[:3],
                evidence=[candidate.item.id],
            ))

        return suggestions


class TriageEngine:
    """Engine for generating triage suggestions.

    Coordinates retrieval and reranking to produce suggestion bundles.
    """

    def __init__(
        self,
        storage: WorkStorageProtocol,
        retrieval: RetrievalBackend | None = None,
    ):
        self.storage = storage
        self.retrieval = retrieval or SimpleKeywordRetrieval(storage)
        self.duplicate_reranker = DuplicateReranker()
        self.related_reranker = RelatedReranker()

    def generate_suggestions(
        self,
        item_id: str,
        modes: list[str] | None = None,
    ) -> TriageSuggestionBundle | str:
        """Generate triage suggestions for a WorkItem.

        Args:
            item_id: WorkItem ID to generate suggestions for
            modes: Suggestion types to generate. Default: all
                   Options: "duplicates", "related", "priority"

        Returns:
            TriageSuggestionBundle with suggestions, or error string
        """
        item = self.storage.get_work_item(item_id)
        if item is None:
            return f"WorkItem {item_id} not found"

        modes = modes or ["duplicates", "related"]
        bundle = TriageSuggestionBundle(work_item_id=item_id)

        # Ensure index is current
        if isinstance(self.retrieval, SimpleKeywordRetrieval):
            self.retrieval.rebuild_index()

        # Generate candidates
        query = f"{item.title} {item.body}"
        candidates = self.retrieval.search(
            query,
            filters={"exclude_id": item_id},
            limit=20,
        )

        if "duplicates" in modes:
            bundle.duplicates = self.duplicate_reranker.rerank(item, candidates, limit=3)

        if "related" in modes:
            bundle.related = self.related_reranker.rerank(item, candidates, limit=5)

        if "priority" in modes:
            # Simple priority suggestion based on keywords
            priority_keywords = {
                0: ["urgent", "critical", "blocker", "emergency", "asap"],
                1: ["important", "high", "priority", "soon"],
                2: [],  # default
                3: ["low", "minor", "nice-to-have"],
                4: ["backlog", "someday", "future"],
            }

            text = f"{item.title} {item.body}".lower()
            suggested_priority = 2  # default
            confidence = 0.5
            reasons = ["Default priority"]

            for priority, keywords in priority_keywords.items():
                if any(kw in text for kw in keywords):
                    suggested_priority = priority
                    confidence = 0.8
                    matched = [kw for kw in keywords if kw in text]
                    reasons = [f"Contains: {', '.join(matched[:2])}"]
                    break

            bundle.priority = TriageSuggestion(
                suggestion_type=SuggestionType.PRIORITY,
                suggested_value=suggested_priority,
                confidence=confidence,
                reasons=reasons,
                evidence=[],
            )

        return bundle

