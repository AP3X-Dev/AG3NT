"""Semantic memory search for AG3NT.

This module provides vector-based search over memory files (MEMORY.md, AGENTS.md,
and daily logs) using FAISS for similarity search and LangChain for embeddings.

Enhanced features:
- FAISS IVF indexing for datasets >100 vectors (with flat fallback for small datasets)
- Hybrid search combining semantic (50%), BM25 (25%), keyword (15%), and recency (10%)
- BM25 (Okapi BM25) full-text search for improved term matching
- Memory deduplication with content hashing and 0.95 similarity threshold

Memory files are chunked, embedded, and stored in a persistent FAISS index at
~/.ag3nt/vectors/. The index is refreshed when memory files are modified.

Usage:
    from ag3nt_agent.memory_search import search_memory, get_memory_search_tool

    # Direct search
    results = search_memory("user preferences for code style")

    # Get as LangChain tool
    tool = get_memory_search_tool()
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Constants
VECTORS_DIR = Path.home() / ".ag3nt" / "vectors"
INDEX_FILE = VECTORS_DIR / "faiss.index"
METADATA_FILE = VECTORS_DIR / "metadata.json"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks

# IVF indexing thresholds
IVF_THRESHOLD = 100  # Use IVF index when vectors exceed this count
DEFAULT_NLIST = 100  # Default number of clusters for IVF
MIN_NLIST = 4  # Minimum clusters (FAISS requirement)
NPROBE = 10  # Number of clusters to search

# Hybrid search weights (now includes BM25)
SEMANTIC_WEIGHT = 0.50  # Vector similarity
BM25_WEIGHT = 0.25  # BM25 full-text search
KEYWORD_WEIGHT = 0.15  # Simple keyword matching
RECENCY_WEIGHT = 0.10  # Recency scoring
RECENCY_DECAY_DAYS = 30  # Days for exponential decay half-life

# BM25 parameters
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Document length normalization parameter

# Deduplication settings
DEDUP_SIMILARITY_THRESHOLD = 0.95  # Cosine similarity threshold for deduplication


class IndexType(str, Enum):
    """FAISS index type."""

    FLAT = "flat"
    IVF = "ivf"


@dataclass
class SearchConfig:
    """Configuration for hybrid search.

    Attributes:
        semantic_weight: Weight for semantic similarity (0-1)
        bm25_weight: Weight for BM25 full-text search (0-1)
        keyword_weight: Weight for keyword matching (0-1)
        recency_weight: Weight for recency scoring (0-1)
        recency_decay_days: Half-life for recency exponential decay
        bm25_k1: BM25 term frequency saturation parameter
        bm25_b: BM25 document length normalization parameter
        enable_hybrid: Whether to use hybrid search (vs pure semantic)
        enable_bm25: Whether to include BM25 scoring in hybrid search
    """

    semantic_weight: float = SEMANTIC_WEIGHT
    bm25_weight: float = BM25_WEIGHT
    keyword_weight: float = KEYWORD_WEIGHT
    recency_weight: float = RECENCY_WEIGHT
    recency_decay_days: float = RECENCY_DECAY_DAYS
    bm25_k1: float = BM25_K1
    bm25_b: float = BM25_B
    enable_hybrid: bool = True
    enable_bm25: bool = True

    def __post_init__(self):
        total = self.semantic_weight + self.bm25_weight + self.keyword_weight + self.recency_weight
        if not (0.99 <= total <= 1.01):  # Allow small float errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")
        if self.recency_decay_days <= 0:
            raise ValueError("recency_decay_days must be positive")


@dataclass
class DeduplicationConfig:
    """Configuration for memory deduplication.

    Attributes:
        enabled: Whether to deduplicate chunks
        similarity_threshold: Cosine similarity threshold (0-1)
        use_content_hash: Also dedupe by exact content hash
    """

    enabled: bool = True
    similarity_threshold: float = DEDUP_SIMILARITY_THRESHOLD
    use_content_hash: bool = True

    def __post_init__(self):
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")


# Default configurations
DEFAULT_SEARCH_CONFIG = SearchConfig()
DEFAULT_DEDUP_CONFIG = DeduplicationConfig()


def _get_memory_dir() -> Path:
    """Get the memory directory path."""
    return Path.home() / ".ag3nt"


def _get_memory_files() -> list[Path]:
    """Get all memory files to index.

    Returns:
        List of memory file paths
    """
    memory_dir = _get_memory_dir()
    files: list[Path] = []

    # Main memory files
    for name in ["MEMORY.md", "AGENTS.md"]:
        f = memory_dir / name
        if f.exists():
            files.append(f)

    # Daily logs
    logs_dir = memory_dir / "memory"
    if logs_dir.exists():
        files.extend(logs_dir.glob("*.md"))

    return files


def _compute_files_hash(files: list[Path]) -> str:
    """Compute a hash of file paths and modification times.

    Used to detect when reindexing is needed.
    """
    hasher = hashlib.md5()
    for f in sorted(files):
        if f.exists():
            hasher.update(f.name.encode())
            hasher.update(str(f.stat().st_mtime).encode())
    return hasher.hexdigest()


def _chunk_text(text: str, source: str, mtime: float | None = None) -> list[dict[str, Any]]:
    """Split text into chunks with metadata.

    Args:
        text: The text content to chunk
        source: Source file path for metadata
        mtime: File modification time (Unix timestamp) for recency scoring

    Returns:
        List of chunks with text and metadata
    """
    timestamp = mtime or datetime.now().timestamp()

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback to simple chunking if splitter not available
        chunks = []
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i : i + CHUNK_SIZE]
            if chunk_text.strip():
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": source,
                        "chunk_index": len(chunks),
                        "mtime": timestamp,
                        "content_hash": hashlib.md5(chunk_text.encode()).hexdigest(),
                    }
                )
        return chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for i, chunk in enumerate(splitter.split_text(text)):
        chunks.append(
            {
                "text": chunk,
                "source": source,
                "chunk_index": i,
                "mtime": timestamp,
                "content_hash": hashlib.md5(chunk.encode()).hexdigest(),
            }
        )
    return chunks


def _compute_content_hash(text: str) -> str:
    """Compute MD5 hash of content for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()


def _compute_recency_score(mtime: float, decay_days: float = RECENCY_DECAY_DAYS) -> float:
    """Compute recency score with exponential decay.

    Args:
        mtime: Modification time as Unix timestamp
        decay_days: Half-life in days for decay

    Returns:
        Recency score between 0 and 1
    """
    now = datetime.now().timestamp()
    age_seconds = max(0, now - mtime)
    age_days = age_seconds / (24 * 3600)

    # Exponential decay with half-life = decay_days
    # score = 0.5^(age/half_life)
    return math.pow(0.5, age_days / decay_days)


def _compute_keyword_score(query: str, text: str) -> float:
    """Compute keyword matching score.

    Args:
        query: Search query
        text: Text to match against

    Returns:
        Score between 0 and 1 based on term overlap
    """
    query_terms = set(query.lower().split())
    text_lower = text.lower()

    if not query_terms:
        return 0.0

    matches = sum(1 for term in query_terms if term in text_lower)
    return matches / len(query_terms)


class BM25Index:
    """BM25 full-text search index for memory chunks.

    Implements the Okapi BM25 ranking function for efficient text retrieval.
    BM25 considers term frequency, document length, and inverse document frequency.
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        """Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter (default 1.5)
            b: Document length normalization parameter (default 0.75)
        """
        self._k1 = k1
        self._b = b
        self._documents: list[list[str]] = []
        self._doc_lengths: list[int] = []
        self._avg_doc_length: float = 0.0
        self._doc_freqs: dict[str, int] = {}  # Term -> num docs containing term
        self._idf: dict[str, float] = {}
        self._term_freqs: list[dict[str, int]] = []  # Per-doc term frequency maps
        self._initialized = False

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase and split on whitespace/punctuation."""
        import re
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def build(self, documents: list[str]) -> None:
        """Build the BM25 index from documents.

        Pre-computes per-document term frequency maps so that score()
        can look up term counts in O(1) instead of rebuilding them.

        Args:
            documents: List of document texts to index
        """
        self._documents = []
        self._doc_lengths = []
        self._doc_freqs = {}
        self._term_freqs = []

        # Tokenize, compute document frequencies, and pre-compute term freqs
        for doc in documents:
            tokens = self._tokenize(doc)
            self._documents.append(tokens)
            self._doc_lengths.append(len(tokens))

            # Build per-document term frequency map
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._term_freqs.append(tf)

            # Count unique terms in this document (for IDF)
            for term in tf:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        # Compute average document length
        total_length = sum(self._doc_lengths)
        self._avg_doc_length = total_length / len(documents) if documents else 1.0

        # Pre-compute IDF for all terms
        n = len(documents)
        for term, df in self._doc_freqs.items():
            # IDF with smoothing to avoid division by zero
            self._idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

        self._initialized = True
        logger.debug(f"Built BM25 index with {len(documents)} documents, {len(self._doc_freqs)} unique terms")

    def score(self, query: str, doc_idx: int, *, _query_tokens: list[str] | None = None) -> float:
        """Compute BM25 score for a query against a document.

        Uses precomputed per-document term frequency maps for O(1) lookups.

        Args:
            query: Search query
            doc_idx: Index of document to score
            _query_tokens: Pre-tokenized query tokens (avoids re-tokenizing
                per document when called from search()).

        Returns:
            BM25 score (higher is more relevant)
        """
        if not self._initialized or doc_idx >= len(self._documents):
            return 0.0

        query_tokens = _query_tokens if _query_tokens is not None else self._tokenize(query)
        doc_len = self._doc_lengths[doc_idx]
        term_freqs = self._term_freqs[doc_idx]

        score = 0.0
        for term in query_tokens:
            if term not in self._idf:
                continue

            idf = self._idf[term]
            tf = term_freqs.get(term, 0)

            # BM25 formula
            numerator = tf * (self._k1 + 1)
            denominator = tf + self._k1 * (1 - self._b + self._b * doc_len / self._avg_doc_length)
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Search for documents matching query.

        Tokenizes the query once and reuses the tokens for all document scores,
        avoiding O(n) redundant tokenization.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_idx, score) tuples, sorted by score descending
        """
        if not self._initialized:
            return []

        query_tokens = self._tokenize(query)
        scores = []
        for idx in range(len(self._documents)):
            score = self.score(query, idx, _query_tokens=query_tokens)
            if score > 0:
                scores.append((idx, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_normalized_score(self, query: str, doc_idx: int, *, _query_tokens: list[str] | None = None) -> float:
        """Get a normalized BM25 score between 0 and 1.

        Args:
            query: Search query
            doc_idx: Document index
            _query_tokens: Pre-tokenized query tokens to avoid redundant tokenization.

        Returns:
            Normalized score between 0 and 1
        """
        raw_score = self.score(query, doc_idx, _query_tokens=_query_tokens)
        # Normalize using sigmoid-like function
        # Score of ~10 maps to ~0.9
        return raw_score / (raw_score + 10.0) if raw_score > 0 else 0.0


def _get_embeddings():
    """Get the embeddings model.

    Uses OpenAI embeddings by default, configured via environment variables.
    Falls back to a simple TF-IDF approach if OpenAI is not available.
    """
    # Try OpenAI embeddings first (works with OpenRouter too)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )

    if api_key:
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=api_key,
                openai_api_base=base_url,
            )
        except ImportError:
            logger.warning("langchain-openai not installed, using fallback embeddings")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI embeddings: {e}")

    # Fallback: return None and use keyword matching
    logger.info("Using keyword-based memory search (no embeddings configured)")
    return None


class MemoryVectorStore:
    """Enhanced vector store with IVF indexing, hybrid search, and deduplication.

    Features:
    - Automatic IVF index selection for datasets >100 vectors
    - Hybrid search combining semantic, BM25, keyword, and recency scoring
    - Content deduplication using hashing and similarity threshold
    - BM25 full-text search for improved term matching
    """

    def __init__(
        self,
        search_config: SearchConfig | None = None,
        dedup_config: DeduplicationConfig | None = None,
    ):
        self._index = None
        self._metadata: list[dict] = []
        self._embeddings = None
        self._embeddings_matrix = None  # Cached for deduplication
        self._files_hash: str | None = None
        self._index_type: IndexType = IndexType.FLAT
        self._initialized = False
        self._search_config = search_config or DEFAULT_SEARCH_CONFIG
        self._dedup_config = dedup_config or DEFAULT_DEDUP_CONFIG
        self._dedup_stats = {"removed_by_hash": 0, "removed_by_similarity": 0}
        self._bm25_index: BM25Index | None = None

    def mark_dirty(self) -> None:
        """Mark the index as stale so it will be rebuilt on next search.

        Unlike destroying the singleton, this keeps the existing index
        usable for searches until the rebuild completes lazily.
        """
        self._files_hash = None

    @property
    def index_type(self) -> IndexType:
        """Get the current index type."""
        return self._index_type

    @property
    def dedup_stats(self) -> dict:
        """Get deduplication statistics."""
        return self._dedup_stats.copy()

    def _ensure_initialized(self) -> bool:
        """Ensure the index is initialized, rebuilding if needed.

        When the index has been marked dirty (``_files_hash is None``),
        re-checks the files and rebuilds if content has changed, while
        keeping the stale index usable until the new one is ready.

        Returns:
            True if vector search is available, False if falling back to keyword
        """
        # Fast path: already initialized and not marked dirty
        if self._initialized and self._files_hash is not None:
            return self._index is not None

        if not self._initialized:
            self._initialized = True
            try:
                self._embeddings = _get_embeddings()
            except Exception as e:
                logger.warning("Embeddings initialization failed (network?): %s", e)
                self._embeddings = None

        if self._embeddings is None:
            return False

        # Check if we need to rebuild
        memory_files = _get_memory_files()
        current_hash = _compute_files_hash(memory_files)

        # Try to load existing index (only on first init, not on dirty refresh)
        if self._index is None and INDEX_FILE.exists() and METADATA_FILE.exists():
            try:
                self._load_index()
                if self._files_hash == current_hash:
                    logger.debug("Memory index up to date")
                    return True
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")

        # Skip rebuild if hash matches (no actual change since mark_dirty)
        if self._files_hash == current_hash:
            return self._index is not None

        # Rebuild index
        try:
            self._rebuild_index(memory_files, current_hash)
        except (ImportError, Exception) as e:
            logger.warning("Memory indexing failed: %s", e)
            return False
        return self._index is not None

    def _load_index(self):
        """Load index from disk."""
        import faiss

        self._index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE) as f:
            data = json.load(f)
            self._metadata = data.get("chunks", [])
            self._files_hash = data.get("files_hash")
            self._index_type = IndexType(data.get("index_type", "flat"))

        # Rebuild BM25 index from loaded metadata
        if self._metadata and self._search_config.enable_bm25:
            self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Build the BM25 index from current metadata."""
        if not self._metadata:
            return

        texts = [chunk.get("contextualized_text", chunk["text"]) for chunk in self._metadata]
        self._bm25_index = BM25Index(
            k1=self._search_config.bm25_k1,
            b=self._search_config.bm25_b,
        )
        self._bm25_index.build(texts)
        logger.debug(f"Built BM25 index with {len(texts)} documents")

    def _save_index(self):
        """Save index to disk."""
        import faiss

        VECTORS_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(INDEX_FILE))
        with open(METADATA_FILE, "w") as f:
            json.dump(
                {
                    "chunks": self._metadata,
                    "files_hash": self._files_hash,
                    "index_type": self._index_type.value,
                    "dedup_stats": self._dedup_stats,
                },
                f,
            )

    def _deduplicate_chunks(
        self, chunks: list[dict], embeddings_np: "np.ndarray"
    ) -> tuple[list[dict], "np.ndarray"]:
        """Remove duplicate chunks using content hash and similarity.

        Args:
            chunks: List of chunk dictionaries
            embeddings_np: Corresponding embeddings matrix

        Returns:
            Tuple of (deduplicated chunks, deduplicated embeddings)
        """
        import numpy as np

        if not self._dedup_config.enabled or len(chunks) <= 1:
            return chunks, embeddings_np

        self._dedup_stats = {"removed_by_hash": 0, "removed_by_similarity": 0}

        # First pass: remove exact duplicates by content hash
        if self._dedup_config.use_content_hash:
            seen_hashes: set[str] = set()
            unique_indices: list[int] = []

            for i, chunk in enumerate(chunks):
                content_hash = chunk.get("content_hash") or _compute_content_hash(chunk["text"])
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_indices.append(i)
                else:
                    self._dedup_stats["removed_by_hash"] += 1

            chunks = [chunks[i] for i in unique_indices]
            embeddings_np = embeddings_np[unique_indices]

        # Second pass: remove near-duplicates by cosine similarity
        if len(chunks) > 1 and self._dedup_config.similarity_threshold < 1.0:
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            normalized = embeddings_np / norms

            keep_mask = np.ones(len(chunks), dtype=bool)

            for i in range(len(chunks)):
                if not keep_mask[i]:
                    continue
                # Compute similarity with remaining chunks
                for j in range(i + 1, len(chunks)):
                    if not keep_mask[j]:
                        continue
                    similarity = float(np.dot(normalized[i], normalized[j]))
                    if similarity >= self._dedup_config.similarity_threshold:
                        # Keep the more recent one
                        mtime_i = chunks[i].get("mtime", 0)
                        mtime_j = chunks[j].get("mtime", 0)
                        if mtime_i >= mtime_j:
                            keep_mask[j] = False
                        else:
                            keep_mask[i] = False
                            break
                        self._dedup_stats["removed_by_similarity"] += 1

            chunks = [c for c, keep in zip(chunks, keep_mask) if keep]
            embeddings_np = embeddings_np[keep_mask]

        logger.info(
            f"Deduplication: removed {self._dedup_stats['removed_by_hash']} by hash, "
            f"{self._dedup_stats['removed_by_similarity']} by similarity"
        )
        return chunks, embeddings_np

    def _create_index(self, dimension: int, num_vectors: int) -> tuple[Any, IndexType]:
        """Create the appropriate FAISS index based on dataset size.

        Args:
            dimension: Embedding dimension
            num_vectors: Number of vectors to index

        Returns:
            Tuple of (index, index_type)
        """
        import faiss

        if num_vectors < IVF_THRESHOLD:
            # Use flat index for small datasets (L2 distance)
            logger.info(f"Creating flat index for {num_vectors} vectors")
            return faiss.IndexFlatL2(dimension), IndexType.FLAT

        # Use IVF index for larger datasets
        # nlist = number of clusters, should be roughly sqrt(n) to 4*sqrt(n)
        nlist = min(DEFAULT_NLIST, max(MIN_NLIST, int(math.sqrt(num_vectors))))
        logger.info(f"Creating IVF index with {nlist} clusters for {num_vectors} vectors")

        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        return index, IndexType.IVF

    def _rebuild_index(self, memory_files: list[Path], files_hash: str):
        """Rebuild the index from memory files with IVF support and deduplication."""
        import faiss
        import numpy as np

        logger.info(f"Rebuilding memory index from {len(memory_files)} files")

        # Collect all chunks with modification time
        all_chunks: list[dict] = []
        source_documents: dict[str, str] = {}
        for file_path in memory_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                source = str(file_path.relative_to(_get_memory_dir()))
                mtime = file_path.stat().st_mtime
                chunks = _chunk_text(content, source, mtime)
                all_chunks.extend(chunks)
                source_documents[source] = content
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

        if not all_chunks:
            logger.warning("No memory content to index")
            return

        # Contextual enrichment: add LLM-generated context to each chunk
        try:
            from ag3nt_agent.context_enrichment import get_context_enricher

            enricher = get_context_enricher()
            all_chunks = enricher.enrich_chunks(all_chunks, source_documents)
            enriched_count = sum(
                1 for c in all_chunks
                if c.get("contextualized_text", c["text"]) != c["text"]
            )
            logger.info(
                f"Contextual enrichment: {enriched_count}/{len(all_chunks)} chunks enriched"
            )
        except ImportError:
            logger.debug("context_enrichment module not available, skipping enrichment")
            for chunk in all_chunks:
                chunk.setdefault("contextualized_text", chunk["text"])
        except Exception as e:
            logger.warning(f"Contextual enrichment failed: {e}, using raw chunks")
            for chunk in all_chunks:
                chunk.setdefault("contextualized_text", chunk["text"])

        # Get embeddings using contextualized text (better semantic capture)
        texts = [c.get("contextualized_text", c["text"]) for c in all_chunks]
        try:
            from ag3nt_agent.embedding_cache import get_embedding_cache

            cache = get_embedding_cache()

            def embed_batch(batch: list[str]) -> list[list[float]]:
                return self._embeddings.embed_documents(batch)

            embeddings = cache.get_or_compute_batch(
                texts, embed_batch, provider="openai", model="text-embedding-3-small"
            )
            embeddings_np = np.array(embeddings, dtype=np.float32)
            logger.debug(f"Embedding cache stats: {cache.get_stats().hit_rate:.1%} hit rate")
        except ImportError:
            # Fallback if cache not available
            embeddings = self._embeddings.embed_documents(texts)
            embeddings_np = np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to embed memory chunks: {e}")
            return

        # Deduplicate chunks
        all_chunks, embeddings_np = self._deduplicate_chunks(all_chunks, embeddings_np)

        if len(all_chunks) == 0:
            logger.warning("No chunks remaining after deduplication")
            return

        # Create appropriate index based on dataset size
        dimension = embeddings_np.shape[1]
        self._index, self._index_type = self._create_index(dimension, len(all_chunks))

        # Train IVF index if needed
        if self._index_type == IndexType.IVF:
            logger.info("Training IVF index...")
            self._index.train(embeddings_np)
            self._index.nprobe = NPROBE

        # Add vectors to index
        self._index.add(embeddings_np)
        self._metadata = all_chunks
        self._files_hash = files_hash
        self._embeddings_matrix = embeddings_np  # Cache for potential later use

        # Build BM25 index for full-text search
        if self._search_config.enable_bm25:
            self._build_bm25_index()

        # Save to disk
        self._save_index()
        logger.info(
            f"Indexed {len(all_chunks)} memory chunks using {self._index_type.value} index"
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search memory for relevant content using hybrid search.

        Combines semantic similarity (vector search), keyword matching, and
        recency scoring with configurable weights.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of relevant memory chunks with metadata and hybrid scores
        """
        has_vectors = self._ensure_initialized()

        if has_vectors:
            if self._search_config.enable_hybrid:
                return self._hybrid_search(query, top_k)
            else:
                return self._vector_search(query, top_k)
        else:
            return self._keyword_search(query, top_k)

    def _hybrid_search(self, query: str, top_k: int) -> list[dict]:
        """Perform hybrid search combining semantic, BM25, keyword, and recency scores.

        Args:
            query: Natural language query
            top_k: Number of results to return

        Returns:
            List of results with hybrid scores
        """
        import numpy as np

        try:
            # Get query embedding
            query_embedding = self._embeddings.embed_query(query)
            query_np = np.array([query_embedding], dtype=np.float32)

            # Retrieve more candidates for re-ranking (3x top_k)
            candidate_k = min(top_k * 3, len(self._metadata))
            distances, indices = self._index.search(query_np, candidate_k)

            # Compute hybrid scores for each candidate
            candidates = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self._metadata):
                    continue

                chunk = self._metadata[idx].copy()

                # Semantic score: convert L2 distance to similarity (0-1)
                semantic_score = float(1 / (1 + distances[0][i]))

                # BM25 score (if enabled and index available)
                bm25_score = 0.0
                if (
                    self._search_config.enable_bm25
                    and self._bm25_index is not None
                ):
                    bm25_score = self._bm25_index.get_normalized_score(query, idx)

                # Keyword score
                keyword_score = _compute_keyword_score(query, chunk["text"])

                # Recency score (use mtime if available, else default to recent)
                mtime = chunk.get("mtime", datetime.now().timestamp())
                recency_score = _compute_recency_score(
                    mtime, self._search_config.recency_decay_days
                )

                # Compute hybrid score with all components
                hybrid_score = (
                    self._search_config.semantic_weight * semantic_score
                    + self._search_config.bm25_weight * bm25_score
                    + self._search_config.keyword_weight * keyword_score
                    + self._search_config.recency_weight * recency_score
                )

                # Store component scores for transparency
                chunk["score"] = hybrid_score
                chunk["semantic_score"] = semantic_score
                chunk["bm25_score"] = bm25_score
                chunk["keyword_score"] = keyword_score
                chunk["recency_score"] = recency_score
                candidates.append(chunk)

            # Sort by hybrid score and return top_k
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return self._keyword_search(query, top_k)

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """Perform pure vector similarity search (no hybrid scoring)."""
        import numpy as np

        try:
            query_embedding = self._embeddings.embed_query(query)
            query_np = np.array([query_embedding], dtype=np.float32)

            distances, indices = self._index.search(query_np, min(top_k, len(self._metadata)))

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self._metadata):
                    continue
                chunk = self._metadata[idx].copy()
                chunk["score"] = float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                chunk["semantic_score"] = chunk["score"]
                results.append(chunk)

            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return self._keyword_search(query, top_k)

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """Fallback keyword-based search when vectors unavailable."""
        memory_files = _get_memory_files()
        results = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for file_path in memory_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                source = str(file_path.relative_to(_get_memory_dir()))
                mtime = file_path.stat().st_mtime

                # Score each line
                for line in content.split("\n"):
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue

                    keyword_score = _compute_keyword_score(query, line_stripped)
                    if keyword_score > 0:
                        recency_score = _compute_recency_score(
                            mtime, self._search_config.recency_decay_days
                        )
                        # Weight keyword higher since we don't have semantic
                        score = 0.7 * keyword_score + 0.3 * recency_score

                        results.append({
                            "text": line_stripped,
                            "source": source,
                            "score": score,
                            "keyword_score": keyword_score,
                            "recency_score": recency_score,
                        })
            except Exception:
                continue

        # Sort by score and limit
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


# Singleton instance
_memory_store: MemoryVectorStore | None = None
_memory_store_lock = threading.Lock()


def _get_memory_store() -> MemoryVectorStore:
    """Get or create the singleton memory store."""
    global _memory_store
    if _memory_store is None:
        with _memory_store_lock:
            if _memory_store is None:
                _memory_store = MemoryVectorStore()
    return _memory_store


def reset_memory_store() -> None:
    """Invalidate the memory store index so it rebuilds on next search.

    Instead of destroying the singleton (which forces a full re-embed),
    this marks the index as dirty. The existing stale index remains
    usable for searches until the lazy rebuild completes.
    """
    with _memory_store_lock:
        if _memory_store is not None:
            _memory_store.mark_dirty()


def get_memory_store_info() -> dict:
    """Get information about the current memory store.

    Returns:
        Dictionary with index type, chunk count, dedup stats, etc.
    """
    store = _get_memory_store()
    store._ensure_initialized()

    return {
        "index_type": store.index_type.value,
        "chunk_count": len(store._metadata),
        "dedup_stats": store.dedup_stats,
        "files_hash": store._files_hash,
        "has_vectors": store._index is not None,
        "search_config": {
            "semantic_weight": store._search_config.semantic_weight,
            "keyword_weight": store._search_config.keyword_weight,
            "recency_weight": store._search_config.recency_weight,
            "recency_decay_days": store._search_config.recency_decay_days,
            "enable_hybrid": store._search_config.enable_hybrid,
        },
        "dedup_config": {
            "enabled": store._dedup_config.enabled,
            "similarity_threshold": store._dedup_config.similarity_threshold,
            "use_content_hash": store._dedup_config.use_content_hash,
        },
    }


def search_memory(query: str, top_k: int = 5) -> dict:
    """Search memory for relevant information using hybrid search.

    Combines semantic similarity (60%), keyword matching (25%), and
    recency scoring (15%) for optimal relevance.

    Args:
        query: Natural language query describing what you're looking for
        top_k: Number of results to return (default: 5)

    Returns:
        Dictionary with results, metadata, and index information
    """
    store = _get_memory_store()
    results = store.search(query, top_k)

    if not results:
        return {
            "results": [],
            "message": "No relevant memories found. Memory files may be empty.",
            "index_type": store.index_type.value,
        }

    return {
        "results": results,
        "count": len(results),
        "query": query,
        "index_type": store.index_type.value,
        "search_mode": "hybrid" if store._search_config.enable_hybrid else "semantic",
    }


def get_memory_search_tool():
    """Get the memory search tool for the agent.

    Returns:
        LangChain tool for memory search
    """
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def memory_search(query: str, top_k: int = 5) -> dict:
        """Search your memory for relevant information.

        Use this tool to recall information from past conversations, user preferences,
        project context, and other stored knowledge. This uses hybrid search combining:
        - Semantic similarity (60%): Understanding meaning and context
        - Keyword matching (25%): Exact term matches
        - Recency (15%): Prioritizing recent memories

        Args:
            query: What you're looking for (natural language description)
            top_k: Number of results to return (default: 5)

        Returns:
            Relevant memory entries with source files and relevance scores
        """
        return search_memory(query, top_k)

    return memory_search


def get_memory_store_tool():
    """Get the memory store tool for the agent.

    Returns:
        LangChain tool for storing notes to memory
    """
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def memory_store(
        content: str,
        category: str = "General",
    ) -> str:
        """Store a note or piece of information to long-term memory.

        Use this to persist user preferences, project decisions, important facts,
        or any information that should be recalled in future sessions. Notes are
        appended to the daily memory log at ~/.ag3nt/memory/.

        Args:
            content: The information to store (e.g., "User prefers conventional commits")
            category: Optional category label (e.g., "User Preferences", "Project Decisions")

        Returns:
            Confirmation message with the file path where the note was saved
        """
        import datetime

        today = datetime.date.today().isoformat()
        memory_dir = Path.home() / ".ag3nt" / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        daily_file = memory_dir / f"{today}.md"

        # Build entry
        timestamp = datetime.datetime.now().strftime("%H:%M")
        entry = f"\n## {category} ({timestamp})\n- {content}\n"

        # Append to daily log (create with header if new)
        if not daily_file.exists():
            daily_file.write_text(
                f"# Memory Notes - {today}\n{entry}",
                encoding="utf-8",
            )
        else:
            with open(daily_file, "a", encoding="utf-8") as f:
                f.write(entry)

        # Invalidate the vector index so the new note is found on next search
        reset_memory_store()

        return f"Stored to {daily_file}: [{category}] {content}"

    return memory_store
