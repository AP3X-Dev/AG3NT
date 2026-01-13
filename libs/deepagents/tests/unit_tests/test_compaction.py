"""Unit tests for the Context Window Compaction System."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from deepagents.compaction import (
    ArtifactStore,
    AssembledContext,
    CompactionConfig,
    ContextAssembler,
    EvidenceRecord,
    Finding,
    MaskedObservationPlaceholder,
    ObservationMasker,
    ReasoningState,
    ReasoningStateSummarizer,
    ResearchBundle,
    ResearchSubagentRunner,
    RetrievalIndex,
)
from deepagents.compaction.models import Confidence


class TestCompactionConfig:
    """Tests for CompactionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CompactionConfig()
        assert config.mask_tool_output_if_chars_gt == 6000
        assert config.keep_last_unmasked_tool_outputs == 3
        assert config.summarize_every_steps == 8
        assert config.model_context_limit == 128000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CompactionConfig(
            mask_tool_output_if_chars_gt=10000,
            keep_last_unmasked_tool_outputs=5,
            summarize_every_steps=10,
        )
        assert config.mask_tool_output_if_chars_gt == 10000
        assert config.keep_last_unmasked_tool_outputs == 5
        assert config.summarize_every_steps == 10

    def test_workspace_dir_creation(self):
        """Test workspace directory is created."""
        config = CompactionConfig()
        workspace = config.get_workspace_dir()
        assert workspace.exists()
        assert workspace.is_dir()

    def test_estimate_tokens(self):
        """Test token estimation."""
        config = CompactionConfig()
        text = "Hello world, this is a test."
        tokens = config.estimate_tokens(text)
        # ~4 chars per token
        assert tokens == len(text) // 4


class TestArtifactStore:
    """Tests for ArtifactStore."""

    @pytest.fixture
    def store(self):
        """Create a store with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompactionConfig(workspace_dir=Path(tmpdir))
            yield ArtifactStore(config)

    def test_write_artifact(self, store):
        """Test writing an artifact."""
        content = "This is test content for the artifact."
        artifact_id, path = store.write_artifact(
            content,
            tool_name="test_tool",
            title="Test Artifact",
        )

        assert artifact_id.startswith("art_")
        assert Path(path).exists()

    def test_read_artifact(self, store):
        """Test reading an artifact."""
        content = "This is test content for the artifact."
        artifact_id, _ = store.write_artifact(
            content,
            tool_name="test_tool",
        )

        read_content = store.read_artifact(artifact_id)
        assert read_content == content

    def test_artifact_not_found(self, store):
        """Test reading non-existent artifact."""
        result = store.read_artifact("art_nonexistent")
        assert result is None

    def test_get_metadata(self, store):
        """Test getting artifact metadata."""
        content = "Test content"
        artifact_id, _ = store.write_artifact(
            content,
            tool_name="test_tool",
            title="My Title",
            tags=["tag1", "tag2"],
        )

        meta = store.get_metadata(artifact_id)
        assert meta is not None
        assert meta.artifact_id == artifact_id
        assert meta.tool_name == "test_tool"
        assert meta.title == "My Title"
        assert "tag1" in meta.tags

    def test_list_artifacts(self, store):
        """Test listing artifacts."""
        store.write_artifact("Content 1", tool_name="tool_a")
        store.write_artifact("Content 2", tool_name="tool_b")
        store.write_artifact("Content 3", tool_name="tool_a")

        all_artifacts = store.list_artifacts()
        assert len(all_artifacts) == 3

        tool_a_only = store.list_artifacts(tool_name="tool_a")
        assert len(tool_a_only) == 2

    def test_secret_redaction(self, store):
        """Test that secrets are redacted."""
        content = 'api_key="sk-1234567890abcdefghij"'
        artifact_id, _ = store.write_artifact(content, tool_name="test")

        read_content = store.read_artifact(artifact_id)
        assert "sk-1234567890" not in read_content
        assert "REDACTED" in read_content

    def test_content_hash(self, store):
        """Test content hashing."""
        content = "Same content"
        id1, _ = store.write_artifact(content, tool_name="test")
        id2, _ = store.write_artifact(content, tool_name="test")

        meta1 = store.get_metadata(id1)
        meta2 = store.get_metadata(id2)
        assert meta1.content_hash == meta2.content_hash


class TestObservationMasker:
    """Tests for ObservationMasker."""

    @pytest.fixture
    def masker(self):
        """Create a masker with temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompactionConfig(
                workspace_dir=Path(tmpdir),
                mask_tool_output_if_chars_gt=100,  # Low threshold for testing
                keep_last_unmasked_tool_outputs=2,
            )
            store = ArtifactStore(config)
            yield ObservationMasker(store=store, config=config)

    def test_should_mask_large_content(self, masker):
        """Test masking decision for large content."""
        small_content = "Small content"
        large_content = "x" * 200

        assert masker.should_mask(small_content) is False
        assert masker.should_mask(large_content) is True

    def test_mask_observation_small(self, masker):
        """Test that small content is not masked."""
        result = masker.mask_observation(
            tool_call_id="call_1",
            tool_name="test_tool",
            content="Small content",
        )

        assert isinstance(result, str)
        assert result == "Small content"

    def test_mask_observation_large(self, masker):
        """Test that large content is masked."""
        large_content = "Important finding: " + "x" * 200

        result = masker.mask_observation(
            tool_call_id="call_1",
            tool_name="web_fetch",
            content=large_content,
        )

        assert isinstance(result, MaskedObservationPlaceholder)
        assert result.tool_name == "web_fetch"
        assert result.artifact_id.startswith("art_")

    def test_placeholder_text(self, masker):
        """Test placeholder text generation."""
        large_content = "x" * 200

        result = masker.mask_observation(
            tool_call_id="call_1",
            tool_name="test_tool",
            content=large_content,
        )

        placeholder_text = result.to_placeholder_text()
        assert "[MASKED OUTPUT: test_tool]" in placeholder_text
        assert "Artifact:" in placeholder_text
        assert "retrieve_snippets" in placeholder_text

    def test_evidence_ledger(self, masker):
        """Test evidence ledger tracking."""
        content = "Content from https://example.com/page " + "x" * 200

        masker.mask_observation(
            tool_call_id="call_1",
            tool_name="web_fetch",
            content=content,
            source_url="https://example.com/page",
        )

        evidence = masker.get_evidence_ledger()
        assert len(evidence) == 1
        assert evidence[0].url == "https://example.com/page"

    def test_masked_count(self, masker):
        """Test masked count tracking."""
        assert masker.get_masked_count() == 0

        masker.mask_observation("call_1", "tool", "x" * 200)
        assert masker.get_masked_count() == 1

        masker.mask_observation("call_2", "tool", "x" * 200)
        assert masker.get_masked_count() == 2


class TestReasoningStateSummarizer:
    """Tests for ReasoningStateSummarizer."""

    @pytest.fixture
    def summarizer(self):
        """Create a summarizer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CompactionConfig(
                workspace_dir=Path(tmpdir),
                summarize_every_steps=5,
            )
            store = ArtifactStore(config)
            masker = ObservationMasker(store=store, config=config)
            yield ReasoningStateSummarizer(config=config, masker=masker)

    def test_should_summarize_by_steps(self, summarizer):
        """Test step-based summarization trigger."""
        assert summarizer.should_summarize(step_number=1) is False
        assert summarizer.should_summarize(step_number=5) is True
        assert summarizer.should_summarize(step_number=10) is True

    def test_should_summarize_by_tokens(self, summarizer):
        """Test token-based summarization trigger."""
        # 70% of 128000 = 89600
        assert summarizer.should_summarize(step_number=1, estimated_tokens=50000) is False
        assert summarizer.should_summarize(step_number=1, estimated_tokens=100000) is True

    def test_summarize_creates_state(self, summarizer):
        """Test that summarize creates a reasoning state."""
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="Research the topic"),
            AIMessage(content="I found that the answer is 42. This is confirmed."),
        ]

        state = summarizer.summarize(messages, step_number=5)

        assert isinstance(state, ReasoningState)
        assert state.step_number == 5
        assert state.executive_summary != ""

    def test_format_for_context(self, summarizer):
        """Test formatting reasoning state for context."""
        state = ReasoningState(
            executive_summary="Research in progress",
            confirmed_facts=["Fact 1", "Fact 2"],
            hypotheses=["Hypothesis 1"],
            open_questions=["Question 1"],
            visited_sources=["https://example.com"],
            step_number=5,
        )

        formatted = summarizer.format_for_context(state)

        assert "Reasoning State Summary" in formatted
        assert "Fact 1" in formatted
        assert "Hypothesis 1" in formatted


class TestRetrievalIndex:
    """Tests for RetrievalIndex."""

    @pytest.fixture
    def index(self):
        """Create an index with temporary directory."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            config = CompactionConfig(workspace_dir=Path(tmpdir))
            store = ArtifactStore(config)
            retrieval_index = RetrievalIndex(config=config, store=store)
            yield retrieval_index, store
            # Close the database connection to release the file lock
            retrieval_index.close()

    def test_index_artifact(self, index):
        """Test indexing an artifact."""
        retrieval_index, store = index

        content = "This is a test document about Python programming."
        artifact_id, _ = store.write_artifact(content, tool_name="test")

        chunks = retrieval_index.index_artifact(artifact_id)
        assert chunks > 0

    def test_search(self, index):
        """Test searching indexed artifacts."""
        retrieval_index, store = index

        content = """
        Python is a programming language.
        It is widely used for web development.
        Machine learning uses Python extensively.
        """
        artifact_id, _ = store.write_artifact(content, tool_name="test")
        retrieval_index.index_artifact(artifact_id)

        results = retrieval_index.search("Python programming")
        assert len(results) > 0
        assert results[0].artifact_id == artifact_id

    def test_search_specific_artifact(self, index):
        """Test searching within a specific artifact."""
        retrieval_index, store = index

        content1 = "Document about cats and dogs."
        content2 = "Document about Python and JavaScript."

        id1, _ = store.write_artifact(content1, tool_name="test")
        id2, _ = store.write_artifact(content2, tool_name="test")

        retrieval_index.index_artifact(id1)
        retrieval_index.index_artifact(id2)

        results = retrieval_index.search("Python", artifact_id=id2)
        assert len(results) > 0
        assert all(r.artifact_id == id2 for r in results)

    def test_no_results(self, index):
        """Test search with no matching results."""
        retrieval_index, store = index

        content = "Document about something else entirely."
        artifact_id, _ = store.write_artifact(content, tool_name="test")
        retrieval_index.index_artifact(artifact_id)

        results = retrieval_index.search("xyznonexistent")
        assert len(results) == 0

    def test_thread_safety(self, index):
        """Test that the index works correctly across multiple threads."""
        import concurrent.futures
        import threading

        retrieval_index, store = index

        # Write some content
        content = "Python is a great programming language for data science."
        artifact_id, _ = store.write_artifact(content, tool_name="test")
        retrieval_index.index_artifact(artifact_id)

        errors = []
        results_from_threads = []
        threads_used = set()

        def search_in_thread(query: str, thread_num: int):
            """Search from a different thread."""
            try:
                threads_used.add(threading.current_thread().name)
                results = retrieval_index.search(query)
                results_from_threads.append((thread_num, len(results)))
            except Exception as e:
                errors.append((thread_num, str(e)))

        # Run searches from multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(10):
                futures.append(executor.submit(search_in_thread, "Python", i))
            concurrent.futures.wait(futures)

        # Verify no errors occurred
        assert len(errors) == 0, f"Thread errors: {errors}"

        # Verify all searches returned results
        assert len(results_from_threads) == 10
        for thread_num, count in results_from_threads:
            assert count > 0, f"Thread {thread_num} got no results"


class TestContextAssembler:
    """Tests for ContextAssembler."""

    @pytest.fixture
    def assembler(self):
        """Create an assembler."""
        config = CompactionConfig()
        return ContextAssembler(config)

    def test_assemble_empty(self, assembler):
        """Test assembling with no content."""
        result = assembler.assemble()

        assert isinstance(result, AssembledContext)
        assert result.total_tokens == 0
        assert len(result.blocks) == 0

    def test_assemble_with_content(self, assembler):
        """Test assembling with content."""
        result = assembler.assemble(
            working_memory="Current task: Research topic",
            plan_state="Step 1: Search\nStep 2: Analyze",
        )

        assert result.total_tokens > 0
        assert len(result.blocks) == 2

    def test_block_priorities(self, assembler):
        """Test that blocks are ordered by priority."""
        result = assembler.assemble(
            working_memory="Memory content",
            plan_state="Plan content",
            decision_ledger="Decisions made",
        )

        # Blocks should be sorted by priority in to_text
        text = result.to_text()
        # Working memory has priority 1, should come first
        assert text.index("Working Memory") < text.index("Plan State")

    def test_truncation(self, assembler):
        """Test content truncation."""
        # Create content that exceeds budget
        large_content = "x" * 10000

        result = assembler.assemble(
            working_memory=large_content,
            total_budget=100,  # Very small budget
        )

        assert "working_memory" in result.blocks_truncated

    def test_to_text(self, assembler):
        """Test converting to text."""
        result = assembler.assemble(
            working_memory="Memory content",
            plan_state="Plan content",
        )

        text = result.to_text()
        assert "## Working Memory" in text
        assert "## Plan State" in text
        assert "Memory content" in text


class TestResearchBundle:
    """Tests for ResearchBundle model."""

    def test_create_bundle(self):
        """Test creating a research bundle."""
        bundle = ResearchBundle(
            executive_summary="Research complete",
            findings=[
                Finding(
                    claim="The sky is blue",
                    confidence=Confidence.HIGH,
                    evidence_artifact_ids=["art_123"],
                )
            ],
            evidence=[
                EvidenceRecord(
                    url="https://example.com",
                    artifact_id="art_123",
                )
            ],
        )

        assert bundle.executive_summary == "Research complete"
        assert len(bundle.findings) == 1
        assert bundle.findings[0].confidence == Confidence.HIGH


class TestResearchSubagentRunner:
    """Tests for ResearchSubagentRunner."""

    @pytest.fixture
    def runner(self):
        """Create a runner."""
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            from deepagents.compaction.middleware import CompactionMiddleware

            middleware = CompactionMiddleware(workspace_dir=Path(tmpdir))
            yield ResearchSubagentRunner(middleware=middleware)
            # Close the database connection to release the file lock
            middleware.retrieval_index.close()

    def test_create_bundle_from_messages(self, runner):
        """Test creating bundle from messages."""
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="Research Python"),
            AIMessage(content="Finding: Python is widely used for data science."),
        ]

        bundle = runner.create_bundle_from_messages(
            task="Research Python",
            messages=messages,
        )

        assert isinstance(bundle, ResearchBundle)
        assert "Python" in bundle.executive_summary

    def test_review_bundle_pass(self, runner):
        """Test bundle review passing."""
        bundle = ResearchBundle(
            executive_summary="Complete research",
            findings=[
                Finding(
                    claim="Test finding",
                    confidence=Confidence.HIGH,
                    evidence_artifact_ids=["art_1"],  # Link to evidence
                )
            ],
            evidence=[EvidenceRecord(url="https://example.com", artifact_id="art_1")],
        )

        passed, issues = runner.review_bundle(bundle)
        assert passed is True
        assert len(issues) == 0

    def test_review_bundle_fail(self, runner):
        """Test bundle review failing."""
        bundle = ResearchBundle(
            executive_summary="Incomplete",
            findings=[],
            evidence=[],
        )

        passed, issues = runner.review_bundle(bundle, min_findings=1, min_evidence=1)
        assert passed is False
        assert len(issues) > 0

    def test_format_bundle_for_response(self, runner):
        """Test formatting bundle for response."""
        bundle = ResearchBundle(
            executive_summary="Research summary",
            findings=[Finding(claim="Important finding", confidence=Confidence.HIGH)],
            evidence=[EvidenceRecord(url="https://example.com", artifact_id="art_1")],
        )

        formatted = runner.format_bundle_for_response(bundle)

        assert "## Research Results" in formatted
        assert "Important finding" in formatted
        assert "Sources Consulted" in formatted
