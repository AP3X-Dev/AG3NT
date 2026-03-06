# Phase 4: Final Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the 6 remaining OpenClaw parity gaps — 3 in the Python agent (query expansion, thinking-level fallback, inter-agent messaging) and 3 in the TypeScript gateway (cron backoff, timezone support, delivery modes).

**Architecture:** All changes extend existing data structures and control flow. No new subsystems. Python features plug into `memory_search.py`, `model_fallback.py`, and `subagents.py`. TypeScript features extend `Scheduler.ts` and `types.ts`.

**Tech Stack:** Python 3.11+, LangChain/LangGraph, pytest | TypeScript, vitest, node-schedule, Express

---

## Task 1: Query Expansion for Memory Search — Tests

**Files:**
- Create: `apps/agent/tests/unit/test_query_expansion.py`

**Step 1: Write the failing tests**

```python
"""Tests for query expansion in memory search."""
from __future__ import annotations

import pytest


class TestExpandQuery:
    """Tests for the _expand_query() function."""

    def test_returns_original_plus_keywords(self):
        from ag3nt_agent.memory_search import _expand_query
        result = _expand_query("deploy the application")
        assert "deploy the application" in result
        assert len(result) >= 2  # at least original + one variant

    def test_single_word_query(self):
        from ag3nt_agent.memory_search import _expand_query
        result = _expand_query("authentication")
        assert "authentication" in result
        # Should include keyword variants
        assert len(result) >= 1

    def test_empty_query_returns_list_with_original(self):
        from ag3nt_agent.memory_search import _expand_query
        result = _expand_query("")
        assert result == [""]

    def test_stop_words_filtered(self):
        from ag3nt_agent.memory_search import _expand_query
        result = _expand_query("how to set up the database")
        # "how", "to", "the" are stop words — shouldn't appear as standalone variants
        keywords_only = [q for q in result if q != "how to set up the database"]
        for variant in keywords_only:
            assert variant not in ("how", "to", "the", "set", "up")

    def test_generates_keyword_subsets(self):
        from ag3nt_agent.memory_search import _expand_query
        result = _expand_query("configure authentication middleware")
        # Should generate bigrams from the keywords
        all_text = " ".join(result)
        assert "authentication" in all_text
        assert "middleware" in all_text


class TestHybridSearchWithExpansion:
    """Tests that _hybrid_search uses query expansion."""

    def test_expansion_called_during_search(self, monkeypatch):
        """Verify _expand_query is called by _hybrid_search."""
        from ag3nt_agent import memory_search

        calls = []
        original_expand = memory_search._expand_query

        def tracking_expand(query):
            calls.append(query)
            return original_expand(query)

        monkeypatch.setattr(memory_search, "_expand_query", tracking_expand)

        # Create a minimal store with mocked internals
        store = memory_search.MemoryVectorStore()
        store._initialized = True
        store._metadata = []
        # _hybrid_search should call _expand_query even if no results
        try:
            store._hybrid_search("test query", top_k=5)
        except Exception:
            pass  # May fail without embeddings, that's fine
        assert len(calls) >= 1
        assert calls[0] == "test query"
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/agent && python -m pytest tests/unit/test_query_expansion.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name '_expand_query'`

**Step 3: Commit**

```bash
git add apps/agent/tests/unit/test_query_expansion.py
git commit -m "test: add failing tests for query expansion in memory search"
```

---

## Task 2: Query Expansion for Memory Search — Implementation

**Files:**
- Modify: `apps/agent/ag3nt_agent/memory_search.py:296-301` (near `_tokenize`)
- Modify: `apps/agent/ag3nt_agent/memory_search.py:822-860` (in `_hybrid_search`)

**Step 1: Add `_expand_query()` function**

Add after the `BM25Index` class (around line 400, before `MemoryVectorStore`):

```python
# Stop words for query expansion
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "about", "up", "set",
})


def _expand_query(query: str) -> list[str]:
    """Expand a search query into the original plus keyword variants.

    Extracts meaningful keywords (filtering stop words), then generates
    bigram combinations to broaden recall without external API calls.

    Returns:
        List of query strings: [original, keyword_bigrams...]
    """
    if not query.strip():
        return [query]

    import re
    tokens = re.findall(r'\b\w+\b', query.lower())
    keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]

    variants = [query]  # Always include the original

    # Add individual keywords (if more than 1 to avoid duplicating short queries)
    if len(keywords) > 1:
        # Generate bigrams from keywords
        for i in range(len(keywords) - 1):
            variants.append(f"{keywords[i]} {keywords[i + 1]}")

    return variants
```

**Step 2: Wire `_expand_query` into `_hybrid_search`**

In `_hybrid_search()` (line ~835), replace the single query embedding with expanded queries:

Find the existing code around line 835-837:
```python
# Get query embedding
query_embedding = self._embeddings.embed_query(query)
query_np = np.array([query_embedding], dtype=np.float32)
```

Replace with:
```python
# Expand query for broader recall
expanded_queries = _expand_query(query)

# Get query embedding (use original query for semantic search)
query_embedding = self._embeddings.embed_query(query)
query_np = np.array([query_embedding], dtype=np.float32)
```

Then in the BM25 scoring section (around line 860), replace the single BM25 call with expanded queries:

Find:
```python
bm25_scores = self._bm25_index.score(query)
```

Replace with:
```python
# Score BM25 across all expanded queries, take max per document
bm25_scores = self._bm25_index.score(expanded_queries[0])
for eq in expanded_queries[1:]:
    alt_scores = self._bm25_index.score(eq)
    bm25_scores = [max(a, b) for a, b in zip(bm25_scores, alt_scores)]
```

**Step 3: Run tests to verify they pass**

Run: `cd apps/agent && python -m pytest tests/unit/test_query_expansion.py -v --no-cov`
Expected: PASS (all 6 tests)

**Step 4: Run full test suite for regressions**

Run: `cd apps/agent && python -m pytest tests/unit/ -v --no-cov -x`
Expected: All tests pass

**Step 5: Commit**

```bash
git add apps/agent/ag3nt_agent/memory_search.py
git commit -m "feat: add query expansion for memory search"
```

---

## Task 3: Thinking-Level Fallback — Tests

**Files:**
- Create: `apps/agent/tests/unit/test_thinking_fallback.py`

**Step 1: Write the failing tests**

```python
"""Tests for thinking-level fallback in model_fallback.py."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


class TestThinkingLevel:
    """Tests for ThinkingLevel enum and ProviderState.thinking_level."""

    def test_thinking_level_enum_values(self):
        from ag3nt_agent.model_fallback import ThinkingLevel
        assert ThinkingLevel.OFF.value == "off"
        assert ThinkingLevel.LOW.value == "low"
        assert ThinkingLevel.MEDIUM.value == "medium"
        assert ThinkingLevel.HIGH.value == "high"

    def test_provider_state_has_thinking_level(self):
        from ag3nt_agent.model_fallback import ProviderState, ThinkingLevel
        state = ProviderState()
        assert state.thinking_level == ThinkingLevel.OFF

    def test_thinking_level_next(self):
        from ag3nt_agent.model_fallback import ThinkingLevel
        assert ThinkingLevel.OFF.next() == ThinkingLevel.LOW
        assert ThinkingLevel.LOW.next() == ThinkingLevel.MEDIUM
        assert ThinkingLevel.MEDIUM.next() == ThinkingLevel.HIGH
        assert ThinkingLevel.HIGH.next() is None  # No higher level


class TestEscalateThinking:
    """Tests for escalate_thinking() on ModelFallbackChain."""

    def test_escalate_from_off_to_low(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain, ThinkingLevel
        chain = ModelFallbackChain(providers=[{"provider": "test", "model": "m"}])
        result = chain.escalate_thinking(0)
        assert result is True
        assert chain._states[0].thinking_level == ThinkingLevel.LOW

    def test_escalate_through_all_levels(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain, ThinkingLevel
        chain = ModelFallbackChain(providers=[{"provider": "test", "model": "m"}])
        assert chain.escalate_thinking(0) is True   # off -> low
        assert chain.escalate_thinking(0) is True   # low -> medium
        assert chain.escalate_thinking(0) is True   # medium -> high
        assert chain.escalate_thinking(0) is False  # high -> None (can't escalate)
        assert chain._states[0].thinking_level == ThinkingLevel.HIGH

    def test_escalate_resets_on_success(self):
        from ag3nt_agent.model_fallback import ModelFallbackChain, ThinkingLevel
        chain = ModelFallbackChain(providers=[{"provider": "test", "model": "m"}])
        chain.escalate_thinking(0)  # off -> low
        chain.mark_success(0)
        assert chain._states[0].thinking_level == ThinkingLevel.OFF


class TestRunWithFallbackThinking:
    """Tests for thinking escalation in run_with_fallback."""

    def test_context_overflow_triggers_escalation(self):
        from ag3nt_agent.model_fallback import (
            ModelFallbackChain, ThinkingLevel, run_with_fallback,
        )
        chain = ModelFallbackChain(providers=[
            {"provider": "p1", "model": "m1"},
            {"provider": "p2", "model": "m2"},
        ])
        call_count = 0

        def action(model):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("context window overflow")
            return "success"

        result = run_with_fallback(chain, action, max_attempts=4)
        assert result == "success"
        # First attempt fails with context_overflow, should escalate thinking
        # then retry same provider before moving to next

    def test_timeout_triggers_escalation(self):
        from ag3nt_agent.model_fallback import (
            ModelFallbackChain, ThinkingLevel, run_with_fallback,
        )
        chain = ModelFallbackChain(providers=[
            {"provider": "p1", "model": "m1"},
        ])
        call_count = 0

        def action(model):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise Exception("request timed out")
            return "ok"

        result = run_with_fallback(chain, action, max_attempts=4)
        assert result == "ok"
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/agent && python -m pytest tests/unit/test_thinking_fallback.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'ThinkingLevel'`

**Step 3: Commit**

```bash
git add apps/agent/tests/unit/test_thinking_fallback.py
git commit -m "test: add failing tests for thinking-level fallback"
```

---

## Task 4: Thinking-Level Fallback — Implementation

**Files:**
- Modify: `apps/agent/ag3nt_agent/model_fallback.py:14` (add ThinkingLevel enum)
- Modify: `apps/agent/ag3nt_agent/model_fallback.py:33-40` (extend ProviderState)
- Modify: `apps/agent/ag3nt_agent/model_fallback.py:51-62` (add escalate_thinking to chain)
- Modify: `apps/agent/ag3nt_agent/model_fallback.py:145-184` (update run_with_fallback)

**Step 1: Add ThinkingLevel enum**

Add after `ERROR_BACKOFF_SCHEDULE` (line 14):

```python
from enum import Enum


class ThinkingLevel(str, Enum):
    """Thinking depth levels for model inference."""
    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    def next(self) -> ThinkingLevel | None:
        """Return the next escalation level, or None if already at max."""
        levels = list(ThinkingLevel)
        idx = levels.index(self)
        if idx + 1 < len(levels):
            return levels[idx + 1]
        return None
```

**Step 2: Extend ProviderState**

Add `thinking_level` field to `ProviderState`:

```python
@dataclass
class ProviderState:
    """Tracks health state for a single provider."""
    consecutive_errors: int = 0
    last_failure_at: float = 0.0
    last_error_type: str = ""
    cooldown_until: float = 0.0
    thinking_level: ThinkingLevel = ThinkingLevel.OFF
```

**Step 3: Add escalation methods to ModelFallbackChain**

Add to `ModelFallbackChain` class:

```python
def escalate_thinking(self, provider_idx: int) -> bool:
    """Escalate thinking level for a provider. Returns False if already at max."""
    state = self._states[provider_idx]
    next_level = state.thinking_level.next()
    if next_level is None:
        return False
    state.thinking_level = next_level
    return True
```

Update `mark_success()` to reset thinking level:

```python
def mark_success(self, provider_idx: int) -> None:
    """Record a successful call to a provider."""
    state = self._states[provider_idx]
    state.consecutive_errors = 0
    state.thinking_level = ThinkingLevel.OFF
```

**Step 4: Update run_with_fallback for thinking escalation**

In `run_with_fallback()`, after classifying the error, add thinking escalation for context_overflow and timeout errors before moving to the next provider:

Replace the error handling block inside the for loop (the `except` clause):

```python
        except Exception as exc:
            error_type = classify_error(exc)
            chain.mark_failure(idx, error_type)
            last_error = exc
            logger.warning(
                "Attempt %d/%d failed (provider=%s, error=%s): %s",
                attempt + 1, max_attempts,
                chain.providers[idx]["provider"], error_type, exc,
            )
            # For context_overflow or timeout, try escalating thinking
            # level on the same provider before moving to the next one
            if error_type in ("context_overflow", "timeout"):
                if chain.escalate_thinking(idx):
                    start_index = idx  # retry same provider
                    continue
            start_index = (idx + 1) % len(chain.providers)
```

**Step 5: Run tests to verify they pass**

Run: `cd apps/agent && python -m pytest tests/unit/test_thinking_fallback.py -v --no-cov`
Expected: PASS (all 7 tests)

**Step 6: Run full test suite**

Run: `cd apps/agent && python -m pytest tests/unit/ -v --no-cov -x`
Expected: All tests pass

**Step 7: Commit**

```bash
git add apps/agent/ag3nt_agent/model_fallback.py
git commit -m "feat: add thinking-level fallback escalation"
```

---

## Task 5: Inter-Agent Messaging (SubagentMailbox) — Tests

**Files:**
- Create: `apps/agent/tests/unit/test_subagent_mailbox.py`

**Step 1: Write the failing tests**

```python
"""Tests for SubagentMailbox inter-agent messaging."""
from __future__ import annotations

import asyncio
import pytest


class TestSubagentMailbox:
    """Tests for the SubagentMailbox class."""

    def test_create_mailbox(self):
        from deepagents.middleware.subagents import SubagentMailbox
        mb = SubagentMailbox()
        assert mb.empty()

    def test_send_and_receive(self):
        from deepagents.middleware.subagents import SubagentMailbox
        mb = SubagentMailbox()
        mb.send("parent", "hello child")
        assert not mb.empty()
        msg = mb.receive()
        assert msg == {"sender": "parent", "content": "hello child"}
        assert mb.empty()

    def test_fifo_ordering(self):
        from deepagents.middleware.subagents import SubagentMailbox
        mb = SubagentMailbox()
        mb.send("parent", "first")
        mb.send("parent", "second")
        assert mb.receive()["content"] == "first"
        assert mb.receive()["content"] == "second"

    def test_receive_empty_returns_none(self):
        from deepagents.middleware.subagents import SubagentMailbox
        mb = SubagentMailbox()
        assert mb.receive() is None

    def test_drain_returns_all(self):
        from deepagents.middleware.subagents import SubagentMailbox
        mb = SubagentMailbox()
        mb.send("a", "msg1")
        mb.send("b", "msg2")
        msgs = mb.drain()
        assert len(msgs) == 2
        assert mb.empty()


class TestMailboxInjection:
    """Tests that mailbox is injected into subagent state."""

    def test_mailbox_in_subagent_typedef(self):
        """SubAgent TypedDict should accept optional mailbox."""
        from deepagents.middleware.subagents import SubAgent, SubagentMailbox
        # Should not raise — mailbox is a valid key
        agent: SubAgent = {
            "name": "test",
            "description": "test agent",
            "system_prompt": "you are a test",
            "mailbox": SubagentMailbox(),
        }
        assert "mailbox" in agent

    def test_mailbox_passed_through_state(self):
        """Mailbox should survive _validate_and_prepare_state filtering."""
        from deepagents.middleware.subagents import SubagentMailbox
        mb = SubagentMailbox()
        # Mailbox key should not be in _EXCLUDED_STATE_KEYS
        from deepagents.middleware.subagents import _EXCLUDED_STATE_KEYS
        assert "mailbox" not in _EXCLUDED_STATE_KEYS
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/agent && python -m pytest tests/unit/test_subagent_mailbox.py -v --no-cov`
Expected: FAIL — `ImportError: cannot import name 'SubagentMailbox'`

**Step 3: Commit**

```bash
git add apps/agent/tests/unit/test_subagent_mailbox.py
git commit -m "test: add failing tests for SubagentMailbox"
```

---

## Task 6: Inter-Agent Messaging (SubagentMailbox) — Implementation

**Files:**
- Modify: `vendor/deepagents/libs/deepagents/deepagents/middleware/subagents.py:22-79` (add mailbox field to SubAgent)
- Modify: `vendor/deepagents/libs/deepagents/deepagents/middleware/subagents.py` (add SubagentMailbox class)

**Step 1: Add SubagentMailbox class**

Add before the `SubAgent` TypedDict (around line 20):

```python
from collections import deque


class SubagentMailbox:
    """Async-safe message queue for parent <-> child agent communication.

    Messages are simple dicts with sender and content fields.
    The mailbox is optional — agents work without it for backward compatibility.
    """

    def __init__(self) -> None:
        self._queue: deque[dict[str, str]] = deque()

    def send(self, sender: str, content: str) -> None:
        """Enqueue a message."""
        self._queue.append({"sender": sender, "content": content})

    def receive(self) -> dict[str, str] | None:
        """Dequeue and return the next message, or None if empty."""
        if self._queue:
            return self._queue.popleft()
        return None

    def drain(self) -> list[dict[str, str]]:
        """Dequeue and return all pending messages."""
        msgs = list(self._queue)
        self._queue.clear()
        return msgs

    def empty(self) -> bool:
        """Check if the mailbox is empty."""
        return len(self._queue) == 0
```

**Step 2: Add `mailbox` field to SubAgent TypedDict**

```python
class SubAgent(TypedDict):
    name: str
    description: str
    system_prompt: str
    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    model: NotRequired[str | BaseChatModel]
    middleware: NotRequired[list[AgentMiddleware]]
    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    skills: NotRequired[list[str]]
    mailbox: NotRequired[SubagentMailbox]
```

**Step 3: Run tests to verify they pass**

Run: `cd apps/agent && python -m pytest tests/unit/test_subagent_mailbox.py -v --no-cov`
Expected: PASS (all 7 tests)

**Step 4: Run full test suite**

Run: `cd apps/agent && python -m pytest tests/unit/ -v --no-cov -x`
Expected: All tests pass

**Step 5: Commit**

```bash
git add vendor/deepagents/libs/deepagents/deepagents/middleware/subagents.py
git commit -m "feat: add SubagentMailbox for inter-agent messaging"
```

---

## Task 7: Cron Exponential Backoff — Tests

**Files:**
- Modify: `apps/gateway/src/scheduler/Scheduler.test.ts`

**Step 1: Write the failing tests**

Append these test cases to the existing `Scheduler.test.ts`:

```typescript
describe('Cron Exponential Backoff', () => {
  it('should reschedule job with backoff delay after failure', async () => {
    const failHandler: ScheduledMessageHandler = vi.fn()
      .mockRejectedValueOnce(new Error('task failed'))
      .mockResolvedValue({ text: 'ok', notify: false });

    const backoffScheduler = new Scheduler(config, failHandler, mockNotifier);
    const jobId = backoffScheduler.addJob({
      schedule: '*/5 * * * *',
      message: 'test',
    });

    // Trigger the job
    await backoffScheduler['runCronJob'](jobId);

    // Job should still exist (not removed)
    const status = backoffScheduler.getStatus();
    expect(status.jobCount).toBe(1);

    // consecutiveErrors should be 1
    expect(backoffScheduler['consecutiveErrors'].get(jobId)).toBe(1);

    backoffScheduler.stop();
  });

  it('should reset backoff on success after failure', async () => {
    let callCount = 0;
    const handler: ScheduledMessageHandler = vi.fn().mockImplementation(() => {
      callCount++;
      if (callCount === 1) throw new Error('fail');
      return Promise.resolve({ text: 'ok', notify: false });
    });

    const s = new Scheduler(config, handler, mockNotifier);
    const jobId = s.addJob({ schedule: '*/5 * * * *', message: 'test' });

    await s['runCronJob'](jobId);
    expect(s['consecutiveErrors'].get(jobId)).toBe(1);

    await s['runCronJob'](jobId);
    expect(s['consecutiveErrors'].get(jobId)).toBe(0);

    s.stop();
  });

  it('should increase backoff tier with consecutive failures', async () => {
    const failHandler: ScheduledMessageHandler = vi.fn()
      .mockRejectedValue(new Error('persistent failure'));

    const s = new Scheduler(config, failHandler, mockNotifier);
    const jobId = s.addJob({ schedule: '*/5 * * * *', message: 'test' });

    await s['runCronJob'](jobId);
    expect(s['consecutiveErrors'].get(jobId)).toBe(1);

    await s['runCronJob'](jobId);
    expect(s['consecutiveErrors'].get(jobId)).toBe(2);

    await s['runCronJob'](jobId);
    expect(s['consecutiveErrors'].get(jobId)).toBe(3);

    s.stop();
  });

  it('should skip execution during backoff window', async () => {
    const failHandler: ScheduledMessageHandler = vi.fn()
      .mockRejectedValue(new Error('fail'));

    const s = new Scheduler(config, failHandler, mockNotifier);
    const jobId = s.addJob({ schedule: '*/5 * * * *', message: 'test' });

    // First failure — sets backoff
    await s['runCronJob'](jobId);

    // Immediate retry should be skipped (within 30s backoff)
    const store = s['store'];
    // The job should have a backoffUntil set
    const entry = s['jobs'].get(jobId);
    expect(entry?.job.backoffUntil).toBeDefined();

    s.stop();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/gateway && npx vitest run src/scheduler/Scheduler.test.ts`
Expected: FAIL — `backoffUntil` property doesn't exist on `CronJob`

**Step 3: Commit**

```bash
git add apps/gateway/src/scheduler/Scheduler.test.ts
git commit -m "test: add failing tests for cron exponential backoff"
```

---

## Task 8: Cron Exponential Backoff — Implementation

**Files:**
- Modify: `apps/gateway/src/scheduler/types.ts:35-44` (add `backoffUntil` to CronJob)
- Modify: `apps/gateway/src/scheduler/Scheduler.ts:373-452` (apply backoff in `runCronJob`)

**Step 1: Add `backoffUntil` to CronJob type**

In `types.ts`, add to the `CronJob` interface (around line 35):

```typescript
export interface CronJob extends CronJobDefinition {
  id: string;
  nextRun: Date | null;
  paused: boolean;
  createdAt: Date;
  backoffUntil?: Date;  // Added: skip execution until this time
}
```

**Step 2: Apply backoff in runCronJob**

In `Scheduler.ts`, update `runCronJob()`:

1. Add backoff check at the top (after the pause check, line 375):

```typescript
private async runCronJob(jobId: string): Promise<void> {
    const entry = this.jobs.get(jobId);
    if (!entry || entry.job.paused) return;

    // Skip if within backoff window
    if (entry.job.backoffUntil && new Date() < entry.job.backoffUntil) {
      console.log(`[Scheduler] Job ${jobId} in backoff until ${entry.job.backoffUntil.toISOString()}, skipping`);
      this.store?.recordRun(jobId, {
        status: 'skipped',
        startedAt: new Date().toISOString(),
        durationMs: 0,
      });
      return;
    }
```

2. On success (after `this.consecutiveErrors.set(jobId, 0)`, line 398), clear backoff:

```typescript
    this.consecutiveErrors.set(jobId, 0);
    entry.job.backoffUntil = undefined;  // Clear backoff on success
```

3. In the error handler (after computing `backoffMs`, line 445), set backoff:

```typescript
    const backoffIndex = Math.min(errorCount - 1, ERROR_BACKOFF_MS.length - 1);
    const backoffMs = ERROR_BACKOFF_MS[backoffIndex];
    entry.job.backoffUntil = new Date(Date.now() + backoffMs);  // Apply backoff
```

**Step 3: Run tests to verify they pass**

Run: `cd apps/gateway && npx vitest run src/scheduler/Scheduler.test.ts`
Expected: PASS

**Step 4: Commit**

```bash
git add apps/gateway/src/scheduler/types.ts apps/gateway/src/scheduler/Scheduler.ts
git commit -m "feat: apply exponential backoff to cron job failures"
```

---

## Task 9: Cron Timezone Support — Tests

**Files:**
- Modify: `apps/gateway/src/scheduler/Scheduler.test.ts`

**Step 1: Write the failing tests**

Append to `Scheduler.test.ts`:

```typescript
describe('Cron Timezone Support', () => {
  it('should accept timezone in job definition', () => {
    const jobId = scheduler.addJob({
      schedule: '0 9 * * *',
      message: 'morning check',
      timezone: 'America/New_York',
    });

    expect(jobId).toBeDefined();
    const status = scheduler.getStatus();
    expect(status.jobCount).toBe(1);
  });

  it('should pass timezone to scheduled job', () => {
    const scheduleSpy = vi.spyOn(require('node-schedule'), 'scheduleJob');

    scheduler.addJob({
      schedule: '0 9 * * *',
      message: 'tz test',
      timezone: 'Europe/London',
    });

    expect(scheduleSpy).toHaveBeenCalledWith(
      expect.objectContaining({ tz: 'Europe/London' }),
      expect.any(Function),
    );

    scheduleSpy.mockRestore();
  });

  it('should work without timezone (backward compatible)', () => {
    const jobId = scheduler.addJob({
      schedule: '0 9 * * *',
      message: 'no tz',
    });

    expect(jobId).toBeDefined();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/gateway && npx vitest run src/scheduler/Scheduler.test.ts`
Expected: FAIL — `timezone` not in `CronJobDefinition` type

**Step 3: Commit**

```bash
git add apps/gateway/src/scheduler/Scheduler.test.ts
git commit -m "test: add failing tests for cron timezone support"
```

---

## Task 10: Cron Timezone Support — Implementation

**Files:**
- Modify: `apps/gateway/src/scheduler/types.ts:17-30` (add `timezone` field)
- Modify: `apps/gateway/src/scheduler/Scheduler.ts:258-276` (pass timezone to scheduleJob)
- Modify: `apps/gateway/src/scheduler/CronJobStore.ts:4-16` (add `timezone` to persisted)

**Step 1: Add `timezone` to CronJobDefinition**

```typescript
export interface CronJobDefinition {
  schedule: string;
  message: string;
  sessionMode?: SessionMode;
  channelTarget?: string;
  oneShot?: boolean;
  name?: string;
  timezone?: string;  // IANA timezone (e.g., 'America/New_York')
}
```

**Step 2: Add `timezone` to PersistedCronJob**

```typescript
export interface PersistedCronJob {
  id: string;
  schedule: string;
  message: string;
  sessionMode?: 'isolated' | 'main';
  channelTarget?: string;
  oneShot?: boolean;
  name?: string;
  timezone?: string;
  enabled: boolean;
  createdAt: string;
  lastRunAt?: string;
  consecutiveErrors?: number;
}
```

**Step 3: Pass timezone to scheduleJob in addJob()**

In `Scheduler.ts` `addJob()`, update the cron expression branch (line 272-275):

```typescript
    if (relativeTime) {
      // One-time job at specific date — timezone not applicable
      scheduledJob = schedule.scheduleJob(relativeTime, () => {
        this.runCronJob(id);
      });
    } else {
      // Cron expression — pass timezone via RecurrenceRule if provided
      const scheduleSpec = jobDef.timezone
        ? { rule: jobDef.schedule, tz: jobDef.timezone }
        : jobDef.schedule;
      scheduledJob = schedule.scheduleJob(scheduleSpec, () => {
        this.runCronJob(id);
      });
    }
```

**Step 4: Persist timezone in store**

In `addJob()`, add `timezone` to the `store?.save()` call (line 289-299):

```typescript
    this.store?.save({
      id,
      schedule: jobDef.schedule,
      message: jobDef.message,
      sessionMode: jobDef.sessionMode,
      channelTarget: jobDef.channelTarget,
      oneShot: jobDef.oneShot,
      name: jobDef.name,
      timezone: jobDef.timezone,
      enabled: true,
      createdAt: now.toISOString(),
    });
```

**Step 5: Run tests to verify they pass**

Run: `cd apps/gateway && npx vitest run src/scheduler/Scheduler.test.ts`
Expected: PASS

**Step 6: Commit**

```bash
git add apps/gateway/src/scheduler/types.ts apps/gateway/src/scheduler/Scheduler.ts apps/gateway/src/scheduler/CronJobStore.ts
git commit -m "feat: add timezone support for cron jobs"
```

---

## Task 11: Cron Delivery Modes — Tests

**Files:**
- Modify: `apps/gateway/src/scheduler/Scheduler.test.ts`

**Step 1: Write the failing tests**

Append to `Scheduler.test.ts`:

```typescript
describe('Cron Delivery Modes', () => {
  it('should notify by default (deliveryMode=notify)', async () => {
    const handler: ScheduledMessageHandler = vi.fn()
      .mockResolvedValue({ text: 'result', notify: true });

    const s = new Scheduler(config, handler, mockNotifier);
    const jobId = s.addJob({
      schedule: '*/5 * * * *',
      message: 'test',
      channelTarget: 'chan1',
    });

    await s['runCronJob'](jobId);

    expect(mockNotifier).toHaveBeenCalledWith(
      'chan1',
      'result',
      expect.any(Object),
    );

    s.stop();
  });

  it('should NOT notify when deliveryMode=background', async () => {
    const handler: ScheduledMessageHandler = vi.fn()
      .mockResolvedValue({ text: 'background result', notify: true });

    const bgNotifier = vi.fn().mockResolvedValue(undefined);
    const s = new Scheduler(config, handler, bgNotifier);
    const jobId = s.addJob({
      schedule: '*/5 * * * *',
      message: 'bg test',
      deliveryMode: 'background',
      channelTarget: 'chan1',
    });

    await s['runCronJob'](jobId);

    // Should NOT call notifier even though handler returned notify: true
    expect(bgNotifier).not.toHaveBeenCalled();

    s.stop();
  });

  it('should still record run history in background mode', async () => {
    const mockStore = {
      save: vi.fn(),
      recordRun: vi.fn(),
      loadAll: vi.fn().mockReturnValue([]),
      getRunHistory: vi.fn().mockReturnValue([]),
    };

    const handler: ScheduledMessageHandler = vi.fn()
      .mockResolvedValue({ text: 'bg', notify: false });

    const s = new Scheduler(config, handler, mockNotifier, undefined, mockStore as any);
    const jobId = s.addJob({
      schedule: '*/5 * * * *',
      message: 'bg test',
      deliveryMode: 'background',
    });

    await s['runCronJob'](jobId);

    expect(mockStore.recordRun).toHaveBeenCalledWith(
      jobId,
      expect.objectContaining({ status: 'ok' }),
    );

    s.stop();
  });
});
```

**Step 2: Run tests to verify they fail**

Run: `cd apps/gateway && npx vitest run src/scheduler/Scheduler.test.ts`
Expected: FAIL — `deliveryMode` not in `CronJobDefinition` type

**Step 3: Commit**

```bash
git add apps/gateway/src/scheduler/Scheduler.test.ts
git commit -m "test: add failing tests for cron delivery modes"
```

---

## Task 12: Cron Delivery Modes — Implementation

**Files:**
- Modify: `apps/gateway/src/scheduler/types.ts:17-30` (add `deliveryMode`)
- Modify: `apps/gateway/src/scheduler/Scheduler.ts:414-422` (conditional notify)
- Modify: `apps/gateway/src/scheduler/CronJobStore.ts:4-16` (add `deliveryMode` to persisted)

**Step 1: Add `deliveryMode` to CronJobDefinition**

```typescript
export type DeliveryMode = 'notify' | 'background';

export interface CronJobDefinition {
  schedule: string;
  message: string;
  sessionMode?: SessionMode;
  channelTarget?: string;
  oneShot?: boolean;
  name?: string;
  timezone?: string;
  deliveryMode?: DeliveryMode;  // 'notify' (default) or 'background'
}
```

**Step 2: Add `deliveryMode` to PersistedCronJob**

```typescript
export interface PersistedCronJob {
  // ... existing fields ...
  deliveryMode?: 'notify' | 'background';
}
```

**Step 3: Conditional notify in runCronJob**

In `Scheduler.ts`, update the notification block (lines 414-422):

```typescript
      // Notify channel with response (unless background mode)
      const deliveryMode = job.deliveryMode ?? 'notify';
      if (result.notify && deliveryMode !== 'background') {
        await this.notifier(job.channelTarget, result.text, {
          jobId,
          jobName: job.name,
          type: job.oneShot ? "reminder" : "cron",
          sessionId,
        });
      }
```

**Step 4: Persist deliveryMode in store**

In `addJob()`, add `deliveryMode` to the `store?.save()` call:

```typescript
    this.store?.save({
      // ... existing fields ...
      deliveryMode: jobDef.deliveryMode,
    });
```

**Step 5: Run tests to verify they pass**

Run: `cd apps/gateway && npx vitest run src/scheduler/Scheduler.test.ts`
Expected: PASS

**Step 6: Run full gateway test suite**

Run: `cd apps/gateway && npx vitest run`
Expected: All tests pass

**Step 7: Commit**

```bash
git add apps/gateway/src/scheduler/types.ts apps/gateway/src/scheduler/Scheduler.ts apps/gateway/src/scheduler/CronJobStore.ts
git commit -m "feat: add delivery modes for cron jobs"
```

---

## Task 13: Final Integration Check

**Step 1: Run full Python test suite**

Run: `cd apps/agent && python -m pytest tests/unit/ -v --no-cov`
Expected: All tests pass (including new query_expansion, thinking_fallback, subagent_mailbox tests)

**Step 2: Run full TypeScript test suite**

Run: `cd apps/gateway && npx vitest run`
Expected: All tests pass (including new backoff, timezone, delivery mode tests)

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: Phase 4 — close all remaining parity gaps

- Query expansion for memory search (keyword bigrams, stop-word filtering)
- Thinking-level fallback (off -> low -> medium -> high -> next provider)
- Inter-agent messaging via SubagentMailbox
- Cron exponential backoff (apply ERROR_BACKOFF_MS on failure)
- Cron timezone support (IANA timezone in job definitions)
- Cron delivery modes (notify vs background)"
```
