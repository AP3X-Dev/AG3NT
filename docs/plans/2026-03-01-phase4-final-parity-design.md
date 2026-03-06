# Phase 4: Final Parity — Closing All Remaining Gaps

## Goal

Close the 6 remaining OpenClaw parity gaps across both the Python agent and TypeScript gateway, achieving full feature parity on the analysis matrix.

## Architecture

Two workstreams running in sequence:
1. **Python agent** (3 features): query expansion, thinking-level fallback, inter-agent messaging
2. **TypeScript gateway** (3 features): cron backoff, timezone support, delivery modes

All changes extend existing data structures and control flow — no new subsystems.

## Features

### 1. Query Expansion for Memory Search
- Add `_expand_query()` to memory search store
- Extracts keywords, generates simple synonym variants
- Runs expanded terms against embeddings, blends scores
- No external API calls — pure keyword-level expansion

### 2. Thinking-Level Fallback
- Extend `ProviderState` with `thinking_level` tracking
- Escalation: off -> low -> medium -> high -> next provider
- Triggers on context overflow or timeout errors
- Integrates with existing `run_with_fallback()`

### 3. Inter-Agent Messaging (SubagentMailbox)
- Async queue for parent <-> child communication
- Passed in subagent state during spawn
- Parent sends context updates; child streams intermediate results
- Backward-compatible — mailbox is optional

### 4. Cron Exponential Backoff
- Apply existing `ERROR_BACKOFF_MS` array (computed but unused)
- Cancel and reschedule with delay after failure
- Reset consecutive errors on success

### 5. Cron Timezone Support
- Add optional `timezone` field to job definitions
- Pass to node-schedule via `{ tz }` option
- Default to system timezone for backward compatibility

### 6. Cron Delivery Modes
- Add `deliveryMode` to job definition: 'notify' | 'background'
- notify (default): current behavior
- background: store result in history only, no notification

## Tech Stack
- Python 3.11+, LangChain/LangGraph, pytest
- TypeScript, vitest, node-schedule, Express
