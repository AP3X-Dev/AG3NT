/**
 * Tests for QueueModeManager.
 *
 * Covers all three queue modes (follow-up, collect, steer),
 * drop policies (new, old, summarize), debounce, and per-session isolation.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  QueueModeManager,
  type QueuedMessage,
  type QueueModeConfig,
} from './QueueModeManager.js';

function makeMsg(
  text: string,
  sessionId = 'sess-1',
  timestamp?: number,
): QueuedMessage {
  return {
    text,
    sessionId,
    channelType: 'web',
    chatId: 'chat-1',
    timestamp: timestamp ?? Date.now(),
  };
}

describe('QueueModeManager', () => {
  // ----------------------------------------------------------------
  // 1. follow-up mode queues messages in FIFO order
  // ----------------------------------------------------------------
  it('follow-up mode queues messages in FIFO order', () => {
    const mgr = new QueueModeManager({
      mode: 'follow-up',
      cap: 10,
      debounceMs: 0,
    });

    mgr.enqueue(makeMsg('first'));
    mgr.enqueue(makeMsg('second'));
    mgr.enqueue(makeMsg('third'));

    const drained = mgr.drain();
    expect(drained).toHaveLength(3);
    expect(drained.map((m) => m.text)).toEqual(['first', 'second', 'third']);
    // Queue should be empty after drain
    expect(mgr.size()).toBe(0);
  });

  // ----------------------------------------------------------------
  // 2. collect mode batches messages into a single prompt
  // ----------------------------------------------------------------
  it('collect mode batches messages into a single prompt', () => {
    const mgr = new QueueModeManager({
      mode: 'collect',
      cap: 10,
      debounceMs: 0,
    });

    mgr.enqueue(makeMsg('alpha'));
    mgr.enqueue(makeMsg('beta'));
    mgr.enqueue(makeMsg('gamma'));

    const prompt = mgr.drainCollected();
    expect(prompt).toContain('1. alpha');
    expect(prompt).toContain('2. beta');
    expect(prompt).toContain('3. gamma');
    expect(mgr.size()).toBe(0);
  });

  // ----------------------------------------------------------------
  // 3. steer mode returns the newest message and discards older
  // ----------------------------------------------------------------
  it('steer mode returns the newest message and discards older', () => {
    const mgr = new QueueModeManager({
      mode: 'steer',
      cap: 10,
      debounceMs: 0,
    });

    mgr.enqueue(makeMsg('old-1'));
    mgr.enqueue(makeMsg('old-2'));
    mgr.enqueue(makeMsg('newest'));

    const drained = mgr.drain();
    expect(drained).toHaveLength(1);
    expect(drained[0].text).toBe('newest');
    expect(mgr.size()).toBe(0);
  });

  // ----------------------------------------------------------------
  // 4. steer mode signals abort on enqueue
  // ----------------------------------------------------------------
  it('steer mode signals abort on enqueue', () => {
    const mgr = new QueueModeManager({
      mode: 'steer',
      cap: 10,
      debounceMs: 0,
    });

    expect(mgr.shouldAbortCurrent()).toBe(false);

    mgr.enqueue(makeMsg('interrupt'));
    expect(mgr.shouldAbortCurrent()).toBe(true);

    // drain clears the abort flag
    mgr.drain();
    expect(mgr.shouldAbortCurrent()).toBe(false);
  });

  // ----------------------------------------------------------------
  // 5. drop policy "new" rejects when at cap
  // ----------------------------------------------------------------
  it('drop policy "new" rejects when at cap', () => {
    const mgr = new QueueModeManager({
      mode: 'follow-up',
      cap: 2,
      debounceMs: 0,
      dropPolicy: 'new',
    });

    expect(mgr.enqueue(makeMsg('a'))).toBe(true);
    expect(mgr.enqueue(makeMsg('b'))).toBe(true);
    // At cap — new message should be rejected
    expect(mgr.enqueue(makeMsg('c'))).toBe(false);
    expect(mgr.size()).toBe(2);

    const drained = mgr.drain();
    expect(drained.map((m) => m.text)).toEqual(['a', 'b']);
  });

  // ----------------------------------------------------------------
  // 6. drop policy "old" removes oldest when at cap
  // ----------------------------------------------------------------
  it('drop policy "old" removes oldest when at cap', () => {
    const mgr = new QueueModeManager({
      mode: 'follow-up',
      cap: 2,
      debounceMs: 0,
      dropPolicy: 'old',
    });

    mgr.enqueue(makeMsg('a'));
    mgr.enqueue(makeMsg('b'));
    // At cap — oldest should be dropped
    mgr.enqueue(makeMsg('c'));

    expect(mgr.size()).toBe(2);
    const drained = mgr.drain();
    expect(drained.map((m) => m.text)).toEqual(['b', 'c']);
  });

  // ----------------------------------------------------------------
  // 7. drop policy "summarize" keeps summary of dropped messages
  // ----------------------------------------------------------------
  it('drop policy "summarize" keeps summary of dropped messages', () => {
    const mgr = new QueueModeManager({
      mode: 'collect',
      cap: 2,
      debounceMs: 0,
      dropPolicy: 'summarize',
    });

    mgr.enqueue(makeMsg('first'));
    mgr.enqueue(makeMsg('second'));
    // Exceeds cap — "first" is dropped but summarized
    mgr.enqueue(makeMsg('third'));

    const summary = mgr.getDroppedSummary();
    expect(summary).toContain('first');

    // drainCollected should include the dropped summary as prefix
    const prompt = mgr.drainCollected();
    expect(prompt).toContain('first');
    expect(prompt).toContain('second');
    expect(prompt).toContain('third');
  });

  // ----------------------------------------------------------------
  // 8. debounce delays drain readiness
  // ----------------------------------------------------------------
  it('debounce delays drain readiness', () => {
    const now = 1000;
    vi.spyOn(Date, 'now').mockReturnValue(now);

    const mgr = new QueueModeManager({
      mode: 'follow-up',
      cap: 10,
      debounceMs: 200,
    });

    mgr.enqueue(makeMsg('hello'));

    // Immediately after enqueue — not ready yet
    expect(mgr.isReady()).toBe(false);

    // Advance time past debounce window
    vi.spyOn(Date, 'now').mockReturnValue(now + 201);
    expect(mgr.isReady()).toBe(true);

    vi.restoreAllMocks();
  });

  // ----------------------------------------------------------------
  // 9. setMode changes the queue mode dynamically
  // ----------------------------------------------------------------
  it('setMode changes the queue mode dynamically', () => {
    const mgr = new QueueModeManager({
      mode: 'follow-up',
      cap: 10,
      debounceMs: 0,
    });

    mgr.enqueue(makeMsg('a'));
    mgr.enqueue(makeMsg('b'));

    // Switch to steer mode
    mgr.setMode('steer');

    // Now drain should return only newest
    const drained = mgr.drain();
    expect(drained).toHaveLength(1);
    expect(drained[0].text).toBe('b');
  });

  // ----------------------------------------------------------------
  // 10. getStats returns queue statistics
  // ----------------------------------------------------------------
  it('getStats returns queue statistics', () => {
    const mgr = new QueueModeManager({
      mode: 'follow-up',
      cap: 5,
      debounceMs: 0,
      dropPolicy: 'new',
    });

    mgr.enqueue(makeMsg('a'));
    mgr.enqueue(makeMsg('b'));

    const stats = mgr.getStats();
    expect(stats.size).toBe(2);
    expect(stats.mode).toBe('follow-up');
    expect(stats.droppedCount).toBe(0);
    expect(stats.cap).toBe(5);
  });

  // ----------------------------------------------------------------
  // 11. drainSession isolates queues by session
  // ----------------------------------------------------------------
  it('drainSession isolates queues by session', () => {
    const mgr = new QueueModeManager({
      mode: 'follow-up',
      cap: 10,
      debounceMs: 0,
    });

    mgr.enqueue(makeMsg('s1-a', 'sess-1'));
    mgr.enqueue(makeMsg('s2-a', 'sess-2'));
    mgr.enqueue(makeMsg('s1-b', 'sess-1'));
    mgr.enqueue(makeMsg('s2-b', 'sess-2'));

    // Drain only sess-1
    const s1 = mgr.drainSession('sess-1');
    expect(s1).toHaveLength(2);
    expect(s1.map((m) => m.text)).toEqual(['s1-a', 's1-b']);

    // sess-2 messages still in the queue
    expect(mgr.size()).toBe(2);
    const remaining = mgr.drain();
    expect(remaining.map((m) => m.text)).toEqual(['s2-a', 's2-b']);
  });
});
