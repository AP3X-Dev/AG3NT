/**
 * Queue Mode Manager — supports follow-up, collect, and steer queue modes.
 *
 * - follow-up: FIFO — queue behind current run, process in order
 * - collect:   Batch — accumulate messages, combine into single prompt after current run finishes
 * - steer:     Interrupt — abort current run, process newest message only
 *
 * Also provides debounce, drop policies (new, old, summarize),
 * and per-session isolation via drainSession().
 */

// ─────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────

export type QueueMode = 'follow-up' | 'collect' | 'steer';
export type QueueDropPolicy = 'old' | 'new' | 'summarize';

export interface QueuedMessage {
  text: string;
  sessionId: string;
  channelType: string;
  chatId: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export interface QueueModeConfig {
  mode: QueueMode;
  cap: number;
  debounceMs: number;
  dropPolicy?: QueueDropPolicy;
}

export interface QueueModeStats {
  size: number;
  mode: QueueMode;
  droppedCount: number;
  cap: number;
}

// ─────────────────────────────────────────────────────────────────
// QueueModeManager
// ─────────────────────────────────────────────────────────────────

export class QueueModeManager {
  private queue: QueuedMessage[] = [];
  private config: QueueModeConfig;
  private abortFlag = false;
  private lastEnqueueAt = 0;
  private droppedSummaries: string[] = [];
  private droppedCount = 0;

  constructor(config: QueueModeConfig) {
    this.config = { ...config };
  }

  /**
   * Enqueue a message. Returns false if the message was dropped (drop policy "new").
   */
  enqueue(msg: QueuedMessage): boolean {
    // Handle cap overflow with drop policy
    if (this.queue.length >= this.config.cap) {
      const policy = this.config.dropPolicy ?? 'new';

      switch (policy) {
        case 'new':
          // Reject the incoming message
          this.droppedCount++;
          return false;

        case 'old': {
          // Drop oldest item
          this.queue.shift();
          this.droppedCount++;
          break;
        }

        case 'summarize': {
          // Drop oldest item but keep its text as a summary
          const dropped = this.queue.shift();
          if (dropped) {
            this.droppedSummaries.push(dropped.text);
            this.droppedCount++;
          }
          break;
        }
      }
    }

    this.queue.push(msg);
    this.lastEnqueueAt = Date.now();

    // In steer mode, signal that current processing should abort
    if (this.config.mode === 'steer') {
      this.abortFlag = true;
    }

    return true;
  }

  /**
   * Drain the queue based on current mode.
   *
   * - follow-up: Returns all items in FIFO order
   * - steer: Returns only the newest (last) item, discards rest
   */
  drain(): QueuedMessage[] {
    if (this.queue.length === 0) return [];

    let result: QueuedMessage[];

    switch (this.config.mode) {
      case 'steer': {
        // Return only the newest message
        result = [this.queue[this.queue.length - 1]];
        break;
      }

      case 'follow-up':
      case 'collect':
      default: {
        // Return all items in FIFO order
        result = [...this.queue];
        break;
      }
    }

    this.queue = [];
    this.abortFlag = false;
    return result;
  }

  /**
   * Drain only messages belonging to a specific session.
   * Other sessions' messages remain in the queue.
   */
  drainSession(sessionId: string): QueuedMessage[] {
    const matching: QueuedMessage[] = [];
    const remaining: QueuedMessage[] = [];

    for (const msg of this.queue) {
      if (msg.sessionId === sessionId) {
        matching.push(msg);
      } else {
        remaining.push(msg);
      }
    }

    this.queue = remaining;
    return matching;
  }

  /**
   * Drain all messages in collect mode, formatting them as a numbered list.
   * Includes dropped summary as prefix if any messages were summarized.
   */
  drainCollected(): string {
    const parts: string[] = [];

    // Include dropped summaries as a prefix
    if (this.droppedSummaries.length > 0) {
      parts.push(
        `[Dropped messages summary: ${this.droppedSummaries.join('; ')}]`,
      );
    }

    // Format remaining messages as a numbered list
    this.queue.forEach((msg, i) => {
      parts.push(`${i + 1}. ${msg.text}`);
    });

    this.queue = [];
    this.droppedSummaries = [];
    return parts.join('\n');
  }

  /**
   * Returns true if the current processing should be aborted (steer mode).
   */
  shouldAbortCurrent(): boolean {
    return this.abortFlag;
  }

  /**
   * Returns true if the debounce window has elapsed since the last enqueue.
   * Always true when debounceMs is 0 and queue is non-empty.
   */
  isReady(): boolean {
    if (this.queue.length === 0) return false;
    if (this.config.debounceMs === 0) return true;
    return Date.now() - this.lastEnqueueAt >= this.config.debounceMs;
  }

  /**
   * Current number of messages in the queue.
   */
  size(): number {
    return this.queue.length;
  }

  /**
   * Change the queue mode dynamically.
   */
  setMode(mode: QueueMode): void {
    this.config.mode = mode;
  }

  /**
   * Get a human-readable summary of dropped messages (from "summarize" policy).
   */
  getDroppedSummary(): string {
    return this.droppedSummaries.join('; ');
  }

  /**
   * Get queue statistics.
   */
  getStats(): QueueModeStats {
    return {
      size: this.queue.length,
      mode: this.config.mode,
      droppedCount: this.droppedCount,
      cap: this.config.cap,
    };
  }
}
