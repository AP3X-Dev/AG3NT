/**
 * Priority message queue with rate limiting.
 *
 * Provides ordered message processing based on session priority.
 * Includes a rate limiter that enforces per-session quotas and
 * a QueueManager that orchestrates the full flow.
 */
import { randomUUID } from 'crypto';
import type { ChannelMessage, ChannelResponse } from '../channels/types.js';
import type { EnhancedSession, SessionQuotas } from '../session/SessionStore.js';
import type { RoutingDecision } from './AgentRouter.js';
import type { QueueModeManager } from './QueueModeManager.js';

// ─────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────

export interface QueueItem {
  id: string;
  message: ChannelMessage;
  session: EnhancedSession;
  priority: number;
  enqueuedAt: number;
  routingDecision: RoutingDecision;
  resolve: (response: ChannelResponse) => void;
  reject: (error: Error) => void;
}

export interface QueueStats {
  size: number;
  processing: number;
  totalEnqueued: number;
  totalProcessed: number;
  totalRejected: number;
}

// ─────────────────────────────────────────────────────────────────
// MessageQueue - Min-Heap implementation
// ─────────────────────────────────────────────────────────────────

export class MessageQueue {
  private heap: QueueItem[] = [];

  enqueue(item: QueueItem): void {
    this.heap.push(item);
    this.bubbleUp(this.heap.length - 1);
  }

  dequeue(): QueueItem | null {
    if (this.heap.length === 0) return null;
    if (this.heap.length === 1) return this.heap.pop()!;

    const top = this.heap[0];
    this.heap[0] = this.heap.pop()!;
    this.bubbleDown(0);
    return top;
  }

  peek(): QueueItem | null {
    return this.heap.length > 0 ? this.heap[0] : null;
  }

  size(): number {
    return this.heap.length;
  }

  private bubbleUp(index: number): void {
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2);
      if (this.heap[parentIndex].priority <= this.heap[index].priority) break;
      [this.heap[parentIndex], this.heap[index]] = [this.heap[index], this.heap[parentIndex]];
      index = parentIndex;
    }
  }

  private bubbleDown(index: number): void {
    const length = this.heap.length;
    while (true) {
      let smallest = index;
      const left = 2 * index + 1;
      const right = 2 * index + 2;

      if (left < length && this.heap[left].priority < this.heap[smallest].priority) {
        smallest = left;
      }
      if (right < length && this.heap[right].priority < this.heap[smallest].priority) {
        smallest = right;
      }

      if (smallest === index) break;
      [this.heap[smallest], this.heap[index]] = [this.heap[index], this.heap[smallest]];
      index = smallest;
    }
  }
}

// ─────────────────────────────────────────────────────────────────
// RateLimiter - Sliding window per session
// ─────────────────────────────────────────────────────────────────

interface SessionWindow {
  timestamps: number[];
  concurrent: number;
}

export class RateLimiter {
  private windows: Map<string, SessionWindow> = new Map();
  private lastCleanup = 0;
  private static readonly CLEANUP_INTERVAL_MS = 60_000; // 60 seconds

  /**
   * Remove stale session windows that have no active concurrent requests
   * and no timestamps within the last hour. Prevents unbounded Map growth.
   */
  private maybeCleanupStaleWindows(): void {
    const now = Date.now();
    if (now - this.lastCleanup < RateLimiter.CLEANUP_INTERVAL_MS) return;
    this.lastCleanup = now;

    const oneHourAgo = now - 3600_000;
    for (const [sessionId, window] of this.windows) {
      if (
        window.concurrent === 0 &&
        (window.timestamps.length === 0 ||
          window.timestamps[window.timestamps.length - 1] <= oneHourAgo)
      ) {
        this.windows.delete(sessionId);
      }
    }
  }

  tryAcquire(sessionId: string, quotas: SessionQuotas): boolean {
    this.maybeCleanupStaleWindows();

    const now = Date.now();
    const oneHourAgo = now - 3600_000;

    let window = this.windows.get(sessionId);
    if (!window) {
      window = { timestamps: [], concurrent: 0 };
      this.windows.set(sessionId, window);
    }

    // Prune old timestamps using binary search (timestamps are chronological)
    const ts = window.timestamps;
    if (ts.length > 0 && ts[0] <= oneHourAgo) {
      let lo = 0;
      let hi = ts.length;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (ts[mid] <= oneHourAgo) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      if (lo > 0) {
        ts.splice(0, lo);
      }
    }

    // Check turns per hour
    if (ts.length >= quotas.maxTurnsPerHour) {
      return false;
    }

    // Check concurrent
    if (window.concurrent >= quotas.maxConcurrent) {
      return false;
    }

    // Acquire
    ts.push(now);
    window.concurrent++;
    return true;
  }

  release(sessionId: string): void {
    const window = this.windows.get(sessionId);
    if (window && window.concurrent > 0) {
      window.concurrent--;
    }
  }

  getStats(sessionId: string): { turnsInLastHour: number; concurrent: number } | null {
    const window = this.windows.get(sessionId);
    if (!window) return null;

    const oneHourAgo = Date.now() - 3600_000;
    const ts = window.timestamps;
    // Binary search for cutoff instead of creating a new filtered array
    let lo = 0;
    let hi = ts.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (ts[mid] <= oneHourAgo) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return {
      turnsInLastHour: ts.length - lo,
      concurrent: window.concurrent,
    };
  }
}

// ─────────────────────────────────────────────────────────────────
// QueueManager - Orchestrator
// ─────────────────────────────────────────────────────────────────

export interface QueueManagerConfig {
  queueEnabled: boolean;
  queueIntervalMs: number;
  maxQueueSize: number;
  /** Maximum items processed concurrently (default: 3). */
  maxConcurrent?: number;
}

export class QueueManager {
  private queue: MessageQueue;
  private rateLimiter: RateLimiter;
  private config: QueueManagerConfig;
  private processHandler: ((item: QueueItem) => Promise<ChannelResponse>) | null = null;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private activeCount = 0;
  private maxConcurrent: number;
  private stats: QueueStats = {
    size: 0,
    processing: 0,
    totalEnqueued: 0,
    totalProcessed: 0,
    totalRejected: 0,
  };
  private sessionQueueModes = new Map<string, QueueModeManager>();

  constructor(config: QueueManagerConfig) {
    this.config = config;
    this.queue = new MessageQueue();
    this.rateLimiter = new RateLimiter();
    this.maxConcurrent = config.maxConcurrent ?? 3;
  }

  /**
   * Set the handler that processes dequeued items.
   */
  setProcessHandler(handler: (item: QueueItem) => Promise<ChannelResponse>): void {
    this.processHandler = handler;
  }

  /**
   * Submit a message for processing.
   * If queueing is disabled, processes immediately.
   * Returns a promise that resolves with the response.
   */
  submit(
    message: ChannelMessage,
    session: EnhancedSession,
    routingDecision: RoutingDecision,
  ): Promise<ChannelResponse> {
    // Rate limit check
    if (!this.rateLimiter.tryAcquire(session.id, session.quotas)) {
      this.stats.totalRejected++;
      return Promise.resolve({
        text: 'Rate limit exceeded. Please try again later.',
        metadata: { rateLimited: true, sessionId: session.id },
      });
    }

    // If queue is disabled, process immediately
    if (!this.config.queueEnabled) {
      return this.processItem({
        id: randomUUID(),
        message,
        session,
        priority: session.priority,
        enqueuedAt: Date.now(),
        routingDecision,
        resolve: () => {},
        reject: () => {},
      });
    }

    // Check max queue size
    if (this.queue.size() >= this.config.maxQueueSize) {
      this.rateLimiter.release(session.id);
      this.stats.totalRejected++;
      return Promise.resolve({
        text: 'The system is currently at capacity. Please try again later.',
        metadata: { queueFull: true, sessionId: session.id },
      });
    }

    // Enqueue with promise
    return new Promise<ChannelResponse>((resolve, reject) => {
      const item: QueueItem = {
        id: randomUUID(),
        message,
        session,
        priority: session.priority,
        enqueuedAt: Date.now(),
        routingDecision,
        resolve,
        reject,
      };

      this.queue.enqueue(item);
      this.stats.totalEnqueued++;
      this.stats.size = this.queue.size();

      // Trigger immediate processing instead of waiting for next interval
      this.triggerProcess();
    });
  }

  /**
   * Process the next item from the queue.
   */
  async processNext(): Promise<void> {
    const item = this.queue.dequeue();
    if (!item) return;

    this.stats.size = this.queue.size();

    try {
      const response = await this.processItem(item);
      item.resolve(response);
    } catch (err) {
      item.reject(err instanceof Error ? err : new Error(String(err)));
    }
  }

  /**
   * Trigger immediate processing of queued items (event-driven).
   * Allows up to maxConcurrent items to process simultaneously.
   * Each completed item re-triggers to pick up the next queued item.
   */
  private triggerProcess(): void {
    while (this.activeCount < this.maxConcurrent && this.queue.size() > 0) {
      this.activeCount++;
      this.processNext()
        .catch((err) => {
          console.error('[QueueManager] Processing error:', err);
        })
        .finally(() => {
          this.activeCount--;
          // Check if more items arrived during processing
          if (this.queue.size() > 0) {
            this.triggerProcess();
          }
        });
    }
  }

  /**
   * Start the processing loop.
   * Uses a fallback interval (1000ms) to catch any items missed by event-driven triggers.
   */
  start(intervalMs?: number): void {
    if (this.intervalId) return;

    // Use a longer fallback interval — primary processing is event-driven via triggerProcess()
    const interval = intervalMs ?? Math.max(this.config.queueIntervalMs, 1000);
    this.intervalId = setInterval(() => {
      if (this.queue.size() > 0) {
        this.triggerProcess();
      }
    }, interval);
  }

  /**
   * Stop the processing loop.
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  /**
   * Get queue statistics.
   */
  getStats(): QueueStats {
    return { ...this.stats, size: this.queue.size() };
  }

  getRateLimiter(): RateLimiter {
    return this.rateLimiter;
  }

  /**
   * Associate a QueueModeManager with a session for mid-run message handling.
   */
  setSessionQueueMode(sessionId: string, modeMgr: QueueModeManager): void {
    this.sessionQueueModes.set(sessionId, modeMgr);
  }

  /**
   * Remove the QueueModeManager associated with a session.
   */
  removeSessionQueueMode(sessionId: string): void {
    this.sessionQueueModes.delete(sessionId);
  }

  /**
   * Get queue mode statistics for a session, or null if no QueueModeManager is set.
   */
  getSessionQueueModeStats(sessionId: string): ReturnType<QueueModeManager['getStats']> | null {
    return this.sessionQueueModes.get(sessionId)?.getStats() ?? null;
  }

  private async processItem(item: QueueItem): Promise<ChannelResponse> {
    this.stats.processing++;

    try {
      if (!this.processHandler) {
        throw new Error('No process handler configured');
      }

      const response = await this.processHandler(item);
      this.stats.totalProcessed++;
      return response;
    } finally {
      this.stats.processing--;
      this.rateLimiter.release(item.session.id);
    }
  }
}
