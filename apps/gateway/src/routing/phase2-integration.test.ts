/**
 * Phase 2 Integration Tests
 *
 * Verifies that all Phase 2 components work together:
 * - Queue modes with drop policies and channel delivery
 * - Push token persistence with payload building
 * - Steer mode abort behavior
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { QueueModeManager } from './QueueModeManager.js';
import { PushNotificationService } from '../push/PushNotificationService.js';
import { ChannelDeliveryService } from '../channels/ChannelDeliveryService.js';
import type { IChannelAdapter } from '../channels/types.js';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('Phase 2 Integration', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'phase2-'));
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should handle collect mode with drop policy and delivery', async () => {
    // Setup queue with collect mode, cap 3, summarize drops
    const queue = new QueueModeManager({ mode: 'collect', cap: 3, debounceMs: 0, dropPolicy: 'summarize' });

    // Simulate 5 messages arriving during active run
    for (let i = 1; i <= 5; i++) {
      queue.enqueue({
        text: `message ${i}`,
        sessionId: 'session-1',
        channelType: 'telegram',
        chatId: 'chat-1',
        timestamp: Date.now() + i,
      });
    }

    // 2 messages dropped (messages 1 and 2), 3 remain
    expect(queue.size()).toBe(3);
    expect(queue.getDroppedSummary()).toContain('message 1');
    expect(queue.getDroppedSummary()).toContain('message 2');

    // Drain collected into single prompt
    const batched = queue.drainCollected();
    // Remaining messages are present as numbered items
    expect(batched).toContain('message 3');
    expect(batched).toContain('message 4');
    expect(batched).toContain('message 5');
    // Dropped messages appear in the summary prefix
    expect(batched).toContain('Dropped messages summary');
    expect(batched).toContain('message 1');
    expect(batched).toContain('message 2');

    // Setup channel delivery
    const delivery = new ChannelDeliveryService();
    const adapter: IChannelAdapter = {
      type: 'telegram',
      id: 'tg',
      connect: vi.fn().mockResolvedValue(undefined),
      disconnect: vi.fn().mockResolvedValue(undefined),
      isConnected: vi.fn().mockReturnValue(true),
      send: vi.fn().mockResolvedValue(undefined),
      onMessage: vi.fn(),
    };
    // Track incoming message to register default chatId for adapter 'tg'
    delivery.trackIncomingMessage('tg', 'chat-1');

    // Deliver result via channel — finds adapter by type 'telegram', then
    // looks up chatId by adapter.id 'tg'
    const delivered = await delivery.deliverToType([adapter], 'telegram', 'Processed all queued messages');
    expect(delivered).toBe(true);
    expect(adapter.send).toHaveBeenCalledWith('chat-1', { text: 'Processed all queued messages' });
  });

  it('should persist push tokens and deliver via channel adapter', async () => {
    // Register push token
    const push = new PushNotificationService(tmpDir, { teamId: 'T', keyId: 'K', privateKey: 'PK' });
    await push.registerToken({ nodeId: 'iphone-1', token: 'device-token-abc', topic: 'com.ag3nt', environment: 'sandbox' });

    // Verify persistence — new instance reads from disk
    const push2 = new PushNotificationService(tmpDir, { teamId: 'T', keyId: 'K', privateKey: 'PK' });
    const tokens = push2.getTokensForNode('iphone-1');
    expect(tokens).toHaveLength(1);
    expect(tokens[0].token).toBe('device-token-abc');
    expect(tokens[0].topic).toBe('com.ag3nt');
    expect(tokens[0].environment).toBe('sandbox');

    // Build alert payload
    const alert = push2.buildAlertPayload({ title: 'AG3NT', body: 'Task done', nodeId: 'iphone-1' });
    expect(alert.aps.alert.title).toBe('AG3NT');
    expect(alert.aps.alert.body).toBe('Task done');
    expect(alert.aps.sound).toBe('default');
    expect(alert.ag3nt.kind).toBe('push.alert');
    expect(alert.ag3nt.nodeId).toBe('iphone-1');

    // Build wake payload
    const wake = push2.buildWakePayload({ nodeId: 'iphone-1', reason: 'cron.completed' });
    expect(wake.aps['content-available']).toBe(1);
    expect(wake.ag3nt.reason).toBe('cron.completed');
    expect(wake.ag3nt.kind).toBe('node.wake');
    expect(wake.ag3nt.nodeId).toBe('iphone-1');
  });

  it('should steer mode abort and process only newest message', () => {
    const queue = new QueueModeManager({ mode: 'steer', cap: 10, debounceMs: 0 });

    // Multiple messages arrive during active run
    queue.enqueue({ text: 'old-1', sessionId: 's1', channelType: 'cli', chatId: 'c1', timestamp: 1 });
    queue.enqueue({ text: 'old-2', sessionId: 's1', channelType: 'cli', chatId: 'c1', timestamp: 2 });
    queue.enqueue({ text: 'latest', sessionId: 's1', channelType: 'cli', chatId: 'c1', timestamp: 3 });

    // Should signal abort
    expect(queue.shouldAbortCurrent()).toBe(true);

    // Drain returns only newest
    const items = queue.drain();
    expect(items).toHaveLength(1);
    expect(items[0].text).toBe('latest');

    // Abort flag cleared after drain
    expect(queue.shouldAbortCurrent()).toBe(false);
  });
});
