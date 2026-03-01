import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { QueueManager } from './MessageQueue.js';
import { QueueModeManager } from './QueueModeManager.js';

describe('QueueManager with QueueModeManager', () => {
  let qm: QueueManager;
  let processHandler: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    processHandler = vi.fn().mockResolvedValue({ text: 'ok', metadata: {} });
    qm = new QueueManager({
      queueEnabled: true,
      queueIntervalMs: 100,
      maxQueueSize: 50,
      maxConcurrent: 1,
    });
    qm.setProcessHandler(processHandler);
  });

  afterEach(() => {
    qm.stop();
  });

  it('should accept a QueueModeManager per session', () => {
    const modeMgr = new QueueModeManager({ mode: 'follow-up', cap: 10, debounceMs: 0 });
    qm.setSessionQueueMode('session-1', modeMgr);
    const stats = qm.getSessionQueueModeStats('session-1');
    expect(stats?.mode).toBe('follow-up');
  });

  it('should return null stats for unknown session', () => {
    expect(qm.getSessionQueueModeStats('nonexistent')).toBeNull();
  });

  it('should remove a session queue mode', () => {
    const modeMgr = new QueueModeManager({ mode: 'steer', cap: 5, debounceMs: 0 });
    qm.setSessionQueueMode('s1', modeMgr);
    qm.removeSessionQueueMode('s1');
    expect(qm.getSessionQueueModeStats('s1')).toBeNull();
  });

  it('should work without QueueModeManager (backward compatible)', async () => {
    const session = {
      id: 'session-1',
      channelType: 'web',
      channelId: 'ch-1',
      chatId: 'chat-1',
      priority: 1,
      assignedAgent: null,
      directives: [],
      quotas: { maxTurnsPerHour: 100, maxConcurrent: 5 },
      activationMode: 'always' as const,
      activationKeywords: [],
    };
    const routing = {
      agentName: 'default',
      workerUrl: 'http://localhost:8000',
      reason: 'default',
      priority: 1,
    };
    const result = await qm.submit(
      { text: 'hello', session_id: 'session-1', metadata: {} },
      session as any,
      routing,
    );
    expect(processHandler).toHaveBeenCalled();
  });
});
