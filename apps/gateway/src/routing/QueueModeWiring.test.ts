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
    const session = makeSession('session-1');
    const routing = makeRouting();
    await qm.submit(
      makeMessage('hello'),
      session as any,
      routing,
    );
    expect(processHandler).toHaveBeenCalled();
  });

  it('should route mid-run messages through QueueModeManager when active', async () => {
    const session = makeSession('session-1');
    const routing = makeRouting();
    const modeMgr = new QueueModeManager({ mode: 'collect', cap: 10, debounceMs: 0 });
    qm.setSessionQueueMode('session-1', modeMgr);

    // Submit while QueueModeManager is active — should buffer, not process
    const result = await qm.submit(makeMessage('mid-run msg'), session as any, routing);
    expect(result.metadata?.queued).toBe(true);
    expect(result.metadata?.mode).toBe('collect');
    expect(processHandler).not.toHaveBeenCalled();
    expect(modeMgr.size()).toBe(1);
  });

  it('should return drop message when QueueModeManager cap is exceeded', async () => {
    const session = makeSession('session-1');
    const routing = makeRouting();
    const modeMgr = new QueueModeManager({ mode: 'collect', cap: 1, debounceMs: 0, dropPolicy: 'new' });
    qm.setSessionQueueMode('session-1', modeMgr);

    await qm.submit(makeMessage('first'), session as any, routing);
    const result = await qm.submit(makeMessage('second'), session as any, routing);
    expect(result.metadata?.queueModeDrop).toBe(true);
  });

  it('should process collected messages after primary turn completes', async () => {
    const session = makeSession('session-1');
    const routing = makeRouting();
    const modeMgr = new QueueModeManager({ mode: 'follow-up', cap: 10, debounceMs: 0 });

    // Simulate: processHandler sets QueueModeManager and a mid-run message arrives
    let callCount = 0;
    processHandler.mockImplementation(async () => {
      callCount++;
      if (callCount === 1) {
        // During first turn, set queue mode and simulate a mid-run message
        qm.setSessionQueueMode('session-1', modeMgr);
        modeMgr.enqueue({
          text: 'follow-up msg',
          sessionId: 'session-1',
          channelType: 'web',
          chatId: 'chat-1',
          timestamp: Date.now(),
        });
      }
      return { text: `response-${callCount}`, metadata: {} };
    });

    qm.start();
    await qm.submit(makeMessage('primary msg'), session as any, routing);
    // Wait for async follow-up processing
    await new Promise(r => setTimeout(r, 100));

    // processHandler called for primary turn + follow-up drain
    expect(processHandler).toHaveBeenCalledTimes(2);
  });
});

function makeSession(id: string) {
  return {
    id,
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
}

function makeMessage(text: string) {
  return { id: 'msg-1', text, channelType: 'web', channelId: 'ch-1', chatId: 'chat-1', userId: 'u1', timestamp: new Date() } as any;
}

function makeRouting() {
  return { agentName: 'default', workerUrl: 'http://localhost:8000', reason: 'default', priority: 1 };
}
