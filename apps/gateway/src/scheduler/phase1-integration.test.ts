// apps/gateway/src/scheduler/phase1-integration.test.ts
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { CronJobStore } from './CronJobStore.js';
import { HeartbeatRunner } from './HeartbeatRunner.js';
import { SessionRecovery } from './SessionRecovery.js';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('Phase 1 Integration', () => {
  let tmpDir: string;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-02-28T10:00:00Z'));
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'phase1-test-'));
  });

  afterEach(() => {
    vi.useRealTimers();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should persist cron jobs, run heartbeat, and recover state', async () => {
    // 1. Create and persist a cron job
    const cronStore = new CronJobStore(path.join(tmpDir, 'cron'));
    cronStore.save({
      id: 'job-morning',
      schedule: '0 7 * * *',
      message: 'Morning briefing',
      enabled: true,
      createdAt: new Date().toISOString(),
    });

    // 2. Run heartbeat with checklist
    fs.writeFileSync(path.join(tmpDir, 'HEARTBEAT.md'), '- [ ] Check inbox\n- [ ] Review PRs\n');
    const notifications: string[] = [];
    const heartbeat = new HeartbeatRunner({
      intervalMs: 30 * 60 * 1000,
      workspacePath: tmpDir,
      activeHours: { start: '08:00', end: '22:00', timezone: 'UTC' },
      agentHandler: async (_prompt, _sessionId) => ({
        content: 'You have 2 unread PRs that need review.',
      }),
      notifier: async (msg) => { notifications.push(msg); },
    });

    await heartbeat.runOnce();
    expect(notifications).toHaveLength(1);
    expect(notifications[0]).toContain('2 unread PRs');

    // 3. Save and restore gateway state
    const recovery = new SessionRecovery(tmpDir);
    recovery.saveState({
      lastHeartbeat: new Date().toISOString(),
      activeSessions: ['session-1'],
      schedulerRunning: true,
    });

    // Simulate restart
    const recovery2 = new SessionRecovery(tmpDir);
    const state = recovery2.loadState();
    expect(state).not.toBeNull();
    expect(state!.schedulerRunning).toBe(true);

    // Cron jobs survive restart
    const cronStore2 = new CronJobStore(path.join(tmpDir, 'cron'));
    const jobs = cronStore2.loadAll();
    expect(jobs).toHaveLength(1);
    expect(jobs[0].id).toBe('job-morning');

    heartbeat.stop();
  });

  it('should track cron job run history across sessions', () => {
    const cronDir = path.join(tmpDir, 'cron');
    const store1 = new CronJobStore(cronDir);

    // First session: add job and record runs
    store1.save({
      id: 'job-1',
      schedule: '0 * * * *',
      message: 'hourly check',
      enabled: true,
      createdAt: new Date().toISOString(),
    });
    store1.recordRun('job-1', { status: 'ok', startedAt: new Date().toISOString(), durationMs: 500 });
    store1.recordRun('job-1', { status: 'error', startedAt: new Date().toISOString(), durationMs: 100, error: 'timeout' });

    // Second session: history persists
    const store2 = new CronJobStore(cronDir);
    const history = store2.getRunHistory('job-1');
    expect(history).toHaveLength(2);
    expect(history[0].status).toBe('ok');
    expect(history[1].status).toBe('error');
    expect(history[1].error).toBe('timeout');
  });

  it('should suppress heartbeat when agent says HEARTBEAT_OK', async () => {
    fs.writeFileSync(path.join(tmpDir, 'HEARTBEAT.md'), '- [ ] Quick status check\n');
    const notifications: string[] = [];
    const heartbeat = new HeartbeatRunner({
      intervalMs: 30 * 60 * 1000,
      workspacePath: tmpDir,
      activeHours: { start: '08:00', end: '22:00', timezone: 'UTC' },
      agentHandler: async (_prompt, _sessionId) => ({ content: 'HEARTBEAT_OK' }),
      notifier: async (msg) => { notifications.push(msg); },
    });

    await heartbeat.runOnce();
    expect(notifications).toHaveLength(0); // Suppressed

    heartbeat.stop();
  });
});
