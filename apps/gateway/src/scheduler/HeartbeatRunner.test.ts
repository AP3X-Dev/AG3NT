// apps/gateway/src/scheduler/HeartbeatRunner.test.ts
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { HeartbeatRunner } from './HeartbeatRunner.js';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('HeartbeatRunner', () => {
  let runner: HeartbeatRunner;
  let tmpDir: string;
  let mockAgentHandler: ReturnType<typeof vi.fn>;
  let mockNotifier: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-02-28T10:00:00Z')); // 10 AM UTC
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'hb-test-'));
    mockAgentHandler = vi.fn().mockResolvedValue({ content: 'HEARTBEAT_OK' });
    mockNotifier = vi.fn().mockResolvedValue(undefined);
    runner = new HeartbeatRunner({
      intervalMs: 30 * 60 * 1000,
      workspacePath: tmpDir,
      activeHours: { start: '08:00', end: '22:00', timezone: 'UTC' },
      agentHandler: mockAgentHandler,
      notifier: mockNotifier,
    });
  });

  afterEach(() => {
    runner.stop();
    vi.useRealTimers();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should read HEARTBEAT.md and pass contents to agent', async () => {
    fs.writeFileSync(path.join(tmpDir, 'HEARTBEAT.md'), '- [ ] Check inbox\n- [ ] Review calendar\n');
    await runner.runOnce();
    expect(mockAgentHandler).toHaveBeenCalledTimes(1);
    const call = mockAgentHandler.mock.calls[0];
    expect(call[0]).toContain('Check inbox');
    expect(call[0]).toContain('Review calendar');
  });

  it('should suppress HEARTBEAT_OK responses (not notify user)', async () => {
    fs.writeFileSync(path.join(tmpDir, 'HEARTBEAT.md'), '- [ ] Quick check\n');
    mockAgentHandler.mockResolvedValue({ content: 'HEARTBEAT_OK' });
    await runner.runOnce();
    expect(mockNotifier).not.toHaveBeenCalled();
  });

  it('should notify user when agent has real content', async () => {
    fs.writeFileSync(path.join(tmpDir, 'HEARTBEAT.md'), '- [ ] Check inbox\n');
    mockAgentHandler.mockResolvedValue({ content: 'You have 3 urgent emails that need attention.' });
    await runner.runOnce();
    expect(mockNotifier).toHaveBeenCalledTimes(1);
    expect(mockNotifier.mock.calls[0][0]).toContain('3 urgent emails');
  });

  it('should skip heartbeat outside active hours', async () => {
    vi.setSystemTime(new Date('2026-02-28T23:00:00Z')); // 11 PM UTC
    await runner.runOnce();
    expect(mockAgentHandler).not.toHaveBeenCalled();
  });

  it('should handle missing HEARTBEAT.md gracefully', async () => {
    await runner.runOnce();
    expect(mockAgentHandler).toHaveBeenCalledTimes(1);
  });

  it('should skip when HEARTBEAT.md is effectively empty', async () => {
    fs.writeFileSync(path.join(tmpDir, 'HEARTBEAT.md'), '# Heartbeat Checklist\n\n<!-- nothing here -->\n');
    await runner.runOnce();
    expect(mockAgentHandler).toHaveBeenCalledTimes(1);
  });

  it('should strip HEARTBEAT_OK from mixed responses', async () => {
    fs.writeFileSync(path.join(tmpDir, 'HEARTBEAT.md'), '- [ ] Check status\n');
    mockAgentHandler.mockResolvedValue({
      content: 'Your server is healthy. All services running.\nHEARTBEAT_OK'
    });
    await runner.runOnce();
    expect(mockNotifier).toHaveBeenCalledTimes(1);
    const notified = mockNotifier.mock.calls[0][0];
    expect(notified).not.toContain('HEARTBEAT_OK');
    expect(notified).toContain('server is healthy');
  });

  it('should run periodically when started', async () => {
    fs.writeFileSync(path.join(tmpDir, 'HEARTBEAT.md'), '- [ ] Check\n');
    runner.start();
    await vi.advanceTimersByTimeAsync(30 * 60 * 1000);
    expect(mockAgentHandler).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(30 * 60 * 1000);
    expect(mockAgentHandler).toHaveBeenCalledTimes(2);
  });
});
