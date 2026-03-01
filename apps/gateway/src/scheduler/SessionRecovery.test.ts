// apps/gateway/src/scheduler/SessionRecovery.test.ts
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SessionRecovery } from './SessionRecovery.js';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('SessionRecovery', () => {
  let recovery: SessionRecovery;
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'recovery-test-'));
    recovery = new SessionRecovery(tmpDir);
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should save and restore gateway state', () => {
    const state = {
      lastHeartbeat: new Date().toISOString(),
      activeSessions: ['session-1', 'session-2'],
      schedulerRunning: true,
    };
    recovery.saveState(state);
    const restored = recovery.loadState();
    expect(restored).not.toBeNull();
    expect(restored!.activeSessions).toEqual(['session-1', 'session-2']);
    expect(restored!.schedulerRunning).toBe(true);
  });

  it('should return null when no state file exists', () => {
    const restored = recovery.loadState();
    expect(restored).toBeNull();
  });

  it('should handle corrupted state file gracefully', () => {
    fs.writeFileSync(path.join(tmpDir, 'gateway-state.json'), 'not json{{{');
    const restored = recovery.loadState();
    expect(restored).toBeNull();
  });

  it('should save state periodically via checkpoint', () => {
    recovery.checkpoint({
      lastHeartbeat: new Date().toISOString(),
      activeSessions: [],
      schedulerRunning: false,
    });
    const restored = recovery.loadState();
    expect(restored).not.toBeNull();
  });

  it('should clear state on explicit reset', () => {
    recovery.saveState({
      lastHeartbeat: new Date().toISOString(),
      activeSessions: [],
      schedulerRunning: true,
    });
    recovery.clearState();
    const restored = recovery.loadState();
    expect(restored).toBeNull();
  });
});
