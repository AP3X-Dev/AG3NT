import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { CronJobStore } from './CronJobStore.js';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('CronJobStore', () => {
  let store: CronJobStore;
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cron-test-'));
    store = new CronJobStore(tmpDir);
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should save and load a job', () => {
    const job = {
      id: 'job-1',
      schedule: '0 9 * * *',
      message: 'Good morning',
      sessionMode: 'isolated' as const,
      channelTarget: 'telegram',
      oneShot: false,
      name: 'morning-briefing',
      enabled: true,
      createdAt: new Date().toISOString(),
    };
    store.save(job);
    const loaded = store.loadAll();
    expect(loaded).toHaveLength(1);
    expect(loaded[0].id).toBe('job-1');
    expect(loaded[0].schedule).toBe('0 9 * * *');
    expect(loaded[0].message).toBe('Good morning');
  });

  it('should delete a job', () => {
    store.save({ id: 'job-1', schedule: '* * * * *', message: 'test', enabled: true, createdAt: new Date().toISOString() });
    store.save({ id: 'job-2', schedule: '* * * * *', message: 'test2', enabled: true, createdAt: new Date().toISOString() });
    store.delete('job-1');
    const loaded = store.loadAll();
    expect(loaded).toHaveLength(1);
    expect(loaded[0].id).toBe('job-2');
  });

  it('should update a job (save with same id)', () => {
    store.save({ id: 'job-1', schedule: '* * * * *', message: 'old', enabled: true, createdAt: new Date().toISOString() });
    store.save({ id: 'job-1', schedule: '0 7 * * *', message: 'updated', enabled: true, createdAt: new Date().toISOString() });
    const loaded = store.loadAll();
    expect(loaded).toHaveLength(1);
    expect(loaded[0].message).toBe('updated');
  });

  it('should return empty array when no file exists', () => {
    const loaded = store.loadAll();
    expect(loaded).toEqual([]);
  });

  it('should track run history', () => {
    store.recordRun('job-1', { status: 'ok', startedAt: new Date().toISOString(), durationMs: 1500 });
    store.recordRun('job-1', { status: 'error', startedAt: new Date().toISOString(), durationMs: 300, error: 'timeout' });
    const history = store.getRunHistory('job-1');
    expect(history).toHaveLength(2);
    expect(history[0].status).toBe('ok');
    expect(history[1].status).toBe('error');
  });

  it('should prune run history beyond max entries', () => {
    for (let i = 0; i < 150; i++) {
      store.recordRun('job-1', { status: 'ok', startedAt: new Date().toISOString(), durationMs: 100 });
    }
    const history = store.getRunHistory('job-1');
    expect(history.length).toBeLessThanOrEqual(100);
  });

  it('should update job pause state', () => {
    store.save({ id: 'job-1', schedule: '* * * * *', message: 'test', enabled: true, createdAt: new Date().toISOString() });
    store.updateEnabled('job-1', false);
    const loaded = store.loadAll();
    expect(loaded[0].enabled).toBe(false);
  });
});
