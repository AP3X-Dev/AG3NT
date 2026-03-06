/**
 * Tests for Scheduler persistence wiring with CronJobStore.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { Scheduler } from './Scheduler.js';
import { CronJobStore } from './CronJobStore.js';
import type { SchedulerConfig, ScheduledMessageHandler, ChannelNotifier } from './types.js';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('Scheduler Persistence', () => {
  let scheduler: Scheduler;
  let store: CronJobStore;
  let tmpDir: string;
  let mockHandler: ScheduledMessageHandler;
  let mockNotifier: ChannelNotifier;
  let mockEventHandler: ReturnType<typeof vi.fn>;
  let config: SchedulerConfig;

  beforeEach(() => {
    vi.useFakeTimers();
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sched-persist-test-'));
    store = new CronJobStore(tmpDir);

    mockHandler = vi.fn().mockResolvedValue({ text: 'ok', notify: false });
    mockNotifier = vi.fn().mockResolvedValue(undefined);
    mockEventHandler = vi.fn();

    config = {
      heartbeat: { intervalMinutes: 0 },
      cronJobs: [],
    };

    scheduler = new Scheduler(config, mockHandler, mockNotifier, mockEventHandler, store);
  });

  afterEach(() => {
    scheduler.stop();
    vi.useRealTimers();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should persist new jobs to store', () => {
    const jobId = scheduler.addJob({
      schedule: '0 9 * * *',
      message: 'Daily briefing',
      name: 'Morning Check',
    });

    const persisted = store.loadAll();
    expect(persisted).toHaveLength(1);
    expect(persisted[0]).toMatchObject({
      id: jobId,
      schedule: '0 9 * * *',
      message: 'Daily briefing',
      name: 'Morning Check',
      enabled: true,
    });
  });

  it('should remove persisted job on removeJob', () => {
    const jobId = scheduler.addJob({
      schedule: '0 9 * * *',
      message: 'Daily briefing',
    });

    expect(store.loadAll()).toHaveLength(1);

    scheduler.removeJob(jobId);

    expect(store.loadAll()).toHaveLength(0);
  });

  it('should restore jobs from store on start', () => {
    // Pre-populate the store with a persisted job
    store.save({
      id: 'job-42',
      schedule: '0 9 * * *',
      message: 'Restored job',
      name: 'Persisted Morning',
      enabled: true,
      createdAt: new Date().toISOString(),
    });

    // Create a new scheduler with the pre-populated store
    const scheduler2 = new Scheduler(config, mockHandler, mockNotifier, mockEventHandler, store);
    scheduler2.start();

    const jobs = scheduler2.listJobs();
    expect(jobs.length).toBeGreaterThanOrEqual(1);

    const restored = jobs.find(j => j.id === 'job-42');
    expect(restored).toBeDefined();
    expect(restored!.message).toBe('Restored job');
    expect(restored!.name).toBe('Persisted Morning');

    scheduler2.stop();
  });

  it('should not restore one-shot jobs from store', () => {
    // Pre-populate with a one-shot job
    store.save({
      id: 'job-99',
      schedule: '0 9 * * *',
      message: 'One-shot task',
      oneShot: true,
      enabled: true,
      createdAt: new Date().toISOString(),
    });

    const scheduler2 = new Scheduler(config, mockHandler, mockNotifier, mockEventHandler, store);
    scheduler2.start();

    const jobs = scheduler2.listJobs();
    const oneShot = jobs.find(j => j.id === 'job-99');
    expect(oneShot).toBeUndefined();

    scheduler2.stop();
  });

  it('should record run history on successful job completion', async () => {
    const jobId = scheduler.addJob({
      schedule: '* * * * *',
      message: 'Every minute',
      name: 'Frequent job',
    });

    scheduler.start();

    // Trigger the cron by advancing time by 1 minute
    await vi.advanceTimersByTimeAsync(60 * 1000);

    // Allow the async handler to complete
    await vi.advanceTimersByTimeAsync(10);

    const history = store.getRunHistory(jobId);
    expect(history.length).toBeGreaterThanOrEqual(1);
    expect(history[0].status).toBe('ok');
    expect(history[0].durationMs).toBeGreaterThanOrEqual(0);
  });

  it('should record run history on job failure', async () => {
    (mockHandler as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('Agent timeout'));

    const jobId = scheduler.addJob({
      schedule: '* * * * *',
      message: 'Failing job',
      name: 'Bad job',
    });

    scheduler.start();

    // Trigger the cron by advancing time
    await vi.advanceTimersByTimeAsync(60 * 1000);
    await vi.advanceTimersByTimeAsync(10);

    const history = store.getRunHistory(jobId);
    expect(history.length).toBeGreaterThanOrEqual(1);
    expect(history[0].status).toBe('error');
    expect(history[0].error).toBe('Agent timeout');
  });

  it('should persist enabled=false when job is paused', () => {
    const jobId = scheduler.addJob({
      schedule: '0 9 * * *',
      message: 'Pausable job',
    });

    scheduler.pauseJob(jobId);

    const persisted = store.loadAll();
    expect(persisted[0].enabled).toBe(false);
  });

  it('should persist enabled=true when job is resumed', () => {
    const jobId = scheduler.addJob({
      schedule: '0 9 * * *',
      message: 'Resumable job',
    });

    scheduler.pauseJob(jobId);
    expect(store.loadAll()[0].enabled).toBe(false);

    scheduler.resumeJob(jobId);
    expect(store.loadAll()[0].enabled).toBe(true);
  });

  it('should restore paused jobs in paused state', () => {
    // Pre-populate with a paused job
    store.save({
      id: 'job-50',
      schedule: '0 9 * * *',
      message: 'Paused job',
      enabled: false,
      createdAt: new Date().toISOString(),
    });

    const scheduler2 = new Scheduler(config, mockHandler, mockNotifier, mockEventHandler, store);
    scheduler2.start();

    const jobs = scheduler2.listJobs();
    const restored = jobs.find(j => j.id === 'job-50');
    expect(restored).toBeDefined();
    expect(restored!.paused).toBe(true);

    scheduler2.stop();
  });

  it('should avoid job ID collisions with restored jobs', () => {
    // Pre-populate with a job that has a high counter
    store.save({
      id: 'job-100',
      schedule: '0 9 * * *',
      message: 'High counter job',
      enabled: true,
      createdAt: new Date().toISOString(),
    });

    const scheduler2 = new Scheduler(config, mockHandler, mockNotifier, mockEventHandler, store);
    scheduler2.start();

    // Add a new job — its ID should be above 100
    const newId = scheduler2.addJob({
      schedule: '0 17 * * *',
      message: 'New job after restore',
    });

    expect(newId).toBe('job-101');

    scheduler2.stop();
  });

  it('should work without a store (backward compatible)', () => {
    const schedulerNoStore = new Scheduler(config, mockHandler, mockNotifier, mockEventHandler);

    const jobId = schedulerNoStore.addJob({
      schedule: '0 9 * * *',
      message: 'No store job',
    });

    expect(jobId).toBeDefined();
    expect(schedulerNoStore.listJobs()).toHaveLength(1);

    schedulerNoStore.removeJob(jobId);
    expect(schedulerNoStore.listJobs()).toHaveLength(0);

    schedulerNoStore.stop();
  });
});
