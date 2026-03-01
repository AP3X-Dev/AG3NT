/**
 * Tests for Scheduler.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { Scheduler } from './Scheduler.js';
import type { SchedulerConfig, ScheduledMessageHandler, ChannelNotifier } from './types.js';

describe('Scheduler', () => {
  let scheduler: Scheduler;
  let mockMessageHandler: ScheduledMessageHandler;
  let mockNotifier: ChannelNotifier;
  let config: SchedulerConfig;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();

    mockMessageHandler = vi.fn().mockResolvedValue({
      text: 'HEARTBEAT_OK',
      notify: false,
    });

    mockNotifier = vi.fn().mockResolvedValue(undefined);

    config = {
      heartbeat: {
        intervalMinutes: 60,
        message: '[HEARTBEAT] Check-in time',
      },
      cronJobs: [],
    };

    scheduler = new Scheduler(config, mockMessageHandler, mockNotifier);
  });

  afterEach(() => {
    scheduler.stop();
    vi.useRealTimers();
  });

  describe('Initialization', () => {
    it('should create scheduler with config', () => {
      expect(scheduler).toBeDefined();
      expect(scheduler.getStatus()).toMatchObject({
        heartbeatRunning: false,
        heartbeatPaused: false,
        jobCount: 0,
      });
    });
  });

  describe('Heartbeat', () => {
    it('should start heartbeat on scheduler start', () => {
      scheduler.start();

      expect(scheduler.isHeartbeatRunning()).toBe(true);
      expect(scheduler.getStatus().heartbeatRunning).toBe(true);
    });

    it('should run heartbeat at configured interval', async () => {
      scheduler.start();

      // Advance time by 60 minutes
      await vi.advanceTimersByTimeAsync(60 * 60 * 1000);

      expect(mockMessageHandler).toHaveBeenCalledWith(
        'HEARTBEAT',
        expect.stringContaining('heartbeat:'),
        expect.objectContaining({ type: 'heartbeat' })
      );
    });

    it('should pause heartbeat', () => {
      scheduler.start();
      scheduler.pauseHeartbeat();

      expect(scheduler.isHeartbeatRunning()).toBe(false);
      expect(scheduler.getStatus().heartbeatPaused).toBe(true);
    });

    it('should resume heartbeat', () => {
      scheduler.start();
      scheduler.pauseHeartbeat();
      scheduler.resumeHeartbeat();

      expect(scheduler.isHeartbeatRunning()).toBe(true);
      expect(scheduler.getStatus().heartbeatPaused).toBe(false);
    });

    it('should not run heartbeat when paused', async () => {
      scheduler.start();
      scheduler.pauseHeartbeat();

      mockMessageHandler.mockClear();

      // Advance time by 60 minutes
      await vi.advanceTimersByTimeAsync(60 * 60 * 1000);

      expect(mockMessageHandler).not.toHaveBeenCalled();
    });

    it('should track last heartbeat time', async () => {
      scheduler.start();

      expect(scheduler.getLastHeartbeat()).toBeNull();

      // Advance time to trigger heartbeat
      await vi.advanceTimersByTimeAsync(60 * 60 * 1000);

      expect(scheduler.getLastHeartbeat()).toBeInstanceOf(Date);
    });

    it('should notify if heartbeat response is not OK', async () => {
      mockMessageHandler.mockResolvedValueOnce({
        text: 'Warning: High memory usage detected',
        notify: true,
      });

      scheduler.start();

      // Advance time to trigger heartbeat
      await vi.advanceTimersByTimeAsync(60 * 60 * 1000);

      expect(mockNotifier).toHaveBeenCalledWith(
        undefined,
        'Warning: High memory usage detected'
      );
    });

    it('should not notify if heartbeat response is OK', async () => {
      mockMessageHandler.mockResolvedValueOnce({
        text: 'HEARTBEAT_OK',
        notify: true,
      });

      scheduler.start();

      // Advance time to trigger heartbeat
      await vi.advanceTimersByTimeAsync(60 * 60 * 1000);

      expect(mockNotifier).not.toHaveBeenCalled();
    });

    it('should stop heartbeat on scheduler stop', () => {
      scheduler.start();
      scheduler.stop();

      expect(scheduler.isHeartbeatRunning()).toBe(false);
    });
  });

  describe('Cron Jobs', () => {
    it('should add cron job', () => {
      const jobId = scheduler.addJob({
        schedule: '0 9 * * *',
        message: 'Daily reminder',
        name: 'Morning Check',
      });

      expect(jobId).toBeDefined();
      expect(jobId).toMatch(/^job-\d+$/);

      const jobs = scheduler.listJobs();
      expect(jobs).toHaveLength(1);
      expect(jobs[0]).toMatchObject({
        id: jobId,
        message: 'Daily reminder',
        name: 'Morning Check',
      });
    });

    it('should remove cron job', () => {
      const jobId = scheduler.addJob({
        schedule: '* * * * *',
        message: 'Test job',
      });

      const removed = scheduler.removeJob(jobId);

      expect(removed).toBe(true);
      expect(scheduler.listJobs()).toHaveLength(0);
    });

    it('should return false when removing non-existent job', () => {
      const removed = scheduler.removeJob('non-existent');

      expect(removed).toBe(false);
    });

    it('should list all jobs', () => {
      scheduler.addJob({ schedule: '0 9 * * *', message: 'Job 1' });
      scheduler.addJob({ schedule: '0 17 * * *', message: 'Job 2' });
      scheduler.addJob({ schedule: '0 0 * * 0', message: 'Job 3' });

      const jobs = scheduler.listJobs();

      expect(jobs).toHaveLength(3);
      expect(jobs.map((j) => j.message)).toEqual(['Job 1', 'Job 2', 'Job 3']);
    });

    it('should pause cron job', () => {
      const jobId = scheduler.addJob({
        schedule: '* * * * *',
        message: 'Test job',
      });

      const paused = scheduler.pauseJob(jobId);

      expect(paused).toBe(true);

      const jobs = scheduler.listJobs();
      expect(jobs[0].paused).toBe(true);
    });

    it('should resume cron job', () => {
      const jobId = scheduler.addJob({
        schedule: '* * * * *',
        message: 'Test job',
      });

      scheduler.pauseJob(jobId);
      const resumed = scheduler.resumeJob(jobId);

      expect(resumed).toBe(true);

      const jobs = scheduler.listJobs();
      expect(jobs[0].paused).toBe(false);
    });

    it('should return false when pausing non-existent job', () => {
      const paused = scheduler.pauseJob('non-existent');

      expect(paused).toBe(false);
    });

    it('should return false when resuming non-existent job', () => {
      const resumed = scheduler.resumeJob('non-existent');

      expect(resumed).toBe(false);
    });

    it('should support one-shot jobs with relative time', () => {
      const jobId = scheduler.addJob({
        schedule: 'in 10 minutes',
        message: 'Reminder',
        oneShot: true,
      });

      expect(jobId).toBeDefined();

      const jobs = scheduler.listJobs();
      expect(jobs[0].oneShot).toBe(true);
    });

    it('should clear all jobs on stop', () => {
      scheduler.addJob({ schedule: '* * * * *', message: 'Job 1' });
      scheduler.addJob({ schedule: '* * * * *', message: 'Job 2' });

      scheduler.stop();

      expect(scheduler.listJobs()).toHaveLength(0);
    });
  });

  describe('Scheduler Status', () => {
    it('should return correct status', () => {
      scheduler.start();

      const status = scheduler.getStatus();

      expect(status).toMatchObject({
        heartbeatRunning: true,
        heartbeatPaused: false,
        jobCount: 0,
      });
    });

    it('should include job count in status', () => {
      scheduler.addJob({ schedule: '* * * * *', message: 'Job 1' });
      scheduler.addJob({ schedule: '* * * * *', message: 'Job 2' });

      const status = scheduler.getStatus();

      expect(status.jobCount).toBe(2);
    });

    it('should reflect paused state in status', () => {
      scheduler.start();
      scheduler.pauseHeartbeat();

      const status = scheduler.getStatus();

      expect(status.heartbeatPaused).toBe(true);
      // Note: heartbeatRunning checks if interval exists, not if it's paused
      expect(status.heartbeatRunning).toBe(true);
    });
  });

  describe('Reminder Scheduling', () => {
    it('should schedule reminder with relative time', () => {
      const when = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes from now
      const jobId = scheduler.scheduleReminder(when, 'Meeting in 10 minutes');

      expect(jobId).toBeDefined();

      const jobs = scheduler.listJobs();
      expect(jobs).toHaveLength(1);
      // Note: Scheduler adds '⏰ Reminder: ' prefix to reminder messages
      expect(jobs[0].message).toBe('⏰ Reminder: Meeting in 10 minutes');
      expect(jobs[0].oneShot).toBe(true);
    });

    it('should schedule reminder with channel target', () => {
      const when = new Date(Date.now() + 5 * 60 * 1000);
      const jobId = scheduler.scheduleReminder(when, 'Reminder', 'telegram:bot-1:chat-123');

      expect(jobId).toBeDefined();

      const jobs = scheduler.listJobs();
      expect(jobs[0].channelTarget).toBe('telegram:bot-1:chat-123');
    });
  });

  describe('Cron Exponential Backoff', () => {
    it('should apply backoff after job failure', async () => {
      const failHandler: ScheduledMessageHandler = vi.fn()
        .mockRejectedValueOnce(new Error('task failed'))
        .mockResolvedValue({ text: 'ok', notify: false });

      const backoffScheduler = new Scheduler(config, failHandler, mockNotifier);
      const jobId = backoffScheduler.addJob({
        schedule: '*/5 * * * *',
        message: 'test',
      });

      await backoffScheduler['runCronJob'](jobId);

      // Job should still exist and have backoffUntil set
      const entry = backoffScheduler['jobs'].get(jobId);
      expect(entry?.job.backoffUntil).toBeDefined();
      expect(backoffScheduler['consecutiveErrors'].get(jobId)).toBe(1);

      backoffScheduler.stop();
    });

    it('should reset backoff on success after failure', async () => {
      let callCount = 0;
      const handler: ScheduledMessageHandler = vi.fn().mockImplementation(() => {
        callCount++;
        if (callCount === 1) throw new Error('fail');
        return Promise.resolve({ text: 'ok', notify: false });
      });

      const s = new Scheduler(config, handler, mockNotifier);
      const jobId = s.addJob({ schedule: '*/5 * * * *', message: 'test' });

      await s['runCronJob'](jobId);
      expect(s['consecutiveErrors'].get(jobId)).toBe(1);
      const entry1 = s['jobs'].get(jobId);
      expect(entry1?.job.backoffUntil).toBeDefined();

      // Advance past the backoff window (30s) so the next call is not skipped
      await vi.advanceTimersByTimeAsync(30_000);

      await s['runCronJob'](jobId);
      expect(s['consecutiveErrors'].get(jobId)).toBe(0);
      const entry2 = s['jobs'].get(jobId);
      expect(entry2?.job.backoffUntil).toBeUndefined();

      s.stop();
    });

    it('should skip execution during backoff window', async () => {
      const failHandler: ScheduledMessageHandler = vi.fn()
        .mockRejectedValue(new Error('fail'));

      const mockStore = {
        save: vi.fn(),
        recordRun: vi.fn(),
        loadAll: vi.fn().mockReturnValue([]),
        getRunHistory: vi.fn().mockReturnValue([]),
      };

      const s = new Scheduler(config, failHandler, mockNotifier, undefined, mockStore as any);
      const jobId = s.addJob({ schedule: '*/5 * * * *', message: 'test' });

      // First failure — sets backoff
      await s['runCronJob'](jobId);
      expect(failHandler).toHaveBeenCalledTimes(1);

      // Second call within backoff window — should skip
      await s['runCronJob'](jobId);
      // Handler should NOT have been called again (still only 1 call)
      expect(failHandler).toHaveBeenCalledTimes(1);
      // Should record a 'skipped' run
      expect(mockStore.recordRun).toHaveBeenCalledWith(
        jobId,
        expect.objectContaining({ status: 'skipped' }),
      );

      s.stop();
    });
  });

  describe('Cron Timezone Support', () => {
    it('should accept timezone in job definition', () => {
      const jobId = scheduler.addJob({
        schedule: '0 9 * * *',
        message: 'morning check',
        timezone: 'America/New_York',
      });

      expect(jobId).toBeDefined();
      const status = scheduler.getStatus();
      expect(status.jobCount).toBe(1);
    });

    it('should work without timezone (backward compatible)', () => {
      const jobId = scheduler.addJob({
        schedule: '0 9 * * *',
        message: 'no tz',
      });

      expect(jobId).toBeDefined();
      const status = scheduler.getStatus();
      expect(status.jobCount).toBe(1);
    });
  });
});


