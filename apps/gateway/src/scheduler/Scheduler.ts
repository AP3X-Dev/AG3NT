/**
 * AG3NT Scheduler.
 *
 * Manages heartbeat (periodic checks) and cron jobs (scheduled tasks).
 * Injects scheduled messages to the agent and notifies channels of responses.
 */

import schedule from "node-schedule";
import type {
  CronJob,
  CronJobDefinition,
  SchedulerConfig,
  ScheduledMessageHandler,
  ChannelNotifier,
  SchedulerEvent,
  SchedulerEventHandler,
} from "./types.js";
import { ERROR_BACKOFF_MS } from "./types.js";
import type { CronJobStore } from "./CronJobStore.js";
import { HeartbeatRunner } from "./HeartbeatRunner.js";

/**
 * Parse relative time string (e.g., "in 10 minutes") to a Date.
 * Returns null if not a relative time expression.
 */
function parseRelativeTime(expr: string): Date | null {
  const match = expr.match(/^in\s+(\d+)\s+(second|minute|hour|day)s?$/i);
  if (!match) return null;

  const amount = parseInt(match[1], 10);
  const unit = match[2].toLowerCase();

  const now = new Date();
  switch (unit) {
    case "second":
      return new Date(now.getTime() + amount * 1000);
    case "minute":
      return new Date(now.getTime() + amount * 60 * 1000);
    case "hour":
      return new Date(now.getTime() + amount * 60 * 60 * 1000);
    case "day":
      return new Date(now.getTime() + amount * 24 * 60 * 60 * 1000);
    default:
      return null;
  }
}

export class Scheduler {
  private config: SchedulerConfig;
  private messageHandler: ScheduledMessageHandler;
  private notifier: ChannelNotifier;
  private eventHandler?: SchedulerEventHandler;
  private store?: CronJobStore;

  // Heartbeat state
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private heartbeatPaused = false;
  private heartbeatActive = false;
  private lastHeartbeat: Date | null = null;
  private heartbeatRunner?: HeartbeatRunner;

  // Cron jobs
  private jobs = new Map<string, { job: CronJob; scheduled: schedule.Job }>();
  private jobCounter = 0;

  // Error tracking for backoff
  private consecutiveErrors = new Map<string, number>();

  constructor(
    config: SchedulerConfig,
    messageHandler: ScheduledMessageHandler,
    notifier: ChannelNotifier,
    eventHandler?: SchedulerEventHandler,
    store?: CronJobStore
  ) {
    this.config = config;
    this.messageHandler = messageHandler;
    this.notifier = notifier;
    this.eventHandler = eventHandler;
    this.store = store;
  }

  /**
   * Start the scheduler (heartbeat and cron jobs).
   */
  start(): void {
    // Restore persisted jobs from store before anything else
    this.restorePersistedJobs();

    // Start heartbeat if configured
    if (this.config.heartbeat.intervalMinutes > 0) {
      this.startHeartbeat();
    }

    // Register initial cron jobs from config
    for (const jobDef of this.config.cronJobs) {
      this.addJob(jobDef);
    }

    console.log(
      `[Scheduler] Started. Heartbeat: ${this.config.heartbeat.intervalMinutes}min, Jobs: ${this.jobs.size}`
    );
  }

  /**
   * Stop all scheduled tasks.
   */
  stop(): void {
    // Stop HeartbeatRunner if active
    if (this.heartbeatRunner) {
      this.heartbeatRunner.stop();
      this.heartbeatRunner = undefined;
      this.heartbeatActive = false;
    }

    // Stop heartbeat interval (legacy)
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Cancel all cron jobs
    for (const { scheduled } of this.jobs.values()) {
      scheduled.cancel();
    }
    this.jobs.clear();

    console.log("[Scheduler] Stopped");
  }

  // =========================================================================
  // Heartbeat
  // =========================================================================

  private startHeartbeat(): void {
    const intervalMs = this.config.heartbeat.intervalMinutes * 60 * 1000;

    // If workspacePath is configured, delegate to HeartbeatRunner
    // which reads HEARTBEAT.md and supports active-hours filtering.
    if (this.config.workspacePath) {
      this.heartbeatRunner = new HeartbeatRunner({
        intervalMs,
        workspacePath: this.config.workspacePath,
        agentHandler: async (prompt, sessionId) => {
          const result = await this.messageHandler(prompt, sessionId, {
            type: "heartbeat",
            timestamp: new Date().toISOString(),
          });
          // Update Scheduler-level tracking
          this.lastHeartbeat = new Date();
          this.emitEvent({
            type: "heartbeat",
            message: prompt.slice(0, 100),
            timestamp: this.lastHeartbeat,
            response: result.text,
          });
          return { content: result.text };
        },
        notifier: async (message) => {
          await this.notifier(undefined, message);
        },
      });
      this.heartbeatRunner.start();
      this.heartbeatActive = true;
      console.log(
        `[Scheduler] HeartbeatRunner started (every ${this.config.heartbeat.intervalMinutes} minutes, workspace: ${this.config.workspacePath})`
      );
      return;
    }

    // Fallback: basic interval-based heartbeat (no HEARTBEAT.md support)
    this.heartbeatInterval = setInterval(() => {
      if (!this.heartbeatPaused) {
        this.runHeartbeat();
      }
    }, intervalMs);

    console.log(
      `[Scheduler] Heartbeat started (every ${this.config.heartbeat.intervalMinutes} minutes)`
    );
  }

  private async runHeartbeat(): Promise<void> {
    const now = new Date();
    this.lastHeartbeat = now;

    console.log(`[Scheduler] Running heartbeat at ${now.toISOString()}`);

    try {
      // Send HEARTBEAT message to agent with isolated session
      const sessionId = `heartbeat:${now.getTime()}`;
      const result = await this.messageHandler(
        "HEARTBEAT",
        sessionId,
        { type: "heartbeat", timestamp: now.toISOString() }
      );

      // Emit event
      this.emitEvent({
        type: "heartbeat",
        message: "HEARTBEAT",
        timestamp: now,
        response: result.text,
      });

      // If agent has something to report (not HEARTBEAT_OK), notify
      if (result.notify && !this.isHeartbeatOk(result.text)) {
        await this.notifier(undefined, result.text);
      }
    } catch (err) {
      console.error("[Scheduler] Heartbeat error:", err);
    }
  }

  private isHeartbeatOk(response: string): boolean {
    const normalized = response.trim().toUpperCase();
    return normalized === "HEARTBEAT_OK" || normalized.includes("HEARTBEAT_OK");
  }

  pauseHeartbeat(): void {
    this.heartbeatPaused = true;
    if (this.heartbeatRunner) {
      this.heartbeatRunner.stop();
      this.heartbeatActive = false;
    }
    console.log("[Scheduler] Heartbeat paused");
  }

  resumeHeartbeat(): void {
    this.heartbeatPaused = false;
    if (this.heartbeatRunner) {
      this.heartbeatRunner.start();
      this.heartbeatActive = true;
    }
    console.log("[Scheduler] Heartbeat resumed");
  }

  isHeartbeatRunning(): boolean {
    return (this.heartbeatActive || this.heartbeatInterval !== null) && !this.heartbeatPaused;
  }

  getLastHeartbeat(): Date | null {
    // HeartbeatRunner tracks its own lastRun; prefer it when available
    if (this.heartbeatRunner) {
      return this.heartbeatRunner.getLastRun() ?? this.lastHeartbeat;
    }
    return this.lastHeartbeat;
  }

  // =========================================================================
  // Cron Jobs
  // =========================================================================

  /**
   * Add a new cron job.
   * @returns The job ID
   */
  addJob(jobDef: CronJobDefinition): string {
    const id = `job-${++this.jobCounter}`;
    const now = new Date();

    // Parse schedule - either cron expression or relative time
    let scheduledJob: schedule.Job;
    const relativeTime = parseRelativeTime(jobDef.schedule);

    if (relativeTime) {
      // One-time job at specific date
      scheduledJob = schedule.scheduleJob(relativeTime, () => {
        this.runCronJob(id);
      });
    } else {
      // Cron expression — pass timezone if provided
      const scheduleSpec = jobDef.timezone
        ? { rule: jobDef.schedule, tz: jobDef.timezone }
        : jobDef.schedule;
      scheduledJob = schedule.scheduleJob(scheduleSpec, () => {
        this.runCronJob(id);
      });
    }

    const cronJob: CronJob = {
      ...jobDef,
      id,
      nextRun: scheduledJob.nextInvocation() ?? null,
      paused: false,
      createdAt: now,
    };

    this.jobs.set(id, { job: cronJob, scheduled: scheduledJob });

    // Persist to store
    this.store?.save({
      id,
      schedule: jobDef.schedule,
      message: jobDef.message,
      sessionMode: jobDef.sessionMode,
      channelTarget: jobDef.channelTarget,
      oneShot: jobDef.oneShot,
      name: jobDef.name,
      timezone: jobDef.timezone,
      deliveryMode: jobDef.deliveryMode,
      enabled: true,
      createdAt: now.toISOString(),
    });

    console.log(`[Scheduler] Added job ${id}: "${jobDef.name ?? jobDef.message.slice(0, 30)}"`);

    return id;
  }

  /**
   * Remove a cron job.
   */
  removeJob(jobId: string): boolean {
    const entry = this.jobs.get(jobId);
    if (!entry) return false;

    entry.scheduled.cancel();
    this.jobs.delete(jobId);
    this.consecutiveErrors.delete(jobId);
    this.store?.delete(jobId);
    console.log(`[Scheduler] Removed job ${jobId}`);
    return true;
  }

  /**
   * List all cron jobs.
   */
  listJobs(): CronJob[] {
    return Array.from(this.jobs.values()).map(({ job, scheduled }) => ({
      ...job,
      nextRun: scheduled.nextInvocation() ?? null,
    }));
  }

  /**
   * Pause a cron job.
   */
  pauseJob(jobId: string): boolean {
    const entry = this.jobs.get(jobId);
    if (!entry) return false;

    entry.scheduled.cancel();
    entry.job.paused = true;
    this.store?.updateEnabled(jobId, false);
    console.log(`[Scheduler] Paused job ${jobId}`);
    return true;
  }

  /**
   * Resume a paused cron job.
   */
  resumeJob(jobId: string): boolean {
    const entry = this.jobs.get(jobId);
    if (!entry || !entry.job.paused) return false;

    // Re-schedule the job
    const relativeTime = parseRelativeTime(entry.job.schedule);
    let newScheduled: schedule.Job;

    if (relativeTime) {
      newScheduled = schedule.scheduleJob(relativeTime, () => {
        this.runCronJob(jobId);
      });
    } else {
      const resumeSpec = entry.job.timezone
        ? { rule: entry.job.schedule, tz: entry.job.timezone }
        : entry.job.schedule;
      newScheduled = schedule.scheduleJob(resumeSpec, () => {
        this.runCronJob(jobId);
      });
    }

    entry.scheduled = newScheduled;
    entry.job.paused = false;
    this.store?.updateEnabled(jobId, true);
    console.log(`[Scheduler] Resumed job ${jobId}`);
    return true;
  }

  private async runCronJob(jobId: string): Promise<void> {
    const entry = this.jobs.get(jobId);
    if (!entry || entry.job.paused) return;

    // Skip if within backoff window
    if (entry.job.backoffUntil && new Date() < entry.job.backoffUntil) {
      console.log(`[Scheduler] Job ${jobId} in backoff until ${entry.job.backoffUntil.toISOString()}, skipping`);
      this.store?.recordRun(jobId, {
        status: 'skipped',
        startedAt: new Date().toISOString(),
        durationMs: 0,
      });
      return;
    }

    const { job } = entry;
    const now = new Date();
    const startTime = Date.now();

    console.log(`[Scheduler] Running cron job ${jobId}: "${job.name ?? job.message.slice(0, 30)}"`);

    try {
      // Determine session ID based on session mode
      const sessionMode = job.sessionMode ?? "isolated";
      const sessionId = sessionMode === "isolated"
        ? `cron:${jobId}:${now.getTime()}`
        : "main";

      const result = await this.messageHandler(
        job.message,
        sessionId,
        { type: "cron", jobId, timestamp: now.toISOString() }
      );

      // Record successful run
      const durationMs = Date.now() - startTime;
      this.consecutiveErrors.set(jobId, 0);
      entry.job.backoffUntil = undefined;
      this.store?.recordRun(jobId, {
        status: 'ok',
        startedAt: now.toISOString(),
        durationMs,
      });

      // Emit event
      this.emitEvent({
        type: "cron",
        jobId,
        message: job.message,
        timestamp: now,
        response: result.text,
      });

      // Notify channel with response (unless background mode)
      const deliveryMode = job.deliveryMode ?? 'notify';
      if (result.notify && deliveryMode !== 'background') {
        await this.notifier(job.channelTarget, result.text, {
          jobId,
          jobName: job.name,
          type: job.oneShot ? "reminder" : "cron",
          sessionId,
        });
      }

      // Remove one-shot jobs after execution
      if (job.oneShot) {
        this.removeJob(jobId);
      }
    } catch (err) {
      const durationMs = Date.now() - startTime;
      const errorCount = (this.consecutiveErrors.get(jobId) ?? 0) + 1;
      this.consecutiveErrors.set(jobId, errorCount);

      const errorMessage = err instanceof Error ? err.message : String(err);

      // Record failed run
      this.store?.recordRun(jobId, {
        status: 'error',
        startedAt: now.toISOString(),
        durationMs,
        error: errorMessage,
      });

      // Determine backoff tier
      const backoffIndex = Math.min(errorCount - 1, ERROR_BACKOFF_MS.length - 1);
      const backoffMs = ERROR_BACKOFF_MS[backoffIndex];
      entry.job.backoffUntil = new Date(Date.now() + backoffMs);

      console.warn(
        `[Scheduler] Cron job ${jobId} error (consecutive: ${errorCount}, ` +
        `next backoff: ${backoffMs}ms): ${errorMessage}`
      );
    }
  }

  // =========================================================================
  // One-Shot Reminders
  // =========================================================================

  /**
   * Schedule a one-shot reminder.
   * @param when - Date object or milliseconds from now
   * @param message - Reminder message
   * @param channelTarget - Optional channel to send to
   * @returns The job ID
   */
  scheduleReminder(when: Date | number, message: string, channelTarget?: string): string {
    const targetDate = typeof when === "number"
      ? new Date(Date.now() + when)
      : when;

    // Convert to cron-compatible schedule
    const jobDef: CronJobDefinition = {
      schedule: `${targetDate.getSeconds()} ${targetDate.getMinutes()} ${targetDate.getHours()} ${targetDate.getDate()} ${targetDate.getMonth() + 1} *`,
      message: `⏰ Reminder: ${message}`,
      sessionMode: "isolated",
      channelTarget,
      oneShot: true,
      name: `Reminder: ${message.slice(0, 30)}`,
    };

    // Use scheduleJob with Date directly for one-shot
    const id = `reminder-${++this.jobCounter}`;
    const scheduledJob = schedule.scheduleJob(targetDate, () => {
      this.runCronJob(id);
    });

    const cronJob: CronJob = {
      ...jobDef,
      id,
      nextRun: targetDate,
      paused: false,
      createdAt: new Date(),
    };

    this.jobs.set(id, { job: cronJob, scheduled: scheduledJob });
    console.log(`[Scheduler] Scheduled reminder ${id} for ${targetDate.toISOString()}`);

    return id;
  }

  // =========================================================================
  // Persistence
  // =========================================================================

  /**
   * Restore persisted jobs from the CronJobStore.
   * Called during start() before the heartbeat begins.
   */
  private restorePersistedJobs(): void {
    if (!this.store) return;

    const persisted = this.store.loadAll();
    let restored = 0;

    for (const pj of persisted) {
      // Skip one-shot jobs — they are ephemeral by design
      if (pj.oneShot) continue;

      const relativeTime = parseRelativeTime(pj.schedule);

      // Skip expired relative-time jobs
      if (relativeTime && relativeTime.getTime() <= Date.now()) continue;

      // Ensure jobCounter stays above persisted IDs to avoid collisions
      const match = pj.id.match(/^job-(\d+)$/);
      if (match) {
        const num = parseInt(match[1], 10);
        if (num >= this.jobCounter) {
          this.jobCounter = num;
        }
      }

      // Track consecutive errors from persisted state
      if (pj.consecutiveErrors) {
        this.consecutiveErrors.set(pj.id, pj.consecutiveErrors);
      }

      // Only create a schedule for enabled jobs — skip scheduling paused jobs entirely
      if (pj.enabled) {
        const scheduledJob = relativeTime
          ? schedule.scheduleJob(relativeTime, () => this.runCronJob(pj.id))
          : schedule.scheduleJob(
              pj.timezone ? { rule: pj.schedule, tz: pj.timezone } : pj.schedule,
              () => this.runCronJob(pj.id)
            );

        const cronJob: CronJob = {
          schedule: pj.schedule,
          message: pj.message,
          sessionMode: pj.sessionMode,
          channelTarget: pj.channelTarget,
          oneShot: pj.oneShot,
          name: pj.name,
          timezone: pj.timezone,
          id: pj.id,
          nextRun: scheduledJob.nextInvocation() ?? null,
          paused: false,
          createdAt: new Date(pj.createdAt),
        };

        this.jobs.set(pj.id, { job: cronJob, scheduled: scheduledJob });
      } else {
        // Paused job: schedule but immediately cancel so the job entry exists in the map
        const pausedJob = relativeTime
          ? schedule.scheduleJob(new Date(Date.now() + 365 * 24 * 60 * 60 * 1000), () => {})
          : schedule.scheduleJob(
              pj.timezone ? { rule: pj.schedule, tz: pj.timezone } : pj.schedule,
              () => this.runCronJob(pj.id)
            );
        pausedJob.cancel();

        const cronJob: CronJob = {
          schedule: pj.schedule,
          message: pj.message,
          sessionMode: pj.sessionMode,
          channelTarget: pj.channelTarget,
          oneShot: pj.oneShot,
          name: pj.name,
          timezone: pj.timezone,
          id: pj.id,
          nextRun: null,
          paused: true,
          createdAt: new Date(pj.createdAt),
        };

        this.jobs.set(pj.id, { job: cronJob, scheduled: pausedJob });
      }

      restored++;
    }

    if (restored > 0) {
      console.log(`[Scheduler] Restored ${restored} persisted job(s)`);
    }
  }

  // =========================================================================
  // Helpers
  // =========================================================================

  private emitEvent(event: SchedulerEvent): void {
    if (this.eventHandler) {
      this.eventHandler(event);
    }
  }

  /**
   * Get scheduler status.
   */
  getStatus(): {
    heartbeatRunning: boolean;
    heartbeatPaused: boolean;
    lastHeartbeat: Date | null;
    jobCount: number;
    jobs: CronJob[];
  } {
    return {
      heartbeatRunning: this.heartbeatActive || this.heartbeatInterval !== null,
      heartbeatPaused: this.heartbeatPaused,
      lastHeartbeat: this.getLastHeartbeat(),
      jobCount: this.jobs.size,
      jobs: this.listJobs(),
    };
  }
}
