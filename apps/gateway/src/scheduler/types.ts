/**
 * Types for AG3NT Scheduler.
 *
 * Supports heartbeat (periodic checks) and cron jobs (scheduled tasks).
 */

/**
 * Session mode for scheduled jobs.
 * - "isolated": Run in a fresh session (no conversation context)
 * - "main": Run in the main user session (carries context)
 */
export type SessionMode = "isolated" | "main";

/**
 * Definition for creating a cron job.
 */
export interface CronJobDefinition {
  /** Cron expression (e.g., "0 9 * * *" for 9 AM daily) or relative time (e.g., "in 10 minutes") */
  schedule: string;
  /** Message to send to the agent when the job fires */
  message: string;
  /** Session mode for the job */
  sessionMode?: SessionMode;
  /** Target channel type to send response to (default: primary) */
  channelTarget?: string;
  /** If true, job is deleted after first run */
  oneShot?: boolean;
  /** Human-readable name for the job */
  name?: string;
}

/**
 * A scheduled cron job.
 */
export interface CronJob extends CronJobDefinition {
  /** Unique job ID */
  id: string;
  /** Next scheduled run time */
  nextRun: Date | null;
  /** Whether the job is paused */
  paused: boolean;
  /** When the job was created */
  createdAt: Date;
}

/**
 * Heartbeat configuration.
 */
export interface HeartbeatConfig {
  /** Interval in minutes (0 = disabled) */
  intervalMinutes: number;
}

/**
 * Scheduler configuration.
 */
export interface SchedulerConfig {
  heartbeat: HeartbeatConfig;
  /** Initial cron jobs from config */
  cronJobs: CronJobDefinition[];
}

/**
 * Callback function for handling scheduled messages.
 * Returns the agent response text.
 */
export type ScheduledMessageHandler = (
  message: string,
  sessionId: string,
  metadata?: Record<string, unknown>
) => Promise<{ text: string; notify: boolean }>;

/**
 * Metadata attached to a scheduler notification.
 */
export interface NotificationMetadata {
  jobId?: string;
  jobName?: string;
  type?: "reminder" | "cron";
  sessionId?: string;
}

/**
 * Callback for sending notifications to channels.
 */
export type ChannelNotifier = (
  channelTarget: string | undefined,
  message: string,
  metadata?: NotificationMetadata
) => Promise<void>;

/**
 * Scheduler events.
 */
export interface SchedulerEvent {
  type: "heartbeat" | "cron" | "reminder";
  jobId?: string;
  message: string;
  timestamp: Date;
  response?: string;
}

/**
 * Scheduler event handler.
 */
export type SchedulerEventHandler = (event: SchedulerEvent) => void;

/**
 * Record of a single cron job execution.
 */
export interface CronJobRunRecord {
  status: 'ok' | 'error' | 'skipped';
  startedAt: string;
  durationMs: number;
  error?: string;
}

/**
 * Backoff schedule (ms) for consecutive job failures: 30s, 1m, 5m, 15m, 60m.
 */
export const ERROR_BACKOFF_MS = [30_000, 60_000, 300_000, 900_000, 3_600_000];

