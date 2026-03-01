// apps/gateway/src/scheduler/HeartbeatRunner.ts
import fs from 'fs';
import path from 'path';

export interface HeartbeatConfig {
  intervalMs: number;
  workspacePath: string;
  activeHours?: {
    start: string; // "HH:MM"
    end: string;   // "HH:MM"
    timezone?: string;
  };
  agentHandler: (prompt: string, sessionId: string) => Promise<{ content: string }>;
  notifier: (message: string) => Promise<void>;
  ackMaxChars?: number;
}

const DEFAULT_HEARTBEAT_PROMPT = `You are performing a routine heartbeat check. Review the checklist below and determine if anything needs the user's attention.

If there is something actionable, describe it clearly and concisely.
If everything looks fine and there is nothing to report, respond with exactly: HEARTBEAT_OK

Checklist:
`;

const DEFAULT_EMPTY_PROMPT = `You are performing a routine heartbeat check. There is no specific checklist configured. Check if there are any pending tasks, recent errors, or anything that might need the user's attention.

If there is nothing actionable, respond with exactly: HEARTBEAT_OK`;

export class HeartbeatRunner {
  private config: HeartbeatConfig;
  private timer: ReturnType<typeof setInterval> | null = null;
  private running = false;
  private lastRun: Date | null = null;
  private ackMaxChars: number;

  constructor(config: HeartbeatConfig) {
    this.config = config;
    this.ackMaxChars = config.ackMaxChars ?? 300;
  }

  start(): void {
    if (this.timer) return;
    this.running = true;
    this.timer = setInterval(() => {
      this.runOnce().catch(err => {
        console.error('[HeartbeatRunner] Error during heartbeat:', err);
      });
    }, this.config.intervalMs);
  }

  stop(): void {
    this.running = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  isRunning(): boolean {
    return this.running;
  }

  getLastRun(): Date | null {
    return this.lastRun;
  }

  async runOnce(): Promise<void> {
    if (!this.isWithinActiveHours()) return;

    const checklist = this.readHeartbeatFile();
    const prompt = this.buildPrompt(checklist);
    const sessionId = `heartbeat:${Date.now()}`;

    const response = await this.config.agentHandler(prompt, sessionId);
    this.lastRun = new Date();

    const content = response.content ?? '';
    const stripped = this.stripHeartbeatOk(content);

    // If response was purely HEARTBEAT_OK (nothing left after stripping), suppress
    if (!stripped) {
      return;
    }

    await this.config.notifier(stripped);
  }

  private readHeartbeatFile(): string | null {
    const filePath = path.join(this.config.workspacePath, 'HEARTBEAT.md');
    try {
      if (!fs.existsSync(filePath)) return null;
      return fs.readFileSync(filePath, 'utf-8');
    } catch {
      return null;
    }
  }

  private buildPrompt(checklist: string | null): string {
    if (!checklist || this.isEffectivelyEmpty(checklist)) {
      return DEFAULT_EMPTY_PROMPT;
    }
    return DEFAULT_HEARTBEAT_PROMPT + checklist;
  }

  private isEffectivelyEmpty(content: string): boolean {
    const lines = content.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (trimmed.startsWith('#')) continue;
      if (trimmed.startsWith('<!--') && trimmed.endsWith('-->')) continue;
      if (/^[-*+]\s*\[\s*\]\s*$/.test(trimmed)) continue;
      return false;
    }
    return true;
  }

  private isHeartbeatOk(content: string): boolean {
    return content.includes('HEARTBEAT_OK');
  }

  private stripHeartbeatOk(content: string): string {
    return content.replace(/\bHEARTBEAT_OK\b/g, '').trim();
  }

  private isWithinActiveHours(): boolean {
    const { activeHours } = this.config;
    if (!activeHours) return true;

    const now = new Date();
    const formatter = new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
      timeZone: activeHours.timezone ?? 'UTC',
    });
    const timeStr = formatter.format(now);
    const currentMinutes = this.parseTimeToMinutes(timeStr);
    const startMinutes = this.parseTimeToMinutes(activeHours.start);
    const endMinutes = this.parseTimeToMinutes(activeHours.end);

    if (startMinutes <= endMinutes) {
      return currentMinutes >= startMinutes && currentMinutes < endMinutes;
    }
    return currentMinutes >= startMinutes || currentMinutes < endMinutes;
  }

  private parseTimeToMinutes(time: string): number {
    const [h, m] = time.split(':').map(Number);
    return h * 60 + m;
  }
}
