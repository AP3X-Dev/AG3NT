import fs from 'fs';
import path from 'path';

export interface PersistedCronJob {
  id: string;
  schedule: string;
  message: string;
  sessionMode?: 'isolated' | 'main';
  channelTarget?: string;
  oneShot?: boolean;
  name?: string;
  enabled: boolean;
  createdAt: string;
  lastRunAt?: string;
  consecutiveErrors?: number;
}

export interface RunRecord {
  status: 'ok' | 'error' | 'skipped';
  startedAt: string;
  durationMs: number;
  error?: string;
}

const MAX_RUN_HISTORY = 100;

export class CronJobStore {
  private jobsPath: string;
  private runsDir: string;

  constructor(baseDir: string) {
    this.jobsPath = path.join(baseDir, 'jobs.json');
    this.runsDir = path.join(baseDir, 'runs');
  }

  loadAll(): PersistedCronJob[] {
    try {
      if (!fs.existsSync(this.jobsPath)) return [];
      const data = fs.readFileSync(this.jobsPath, 'utf-8');
      return JSON.parse(data);
    } catch {
      return [];
    }
  }

  save(job: PersistedCronJob): void {
    const jobs = this.loadAll();
    const idx = jobs.findIndex(j => j.id === job.id);
    if (idx >= 0) {
      jobs[idx] = job;
    } else {
      jobs.push(job);
    }
    this._writeJobs(jobs);
  }

  delete(jobId: string): void {
    const jobs = this.loadAll().filter(j => j.id !== jobId);
    this._writeJobs(jobs);
  }

  updateEnabled(jobId: string, enabled: boolean): void {
    const jobs = this.loadAll();
    const job = jobs.find(j => j.id === jobId);
    if (job) {
      job.enabled = enabled;
      this._writeJobs(jobs);
    }
  }

  recordRun(jobId: string, record: RunRecord): void {
    const filePath = path.join(this.runsDir, `${jobId}.json`);
    let history: RunRecord[] = [];
    try {
      if (fs.existsSync(filePath)) {
        history = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
      }
    } catch { /* start fresh */ }
    history.push(record);
    if (history.length > MAX_RUN_HISTORY) {
      history = history.slice(history.length - MAX_RUN_HISTORY);
    }
    fs.mkdirSync(this.runsDir, { recursive: true });
    fs.writeFileSync(filePath, JSON.stringify(history, null, 2));
  }

  getRunHistory(jobId: string): RunRecord[] {
    const filePath = path.join(this.runsDir, `${jobId}.json`);
    try {
      if (!fs.existsSync(filePath)) return [];
      return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    } catch {
      return [];
    }
  }

  private _writeJobs(jobs: PersistedCronJob[]): void {
    const dir = path.dirname(this.jobsPath);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(this.jobsPath, JSON.stringify(jobs, null, 2));
  }
}
