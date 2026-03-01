// apps/gateway/src/scheduler/SessionRecovery.ts
import fs from 'fs';
import path from 'path';

export interface GatewayState {
  lastHeartbeat: string | null;
  activeSessions: string[];
  schedulerRunning: boolean;
  savedAt?: string;
}

const STATE_FILE = 'gateway-state.json';

export class SessionRecovery {
  private statePath: string;

  constructor(baseDir: string) {
    this.statePath = path.join(baseDir, STATE_FILE);
  }

  saveState(state: GatewayState): void {
    const dir = path.dirname(this.statePath);
    fs.mkdirSync(dir, { recursive: true });
    const data = { ...state, savedAt: new Date().toISOString() };
    fs.writeFileSync(this.statePath, JSON.stringify(data, null, 2));
  }

  loadState(): GatewayState | null {
    try {
      if (!fs.existsSync(this.statePath)) return null;
      const data = fs.readFileSync(this.statePath, 'utf-8');
      return JSON.parse(data);
    } catch {
      return null;
    }
  }

  checkpoint(state: GatewayState): void {
    this.saveState(state);
  }

  clearState(): void {
    try {
      if (fs.existsSync(this.statePath)) {
        fs.unlinkSync(this.statePath);
      }
    } catch {
      // ignore
    }
  }
}
