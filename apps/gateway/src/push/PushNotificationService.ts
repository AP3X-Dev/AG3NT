import fs from 'fs';
import fsp from 'fs/promises';
import path from 'path';

export interface ApnsAuthConfig {
  teamId: string;
  keyId: string;
  privateKey: string;
}

export interface PushTokenRegistration {
  nodeId: string;
  token: string;
  topic: string;
  environment: 'sandbox' | 'production';
  updatedAt: string;
}

export interface AlertPayloadOptions {
  title: string;
  body: string;
  nodeId: string;
  sound?: string;
}

export interface WakePayloadOptions {
  nodeId: string;
  reason?: string;
}

export interface ApnsAlertPayload {
  aps: { alert: { title: string; body: string }; sound: string };
  ag3nt: { kind: string; nodeId: string; ts: number };
}

export interface ApnsWakePayload {
  aps: { 'content-available': 1 };
  ag3nt: { kind: string; nodeId: string; ts: number; reason: string };
}

export class PushNotificationService {
  private tokensPath: string;
  private auth: ApnsAuthConfig;
  private tokens: PushTokenRegistration[];

  constructor(baseDir: string, auth: ApnsAuthConfig) {
    this.tokensPath = path.join(baseDir, 'tokens.json');
    this.auth = auth;
    this.tokens = this.loadTokens();
  }

  private loadTokens(): PushTokenRegistration[] {
    try {
      if (!fs.existsSync(this.tokensPath)) return [];
      const data = fs.readFileSync(this.tokensPath, 'utf-8');
      return JSON.parse(data);
    } catch {
      return [];
    }
  }

  private async saveTokens(): Promise<void> {
    const dir = path.dirname(this.tokensPath);
    await fsp.mkdir(dir, { recursive: true });
    await fsp.writeFile(this.tokensPath, JSON.stringify(this.tokens, null, 2));
  }

  async registerToken(params: Omit<PushTokenRegistration, 'updatedAt'>): Promise<PushTokenRegistration> {
    const registration: PushTokenRegistration = {
      ...params,
      updatedAt: new Date().toISOString(),
    };

    const idx = this.tokens.findIndex(t => t.nodeId === params.nodeId);
    if (idx >= 0) {
      this.tokens[idx] = registration;
    } else {
      this.tokens.push(registration);
    }

    await this.saveTokens();
    return registration;
  }

  async removeToken(nodeId: string): Promise<void> {
    this.tokens = this.tokens.filter(t => t.nodeId !== nodeId);
    await this.saveTokens();
  }

  getTokensForNode(nodeId: string): PushTokenRegistration[] {
    return this.tokens.filter(t => t.nodeId === nodeId);
  }

  listNodes(): string[] {
    return [...new Set(this.tokens.map(t => t.nodeId))];
  }

  buildAlertPayload(opts: AlertPayloadOptions): ApnsAlertPayload {
    return {
      aps: {
        alert: {
          title: opts.title,
          body: opts.body,
        },
        sound: opts.sound ?? 'default',
      },
      ag3nt: {
        kind: 'push.alert',
        nodeId: opts.nodeId,
        ts: Date.now(),
      },
    };
  }

  buildWakePayload(opts: WakePayloadOptions): ApnsWakePayload {
    return {
      aps: {
        'content-available': 1,
      },
      ag3nt: {
        kind: 'node.wake',
        nodeId: opts.nodeId,
        ts: Date.now(),
        reason: opts.reason ?? 'node.invoke',
      },
    };
  }
}
