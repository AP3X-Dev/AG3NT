import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { PushNotificationService } from './PushNotificationService.js';
import fs from 'fs';
import path from 'path';
import os from 'os';

describe('PushNotificationService', () => {
  let service: PushNotificationService;
  let tmpDir: string;
  const auth = {
    teamId: 'TEAM123',
    keyId: 'KEY456',
    privateKey: '-----BEGIN PRIVATE KEY-----\nfake\n-----END PRIVATE KEY-----',
  };

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'push-test-'));
    service = new PushNotificationService(tmpDir, auth);
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('should register a push token', async () => {
    const result = await service.registerToken({
      nodeId: 'node-1',
      token: 'device-token-abc',
      topic: 'com.ag3nt.app',
      environment: 'sandbox',
    });

    expect(result.nodeId).toBe('node-1');
    expect(result.token).toBe('device-token-abc');
    expect(result.topic).toBe('com.ag3nt.app');
    expect(result.environment).toBe('sandbox');
    expect(result.updatedAt).toBeDefined();
    // updatedAt should be a valid ISO string
    expect(new Date(result.updatedAt).toISOString()).toBe(result.updatedAt);
  });

  it('should persist and reload tokens', async () => {
    await service.registerToken({
      nodeId: 'node-1',
      token: 'device-token-abc',
      topic: 'com.ag3nt.app',
      environment: 'sandbox',
    });

    // Create a new service instance pointing at the same directory
    const service2 = new PushNotificationService(tmpDir, auth);
    const tokens = service2.getTokensForNode('node-1');

    expect(tokens).toHaveLength(1);
    expect(tokens[0].token).toBe('device-token-abc');
    expect(tokens[0].nodeId).toBe('node-1');
  });

  it('should update token for same nodeId', async () => {
    await service.registerToken({
      nodeId: 'node-1',
      token: 'old-token',
      topic: 'com.ag3nt.app',
      environment: 'sandbox',
    });

    await service.registerToken({
      nodeId: 'node-1',
      token: 'new-token',
      topic: 'com.ag3nt.app',
      environment: 'production',
    });

    const tokens = service.getTokensForNode('node-1');
    expect(tokens).toHaveLength(1);
    expect(tokens[0].token).toBe('new-token');
    expect(tokens[0].environment).toBe('production');
  });

  it('should remove a token', async () => {
    await service.registerToken({
      nodeId: 'node-1',
      token: 'device-token-abc',
      topic: 'com.ag3nt.app',
      environment: 'sandbox',
    });

    await service.registerToken({
      nodeId: 'node-2',
      token: 'device-token-def',
      topic: 'com.ag3nt.app',
      environment: 'production',
    });

    await service.removeToken('node-1');

    const tokens1 = service.getTokensForNode('node-1');
    const tokens2 = service.getTokensForNode('node-2');

    expect(tokens1).toHaveLength(0);
    expect(tokens2).toHaveLength(1);
    expect(tokens2[0].token).toBe('device-token-def');
  });

  it('should build APNS alert payload', () => {
    const now = Date.now();
    const payload = service.buildAlertPayload({
      title: 'Task Complete',
      body: 'Your agent finished running.',
      nodeId: 'node-1',
    });

    expect(payload.aps.alert.title).toBe('Task Complete');
    expect(payload.aps.alert.body).toBe('Your agent finished running.');
    expect(payload.aps.sound).toBe('default');
    expect(payload.ag3nt.kind).toBe('push.alert');
    expect(payload.ag3nt.nodeId).toBe('node-1');
    expect(payload.ag3nt.ts).toBeGreaterThanOrEqual(now);
    expect(payload.ag3nt.ts).toBeLessThanOrEqual(Date.now());

    // Test custom sound
    const payload2 = service.buildAlertPayload({
      title: 'Alert',
      body: 'Test',
      nodeId: 'node-2',
      sound: 'chime.caf',
    });
    expect(payload2.aps.sound).toBe('chime.caf');
  });

  it('should build APNS background wake payload', () => {
    const now = Date.now();
    const payload = service.buildWakePayload({
      nodeId: 'node-1',
    });

    expect(payload.aps['content-available']).toBe(1);
    expect(payload.ag3nt.kind).toBe('node.wake');
    expect(payload.ag3nt.nodeId).toBe('node-1');
    expect(payload.ag3nt.reason).toBe('node.invoke');
    expect(payload.ag3nt.ts).toBeGreaterThanOrEqual(now);
    expect(payload.ag3nt.ts).toBeLessThanOrEqual(Date.now());

    // Test custom reason
    const payload2 = service.buildWakePayload({
      nodeId: 'node-2',
      reason: 'cron.trigger',
    });
    expect(payload2.ag3nt.reason).toBe('cron.trigger');
  });

  it('should handle missing tokens file gracefully', () => {
    // Service constructed with a directory that has no tokens.json
    const emptyDir = fs.mkdtempSync(path.join(os.tmpdir(), 'push-empty-'));
    try {
      const freshService = new PushNotificationService(emptyDir, auth);
      const tokens = freshService.getTokensForNode('nonexistent');
      expect(tokens).toEqual([]);
      expect(freshService.listNodes()).toEqual([]);
    } finally {
      fs.rmSync(emptyDir, { recursive: true, force: true });
    }

    // Also test corrupted file
    const corruptDir = fs.mkdtempSync(path.join(os.tmpdir(), 'push-corrupt-'));
    try {
      fs.writeFileSync(path.join(corruptDir, 'tokens.json'), '{{not valid json}}');
      const corruptService = new PushNotificationService(corruptDir, auth);
      const tokens = corruptService.getTokensForNode('anything');
      expect(tokens).toEqual([]);
      expect(corruptService.listNodes()).toEqual([]);
    } finally {
      fs.rmSync(corruptDir, { recursive: true, force: true });
    }
  });

  it('should list all registered nodes', async () => {
    await service.registerToken({
      nodeId: 'node-alpha',
      token: 'token-a',
      topic: 'com.ag3nt.app',
      environment: 'sandbox',
    });

    await service.registerToken({
      nodeId: 'node-beta',
      token: 'token-b',
      topic: 'com.ag3nt.app',
      environment: 'production',
    });

    await service.registerToken({
      nodeId: 'node-gamma',
      token: 'token-c',
      topic: 'com.ag3nt.app',
      environment: 'sandbox',
    });

    const nodes = service.listNodes();
    expect(nodes).toHaveLength(3);
    expect(nodes).toContain('node-alpha');
    expect(nodes).toContain('node-beta');
    expect(nodes).toContain('node-gamma');
  });
});
