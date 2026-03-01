/**
 * Tests for push token registration API routes.
 */

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import express from "express";
import request from "supertest";
import { createPushRouter } from "./push.js";
import { PushNotificationService } from "../push/PushNotificationService.js";
import fs from "fs";
import path from "path";
import os from "os";

describe("Push Routes", () => {
  let app: express.Express;
  let service: PushNotificationService;
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "push-route-"));
    service = new PushNotificationService(tmpDir, {
      teamId: "T",
      keyId: "K",
      privateKey: "PK",
    });
    app = express();
    app.use(express.json());
    app.use("/push", createPushRouter(service));
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe("POST /push/register", () => {
    it("should register a token", async () => {
      const res = await request(app)
        .post("/push/register")
        .send({
          nodeId: "node-1",
          token: "abc123",
          topic: "com.app",
          environment: "sandbox",
        });
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
      expect(res.body.registration.nodeId).toBe("node-1");
      expect(res.body.registration.token).toBe("abc123");
      expect(res.body.registration.topic).toBe("com.app");
      expect(res.body.registration.environment).toBe("sandbox");
      expect(res.body.registration.updatedAt).toBeDefined();
    });

    it("should default environment to sandbox when not specified", async () => {
      const res = await request(app)
        .post("/push/register")
        .send({ nodeId: "node-1", token: "abc", topic: "com.app" });
      expect(res.status).toBe(200);
      expect(res.body.registration.environment).toBe("sandbox");
    });

    it("should accept production environment", async () => {
      const res = await request(app)
        .post("/push/register")
        .send({
          nodeId: "node-1",
          token: "abc",
          topic: "com.app",
          environment: "production",
        });
      expect(res.status).toBe(200);
      expect(res.body.registration.environment).toBe("production");
    });

    it("should reject missing nodeId", async () => {
      const res = await request(app)
        .post("/push/register")
        .send({ token: "abc", topic: "com.app" });
      expect(res.status).toBe(400);
      expect(res.body.ok).toBe(false);
      expect(res.body.error).toContain("Missing required fields");
    });

    it("should reject missing token", async () => {
      const res = await request(app)
        .post("/push/register")
        .send({ nodeId: "node-1", topic: "com.app" });
      expect(res.status).toBe(400);
    });

    it("should reject missing topic", async () => {
      const res = await request(app)
        .post("/push/register")
        .send({ nodeId: "node-1", token: "abc" });
      expect(res.status).toBe(400);
    });

    it("should reject empty body", async () => {
      const res = await request(app).post("/push/register").send({});
      expect(res.status).toBe(400);
    });

    it("should reject oversized nodeId", async () => {
      const res = await request(app)
        .post("/push/register")
        .send({ nodeId: "x".repeat(257), token: "abc", topic: "com.app" });
      expect(res.status).toBe(400);
      expect(res.body.error).toBe("Invalid nodeId");
    });

    it("should reject oversized token", async () => {
      const res = await request(app)
        .post("/push/register")
        .send({ nodeId: "n1", token: "x".repeat(513), topic: "com.app" });
      expect(res.status).toBe(400);
      expect(res.body.error).toBe("Invalid token");
    });
  });

  describe("DELETE /push/register/:nodeId", () => {
    it("should unregister a node", async () => {
      await service.registerToken({
        nodeId: "node-1",
        token: "abc",
        topic: "com.app",
        environment: "sandbox",
      });
      const res = await request(app).delete("/push/register/node-1");
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
      expect(service.getTokensForNode("node-1")).toHaveLength(0);
    });

    it("should succeed even if node does not exist", async () => {
      const res = await request(app).delete("/push/register/nonexistent");
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
    });
  });

  describe("GET /push/tokens", () => {
    it("should return empty list when no nodes registered", async () => {
      const res = await request(app).get("/push/tokens");
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
      expect(res.body.nodes).toHaveLength(0);
    });

    it("should list all registered nodes", async () => {
      await service.registerToken({
        nodeId: "n1",
        token: "a",
        topic: "com.app",
        environment: "sandbox",
      });
      await service.registerToken({
        nodeId: "n2",
        token: "b",
        topic: "com.app",
        environment: "sandbox",
      });
      const res = await request(app).get("/push/tokens");
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
      expect(res.body.nodes).toHaveLength(2);

      const nodeIds = res.body.nodes.map(
        (n: { nodeId: string }) => n.nodeId,
      );
      expect(nodeIds).toContain("n1");
      expect(nodeIds).toContain("n2");
    });

    it("should include token details for each node", async () => {
      await service.registerToken({
        nodeId: "n1",
        token: "tok-a",
        topic: "com.app",
        environment: "sandbox",
      });
      const res = await request(app).get("/push/tokens");
      expect(res.status).toBe(200);

      const node = res.body.nodes[0];
      expect(node.nodeId).toBe("n1");
      expect(node.tokens).toHaveLength(1);
      expect(node.tokens[0].token).toBe("tok-a");
    });
  });

  describe("POST /push/test", () => {
    it("should return test payload for registered node", async () => {
      await service.registerToken({
        nodeId: "n1",
        token: "device-tok",
        topic: "com.app",
        environment: "sandbox",
      });
      const res = await request(app)
        .post("/push/test")
        .send({ nodeId: "n1" });
      expect(res.status).toBe(200);
      expect(res.body.ok).toBe(true);
      expect(res.body.payload.aps.alert.title).toBe("Test Notification");
      expect(res.body.payload.aps.alert.body).toBe("This is a test push from AG3NT");
      expect(res.body.token).toBe("device-tok");
    });

    it("should accept custom title and body", async () => {
      await service.registerToken({
        nodeId: "n1",
        token: "tok",
        topic: "com.app",
        environment: "sandbox",
      });
      const res = await request(app)
        .post("/push/test")
        .send({ nodeId: "n1", title: "Hello", body: "World" });
      expect(res.status).toBe(200);
      expect(res.body.payload.aps.alert.title).toBe("Hello");
      expect(res.body.payload.aps.alert.body).toBe("World");
    });

    it("should reject missing nodeId", async () => {
      const res = await request(app).post("/push/test").send({});
      expect(res.status).toBe(400);
      expect(res.body.error).toBe("Missing nodeId");
    });

    it("should return 404 for unregistered node", async () => {
      const res = await request(app)
        .post("/push/test")
        .send({ nodeId: "unknown" });
      expect(res.status).toBe(404);
      expect(res.body.error).toContain("No tokens");
    });
  });
});
