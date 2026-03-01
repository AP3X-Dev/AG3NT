/**
 * REST API routes for push token management.
 *
 * Endpoints (mounted at `${httpPath}/push`):
 *   POST   /register        — Register a device push token
 *   DELETE /register/:nodeId — Unregister a device
 *   GET    /tokens           — List all registered nodes with their tokens
 *   POST   /test             — Build a test push payload (dry-run)
 */

import { Router } from "express";
import type { PushNotificationService } from "../push/PushNotificationService.js";

export function createPushRouter(service: PushNotificationService): Router {
  const router = Router();

  router.post("/register", (req, res) => {
    const { nodeId, token, topic, environment } = req.body ?? {};
    if (!nodeId || !token || !topic) {
      res.status(400).json({
        ok: false,
        error: "Missing required fields: nodeId, token, topic",
      });
      return;
    }
    if (typeof nodeId !== "string" || nodeId.length > 256) {
      res.status(400).json({ ok: false, error: "Invalid nodeId" });
      return;
    }
    if (typeof token !== "string" || token.length > 512) {
      res.status(400).json({ ok: false, error: "Invalid token" });
      return;
    }
    if (typeof topic !== "string" || topic.length > 256) {
      res.status(400).json({ ok: false, error: "Invalid topic" });
      return;
    }
    const env = environment === "production" ? "production" : "sandbox";
    const registration = service.registerToken({
      nodeId,
      token,
      topic,
      environment: env,
    });
    res.json({ ok: true, registration });
  });

  router.delete("/register/:nodeId", (req, res) => {
    service.removeToken(req.params.nodeId);
    res.json({ ok: true });
  });

  router.get("/tokens", (_req, res) => {
    const nodes = service.listNodes();
    const tokens = nodes.map((nodeId) => ({
      nodeId,
      tokens: service.getTokensForNode(nodeId),
    }));
    res.json({ ok: true, nodes: tokens });
  });

  router.post("/test", (req, res) => {
    const { nodeId, title, body } = req.body ?? {};
    if (!nodeId) {
      res.status(400).json({ ok: false, error: "Missing nodeId" });
      return;
    }
    const tokens = service.getTokensForNode(nodeId);
    if (tokens.length === 0) {
      res.status(404).json({ ok: false, error: "No tokens registered for node" });
      return;
    }
    const payload = service.buildAlertPayload({
      title: title ?? "Test Notification",
      body: body ?? "This is a test push from AG3NT",
      nodeId,
    });
    res.json({ ok: true, payload, token: tokens[0].token });
  });

  return router;
}
