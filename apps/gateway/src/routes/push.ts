/**
 * REST API routes for push token management.
 *
 * Endpoints (mounted at `${httpPath}/push`):
 *   POST   /register        — Register a device push token
 *   DELETE /register/:nodeId — Unregister a device
 *   GET    /tokens           — List all registered nodes with their tokens
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

  return router;
}
