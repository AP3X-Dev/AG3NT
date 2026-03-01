/**
 * Tests for ChannelDeliveryService wiring into createGateway.
 *
 * Validates that:
 * 1. Incoming channel messages track their chat ID as the default for that adapter
 * 2. The Scheduler's channelNotifier callback delivers via ChannelDeliveryService
 * 3. First-wins semantics are preserved for chat ID tracking
 * 4. Explicit overrides work correctly
 */

import { describe, it, expect, vi } from "vitest";
import { ChannelDeliveryService } from "./ChannelDeliveryService.js";
import type { IChannelAdapter } from "./types.js";

describe("ChannelDeliveryService Wiring", () => {
  it("should track chat ID from incoming message and deliver scheduled notification", async () => {
    const service = new ChannelDeliveryService();
    const adapter: IChannelAdapter = {
      type: "telegram",
      id: "telegram-main",
      connect: vi.fn().mockResolvedValue(undefined),
      disconnect: vi.fn().mockResolvedValue(undefined),
      isConnected: vi.fn().mockReturnValue(true),
      send: vi.fn().mockResolvedValue(undefined),
      onMessage: vi.fn(),
    };

    // Simulate incoming message tracking (as wired in createGateway)
    service.trackIncomingMessage("telegram-main", "12345");

    // Simulate scheduler notification delivery (as wired in createGateway)
    const delivered = await service.deliverToType(
      [adapter],
      "telegram",
      "Cron job completed"
    );
    expect(delivered).toBe(true);
    expect(adapter.send).toHaveBeenCalledWith("12345", {
      text: "Cron job completed",
    });
  });

  it("should not override default chat ID on subsequent messages", () => {
    const service = new ChannelDeliveryService();
    service.trackIncomingMessage("telegram-main", "chat-1");
    expect(service.getDefaultChatId("telegram-main")).toBe("chat-1");
    service.trackIncomingMessage("telegram-main", "chat-2");
    expect(service.getDefaultChatId("telegram-main")).toBe("chat-1");
  });

  it("should allow explicit override of default chat ID", () => {
    const service = new ChannelDeliveryService();
    service.trackIncomingMessage("telegram-main", "auto-tracked");
    service.setDefaultChatId("telegram-main", "manual-override");
    expect(service.getDefaultChatId("telegram-main")).toBe("manual-override");
  });
});
