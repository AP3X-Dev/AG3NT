/**
 * Tests for ChannelDeliveryService.
 *
 * Validates that scheduled notifications can be delivered through
 * channel adapters (Telegram, Discord, Slack) by tracking default
 * chat IDs and dispatching messages.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { ChannelDeliveryService } from "./ChannelDeliveryService.js";
import type { IChannelAdapter } from "./types.js";

/** Create a mock adapter with vi.fn() stubs for all IChannelAdapter methods. */
function createMockAdapter(
  id: string,
  type: string = "test",
  connected: boolean = true
): IChannelAdapter {
  return {
    id,
    type,
    connect: vi.fn(async () => {}),
    disconnect: vi.fn(async () => {}),
    isConnected: vi.fn(() => connected),
    send: vi.fn(async () => {}),
    onMessage: vi.fn(),
  };
}

describe("ChannelDeliveryService", () => {
  let service: ChannelDeliveryService;

  beforeEach(() => {
    service = new ChannelDeliveryService();
  });

  it("should register a default chat ID for an adapter", () => {
    service.setDefaultChatId("telegram-1", "chat-123");
    expect(service.getDefaultChatId("telegram-1")).toBe("chat-123");
  });

  it("should deliver to adapter when chat ID is known", async () => {
    const adapter = createMockAdapter("telegram-1", "telegram");
    service.setDefaultChatId("telegram-1", "chat-123");

    const result = await service.deliver(adapter, "Hello from scheduler");

    expect(result).toBe(true);
    expect(adapter.send).toHaveBeenCalledWith("chat-123", {
      text: "Hello from scheduler",
    });
  });

  it("should return false when no chat ID is registered", async () => {
    const adapter = createMockAdapter("telegram-1", "telegram");

    const result = await service.deliver(adapter, "Hello");

    expect(result).toBe(false);
    expect(adapter.send).not.toHaveBeenCalled();
  });

  it("should deliver to adapter by type", async () => {
    const telegram = createMockAdapter("telegram-1", "telegram");
    const slack = createMockAdapter("slack-1", "slack");
    service.setDefaultChatId("telegram-1", "chat-tg");
    service.setDefaultChatId("slack-1", "chat-sl");

    const result = await service.deliverToType(
      [telegram, slack],
      "telegram",
      "Notification"
    );

    expect(result).toBe(true);
    expect(telegram.send).toHaveBeenCalledWith("chat-tg", {
      text: "Notification",
    });
    expect(slack.send).not.toHaveBeenCalled();
  });

  it("should deliver to first connected adapter when no type specified", async () => {
    const disconnected = createMockAdapter("disc-1", "discord", false);
    const connected = createMockAdapter("telegram-1", "telegram", true);
    service.setDefaultChatId("telegram-1", "chat-tg");

    const result = await service.deliverToType(
      [disconnected, connected],
      undefined,
      "Fallback notification"
    );

    expect(result).toBe(true);
    expect(connected.send).toHaveBeenCalledWith("chat-tg", {
      text: "Fallback notification",
    });
    expect(disconnected.send).not.toHaveBeenCalled();
  });

  it("should handle adapter send failure gracefully", async () => {
    const adapter = createMockAdapter("telegram-1", "telegram");
    (adapter.send as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error("Network timeout")
    );
    service.setDefaultChatId("telegram-1", "chat-123");

    const result = await service.deliver(adapter, "Will fail");

    expect(result).toBe(false);
  });

  it("should track chat IDs from incoming messages (first-wins)", () => {
    // First message sets the chat ID
    service.trackIncomingMessage("telegram-1", "chat-100");
    expect(service.getDefaultChatId("telegram-1")).toBe("chat-100");

    // Second message from a different chat does NOT override
    service.trackIncomingMessage("telegram-1", "chat-200");
    expect(service.getDefaultChatId("telegram-1")).toBe("chat-100");

    // Explicit set DOES override
    service.setDefaultChatId("telegram-1", "chat-300");
    expect(service.getDefaultChatId("telegram-1")).toBe("chat-300");
  });
});
