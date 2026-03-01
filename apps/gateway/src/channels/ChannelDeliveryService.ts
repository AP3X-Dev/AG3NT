/**
 * ChannelDeliveryService — Deliver Scheduled Notifications via Channel Adapters.
 *
 * Tracks default chat IDs per channel adapter and delivers scheduled
 * notifications (cron job results, heartbeat alerts) through the actual
 * channel adapters (Telegram, Discord, Slack) instead of only through
 * the SSE bus.
 *
 * Usage:
 *   1. When a user sends a message via Telegram/Discord/Slack, call
 *      trackIncomingMessage() to record their chatId as the default
 *      delivery target for that adapter.
 *   2. When the scheduler needs to send a notification, call deliver()
 *      or deliverToType() which look up the default chat ID and send
 *      the message through the adapter.
 *   3. If no chat ID is registered for the target adapter, delivery
 *      returns false (SSE-only fallback).
 */

import type { IChannelAdapter } from "./types.js";

export class ChannelDeliveryService {
  /** Map from adapter ID to default chat ID for delivery. */
  private defaultChatIds: Map<string, string> = new Map();

  /**
   * Explicitly set (or override) the default chat ID for an adapter.
   * Always sets, even if a chat ID was previously tracked.
   */
  setDefaultChatId(adapterId: string, chatId: string): void {
    this.defaultChatIds.set(adapterId, chatId);
  }

  /**
   * Get the default chat ID registered for an adapter.
   * Returns undefined if no chat ID has been set or tracked.
   */
  getDefaultChatId(adapterId: string): string | undefined {
    return this.defaultChatIds.get(adapterId);
  }

  /**
   * Track chat ID from an incoming message — first-wins semantics.
   * Sets the default chat ID only if one is not already registered
   * for the given adapter. This ensures a stable delivery target.
   */
  trackIncomingMessage(adapterId: string, chatId: string): void {
    if (!this.defaultChatIds.has(adapterId)) {
      this.defaultChatIds.set(adapterId, chatId);
    }
  }

  /**
   * Deliver a message through a specific adapter.
   *
   * Looks up the default chat ID for the adapter's ID, then calls
   * adapter.send(). Returns true on success, false if no chat ID
   * is registered or if the send fails.
   */
  async deliver(adapter: IChannelAdapter, message: string): Promise<boolean> {
    const chatId = this.defaultChatIds.get(adapter.id);
    if (!chatId) {
      return false;
    }

    try {
      await adapter.send(chatId, { text: message });
      return true;
    } catch (error) {
      // Log but do not throw — caller can fall back to SSE.
      console.error(
        `[ChannelDeliveryService] Failed to deliver to adapter ${adapter.id}:`,
        error
      );
      return false;
    }
  }

  /**
   * Deliver a message to an adapter matching the given channel type.
   *
   * - If channelType is provided, finds an adapter whose type or id
   *   matches (type match first, then id match).
   * - If channelType is undefined, finds the first connected adapter
   *   that has a known default chat ID.
   *
   * Returns true on successful delivery, false otherwise.
   */
  async deliverToType(
    adapters: IChannelAdapter[],
    channelType: string | undefined,
    message: string
  ): Promise<boolean> {
    let target: IChannelAdapter | undefined;

    if (channelType) {
      // Try type match first, then id match
      target =
        adapters.find((a) => a.type === channelType) ??
        adapters.find((a) => a.id === channelType);
    } else {
      // No type specified — find first connected adapter with a known chat ID
      target = adapters.find(
        (a) => a.isConnected() && this.defaultChatIds.has(a.id)
      );
    }

    if (!target) {
      return false;
    }

    return this.deliver(target, message);
  }
}
