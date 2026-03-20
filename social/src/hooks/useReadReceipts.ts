"use client";

import { useEffect, useRef, useCallback } from "react";
import { apiUrl } from "@/lib/apiUrl";

interface ReadReceiptData {
  userId: string;
  messageId: string;
  readAt: string;
  user: { id: string; displayName: string; avatarUrl: string | null };
}

/**
 * Sends read receipts when:
 * 1. The channel first loads (marks last message as read)
 * 2. New messages arrive while the window is focused
 * 3. The window regains focus after being blurred
 */
export function useReadReceipts(
  channelId: string,
  lastMessageId: string | null,
  _currentUserId: string,
) {
  const lastSentRef = useRef<string | null>(null);

  const sendReadReceipt = useCallback(
    async (messageId: string) => {
      if (!messageId || messageId === lastSentRef.current) return;
      lastSentRef.current = messageId;

      try {
        await fetch(apiUrl(`/api/channels/${channelId}/read`), {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messageId }),
        });
      } catch {
        // Silently fail — read receipts are best-effort
      }
    },
    [channelId],
  );

  // Send on mount and when lastMessageId changes (new messages)
  useEffect(() => {
    if (lastMessageId && document.hasFocus()) {
      sendReadReceipt(lastMessageId);
    }
  }, [lastMessageId, sendReadReceipt]);

  // Send when window regains focus
  useEffect(() => {
    const handleFocus = () => {
      if (lastMessageId) {
        sendReadReceipt(lastMessageId);
      }
    };
    window.addEventListener("focus", handleFocus);
    return () => window.removeEventListener("focus", handleFocus);
  }, [lastMessageId, sendReadReceipt]);

  return { sendReadReceipt };
}

/**
 * Fetches initial read receipts for a channel.
 * Returns a Map<messageId, userId[]> showing who has read up to each message.
 */
export async function fetchReadReceipts(channelId: string): Promise<ReadReceiptData[]> {
  try {
    const res = await fetch(apiUrl(`/api/channels/${channelId}/read`));
    if (res.ok) return res.json();
  } catch {
    // best-effort
  }
  return [];
}
