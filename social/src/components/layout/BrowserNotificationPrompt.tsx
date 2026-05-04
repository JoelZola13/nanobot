"use client";

import { useEffect, useState } from "react";
import { Bell, X } from "lucide-react";
import {
  dismissBrowserNotificationPrompt,
  requestBrowserNotificationPermission,
  shouldShowBrowserNotificationPrompt,
} from "@/lib/browserNotifications";

export default function BrowserNotificationPrompt() {
  const [visible, setVisible] = useState(false);
  const [requesting, setRequesting] = useState(false);

  useEffect(() => {
    setVisible(shouldShowBrowserNotificationPrompt());
  }, []);

  if (!visible) return null;

  const dismiss = () => {
    dismissBrowserNotificationPrompt();
    setVisible(false);
  };

  const enable = async () => {
    setRequesting(true);
    const permission = await requestBrowserNotificationPermission();
    setRequesting(false);
    if (permission !== "default") setVisible(false);
  };

  return (
    <div
      className="mx-2 rounded-md border px-2.5 py-2"
      style={{
        background: "var(--sv-sidebar-elevated)",
        borderColor: "var(--sv-sidebar-border)",
      }}
    >
      <div className="flex items-start gap-2">
        <Bell size={14} className="mt-0.5 shrink-0 text-accent" />
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold">Desktop alerts</div>
          <div className="mt-0.5 text-2xs leading-snug" style={{ color: "var(--sv-sidebar-muted)" }}>
            Get notified while Messages is in the background.
          </div>
          <button
            type="button"
            onClick={() => void enable()}
            disabled={requesting}
            className="mt-2 rounded-md bg-accent px-2 py-1 text-2xs font-semibold text-white transition-colors hover:bg-accent-hover disabled:opacity-60"
          >
            {requesting ? "Waiting..." : "Enable"}
          </button>
        </div>
        <button
          type="button"
          onClick={dismiss}
          className="sidebar-icon-button h-6 w-6 shrink-0"
          title="Dismiss desktop alerts prompt"
          aria-label="Dismiss desktop alerts prompt"
        >
          <X size={13} />
        </button>
      </div>
    </div>
  );
}
