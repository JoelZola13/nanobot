"use client";

import { useEffect, useState } from "react";
import { AlertCircle, Bell, Check, Loader2, X } from "lucide-react";
import {
  dismissBrowserNotificationPrompt,
  requestBrowserNotificationPermission,
  shouldShowBrowserNotificationPrompt,
} from "@/lib/browserNotifications";

export default function BrowserNotificationPrompt() {
  const [visible, setVisible] = useState(false);
  const [requesting, setRequesting] = useState(false);
  const [status, setStatus] = useState<{
    kind: "error" | "success";
    message: string;
  } | null>(null);

  useEffect(() => {
    setVisible(shouldShowBrowserNotificationPrompt());
  }, []);

  if (!visible) return null;

  const dismiss = () => {
    dismissBrowserNotificationPrompt();
    setStatus(null);
    setVisible(false);
  };

  const enable = async () => {
    setRequesting(true);
    setStatus(null);

    try {
      const permission = await requestBrowserNotificationPermission();
      if (permission === "granted") {
        dismissBrowserNotificationPrompt();
        setStatus({
          kind: "success",
          message: "Desktop alerts are enabled.",
        });
        setVisible(false);
        return;
      }

      if (permission === "denied") {
        setStatus({
          kind: "error",
          message: "Desktop alerts are blocked in this browser.",
        });
        return;
      }

      if (permission === "unsupported") {
        setStatus({
          kind: "error",
          message: "Desktop alerts are not supported in this browser.",
        });
        return;
      }

      setStatus({
        kind: "error",
        message: "Desktop alerts were not enabled.",
      });
    } catch {
      setStatus({
        kind: "error",
        message: "Desktop alerts could not be enabled.",
      });
    } finally {
      setRequesting(false);
    }
  };

  return (
    <div
      data-testid="browser-notification-prompt"
      role="region"
      aria-label="Desktop alerts prompt"
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
          <div
            className="mt-0.5 text-2xs leading-snug"
            style={{ color: "var(--sv-sidebar-muted)" }}
          >
            Get notified while Messages is in the background.
          </div>
          {status && (
            <div
              data-testid="browser-notification-prompt-status"
              role={status.kind === "error" ? "alert" : "status"}
              className={`mt-2 flex items-start gap-1.5 rounded-md border px-2 py-1.5 text-2xs leading-snug ${
                status.kind === "error"
                  ? "border-red-300 bg-red-50 text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
                  : "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-900/50 dark:bg-emerald-950/30 dark:text-emerald-200"
              }`}
            >
              {status.kind === "error" ? (
                <AlertCircle size={12} className="mt-0.5 shrink-0" />
              ) : (
                <Check size={12} className="mt-0.5 shrink-0" />
              )}
              <span className="min-w-0 flex-1">{status.message}</span>
            </div>
          )}
          <button
            type="button"
            onClick={() => void enable()}
            disabled={requesting}
            className="mt-2 inline-flex items-center gap-1.5 rounded-md bg-accent px-2 py-1 text-2xs font-semibold text-black transition-colors hover:bg-accent-hover disabled:opacity-60"
            style={{ color: "#000" }}
            aria-label="Enable desktop alerts"
          >
            {requesting && <Loader2 size={11} className="animate-spin" />}
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
