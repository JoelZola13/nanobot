"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  AtSign,
  Bell,
  BellOff,
  Check,
  Loader2,
  RefreshCw,
  X,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";
import type { NotificationLevel } from "@/lib/notificationPreferences";

type NotificationPreference = {
  channelId: string;
  channelName: string | null;
  channelType: "PUBLIC" | "PRIVATE" | "DM" | "GROUP_DM";
  level: NotificationLevel;
  mutedAt: string | null;
};

const OPTIONS: {
  level: NotificationLevel;
  label: string;
  icon: typeof Bell;
}[] = [
  { level: "ALL", label: "All activity", icon: Bell },
  { level: "MENTIONS", label: "Mentions", icon: AtSign },
  { level: "MUTED", label: "Muted", icon: BellOff },
];

async function getApiErrorMessage(response: Response, fallback: string) {
  const payload = (await response.json().catch(() => null)) as {
    error?: unknown;
    message?: unknown;
  } | null;

  if (typeof payload?.error === "string" && payload.error.trim()) {
    return payload.error;
  }

  if (typeof payload?.message === "string" && payload.message.trim()) {
    return payload.message;
  }

  return fallback;
}

export default function NotificationPreferencesPanel({
  channelId,
  onClose,
  onLevelChange,
  className,
}: {
  channelId: string;
  onClose: () => void;
  onLevelChange?: (level: NotificationLevel) => void;
  className?: string;
}) {
  const [preference, setPreference] = useState<NotificationPreference | null>(
    null,
  );
  const [loading, setLoading] = useState(true);
  const [savingLevel, setSavingLevel] = useState<NotificationLevel | null>(
    null,
  );
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const onLevelChangeRef = useRef(onLevelChange);

  useEffect(() => {
    onLevelChangeRef.current = onLevelChange;
  }, [onLevelChange]);

  const loadPreference = useCallback(
    async (signal?: AbortSignal) => {
      setLoading(true);
      setPreference(null);
      setLoadError(null);
      setSaveError(null);

      try {
        const response = await fetch(
          apiUrl(`/api/channels/${channelId}/notifications`),
          {
            cache: "no-store",
            signal,
          },
        );
        if (!response.ok) {
          throw new Error(
            await getApiErrorMessage(
              response,
              "Notification preferences could not load.",
            ),
          );
        }

        const data = (await response.json()) as NotificationPreference;
        if (signal?.aborted) return;
        setPreference(data);
        onLevelChangeRef.current?.(data.level);
      } catch (error) {
        if (signal?.aborted) return;
        setLoadError(
          error instanceof Error
            ? error.message
            : "Notification preferences could not load.",
        );
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [channelId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadPreference(controller.signal);

    return () => {
      controller.abort();
    };
  }, [loadPreference]);

  const updatePreference = async (level: NotificationLevel) => {
    if (savingLevel || loading || loadError || preference?.level === level)
      return;

    setSavingLevel(level);
    setSaveError(null);

    try {
      const response = await fetch(
        apiUrl(`/api/channels/${channelId}/notifications`),
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ level }),
        },
      );
      if (!response.ok) {
        throw new Error(
          await getApiErrorMessage(
            response,
            "Notification preference could not be saved.",
          ),
        );
      }

      const updatedPreference =
        (await response.json()) as NotificationPreference;
      setPreference(updatedPreference);
      onLevelChangeRef.current?.(updatedPreference.level);
      window.dispatchEvent(
        new CustomEvent("social:notification-preference", {
          detail: {
            channelId,
            level: updatedPreference.level,
          },
        }),
      );
    } catch (error) {
      setSaveError(
        error instanceof Error
          ? error.message
          : "Notification preference could not be saved.",
      );
    } finally {
      setSavingLevel(null);
    }
  };

  return (
    <div
      data-testid="notification-preferences-panel"
      role="dialog"
      aria-label="Notification preferences"
      className={
        className ||
        "absolute right-4 top-14 z-40 w-80 overflow-hidden rounded-xl border border-border bg-bg-surface shadow-xl"
      }
    >
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <Bell size={14} className="text-accent" />
          <span className="font-heading text-sm font-semibold text-text-primary">
            Notifications
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-text-muted hover:bg-bg-hover"
          aria-label="Close notification preferences"
          title="Close"
        >
          <X size={14} />
        </button>
      </div>

      <div className="space-y-1 p-2">
        {loading && (
          <div
            className="flex items-center justify-center gap-2 px-3 py-4 text-sm text-text-muted"
            role="status"
            aria-label="Loading notification preferences"
          >
            <Loader2 size={14} className="animate-spin" />
            Loading
          </div>
        )}

        {!loading && loadError && (
          <div
            className="rounded-lg border border-red-300 bg-red-50 px-3 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
            data-testid="notification-preferences-load-error"
            role="alert"
          >
            <div className="flex items-start gap-2">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span className="min-w-0 flex-1">{loadError}</span>
            </div>
            <button
              type="button"
              onClick={() => void loadPreference()}
              className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-2.5 py-1.5 text-xs font-medium text-red-700 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-400 dark:border-red-800 dark:bg-red-950/20 dark:text-red-100 dark:hover:bg-red-950/50"
              aria-label="Retry notification preferences"
            >
              <RefreshCw size={12} />
              Retry
            </button>
          </div>
        )}

        {!loading &&
          !loadError &&
          OPTIONS.map((option) => {
            const Icon = option.icon;
            const active = preference?.level === option.level;
            const saving = savingLevel === option.level;

            return (
              <button
                key={option.level}
                type="button"
                data-testid={`notification-preference-${option.level.toLowerCase()}`}
                onClick={() => updatePreference(option.level)}
                disabled={Boolean(savingLevel)}
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                  active
                    ? "bg-accent-muted text-accent"
                    : "text-text-secondary hover:bg-bg-hover hover:text-text-primary"
                } disabled:opacity-70`}
                aria-pressed={active}
                aria-label={`${option.label} notifications${active ? ", selected" : ""}`}
              >
                <Icon size={15} className="shrink-0" />
                <span className="min-w-0 flex-1 truncate">
                  {saving ? "Saving..." : option.label}
                </span>
                {saving ? (
                  <Loader2 size={14} className="shrink-0 animate-spin" />
                ) : active ? (
                  <Check size={14} className="shrink-0" />
                ) : null}
              </button>
            );
          })}

        {!loading && !loadError && saveError && (
          <div
            className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
            data-testid="notification-preferences-save-error"
            role="alert"
          >
            {saveError}
          </div>
        )}
      </div>
    </div>
  );
}
