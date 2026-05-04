"use client";

import { useEffect, useState } from "react";
import { AtSign, Bell, BellOff, Check, X } from "lucide-react";
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

export default function NotificationPreferencesPanel({
  channelId,
  onClose,
  onLevelChange,
}: {
  channelId: string;
  onClose: () => void;
  onLevelChange?: (level: NotificationLevel) => void;
}) {
  const [preference, setPreference] = useState<NotificationPreference | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingLevel, setSavingLevel] = useState<NotificationLevel | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);

    fetch(apiUrl(`/api/channels/${channelId}/notifications`))
      .then((response) => {
        if (!response.ok) throw new Error("Failed to load notification preferences");
        return response.json();
      })
      .then((data: NotificationPreference) => {
        if (cancelled) return;
        setPreference(data);
        onLevelChange?.(data.level);
      })
      .catch(() => {
        if (!cancelled) setError("Notification preferences could not load.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [channelId, onLevelChange]);

  const updatePreference = async (level: NotificationLevel) => {
    if (savingLevel || preference?.level === level) return;

    setSavingLevel(level);
    setError(null);

    try {
      const response = await fetch(apiUrl(`/api/channels/${channelId}/notifications`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ level }),
      });
      if (!response.ok) throw new Error("Failed to save notification preference");

      const updatedPreference = (await response.json()) as NotificationPreference;
      setPreference(updatedPreference);
      onLevelChange?.(updatedPreference.level);
      window.dispatchEvent(
        new CustomEvent("social:notification-preference", {
          detail: {
            channelId,
            level: updatedPreference.level,
          },
        }),
      );
    } catch {
      setError("Notification preference could not be saved.");
    } finally {
      setSavingLevel(null);
    }
  };

  return (
    <div className="absolute right-4 top-14 z-40 w-80 overflow-hidden rounded-xl border border-border bg-bg-surface shadow-xl">
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
          <div className="px-3 py-4 text-center text-sm text-text-muted">
            Loading...
          </div>
        )}

        {!loading &&
          OPTIONS.map((option) => {
            const Icon = option.icon;
            const active = preference?.level === option.level;
            const saving = savingLevel === option.level;

            return (
              <button
                key={option.level}
                type="button"
                onClick={() => updatePreference(option.level)}
                disabled={Boolean(savingLevel)}
                className={`flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                  active
                    ? "bg-accent-muted text-accent"
                    : "text-text-secondary hover:bg-bg-hover hover:text-text-primary"
                } disabled:opacity-70`}
                aria-pressed={active}
              >
                <Icon size={15} className="shrink-0" />
                <span className="min-w-0 flex-1 truncate">
                  {saving ? "Saving..." : option.label}
                </span>
                {active && <Check size={14} className="shrink-0" />}
              </button>
            );
          })}

        {error && (
          <div className="rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
