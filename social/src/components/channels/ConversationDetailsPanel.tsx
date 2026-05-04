"use client";

import { useEffect, useState } from "react";
import {
  Bell,
  Check,
  FileText,
  Hash,
  Info,
  Pencil,
  Pin,
  UserRound,
  Users,
  X,
  type LucideIcon,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";

type ConversationDetailsPanelProps = {
  channelId: string;
  title: string;
  description?: string;
  type: "channel" | "dm";
  channelVisibility?: "PUBLIC" | "PRIVATE";
  memberCount?: number;
  canEditTopic?: boolean;
  onClose: () => void;
  onTopicChange?: (description?: string) => void;
  onOpenMembers?: () => void;
  onOpenPins?: () => void;
  onOpenFiles?: () => void;
  onOpenNotifications?: () => void;
};

const formatMembers = (memberCount?: number) => {
  if (memberCount === undefined) return null;
  return `${memberCount} member${memberCount === 1 ? "" : "s"}`;
};

const formatCount = (count: number | null, singular: string, plural: string) => {
  if (count === null) return "Loading";
  return `${count} ${count === 1 ? singular : plural}`;
};

export default function ConversationDetailsPanel({
  channelId,
  title,
  description,
  type,
  channelVisibility = "PUBLIC",
  memberCount,
  canEditTopic = false,
  onClose,
  onTopicChange,
  onOpenMembers,
  onOpenPins,
  onOpenFiles,
  onOpenNotifications,
}: ConversationDetailsPanelProps) {
  const [pinCount, setPinCount] = useState<number | null>(null);
  const [fileCount, setFileCount] = useState<number | null>(null);
  const [currentDescription, setCurrentDescription] = useState(description || "");
  const [topicDraft, setTopicDraft] = useState(description || "");
  const [editingTopic, setEditingTopic] = useState(false);
  const [savingTopic, setSavingTopic] = useState(false);
  const [topicError, setTopicError] = useState<string | null>(null);
  const isDm = type === "dm";
  const Icon = isDm ? UserRound : Hash;
  const membersLabel = formatMembers(memberCount);
  const topicText = currentDescription.trim();

  useEffect(() => {
    let cancelled = false;

    setPinCount(null);
    setFileCount(null);

    Promise.all([
      fetch(apiUrl(`/api/channels/${channelId}/pins`))
        .then((response) => (response.ok ? response.json() : { pins: [] }))
        .catch(() => ({ pins: [] })),
      fetch(apiUrl(`/api/channels/${channelId}/files`))
        .then((response) => (response.ok ? response.json() : { files: [] }))
        .catch(() => ({ files: [] })),
    ]).then(([pinsData, filesData]) => {
      if (cancelled) return;
      setPinCount(Array.isArray(pinsData.pins) ? pinsData.pins.length : 0);
      setFileCount(Array.isArray(filesData.files) ? filesData.files.length : 0);
    });

    return () => {
      cancelled = true;
    };
  }, [channelId]);

  useEffect(() => {
    const nextDescription = description || "";
    setCurrentDescription(nextDescription);
    setTopicDraft(nextDescription);
    setEditingTopic(false);
    setTopicError(null);
  }, [description]);

  const handleStartTopicEdit = () => {
    setTopicDraft(currentDescription);
    setTopicError(null);
    setEditingTopic(true);
  };

  const handleCancelTopicEdit = () => {
    setTopicDraft(currentDescription);
    setTopicError(null);
    setEditingTopic(false);
  };

  const handleSaveTopic = async () => {
    if (!canEditTopic || savingTopic) return;

    setSavingTopic(true);
    setTopicError(null);

    try {
      const response = await fetch(apiUrl(`/api/channels/${channelId}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: title,
          description: topicDraft,
          type: channelVisibility,
        }),
      });
      if (!response.ok) throw new Error("Failed to update topic");

      const updated = (await response.json()) as { description?: string | null };
      const nextDescription = updated.description || "";
      setCurrentDescription(nextDescription);
      setTopicDraft(nextDescription);
      setEditingTopic(false);
      onTopicChange?.(nextDescription || undefined);
    } catch {
      setTopicError("Topic could not be saved.");
    } finally {
      setSavingTopic(false);
    }
  };

  return (
    <div
      data-testid="conversation-details-panel"
      className="absolute right-4 top-14 z-40 w-80 overflow-hidden rounded-xl border border-border bg-bg-surface shadow-xl"
    >
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <Info size={14} className="shrink-0 text-accent" />
          <span className="truncate font-heading text-sm font-semibold text-text-primary">
            Details
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-text-muted hover:bg-bg-hover"
          aria-label="Close details"
          title="Close"
        >
          <X size={14} />
        </button>
      </div>

      <div className="border-b border-border px-4 py-4">
        <div className="flex items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-border bg-bg-elevated text-text-secondary">
            <Icon size={18} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="truncate font-heading text-base font-semibold text-text-primary">
              {isDm ? title : `#${title}`}
            </div>
            {isDm && currentDescription && (
              <div className="mt-1 text-sm leading-5 text-text-secondary">
                {currentDescription}
              </div>
            )}
            {membersLabel && (
              <div className="mt-1 text-xs text-text-muted">{membersLabel}</div>
            )}
          </div>
        </div>
      </div>

      {!isDm && (
        <div className="border-b border-border px-4 py-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-xs font-semibold uppercase text-text-muted">
              Topic
            </span>
            {canEditTopic && !editingTopic && (
              <button
                type="button"
                data-testid="conversation-details-topic-edit"
                onClick={handleStartTopicEdit}
                className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-xs font-medium text-text-muted hover:bg-bg-hover hover:text-text-primary"
              >
                <Pencil size={12} />
                <span>Edit</span>
              </button>
            )}
          </div>

          {editingTopic ? (
            <div className="space-y-2">
              <textarea
                data-testid="conversation-details-topic-input"
                value={topicDraft}
                onChange={(event) => setTopicDraft(event.target.value)}
                className="min-h-20 w-full resize-none rounded-lg border border-border bg-bg-elevated px-3 py-2 text-sm text-text-primary outline-none focus:border-accent"
                placeholder="What is this channel about?"
                autoFocus
              />
              {topicError && (
                <div className="text-xs text-danger">{topicError}</div>
              )}
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  data-testid="conversation-details-topic-cancel"
                  onClick={handleCancelTopicEdit}
                  disabled={savingTopic}
                  className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-xs font-medium text-text-muted hover:bg-bg-hover hover:text-text-primary disabled:opacity-60"
                >
                  <X size={12} />
                  <span>Cancel</span>
                </button>
                <button
                  type="button"
                  data-testid="conversation-details-topic-save"
                  onClick={handleSaveTopic}
                  disabled={savingTopic}
                  className="inline-flex items-center gap-1 rounded-md bg-accent px-2 py-1.5 text-xs font-semibold text-black transition-colors hover:bg-accent-hover disabled:opacity-60"
                >
                  <Check size={12} />
                  <span>{savingTopic ? "Saving..." : "Save"}</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="text-sm leading-5 text-text-secondary">
              {topicText || "No topic set"}
            </div>
          )}
        </div>
      )}

      <div className="space-y-1 p-2">
        {onOpenMembers && (
          <DetailsAction
            icon={Users}
            label="Members"
            meta={membersLabel || undefined}
            testId="conversation-details-members"
            onClick={onOpenMembers}
          />
        )}
        {onOpenPins && (
          <DetailsAction
            icon={Pin}
            label="Pinned messages"
            meta={formatCount(pinCount, "pin", "pins")}
            testId="conversation-details-pins"
            onClick={onOpenPins}
          />
        )}
        {onOpenFiles && (
          <DetailsAction
            icon={FileText}
            label="Files"
            meta={formatCount(fileCount, "file", "files")}
            testId="conversation-details-files"
            onClick={onOpenFiles}
          />
        )}
        {onOpenNotifications && (
          <DetailsAction
            icon={Bell}
            label="Notifications"
            testId="conversation-details-notifications"
            onClick={onOpenNotifications}
          />
        )}
      </div>
    </div>
  );
}

function DetailsAction({
  icon: Icon,
  label,
  meta,
  testId,
  onClick,
}: {
  icon: LucideIcon;
  label: string;
  meta?: string;
  testId: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      data-testid={testId}
      onClick={onClick}
      className="flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-sm text-text-secondary transition-colors hover:bg-bg-hover hover:text-text-primary"
    >
      <Icon size={15} className="shrink-0" />
      <span className="min-w-0 flex-1 truncate">{label}</span>
      {meta && (
        <span
          data-testid={`${testId}-count`}
          className="shrink-0 text-2xs text-text-muted"
        >
          {meta}
        </span>
      )}
    </button>
  );
}
