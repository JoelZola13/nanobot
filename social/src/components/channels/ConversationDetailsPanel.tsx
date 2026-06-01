"use client";

import { useCallback, useEffect, useState } from "react";
import {
  AlertCircle,
  Bell,
  Check,
  ExternalLink,
  File,
  FileText,
  Hash,
  Info,
  Image,
  Loader2,
  Music,
  Pencil,
  Pin,
  RefreshCw,
  UserRound,
  Users,
  Video,
  X,
  type LucideIcon,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";

type ChannelFile = {
  id: string;
  fileName: string;
  fileSize: number;
  mimeType: string;
  url: string;
  href: string;
};

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

const formatCount = (
  count: number | null,
  singular: string,
  plural: string,
) => {
  if (count === null) return "Loading";
  return `${count} ${count === 1 ? singular : plural}`;
};

const iconForFile = (mimeType: string) => {
  if (mimeType.startsWith("image/")) return Image;
  if (mimeType.startsWith("audio/")) return Music;
  if (mimeType.startsWith("video/")) return Video;
  if (mimeType.includes("pdf") || mimeType.includes("text")) return FileText;
  return File;
};

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
  const [sharedFiles, setSharedFiles] = useState<ChannelFile[] | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [currentDescription, setCurrentDescription] = useState(
    description || "",
  );
  const [topicDraft, setTopicDraft] = useState(description || "");
  const [editingTopic, setEditingTopic] = useState(false);
  const [savingTopic, setSavingTopic] = useState(false);
  const [topicError, setTopicError] = useState<string | null>(null);
  const isDm = type === "dm";
  const Icon = isDm ? UserRound : Hash;
  const membersLabel = formatMembers(memberCount);
  const topicText = currentDescription.trim();
  const sharedPreviewFiles = sharedFiles?.slice(0, 4) || [];

  const loadSummary = useCallback(
    async (signal?: AbortSignal) => {
      setPinCount(null);
      setFileCount(null);
      setSharedFiles(null);
      setSummaryError(null);

      try {
        const [pinsResponse, filesResponse] = await Promise.all([
          fetch(apiUrl(`/api/channels/${channelId}/pins`), {
            cache: "no-store",
            signal,
          }),
          fetch(apiUrl(`/api/channels/${channelId}/files`), {
            cache: "no-store",
            signal,
          }),
        ]);

        if (!pinsResponse.ok) {
          throw new Error(
            await getApiErrorMessage(
              pinsResponse,
              "Conversation summary could not load.",
            ),
          );
        }

        if (!filesResponse.ok) {
          throw new Error(
            await getApiErrorMessage(
              filesResponse,
              "Conversation summary could not load.",
            ),
          );
        }

        const [pinsData, filesData] = (await Promise.all([
          pinsResponse.json(),
          filesResponse.json(),
        ])) as [{ pins?: unknown[] }, { files?: ChannelFile[] }];
        if (signal?.aborted) return;
        const files = Array.isArray(filesData.files) ? filesData.files : [];
        setPinCount(Array.isArray(pinsData.pins) ? pinsData.pins.length : 0);
        setFileCount(files.length);
        setSharedFiles(files);
      } catch (error) {
        if (signal?.aborted) return;
        setPinCount(null);
        setFileCount(null);
        setSharedFiles([]);
        setSummaryError(
          error instanceof Error
            ? error.message
            : "Conversation summary could not load.",
        );
      }
    },
    [channelId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadSummary(controller.signal);

    return () => {
      controller.abort();
    };
  }, [loadSummary]);

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
      if (!response.ok) {
        throw new Error(
          await getApiErrorMessage(response, "Topic could not be saved."),
        );
      }

      const updated = (await response.json()) as {
        description?: string | null;
      };
      const nextDescription = updated.description || "";
      setCurrentDescription(nextDescription);
      setTopicDraft(nextDescription);
      setEditingTopic(false);
      onTopicChange?.(nextDescription || undefined);
    } catch (error) {
      setTopicError(
        error instanceof Error ? error.message : "Topic could not be saved.",
      );
    } finally {
      setSavingTopic(false);
    }
  };

  return (
    <div
      data-testid="conversation-details-panel"
      role="dialog"
      aria-label="Conversation details"
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
                aria-label="Edit channel topic"
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
                aria-label="Channel topic"
                autoFocus
              />
              {topicError && (
                <div
                  className="text-xs text-danger"
                  data-testid="conversation-details-topic-error"
                  role="alert"
                >
                  {topicError}
                </div>
              )}
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  data-testid="conversation-details-topic-cancel"
                  onClick={handleCancelTopicEdit}
                  disabled={savingTopic}
                  className="inline-flex items-center gap-1 rounded-md px-2 py-1.5 text-xs font-medium text-text-muted hover:bg-bg-hover hover:text-text-primary disabled:opacity-60"
                  aria-label="Cancel topic edit"
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
                  aria-label="Save channel topic"
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

      <div
        data-testid="conversation-details-media-strip"
        className="border-b border-border px-4 py-3"
      >
        <div className="mb-2 flex items-center justify-between gap-2">
          <span className="text-xs font-semibold uppercase text-text-muted">
            Shared media
          </span>
          {onOpenFiles && fileCount !== null && fileCount > 0 && (
            <button
              type="button"
              onClick={onOpenFiles}
              className="rounded-md px-1.5 py-1 text-xs font-medium text-text-muted hover:bg-bg-hover hover:text-text-primary"
              aria-label="View all shared files"
            >
              View all
            </button>
          )}
        </div>

        {summaryError ? (
          <div
            className="rounded-lg border border-red-300 bg-red-50 px-3 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
            data-testid="conversation-details-summary-error"
            role="alert"
          >
            <div className="flex items-start gap-2">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span className="min-w-0 flex-1">{summaryError}</span>
            </div>
            <button
              type="button"
              onClick={() => void loadSummary()}
              className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-2.5 py-1.5 text-xs font-medium text-red-700 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-400 dark:border-red-800 dark:bg-red-950/20 dark:text-red-100 dark:hover:bg-red-950/50"
              aria-label="Retry details summary"
            >
              <RefreshCw size={12} />
              Retry
            </button>
          </div>
        ) : sharedFiles === null ? (
          <div className="grid grid-cols-4 gap-2">
            {[0, 1, 2, 3].map((index) => (
              <div
                key={index}
                className="flex aspect-square items-center justify-center rounded-lg bg-bg-elevated text-text-muted"
                role="status"
                aria-label="Loading shared media"
              >
                <Loader2 size={14} className="animate-spin" />
              </div>
            ))}
          </div>
        ) : sharedPreviewFiles.length > 0 ? (
          <div className="grid grid-cols-4 gap-2">
            {sharedPreviewFiles.map((file) => (
              <SharedMediaPreview key={file.id} file={file} />
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-border px-3 py-4 text-center text-xs text-text-muted">
            Attachments shared here will appear in this strip.
          </div>
        )}
      </div>

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
            meta={
              summaryError
                ? "Unavailable"
                : formatCount(pinCount, "pin", "pins")
            }
            testId="conversation-details-pins"
            onClick={onOpenPins}
          />
        )}
        {onOpenFiles && (
          <DetailsAction
            icon={FileText}
            label="Files"
            meta={
              summaryError
                ? "Unavailable"
                : formatCount(fileCount, "file", "files")
            }
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

function SharedMediaPreview({ file }: { file: ChannelFile }) {
  const Icon = iconForFile(file.mimeType);
  const isImage = file.mimeType.startsWith("image/");

  return (
    <a
      href={file.url}
      target="_blank"
      rel="noreferrer"
      data-testid="conversation-details-media-item"
      className="group relative flex aspect-square min-w-0 items-center justify-center overflow-hidden rounded-lg border border-border bg-bg-elevated text-text-secondary transition-colors hover:border-accent hover:text-accent"
      title={file.fileName}
      aria-label={`Open ${file.fileName}`}
    >
      {isImage ? (
        <img
          src={file.url}
          alt={file.fileName}
          className="h-full w-full object-cover"
          loading="lazy"
        />
      ) : (
        <Icon size={20} />
      )}
      <div className="absolute inset-x-0 bottom-0 flex items-center gap-1 bg-black/60 px-1.5 py-1 text-[10px] font-medium text-white opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
        <span className="min-w-0 flex-1 truncate">{file.fileName}</span>
        <ExternalLink size={10} className="shrink-0" />
      </div>
    </a>
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
      aria-label={meta ? `${label}, ${meta}` : label}
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
