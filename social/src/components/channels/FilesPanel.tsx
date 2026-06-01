"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import {
  AlertCircle,
  Archive,
  Bot,
  ExternalLink,
  File,
  FileText,
  Image,
  Loader2,
  Music,
  RefreshCw,
  Video,
  X,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";
import { useEmbeddedNavigation } from "@/lib/useEmbeddedNavigation";

type ChannelFile = {
  id: string;
  fileName: string;
  fileSize: number;
  mimeType: string;
  url: string;
  width: number | null;
  height: number | null;
  messageId: string;
  channelId: string;
  createdAt: string;
  messageContent: string;
  href: string;
  author: {
    id: string;
    username: string;
    displayName: string;
    avatarUrl: string | null;
    isAgent: boolean;
  };
};

type FileFilter = "all" | "images" | "documents" | "audio" | "video";

const FILTERS: { id: FileFilter; label: string }[] = [
  { id: "all", label: "All" },
  { id: "images", label: "Images" },
  { id: "documents", label: "Docs" },
  { id: "audio", label: "Audio" },
  { id: "video", label: "Video" },
];

const formatFileSize = (bytes: number) => {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";

  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** exponent;

  return `${value >= 10 || exponent === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[exponent]}`;
};

const fileFilterFor = (mimeType: string): FileFilter => {
  if (mimeType.startsWith("image/")) return "images";
  if (mimeType.startsWith("audio/")) return "audio";
  if (mimeType.startsWith("video/")) return "video";
  return "documents";
};

const iconFor = (mimeType: string) => {
  if (mimeType.startsWith("image/")) return Image;
  if (mimeType.startsWith("audio/")) return Music;
  if (mimeType.startsWith("video/")) return Video;
  if (
    mimeType.includes("zip") ||
    mimeType.includes("tar") ||
    mimeType.includes("compressed")
  ) {
    return Archive;
  }
  if (mimeType.includes("pdf") || mimeType.includes("text")) return FileText;
  return File;
};

const apiErrorMessage = async (response: Response, fallback: string) => {
  const data = (await response.json().catch(() => null)) as {
    error?: string;
  } | null;

  return data?.error || fallback;
};

export default function FilesPanel({
  channelId,
  onClose,
}: {
  channelId: string;
  onClose: () => void;
}) {
  const [files, setFiles] = useState<ChannelFile[]>([]);
  const [activeFilter, setActiveFilter] = useState<FileFilter>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { withEmbed } = useEmbeddedNavigation();

  const loadFiles = useCallback(
    async (signal?: AbortSignal) => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch(
          apiUrl(`/api/channels/${channelId}/files`),
          {
            cache: "no-store",
            signal,
          },
        );
        if (!response.ok) {
          throw new Error(
            await apiErrorMessage(response, "Files could not load."),
          );
        }

        const data = (await response.json()) as { files?: ChannelFile[] };
        if (signal?.aborted) return;
        setFiles(data.files || []);
      } catch (loadError) {
        if (signal?.aborted) return;
        setFiles([]);
        setError(
          loadError instanceof Error
            ? loadError.message
            : "Files could not load.",
        );
      } finally {
        if (!signal?.aborted) setLoading(false);
      }
    },
    [channelId],
  );

  useEffect(() => {
    const controller = new AbortController();
    void loadFiles(controller.signal);

    return () => {
      controller.abort();
    };
  }, [loadFiles]);

  const filteredFiles = useMemo(() => {
    if (activeFilter === "all") return files;
    return files.filter(
      (file) => fileFilterFor(file.mimeType) === activeFilter,
    );
  }, [activeFilter, files]);
  const activeFilterLabel =
    FILTERS.find((filter) => filter.id === activeFilter)?.label || "Files";

  return (
    <div
      className="absolute right-4 top-14 z-40 w-[24rem] max-w-[calc(100vw-2rem)] overflow-hidden rounded-xl border border-border bg-bg-surface shadow-xl"
      data-testid="files-panel"
      role="dialog"
      aria-label="Files"
    >
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex min-w-0 items-center gap-2">
          <FileText size={14} className="text-accent" />
          <span className="font-heading text-sm font-semibold text-text-primary">
            Files
          </span>
          <span className="rounded-full bg-bg-elevated px-1.5 py-0.5 text-2xs font-medium text-text-muted">
            {files.length}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="rounded p-1 text-text-muted hover:bg-bg-hover"
          aria-label="Close files"
          title="Close"
        >
          <X size={14} />
        </button>
      </div>

      <div className="border-b border-border px-3 py-2">
        <div
          className="grid grid-cols-5 gap-1 rounded-lg bg-bg-elevated p-1"
          role="group"
          aria-label="File filters"
        >
          {FILTERS.map((filter) => (
            <button
              key={filter.id}
              type="button"
              onClick={() => setActiveFilter(filter.id)}
              className={`rounded-md px-2 py-1.5 text-xs font-medium transition-colors ${
                activeFilter === filter.id
                  ? "bg-bg-surface text-text-primary shadow-sm"
                  : "text-text-muted hover:text-text-primary"
              }`}
              aria-pressed={activeFilter === filter.id}
              aria-label={`Show ${filter.label} files`}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </div>

      <div className="max-h-[28rem] overflow-y-auto">
        {loading && (
          <div
            className="flex items-center justify-center gap-2 px-4 py-6 text-center text-sm text-text-muted"
            role="status"
          >
            <Loader2 size={14} className="animate-spin" />
            Loading files...
          </div>
        )}

        {!loading && error && (
          <div
            className="m-3 rounded-lg border border-red-300 bg-red-50 px-3 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
            data-testid="files-panel-load-error"
            role="alert"
          >
            <div className="flex items-start gap-2">
              <AlertCircle size={16} className="mt-0.5 shrink-0" />
              <div className="min-w-0 flex-1">
                <div className="font-medium">Files could not load</div>
                <div className="mt-1">{error}</div>
              </div>
            </div>
            <button
              type="button"
              onClick={() => void loadFiles()}
              className="mt-3 inline-flex items-center gap-1.5 rounded-md border border-red-300 bg-white px-2.5 py-1.5 text-xs font-medium text-red-700 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-400 dark:border-red-800 dark:bg-red-950/20 dark:text-red-100 dark:hover:bg-red-950/50"
              aria-label="Retry files"
            >
              <RefreshCw size={12} />
              Retry
            </button>
          </div>
        )}

        {!loading && !error && filteredFiles.length === 0 && (
          <div
            className="px-6 py-10 text-center"
            data-testid="files-panel-empty"
          >
            <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-lg border border-border bg-bg-elevated text-text-muted">
              <FileText size={20} />
            </div>
            <div className="font-heading text-sm font-semibold text-text-primary">
              {activeFilter === "all"
                ? "No files shared yet"
                : `No ${activeFilterLabel.toLowerCase()} files`}
            </div>
            <div className="mt-1 text-sm leading-5 text-text-muted">
              {activeFilter === "all"
                ? "Attachments from this conversation will collect here."
                : "Try another file type or share a matching attachment."}
            </div>
            {activeFilter !== "all" && (
              <button
                type="button"
                onClick={() => setActiveFilter("all")}
                className="btn-secondary mt-4 text-xs"
                aria-label="Show all files from empty state"
                data-testid="files-panel-empty-show-all"
              >
                Show all files
              </button>
            )}
          </div>
        )}

        {!loading &&
          !error &&
          filteredFiles.map((file) => (
            <FileRow key={file.id} file={file} withEmbed={withEmbed} />
          ))}
      </div>
    </div>
  );
}

function FileRow({
  file,
  withEmbed,
}: {
  file: ChannelFile;
  withEmbed: (href: string) => string;
}) {
  const Icon = iconFor(file.mimeType);
  const isImage = file.mimeType.startsWith("image/");

  return (
    <div
      className="group border-b border-border/50 px-4 py-3 last:border-0 hover:bg-bg-hover/60 focus-within:bg-bg-hover/60"
      data-testid="files-panel-row"
    >
      <div className="flex gap-3">
        <a
          href={file.url}
          target="_blank"
          rel="noreferrer"
          className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-border bg-bg-elevated text-text-secondary"
          title="Open file"
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
        </a>

        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 items-start gap-2">
            <div className="min-w-0 flex-1">
              <a
                href={file.url}
                target="_blank"
                rel="noreferrer"
                className="block truncate text-sm font-medium text-text-primary hover:text-accent"
                title={file.fileName}
                aria-label={`Open ${file.fileName}`}
              >
                {file.fileName}
              </a>
              <div className="mt-0.5 flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-1 text-2xs text-text-muted">
                <span>{formatFileSize(file.fileSize)}</span>
                <span>/</span>
                <span>
                  {formatDistanceToNow(new Date(file.createdAt), {
                    addSuffix: true,
                  })}
                </span>
                <span>/</span>
                <span className="truncate">{file.author.displayName}</span>
                {file.author.isAgent && <Bot size={10} className="text-teal" />}
              </div>
            </div>
            <a
              href={file.url}
              target="_blank"
              rel="noreferrer"
              className="rounded p-1 text-text-muted opacity-0 transition-opacity hover:bg-bg-elevated hover:text-accent focus:opacity-100 group-focus-within:opacity-100 group-hover:opacity-100"
              title="Open file"
              aria-label={`Open ${file.fileName}`}
            >
              <ExternalLink size={14} />
            </a>
          </div>

          {file.messageContent && (
            <p className="mt-1 line-clamp-1 text-xs text-text-muted">
              {file.messageContent}
            </p>
          )}

          <Link
            href={withEmbed(file.href)}
            className="mt-2 inline-flex rounded-md border border-border px-2 py-1 text-2xs font-medium text-text-secondary hover:border-accent hover:text-accent"
            aria-label={`Open message context for ${file.fileName}`}
          >
            Open context
          </Link>
        </div>
      </div>
    </div>
  );
}
