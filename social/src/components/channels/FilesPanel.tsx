"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import {
  Archive,
  Bot,
  ExternalLink,
  File,
  FileText,
  Image,
  Music,
  Video,
  X,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";

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

  useEffect(() => {
    let cancelled = false;

    setLoading(true);
    setError(null);
    fetch(apiUrl(`/api/channels/${channelId}/files`))
      .then((response) => {
        if (!response.ok) throw new Error("Failed to load files");
        return response.json();
      })
      .then((data: { files?: ChannelFile[] }) => {
        if (!cancelled) setFiles(data.files || []);
      })
      .catch(() => {
        if (!cancelled) setError("Files could not load.");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [channelId]);

  const filteredFiles = useMemo(() => {
    if (activeFilter === "all") return files;
    return files.filter((file) => fileFilterFor(file.mimeType) === activeFilter);
  }, [activeFilter, files]);

  return (
    <div className="absolute right-4 top-14 z-40 w-[24rem] overflow-hidden rounded-xl border border-border bg-bg-surface shadow-xl">
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
        <div className="grid grid-cols-5 gap-1 rounded-lg bg-bg-elevated p-1">
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
            >
              {filter.label}
            </button>
          ))}
        </div>
      </div>

      <div className="max-h-[28rem] overflow-y-auto">
        {loading && (
          <div className="px-4 py-6 text-center text-sm text-text-muted">
            Loading...
          </div>
        )}

        {!loading && error && (
          <div className="m-3 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200">
            {error}
          </div>
        )}

        {!loading && !error && filteredFiles.length === 0 && (
          <div className="px-6 py-10 text-center">
            <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-lg border border-border bg-bg-elevated text-text-muted">
              <FileText size={20} />
            </div>
            <div className="font-heading text-sm font-semibold text-text-primary">
              No files shared yet
            </div>
            <div className="mt-1 text-sm leading-5 text-text-muted">
              Attachments from this conversation will collect here.
            </div>
          </div>
        )}

        {!loading &&
          !error &&
          filteredFiles.map((file) => (
            <FileRow key={file.id} file={file} />
          ))}
      </div>
    </div>
  );
}

function FileRow({ file }: { file: ChannelFile }) {
  const Icon = iconFor(file.mimeType);
  const isImage = file.mimeType.startsWith("image/");

  return (
    <div className="group border-b border-border/50 px-4 py-3 last:border-0 hover:bg-bg-hover/60">
      <div className="flex gap-3">
        <a
          href={file.url}
          target="_blank"
          rel="noreferrer"
          className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-border bg-bg-elevated text-text-secondary"
          title="Open file"
        >
          {isImage ? (
            <img
              src={file.url}
              alt=""
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
              >
                {file.fileName}
              </a>
              <div className="mt-0.5 flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-1 text-2xs text-text-muted">
                <span>{formatFileSize(file.fileSize)}</span>
                <span>/</span>
                <span>{formatDistanceToNow(new Date(file.createdAt), { addSuffix: true })}</span>
                <span>/</span>
                <span className="truncate">{file.author.displayName}</span>
                {file.author.isAgent && <Bot size={10} className="text-teal" />}
              </div>
            </div>
            <a
              href={file.url}
              target="_blank"
              rel="noreferrer"
              className="rounded p-1 text-text-muted opacity-0 transition-opacity hover:bg-bg-elevated hover:text-accent group-hover:opacity-100"
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
            href={file.href}
            className="mt-2 inline-flex rounded-md border border-border px-2 py-1 text-2xs font-medium text-text-secondary hover:border-accent hover:text-accent"
          >
            Open context
          </Link>
        </div>
      </div>
    </div>
  );
}
