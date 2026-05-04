"use client";

import {
  Bell,
  FileText,
  Hash,
  Info,
  Pin,
  UserRound,
  Users,
  X,
  type LucideIcon,
} from "lucide-react";

type ConversationDetailsPanelProps = {
  title: string;
  description?: string;
  type: "channel" | "dm";
  memberCount?: number;
  onClose: () => void;
  onOpenMembers?: () => void;
  onOpenPins?: () => void;
  onOpenFiles?: () => void;
  onOpenNotifications?: () => void;
};

const formatMembers = (memberCount?: number) => {
  if (memberCount === undefined) return null;
  return `${memberCount} member${memberCount === 1 ? "" : "s"}`;
};

export default function ConversationDetailsPanel({
  title,
  description,
  type,
  memberCount,
  onClose,
  onOpenMembers,
  onOpenPins,
  onOpenFiles,
  onOpenNotifications,
}: ConversationDetailsPanelProps) {
  const isDm = type === "dm";
  const Icon = isDm ? UserRound : Hash;
  const membersLabel = formatMembers(memberCount);

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
            {description && (
              <div className="mt-1 text-sm leading-5 text-text-secondary">
                {description}
              </div>
            )}
            {membersLabel && (
              <div className="mt-1 text-xs text-text-muted">{membersLabel}</div>
            )}
          </div>
        </div>
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
            testId="conversation-details-pins"
            onClick={onOpenPins}
          />
        )}
        {onOpenFiles && (
          <DetailsAction
            icon={FileText}
            label="Files"
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
      {meta && <span className="shrink-0 text-2xs text-text-muted">{meta}</span>}
    </button>
  );
}
