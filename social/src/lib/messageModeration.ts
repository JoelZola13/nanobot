import { canManageChannel } from "@/lib/channelManagement";

type SessionUserForModeration = {
  id?: string | null;
  name?: string | null;
  role?: string | null;
};

export type MessageRemovalMode = "author" | "moderator";

export type MessageDeletionAudit = {
  actorId: string;
  actorName: string;
  mode: MessageRemovalMode;
  reason: string | null;
  deletedAt: string;
};

const MAX_DELETION_REASON_LENGTH = 240;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

export function normalizeDeletionReason(value: unknown) {
  if (typeof value !== "string") return null;
  const reason = value.trim().slice(0, MAX_DELETION_REASON_LENGTH);
  return reason || null;
}

export function getMessageRemovalMode(
  user: SessionUserForModeration | undefined | null,
  authorId: string,
  channelRole?: string | null,
): MessageRemovalMode | null {
  if (!user?.id) return null;
  if (user.id === authorId) return "author";
  return canManageChannel(user, channelRole) ? "moderator" : null;
}

export function buildMessageDeletionAudit({
  user,
  mode,
  reason,
  deletedAt,
}: {
  user: SessionUserForModeration;
  mode: MessageRemovalMode;
  reason: string | null;
  deletedAt: Date;
}): MessageDeletionAudit {
  return {
    actorId: user.id || "",
    actorName: user.name || "Unknown user",
    mode,
    reason,
    deletedAt: deletedAt.toISOString(),
  };
}

export function withMessageDeletionAudit(
  metadata: unknown,
  audit: MessageDeletionAudit,
) {
  return {
    ...(isRecord(metadata) ? metadata : {}),
    deletionAudit: audit,
  };
}

export function readMessageDeletionAudit(
  metadata: unknown,
): MessageDeletionAudit | null {
  if (!isRecord(metadata) || !isRecord(metadata.deletionAudit)) return null;
  const audit = metadata.deletionAudit;

  if (
    typeof audit.actorId !== "string" ||
    typeof audit.actorName !== "string" ||
    typeof audit.deletedAt !== "string"
  ) {
    return null;
  }

  return {
    actorId: audit.actorId,
    actorName: audit.actorName,
    mode: audit.mode === "moderator" ? "moderator" : "author",
    reason: typeof audit.reason === "string" ? audit.reason : null,
    deletedAt: audit.deletedAt,
  };
}
