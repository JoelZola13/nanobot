import {
  isNotificationLevel,
  type NotificationLevel,
} from "@/lib/notificationPreferences";
import { isWorkspaceAdminRole } from "@/lib/channelManagement";

type SessionUserWithRole = {
  role?: string | null;
};

export type WorkspaceChannelVisibility = "PUBLIC" | "PRIVATE";
export type PublicChannelJoinPolicy = "OPEN" | "WORKSPACE_ADMINS";

export type WorkspacePolicies = {
  defaultChannelVisibility: WorkspaceChannelVisibility;
  defaultNotificationLevel: NotificationLevel;
  publicChannelJoinPolicy: PublicChannelJoinPolicy;
  privateChannelJoinPolicy: "INVITE_ONLY";
  channelCreationPolicy: "MEMBERS";
};

const CHANNEL_VISIBILITIES = new Set<WorkspaceChannelVisibility>([
  "PUBLIC",
  "PRIVATE",
]);
const PUBLIC_JOIN_POLICIES = new Set<PublicChannelJoinPolicy>([
  "OPEN",
  "WORKSPACE_ADMINS",
]);

const normalizeEnvValue = (value: string | undefined) =>
  value?.trim().toUpperCase().replace(/-/g, "_");

export function normalizeWorkspaceChannelVisibility(
  value: unknown,
  fallback: WorkspaceChannelVisibility = "PUBLIC",
): WorkspaceChannelVisibility {
  if (typeof value !== "string") return fallback;
  const normalized = normalizeEnvValue(value);
  return CHANNEL_VISIBILITIES.has(normalized as WorkspaceChannelVisibility)
    ? (normalized as WorkspaceChannelVisibility)
    : fallback;
}

export function normalizePublicChannelJoinPolicy(
  value: unknown,
): PublicChannelJoinPolicy {
  if (typeof value !== "string") return "OPEN";
  const normalized = normalizeEnvValue(value);
  return PUBLIC_JOIN_POLICIES.has(normalized as PublicChannelJoinPolicy)
    ? (normalized as PublicChannelJoinPolicy)
    : "OPEN";
}

export function getWorkspacePolicies(): WorkspacePolicies {
  const defaultNotificationLevel = normalizeEnvValue(
    process.env.SOCIAL_DEFAULT_NOTIFICATION_LEVEL,
  );

  return {
    defaultChannelVisibility: normalizeWorkspaceChannelVisibility(
      process.env.SOCIAL_DEFAULT_CHANNEL_VISIBILITY,
    ),
    defaultNotificationLevel: isNotificationLevel(defaultNotificationLevel)
      ? defaultNotificationLevel
      : "ALL",
    publicChannelJoinPolicy: normalizePublicChannelJoinPolicy(
      process.env.SOCIAL_PUBLIC_CHANNEL_JOIN_POLICY,
    ),
    privateChannelJoinPolicy: "INVITE_ONLY",
    channelCreationPolicy: "MEMBERS",
  };
}

export function getDefaultChannelVisibility(value: unknown) {
  return normalizeWorkspaceChannelVisibility(
    value,
    getWorkspacePolicies().defaultChannelVisibility,
  );
}

export function getDefaultMembershipPreferences(now = new Date()) {
  const notificationLevel = getWorkspacePolicies().defaultNotificationLevel;

  return {
    notificationLevel,
    mutedAt: notificationLevel === "MUTED" ? now : null,
  };
}

export function canJoinPublicChannel(
  user?: SessionUserWithRole | null,
  existingRole?: string | null,
) {
  if (existingRole) return true;
  const policy = getWorkspacePolicies().publicChannelJoinPolicy;
  return policy === "OPEN" || isWorkspaceAdminRole(user?.role);
}
