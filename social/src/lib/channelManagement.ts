type SessionUserWithRole = {
  role?: string | null;
};

const CHANNEL_MANAGER_ROLES = new Set(["admin", "owner"]);

export function isWorkspaceAdminRole(role?: string | null) {
  return role?.toUpperCase() === "ADMIN";
}

export function canCreateWorkspaceChannels(user?: SessionUserWithRole | null) {
  return isWorkspaceAdminRole(user?.role);
}

export function canManageChannelRole(role?: string | null) {
  return CHANNEL_MANAGER_ROLES.has(role?.toLowerCase() || "");
}

export function canManageChannel(
  user?: SessionUserWithRole | null,
  channelRole?: string | null,
) {
  return isWorkspaceAdminRole(user?.role) || canManageChannelRole(channelRole);
}

export function normalizeChannelName(value: unknown) {
  if (typeof value !== "string") return "";

  return value
    .trim()
    .replace(/^#+/, "")
    .toLowerCase()
    .replace(/[^a-z0-9_\-\s]+/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

export function normalizeChannelDescription(value: unknown) {
  if (typeof value !== "string") return null;
  const description = value.trim().slice(0, 240);
  return description || null;
}

export function normalizeChannelVisibility(value: unknown) {
  return value === "PRIVATE" ? "PRIVATE" : "PUBLIC";
}
