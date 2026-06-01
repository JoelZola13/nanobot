"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import {
  type FormEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import TopBar from "@/components/layout/TopBar";
import {
  Archive,
  ArrowRight,
  Check,
  Globe2,
  Hash,
  Lock,
  LogOut,
  MessageSquare,
  Pencil,
  Plus,
  RotateCcw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Users,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";
import { useEmbeddedNavigation } from "@/lib/useEmbeddedNavigation";

type ChannelSummary = {
  id: string;
  name: string | null;
  slug: string | null;
  description: string | null;
  type: "PUBLIC" | "PRIVATE" | "DM" | "GROUP_DM";
  iconEmoji: string | null;
  isDefault?: boolean;
  isArchived?: boolean;
  isMember?: boolean;
  memberCount?: number;
  messageCount?: number;
  role?: string;
  canCreate?: boolean;
  canManage?: boolean;
};

type ChannelVisibility = "PUBLIC" | "PRIVATE";

type WorkspacePolicies = {
  defaultChannelVisibility: ChannelVisibility;
  defaultNotificationLevel: "ALL" | "MENTIONS" | "MUTED";
  publicChannelJoinPolicy: "OPEN" | "WORKSPACE_ADMINS";
  privateChannelJoinPolicy: "INVITE_ONLY";
  channelCreationPolicy: "MEMBERS";
  canManage: boolean;
};

const formatCount = (count: number | undefined, label: string) => {
  const value = count ?? 0;
  return `${value} ${label}${value === 1 ? "" : "s"}`;
};

const normalizeChannelName = (channel: ChannelSummary) =>
  channel.name || channel.slug || "unnamed";

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

export default function ChannelsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { shouldPreserveEmbed, withEmbed } = useEmbeddedNavigation();
  const [channels, setChannels] = useState<ChannelSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [channelType, setChannelType] = useState<ChannelVisibility>("PUBLIC");
  const [createVisibilityTouched, setCreateVisibilityTouched] = useState(false);
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [editingChannel, setEditingChannel] = useState<ChannelSummary | null>(
    null,
  );
  const [editName, setEditName] = useState("");
  const [editDescription, setEditDescription] = useState("");
  const [editType, setEditType] = useState<ChannelVisibility>("PUBLIC");
  const [updating, setUpdating] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);
  const [busyChannelId, setBusyChannelId] = useState<string | null>(null);
  const [archivingChannelId, setArchivingChannelId] = useState<string | null>(
    null,
  );
  const [showArchived, setShowArchived] = useState(false);
  const [query, setQuery] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [channelActionErrors, setChannelActionErrors] = useState<
    Record<string, string>
  >({});
  const [workspacePolicies, setWorkspacePolicies] =
    useState<WorkspacePolicies | null>(null);
  const [workspacePoliciesLoading, setWorkspacePoliciesLoading] =
    useState(false);
  const [workspacePolicyError, setWorkspacePolicyError] = useState<
    string | null
  >(null);

  const loadChannels = useCallback(async () => {
    setLoading(true);
    setLoadError(null);

    try {
      const url = showArchived
        ? "/api/channels?includeArchived=true"
        : "/api/channels";
      const res = await fetch(apiUrl(url), { cache: "no-store" });
      if (!res.ok) {
        throw new Error(
          await getApiErrorMessage(res, "Channels could not load."),
        );
      }

      const data = (await res.json()) as ChannelSummary[];
      setChannels(
        data
          .filter(
            (channel) =>
              channel.type === "PUBLIC" || channel.type === "PRIVATE",
          )
          .map((channel) => ({
            ...channel,
            isArchived: Boolean(channel.isArchived),
            isMember: Boolean(channel.isMember),
          })),
      );
    } catch (error) {
      setLoadError(
        error instanceof Error ? error.message : "Channels could not load.",
      );
    } finally {
      setLoading(false);
    }
  }, [showArchived]);

  useEffect(() => {
    void loadChannels();
  }, [loadChannels]);

  const visibleChannels = useMemo(() => {
    const trimmedQuery = query.trim().toLowerCase();
    if (!trimmedQuery) return channels;

    return channels.filter((channel) => {
      const haystack = [
        channel.name,
        channel.slug,
        channel.description,
        channel.type === "PRIVATE" ? "private" : "public",
        channel.isArchived ? "archived" : "active",
        channel.isMember ? "joined" : "available",
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return haystack.includes(trimmedQuery);
    });
  }, [channels, query]);

  const activeChannels = useMemo(
    () => visibleChannels.filter((channel) => !channel.isArchived),
    [visibleChannels],
  );
  const archivedChannels = useMemo(
    () => visibleChannels.filter((channel) => channel.isArchived),
    [visibleChannels],
  );
  const defaultChannels = useMemo(
    () => activeChannels.filter((channel) => channel.isDefault),
    [activeChannels],
  );
  const joinedChannels = useMemo(
    () =>
      activeChannels.filter(
        (channel) =>
          channel.isMember && !channel.isDefault && channel.type === "PUBLIC",
      ),
    [activeChannels],
  );
  const privateChannels = useMemo(
    () => activeChannels.filter((channel) => channel.type === "PRIVATE"),
    [activeChannels],
  );
  const browseChannels = useMemo(
    () =>
      activeChannels.filter(
        (channel) => !channel.isMember && channel.type === "PUBLIC",
      ),
    [activeChannels],
  );
  const canCreateChannels = channels.some((channel) => channel.canCreate);
  const canViewArchivedChannels = channels.some((channel) => channel.canManage);
  const defaultChannelType =
    workspacePolicies?.defaultChannelVisibility ?? "PUBLIC";

  useEffect(() => {
    if (searchParams.get("create") === "true" && canCreateChannels) {
      setEditingChannel(null);
      setShowCreate(true);
    }
  }, [canCreateChannels, searchParams]);

  const replaceCreateParam = useCallback(
    (visible: boolean) => {
      const params = new URLSearchParams(searchParams.toString());
      if (visible) {
        params.set("create", "true");
      } else {
        params.delete("create");
      }
      if (shouldPreserveEmbed) params.set("embed", "true");

      const queryString = params.toString();
      router.replace(`/channels${queryString ? `?${queryString}` : ""}`, {
        scroll: false,
      });
    },
    [router, searchParams, shouldPreserveEmbed],
  );

  const openCreateForm = useCallback(() => {
    setEditingChannel(null);
    setEditError(null);
    setCreateError(null);
    setActionError(null);
    setCreateVisibilityTouched(false);
    setChannelType(defaultChannelType);
    setShowCreate(true);
    replaceCreateParam(true);
  }, [defaultChannelType, replaceCreateParam]);

  const closeCreateForm = useCallback(() => {
    setShowCreate(false);
    setName("");
    setDescription("");
    setChannelType(defaultChannelType);
    setCreateVisibilityTouched(false);
    setCreateError(null);
    replaceCreateParam(false);
  }, [defaultChannelType, replaceCreateParam]);

  const closeEditForm = useCallback(() => {
    setEditingChannel(null);
    setEditError(null);
  }, []);

  const loadWorkspacePolicies = useCallback(async () => {
    if (!canCreateChannels) {
      setWorkspacePolicies(null);
      setWorkspacePolicyError(null);
      setWorkspacePoliciesLoading(false);
      return;
    }

    setWorkspacePoliciesLoading(true);
    setWorkspacePolicyError(null);

    try {
      const res = await fetch(apiUrl("/api/workspace/policies"), {
        cache: "no-store",
      });
      if (!res.ok) {
        throw new Error(
          await getApiErrorMessage(res, "Workspace defaults could not load."),
        );
      }

      const data = (await res.json()) as WorkspacePolicies;
      setWorkspacePolicies(data);
    } catch (error) {
      setWorkspacePolicies(null);
      setWorkspacePolicyError(
        error instanceof Error
          ? error.message
          : "Workspace defaults could not load.",
      );
    } finally {
      setWorkspacePoliciesLoading(false);
    }
  }, [canCreateChannels]);

  useEffect(() => {
    void loadWorkspacePolicies();
  }, [loadWorkspacePolicies]);

  useEffect(() => {
    if (!showCreate || createVisibilityTouched) return;
    setChannelType(defaultChannelType);
  }, [createVisibilityTouched, defaultChannelType, showCreate]);

  const handleCreate = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || creating) return;
    setCreating(true);
    setActionError(null);
    setCreateError(null);

    try {
      const res = await fetch(apiUrl("/api/channels"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description, type: channelType }),
      });
      if (!res.ok) {
        throw new Error(
          await getApiErrorMessage(res, "Channel could not be created."),
        );
      }

      const channel = await res.json();
      setShowCreate(false);
      setName("");
      setDescription("");
      setChannelType(defaultChannelType);
      setCreateVisibilityTouched(false);
      router.push(withEmbed(`/channels/${channel.id}`));
    } catch (error) {
      setCreateError(
        error instanceof Error
          ? error.message
          : "Channel could not be created.",
      );
    } finally {
      setCreating(false);
    }
  };

  const openEditChannel = (channel: ChannelSummary) => {
    closeCreateForm();
    setEditingChannel(channel);
    setEditName(normalizeChannelName(channel));
    setEditDescription(channel.description || "");
    setEditType(channel.type === "PRIVATE" ? "PRIVATE" : "PUBLIC");
    setEditError(null);
    setActionError(null);
    clearChannelActionError(channel.id);
  };

  const updateChannel = (updatedChannel: ChannelSummary) => {
    setChannels((currentChannels) =>
      currentChannels.map((channel) =>
        channel.id === updatedChannel.id
          ? {
              ...channel,
              ...updatedChannel,
              isMember: Boolean(updatedChannel.isMember),
            }
          : channel,
      ),
    );
  };

  const handleUpdate = async (e: FormEvent) => {
    e.preventDefault();
    if (!editingChannel || !editName.trim() || updating) return;
    setUpdating(true);
    setActionError(null);
    setEditError(null);

    try {
      const res = await fetch(apiUrl(`/api/channels/${editingChannel.id}`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: editName,
          description: editDescription,
          type: editType,
        }),
      });
      if (!res.ok) {
        throw new Error(
          await getApiErrorMessage(
            res,
            `#${normalizeChannelName(editingChannel)} could not be updated.`,
          ),
        );
      }

      const updatedChannel = (await res.json()) as ChannelSummary;
      updateChannel(updatedChannel);
      closeEditForm();
    } catch (error) {
      setEditError(
        error instanceof Error
          ? error.message
          : `#${normalizeChannelName(editingChannel)} could not be updated.`,
      );
    } finally {
      setUpdating(false);
    }
  };

  const applyChannelArchiveState = (updatedChannel: ChannelSummary) => {
    setChannels((currentChannels) => {
      if (updatedChannel.isArchived && !showArchived) {
        return currentChannels.filter(
          (channel) => channel.id !== updatedChannel.id,
        );
      }

      const exists = currentChannels.some(
        (channel) => channel.id === updatedChannel.id,
      );
      const normalizedChannel = {
        ...updatedChannel,
        isArchived: Boolean(updatedChannel.isArchived),
        isMember: Boolean(updatedChannel.isMember),
      };

      if (!exists) return [...currentChannels, normalizedChannel];

      return currentChannels.map((channel) =>
        channel.id === updatedChannel.id
          ? { ...channel, ...normalizedChannel }
          : channel,
      );
    });
  };

  const handleArchiveChange = async (
    channel: ChannelSummary,
    archived: boolean,
  ) => {
    if (archivingChannelId || channel.isDefault) return;
    if (
      archived &&
      !window.confirm(`Archive #${normalizeChannelName(channel)}?`)
    ) {
      return;
    }

    setArchivingChannelId(channel.id);
    setActionError(null);
    clearChannelActionError(channel.id);

    try {
      const res = await fetch(apiUrl(`/api/channels/${channel.id}/archive`), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archived }),
      });
      if (!res.ok) {
        throw new Error(
          await getApiErrorMessage(
            res,
            `#${normalizeChannelName(channel)} could not be ${
              archived ? "archived" : "restored"
            }.`,
          ),
        );
      }

      const updatedChannel = (await res.json()) as ChannelSummary;
      applyChannelArchiveState(updatedChannel);
      setEditingChannel((current) =>
        current?.id === channel.id ? null : current,
      );
      clearChannelActionError(channel.id);
    } catch (error) {
      setChannelActionError(
        channel.id,
        error instanceof Error
          ? error.message
          : `#${normalizeChannelName(channel)} could not be ${
              archived ? "archived" : "restored"
            }.`,
      );
    } finally {
      setArchivingChannelId(null);
    }
  };

  const updateChannelMembership = (
    channelId: string,
    isMember: boolean,
    role?: string,
  ) => {
    setChannels((currentChannels) =>
      currentChannels.map((channel) => {
        if (channel.id !== channelId) return channel;
        const wasMember = Boolean(channel.isMember);
        const memberDelta =
          isMember && !wasMember ? 1 : !isMember && wasMember ? -1 : 0;

        return {
          ...channel,
          isMember,
          role: isMember ? role || "member" : undefined,
          memberCount: Math.max((channel.memberCount || 0) + memberDelta, 0),
        };
      }),
    );
  };

  const clearChannelActionError = (channelId: string) => {
    setChannelActionErrors((currentErrors) => {
      if (!currentErrors[channelId]) return currentErrors;
      const nextErrors = { ...currentErrors };
      delete nextErrors[channelId];
      return nextErrors;
    });
  };

  const setChannelActionError = (channelId: string, error: string) => {
    setChannelActionErrors((currentErrors) => ({
      ...currentErrors,
      [channelId]: error,
    }));
  };

  const handleJoin = async (channel: ChannelSummary) => {
    if (busyChannelId) return;
    setBusyChannelId(channel.id);
    setActionError(null);
    clearChannelActionError(channel.id);

    try {
      const res = await fetch(
        apiUrl(`/api/channels/${channel.id}/membership`),
        {
          method: "POST",
        },
      );
      if (!res.ok) {
        throw new Error(
          await getApiErrorMessage(
            res,
            `#${normalizeChannelName(channel)} could not be joined.`,
          ),
        );
      }

      const data = await res.json();
      updateChannelMembership(channel.id, true, data.role);
      clearChannelActionError(channel.id);
    } catch (error) {
      setChannelActionError(
        channel.id,
        error instanceof Error
          ? error.message
          : `#${normalizeChannelName(channel)} could not be joined.`,
      );
    } finally {
      setBusyChannelId(null);
    }
  };

  const handleLeave = async (channel: ChannelSummary) => {
    if (busyChannelId || channel.isDefault) return;
    setBusyChannelId(channel.id);
    setActionError(null);
    clearChannelActionError(channel.id);

    try {
      const res = await fetch(
        apiUrl(`/api/channels/${channel.id}/membership`),
        {
          method: "DELETE",
        },
      );
      if (!res.ok) {
        throw new Error(
          await getApiErrorMessage(
            res,
            `#${normalizeChannelName(channel)} could not be left.`,
          ),
        );
      }

      updateChannelMembership(channel.id, false);
      clearChannelActionError(channel.id);
    } catch (error) {
      setChannelActionError(
        channel.id,
        error instanceof Error
          ? error.message
          : `#${normalizeChannelName(channel)} could not be left.`,
      );
    } finally {
      setBusyChannelId(null);
    }
  };

  const hasVisibleChannels =
    activeChannels.length > 0 || archivedChannels.length > 0;
  const hasQuery = query.trim().length > 0;

  return (
    <>
      <TopBar
        title="Channels"
        type="channel"
        description="Browse workspace channels"
      />
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl px-4 py-6">
          <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="min-w-0">
              <h2 className="font-heading text-xl font-semibold text-text-primary">
                Channel browser
              </h2>
              <p className="text-sm text-text-muted">
                {loading
                  ? "Loading workspace channels"
                  : formatCount(channels.length, "workspace channel")}
              </p>
            </div>
            {canCreateChannels && (
              <button
                type="button"
                onClick={() => {
                  if (showCreate) {
                    closeCreateForm();
                  } else {
                    openCreateForm();
                  }
                }}
                className="btn-primary inline-flex items-center gap-2 self-start text-sm"
                aria-label={
                  showCreate ? "Close new channel form" : "Create a new channel"
                }
                aria-expanded={showCreate}
                aria-controls="channel-browser-create-form"
              >
                <Plus size={16} />
                <span>New Channel</span>
              </button>
            )}
          </div>

          <div className="mb-5 flex flex-col gap-2 sm:flex-row sm:items-center">
            <div className="flex min-w-0 flex-1 items-center gap-2 rounded-lg border border-border bg-bg-surface px-3 py-2">
              <Search size={16} className="shrink-0 text-text-muted" />
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search channels"
                className="min-w-0 flex-1 bg-transparent text-sm text-text-primary outline-none placeholder:text-text-muted"
                aria-label="Search channels"
                aria-controls="channel-browser-results"
                data-testid="channel-browser-search"
              />
              {hasQuery && (
                <button
                  type="button"
                  onClick={() => setQuery("")}
                  className="rounded-md px-2 py-1 text-xs font-medium text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
                  aria-label="Clear channel search"
                >
                  Clear
                </button>
              )}
            </div>
            {canViewArchivedChannels && (
              <button
                type="button"
                onClick={() => setShowArchived((current) => !current)}
                className={`btn-ghost inline-flex h-10 items-center gap-2 self-start px-3 text-sm ${
                  showArchived ? "text-accent" : ""
                }`}
                aria-pressed={showArchived}
                aria-controls="channel-browser-results"
                aria-label={
                  showArchived
                    ? "Hide archived channels"
                    : "Show archived channels"
                }
              >
                <Archive size={15} />
                <span>{showArchived ? "Hide archived" : "Show archived"}</span>
              </button>
            )}
          </div>

          {workspacePoliciesLoading && !workspacePolicies ? (
            <WorkspacePolicyLoading />
          ) : workspacePolicyError ? (
            <WorkspacePolicyLoadError
              error={workspacePolicyError}
              onRetry={() => void loadWorkspacePolicies()}
            />
          ) : workspacePolicies?.canManage ? (
            <WorkspacePolicySummary policies={workspacePolicies} />
          ) : null}

          {showCreate && (
            <form
              id="channel-browser-create-form"
              data-testid="channel-browser-create-form"
              onSubmit={handleCreate}
              className="mb-6 space-y-4 rounded-lg border border-border bg-bg-surface p-5"
            >
              <div>
                <label
                  htmlFor="new-channel-name"
                  className="mb-1.5 block text-sm font-medium text-text-primary"
                >
                  Channel name
                </label>
                <div className="flex items-center gap-2">
                  <Hash size={16} className="shrink-0 text-text-muted" />
                  <input
                    id="new-channel-name"
                    type="text"
                    value={name}
                    onChange={(e) => {
                      setName(e.target.value);
                      setCreateError(null);
                    }}
                    placeholder="e.g. content-team"
                    className="input-field flex-1"
                    autoFocus
                  />
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium text-text-primary">
                  Visibility
                </label>
                <div
                  className="grid gap-2 sm:grid-cols-2"
                  role="group"
                  aria-label="Channel visibility"
                >
                  <VisibilityButton
                    active={channelType === "PUBLIC"}
                    icon={<Globe2 size={15} />}
                    title="Public"
                    subtitle="Open access"
                    ariaLabel="New channel visibility Public"
                    testId="channel-browser-create-visibility-public"
                    onClick={() => {
                      setCreateVisibilityTouched(true);
                      setChannelType("PUBLIC");
                    }}
                  />
                  <VisibilityButton
                    active={channelType === "PRIVATE"}
                    icon={<Lock size={15} />}
                    title="Private"
                    subtitle="Invite-only"
                    ariaLabel="New channel visibility Private"
                    testId="channel-browser-create-visibility-private"
                    onClick={() => {
                      setCreateVisibilityTouched(true);
                      setChannelType("PRIVATE");
                    }}
                  />
                </div>
              </div>

              <div>
                <label
                  htmlFor="new-channel-description"
                  className="mb-1.5 block text-sm font-medium text-text-primary"
                >
                  Description{" "}
                  <span className="font-normal text-text-muted">
                    (optional)
                  </span>
                </label>
                <input
                  id="new-channel-description"
                  type="text"
                  value={description}
                  onChange={(e) => {
                    setDescription(e.target.value);
                    setCreateError(null);
                  }}
                  placeholder="What's this channel about?"
                  className="input-field w-full"
                />
              </div>
              {createError && (
                <div
                  className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm font-medium text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
                  role="status"
                  data-testid="channel-browser-create-error"
                >
                  {createError}
                </div>
              )}
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={closeCreateForm}
                  className="btn-ghost text-sm"
                  aria-label="Cancel new channel"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!name.trim() || creating}
                  className="btn-primary text-sm"
                >
                  {creating ? "Creating..." : "Create Channel"}
                </button>
              </div>
            </form>
          )}

          {editingChannel && (
            <form
              id="channel-browser-edit-form"
              data-testid="channel-browser-edit-form"
              onSubmit={handleUpdate}
              className="mb-6 space-y-4 rounded-lg border border-border bg-bg-surface p-5"
            >
              <div className="flex items-center justify-between gap-3">
                <div>
                  <h3 className="font-heading text-base font-semibold text-text-primary">
                    Edit channel
                  </h3>
                  <p className="text-sm text-text-muted">
                    Manage #{normalizeChannelName(editingChannel)}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={closeEditForm}
                  className="btn-ghost text-sm"
                >
                  Close
                </button>
              </div>

              <div>
                <label
                  htmlFor="edit-channel-name"
                  className="mb-1.5 block text-sm font-medium text-text-primary"
                >
                  Channel name
                </label>
                <div className="flex items-center gap-2">
                  <Hash size={16} className="shrink-0 text-text-muted" />
                  <input
                    id="edit-channel-name"
                    type="text"
                    value={editName}
                    onChange={(e) => {
                      setEditName(e.target.value);
                      setEditError(null);
                    }}
                    className="input-field flex-1"
                    autoFocus
                  />
                </div>
              </div>

              <div>
                <label className="mb-1.5 block text-sm font-medium text-text-primary">
                  Visibility
                </label>
                <div
                  className="grid gap-2 sm:grid-cols-2"
                  role="group"
                  aria-label="Channel visibility"
                >
                  <VisibilityButton
                    active={editType === "PUBLIC"}
                    icon={<Globe2 size={15} />}
                    title="Public"
                    subtitle="Open access"
                    ariaLabel="Edit channel visibility Public"
                    testId="channel-browser-edit-visibility-public"
                    onClick={() => setEditType("PUBLIC")}
                  />
                  <VisibilityButton
                    active={editType === "PRIVATE"}
                    icon={<Lock size={15} />}
                    title="Private"
                    subtitle="Invite-only"
                    ariaLabel="Edit channel visibility Private"
                    testId="channel-browser-edit-visibility-private"
                    onClick={() => setEditType("PRIVATE")}
                  />
                </div>
              </div>

              <div>
                <label
                  htmlFor="edit-channel-description"
                  className="mb-1.5 block text-sm font-medium text-text-primary"
                >
                  Description{" "}
                  <span className="font-normal text-text-muted">
                    (optional)
                  </span>
                </label>
                <input
                  id="edit-channel-description"
                  type="text"
                  value={editDescription}
                  onChange={(e) => {
                    setEditDescription(e.target.value);
                    setEditError(null);
                  }}
                  placeholder="What's this channel about?"
                  className="input-field w-full"
                />
              </div>
              {editError && (
                <div
                  className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-sm font-medium text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
                  role="status"
                  data-testid="channel-browser-edit-error"
                >
                  {editError}
                </div>
              )}
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={closeEditForm}
                  className="btn-ghost text-sm"
                  aria-label="Cancel channel edit"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!editName.trim() || updating}
                  className="btn-primary text-sm"
                >
                  {updating ? "Saving..." : "Save changes"}
                </button>
              </div>
            </form>
          )}

          {actionError && (
            <div
              className="mb-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
              role="status"
              data-testid="channel-browser-action-error"
            >
              {actionError}
            </div>
          )}

          {loadError && channels.length > 0 && (
            <ChannelLoadError
              error={loadError}
              onRetry={() => void loadChannels()}
            />
          )}

          {loading ? (
            <div
              className="rounded-lg border border-border bg-bg-surface px-4 py-10 text-center text-sm text-text-muted"
              aria-busy="true"
              data-testid="channel-browser-loading"
            >
              Loading channels...
            </div>
          ) : loadError && channels.length === 0 ? (
            <ChannelLoadError
              error={loadError}
              onRetry={() => void loadChannels()}
            />
          ) : hasVisibleChannels ? (
            <div
              id="channel-browser-results"
              className="space-y-6"
              aria-live="polite"
              data-testid="channel-browser-results"
            >
              <ChannelSection
                title="Default channels"
                channels={defaultChannels}
                busyChannelId={busyChannelId}
                onJoin={handleJoin}
                onLeave={handleLeave}
                onEdit={openEditChannel}
                onArchiveChange={handleArchiveChange}
                archivingChannelId={archivingChannelId}
                channelActionErrors={channelActionErrors}
                withEmbed={withEmbed}
              />
              <ChannelSection
                title="Your public channels"
                channels={joinedChannels}
                busyChannelId={busyChannelId}
                onJoin={handleJoin}
                onLeave={handleLeave}
                onEdit={openEditChannel}
                onArchiveChange={handleArchiveChange}
                archivingChannelId={archivingChannelId}
                channelActionErrors={channelActionErrors}
                withEmbed={withEmbed}
              />
              <ChannelSection
                title="Private channels"
                channels={privateChannels}
                busyChannelId={busyChannelId}
                onJoin={handleJoin}
                onLeave={handleLeave}
                onEdit={openEditChannel}
                onArchiveChange={handleArchiveChange}
                archivingChannelId={archivingChannelId}
                channelActionErrors={channelActionErrors}
                withEmbed={withEmbed}
              />
              <ChannelSection
                title="Browse public channels"
                channels={browseChannels}
                busyChannelId={busyChannelId}
                onJoin={handleJoin}
                onLeave={handleLeave}
                onEdit={openEditChannel}
                onArchiveChange={handleArchiveChange}
                archivingChannelId={archivingChannelId}
                channelActionErrors={channelActionErrors}
                withEmbed={withEmbed}
              />
              {showArchived && (
                <ChannelSection
                  title="Archived channels"
                  channels={archivedChannels}
                  busyChannelId={busyChannelId}
                  onJoin={handleJoin}
                  onLeave={handleLeave}
                  onEdit={openEditChannel}
                  onArchiveChange={handleArchiveChange}
                  archivingChannelId={archivingChannelId}
                  channelActionErrors={channelActionErrors}
                  withEmbed={withEmbed}
                />
              )}
            </div>
          ) : (
            <div
              id="channel-browser-results"
              className="rounded-lg border border-border bg-bg-surface px-6 py-12 text-center"
              aria-live="polite"
              data-testid="channel-browser-empty"
            >
              <Users size={42} className="mx-auto mb-3 text-text-muted" />
              <h3 className="font-heading text-lg font-semibold text-text-primary">
                No matching channels
              </h3>
              <p className="mt-1 text-sm text-text-muted">
                No channels match this search.
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

function ChannelLoadError({
  error,
  onRetry,
}: {
  error: string;
  onRetry: () => void;
}) {
  return (
    <div
      className="mb-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
      role="alert"
      data-testid="channel-browser-load-error"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <span>{error}</span>
        <button
          type="button"
          onClick={onRetry}
          className="btn-ghost self-start border-red-300 bg-white/70 text-sm text-red-700 hover:bg-red-100 dark:border-red-900/60 dark:bg-red-950/40 dark:text-red-100 dark:hover:bg-red-900/30"
          aria-label="Retry channels"
        >
          Retry
        </button>
      </div>
    </div>
  );
}

function WorkspacePolicySummary({ policies }: { policies: WorkspacePolicies }) {
  const notificationLabels: Record<
    WorkspacePolicies["defaultNotificationLevel"],
    string
  > = {
    ALL: "All activity",
    MENTIONS: "Mentions",
    MUTED: "Muted",
  };
  const publicJoinLabels: Record<
    WorkspacePolicies["publicChannelJoinPolicy"],
    string
  > = {
    OPEN: "Open",
    WORKSPACE_ADMINS: "Admins only",
  };
  const items = [
    {
      label: "New channels",
      value:
        policies.defaultChannelVisibility === "PRIVATE" ? "Private" : "Public",
    },
    {
      label: "New members",
      value: notificationLabels[policies.defaultNotificationLevel],
    },
    {
      label: "Public joins",
      value: publicJoinLabels[policies.publicChannelJoinPolicy],
    },
    {
      label: "Private joins",
      value: "Invite-only",
    },
  ];

  return (
    <section
      className="mb-5 rounded-lg border border-border bg-bg-surface p-4"
      data-testid="channel-browser-policy-summary"
    >
      <div className="mb-3 flex items-center gap-2 text-sm font-semibold text-text-primary">
        <SlidersHorizontal size={16} className="text-accent" />
        <span>Workspace defaults</span>
      </div>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        {items.map((item) => (
          <div
            key={item.label}
            className="min-w-0 rounded-md border border-border bg-bg-base px-3 py-2"
          >
            <div className="truncate text-2xs font-semibold uppercase tracking-wide text-text-muted">
              {item.label}
            </div>
            <div className="mt-0.5 truncate text-sm font-medium text-text-primary">
              {item.value}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function WorkspacePolicyLoading() {
  return (
    <section
      className="mb-5 rounded-lg border border-border bg-bg-surface p-4 text-sm text-text-muted"
      aria-busy="true"
      data-testid="channel-browser-policy-loading"
    >
      Loading workspace defaults...
    </section>
  );
}

function WorkspacePolicyLoadError({
  error,
  onRetry,
}: {
  error: string;
  onRetry: () => void;
}) {
  return (
    <section
      className="mb-5 rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-800 dark:border-amber-900/50 dark:bg-amber-950/30 dark:text-amber-100"
      role="status"
      data-testid="channel-browser-policy-error"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <span>{error}</span>
        <button
          type="button"
          onClick={onRetry}
          className="btn-ghost self-start border-amber-300 bg-white/70 text-sm text-amber-800 hover:bg-amber-100 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-100 dark:hover:bg-amber-900/30"
          aria-label="Retry workspace defaults"
        >
          Retry
        </button>
      </div>
    </section>
  );
}

function VisibilityButton({
  active,
  icon,
  title,
  subtitle,
  ariaLabel,
  testId,
  onClick,
}: {
  active: boolean;
  icon: ReactNode;
  title: string;
  subtitle: string;
  ariaLabel: string;
  testId: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      data-testid={testId}
      className={`flex min-w-0 items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors ${
        active
          ? "border-accent bg-accent-muted text-accent"
          : "border-border bg-bg-base text-text-secondary hover:border-accent/60 hover:text-text-primary"
      }`}
      aria-pressed={active}
    >
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-current/30">
        {icon}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-semibold">{title}</span>
        <span className="block truncate text-xs opacity-75">{subtitle}</span>
      </span>
      {active && <Check size={15} className="shrink-0" />}
    </button>
  );
}

function ChannelSection({
  title,
  channels,
  busyChannelId,
  onJoin,
  onLeave,
  onEdit,
  onArchiveChange,
  archivingChannelId,
  channelActionErrors,
  withEmbed,
}: {
  title: string;
  channels: ChannelSummary[];
  busyChannelId: string | null;
  onJoin: (channel: ChannelSummary) => void;
  onLeave: (channel: ChannelSummary) => void;
  onEdit: (channel: ChannelSummary) => void;
  onArchiveChange: (channel: ChannelSummary, archived: boolean) => void;
  archivingChannelId: string | null;
  channelActionErrors: Record<string, string>;
  withEmbed: (href: string) => string;
}) {
  if (channels.length === 0) return null;

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between px-1">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-text-muted">
          {title}
        </h3>
        <span className="text-xs text-text-muted">{channels.length}</span>
      </div>
      <div className="space-y-2">
        {channels.map((channel) => (
          <ChannelRow
            key={channel.id}
            channel={channel}
            busy={busyChannelId === channel.id}
            archiving={archivingChannelId === channel.id}
            onJoin={onJoin}
            onLeave={onLeave}
            onEdit={onEdit}
            onArchiveChange={onArchiveChange}
            actionError={channelActionErrors[channel.id]}
            withEmbed={withEmbed}
          />
        ))}
      </div>
    </section>
  );
}

function ChannelRow({
  channel,
  busy,
  archiving,
  onJoin,
  onLeave,
  onEdit,
  onArchiveChange,
  actionError,
  withEmbed,
}: {
  channel: ChannelSummary;
  busy: boolean;
  archiving: boolean;
  onJoin: (channel: ChannelSummary) => void;
  onLeave: (channel: ChannelSummary) => void;
  onEdit: (channel: ChannelSummary) => void;
  onArchiveChange: (channel: ChannelSummary, archived: boolean) => void;
  actionError?: string;
  withEmbed: (href: string) => string;
}) {
  const isPrivate = channel.type === "PRIVATE";
  const isMember = Boolean(channel.isMember);
  const isArchived = Boolean(channel.isArchived);
  const channelName = normalizeChannelName(channel);

  return (
    <article
      className="flex flex-col gap-3 rounded-lg border border-border bg-bg-surface px-4 py-3 transition-colors hover:border-accent/60 sm:flex-row sm:items-start"
      data-testid="channel-browser-row"
      aria-label={`#${channelName} channel`}
    >
      <div className="flex min-w-0 flex-1 gap-3">
        <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-border bg-bg-base text-text-muted">
          {isPrivate ? <Lock size={17} /> : <Hash size={17} />}
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate font-medium text-text-primary">
              {channelName}
            </span>
            {channel.isDefault && (
              <span className="inline-flex items-center gap-1 rounded-full border border-accent/40 bg-accent-muted px-2 py-0.5 text-2xs font-semibold text-accent">
                <ShieldCheck size={11} />
                Default
              </span>
            )}
            {isPrivate && (
              <span className="inline-flex items-center gap-1 rounded-full border border-border bg-bg-elevated px-2 py-0.5 text-2xs font-semibold text-text-secondary">
                <Lock size={11} />
                Private
              </span>
            )}
            {isArchived && (
              <span className="inline-flex items-center gap-1 rounded-full border border-border bg-bg-elevated px-2 py-0.5 text-2xs font-semibold text-text-secondary">
                <Archive size={11} />
                Archived
              </span>
            )}
            {isMember && !channel.isDefault && (
              <span className="inline-flex items-center gap-1 rounded-full border border-teal/30 bg-teal-muted px-2 py-0.5 text-2xs font-semibold text-teal">
                <Check size={11} />
                Joined
              </span>
            )}
          </div>
          {channel.description && (
            <p className="mt-0.5 truncate text-sm text-text-muted">
              {channel.description}
            </p>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-text-muted">
            <span className="inline-flex items-center gap-1">
              <Users size={13} />
              {formatCount(channel.memberCount, "member")}
            </span>
            <span className="inline-flex items-center gap-1">
              <MessageSquare size={13} />
              {formatCount(channel.messageCount, "message")}
            </span>
            {channel.role && <span className="capitalize">{channel.role}</span>}
            {channel.canManage && (
              <span className="inline-flex items-center gap-1 text-accent">
                <ShieldCheck size={13} />
                Manage
              </span>
            )}
          </div>
          {actionError && (
            <div
              className="mt-3 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs font-medium text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
              role="status"
              data-testid="channel-browser-row-action-error"
            >
              {actionError}
            </div>
          )}
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-2 sm:pt-0.5">
        {isArchived ? (
          <>
            {channel.canManage && (
              <button
                type="button"
                onClick={() => onArchiveChange(channel, false)}
                disabled={archiving}
                className="btn-primary inline-flex h-9 items-center gap-1.5 px-3 text-sm"
                aria-label={`Restore #${channelName}`}
              >
                <RotateCcw size={14} />
                <span>{archiving ? "Restoring..." : "Restore"}</span>
              </button>
            )}
          </>
        ) : isMember ? (
          <>
            {channel.canManage && (
              <>
                <button
                  type="button"
                  onClick={() => onEdit(channel)}
                  className="btn-ghost inline-flex h-9 items-center gap-1.5 px-3 text-sm"
                  aria-label={`Edit #${channelName}`}
                >
                  <Pencil size={14} />
                  <span>Edit</span>
                </button>
                {!channel.isDefault && (
                  <button
                    type="button"
                    onClick={() => onArchiveChange(channel, true)}
                    disabled={archiving}
                    className="btn-ghost inline-flex h-9 items-center gap-1.5 px-3 text-sm text-text-muted hover:text-danger"
                    aria-label={`Archive #${channelName}`}
                  >
                    <Archive size={14} />
                    <span>{archiving ? "Archiving..." : "Archive"}</span>
                  </button>
                )}
              </>
            )}
            <Link
              href={withEmbed(`/channels/${channel.id}`)}
              className="btn-primary inline-flex h-9 items-center gap-1.5 px-3 text-sm !text-black"
              aria-label={`Open #${channelName}`}
            >
              <span>Open</span>
              <ArrowRight size={14} />
            </Link>
            {channel.isDefault ? (
              <span className="inline-flex h-9 items-center rounded-md border border-border px-3 text-xs font-medium text-text-muted">
                Required
              </span>
            ) : (
              <button
                type="button"
                onClick={() => onLeave(channel)}
                disabled={busy}
                className="btn-ghost inline-flex h-9 items-center gap-1.5 px-3 text-sm"
                aria-label={`Leave #${channelName}`}
              >
                <LogOut size={14} />
                <span>{busy ? "Leaving..." : "Leave"}</span>
              </button>
            )}
          </>
        ) : (
          <>
            {channel.canManage && (
              <>
                <button
                  type="button"
                  onClick={() => onEdit(channel)}
                  className="btn-ghost inline-flex h-9 items-center gap-1.5 px-3 text-sm"
                  aria-label={`Edit #${channelName}`}
                >
                  <Pencil size={14} />
                  <span>Edit</span>
                </button>
                {!channel.isDefault && (
                  <button
                    type="button"
                    onClick={() => onArchiveChange(channel, true)}
                    disabled={archiving}
                    className="btn-ghost inline-flex h-9 items-center gap-1.5 px-3 text-sm text-text-muted hover:text-danger"
                    aria-label={`Archive #${channelName}`}
                  >
                    <Archive size={14} />
                    <span>{archiving ? "Archiving..." : "Archive"}</span>
                  </button>
                )}
              </>
            )}
            <button
              type="button"
              onClick={() => onJoin(channel)}
              disabled={busy || isPrivate}
              className="btn-primary inline-flex h-9 items-center gap-1.5 px-3 text-sm"
              aria-label={
                isPrivate
                  ? `#${channelName} is invite only`
                  : `Join #${channelName}`
              }
              title={isPrivate ? "Invite only" : undefined}
            >
              <Plus size={14} />
              <span>
                {busy ? "Joining..." : isPrivate ? "Invite only" : "Join"}
              </span>
            </button>
          </>
        )}
      </div>
    </article>
  );
}
