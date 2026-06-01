"use client";

import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { useRouter } from "next/navigation";
import {
  Bot,
  CheckCircle2,
  MapPin,
  MessageSquare,
  Search,
  Sparkles,
  UserPlus,
  UserRound,
  X,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";
import { useEmbeddedNavigation } from "@/lib/useEmbeddedNavigation";
import ProfilePopover from "@/components/users/ProfilePopover";

interface Person {
  id: string;
  username: string;
  displayName: string;
  avatarUrl: string | null;
  bio: string | null;
  status: string;
  location: string | null;
  lastSeenAt: string | null;
  createdAt: string;
  isAgent?: boolean;
}

type DirectoryFilter = "all" | "teammates" | "agents";

const RECENT_TEAMMATE_WINDOW_MS = 30 * 24 * 60 * 60 * 1000;

const isAwaitingFirstSignIn = (person: Person) =>
  !person.isAgent && !person.lastSeenAt && person.status !== "online";

const isRecentlyAdded = (person: Person) => {
  if (person.isAgent) return false;
  const createdAt = new Date(person.createdAt).getTime();
  return (
    Number.isFinite(createdAt) &&
    Date.now() - createdAt <= RECENT_TEAMMATE_WINDOW_MS
  );
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

export default function PeopleList({
  people,
  agents = [],
  currentUserId: _currentUserId,
  initialFilter = "all",
}: {
  people: Person[];
  agents?: Person[];
  currentUserId: string;
  initialFilter?: DirectoryFilter;
}) {
  const router = useRouter();
  const { withEmbed } = useEmbeddedNavigation();
  const [starting, setStarting] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [directoryFilter, setDirectoryFilter] =
    useState<DirectoryFilter>(initialFilter);
  const [dmError, setDmError] = useState<{
    userId: string;
    message: string;
  } | null>(null);

  useEffect(() => {
    setDirectoryFilter(initialFilter);
    setDmError(null);
  }, [initialFilter]);

  const updateDirectoryFilter = (filter: DirectoryFilter) => {
    setDirectoryFilter(filter);
    setDmError(null);
    router.replace(
      withEmbed(filter === "all" ? "/dm" : `/dm?filter=${filter}`),
      {
        scroll: false,
      },
    );
  };

  const startDM = async (userId: string, displayName: string) => {
    if (starting) return;
    setStarting(userId);
    setDmError(null);

    try {
      const res = await fetch(apiUrl("/api/dm"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId }),
      });

      if (!res.ok) {
        throw new Error(
          await getApiErrorMessage(
            res,
            `Could not open a direct message with ${displayName}.`,
          ),
        );
      }

      const { channelId } = (await res.json()) as { channelId?: string };
      if (!channelId) {
        throw new Error("Direct message was missing a channel id.");
      }

      router.push(withEmbed(`/dm/${channelId}`));
      router.refresh();
    } catch (error) {
      setDmError({
        userId,
        message:
          error instanceof Error
            ? error.message
            : `Could not open a direct message with ${displayName}.`,
      });
    } finally {
      setStarting(null);
    }
  };

  const normalizedQuery = query.trim().toLowerCase();
  const matches = (person: Person) => {
    if (!normalizedQuery) return true;
    return [person.displayName, person.username, person.bio, person.location]
      .filter(Boolean)
      .some((value) => value!.toLowerCase().includes(normalizedQuery));
  };

  const visiblePeople = directoryFilter === "agents" ? [] : people;
  const online = visiblePeople.filter(
    (p) => p.status === "online" && matches(p),
  );
  const offline = visiblePeople.filter(
    (p) => p.status !== "online" && matches(p),
  );
  const visibleAgents =
    directoryFilter === "teammates" ? [] : agents.filter(matches);
  const visibleCount = online.length + offline.length + visibleAgents.length;
  const onboardingCount = people.filter(isAwaitingFirstSignIn).length;
  const recentlyAddedCount = people.filter(isRecentlyAdded).length;
  const hasQuery = query.trim().length > 0;
  const clearDirectorySearch = () => {
    setQuery("");
    setDmError(null);
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-5">
        <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="min-w-0">
            <h2 className="font-heading text-xl font-semibold text-text-primary">
              {directoryFilter === "agents"
                ? "AI agents"
                : directoryFilter === "teammates"
                  ? "Teammates"
                  : "Direct messages"}
            </h2>
            <p className="text-sm text-text-muted">
              {directoryFilter === "agents"
                ? `${agents.length} agents`
                : directoryFilter === "teammates"
                  ? `${people.length} teammates`
                  : `${people.length} teammates, ${agents.length} agents`}
            </p>
          </div>
          <div className="flex w-full flex-col gap-2 md:w-auto md:flex-row md:items-center">
            <div
              className="inline-flex rounded-lg border border-border bg-bg-surface p-1"
              role="tablist"
              aria-label="Direct message directory filter"
              data-testid="dm-directory-filter"
            >
              <DirectoryFilterButton
                active={directoryFilter === "all"}
                label="All"
                ariaLabel="Show all direct message contacts"
                onClick={() => updateDirectoryFilter("all")}
              />
              <DirectoryFilterButton
                active={directoryFilter === "teammates"}
                label="Teammates"
                ariaLabel="Show teammates"
                onClick={() => updateDirectoryFilter("teammates")}
              />
              <DirectoryFilterButton
                active={directoryFilter === "agents"}
                label="Agents"
                ariaLabel="Show AI agents"
                onClick={() => updateDirectoryFilter("agents")}
              />
            </div>
            <div className="relative w-full md:w-80">
              <Search
                size={15}
                className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted"
              />
              <input
                type="search"
                value={query}
                onChange={(event) => {
                  setQuery(event.target.value);
                  setDmError(null);
                }}
                placeholder={
                  directoryFilter === "agents"
                    ? "Search agents"
                    : directoryFilter === "teammates"
                      ? "Search teammates"
                      : "Search people and agents"
                }
                className="input-field w-full pl-9 pr-16 text-sm"
                aria-label={
                  directoryFilter === "agents"
                    ? "Search agents"
                    : directoryFilter === "teammates"
                      ? "Search teammates"
                      : "Search people and agents"
                }
                aria-controls="dm-directory-results"
                data-testid="dm-directory-search"
              />
              {hasQuery && (
                <button
                  type="button"
                  onClick={clearDirectorySearch}
                  className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md px-2 py-1 text-xs font-medium text-text-muted transition-colors hover:bg-bg-hover hover:text-text-primary"
                  aria-label="Clear people search"
                >
                  Clear
                </button>
              )}
            </div>
          </div>
        </div>

        {onboardingCount > 0 && directoryFilter === "all" && (
          <div className="mb-4 border-y border-border bg-bg-surface px-4 py-3">
            <div className="flex flex-col gap-3 md:flex-row md:items-center">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-border bg-bg-elevated text-accent">
                <UserPlus size={17} />
              </div>
              <div className="min-w-0 flex-1">
                <div className="break-words text-sm font-semibold leading-5 text-text-primary">
                  {onboardingCount} teammate{onboardingCount === 1 ? "" : "s"}{" "}
                  waiting on first sign-in
                </div>
                <div className="mt-0.5 break-words text-xs leading-5 text-text-muted">
                  Share LibChatMain + NanobotMain, then have them sign in to
                  LibreChat once. Messages uses that same session.
                </div>
              </div>
              {recentlyAddedCount > 0 && (
                <div className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border px-2 py-1 text-2xs font-semibold uppercase text-text-muted">
                  <CheckCircle2 size={12} className="text-teal" />
                  {recentlyAddedCount} added recently
                </div>
              )}
            </div>
          </div>
        )}

        <div
          id="dm-directory-results"
          className="overflow-hidden rounded-lg border border-border bg-bg-surface"
          aria-live="polite"
          data-testid="dm-directory-results"
        >
          {visibleCount === 0 ? (
            <div
              className="px-5 py-10 text-center"
              data-testid="dm-directory-empty"
            >
              <UserRound size={32} className="mx-auto mb-2 text-text-muted" />
              <h3 className="font-heading text-base font-semibold text-text-primary">
                No matches
              </h3>
              <p className="text-sm text-text-muted">
                {directoryFilter === "agents"
                  ? "Try a different agent name or role."
                  : directoryFilter === "teammates"
                    ? "Try a different teammate name, username, or location."
                    : "Try a different name, username, location, or agent."}
              </p>
              {hasQuery && (
                <button
                  type="button"
                  onClick={clearDirectorySearch}
                  className="btn-secondary mt-4 text-sm"
                  aria-label="Clear directory search"
                  data-testid="dm-directory-empty-clear"
                >
                  Clear search
                </button>
              )}
            </div>
          ) : (
            <>
              <DirectorySection
                title="Online"
                count={online.length}
                icon={<span className="h-2 w-2 rounded-full bg-teal" />}
                people={online}
                onMessage={startDM}
                loadingId={starting}
                error={dmError}
                onDismissError={() => setDmError(null)}
              />
              <DirectorySection
                title="Teammates"
                count={offline.length}
                icon={<UserRound size={13} />}
                people={offline}
                onMessage={startDM}
                loadingId={starting}
                error={dmError}
                onDismissError={() => setDmError(null)}
              />
              <DirectorySection
                title="AI agents"
                count={visibleAgents.length}
                icon={<Sparkles size={13} className="text-teal" />}
                people={visibleAgents}
                onMessage={startDM}
                loadingId={starting}
                error={dmError}
                onDismissError={() => setDmError(null)}
                isAgentSection
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function DirectoryFilterButton({
  active,
  label,
  ariaLabel,
  onClick,
}: {
  active: boolean;
  label: string;
  ariaLabel: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-label={ariaLabel}
      aria-selected={active}
      onClick={onClick}
      className={`rounded-md px-3 py-1.5 text-xs font-semibold transition-colors ${
        active
          ? "bg-accent text-[#1a1c24]"
          : "text-text-muted hover:bg-bg-hover hover:text-text-primary"
      }`}
    >
      {label}
    </button>
  );
}

function DirectorySection({
  title,
  count,
  icon,
  people,
  onMessage,
  loadingId,
  error,
  onDismissError,
  isAgentSection = false,
}: {
  title: string;
  count: number;
  icon: ReactNode;
  people: Person[];
  onMessage: (id: string, displayName: string) => void;
  loadingId: string | null;
  error: { userId: string; message: string } | null;
  onDismissError: () => void;
  isAgentSection?: boolean;
}) {
  if (count === 0) return null;

  return (
    <section className="border-t border-border first:border-t-0">
      <div className="flex items-center gap-2 bg-bg-elevated px-4 py-2 text-xs font-semibold uppercase text-text-muted">
        {icon}
        <span>{title}</span>
        <span className="ml-auto">{count}</span>
      </div>
      <div>
        {people.map((person) => (
          <PersonRow
            key={person.id}
            person={person}
            onMessage={onMessage}
            loading={loadingId === person.id}
            error={error?.userId === person.id ? error.message : null}
            onDismissError={onDismissError}
            isAgent={isAgentSection || person.isAgent}
          />
        ))}
      </div>
    </section>
  );
}

function PersonRow({
  person,
  onMessage,
  loading,
  error,
  onDismissError,
  isAgent = false,
}: {
  person: Person;
  onMessage: (id: string, displayName: string) => void;
  loading: boolean;
  error: string | null;
  onDismissError: () => void;
  isAgent?: boolean;
}) {
  const isOnline = person.status === "online" || isAgent;
  const awaitingFirstSignIn = isAwaitingFirstSignIn(person);
  const recentlyAdded = isRecentlyAdded(person);
  const statusText = isAgent
    ? "Active"
    : awaitingFirstSignIn
      ? "Needs first sign-in"
      : isOnline
        ? "Active"
        : "Away";

  return (
    <div
      className="group border-t border-border first:border-t-0"
      data-testid="dm-directory-row"
      data-user-id={person.id}
      aria-label={`${person.displayName} directory row`}
    >
      <div className="flex items-center gap-3 px-4 py-3 hover:bg-bg-hover">
        <ProfilePopover
          user={person}
          className="min-w-0 flex-1"
          triggerClassName="flex min-w-0 flex-1 items-center gap-3 rounded-md text-left focus:outline-none focus:ring-2 focus:ring-accent/40"
        >
          <span className="relative shrink-0">
            <span
              className={`h-10 w-10 avatar text-sm ${isAgent ? "bg-teal-muted text-teal" : "bg-accent-muted text-accent"}`}
            >
              {person.avatarUrl ? (
                <img
                  src={person.avatarUrl}
                  alt=""
                  className="h-full w-full rounded-full object-cover"
                />
              ) : isAgent ? (
                <Bot size={18} />
              ) : (
                person.displayName[0]?.toUpperCase()
              )}
            </span>
            <span
              className={`absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-bg ${isOnline ? "bg-teal" : "bg-border"}`}
            />
          </span>

          <span className="min-w-0 flex-1">
            <span className="flex min-w-0 items-center gap-2">
              <span
                className={`truncate text-sm font-semibold ${isAgent ? "text-teal" : "text-text-primary"}`}
              >
                {person.displayName}
              </span>
              {isAgent ? (
                <span className="badge-teal text-2xs">agent</span>
              ) : recentlyAdded ? (
                <span className="rounded-full bg-accent-muted px-1.5 py-0.5 text-2xs font-semibold uppercase text-accent">
                  New
                </span>
              ) : (
                <span className="truncate text-2xs text-text-muted">
                  @{person.username}
                </span>
              )}
              {!isAgent && recentlyAdded && (
                <span className="truncate text-2xs text-text-muted">
                  @{person.username}
                </span>
              )}
            </span>
            <span className="mt-0.5 flex min-w-0 items-center gap-2 text-xs text-text-muted">
              <span>{statusText}</span>
              {person.location && !isAgent && (
                <>
                  <span>/</span>
                  <span className="inline-flex min-w-0 items-center gap-1 truncate">
                    <MapPin size={11} className="shrink-0" />
                    <span className="truncate">{person.location}</span>
                  </span>
                </>
              )}
              {person.bio && (
                <>
                  <span>/</span>
                  <span className="truncate">{person.bio}</span>
                </>
              )}
            </span>
            {awaitingFirstSignIn && (
              <span className="mt-1 block break-words text-2xs leading-4 text-text-muted">
                LibreChat sign-in unlocks Messages automatically.
              </span>
            )}
          </span>
        </ProfilePopover>

        <button
          data-testid="start-dm-button"
          data-user-id={person.id}
          onClick={() => onMessage(person.id, person.displayName)}
          disabled={loading}
          className="btn-ghost flex h-8 items-center gap-1.5 px-2 text-xs disabled:cursor-wait disabled:opacity-70 md:opacity-0 md:group-hover:opacity-100 md:focus:opacity-100"
          type="button"
          aria-label={`${loading ? "Opening" : "Message"} ${person.displayName}`}
        >
          <MessageSquare size={13} />
          <span>{loading ? "Opening" : "Message"}</span>
        </button>
      </div>
      {error && (
        <div
          className="mx-4 mb-3 flex items-start justify-between gap-3 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
          role="status"
          data-testid="dm-directory-action-error"
        >
          <span className="min-w-0 flex-1">{error}</span>
          <button
            type="button"
            onClick={onDismissError}
            className="rounded-md p-0.5 text-red-700 hover:bg-red-100 dark:text-red-200 dark:hover:bg-red-900/30"
            aria-label={`Dismiss direct message error for ${person.displayName}`}
          >
            <X size={13} />
          </button>
        </div>
      )}
    </div>
  );
}
