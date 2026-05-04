"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import TopBar from "@/components/layout/TopBar";
import {
  Hash,
  Lock,
  MessageSquare,
  Plus,
  ShieldCheck,
  Users,
} from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";

type ChannelSummary = {
  id: string;
  name: string | null;
  slug: string | null;
  description: string | null;
  type: "PUBLIC" | "PRIVATE" | "DM" | "GROUP_DM";
  iconEmoji: string | null;
  isDefault?: boolean;
  memberCount?: number;
  messageCount?: number;
  role?: string;
};

const formatCount = (count: number | undefined, label: string) => {
  const value = count ?? 0;
  return `${value} ${label}${value === 1 ? "" : "s"}`;
};

export default function ChannelsPage() {
  const router = useRouter();
  const [channels, setChannels] = useState<ChannelSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);

  const loadChannels = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(apiUrl("/api/channels"), { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load channels");

      const data = (await res.json()) as ChannelSummary[];
      setChannels(
        data.filter(
          (channel) =>
            channel.type === "PUBLIC" || channel.type === "PRIVATE",
        ),
      );
    } catch {
      setError("Channels could not load.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadChannels();
  }, [loadChannels]);

  const defaultChannels = useMemo(
    () => channels.filter((channel) => channel.isDefault),
    [channels],
  );
  const customChannels = useMemo(
    () => channels.filter((channel) => !channel.isDefault),
    [channels],
  );

  const handleCreate = async (e: FormEvent) => {
    e.preventDefault();
    if (!name.trim() || creating) return;
    setCreating(true);
    setError(null);

    try {
      const res = await fetch(apiUrl("/api/channels"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description }),
      });
      if (!res.ok) throw new Error("Failed to create channel");

      const channel = await res.json();
      router.push(`/channels/${channel.id}`);
      router.refresh();
    } catch {
      setError("Channel could not be created.");
    } finally {
      setCreating(false);
    }
  };

  return (
    <>
      <TopBar title="Channels" type="channel" />
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto py-8 px-4">
          <div className="flex items-center justify-between mb-6">
            <h2 className="font-heading text-xl font-semibold text-text-primary">
              Your Channels
            </h2>
            <button
              onClick={() => setShowCreate(!showCreate)}
              className="btn-primary flex items-center gap-2 text-sm"
            >
              <Plus size={16} />
              <span>New Channel</span>
            </button>
          </div>

          {showCreate && (
            <form
              onSubmit={handleCreate}
              className="bg-bg-surface border border-border rounded-lg p-6 mb-6 space-y-4"
            >
              <div>
                <label className="block text-sm font-medium text-text-primary mb-1.5">
                  Channel Name
                </label>
                <div className="flex items-center gap-2">
                  <Hash size={16} className="text-text-muted shrink-0" />
                  <input
                    type="text"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    placeholder="e.g. content-team"
                    className="input-field flex-1"
                    autoFocus
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-text-primary mb-1.5">
                  Description{" "}
                  <span className="text-text-muted font-normal">
                    (optional)
                  </span>
                </label>
                <input
                  type="text"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What's this channel about?"
                  className="input-field w-full"
                />
              </div>
              <div className="flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowCreate(false)}
                  className="btn-ghost text-sm"
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

          {error && (
            <div className="mb-4 rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200">
              {error}
            </div>
          )}

          {loading ? (
            <div className="rounded-lg border border-border bg-bg-surface px-4 py-10 text-center text-sm text-text-muted">
              Loading channels...
            </div>
          ) : channels.length > 0 ? (
            <div className="space-y-6">
              <ChannelSection
                title="Default channels"
                channels={defaultChannels}
              />
              <ChannelSection title="Custom channels" channels={customChannels} />
            </div>
          ) : (
            <div className="text-center py-12">
              <Users size={48} className="mx-auto text-text-muted mb-3" />
              <h3 className="font-heading text-lg font-semibold text-text-primary mb-1">
                Create your first channel
              </h3>
              <p className="text-sm text-text-muted">
                Channels are where your team communicates. Create one for each
                team, project, or topic.
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

function ChannelSection({
  title,
  channels,
}: {
  title: string;
  channels: ChannelSummary[];
}) {
  if (channels.length === 0) return null;

  return (
    <section className="space-y-2">
      <h3 className="px-1 text-xs font-semibold uppercase tracking-wide text-text-muted">
        {title}
      </h3>
      <div className="space-y-2">
        {channels.map((channel) => (
          <ChannelRow key={channel.id} channel={channel} />
        ))}
      </div>
    </section>
  );
}

function ChannelRow({ channel }: { channel: ChannelSummary }) {
  const isPrivate = channel.type === "PRIVATE";

  return (
    <Link
      href={`/channels/${channel.id}`}
      className="group flex items-start gap-3 rounded-lg border border-border bg-bg-surface px-4 py-3 transition-colors hover:border-accent/70 hover:bg-bg-hover"
    >
      <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-border bg-bg-base text-text-muted">
        {isPrivate ? <Lock size={17} /> : <Hash size={17} />}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="truncate font-medium text-text-primary">
            {channel.name || "unnamed"}
          </span>
          {channel.isDefault && (
            <span className="inline-flex items-center gap-1 rounded-full border border-accent/40 bg-accent-muted px-2 py-0.5 text-2xs font-semibold text-accent">
              <ShieldCheck size={11} />
              Default
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
        </div>
      </div>
    </Link>
  );
}
