"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import TopBar from "@/components/layout/TopBar";
import { Hash, Plus, Users } from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";

export default function ChannelsPage() {
  const router = useRouter();
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [creating, setCreating] = useState(false);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || creating) return;
    setCreating(true);
    try {
      const res = await fetch(apiUrl("/api/channels"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, description }),
      });
      if (res.ok) {
        const channel = await res.json();
        router.push(`/channels/${channel.id}`);
        router.refresh();
      }
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
              className="bg-bg-surface border border-border rounded-xl p-6 mb-6 space-y-4"
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
        </div>
      </div>
    </>
  );
}
