"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Pencil, Save, X } from "lucide-react";
import { apiUrl } from "@/lib/apiUrl";
import { useEmbeddedNavigation } from "@/lib/useEmbeddedNavigation";

type ProfileActionsProps = {
  userId: string;
  isOwnProfile: boolean;
  initialProfile?: {
    displayName: string;
    bio: string | null;
    location: string | null;
    website: string | null;
  };
  onSaved?: (profile: {
    displayName: string;
    bio: string | null;
    location: string | null;
    website: string | null;
  }) => void;
};

async function apiErrorMessage(res: Response, fallback: string) {
  const data = (await res.json().catch(() => null)) as {
    error?: string;
  } | null;
  return data?.error || fallback;
}

export default function ProfileActions({
  userId,
  isOwnProfile,
  initialProfile,
  onSaved,
}: ProfileActionsProps) {
  const router = useRouter();
  const { withEmbed } = useEmbeddedNavigation();
  const [opening, setOpening] = useState(false);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [profile, setProfile] = useState(
    initialProfile || {
      displayName: "",
      bio: null,
      location: null,
      website: null,
    },
  );
  const [draft, setDraft] = useState(profile);

  useEffect(() => {
    if (!initialProfile) return;
    setProfile(initialProfile);
    setDraft(initialProfile);
  }, [initialProfile]);

  const openDM = async () => {
    if (opening) return;
    setOpening(true);
    setError(null);

    try {
      const res = await fetch(apiUrl("/api/dm"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId }),
      });

      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Could not open a direct message."),
        );
      }

      const { channelId } = (await res.json()) as { channelId?: string };
      if (!channelId) {
        throw new Error("Direct message was missing a channel id.");
      }

      router.push(withEmbed(`/dm/${channelId}`));
      router.refresh();
    } catch (openError) {
      setError(
        openError instanceof Error
          ? openError.message
          : "Could not open a direct message.",
      );
    } finally {
      setOpening(false);
    }
  };

  const updateDraft = (field: keyof typeof draft, value: string) => {
    setDraft((current) => ({
      ...current,
      [field]: value,
    }));
    setError(null);
    setNotice(null);
  };

  const openEditor = () => {
    setDraft(profile);
    setError(null);
    setNotice(null);
    setEditing(true);
  };

  const closeEditor = () => {
    setDraft(profile);
    setError(null);
    setEditing(false);
  };

  const saveProfile = async () => {
    if (saving) return;
    setSaving(true);
    setError(null);
    setNotice(null);

    try {
      const res = await fetch(apiUrl("/api/users/profile"), {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });

      if (!res.ok) {
        throw new Error(
          await apiErrorMessage(res, "Profile could not be updated."),
        );
      }

      const updated = (await res.json()) as typeof profile;
      const nextProfile = {
        displayName: updated.displayName,
        bio: updated.bio,
        location: updated.location,
        website: updated.website,
      };
      setProfile(nextProfile);
      setDraft(nextProfile);
      onSaved?.(nextProfile);
      setEditing(false);
      setNotice("Profile updated");
      router.refresh();
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : "Profile could not be updated.",
      );
    } finally {
      setSaving(false);
    }
  };

  if (isOwnProfile) {
    return (
      <span className="relative flex flex-col items-end gap-1">
        <button
          type="button"
          data-testid="profile-edit-open"
          className="btn-ghost inline-flex items-center gap-1.5 border border-border text-sm"
          onClick={openEditor}
          aria-label="Edit profile"
        >
          <Pencil size={14} />
          Edit Profile
        </button>
        {notice && (
          <span
            data-testid="profile-edit-notice"
            className="text-2xs font-medium text-teal"
          >
            {notice}
          </span>
        )}
        {editing && (
          <span
            role="dialog"
            aria-label="Edit profile"
            data-testid="profile-edit-dialog"
            className="absolute right-0 top-full z-[90] mt-2 block w-80 rounded-lg border border-border bg-bg-surface p-3 text-left shadow-xl"
          >
            <span className="mb-3 flex items-center justify-between gap-3">
              <span className="text-sm font-semibold text-text-primary">
                Edit profile
              </span>
              <button
                type="button"
                onClick={closeEditor}
                className="sidebar-icon-button h-7 w-7"
                aria-label="Close profile editor"
              >
                <X size={14} />
              </button>
            </span>

            <label className="mb-2 block text-xs font-medium text-text-secondary">
              Display name
              <input
                data-testid="profile-edit-display-name"
                value={draft.displayName}
                onChange={(event) =>
                  updateDraft("displayName", event.target.value)
                }
                className="mt-1 w-full rounded-md border border-border bg-bg-base px-2.5 py-2 text-sm text-text-primary outline-none focus:border-accent"
              />
            </label>
            <label className="mb-2 block text-xs font-medium text-text-secondary">
              Bio
              <textarea
                data-testid="profile-edit-bio"
                value={draft.bio || ""}
                onChange={(event) => updateDraft("bio", event.target.value)}
                rows={3}
                className="mt-1 w-full resize-y rounded-md border border-border bg-bg-base px-2.5 py-2 text-sm text-text-primary outline-none focus:border-accent"
              />
            </label>
            <label className="mb-2 block text-xs font-medium text-text-secondary">
              Location
              <input
                data-testid="profile-edit-location"
                value={draft.location || ""}
                onChange={(event) =>
                  updateDraft("location", event.target.value)
                }
                className="mt-1 w-full rounded-md border border-border bg-bg-base px-2.5 py-2 text-sm text-text-primary outline-none focus:border-accent"
              />
            </label>
            <label className="block text-xs font-medium text-text-secondary">
              Website
              <input
                data-testid="profile-edit-website"
                value={draft.website || ""}
                onChange={(event) => updateDraft("website", event.target.value)}
                className="mt-1 w-full rounded-md border border-border bg-bg-base px-2.5 py-2 text-sm text-text-primary outline-none focus:border-accent"
              />
            </label>

            {error && (
              <span
                role="alert"
                data-testid="profile-edit-error"
                className="mt-3 block rounded-md border border-red-300 bg-red-50 px-2.5 py-2 text-xs font-medium text-red-700 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200"
              >
                {error}
              </span>
            )}

            <span className="mt-3 flex justify-end gap-2">
              <button
                type="button"
                onClick={closeEditor}
                className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-text-secondary hover:border-text-muted hover:text-text-primary"
              >
                Cancel
              </button>
              <button
                type="button"
                data-testid="profile-edit-save"
                onClick={() => void saveProfile()}
                disabled={saving || !draft.displayName.trim()}
                className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-xs font-semibold text-white hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-60"
                aria-label={saving ? "Saving profile" : "Save profile"}
              >
                {saving ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Save size={13} />
                )}
                Save
              </button>
            </span>
          </span>
        )}
      </span>
    );
  }

  return (
    <span className="flex flex-col items-end gap-1">
      <button
        type="button"
        data-testid="profile-open-dm"
        className="btn-primary text-sm disabled:cursor-wait disabled:opacity-70"
        onClick={() => void openDM()}
        disabled={opening}
        aria-label={opening ? "Opening direct message" : "Open direct message"}
      >
        {opening ? "Opening" : "Message"}
      </button>
      {error && (
        <span
          data-testid="profile-action-error"
          className="max-w-48 text-right text-2xs font-medium text-danger"
        >
          {error}
        </span>
      )}
    </span>
  );
}
