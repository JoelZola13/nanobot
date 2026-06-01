"use client";

import { useState } from "react";
import { Bot, Calendar, Globe, MapPin } from "lucide-react";
import { format } from "date-fns";
import ProfileActions from "./ProfileActions";

type ProfileSummaryProps = {
  user: {
    id: string;
    username: string;
    displayName: string;
    avatarUrl: string | null;
    bannerUrl: string | null;
    bio: string | null;
    location: string | null;
    website: string | null;
    isAgent: boolean;
    createdAt: string;
    postCount: number;
    channelCount: number;
  };
  isOwnProfile: boolean;
};

export default function ProfileSummary({
  user,
  isOwnProfile,
}: ProfileSummaryProps) {
  const [profile, setProfile] = useState({
    displayName: user.displayName,
    bio: user.bio,
    location: user.location,
    website: user.website,
  });

  return (
    <>
      {/* Banner */}
      <div className="h-40 bg-gradient-to-br from-accent/20 to-teal/10 relative">
        {user.bannerUrl && (
          <img
            src={user.bannerUrl}
            alt=""
            className="w-full h-full object-cover"
          />
        )}
      </div>

      <div className="max-w-2xl mx-auto px-4">
        {/* Avatar + name */}
        <div className="flex items-end gap-4 -mt-10 mb-4">
          <div
            className={`w-20 h-20 avatar text-2xl border-4 border-bg ${
              user.isAgent
                ? "bg-teal-muted text-teal"
                : "bg-accent-muted text-accent"
            }`}
          >
            {user.isAgent ? (
              <Bot size={32} />
            ) : (
              profile.displayName[0]?.toUpperCase()
            )}
          </div>
          <div className="flex-1 pb-1">
            <h2
              data-testid="profile-display-name"
              className="font-heading text-xl font-bold text-text-primary"
            >
              {profile.displayName}
            </h2>
            <div className="flex items-center gap-2">
              <span className="text-sm text-text-muted">@{user.username}</span>
              {user.isAgent && <span className="badge-teal">AI Agent</span>}
            </div>
          </div>
          <ProfileActions
            userId={user.id}
            isOwnProfile={isOwnProfile}
            initialProfile={profile}
            onSaved={setProfile}
          />
        </div>

        {/* Bio */}
        {profile.bio && (
          <p className="text-sm text-text-primary/90 mb-4 leading-relaxed">
            {profile.bio}
          </p>
        )}

        {/* Meta */}
        <div className="flex flex-wrap items-center gap-4 text-sm text-text-muted mb-6">
          {profile.location && (
            <span className="flex items-center gap-1">
              <MapPin size={14} />
              {profile.location}
            </span>
          )}
          {profile.website && (
            <a
              href={profile.website}
              className="flex items-center gap-1 text-accent hover:underline"
            >
              <Globe size={14} />
              {profile.website.replace(/^https?:\/\//, "")}
            </a>
          )}
          <span className="flex items-center gap-1">
            <Calendar size={14} />
            Joined {format(new Date(user.createdAt), "MMM yyyy")}
          </span>
        </div>

        {/* Stats */}
        <div className="flex gap-6 mb-8 pb-6 border-b border-border">
          <div>
            <span className="font-heading font-bold text-text-primary">
              {user.postCount}
            </span>
            <span className="text-sm text-text-muted ml-1">posts</span>
          </div>
          <div>
            <span className="font-heading font-bold text-text-primary">
              {user.channelCount}
            </span>
            <span className="text-sm text-text-muted ml-1">channels</span>
          </div>
        </div>
      </div>
    </>
  );
}
