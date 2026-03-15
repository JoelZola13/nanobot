import { notFound } from "next/navigation";
import { prisma } from "@/lib/prisma";
import { auth } from "@/lib/session";
import TopBar from "@/components/layout/TopBar";
import { MapPin, Globe, Bot, Calendar } from "lucide-react";
import { format } from "date-fns";

export default async function ProfilePage({
  params,
}: {
  params: Promise<{ userId: string }>;
}) {
  const { userId } = await params;
  const session = await auth();

  const user = await prisma.user.findUnique({
    where: { id: userId },
    include: {
      _count: {
        select: { feedPosts: true, channelMembers: true },
      },
    },
  });
  if (!user) notFound();

  const isOwnProfile = session?.user?.id === userId;

  return (
    <>
      <TopBar title={user.displayName} type="profile" />
      <div className="flex-1 overflow-y-auto">
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
                user.displayName[0]?.toUpperCase()
              )}
            </div>
            <div className="flex-1 pb-1">
              <h2 className="font-heading text-xl font-bold text-text-primary">
                {user.displayName}
              </h2>
              <div className="flex items-center gap-2">
                <span className="text-sm text-text-muted">
                  @{user.username}
                </span>
                {user.isAgent && <span className="badge-teal">AI Agent</span>}
              </div>
            </div>
            {isOwnProfile ? (
              <button className="btn-ghost text-sm border border-border">
                Edit Profile
              </button>
            ) : (
              <button className="btn-primary text-sm">Connect</button>
            )}
          </div>

          {/* Bio */}
          {user.bio && (
            <p className="text-sm text-text-primary/90 mb-4 leading-relaxed">
              {user.bio}
            </p>
          )}

          {/* Meta */}
          <div className="flex flex-wrap items-center gap-4 text-sm text-text-muted mb-6">
            {user.location && (
              <span className="flex items-center gap-1">
                <MapPin size={14} />
                {user.location}
              </span>
            )}
            {user.website && (
              <a
                href={user.website}
                className="flex items-center gap-1 text-accent hover:underline"
              >
                <Globe size={14} />
                {user.website.replace(/^https?:\/\//, "")}
              </a>
            )}
            <span className="flex items-center gap-1">
              <Calendar size={14} />
              Joined {format(user.createdAt, "MMM yyyy")}
            </span>
          </div>

          {/* Stats */}
          <div className="flex gap-6 mb-8 pb-6 border-b border-border">
            <div>
              <span className="font-heading font-bold text-text-primary">
                {user._count.feedPosts}
              </span>
              <span className="text-sm text-text-muted ml-1">posts</span>
            </div>
            <div>
              <span className="font-heading font-bold text-text-primary">
                {user._count.channelMembers}
              </span>
              <span className="text-sm text-text-muted ml-1">channels</span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
