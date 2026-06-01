import { notFound } from "next/navigation";
import { prisma } from "@/lib/prisma";
import { auth } from "@/lib/session";
import TopBar from "@/components/layout/TopBar";
import ProfileSummary from "@/components/users/ProfileSummary";

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
        <ProfileSummary
          user={{
            id: user.id,
            username: user.username,
            displayName: user.displayName,
            avatarUrl: user.avatarUrl,
            bannerUrl: user.bannerUrl,
            bio: user.bio,
            location: user.location,
            website: user.website,
            isAgent: user.isAgent,
            createdAt: user.createdAt.toISOString(),
            postCount: user._count.feedPosts,
            channelCount: user._count.channelMembers,
          }}
          isOwnProfile={isOwnProfile}
        />
      </div>
    </>
  );
}
