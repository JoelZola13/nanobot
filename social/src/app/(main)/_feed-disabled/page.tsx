import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import TopBar from "@/components/layout/TopBar";
import FeedView from "@/components/feed/FeedView";

export default async function FeedPage() {
  const session = await auth();
  const userId = session?.user?.id || "";

  const posts = await prisma.feedPost.findMany({
    where: { deletedAt: null, visibility: "public" },
    include: {
      author: {
        select: {
          id: true,
          username: true,
          displayName: true,
          avatarUrl: true,
        },
      },
      media: { orderBy: { order: "asc" } },
      _count: { select: { likes: true, comments: true } },
      likes: {
        where: { userId },
        select: { id: true },
        take: 1,
      },
    },
    orderBy: { createdAt: "desc" },
    take: 20,
  });

  const formatted = posts.map((p) => ({
    id: p.id,
    content: p.content,
    createdAt: p.createdAt.toISOString(),
    author: p.author,
    likeCount: p._count.likes,
    commentCount: p._count.comments,
    userLiked: p.likes.length > 0,
    media: p.media.map((m) => ({
      url: m.url,
      mimeType: m.mimeType,
      width: m.width,
      height: m.height,
    })),
  }));

  return (
    <>
      <TopBar title="Feed" type="feed" />
      <FeedView initialPosts={formatted} userId={userId} />
    </>
  );
}
