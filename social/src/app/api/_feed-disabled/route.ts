import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

// GET /api/feed — list feed posts
export async function GET(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { searchParams } = new URL(req.url);
  const cursor = searchParams.get("cursor");
  const limit = Math.min(parseInt(searchParams.get("limit") || "20"), 50);

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
        where: { userId: session.user.id },
        select: { id: true },
        take: 1,
      },
    },
    orderBy: { createdAt: "desc" },
    take: limit,
    ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
  });

  return NextResponse.json({
    posts: posts.map((p) => ({
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
    })),
    nextCursor:
      posts.length === limit ? posts[posts.length - 1].id : null,
  });
}

// POST /api/feed — create a post
export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json();
  const { content, visibility = "public" } = body;

  if (!content?.trim())
    return NextResponse.json(
      { error: "Post content required" },
      { status: 400 },
    );

  const post = await prisma.feedPost.create({
    data: {
      authorId: session.user.id,
      content: content.trim(),
      visibility,
    },
    include: {
      author: {
        select: {
          id: true,
          username: true,
          displayName: true,
          avatarUrl: true,
        },
      },
    },
  });

  return NextResponse.json(
    {
      id: post.id,
      content: post.content,
      createdAt: post.createdAt.toISOString(),
      author: post.author,
      likeCount: 0,
      commentCount: 0,
      userLiked: false,
      media: [],
    },
    { status: 201 },
  );
}
