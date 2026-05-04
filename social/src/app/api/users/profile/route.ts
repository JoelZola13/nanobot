import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

// GET /api/users/profile?userId=id or ?username=name — fetch a compact profile card
export async function GET(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const userId = req.nextUrl.searchParams.get("userId")?.trim();
  const username = req.nextUrl.searchParams.get("username")?.trim();

  if (!userId && !username) {
    return NextResponse.json(
      { error: "userId or username is required" },
      { status: 400 },
    );
  }

  const user = await prisma.user.findFirst({
    where: userId ? { id: userId } : { username },
    select: {
      id: true,
      username: true,
      displayName: true,
      avatarUrl: true,
      bio: true,
      location: true,
      website: true,
      status: true,
      isAgent: true,
      createdAt: true,
      _count: {
        select: {
          channelMembers: true,
          feedPosts: true,
        },
      },
    },
  });

  if (!user) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  return NextResponse.json({
    id: user.id,
    username: user.username,
    displayName: user.displayName,
    avatarUrl: user.avatarUrl,
    bio: user.bio,
    location: user.location,
    website: user.website,
    status: user.status,
    isAgent: user.isAgent,
    createdAt: user.createdAt.toISOString(),
    channelCount: user._count.channelMembers,
    postCount: user._count.feedPosts,
  });
}
