import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

// GET /api/search?q=term — search messages across user's channels
export async function GET(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const q = req.nextUrl.searchParams.get("q")?.trim();
  if (!q || q.length < 2)
    return NextResponse.json({ results: [] });

  // Get user's channel IDs
  const memberships = await prisma.channelMember.findMany({
    where: { userId: session.user.id },
    select: { channelId: true },
  });
  const channelIds = memberships.map((m) => m.channelId);

  if (channelIds.length === 0)
    return NextResponse.json({ results: [] });

  const messages = await prisma.message.findMany({
    where: {
      channelId: { in: channelIds },
      deletedAt: null,
      content: { contains: q, mode: "insensitive" },
    },
    include: {
      author: { select: { id: true, username: true, displayName: true, avatarUrl: true, isAgent: true } },
      channel: { select: { id: true, name: true, slug: true, type: true } },
    },
    orderBy: { createdAt: "desc" },
    take: 20,
  });

  return NextResponse.json({
    results: messages.map((msg) => ({
      id: msg.id,
      channelId: msg.channelId,
      content: msg.content,
      createdAt: msg.createdAt.toISOString(),
      author: msg.author,
      channel: msg.channel,
    })),
  });
}
