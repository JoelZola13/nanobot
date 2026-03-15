import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

// POST /api/dm — create or find existing DM channel with another user
export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const { userId: otherUserId } = await req.json();
  if (!otherUserId)
    return NextResponse.json({ error: "userId required" }, { status: 400 });

  const currentUserId = session.user.id;

  // Check if DM already exists between these two users
  const existingDM = await prisma.channel.findFirst({
    where: {
      type: "DM",
      AND: [
        { members: { some: { userId: currentUserId } } },
        { members: { some: { userId: otherUserId } } },
      ],
    },
    include: { _count: { select: { members: true } } },
  });

  if (existingDM) {
    return NextResponse.json({ channelId: existingDM.id });
  }

  // Get the other user's display name for the channel
  const otherUser = await prisma.user.findUnique({
    where: { id: otherUserId },
    select: { displayName: true, username: true },
  });

  if (!otherUser)
    return NextResponse.json({ error: "User not found" }, { status: 404 });

  // Create new DM channel
  const channel = await prisma.channel.create({
    data: {
      name: null, // DMs don't need names, we show the other user's name
      slug: `dm-${currentUserId}-${otherUserId}`,
      type: "DM",
      members: {
        create: [
          { userId: currentUserId, role: "member" },
          { userId: otherUserId, role: "member" },
        ],
      },
    },
  });

  return NextResponse.json({ channelId: channel.id }, { status: 201 });
}
