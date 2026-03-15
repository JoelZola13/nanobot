import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

// GET /api/channels — list user's channels
export async function GET() {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const memberships = await prisma.channelMember.findMany({
    where: { userId: session.user.id },
    include: {
      channel: {
        include: { _count: { select: { members: true, messages: true } } },
      },
    },
    orderBy: { channel: { updatedAt: "desc" } },
  });

  return NextResponse.json(
    memberships.map((m) => ({
      id: m.channel.id,
      name: m.channel.name,
      slug: m.channel.slug,
      description: m.channel.description,
      type: m.channel.type,
      iconEmoji: m.channel.iconEmoji,
      memberCount: m.channel._count.members,
      role: m.role,
    })),
  );
}

// POST /api/channels — create a channel
export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json();
  const { name, description, type = "PUBLIC" } = body;

  if (!name)
    return NextResponse.json(
      { error: "Channel name is required" },
      { status: 400 },
    );

  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");

  const channel = await prisma.channel.create({
    data: {
      name,
      slug,
      description,
      type,
      createdById: session.user.id,
      members: {
        create: {
          userId: session.user.id,
          role: "owner",
        },
      },
    },
    include: { _count: { select: { members: true } } },
  });

  return NextResponse.json(channel, { status: 201 });
}
