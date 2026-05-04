import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import {
  DEFAULT_CHANNELS,
  ensureDefaultChannelsForUser,
} from "@/lib/defaultChannels";

const DEFAULT_CHANNEL_ORDER = new Map<string, number>(
  DEFAULT_CHANNELS.map((channel, index) => [channel.slug, index]),
);

type BrowserChannel = {
  isDefault: boolean;
  slug: string | null;
  updatedAt: Date;
};

function sortBrowserChannels<T extends BrowserChannel>(channels: T[]) {
  return [...channels].sort((a, b) => {
    if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1;

    if (a.isDefault && b.isDefault) {
      const aRank =
        DEFAULT_CHANNEL_ORDER.get(a.slug || "") ?? Number.MAX_SAFE_INTEGER;
      const bRank =
        DEFAULT_CHANNEL_ORDER.get(b.slug || "") ?? Number.MAX_SAFE_INTEGER;
      if (aRank !== bRank) return aRank - bRank;
    }

    return b.updatedAt.getTime() - a.updatedAt.getTime();
  });
}

// GET /api/channels — browse public channels plus joined private channels
export async function GET() {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  await ensureDefaultChannelsForUser(session.user.id);

  const channels = await prisma.channel.findMany({
    where: {
      isArchived: false,
      OR: [
        { type: "PUBLIC" },
        {
          type: "PRIVATE",
          members: { some: { userId: session.user.id } },
        },
      ],
    },
    include: {
      _count: { select: { members: true, messages: true } },
      members: {
        where: { userId: session.user.id },
        select: { role: true },
      },
    },
    orderBy: { updatedAt: "desc" },
  });

  return NextResponse.json(
    sortBrowserChannels(channels).map((channel) => {
      const membership = channel.members[0];

      return {
        id: channel.id,
        name: channel.name,
        slug: channel.slug,
        description: channel.description,
        type: channel.type,
        iconEmoji: channel.iconEmoji,
        isDefault: channel.isDefault,
        isMember: Boolean(membership),
        memberCount: channel._count.members,
        messageCount: channel._count.messages,
        role: membership?.role,
      };
    }),
  );
}

// POST /api/channels — create a channel
export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const body = await req.json();
  const { name, description } = body;
  const type = body.type === "PRIVATE" ? "PRIVATE" : "PUBLIC";

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
