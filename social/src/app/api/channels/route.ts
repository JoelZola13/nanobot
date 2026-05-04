import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import {
  DEFAULT_CHANNELS,
  ensureDefaultChannelsForUser,
} from "@/lib/defaultChannels";
import {
  canCreateWorkspaceChannels,
  canManageChannel,
  normalizeChannelDescription,
  normalizeChannelName,
  normalizeChannelVisibility,
} from "@/lib/channelManagement";

const DEFAULT_CHANNEL_ORDER = new Map<string, number>(
  DEFAULT_CHANNELS.map((channel, index) => [channel.slug, index]),
);

type BrowserChannel = {
  isDefault: boolean;
  isArchived: boolean;
  slug: string | null;
  updatedAt: Date;
};

function sortBrowserChannels<T extends BrowserChannel>(channels: T[]) {
  return [...channels].sort((a, b) => {
    if (a.isArchived !== b.isArchived) return a.isArchived ? 1 : -1;
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
export async function GET(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  await ensureDefaultChannelsForUser(session.user.id);
  const canCreate = canCreateWorkspaceChannels(session.user);
  const includeArchived =
    req.nextUrl.searchParams.get("includeArchived") === "true";

  const channels = await prisma.channel.findMany({
    where: includeArchived
      ? canCreate
        ? { type: { in: ["PUBLIC", "PRIVATE"] } }
        : {
            type: { in: ["PUBLIC", "PRIVATE"] },
            members: {
              some: {
                userId: session.user.id,
                role: { in: ["owner", "admin"] },
              },
            },
          }
      : {
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
        isArchived: channel.isArchived,
        isMember: Boolean(membership),
        memberCount: channel._count.members,
        messageCount: channel._count.messages,
        role: membership?.role,
        canCreate,
        canManage: canManageChannel(session.user, membership?.role) && !channel.isDefault,
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
  if (!canCreateWorkspaceChannels(session.user)) {
    return NextResponse.json(
      { error: "Workspace admin access required" },
      { status: 403 },
    );
  }

  const name = normalizeChannelName(body.name);
  const description = normalizeChannelDescription(body.description);
  const type = normalizeChannelVisibility(body.type);

  if (!name)
    return NextResponse.json(
      { error: "Channel name is required" },
      { status: 400 },
    );

  const existing = await prisma.channel.findUnique({
    where: { slug: name },
    select: { id: true },
  });
  if (existing) {
    return NextResponse.json(
      { error: "A channel with that name already exists" },
      { status: 409 },
    );
  }

  const channel = await prisma.channel.create({
    data: {
      name,
      slug: name,
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
    include: { _count: { select: { members: true, messages: true } } },
  });

  return NextResponse.json(
    {
      id: channel.id,
      name: channel.name,
      slug: channel.slug,
      description: channel.description,
      type: channel.type,
      iconEmoji: channel.iconEmoji,
      isDefault: channel.isDefault,
      isArchived: channel.isArchived,
      isMember: true,
      memberCount: channel._count.members,
      messageCount: channel._count.messages,
      role: "owner",
      canCreate: true,
      canManage: true,
    },
    { status: 201 },
  );
}
