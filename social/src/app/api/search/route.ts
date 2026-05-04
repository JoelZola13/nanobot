import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import type { Prisma } from "@/generated/prisma/client";

const DATE_RANGES = ["today", "7d", "30d"] as const;
const ATTACHMENT_TYPES = ["file", "image", "document", "audio", "video"] as const;

type DateRange = (typeof DATE_RANGES)[number];
type AttachmentType = (typeof ATTACHMENT_TYPES)[number];

const isDateRange = (value: string | null): value is DateRange =>
  Boolean(value && DATE_RANGES.includes(value as DateRange));

const isAttachmentType = (value: string | null): value is AttachmentType =>
  Boolean(value && ATTACHMENT_TYPES.includes(value as AttachmentType));

const getDateStart = (range: DateRange) => {
  const now = new Date();

  if (range === "today") {
    return new Date(now.getFullYear(), now.getMonth(), now.getDate());
  }

  const days = range === "7d" ? 7 : 30;
  return new Date(now.getTime() - days * 24 * 60 * 60 * 1000);
};

const getAttachmentFilter = (
  attachmentType: AttachmentType,
): Prisma.AttachmentListRelationFilter => {
  if (attachmentType === "file") return { some: {} };
  if (attachmentType === "image") {
    return { some: { mimeType: { startsWith: "image/" } } };
  }
  if (attachmentType === "audio") {
    return { some: { mimeType: { startsWith: "audio/" } } };
  }
  if (attachmentType === "video") {
    return { some: { mimeType: { startsWith: "video/" } } };
  }

  return {
    some: {
      NOT: [
        { mimeType: { startsWith: "image/" } },
        { mimeType: { startsWith: "audio/" } },
        { mimeType: { startsWith: "video/" } },
      ],
    },
  };
};

// GET /api/search?q=term - search messages across user's channels
export async function GET(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const q = req.nextUrl.searchParams.get("q")?.trim();
  const requestedChannelId = req.nextUrl.searchParams.get("channelId")?.trim();
  const requestedAuthorId = req.nextUrl.searchParams.get("authorId")?.trim();
  const dateRange = req.nextUrl.searchParams.get("date");
  const attachmentType = req.nextUrl.searchParams.get("attachment");

  // Get user's channel IDs
  const memberships = await prisma.channelMember.findMany({
    where: { userId: session.user.id, channel: { isArchived: false } },
    include: {
      channel: {
        include: {
          members: {
            include: {
              user: {
                select: {
                  id: true,
                  displayName: true,
                },
              },
            },
          },
        },
      },
    },
  });
  const channelIds = memberships.map((m) => m.channelId);
  const channels = memberships.map((membership) => {
    const otherMember = membership.channel.members.find(
      (member) => member.userId !== session.user!.id,
    );
    const isDm =
      membership.channel.type === "DM" || membership.channel.type === "GROUP_DM";

    return {
      id: membership.channel.id,
      name: membership.channel.name,
      slug: membership.channel.slug,
      type: membership.channel.type,
      label: isDm
        ? otherMember?.user.displayName || membership.channel.name || "Direct message"
        : `#${membership.channel.name || membership.channel.slug || "channel"}`,
    };
  });

  if (channelIds.length === 0) {
    return NextResponse.json({
      results: [],
      filters: { channels: [], authors: [] },
    });
  }

  const authors = await prisma.user.findMany({
    where: {
      messages: {
        some: {
          channelId: { in: channelIds },
          deletedAt: null,
        },
      },
    },
    select: {
      id: true,
      displayName: true,
      username: true,
      isAgent: true,
    },
    orderBy: { displayName: "asc" },
    take: 100,
  });

  const filters = {
    channels,
    authors: authors.map((author) => ({
      id: author.id,
      label: author.displayName,
      username: author.username,
      isAgent: author.isAgent,
    })),
  };

  const requestedChannelIsValid = requestedChannelId
    ? channelIds.includes(requestedChannelId)
    : true;
  const selectedChannelIds =
    requestedChannelId && requestedChannelIsValid
      ? [requestedChannelId]
      : channelIds;
  const validAuthorId = requestedAuthorId || null;
  const validDateRange = isDateRange(dateRange) ? dateRange : null;
  const validAttachmentType = isAttachmentType(attachmentType)
    ? attachmentType
    : null;
  const hasSearchTerm = Boolean(q && q.length >= 2);
  const hasFilters = Boolean(
    requestedChannelId ||
      validAuthorId ||
      validDateRange ||
      validAttachmentType,
  );

  if (!hasSearchTerm && !hasFilters) {
    return NextResponse.json({ results: [], filters });
  }

  if (!requestedChannelIsValid) {
    return NextResponse.json({ results: [], filters });
  }

  const andFilters: Prisma.MessageWhereInput[] = [
    {
      channelId: { in: selectedChannelIds },
      deletedAt: null,
    },
  ];

  if (validAuthorId) {
    andFilters.push({ authorId: validAuthorId });
  }

  if (validDateRange) {
    andFilters.push({ createdAt: { gte: getDateStart(validDateRange) } });
  }

  if (validAttachmentType) {
    andFilters.push({ attachments: getAttachmentFilter(validAttachmentType) });
  }

  if (hasSearchTerm && q) {
    andFilters.push({
      OR: [
        { content: { contains: q, mode: "insensitive" } },
        {
          attachments: {
            some: { fileName: { contains: q, mode: "insensitive" } },
          },
        },
      ],
    });
  }

  const messages = await prisma.message.findMany({
    where: { AND: andFilters },
    include: {
      author: { select: { id: true, username: true, displayName: true, avatarUrl: true, isAgent: true } },
      channel: { select: { id: true, name: true, slug: true, type: true } },
      attachments: {
        select: {
          id: true,
          fileName: true,
          mimeType: true,
          url: true,
        },
      },
    },
    orderBy: { createdAt: "desc" },
    take: 50,
  });

  return NextResponse.json({
    filters,
    results: messages.map((msg) => ({
      id: msg.id,
      channelId: msg.channelId,
      content: msg.content,
      createdAt: msg.createdAt.toISOString(),
      author: msg.author,
      channel: msg.channel,
      attachments: msg.attachments,
    })),
  });
}
