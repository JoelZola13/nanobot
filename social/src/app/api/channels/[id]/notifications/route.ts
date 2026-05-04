import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import {
  isNotificationLevel,
  normalizeNotificationLevel,
} from "@/lib/notificationPreferences";

async function getMembership(userId: string, channelId: string) {
  return prisma.channelMember.findUnique({
    where: {
      channelId_userId: {
        channelId,
        userId,
      },
    },
    include: {
      channel: {
        select: {
          id: true,
          name: true,
          type: true,
          isArchived: true,
        },
      },
    },
  });
}

function serializePreference(
  membership: NonNullable<Awaited<ReturnType<typeof getMembership>>>,
) {
  return {
    channelId: membership.channelId,
    channelName: membership.channel.name,
    channelType: membership.channel.type,
    level: normalizeNotificationLevel(membership.notificationLevel, membership.mutedAt),
    mutedAt: membership.mutedAt?.toISOString() || null,
  };
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;
  const membership = await getMembership(session.user.id, channelId);

  if (!membership || membership.channel.isArchived) {
    return NextResponse.json({ error: "Channel not found" }, { status: 404 });
  }

  return NextResponse.json(serializePreference(membership));
}

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;
  const body = await req.json();
  const level = body?.level;

  if (!isNotificationLevel(level)) {
    return NextResponse.json(
      { error: "Notification level is invalid" },
      { status: 400 },
    );
  }

  const membership = await getMembership(session.user.id, channelId);
  if (!membership || membership.channel.isArchived) {
    return NextResponse.json({ error: "Channel not found" }, { status: 404 });
  }

  const updatedMembership = await prisma.channelMember.update({
    where: {
      channelId_userId: {
        channelId,
        userId: session.user.id,
      },
    },
    data: {
      notificationLevel: level,
      mutedAt: level === "MUTED" ? new Date() : null,
    },
    include: {
      channel: {
        select: {
          id: true,
          name: true,
          type: true,
          isArchived: true,
        },
      },
    },
  });

  return NextResponse.json(serializePreference(updatedMembership));
}
