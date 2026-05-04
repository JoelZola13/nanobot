import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import {
  canCreateWorkspaceChannels,
  canManageChannel,
} from "@/lib/channelManagement";

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;
  const body = await request.json();
  const archived = body.archived === true;

  const channel = await prisma.channel.findUnique({
    where: { id: channelId },
    include: {
      _count: { select: { members: true, messages: true } },
      members: {
        where: { userId: session.user.id },
        select: { role: true },
      },
    },
  });

  if (!channel) {
    return NextResponse.json({ error: "Channel not found" }, { status: 404 });
  }

  if (channel.type !== "PUBLIC" && channel.type !== "PRIVATE") {
    return NextResponse.json(
      { error: "Channel cannot be archived" },
      { status: 400 },
    );
  }

  if (channel.isDefault) {
    return NextResponse.json(
      { error: "Default channels cannot be archived" },
      { status: 400 },
    );
  }

  const membership = channel.members[0];
  if (!canManageChannel(session.user, membership?.role)) {
    return NextResponse.json(
      { error: "Channel admin access required" },
      { status: 403 },
    );
  }

  const updated = await prisma.channel.update({
    where: { id: channelId },
    data: { isArchived: archived },
    include: {
      _count: { select: { members: true, messages: true } },
      members: {
        where: { userId: session.user.id },
        select: { role: true },
      },
    },
  });
  const updatedMembership = updated.members[0];

  return NextResponse.json({
    id: updated.id,
    name: updated.name,
    slug: updated.slug,
    description: updated.description,
    type: updated.type,
    iconEmoji: updated.iconEmoji,
    isDefault: updated.isDefault,
    isArchived: updated.isArchived,
    isMember: Boolean(updatedMembership),
    memberCount: updated._count.members,
    messageCount: updated._count.messages,
    role: updatedMembership?.role,
    canCreate: canCreateWorkspaceChannels(session.user),
    canManage:
      canManageChannel(session.user, updatedMembership?.role) &&
      !updated.isDefault,
  });
}
