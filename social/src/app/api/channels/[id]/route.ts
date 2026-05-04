import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import {
  canCreateWorkspaceChannels,
  canManageChannel,
  normalizeChannelDescription,
  normalizeChannelName,
  normalizeChannelVisibility,
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

  if (!channel || channel.isArchived) {
    return NextResponse.json({ error: "Channel not found" }, { status: 404 });
  }

  if (channel.type !== "PUBLIC" && channel.type !== "PRIVATE") {
    return NextResponse.json(
      { error: "Channel cannot be edited" },
      { status: 400 },
    );
  }

  if (channel.isDefault) {
    return NextResponse.json(
      { error: "Default channels cannot be edited" },
      { status: 400 },
    );
  }

  const membership = channel.members[0];
  if (!canManageChannel(session.user, membership?.role)) {
    return NextResponse.json(
      { error: "Workspace admin access required" },
      { status: 403 },
    );
  }

  const name = normalizeChannelName(body.name);
  if (!name) {
    return NextResponse.json(
      { error: "Channel name is required" },
      { status: 400 },
    );
  }

  const existing = await prisma.channel.findUnique({
    where: { slug: name },
    select: { id: true },
  });
  if (existing && existing.id !== channel.id) {
    return NextResponse.json(
      { error: "A channel with that name already exists" },
      { status: 409 },
    );
  }

  const updated = await prisma.channel.update({
    where: { id: channelId },
    data: {
      name,
      slug: name,
      description: normalizeChannelDescription(body.description),
      type: normalizeChannelVisibility(body.type),
    },
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
    isMember: Boolean(updatedMembership),
    memberCount: updated._count.members,
    messageCount: updated._count.messages,
    role: updatedMembership?.role,
    canCreate: canCreateWorkspaceChannels(session.user),
    canManage: canManageChannel(session.user, updatedMembership?.role) && !updated.isDefault,
  });
}
