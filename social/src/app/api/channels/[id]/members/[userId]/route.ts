import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import {
  canManageChannel,
  canRemoveChannelMemberRole,
  normalizeAssignableChannelRole,
} from "@/lib/channelManagement";

const userSelect = {
  id: true,
  username: true,
  displayName: true,
  avatarUrl: true,
  isAgent: true,
  status: true,
} as const;

const memberSelect = {
  id: true,
  channelId: true,
  userId: true,
  role: true,
  joinedAt: true,
  user: {
    select: userSelect,
  },
} as const;

function formatMember(member: {
  channelId: string;
  userId: string;
  role: string;
  joinedAt: Date;
  user: {
    id: string;
    username: string;
    displayName: string;
    avatarUrl: string | null;
    isAgent: boolean;
    status: string;
  };
}) {
  return {
    channelId: member.channelId,
    userId: member.userId,
    role: member.role,
    joinedAt: member.joinedAt.toISOString(),
    user: member.user,
  };
}

async function getChannelWithRequesterRole(channelId: string, userId: string) {
  return prisma.channel.findUnique({
    where: { id: channelId },
    select: {
      id: true,
      type: true,
      isArchived: true,
      isDefault: true,
      members: {
        where: { userId },
        select: { role: true },
      },
    },
  });
}

async function getTargetMembership(channelId: string, userId: string) {
  return prisma.channelMember.findUnique({
    where: { channelId_userId: { channelId, userId } },
    select: memberSelect,
  });
}

export async function PATCH(
  request: Request,
  { params }: { params: Promise<{ id: string; userId: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId, userId } = await params;
  const body = await request.json();
  const role = normalizeAssignableChannelRole(body.role);

  const channel = await getChannelWithRequesterRole(channelId, session.user.id);
  if (!channel || channel.isArchived) {
    return NextResponse.json({ error: "Channel not found" }, { status: 404 });
  }

  if (channel.type !== "PUBLIC" && channel.type !== "PRIVATE") {
    return NextResponse.json(
      { error: "Channel members cannot be managed" },
      { status: 400 },
    );
  }

  const requesterRole = channel.members[0]?.role;
  if (!canManageChannel(session.user, requesterRole)) {
    return NextResponse.json(
      { error: "Channel admin access required" },
      { status: 403 },
    );
  }

  const targetMember = await getTargetMembership(channelId, userId);
  if (!targetMember) {
    return NextResponse.json({ error: "Member not found" }, { status: 404 });
  }

  if (targetMember.role === "owner") {
    return NextResponse.json(
      { error: "Channel owner role cannot be changed" },
      { status: 400 },
    );
  }

  const updatedMember = await prisma.channelMember.update({
    where: { channelId_userId: { channelId, userId } },
    data: { role },
    select: memberSelect,
  });

  return NextResponse.json(formatMember(updatedMember));
}

export async function DELETE(
  _request: Request,
  { params }: { params: Promise<{ id: string; userId: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId, userId } = await params;
  const channel = await getChannelWithRequesterRole(channelId, session.user.id);
  if (!channel || channel.isArchived) {
    return NextResponse.json({ error: "Channel not found" }, { status: 404 });
  }

  if (channel.type !== "PUBLIC" && channel.type !== "PRIVATE") {
    return NextResponse.json(
      { error: "Channel members cannot be managed" },
      { status: 400 },
    );
  }

  if (channel.isDefault) {
    return NextResponse.json(
      { error: "Default channel members cannot be removed" },
      { status: 400 },
    );
  }

  const requesterRole = channel.members[0]?.role;
  if (!canManageChannel(session.user, requesterRole)) {
    return NextResponse.json(
      { error: "Channel admin access required" },
      { status: 403 },
    );
  }

  if (userId === session.user.id) {
    return NextResponse.json(
      { error: "Use Leave channel to remove yourself" },
      { status: 400 },
    );
  }

  const targetMember = await getTargetMembership(channelId, userId);
  if (!targetMember) {
    return NextResponse.json({ error: "Member not found" }, { status: 404 });
  }

  if (!canRemoveChannelMemberRole(targetMember.role)) {
    return NextResponse.json(
      { error: "Channel owner cannot be removed" },
      { status: 400 },
    );
  }

  await prisma.channelMember.deleteMany({
    where: { channelId, userId },
  });

  return NextResponse.json({ channelId, userId, removed: true });
}
