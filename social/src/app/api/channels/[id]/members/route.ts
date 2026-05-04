import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import {
  canManageChannel,
  normalizeAssignableChannelRole,
} from "@/lib/channelManagement";
import { getDefaultMembershipPreferences } from "@/lib/workspacePolicies";

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

const roleRank = (role: string) => {
  if (role === "owner") return 0;
  if (role === "admin") return 1;
  return 2;
};

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

async function getManageableChannel(channelId: string, userId: string) {
  return prisma.channel.findUnique({
    where: { id: channelId },
    select: {
      id: true,
      type: true,
      isArchived: true,
      members: {
        where: { userId },
        select: { role: true },
      },
    },
  });
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;
  const channel = await getManageableChannel(channelId, session.user.id);

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
  if (!requesterRole && !canManageChannel(session.user, requesterRole)) {
    return NextResponse.json({ error: "Not a member" }, { status: 403 });
  }

  const canManage = canManageChannel(session.user, requesterRole);
  const members = await prisma.channelMember.findMany({
    where: { channelId },
    select: memberSelect,
    orderBy: { joinedAt: "asc" },
  });

  return NextResponse.json({
    channelId,
    canManage,
    members: members
      .sort((a, b) => roleRank(a.role) - roleRank(b.role))
      .map(formatMember),
  });
}

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;
  const body = await request.json();
  const userId = typeof body.userId === "string" ? body.userId.trim() : "";
  const role = normalizeAssignableChannelRole(body.role);

  if (!userId) {
    return NextResponse.json({ error: "User is required" }, { status: 400 });
  }

  const channel = await getManageableChannel(channelId, session.user.id);
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

  const userExists = await prisma.user.findUnique({
    where: { id: userId },
    select: userSelect,
  });
  if (!userExists) {
    return NextResponse.json({ error: "User not found" }, { status: 404 });
  }

  const existingMember = await prisma.channelMember.findUnique({
    where: { channelId_userId: { channelId, userId } },
    select: { role: true },
  });

  if (existingMember?.role === "owner") {
    return NextResponse.json(
      { error: "Channel owner role cannot be changed" },
      { status: 400 },
    );
  }

  const member = existingMember
    ? await prisma.channelMember.update({
        where: { channelId_userId: { channelId, userId } },
        data: { role },
        select: memberSelect,
      })
    : await prisma.channelMember.create({
        data: { channelId, userId, role, ...getDefaultMembershipPreferences() },
        select: memberSelect,
      });

  return NextResponse.json(formatMember(member), {
    status: existingMember ? 200 : 201,
  });
}
