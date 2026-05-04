import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

export async function POST(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;

  const channel = await prisma.channel.findUnique({
    where: { id: channelId },
    select: { id: true, type: true, isArchived: true },
  });
  if (!channel || channel.isArchived) {
    return NextResponse.json({ error: "Channel not found" }, { status: 404 });
  }

  const existingMembership = await prisma.channelMember.findUnique({
    where: {
      channelId_userId: {
        channelId,
        userId: session.user.id,
      },
    },
    select: { role: true },
  });

  if (channel.type === "PRIVATE" && !existingMembership) {
    return NextResponse.json(
      { error: "Private channels require an invitation" },
      { status: 403 },
    );
  }

  if (channel.type !== "PUBLIC" && channel.type !== "PRIVATE") {
    return NextResponse.json(
      { error: "Channel cannot be joined" },
      { status: 400 },
    );
  }

  const membership = await prisma.channelMember.upsert({
    where: {
      channelId_userId: {
        channelId,
        userId: session.user.id,
      },
    },
    update: {},
    create: {
      channelId,
      userId: session.user.id,
      role: "member",
    },
    select: { role: true },
  });

  return NextResponse.json({
    channelId,
    isMember: true,
    role: membership.role,
  });
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;

  const channel = await prisma.channel.findUnique({
    where: { id: channelId },
    select: { id: true, type: true, isArchived: true, isDefault: true },
  });
  if (!channel || channel.isArchived) {
    return NextResponse.json({ error: "Channel not found" }, { status: 404 });
  }

  if (channel.type !== "PUBLIC" && channel.type !== "PRIVATE") {
    return NextResponse.json(
      { error: "Channel membership cannot be changed" },
      { status: 400 },
    );
  }

  if (channel.isDefault) {
    return NextResponse.json(
      { error: "Default channels cannot be left" },
      { status: 400 },
    );
  }

  await prisma.channelMember.deleteMany({
    where: {
      channelId,
      userId: session.user.id,
    },
  });

  return NextResponse.json({ channelId, isMember: false });
}
