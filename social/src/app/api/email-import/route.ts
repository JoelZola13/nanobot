import { NextRequest, NextResponse } from "next/server";
import { ZodError } from "zod";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { formatMessageForClient } from "@/lib/messageFormat";
import { getIO } from "@/lib/socketServer";
import {
  EmailImportRequest,
  EmailImportRequestSchema,
  formatEmailImportForMessage,
} from "@/lib/emailImport";
import {
  canJoinPublicChannel,
  getDefaultMembershipPreferences,
} from "@/lib/workspacePolicies";

type DestinationResult =
  | { channelId: string }
  | { error: string; status: number };

async function resolveChannelDestination(
  channelId: string,
  user: { id: string; role?: string | null },
): Promise<DestinationResult> {
  const membership = await prisma.channelMember.findUnique({
    where: { channelId_userId: { channelId, userId: user.id } },
    include: { channel: { select: { isArchived: true } } },
  });

  if (membership?.channel.isArchived) {
    return { error: "Channel is archived", status: 403 };
  }
  if (membership) return { channelId };

  const channel = await prisma.channel.findUnique({
    where: { id: channelId },
    select: { id: true, type: true, isArchived: true },
  });
  if (!channel || channel.isArchived) {
    return { error: "Channel not found", status: 404 };
  }
  if (channel.type !== "PUBLIC") {
    return { error: "Not a member", status: 403 };
  }
  if (!canJoinPublicChannel(user)) {
    return { error: "Workspace admin access required to join public channels", status: 403 };
  }

  await prisma.channelMember.upsert({
    where: { channelId_userId: { channelId, userId: user.id } },
    update: {},
    create: {
      channelId,
      userId: user.id,
      role: "member",
      ...getDefaultMembershipPreferences(),
    },
  });

  return { channelId };
}

async function resolveDmDestination(
  otherUserId: string,
  currentUserId: string,
): Promise<DestinationResult> {
  if (otherUserId === currentUserId) {
    return { error: "Cannot import email into a DM with yourself", status: 400 };
  }

  const [currentUser, otherUser] = await Promise.all([
    prisma.user.findUnique({
      where: { id: currentUserId },
      select: { id: true },
    }),
    prisma.user.findUnique({
      where: { id: otherUserId },
      select: { id: true },
    }),
  ]);

  if (!currentUser) {
    return { error: "Current user not found in Messages", status: 409 };
  }
  if (!otherUser) return { error: "User not found", status: 404 };

  const existingDM = await prisma.channel.findFirst({
    where: {
      type: "DM",
      AND: [
        { members: { some: { userId: currentUserId } } },
        { members: { some: { userId: otherUserId } } },
      ],
    },
    select: { id: true },
  });

  if (existingDM) return { channelId: existingDM.id };

  const membershipPreferences = getDefaultMembershipPreferences();
  const channel = await prisma.channel.create({
    data: {
      name: null,
      slug: `dm-${currentUserId}-${otherUserId}`,
      type: "DM",
      members: {
        create: [
          { userId: currentUserId, role: "member", ...membershipPreferences },
          { userId: otherUserId, role: "member", ...membershipPreferences },
        ],
      },
    },
    select: { id: true },
  });

  return { channelId: channel.id };
}

async function resolveDestination(
  destination: EmailImportRequest["destination"],
  user: { id: string; role?: string | null },
): Promise<DestinationResult> {
  if (destination.type === "channel") {
    return resolveChannelDestination(destination.channelId, user);
  }

  return resolveDmDestination(destination.userId, user.id);
}

// POST /api/email-import - import the currently selected email into Messages.
export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let parsed: EmailImportRequest;
  try {
    parsed = EmailImportRequestSchema.parse(await req.json());
  } catch (error) {
    const details = error instanceof ZodError ? error.issues : undefined;
    return NextResponse.json(
      { error: "Invalid email import payload", details },
      { status: 400 },
    );
  }

  const destination = await resolveDestination(parsed.destination, session.user);
  if ("error" in destination) {
    return NextResponse.json(
      { error: destination.error },
      { status: destination.status },
    );
  }

  const { content, metadata } = formatEmailImportForMessage(parsed.email);
  const message = await prisma.message.create({
    data: {
      channelId: destination.channelId,
      authorId: session.user.id,
      content,
      metadata,
    },
    include: {
      author: {
        select: {
          id: true,
          username: true,
          displayName: true,
          avatarUrl: true,
          isAgent: true,
        },
      },
      attachments: true,
    },
  });

  await prisma.channel.update({
    where: { id: destination.channelId },
    data: { updatedAt: new Date() },
  });

  const formatted = formatMessageForClient(message, session.user.id);
  getIO()?.to(`channel:${destination.channelId}`).emit("message:new", formatted);

  return NextResponse.json(
    {
      channelId: destination.channelId,
      message: formatted,
    },
    { status: 201 },
  );
}
