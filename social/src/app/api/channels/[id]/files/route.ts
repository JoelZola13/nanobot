import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id: channelId } = await params;
  const membership = await prisma.channelMember.findUnique({
    where: {
      channelId_userId: {
        channelId,
        userId: session.user.id,
      },
    },
    include: {
      channel: {
        select: {
          id: true,
          type: true,
          isArchived: true,
        },
      },
    },
  });

  if (!membership || membership.channel.isArchived) {
    return NextResponse.json({ error: "Channel not found" }, { status: 404 });
  }

  const files = await prisma.attachment.findMany({
    where: {
      message: {
        channelId,
        deletedAt: null,
      },
    },
    select: {
      id: true,
      fileName: true,
      fileSize: true,
      mimeType: true,
      url: true,
      width: true,
      height: true,
      message: {
        select: {
          id: true,
          channelId: true,
          content: true,
          createdAt: true,
          author: {
            select: {
              id: true,
              username: true,
              displayName: true,
              avatarUrl: true,
              isAgent: true,
            },
          },
        },
      },
    },
    orderBy: {
      message: {
        createdAt: "desc",
      },
    },
    take: 100,
  });

  const basePath =
    membership.channel.type === "DM" || membership.channel.type === "GROUP_DM"
      ? "dm"
      : "channels";

  return NextResponse.json({
    files: files.map((file) => ({
      id: file.id,
      fileName: file.fileName,
      fileSize: file.fileSize,
      mimeType: file.mimeType,
      url: file.url,
      width: file.width,
      height: file.height,
      messageId: file.message.id,
      channelId: file.message.channelId,
      createdAt: file.message.createdAt.toISOString(),
      messageContent: file.message.content,
      href: `/${basePath}/${file.message.channelId}?message=${file.message.id}`,
      author: file.message.author,
    })),
  });
}
