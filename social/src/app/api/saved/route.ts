import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";
import { getSavedItemsForUser } from "@/lib/savedItems";

async function readMessageId(req: NextRequest) {
  const fromQuery = req.nextUrl.searchParams.get("messageId");
  if (fromQuery) return fromQuery;

  try {
    const body = await req.json();
    return typeof body?.messageId === "string" ? body.messageId : null;
  } catch {
    return null;
  }
}

async function canAccessMessage(userId: string, messageId: string) {
  const message = await prisma.message.findFirst({
    where: {
      id: messageId,
      deletedAt: null,
      channel: {
        members: {
          some: { userId },
        },
      },
    },
    select: { id: true },
  });

  return Boolean(message);
}

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const savedItems = await getSavedItemsForUser(session.user.id);
  return NextResponse.json({ savedItems });
}

export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const messageId = await readMessageId(req);
  if (!messageId) {
    return NextResponse.json({ error: "Message id required" }, { status: 400 });
  }

  if (!(await canAccessMessage(session.user.id, messageId))) {
    return NextResponse.json({ error: "Message not found" }, { status: 404 });
  }

  await prisma.savedItem.upsert({
    where: {
      userId_messageId: {
        userId: session.user.id,
        messageId,
      },
    },
    update: {},
    create: {
      userId: session.user.id,
      messageId,
    },
  });

  return NextResponse.json({ saved: true });
}

export async function DELETE(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const messageId = await readMessageId(req);
  if (!messageId) {
    return NextResponse.json({ error: "Message id required" }, { status: 400 });
  }

  await prisma.savedItem.deleteMany({
    where: {
      userId: session.user.id,
      messageId,
    },
  });

  return NextResponse.json({ saved: false });
}
