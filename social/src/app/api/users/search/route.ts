import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { prisma } from "@/lib/prisma";

// GET /api/users/search?q=query — search users
export async function GET(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const q = req.nextUrl.searchParams.get("q")?.trim();
  if (!q || q.length < 2)
    return NextResponse.json([]);

  const users = await prisma.user.findMany({
    where: {
      OR: [
        { displayName: { contains: q, mode: "insensitive" } },
        { username: { contains: q, mode: "insensitive" } },
        { email: { contains: q, mode: "insensitive" } },
      ],
      id: { not: session.user.id },
    },
    select: {
      id: true,
      username: true,
      displayName: true,
      avatarUrl: true,
      isAgent: true,
      status: true,
    },
    take: 10,
    orderBy: { displayName: "asc" },
  });

  return NextResponse.json(users);
}
