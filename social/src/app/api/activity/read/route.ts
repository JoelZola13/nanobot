import { NextResponse } from "next/server";
import { markActivityReadForUser } from "@/lib/activity";
import { auth } from "@/lib/session";

export async function POST() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const readAt = await markActivityReadForUser(session.user.id);
  return NextResponse.json({ ok: true, readAt: readAt.toISOString() });
}
