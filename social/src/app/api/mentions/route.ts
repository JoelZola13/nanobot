import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { getMentionMessagesForUser } from "@/lib/mentions";

// GET /api/mentions — messages in accessible channels that mention the user
export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const result = await getMentionMessagesForUser(session.user.id);
  return NextResponse.json(result);
}
