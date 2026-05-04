import { NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { canCreateWorkspaceChannels } from "@/lib/channelManagement";
import { getWorkspacePolicies } from "@/lib/workspacePolicies";

export async function GET() {
  const session = await auth();
  if (!session?.user?.id) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  return NextResponse.json({
    ...getWorkspacePolicies(),
    canManage: canCreateWorkspaceChannels(session.user),
  });
}
