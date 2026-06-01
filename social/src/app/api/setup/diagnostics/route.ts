import { NextResponse } from "next/server";
import { runSetupDiagnostics } from "@/lib/setupDiagnostics";

export const dynamic = "force-dynamic";

export async function GET() {
  const diagnostics = await runSetupDiagnostics();
  const status = diagnostics.status === "error" ? 503 : 200;

  return NextResponse.json(diagnostics, {
    status,
    headers: {
      "Cache-Control": "no-store",
    },
  });
}
