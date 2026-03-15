import { NextRequest, NextResponse } from "next/server";
import { auth } from "@/lib/session";
import { uploadToS3 } from "@/lib/s3";
import { randomUUID } from "crypto";

// POST /api/upload — upload file to S3
export async function POST(req: NextRequest) {
  const session = await auth();
  if (!session?.user?.id)
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  const formData = await req.formData();
  const file = formData.get("file") as File | null;
  if (!file)
    return NextResponse.json({ error: "No file" }, { status: 400 });

  const ext = file.name.split(".").pop() || "bin";
  const s3Key = `uploads/${session.user.id}/${randomUUID()}.${ext}`;
  const bytes = await file.arrayBuffer();
  const url = await uploadToS3(s3Key, Buffer.from(bytes), file.type);

  return NextResponse.json({
    s3Key,
    url,
    fileName: file.name,
    fileSize: file.size,
    mimeType: file.type,
  });
}
