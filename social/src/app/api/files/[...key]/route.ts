import { NextRequest, NextResponse } from "next/server";
import { getObjectFromS3, headObjectFromS3 } from "@/lib/s3";

export const dynamic = "force-dynamic";

function normalizeKey(parts: string[]) {
  return parts.map((part) => decodeURIComponent(part)).join("/");
}

function parseRange(rangeHeader: string | null, size: number) {
  if (!rangeHeader) return null;
  const match = rangeHeader.match(/^bytes=(\d*)-(\d*)$/);
  if (!match) return null;

  const [, startValue, endValue] = match;
  let start = startValue ? Number(startValue) : 0;
  let end = endValue ? Number(endValue) : size - 1;

  if (!startValue && endValue) {
    const suffixLength = Number(endValue);
    start = Math.max(size - suffixLength, 0);
    end = size - 1;
  }

  if (
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    start < 0 ||
    end < start ||
    start >= size
  ) {
    return null;
  }

  return {
    start,
    end: Math.min(end, size - 1),
  };
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ key?: string[] }> },
) {
  const { key: keyParts = [] } = await params;
  const key = normalizeKey(keyParts);
  if (!key || !key.startsWith("uploads/")) {
    return NextResponse.json({ error: "Invalid file key" }, { status: 400 });
  }

  try {
    const head = await headObjectFromS3(key);
    const size = head.contentLength || 0;
    const range = parseRange(request.headers.get("range"), size);

    if (range) {
      const object = await getObjectFromS3(
        key,
        `bytes=${range.start}-${range.end}`,
      );
      return new NextResponse(new Uint8Array(object.body), {
        status: 206,
        headers: {
          "Accept-Ranges": "bytes",
          "Cache-Control": "public, max-age=31536000, immutable",
          "Content-Length": String(object.body.byteLength),
          "Content-Range": `bytes ${range.start}-${range.end}/${size}`,
          "Content-Type": object.contentType || head.contentType,
        },
      });
    }

    const object = await getObjectFromS3(key);
    return new NextResponse(new Uint8Array(object.body), {
      headers: {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=31536000, immutable",
        "Content-Length": String(object.body.byteLength),
        "Content-Type": object.contentType || head.contentType,
      },
    });
  } catch (error) {
    console.error("[FILES] Failed to stream attachment:", error);
    return NextResponse.json({ error: "File could not load" }, { status: 404 });
  }
}
