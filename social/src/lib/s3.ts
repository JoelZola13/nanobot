import {
  CreateBucketCommand,
  GetObjectCommand,
  HeadBucketCommand,
  HeadObjectCommand,
  PutObjectCommand,
  S3Client,
} from "@aws-sdk/client-s3";

const S3_ENDPOINT = process.env.S3_ENDPOINT || "http://localhost:8333";
const S3_BUCKET = process.env.S3_BUCKET || "social";
const S3_PUBLIC_ENDPOINT = process.env.S3_PUBLIC_ENDPOINT;
const S3_ACCESS_KEY = process.env.S3_ACCESS_KEY_ID || "lobehub";
const S3_SECRET_KEY = process.env.S3_SECRET_ACCESS_KEY || "lobehub_s3_secret";
const PUBLIC_ENDPOINT = (S3_PUBLIC_ENDPOINT || "").replace(/\/+$/, "");
const STORAGE_ENDPOINT = S3_ENDPOINT.replace(/\/+$/, "");

let bucketReady: Promise<void> | null = null;

export const s3Client = new S3Client({
  region: "us-east-1",
  endpoint: STORAGE_ENDPOINT,
  forcePathStyle: true,
  credentials: {
    accessKeyId: S3_ACCESS_KEY,
    secretAccessKey: S3_SECRET_KEY,
  },
});

function encodeKey(key: string) {
  return key.split("/").map(encodeURIComponent).join("/");
}

async function ensureBucket() {
  if (!bucketReady) {
    bucketReady = (async () => {
      try {
        await s3Client.send(new HeadBucketCommand({ Bucket: S3_BUCKET }));
      } catch {
        try {
          await s3Client.send(new CreateBucketCommand({ Bucket: S3_BUCKET }));
        } catch (error) {
          const errorName = error instanceof Error ? error.name : String(error);
          if (
            ![
              "BucketAlreadyExists",
              "BucketAlreadyOwnedByYou",
              "BucketAlreadyExistsOwnedByYou",
            ].includes(errorName)
          ) {
            throw error;
          }
        }
      }
    })();
  }

  return bucketReady;
}

export function publicUrlForS3Key(key: string) {
  const encodedKey = encodeKey(key);
  if (PUBLIC_ENDPOINT) return `${PUBLIC_ENDPOINT}/${encodedKey}`;
  return `${STORAGE_ENDPOINT}/${S3_BUCKET}/${encodedKey}`;
}

export function s3KeyFromUrl(url: string) {
  const trimmed = url.trim();
  const decoded = decodeURI(trimmed);

  if (PUBLIC_ENDPOINT && decoded.startsWith(`${PUBLIC_ENDPOINT}/`)) {
    return decoded.slice(PUBLIC_ENDPOINT.length + 1);
  }

  const apiFilesMarker = "/api/files/";
  const apiFilesIndex = decoded.indexOf(apiFilesMarker);
  if (apiFilesIndex >= 0) {
    return decoded.slice(apiFilesIndex + apiFilesMarker.length);
  }

  const bucketMarker = `/${S3_BUCKET}/`;
  const bucketIndex = decoded.indexOf(bucketMarker);
  if (bucketIndex >= 0) {
    return decoded.slice(bucketIndex + bucketMarker.length);
  }

  return decoded.replace(/^\/+/, "");
}

export async function uploadToS3(
  key: string,
  body: Buffer | Uint8Array,
  contentType: string,
): Promise<string> {
  await ensureBucket();
  await s3Client.send(
    new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: key,
      Body: body,
      ContentType: contentType,
    }),
  );
  return publicUrlForS3Key(key);
}

export async function getFromS3(key: string): Promise<Buffer> {
  const object = await getObjectFromS3(key);
  return object.body;
}

async function streamToBuffer(stream: unknown): Promise<Buffer> {
  if (!stream) throw new Error("No body");
  const chunks: Uint8Array[] = [];
  // @ts-expect-error - stream is async iterable
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

export async function getObjectFromS3(key: string, range?: string) {
  const result = await s3Client.send(
    new GetObjectCommand({
      Bucket: S3_BUCKET,
      Key: key,
      Range: range,
    }),
  );

  return {
    body: await streamToBuffer(result.Body),
    contentLength: result.ContentLength,
    contentRange: result.ContentRange,
    contentType: result.ContentType || "application/octet-stream",
    eTag: result.ETag,
  };
}

export async function headObjectFromS3(key: string) {
  const result = await s3Client.send(
    new HeadObjectCommand({
      Bucket: S3_BUCKET,
      Key: key,
    }),
  );

  return {
    contentLength: result.ContentLength,
    contentType: result.ContentType || "application/octet-stream",
    eTag: result.ETag,
  };
}

export { S3_BUCKET };
