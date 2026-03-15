import { S3Client, PutObjectCommand, GetObjectCommand } from "@aws-sdk/client-s3";

const S3_ENDPOINT = process.env.S3_ENDPOINT || "http://localhost:8333";
const S3_BUCKET = process.env.S3_BUCKET || "social";
const S3_ACCESS_KEY = process.env.S3_ACCESS_KEY_ID || "lobehub";
const S3_SECRET_KEY = process.env.S3_SECRET_ACCESS_KEY || "lobehub_s3_secret";

export const s3Client = new S3Client({
  region: "us-east-1",
  endpoint: S3_ENDPOINT,
  forcePathStyle: true,
  credentials: {
    accessKeyId: S3_ACCESS_KEY,
    secretAccessKey: S3_SECRET_KEY,
  },
});

export async function uploadToS3(
  key: string,
  body: Buffer | Uint8Array,
  contentType: string,
): Promise<string> {
  await s3Client.send(
    new PutObjectCommand({
      Bucket: S3_BUCKET,
      Key: key,
      Body: body,
      ContentType: contentType,
    }),
  );
  return `${S3_ENDPOINT}/${S3_BUCKET}/${key}`;
}

export async function getFromS3(key: string): Promise<Buffer> {
  const result = await s3Client.send(
    new GetObjectCommand({
      Bucket: S3_BUCKET,
      Key: key,
    }),
  );
  const stream = result.Body;
  if (!stream) throw new Error("No body");
  const chunks: Uint8Array[] = [];
  // @ts-expect-error - stream is async iterable
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

export { S3_BUCKET };
