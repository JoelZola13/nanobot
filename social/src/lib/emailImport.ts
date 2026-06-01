import { z } from "zod";

const MAX_SUBJECT_CHARS = 500;
const MAX_ADDRESS_NAME_CHARS = 200;
const MAX_ADDRESS_EMAIL_CHARS = 320;
const MAX_URL_CHARS = 2_000;
const MAX_BODY_INPUT_CHARS = 200_000;
const MAX_HTML_INPUT_CHARS = 400_000;
const MAX_HTML_METADATA_CHARS = 180_000;
const MAX_BODY_MESSAGE_CHARS = 28_000;
const MAX_MESSAGE_CHARS = 32_000;
const MAX_BODY_PREVIEW_CHARS = 1_500;

const textField = (max: number) =>
  z
    .preprocess((value) => {
      if (value === null || value === undefined) return undefined;
      if (typeof value === "string") return value.trim();
      return String(value).trim();
    }, z.string().max(max).optional())
    .transform((value) => (value ? value : undefined));

const emailAddressSchema = z
  .object({
    name: textField(MAX_ADDRESS_NAME_CHARS),
    email: textField(MAX_ADDRESS_EMAIL_CHARS),
  })
  .transform((address) => ({
    name: address.name,
    email: address.email,
  }))
  .refine((address) => Boolean(address.name || address.email), {
    message: "Address must include a name or email",
  });

const emailBodySchema = z
  .preprocess((value) => {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value;
    return String(value);
  }, z.string().max(MAX_BODY_INPUT_CHARS))
  .transform((value) => cleanBodyText(value));

const emailHtmlSchema = z
  .preprocess((value) => {
    if (value === null || value === undefined) return undefined;
    if (typeof value === "string") return value;
    return String(value);
  }, z.string().max(MAX_HTML_INPUT_CHARS).optional())
  .transform((value) => (value?.trim() ? value.trim() : undefined));

const emailAttachmentSchema = z.object({
  name: textField(500),
  url: textField(MAX_URL_CHARS),
  mimeType: textField(200),
  sizeLabel: textField(100),
});

export const EmailImportRequestSchema = z.object({
  destination: z.discriminatedUnion("type", [
    z.object({
      type: z.literal("channel"),
      channelId: z.string().trim().min(1),
    }),
    z.object({
      type: z.literal("dm"),
      userId: z.string().trim().min(1),
    }),
  ]),
  email: z.object({
    provider: textField(80),
    subject: textField(MAX_SUBJECT_CHARS),
    from: emailAddressSchema.optional(),
    to: z.array(emailAddressSchema).max(50).optional().default([]),
    cc: z.array(emailAddressSchema).max(50).optional().default([]),
    sentAt: textField(200),
    messageId: textField(500),
    sourceUrl: textField(MAX_URL_CHARS),
    bodyText: emailBodySchema,
    bodyHtml: emailHtmlSchema,
    attachments: z.array(emailAttachmentSchema).max(50).optional().default([]),
  }),
});

export type EmailImportRequest = z.infer<typeof EmailImportRequestSchema>;
export type ImportedEmail = EmailImportRequest["email"];

export type EmailImportMetadata = {
  type: "email_import";
  email: {
    provider?: string;
    subject: string;
    from?: EmailAddress;
    to: EmailAddress[];
    cc: EmailAddress[];
    sentAt?: string;
    messageId?: string;
    sourceUrl?: string;
	    capturedAt: string;
	    bodyPreview?: string;
	    bodyText?: string;
	    bodyHtml?: string;
    bodyTruncated: boolean;
    htmlTruncated: boolean;
    attachments: EmailAttachment[];
  };
};

type EmailAddress = {
  name?: string;
  email?: string;
};

type TruncatedText = {
  text: string;
  truncated: boolean;
};

type EmailAttachment = {
  name?: string;
  url?: string;
  mimeType?: string;
  sizeLabel?: string;
};

function cleanBodyText(value: string): string {
  return value
    .replace(/\r\n?/g, "\n")
    .replace(/\u00a0/g, " ")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{4,}/g, "\n\n\n")
    .trim();
}

function truncateText(value: string, maxChars: number): TruncatedText {
  if (value.length <= maxChars) return { text: value, truncated: false };
  const suffix = "\n\n[Truncated]";
  return {
    text: `${value.slice(0, Math.max(0, maxChars - suffix.length)).trimEnd()}${suffix}`,
    truncated: true,
  };
}

function subjectOrFallback(subject?: string): string {
  return subject || "(no subject)";
}

function formatAddress(address?: EmailAddress): string {
  if (!address) return "Unknown sender";
  if (address.name && address.email) return `${address.name} <${address.email}>`;
  return address.name || address.email || "Unknown sender";
}

function formatAddressList(addresses: EmailAddress[]): string | undefined {
  if (!addresses.length) return undefined;
  return addresses.map(formatAddress).join(", ");
}

function compactMetadataAddress(address?: EmailAddress): EmailAddress | undefined {
  if (!address) return undefined;
  return {
    ...(address.name ? { name: address.name } : {}),
    ...(address.email ? { email: address.email } : {}),
  };
}

function compactMetadataAddresses(addresses: EmailAddress[]): EmailAddress[] {
  return addresses
    .map(compactMetadataAddress)
    .filter((address): address is EmailAddress => Boolean(address));
}

function compactAttachments(attachments: EmailAttachment[]): EmailAttachment[] {
  return attachments.map((attachment) => ({
    ...(attachment.name ? { name: attachment.name } : {}),
    ...(attachment.url ? { url: attachment.url } : {}),
    ...(attachment.mimeType ? { mimeType: attachment.mimeType } : {}),
    ...(attachment.sizeLabel ? { sizeLabel: attachment.sizeLabel } : {}),
  }));
}

export function formatEmailImportForMessage(email: ImportedEmail): {
  content: string;
  metadata: EmailImportMetadata;
} {
  const subject = subjectOrFallback(email.subject);
  const body = truncateText(email.bodyText, MAX_BODY_MESSAGE_CHARS);
  const bodyHtml = truncateText(email.bodyHtml || "", MAX_HTML_METADATA_CHARS);
  const toLine = formatAddressList(email.to);
  const ccLine = formatAddressList(email.cc);
  const attachmentNames = email.attachments
    .map((attachment) => attachment.name || attachment.url)
    .filter(Boolean);

  const headerLines = [
    `Imported email: ${subject}`,
    "",
    `From: ${formatAddress(email.from)}`,
    toLine ? `To: ${toLine}` : undefined,
    ccLine ? `Cc: ${ccLine}` : undefined,
    email.sentAt ? `Date: ${email.sentAt}` : undefined,
    email.sourceUrl ? `Source: ${email.sourceUrl}` : undefined,
    attachmentNames.length ? `Attachments: ${attachmentNames.join(", ")}` : undefined,
  ].filter((line): line is string => Boolean(line));

  const message = truncateText(
    `${headerLines.join("\n")}\n\n---\n\n${body.text || "(No body text captured.)"}`,
    MAX_MESSAGE_CHARS,
  );

  const bodyPreview = truncateText(email.bodyText, MAX_BODY_PREVIEW_CHARS).text;
  const capturedAt = new Date().toISOString();

  return {
    content: message.text,
    metadata: {
      type: "email_import",
      email: {
        ...(email.provider ? { provider: email.provider } : {}),
        subject,
        ...(compactMetadataAddress(email.from) ? { from: compactMetadataAddress(email.from) } : {}),
        to: compactMetadataAddresses(email.to),
        cc: compactMetadataAddresses(email.cc),
        ...(email.sentAt ? { sentAt: email.sentAt } : {}),
        ...(email.messageId ? { messageId: email.messageId } : {}),
        ...(email.sourceUrl ? { sourceUrl: email.sourceUrl } : {}),
	        capturedAt,
	        ...(bodyPreview ? { bodyPreview } : {}),
	        ...(body.text ? { bodyText: body.text } : {}),
	        ...(bodyHtml.text ? { bodyHtml: bodyHtml.text } : {}),
        bodyTruncated: body.truncated || message.truncated,
        htmlTruncated: bodyHtml.truncated,
        attachments: compactAttachments(email.attachments),
      },
    },
  };
}
