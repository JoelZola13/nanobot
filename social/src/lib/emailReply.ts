import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import nodemailer from "nodemailer";
import { z } from "zod";

const MAX_REPLY_CHARS = 20_000;
const DEFAULT_SMTP_PORT = 587;

type JsonRecord = Record<string, unknown>;

export type EmailReplyAddress = {
  name?: string;
  email: string;
};

export type ImportedEmailReplyTarget = {
  provider?: string;
  subject: string;
  from: EmailReplyAddress;
  sentAt?: string;
  messageId?: string;
  sourceUrl?: string;
  bodyText?: string;
  bodyPreview?: string;
};

type EmailReplySmtpConfig = {
  enabled: boolean;
  consentGranted: boolean;
  host: string;
  port: number;
  username: string;
  password: string;
  fromAddress: string;
  useTls: boolean;
  useSsl: boolean;
};

export class EmailReplyConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EmailReplyConfigError";
  }
}

const replyTextField = z
  .preprocess((value) => {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value.trim();
    return String(value).trim();
  }, z.string().min(1, "Reply content is required").max(MAX_REPLY_CHARS));

export const EmailReplyRequestSchema = z.object({
  content: replyTextField,
});

export type EmailReplyRequest = z.infer<typeof EmailReplyRequestSchema>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(record: JsonRecord | undefined, ...keys: string[]): string {
  if (!record) return "";
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "string" && value.trim()) return value.trim();
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
  }
  return "";
}

function readNumber(
  record: JsonRecord | undefined,
  fallback: number,
  ...keys: string[]
): number {
  const value = readString(record, ...keys);
  if (!value) return fallback;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function readBoolean(
  record: JsonRecord | undefined,
  fallback: boolean,
  ...keys: string[]
): boolean {
  if (!record) return fallback;
  for (const key of keys) {
    const value = record[key];
    if (typeof value === "boolean") return value;
    if (typeof value !== "string" && typeof value !== "number") continue;
    const normalized = String(value).trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) return true;
    if (["0", "false", "no", "off"].includes(normalized)) return false;
  }
  return fallback;
}

function envRecord(): JsonRecord {
  return process.env as JsonRecord;
}

function loadEnvSmtpConfig(): EmailReplySmtpConfig | null {
  const env = envRecord();
  const host = readString(
    env,
    "EMAIL_REPLY_SMTP_HOST",
    "NANOBOT_CHANNELS__EMAIL__SMTP_HOST",
  );
  const username = readString(
    env,
    "EMAIL_REPLY_SMTP_USERNAME",
    "EMAIL_REPLY_SMTP_USER",
    "NANOBOT_CHANNELS__EMAIL__SMTP_USERNAME",
  );
  const password = readString(
    env,
    "EMAIL_REPLY_SMTP_PASSWORD",
    "NANOBOT_CHANNELS__EMAIL__SMTP_PASSWORD",
  );

  if (!host && !username && !password) return null;

  const fromAddress =
    readString(
      env,
      "EMAIL_REPLY_FROM",
      "EMAIL_REPLY_FROM_ADDRESS",
      "NANOBOT_CHANNELS__EMAIL__FROM_ADDRESS",
    ) || username;

  return {
    enabled: readBoolean(env, true, "EMAIL_REPLY_ENABLED", "NANOBOT_CHANNELS__EMAIL__ENABLED"),
    consentGranted: readBoolean(
      env,
      true,
      "EMAIL_REPLY_CONSENT_GRANTED",
      "NANOBOT_CHANNELS__EMAIL__CONSENT_GRANTED",
    ),
    host,
    port: readNumber(
      env,
      DEFAULT_SMTP_PORT,
      "EMAIL_REPLY_SMTP_PORT",
      "NANOBOT_CHANNELS__EMAIL__SMTP_PORT",
    ),
    username,
    password,
    fromAddress,
    useTls: readBoolean(
      env,
      true,
      "EMAIL_REPLY_SMTP_USE_TLS",
      "NANOBOT_CHANNELS__EMAIL__SMTP_USE_TLS",
    ),
    useSsl: readBoolean(
      env,
      false,
      "EMAIL_REPLY_SMTP_USE_SSL",
      "NANOBOT_CHANNELS__EMAIL__SMTP_USE_SSL",
    ),
  };
}

function loadNanobotConfigSmtpConfig(): EmailReplySmtpConfig | null {
  const home = process.env.HOME;
  if (!home) return null;

  const configPath = join(home, ".nanobot", "config.json");
  if (!existsSync(configPath)) return null;

  const parsed = JSON.parse(readFileSync(configPath, "utf8")) as unknown;
  if (!isRecord(parsed)) return null;

  const channels = isRecord(parsed.channels) ? parsed.channels : undefined;
  const emailConfig = isRecord(channels?.email)
    ? channels.email
    : isRecord(parsed.email)
      ? parsed.email
      : undefined;
  if (!emailConfig) return null;

  const username = readString(emailConfig, "smtpUsername", "smtp_username");
  const fromAddress =
    readString(emailConfig, "fromAddress", "from_address") || username;

  return {
    enabled: readBoolean(emailConfig, true, "enabled"),
    consentGranted: readBoolean(
      emailConfig,
      false,
      "consentGranted",
      "consent_granted",
    ),
    host: readString(emailConfig, "smtpHost", "smtp_host"),
    port: readNumber(emailConfig, DEFAULT_SMTP_PORT, "smtpPort", "smtp_port"),
    username,
    password: readString(emailConfig, "smtpPassword", "smtp_password"),
    fromAddress,
    useTls: readBoolean(emailConfig, true, "smtpUseTls", "smtp_use_tls"),
    useSsl: readBoolean(emailConfig, false, "smtpUseSsl", "smtp_use_ssl"),
  };
}

function loadSmtpConfig(): EmailReplySmtpConfig {
  const config = loadEnvSmtpConfig() || loadNanobotConfigSmtpConfig();
  if (!config) {
    throw new EmailReplyConfigError("SMTP email replies are not configured.");
  }
  if (!config.enabled) {
    throw new EmailReplyConfigError("Email replies are disabled.");
  }
  if (!config.consentGranted) {
    throw new EmailReplyConfigError("Email reply consent is not enabled.");
  }
  if (!config.host || !config.username || !config.password || !config.fromAddress) {
    throw new EmailReplyConfigError("SMTP email replies are missing required credentials.");
  }
  return config;
}

function isLikelyEmailAddress(value: string): boolean {
  return /^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(value);
}

function cleanHeaderValue(value: string): string {
  return value.replace(/[\r\n]+/g, " ").trim();
}

function formatAddress(address: EmailReplyAddress): string {
  const email = cleanHeaderValue(address.email);
  const name = address.name ? cleanHeaderValue(address.name) : "";
  if (!name) return email;
  return `"${name.replace(/"/g, '\\"')}" <${email}>`;
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function textToHtmlLines(value: string): string {
  return escapeHtml(value)
    .replace(/\r\n?/g, "\n")
    .replace(/\n/g, "<br>");
}

function gmailAttribution(target: ImportedEmailReplyTarget): string {
  const sender = formatAddress(target.from);
  return target.sentAt
    ? `On ${target.sentAt}, ${sender} wrote:`
    : `${sender} wrote:`;
}

function buildGmailStyleReplyText(
  target: ImportedEmailReplyTarget,
  content: string,
): string {
  const quoted = target.bodyText || target.bodyPreview || "";
  if (!quoted) return content;

  const quotedLines = quoted
    .replace(/\r\n?/g, "\n")
    .split("\n")
    .map((line) => `> ${line}`)
    .join("\n");

  return `${content}\n\n${gmailAttribution(target)}\n${quotedLines}`;
}

function buildGmailStyleReplyHtml(
  target: ImportedEmailReplyTarget,
  content: string,
): string {
  const replyHtml = `<div dir="ltr">${textToHtmlLines(content)}</div>`;
  const quoted = target.bodyText || target.bodyPreview || "";
  if (!quoted) return replyHtml;

  return [
    replyHtml,
    "<br>",
    '<div class="gmail_quote gmail_quote_container">',
    `<div dir="ltr" class="gmail_attr">${escapeHtml(gmailAttribution(target))}<br></div>`,
    '<blockquote class="gmail_quote" style="margin:0px 0px 0px 0.8ex;border-left:1px solid rgb(204,204,204);padding-left:1ex">',
    textToHtmlLines(quoted),
    "</blockquote>",
    "</div>",
  ].join("");
}

export function buildReplySubject(subject: string): string {
  const cleaned = cleanHeaderValue(subject || "(no subject)") || "(no subject)";
  return /^re:/i.test(cleaned) ? cleaned : `Re: ${cleaned}`;
}

export function normalizeMessageIdHeader(messageId?: string): string | undefined {
  const value = cleanHeaderValue(messageId || "");
  if (!value || /\s/.test(value)) return undefined;
  if (/^<[^<>@\s]+@[^<>\s]+>$/.test(value)) return value;
  if (/^[^<>@\s]+@[^<>\s]+$/.test(value)) return `<${value}>`;
  return undefined;
}

export function getImportedEmailReplyTarget(
  metadata: unknown,
): ImportedEmailReplyTarget | null {
  if (!isRecord(metadata) || metadata.type !== "email_import") return null;
  const email = isRecord(metadata.email) ? metadata.email : undefined;
  const from = isRecord(email?.from) ? email.from : undefined;
  const fromEmail = readString(from, "email");

  if (!fromEmail || !isLikelyEmailAddress(fromEmail)) return null;

	  return {
	    provider: readString(email, "provider") || undefined,
	    subject: readString(email, "subject") || "(no subject)",
	    from: {
	      email: fromEmail,
	      ...(readString(from, "name") ? { name: readString(from, "name") } : {}),
	    },
	    sentAt: readString(email, "sentAt", "sent_at") || undefined,
	    messageId: readString(email, "messageId", "message_id") || undefined,
	    sourceUrl: readString(email, "sourceUrl", "source_url") || undefined,
	    bodyText: readString(email, "bodyText", "body_text") || undefined,
	    bodyPreview: readString(email, "bodyPreview", "body_preview") || undefined,
	  };
}

export async function sendImportedEmailReply(
  target: ImportedEmailReplyTarget,
  content: string,
): Promise<{
  to: EmailReplyAddress;
  subject: string;
  sentAt: string;
  inReplyTo?: string;
  messageId?: string;
}> {
  const config = loadSmtpConfig();
  const subject = buildReplySubject(target.subject);
  const inReplyTo = normalizeMessageIdHeader(target.messageId);
  const text = buildGmailStyleReplyText(target, content);
  const html = buildGmailStyleReplyHtml(target, content);
  const transport = nodemailer.createTransport({
    host: config.host,
    port: config.port,
    secure: config.useSsl,
    requireTLS: config.useTls && !config.useSsl,
    auth: {
      user: config.username,
      pass: config.password,
    },
  });

  const result = await transport.sendMail({
	    from: config.fromAddress,
	    to: formatAddress(target.from),
	    subject,
	    text,
	    html,
    ...(inReplyTo
      ? {
          inReplyTo,
          references: inReplyTo,
        }
      : {}),
  });

  return {
    to: target.from,
    subject,
    sentAt: new Date().toISOString(),
    ...(inReplyTo ? { inReplyTo } : {}),
    ...(typeof result.messageId === "string" ? { messageId: result.messageId } : {}),
  };
}
