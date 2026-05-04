import type { MessageData } from "@/types";

const PROMPT_DISMISSED_KEY = "street-voices:browser-notifications:dismissed";
const MAX_BODY_LENGTH = 140;

type BrowserNotificationStorage = Pick<Storage, "getItem" | "setItem">;

export type BrowserNotificationDestination = {
  label: string;
  href: string;
  type: "channel" | "dm" | "group_dm";
};

function getBrowserNotificationApi(): typeof Notification | null {
  if (typeof window === "undefined" || !("Notification" in window)) return null;
  return window.Notification;
}

function getBrowserNotificationStorage(): BrowserNotificationStorage | null {
  if (typeof window === "undefined") return null;

  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function getBrowserNotificationPermission() {
  const notificationApi = getBrowserNotificationApi();
  return notificationApi?.permission || "unsupported";
}

export function shouldShowBrowserNotificationPrompt(
  storage: BrowserNotificationStorage | null = getBrowserNotificationStorage(),
) {
  return getBrowserNotificationPermission() === "default" &&
    storage?.getItem(PROMPT_DISMISSED_KEY) !== "true";
}

export function dismissBrowserNotificationPrompt(
  storage: BrowserNotificationStorage | null = getBrowserNotificationStorage(),
) {
  storage?.setItem(PROMPT_DISMISSED_KEY, "true");
}

export async function requestBrowserNotificationPermission() {
  const notificationApi = getBrowserNotificationApi();
  if (!notificationApi || notificationApi.permission === "denied") {
    return getBrowserNotificationPermission();
  }

  if (notificationApi.permission === "granted") return "granted";
  return notificationApi.requestPermission();
}

function stripMessagePreview(content: string) {
  return content
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[*_`>#~-]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function notificationTitle(message: MessageData, destination?: BrowserNotificationDestination) {
  if (!destination || destination.type === "dm") return message.author.displayName;
  if (destination.type === "group_dm") return `${message.author.displayName} in ${destination.label}`;
  return `${message.author.displayName} in #${destination.label}`;
}

export function showBrowserMessageNotification(
  message: MessageData,
  destination?: BrowserNotificationDestination,
) {
  const notificationApi = getBrowserNotificationApi();
  if (!notificationApi || notificationApi.permission !== "granted") return null;

  const preview = stripMessagePreview(message.content) || "Shared an attachment";
  const body = preview.length > MAX_BODY_LENGTH
    ? `${preview.slice(0, MAX_BODY_LENGTH - 1)}...`
    : preview;

  const notification = new notificationApi(notificationTitle(message, destination), {
    body,
    icon: message.author.avatarUrl || undefined,
    tag: `street-voices:${message.channelId}`,
  });

  notification.onclick = () => {
    window.focus();
    if (destination?.href) window.location.href = destination.href;
    notification.close();
  };

  return notification;
}
