import { getServerSession } from "next-auth";
import type { Session } from "next-auth";
import { headers } from "next/headers";
import { authOptions } from "./auth";
import { getString, upsertSocialUserFromIdentity } from "./socialIdentity";
import { socialLog } from "./telemetry";

type LibreChatBridgeUser = {
  _id?: string;
  id?: string;
  openidId?: string;
  idOnTheSource?: string;
  username?: string;
  name?: string;
  email?: string;
  avatar?: string;
  image?: string;
};

type AuthOptions = {
  bridgeUnavailable?: "ignore" | "throw";
};

export class LibreChatBridgeUnavailableError extends Error {
  status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "LibreChatBridgeUnavailableError";
    this.status = status;
  }
}

export function isLibreChatBridgeUnavailableError(
  error: unknown,
): error is LibreChatBridgeUnavailableError {
  return error instanceof LibreChatBridgeUnavailableError;
}

const LIBRECHAT_AUTH_BRIDGE_URL =
  process.env.LIBRECHAT_AUTH_BRIDGE_URL ||
  "http://api:3180/api/auth/social-session";

function getLibreChatCookieHeader() {
  return headers().get("cookie") || "";
}

function shouldTreatBridgeStatusAsUnavailable(status: number) {
  return status === 403 || status === 500 || status === 502 || status === 503 || status === 504;
}

function getBridgeLogUrl() {
  try {
    const url = new URL(LIBRECHAT_AUTH_BRIDGE_URL);
    return `${url.origin}${url.pathname}`;
  } catch {
    return LIBRECHAT_AUTH_BRIDGE_URL;
  }
}

function handleBridgeUnavailable(
  message: string,
  options: AuthOptions,
  status?: number,
) {
  socialLog("error", "social.auth.bridge_unavailable", {
    message,
    status,
    bridgeUnavailableMode: options.bridgeUnavailable || "ignore",
    bridgeUrl: getBridgeLogUrl(),
  });

  if (options.bridgeUnavailable === "throw") {
    throw new LibreChatBridgeUnavailableError(message, status);
  }
}

async function getLibreChatBridgeUser(
  options: AuthOptions = {},
): Promise<LibreChatBridgeUser | null> {
  const cookie = getLibreChatCookieHeader();
  if (!cookie) return null;

  const bridgeSecret = process.env.LIBRECHAT_AUTH_BRIDGE_SECRET;
  const res = await fetch(LIBRECHAT_AUTH_BRIDGE_URL, {
    method: "POST",
    headers: {
      cookie,
      ...(bridgeSecret ? { "x-librechat-social-secret": bridgeSecret } : {}),
    },
    cache: "no-store",
  }).catch((error) => {
    handleBridgeUnavailable(
      `LibreChat bridge request failed: ${(error as Error).message}`,
      options,
    );
    return null;
  });

  if (!res) return null;
  if (!res.ok) {
    if (shouldTreatBridgeStatusAsUnavailable(res.status)) {
      const body = await res.text().catch(() => "");
      handleBridgeUnavailable(
        `LibreChat bridge returned ${res.status}${body ? `: ${body}` : ""}`,
        options,
        res.status,
      );
    }
    return null;
  }

  const payload = (await res.json().catch(() => null)) as { user?: LibreChatBridgeUser } | null;
  return payload?.user || null;
}

async function getLibreChatSession(options: AuthOptions = {}): Promise<Session | null> {
  const libreUser = await getLibreChatBridgeUser(options);
  if (!libreUser) return null;

  const casdoorId =
    getString(libreUser.openidId) ||
    getString(libreUser.idOnTheSource) ||
    getString(libreUser._id) ||
    getString(libreUser.id) ||
    getString(libreUser.email);

  if (!casdoorId) return null;

  const socialUser = await upsertSocialUserFromIdentity({
    casdoorId,
    username: getString(libreUser.username) || getString(libreUser.email),
    displayName: getString(libreUser.name) || getString(libreUser.username),
    email: getString(libreUser.email),
    avatarUrl: getString(libreUser.avatar) || getString(libreUser.image),
  });

  if (!socialUser) return null;

  const name = socialUser.displayName || libreUser.name || libreUser.username || "User";
  const email = socialUser.email || libreUser.email || "";

  return {
    user: {
      id: socialUser.id,
      name,
      email,
      image: socialUser.avatarUrl || libreUser.avatar || libreUser.image || null,
      username: socialUser.username || libreUser.username,
      casdoorId,
    } as Session["user"] & {
      id: string;
      username?: string | null;
      casdoorId?: string;
    },
    expires: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
  };
}

export async function auth(options: AuthOptions = {}) {
  return (await getServerSession(authOptions)) || (await getLibreChatSession(options));
}
