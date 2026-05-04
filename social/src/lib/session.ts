import { getServerSession } from "next-auth";
import type { Session } from "next-auth";
import { headers } from "next/headers";
import { authOptions } from "./auth";
import { getString, upsertSocialUserFromIdentity } from "./socialIdentity";

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

const LIBRECHAT_AUTH_BRIDGE_URL =
  process.env.LIBRECHAT_AUTH_BRIDGE_URL ||
  "http://api:3180/api/auth/social-session";

function getLibreChatCookieHeader() {
  return headers().get("cookie") || "";
}

async function getLibreChatBridgeUser(): Promise<LibreChatBridgeUser | null> {
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
    console.error("[AUTH] LibreChat bridge request failed:", (error as Error).message);
    return null;
  });

  if (!res?.ok) return null;

  const payload = (await res.json().catch(() => null)) as { user?: LibreChatBridgeUser } | null;
  return payload?.user || null;
}

async function getLibreChatSession(): Promise<Session | null> {
  const libreUser = await getLibreChatBridgeUser();
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

export async function auth() {
  return (await getServerSession(authOptions)) || (await getLibreChatSession());
}
