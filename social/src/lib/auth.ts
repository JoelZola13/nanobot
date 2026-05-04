import NextAuth from "next-auth";
import type { NextAuthOptions } from "next-auth";
import {
  findSocialUserByCasdoorId,
  getString,
  upsertSocialUserFromIdentity,
} from "./socialIdentity";

// Strip /api/auth from NEXTAUTH_URL to get the site base URL for redirect_uri
const siteUrl = (process.env.NEXTAUTH_URL || "http://localhost:3180/social/api/auth").replace(/\/api\/auth$/, "");

export const authOptions: NextAuthOptions = {
  providers: [
    {
      id: "casdoor",
      name: "Street Voices",
      type: "oauth",
      wellKnown: `${process.env.AUTH_CASDOOR_ISSUER}/.well-known/openid-configuration`,
      clientId: process.env.AUTH_CASDOOR_ID,
      clientSecret: process.env.AUTH_CASDOOR_SECRET,
      authorization: {
        params: {
          scope: "openid profile email",
          redirect_uri: `${siteUrl}/api/auth/callback/casdoor`,
        },
      },
      token: {
        params: {
          redirect_uri: `${siteUrl}/api/auth/callback/casdoor`,
        },
      },
      client: {
        redirect_uris: [`${siteUrl}/api/auth/callback/casdoor`],
      },
      idToken: true,
      profile(profile) {
        return {
          id: profile.sub,
          name: profile.name || (profile as Record<string, unknown>).preferred_username,
          email: profile.email,
          image: profile.picture || null,
        };
      },
    },
  ],
  callbacks: {
    async signIn({ user, profile }) {
      if (!profile?.sub) return false;
      try {
        const casdoorId = profile.sub as string;
        const username =
          ((profile as Record<string, unknown>).preferred_username as string) ||
          (profile.name as string) ||
          casdoorId;
        const displayName = (profile.name as string) || user.name || "User";
        const email = user.email || "";
        const avatarUrl = (user.image as string) || null;

        await upsertSocialUserFromIdentity({
          casdoorId,
          username,
          displayName,
          email,
          avatarUrl,
        });
      } catch (e) {
        console.error("[AUTH] DB upsert failed:", (e as Error).message);
      }
      return true;
    },
    async jwt({ token, profile }) {
      const tokenRecord = token as Record<string, unknown>;
      const casdoorId =
        getString(profile?.sub) ||
        getString(tokenRecord.casdoorId) ||
        getString(token.sub);

      if (casdoorId) tokenRecord.casdoorId = casdoorId;

      if (profile) {
        tokenRecord.username =
          ((profile as Record<string, unknown>).preferred_username as string) ||
          (profile.name as string) ||
          "";
      }

      if (!getString(tokenRecord.userId)) {
        const socialUser = await findSocialUserByCasdoorId(casdoorId);
        if (socialUser) {
          tokenRecord.userId = socialUser.id;
          if (!getString(tokenRecord.username) && socialUser.username) {
            tokenRecord.username = socialUser.username;
          }
        }
      }

      return token;
    },
    async session({ session, token }) {
      const tokenRecord = token as Record<string, unknown>;
      const casdoorId =
        getString(tokenRecord.casdoorId) || getString(token.sub);
      let userId = getString(tokenRecord.userId);
      let username = getString(tokenRecord.username);

      if (!userId) {
        const socialUser = await findSocialUserByCasdoorId(casdoorId);
        if (socialUser) {
          userId = socialUser.id;
          username ||= socialUser.username || undefined;
        }
      }

      const sessionUser = session.user as Record<string, unknown>;
      if (userId) {
        sessionUser.id = userId;
      }
      if (casdoorId) {
        sessionUser.casdoorId = casdoorId;
      }
      if (username) {
        sessionUser.username = username;
      }
      return session;
    },
  },
  pages: {
    signIn: "/login",
  },
  secret: process.env.AUTH_SECRET,
  session: { strategy: "jwt" },
};

export default NextAuth(authOptions);
