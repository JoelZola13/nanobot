import NextAuth from "next-auth";
import type { NextAuthOptions } from "next-auth";
import { prisma } from "./prisma";

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
        params: { scope: "openid profile email" },
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
      await prisma.user.upsert({
        where: { casdoorId: profile.sub as string },
        update: {
          displayName: (profile.name as string) || user.name || "User",
          email: user.email || "",
          avatarUrl: (user.image as string) || null,
        },
        create: {
          casdoorId: profile.sub as string,
          username:
            ((profile as Record<string, unknown>).preferred_username as string) ||
            (profile.name as string) ||
            (profile.sub as string),
          displayName: (profile.name as string) || user.name || "User",
          email: user.email || "",
          avatarUrl: (user.image as string) || null,
        },
      });
      return true;
    },
    async jwt({ token, profile }) {
      if (profile) {
        token.casdoorId = profile.sub;
        token.username =
          ((profile as Record<string, unknown>).preferred_username as string) ||
          (profile.name as string) ||
          "";
        const dbUser = await prisma.user.findUnique({
          where: { casdoorId: profile.sub as string },
          select: { id: true },
        });
        if (dbUser) token.userId = dbUser.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (token.userId) {
        (session.user as Record<string, unknown>).id = token.userId as string;
      }
      if (token.username) {
        (session.user as Record<string, unknown>).username =
          token.username as string;
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
