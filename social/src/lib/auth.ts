import NextAuth from "next-auth";
import type { NextAuthOptions } from "next-auth";
import pg from "pg";

// Strip /api/auth from NEXTAUTH_URL to get the site base URL for redirect_uri
const siteUrl = (process.env.NEXTAUTH_URL || "http://localhost:3180/social/api/auth").replace(/\/api\/auth$/, "");

const DB_URL =
  process.env.DATABASE_URL ||
  "postgresql://lobehub:lobehub_password@localhost:5433/social";

// Use pg directly for auth callbacks — Prisma's WASM query engine has DNS
// resolution issues inside Docker containers with the adapter-pg setup.
const authPool = new pg.Pool({ connectionString: DB_URL, max: 3 });

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

        await authPool.query(
          `INSERT INTO users (id, casdoor_id, username, display_name, email, avatar_url, created_at, updated_at)
           VALUES (gen_random_uuid()::text, $1, $2, $3, $4, $5, NOW(), NOW())
           ON CONFLICT (casdoor_id) DO UPDATE SET
             display_name = EXCLUDED.display_name,
             email = EXCLUDED.email,
             avatar_url = EXCLUDED.avatar_url,
             updated_at = NOW()`,
          [casdoorId, username, displayName, email, avatarUrl],
        );
      } catch (e) {
        console.error("[AUTH] DB upsert failed:", (e as Error).message);
      }
      return true;
    },
    async jwt({ token, profile }) {
      if (profile) {
        token.casdoorId = profile.sub;
        token.username =
          ((profile as Record<string, unknown>).preferred_username as string) ||
          (profile.name as string) ||
          "";
        try {
          const result = await authPool.query(
            `SELECT id FROM users WHERE casdoor_id = $1`,
            [profile.sub],
          );
          if (result.rows[0]) token.userId = result.rows[0].id;
        } catch (e) {
          console.error("[AUTH] DB lookup failed:", (e as Error).message);
        }
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
