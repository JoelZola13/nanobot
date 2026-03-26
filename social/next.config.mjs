/** @type {import('next').NextConfig} */
const nextConfig = {
  basePath: "/social",
  experimental: {
    instrumentationHook: true,
  },
  // Inline env vars at build time via webpack DefinePlugin.
  // Required because Next.js App Router route handlers in production
  // don't have access to runtime process.env from Docker.
  env: {
    NEXTAUTH_URL: process.env.NEXTAUTH_URL || "http://localhost:3180/social/api/auth",
    AUTH_CASDOOR_ISSUER: process.env.AUTH_CASDOOR_ISSUER || "http://localhost:8380",
    AUTH_CASDOOR_ID: process.env.AUTH_CASDOOR_ID || "social-app",
    AUTH_CASDOOR_SECRET: process.env.AUTH_CASDOOR_SECRET || "social-casdoor-secret-2026",
    AUTH_SECRET: process.env.AUTH_SECRET || "street-voices-social-secret-2026",
    DATABASE_URL: process.env.DATABASE_URL || "postgresql://lobehub:lobehub_password@localhost:5433/social",
    NANOBOT_API_URL: process.env.NANOBOT_API_URL || "http://localhost:18790/v1",
    NANOBOT_API_KEY: process.env.NANOBOT_API_KEY || "nanobot",
  },
};

export default nextConfig;
