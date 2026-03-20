export async function register() {
  // Ensure critical env vars are available at runtime for Next.js App Router
  // route handlers. Docker-compose env vars may not be visible in the webpack
  // execution context, so we force-set defaults here.
  const defaults: Record<string, string> = {
    DATABASE_URL:
      "postgresql://lobehub:lobehub_password@streetvoices-postgresql:5432/social",
    NEXTAUTH_URL: "http://localhost:3180/social/api/auth",
    AUTH_CASDOOR_ISSUER: "http://localhost:8380",
    AUTH_CASDOOR_ID: "social-app",
    AUTH_CASDOOR_SECRET: "social-casdoor-secret-2026",
    AUTH_SECRET: "street-voices-social-secret-2026",
  };

  for (const [key, value] of Object.entries(defaults)) {
    if (!process.env[key]) {
      process.env[key] = value;
    }
  }
}
