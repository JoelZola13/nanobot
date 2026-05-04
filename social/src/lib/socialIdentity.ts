import pg from "pg";

const DB_URL =
  process.env.DATABASE_URL ||
  "postgresql://social:social_password@social-postgres:5432/social";

const identityPool = new pg.Pool({ connectionString: DB_URL, max: 3 });

export type SocialUserIdentity = {
  id: string;
  username: string | null;
  displayName?: string | null;
  email?: string | null;
  avatarUrl?: string | null;
};

export type SocialIdentityInput = {
  casdoorId: string;
  username?: string | null;
  displayName?: string | null;
  email?: string | null;
  avatarUrl?: string | null;
};

export function getString(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function normalizeUsername(value: string): string {
  const normalized = value
    .toLowerCase()
    .replace(/@.*$/, "")
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 36);
  return normalized || "user";
}

async function uniqueUsername(baseUsername: string): Promise<string> {
  const base = normalizeUsername(baseUsername);
  let candidate = base;
  for (let i = 1; i < 50; i += 1) {
    const result = await identityPool.query<{ id: string }>(
      `SELECT id FROM users WHERE username = $1 LIMIT 1`,
      [candidate],
    );
    if (result.rowCount === 0) return candidate;
    candidate = `${base}-${i + 1}`;
  }
  return `${base}-${Date.now().toString(36)}`;
}

export async function findSocialUserByCasdoorId(
  casdoorId: unknown,
): Promise<SocialUserIdentity | null> {
  const resolvedCasdoorId = getString(casdoorId);
  if (!resolvedCasdoorId) return null;

  try {
    const result = await identityPool.query<SocialUserIdentity>(
      `SELECT id, username, display_name AS "displayName", email, avatar_url AS "avatarUrl"
       FROM users
       WHERE casdoor_id = $1`,
      [resolvedCasdoorId],
    );
    return result.rows[0] || null;
  } catch (e) {
    console.error("[AUTH] DB lookup failed:", (e as Error).message);
    return null;
  }
}

export async function upsertSocialUserFromIdentity(
  identity: SocialIdentityInput,
): Promise<SocialUserIdentity | null> {
  const casdoorId = getString(identity.casdoorId);
  if (!casdoorId) return null;

  const email =
    getString(identity.email) || `${casdoorId.replace(/[^a-zA-Z0-9._-]/g, "_")}@streetvoices.local`;
  const displayName = getString(identity.displayName) || getString(identity.username) || email;
  const avatarUrl = getString(identity.avatarUrl) || null;

  try {
    const existing = await identityPool.query<SocialUserIdentity>(
      `SELECT id, username, display_name AS "displayName", email, avatar_url AS "avatarUrl"
       FROM users
       WHERE casdoor_id = $1 OR email = $2
       LIMIT 1`,
      [casdoorId, email],
    );

    if (existing.rows[0]) {
      const updated = await identityPool.query<SocialUserIdentity>(
        `UPDATE users
         SET casdoor_id = $2,
             display_name = $3,
             email = $4,
             avatar_url = $5,
             updated_at = NOW()
         WHERE id = $1
         RETURNING id, username, display_name AS "displayName", email, avatar_url AS "avatarUrl"`,
        [existing.rows[0].id, casdoorId, displayName, email, avatarUrl],
      );
      return updated.rows[0] || existing.rows[0];
    }

    const username = await uniqueUsername(
      getString(identity.username) || getString(identity.email) || displayName,
    );
    const inserted = await identityPool.query<SocialUserIdentity>(
      `INSERT INTO users (id, casdoor_id, username, display_name, email, avatar_url, created_at, updated_at)
       VALUES (gen_random_uuid()::text, $1, $2, $3, $4, $5, NOW(), NOW())
       RETURNING id, username, display_name AS "displayName", email, avatar_url AS "avatarUrl"`,
      [casdoorId, username, displayName, email, avatarUrl],
    );
    return inserted.rows[0] || null;
  } catch (e) {
    console.error("[AUTH] DB upsert failed:", (e as Error).message);
    return null;
  }
}
