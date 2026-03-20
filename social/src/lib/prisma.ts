import { PrismaClient } from "@/generated/prisma/client";
import { PrismaPg } from "@prisma/adapter-pg";
import pg from "pg";

const globalForPrisma = globalThis as unknown as { prisma: PrismaClient };

// Use localhost:5433 — the container's /etc/hosts maps localhost to the Docker
// host (via entrypoint script), and PostgreSQL is exposed on host port 5433.
const DB_URL =
  process.env.DATABASE_URL ||
  "postgresql://lobehub:lobehub_password@localhost:5433/social";

function createPrismaClient() {
  const pool = new pg.Pool({ connectionString: DB_URL });
  const adapter = new PrismaPg(pool);
  return new PrismaClient({ adapter });
}

export const prisma = globalForPrisma.prisma || createPrismaClient();

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
