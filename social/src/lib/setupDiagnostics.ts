import { authOptions } from "@/lib/auth";
import { prisma } from "@/lib/prisma";

export type SetupDiagnosticStatus = "ok" | "warning" | "error";

export type SetupDiagnosticCheck = {
  id: string;
  label: string;
  status: SetupDiagnosticStatus;
  summary: string;
  action?: string;
};

export type SetupDiagnostics = {
  service: "street-voices-social";
  generatedAt: string;
  status: SetupDiagnosticStatus;
  command: string;
  checks: SetupDiagnosticCheck[];
};

type QueryablePrisma = Pick<typeof prisma, "$queryRaw">;

type SetupDiagnosticsDeps = {
  env?: NodeJS.ProcessEnv;
  fetchImpl?: typeof fetch;
  now?: () => Date;
  prismaClient?: QueryablePrisma;
  timeoutMs?: number;
};

const DEFAULT_BRIDGE_URL = "http://api:3180/api/auth/social-session";
const HEALTH_COMMAND = "cd social && npm run health";
const DEFAULT_TIMEOUT_MS = 3500;

function resolveStatus(checks: SetupDiagnosticCheck[]): SetupDiagnosticStatus {
  if (checks.some((check) => check.status === "error")) return "error";
  if (checks.some((check) => check.status === "warning")) return "warning";
  return "ok";
}

function hasValue(value: string | undefined) {
  return Boolean(value?.trim());
}

function formatBridgeTarget(bridgeUrl: string) {
  try {
    const url = new URL(bridgeUrl);
    return `${url.host}${url.pathname}`;
  } catch {
    return bridgeUrl;
  }
}

function checkAuthConfig(env: NodeJS.ProcessEnv): SetupDiagnosticCheck {
  const missing = [
    "AUTH_SECRET",
    "AUTH_CASDOOR_ISSUER",
    "AUTH_CASDOOR_ID",
    "AUTH_CASDOOR_SECRET",
  ].filter((key) => !hasValue(env[key]));
  const hasCasdoorProvider = authOptions.providers.some((provider) => provider.id === "casdoor");

  if (!hasCasdoorProvider) {
    return {
      id: "social-auth-provider",
      label: "Social auth provider",
      status: "error",
      summary: "Casdoor is not registered as a Social NextAuth provider.",
      action: "Check social/src/lib/auth.ts and the Social auth environment.",
    };
  }

  if (missing.length > 0) {
    return {
      id: "social-auth-provider",
      label: "Social auth provider",
      status: "error",
      summary: `Missing ${missing.join(", ")}.`,
      action: "Copy the teammate secrets into the Social service and restart sv-social.",
    };
  }

  return {
    id: "social-auth-provider",
    label: "Social auth provider",
    status: "ok",
    summary: "Casdoor provider and Social auth secrets are configured.",
  };
}

async function checkDatabase(prismaClient: QueryablePrisma): Promise<SetupDiagnosticCheck> {
  try {
    const rows = await prismaClient.$queryRaw<Array<Record<string, unknown>>>`
      SELECT
        to_regclass('public.users')::text AS users_table,
        to_regclass('public.channels')::text AS channels_table
    `;
    const row = rows[0] || {};
    const usersTable = typeof row.users_table === "string" ? row.users_table : "";
    const channelsTable = typeof row.channels_table === "string" ? row.channels_table : "";

    if (usersTable && channelsTable) {
      return {
        id: "social-database",
        label: "Social database",
        status: "ok",
        summary: "social-postgres is reachable and core tables are present.",
      };
    }

    return {
      id: "social-database",
      label: "Social database",
      status: "error",
      summary: "social-postgres is reachable, but core tables are missing.",
      action: "Start the normal Docker stack and run the Social Prisma migrations.",
    };
  } catch {
    return {
      id: "social-database",
      label: "Social database",
      status: "error",
      summary: "Social cannot connect to social-postgres.",
      action: `Check the social-postgres container, then run ${HEALTH_COMMAND}.`,
    };
  }
}

async function checkLibreChatBridge(
  env: NodeJS.ProcessEnv,
  fetchImpl: typeof fetch,
  timeoutMs: number,
): Promise<SetupDiagnosticCheck> {
  const bridgeUrl = env.LIBRECHAT_AUTH_BRIDGE_URL || DEFAULT_BRIDGE_URL;
  const bridgeSecret = env.LIBRECHAT_AUTH_BRIDGE_SECRET?.trim();
  const bridgeTarget = formatBridgeTarget(bridgeUrl);

  if (!bridgeSecret) {
    return {
      id: "librechat-auth-bridge",
      label: "LibreChat auth bridge",
      status: "error",
      summary: "LIBRECHAT_AUTH_BRIDGE_SECRET is missing in the Social service.",
      action: "Set the same LIBRECHAT_AUTH_BRIDGE_SECRET for LibreChat and sv-social.",
    };
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetchImpl(bridgeUrl, {
      method: "POST",
      headers: {
        "x-librechat-social-secret": bridgeSecret,
      },
      cache: "no-store",
      signal: controller.signal,
    });

    if (response.status === 401) {
      return {
        id: "librechat-auth-bridge",
        label: "LibreChat auth bridge",
        status: "ok",
        summary: `Bridge route is reachable at ${bridgeTarget}; no user cookie was expected for this diagnostic.`,
      };
    }

    if (response.ok) {
      return {
        id: "librechat-auth-bridge",
        label: "LibreChat auth bridge",
        status: "ok",
        summary: `Bridge route is reachable at ${bridgeTarget}.`,
      };
    }

    if (response.status === 403) {
      return {
        id: "librechat-auth-bridge",
        label: "LibreChat auth bridge",
        status: "error",
        summary: "LibreChat rejected the Social bridge secret.",
        action: "Make LIBRECHAT_AUTH_BRIDGE_SECRET identical in LibreChat and sv-social.",
      };
    }

    if ([500, 502, 503, 504].includes(response.status)) {
      return {
        id: "librechat-auth-bridge",
        label: "LibreChat auth bridge",
        status: "error",
        summary: `LibreChat bridge returned ${response.status}.`,
        action: "Confirm the LibreChat API container is running and reachable from sv-social.",
      };
    }

    return {
      id: "librechat-auth-bridge",
      label: "LibreChat auth bridge",
      status: "warning",
      summary: `LibreChat bridge returned ${response.status}.`,
      action: `Run ${HEALTH_COMMAND} for the full host-side auth check.`,
    };
  } catch (error) {
    return {
      id: "librechat-auth-bridge",
      label: "LibreChat auth bridge",
      status: "error",
      summary: error instanceof Error && error.name === "AbortError"
        ? "LibreChat bridge timed out."
        : "Social cannot reach the LibreChat bridge route.",
      action: "Confirm the LibreChat API service is running on the Docker network.",
    };
  } finally {
    clearTimeout(timeout);
  }
}

export async function runSetupDiagnostics({
  env = process.env,
  fetchImpl = fetch,
  now = () => new Date(),
  prismaClient = prisma,
  timeoutMs = DEFAULT_TIMEOUT_MS,
}: SetupDiagnosticsDeps = {}): Promise<SetupDiagnostics> {
  const checks = [
    {
      id: "social-service",
      label: "Social service",
      status: "ok" as const,
      summary: "sv-social is serving diagnostics.",
    },
    checkAuthConfig(env),
    await checkDatabase(prismaClient),
    await checkLibreChatBridge(env, fetchImpl, timeoutMs),
    {
      id: "host-health-check",
      label: "Host health check",
      status: "ok" as const,
      summary: "Server-side checks passed far enough to reach the host-side health check step.",
      action: `Run ${HEALTH_COMMAND} from your Nanobot checkout for the full local-stack pass.`,
    },
  ];

  return {
    service: "street-voices-social",
    generatedAt: now().toISOString(),
    status: resolveStatus(checks),
    command: HEALTH_COMMAND,
    checks,
  };
}
