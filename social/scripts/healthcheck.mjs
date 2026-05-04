#!/usr/bin/env node

import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const DEFAULT_BRIDGE_SECRET = "street-voices-social-bridge-2026";
const BASE_URL = (process.env.SOCIAL_HEALTH_BASE_URL || "http://localhost:3180").replace(/\/$/, "");
const POSTGRES_CONTAINER = process.env.SOCIAL_POSTGRES_CONTAINER || "nanobot-social-postgres";
const BRIDGE_SECRET =
  process.env.SOCIAL_HEALTH_BRIDGE_SECRET ||
  process.env.LIBRECHAT_AUTH_BRIDGE_SECRET ||
  DEFAULT_BRIDGE_SECRET;
const REQUEST_TIMEOUT_MS = Number(process.env.SOCIAL_HEALTH_TIMEOUT_MS || 5000);

const checks = [];

function addCheck(name, run) {
  checks.push({ name, run });
}

function pass(summary) {
  return { ok: true, summary };
}

function fail(summary, details) {
  return { ok: false, summary, details };
}

async function fetchWithTimeout(url, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    return await fetch(url, {
      redirect: "follow",
      ...options,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeout);
  }
}

async function readText(response) {
  return response.text().catch(() => "");
}

async function getUrl(path) {
  return `${BASE_URL}${path}`;
}

async function dockerExec(args) {
  return execFileAsync("docker", args, { timeout: REQUEST_TIMEOUT_MS });
}

addCheck("LibreChat API health", async () => {
  const url = await getUrl("/health");
  const response = await fetchWithTimeout(url);

  if (response.ok) {
    return pass(`GET ${url} -> ${response.status}`);
  }

  return fail(`GET ${url} -> ${response.status}`, await readText(response));
});

addCheck("LibreChat auth bridge", async () => {
  const url = await getUrl("/api/auth/social-session");
  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: {
      "x-librechat-social-secret": BRIDGE_SECRET,
    },
  });

  if (response.status === 401) {
    return pass("bridge route reachable, secret accepted, no user cookie present");
  }

  if (response.ok) {
    return pass(`bridge returned ${response.status}`);
  }

  return fail(`bridge returned ${response.status}`, await readText(response));
});

addCheck("Social auth providers", async () => {
  const url = await getUrl("/social/api/auth/providers");
  const response = await fetchWithTimeout(url);

  if (!response.ok) {
    return fail(`GET ${url} -> ${response.status}`, await readText(response));
  }

  const providers = await response.json().catch(() => null);
  if (providers?.casdoor) {
    return pass("NextAuth casdoor provider is available");
  }

  return fail("NextAuth providers loaded, but casdoor is missing", JSON.stringify(providers));
});

addCheck("Social routes", async () => {
  const routes = ["/social/login", "/messages"];
  const results = [];

  for (const route of routes) {
    const url = await getUrl(route);
    const response = await fetchWithTimeout(url);
    const contentType = response.headers.get("content-type") || "";

    if (!response.ok) {
      return fail(`GET ${url} -> ${response.status}`, await readText(response));
    }

    if (!contentType.includes("text/html")) {
      return fail(`GET ${url} returned ${contentType || "unknown content type"}`);
    }

    results.push(`${route} ${response.status}`);
  }

  return pass(results.join(", "));
});

addCheck("Social database", async () => {
  try {
    await dockerExec([
      "exec",
      POSTGRES_CONTAINER,
      "pg_isready",
      "-U",
      "social",
      "-d",
      "social",
    ]);

    const { stdout } = await dockerExec([
      "exec",
      POSTGRES_CONTAINER,
      "psql",
      "-U",
      "social",
      "-d",
      "social",
      "-tA",
      "-c",
      "SELECT CASE WHEN to_regclass('public.users') IS NOT NULL AND to_regclass('public.channels') IS NOT NULL THEN 'ok' ELSE 'missing' END;",
    ]);

    if (stdout.trim() === "ok") {
      return pass(`${POSTGRES_CONTAINER} reachable, core tables present`);
    }

    return fail(`${POSTGRES_CONTAINER} reachable, core tables missing`, stdout.trim());
  } catch (error) {
    return fail(
      `${POSTGRES_CONTAINER} is not healthy or docker is unavailable`,
      error.stderr || error.message,
    );
  }
});

addCheck("Social socket route", async () => {
  const url = await getUrl(`/ws-social/?EIO=4&transport=polling&t=${Date.now()}`);
  const response = await fetchWithTimeout(url);
  const body = await readText(response);

  if (response.ok && body.startsWith("0")) {
    return pass("Socket.IO polling handshake succeeded on /ws-social");
  }

  return fail(`Socket.IO handshake returned ${response.status}`, body.slice(0, 300));
});

async function main() {
  console.log(`Street Voices Messages health check`);
  console.log(`Base URL: ${BASE_URL}`);
  console.log("");

  const results = [];
  for (const check of checks) {
    try {
      const result = await check.run();
      results.push({ name: check.name, ...result });
    } catch (error) {
      results.push({
        name: check.name,
        ok: false,
        summary: error.name === "AbortError" ? "timed out" : error.message,
        details: error.stack,
      });
    }
  }

  const nameWidth = Math.max(...results.map((result) => result.name.length));
  for (const result of results) {
    const status = result.ok ? "OK" : "FAIL";
    console.log(`${status.padEnd(4)} ${result.name.padEnd(nameWidth)}  ${result.summary}`);
    if (!result.ok && result.details) {
      console.log(`     ${String(result.details).replace(/\n/g, "\n     ")}`);
    }
  }

  const failed = results.filter((result) => !result.ok);
  console.log("");

  if (failed.length > 0) {
    console.log(`${failed.length} health check${failed.length === 1 ? "" : "s"} failed.`);
    process.exitCode = 1;
    return;
  }

  console.log("All health checks passed.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
