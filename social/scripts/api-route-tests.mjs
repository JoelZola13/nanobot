#!/usr/bin/env node

import assert from "node:assert/strict";
import { afterEach, describe, test } from "node:test";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Module from "node:module";
import ts from "typescript";

const ROOT_DIR = join(dirname(fileURLToPath(import.meta.url)), "..");

const nextServerMock = {
  NextRequest: class NextRequest {},
  NextResponse: {
    json(body, init = {}) {
      return Response.json(body, init);
    },
  },
};

const originalFetch = globalThis.fetch;
const originalConsoleError = console.error;
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;
const originalWindow = globalThis.window;

afterEach(() => {
  globalThis.fetch = originalFetch;
  console.error = originalConsoleError;
  console.log = originalConsoleLog;
  console.warn = originalConsoleWarn;
  if (originalWindow === undefined) {
    delete globalThis.window;
  } else {
    globalThis.window = originalWindow;
  }
  delete process.env.LIBRECHAT_AUTH_BRIDGE_URL;
});

function loadTsModule(relativePath, mocks = {}) {
  const filename = join(ROOT_DIR, relativePath);
  const source = readFileSync(filename, "utf8");
  const { outputText } = ts.transpileModule(source, {
    fileName: filename,
    compilerOptions: {
      esModuleInterop: true,
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
    },
  });

  const loadedModule = new Module(filename);
  loadedModule.filename = filename;
  loadedModule.paths = Module._nodeModulePaths(dirname(filename));

  const previousLoad = Module._load;
  Module._load = (request, parent, isMain) => {
    if (Object.prototype.hasOwnProperty.call(mocks, request)) {
      return mocks[request];
    }
    return previousLoad(request, parent, isMain);
  };

  try {
    loadedModule._compile(outputText, filename);
  } finally {
    Module._load = previousLoad;
  }

  return loadedModule.exports;
}

function jsonRequest(body) {
  return {
    async json() {
      return body;
    },
  };
}

async function responseJson(response) {
  return response.json();
}

function createAuthMock(userId = "current-user") {
  return {
    async auth() {
      if (!userId) return null;
      return { user: { id: userId, name: "Current User" } };
    },
  };
}

function createUnauthenticatedMocks() {
  return {
    "next/server": nextServerMock,
    "@/lib/session": createAuthMock(null),
    "@/lib/prisma": { prisma: {} },
  };
}

describe("Social structured logs", () => {
  test("writes stable JSON log entries", () => {
    const lines = [];
    console.warn = (line) => lines.push(line);
    const { socialLog } = loadTsModule("src/lib/telemetry.ts");

    socialLog("warn", "social.auth.bridge_unavailable", {
      status: 503,
      omitted: undefined,
    });

    assert.equal(lines.length, 1);
    const entry = JSON.parse(lines[0]);
    assert.equal(entry.level, "warn");
    assert.equal(entry.service, "street-voices-social");
    assert.equal(entry.event, "social.auth.bridge_unavailable");
    assert.equal(entry.status, 503);
    assert.equal(typeof entry.timestamp, "string");
    assert.equal(Object.prototype.hasOwnProperty.call(entry, "omitted"), false);
  });
});

describe("Unread count hydration", () => {
  test("counts unread messages after the latest read receipt or join time", async () => {
    const joinedAtA = new Date("2026-05-01T10:00:00.000Z");
    const joinedAtB = new Date("2026-05-02T10:00:00.000Z");
    const readAtA = new Date("2026-05-03T10:00:00.000Z");
    const countCalls = [];
    const { getInitialUnreadCountsForUser } = loadTsModule("src/lib/unreadCounts.ts", {
      "./prisma": {
        prisma: {
          readReceipt: {
            async findMany(args) {
              assert.deepEqual(args.where, {
                userId: "user-1",
                channelId: { in: ["channel-a", "channel-b"] },
              });
              return [{ channelId: "channel-a", readAt: readAtA }];
            },
          },
          message: {
            async count(args) {
              countCalls.push(args);
              return args.where.channelId === "channel-a" ? 2 : 0;
            },
          },
        },
      },
    });

    const counts = await getInitialUnreadCountsForUser("user-1", [
      { channelId: "channel-a", joinedAt: joinedAtA },
      { channelId: "channel-b", joinedAt: joinedAtB },
    ]);

    assert.deepEqual(counts, { "channel-a": 2 });
    assert.equal(countCalls.length, 2);
    assert.deepEqual(countCalls[0].where, {
      channelId: "channel-a",
      deletedAt: null,
      authorId: { not: "user-1" },
      createdAt: { gt: readAtA },
    });
    assert.deepEqual(countCalls[1].where, {
      channelId: "channel-b",
      deletedAt: null,
      authorId: { not: "user-1" },
      createdAt: { gt: joinedAtB },
    });
  });
});

describe("Message draft storage", () => {
  function createDraftStorage() {
    const values = new Map();
    return {
      getItem(key) {
        return values.get(key) ?? null;
      },
      setItem(key, value) {
        values.set(key, value);
      },
      removeItem(key) {
        values.delete(key);
      },
    };
  }

  test("keeps drafts isolated by conversation key", () => {
    const storage = createDraftStorage();
    const {
      readMessageDraft,
      writeMessageDraft,
      getMessageDraftStorageKey,
    } = loadTsModule("src/lib/messageDrafts.ts");

    writeMessageDraft("user-1:channel:general", "hello channel", storage);
    writeMessageDraft("user-1:channel:dm-1", "hello dm", storage);

    assert.equal(readMessageDraft("user-1:channel:general", storage), "hello channel");
    assert.equal(readMessageDraft("user-1:channel:dm-1", storage), "hello dm");
    assert.notEqual(
      getMessageDraftStorageKey("user-1:channel:general"),
      getMessageDraftStorageKey("user-1:channel:dm-1"),
    );
  });

  test("clears blank and sent drafts", () => {
    const storage = createDraftStorage();
    const {
      clearMessageDraft,
      readMessageDraft,
      writeMessageDraft,
    } = loadTsModule("src/lib/messageDrafts.ts");

    writeMessageDraft("user-1:channel:general", "working draft", storage);
    writeMessageDraft("user-1:channel:general", "   ", storage);
    assert.equal(readMessageDraft("user-1:channel:general", storage), "");

    writeMessageDraft("user-1:channel:general", "second draft", storage);
    clearMessageDraft("user-1:channel:general", storage);
    assert.equal(readMessageDraft("user-1:channel:general", storage), "");
  });
});

describe("Browser notification helpers", () => {
  function createStorage() {
    const values = new Map();
    return {
      getItem(key) {
        return values.get(key) ?? null;
      },
      setItem(key, value) {
        values.set(key, value);
      },
    };
  }

  test("shows the quiet prompt only while permission is undecided and not dismissed", () => {
    const storage = createStorage();
    class FakeNotification {
      static permission = "default";
      static requestPermission = async () => "granted";
    }
    globalThis.window = {
      Notification: FakeNotification,
      localStorage: storage,
    };
    const {
      dismissBrowserNotificationPrompt,
      shouldShowBrowserNotificationPrompt,
    } = loadTsModule("src/lib/browserNotifications.ts");

    assert.equal(shouldShowBrowserNotificationPrompt(storage), true);
    dismissBrowserNotificationPrompt(storage);
    assert.equal(shouldShowBrowserNotificationPrompt(storage), false);
  });

  test("creates a clickable message notification for granted permission", () => {
    const notifications = [];
    let focused = false;
    class FakeNotification {
      static permission = "granted";
      static requestPermission = async () => "granted";

      constructor(title, options) {
        this.title = title;
        this.options = options;
        this.closed = false;
        notifications.push(this);
      }

      close() {
        this.closed = true;
      }
    }
    globalThis.window = {
      Notification: FakeNotification,
      localStorage: createStorage(),
      location: { href: "" },
      focus() {
        focused = true;
      },
    };
    const { showBrowserMessageNotification } = loadTsModule("src/lib/browserNotifications.ts");

    const notification = showBrowserMessageNotification(
      {
        id: "message-1",
        channelId: "channel-1",
        content: "**Hello** [team](https://example.test)",
        createdAt: "2026-05-04T12:00:00.000Z",
        isEdited: false,
        isPinned: false,
        parentId: null,
        author: {
          id: "user-2",
          username: "alex",
          displayName: "Alex Rivera",
          avatarUrl: "https://example.test/alex.png",
          isAgent: false,
        },
        reactions: [],
        attachments: [],
      },
      {
        label: "general",
        href: "/channels/channel-1",
        type: "channel",
      },
    );

    assert.equal(notification, notifications[0]);
    assert.equal(notifications[0].title, "Alex Rivera in #general");
    assert.equal(notifications[0].options.body, "Hello team");
    assert.equal(notifications[0].options.icon, "https://example.test/alex.png");
    notifications[0].onclick();
    assert.equal(focused, true);
    assert.equal(globalThis.window.location.href, "/channels/channel-1");
    assert.equal(notifications[0].closed, true);
  });
});

describe("DM API route", () => {
  test("rejects unauthenticated DM creation", async () => {
    const { POST } = loadTsModule("src/app/api/dm/route.ts", createUnauthenticatedMocks());

    const response = await POST(jsonRequest({ userId: "other-user" }));

    assert.equal(response.status, 401);
    assert.deepEqual(await responseJson(response), { error: "Unauthorized" });
  });

  test("requires the other user id", async () => {
    const { POST } = loadTsModule("src/app/api/dm/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(),
      "@/lib/prisma": { prisma: {} },
    });

    const response = await POST(jsonRequest({}));

    assert.equal(response.status, 400);
    assert.deepEqual(await responseJson(response), { error: "userId required" });
  });

  test("returns an existing DM instead of creating a duplicate", async () => {
    let createCalled = false;
    const prisma = {
      user: {
        async findUnique({ where }) {
          if (where.id === "current-user") return { id: "current-user" };
          if (where.id === "other-user") return { displayName: "Other User", username: "other" };
          return null;
        },
      },
      channel: {
        async findFirst() {
          return { id: "existing-dm", _count: { members: 2 } };
        },
        async create() {
          createCalled = true;
          return { id: "should-not-be-used" };
        },
      },
    };
    const { POST } = loadTsModule("src/app/api/dm/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(),
      "@/lib/prisma": { prisma },
    });

    const response = await POST(jsonRequest({ userId: "other-user" }));

    assert.equal(response.status, 200);
    assert.deepEqual(await responseJson(response), { channelId: "existing-dm" });
    assert.equal(createCalled, false);
  });

  test("creates a two-member DM channel when none exists", async () => {
    let createArgs;
    const prisma = {
      user: {
        async findUnique({ where }) {
          if (where.id === "current-user") return { id: "current-user" };
          if (where.id === "other-user") return { displayName: "Other User", username: "other" };
          return null;
        },
      },
      channel: {
        async findFirst() {
          return null;
        },
        async create(args) {
          createArgs = args;
          return { id: "new-dm" };
        },
      },
    };
    const { POST } = loadTsModule("src/app/api/dm/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(),
      "@/lib/prisma": { prisma },
    });

    const response = await POST(jsonRequest({ userId: "other-user" }));

    assert.equal(response.status, 201);
    assert.deepEqual(await responseJson(response), { channelId: "new-dm" });
    assert.equal(createArgs.data.type, "DM");
    assert.equal(createArgs.data.slug, "dm-current-user-other-user");
    assert.deepEqual(createArgs.data.members.create, [
      { userId: "current-user", role: "member" },
      { userId: "other-user", role: "member" },
    ]);
  });
});

describe("Channel membership API route", () => {
  test("joins a public channel with a member role", async () => {
    let upsertArgs;
    const prisma = {
      channel: {
        async findUnique() {
          return { id: "channel-1", type: "PUBLIC", isArchived: false };
        },
      },
      channelMember: {
        async findUnique() {
          return null;
        },
        async upsert(args) {
          upsertArgs = args;
          return { role: "member" };
        },
      },
    };
    const { POST } = loadTsModule("src/app/api/channels/[id]/membership/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(),
      "@/lib/prisma": { prisma },
    });

    const response = await POST(jsonRequest({}), { params: Promise.resolve({ id: "channel-1" }) });

    assert.equal(response.status, 200);
    assert.deepEqual(await responseJson(response), {
      channelId: "channel-1",
      isMember: true,
      role: "member",
    });
    assert.equal(upsertArgs.create.role, "member");
    assert.deepEqual(upsertArgs.where.channelId_userId, {
      channelId: "channel-1",
      userId: "current-user",
    });
  });

  test("blocks joining a private channel without an existing membership", async () => {
    let upsertCalled = false;
    const prisma = {
      channel: {
        async findUnique() {
          return { id: "private-1", type: "PRIVATE", isArchived: false };
        },
      },
      channelMember: {
        async findUnique() {
          return null;
        },
        async upsert() {
          upsertCalled = true;
          return { role: "member" };
        },
      },
    };
    const { POST } = loadTsModule("src/app/api/channels/[id]/membership/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(),
      "@/lib/prisma": { prisma },
    });

    const response = await POST(jsonRequest({}), { params: Promise.resolve({ id: "private-1" }) });

    assert.equal(response.status, 403);
    assert.deepEqual(await responseJson(response), {
      error: "Private channels require an invitation",
    });
    assert.equal(upsertCalled, false);
  });

  test("leaves a non-default public channel", async () => {
    let deleteArgs;
    const prisma = {
      channel: {
        async findUnique() {
          return { id: "channel-1", type: "PUBLIC", isArchived: false, isDefault: false };
        },
      },
      channelMember: {
        async deleteMany(args) {
          deleteArgs = args;
        },
      },
    };
    const { DELETE } = loadTsModule("src/app/api/channels/[id]/membership/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(),
      "@/lib/prisma": { prisma },
    });

    const response = await DELETE(jsonRequest({}), { params: Promise.resolve({ id: "channel-1" }) });

    assert.equal(response.status, 200);
    assert.deepEqual(await responseJson(response), { channelId: "channel-1", isMember: false });
    assert.deepEqual(deleteArgs.where, { channelId: "channel-1", userId: "current-user" });
  });

  test("does not allow leaving a default channel", async () => {
    let deleteCalled = false;
    const prisma = {
      channel: {
        async findUnique() {
          return { id: "channel-general", type: "PUBLIC", isArchived: false, isDefault: true };
        },
      },
      channelMember: {
        async deleteMany() {
          deleteCalled = true;
        },
      },
    };
    const { DELETE } = loadTsModule("src/app/api/channels/[id]/membership/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(),
      "@/lib/prisma": { prisma },
    });

    const response = await DELETE(jsonRequest({}), { params: Promise.resolve({ id: "channel-general" }) });

    assert.equal(response.status, 400);
    assert.deepEqual(await responseJson(response), {
      error: "Default channels cannot be left",
    });
    assert.equal(deleteCalled, false);
  });
});

describe("Social session bridge auth", () => {
  const getString = (value) => {
    if (typeof value !== "string") return undefined;
    const trimmed = value.trim();
    return trimmed || undefined;
  };

  function loadSessionModule({ cookie = "", serverSession = null, socialUser = null } = {}) {
    const upsertCalls = [];
    const telemetryCalls = [];
    const moduleExports = loadTsModule("src/lib/session.ts", {
      "next-auth": {
        async getServerSession() {
          return serverSession;
        },
      },
      "next/headers": {
        headers() {
          return {
            get(name) {
              return name.toLowerCase() === "cookie" ? cookie : "";
            },
          };
        },
      },
      "./auth": { authOptions: {} },
      "./telemetry": {
        socialLog(level, event, fields) {
          telemetryCalls.push({ level, event, fields });
        },
      },
      "./socialIdentity": {
        getString,
        async upsertSocialUserFromIdentity(identity) {
          upsertCalls.push(identity);
          return socialUser;
        },
      },
    });

    return { moduleExports, upsertCalls, telemetryCalls };
  }

  test("uses an existing NextAuth session before calling the bridge", async () => {
    const existingSession = { user: { id: "social-user" }, expires: "2099-01-01T00:00:00.000Z" };
    globalThis.fetch = async () => {
      throw new Error("bridge should not be called");
    };
    const { moduleExports } = loadSessionModule({ cookie: "refreshToken=token", serverSession: existingSession });

    const session = await moduleExports.auth();

    assert.equal(session, existingSession);
  });

  test("returns null when there is no session cookie", async () => {
    globalThis.fetch = async () => {
      throw new Error("bridge should not be called");
    };
    const { moduleExports } = loadSessionModule();

    const session = await moduleExports.auth();

    assert.equal(session, null);
  });

  test("creates a Social session from the LibreChat bridge user", async () => {
    process.env.LIBRECHAT_AUTH_BRIDGE_URL = "http://bridge.test/session";
    const bridgeRequests = [];
    globalThis.fetch = async (url, options) => {
      bridgeRequests.push({ url, options });
      return Response.json({
        user: {
          id: "casdoor-user-1",
          username: "jane",
          name: "Jane Doe",
          email: "jane@example.test",
          avatar: "https://example.test/jane.png",
        },
      });
    };
    const { moduleExports, upsertCalls } = loadSessionModule({
      cookie: "refreshToken=bridge-cookie",
      socialUser: {
        id: "social-user-1",
        username: "jane",
        displayName: "Jane Local",
        email: "jane@example.test",
        avatarUrl: "https://example.test/local.png",
      },
    });

    const session = await moduleExports.auth();

    assert.equal(bridgeRequests[0].url, "http://bridge.test/session");
    assert.equal(bridgeRequests[0].options.headers.cookie, "refreshToken=bridge-cookie");
    assert.deepEqual(upsertCalls[0], {
      casdoorId: "casdoor-user-1",
      username: "jane",
      displayName: "Jane Doe",
      email: "jane@example.test",
      avatarUrl: "https://example.test/jane.png",
    });
    assert.equal(session.user.id, "social-user-1");
    assert.equal(session.user.name, "Jane Local");
    assert.equal(session.user.username, "jane");
    assert.equal(session.user.casdoorId, "casdoor-user-1");
  });

  test("throws the bridge unavailable error when requested", async () => {
    process.env.LIBRECHAT_AUTH_BRIDGE_URL = "http://bridge.test/session";
    globalThis.fetch = async () => new Response("bridge down", { status: 503 });
    const { moduleExports, telemetryCalls } = loadSessionModule({ cookie: "refreshToken=bridge-cookie" });

    await assert.rejects(
      () => moduleExports.auth({ bridgeUnavailable: "throw" }),
      (error) =>
        error?.name === "LibreChatBridgeUnavailableError" &&
        error?.status === 503 &&
        String(error.message).includes("bridge down"),
    );

    assert.equal(telemetryCalls.length, 1);
    assert.equal(telemetryCalls[0].level, "error");
    assert.equal(telemetryCalls[0].event, "social.auth.bridge_unavailable");
    assert.equal(telemetryCalls[0].fields.status, 503);
    assert.equal(telemetryCalls[0].fields.bridgeUnavailableMode, "throw");
    assert.equal(telemetryCalls[0].fields.bridgeUrl, "http://bridge.test/session");
    assert.match(telemetryCalls[0].fields.message, /bridge down/);
  });
});
