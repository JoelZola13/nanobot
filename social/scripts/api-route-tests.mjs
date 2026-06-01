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
  delete process.env.LIBRECHAT_AUTH_BRIDGE_SECRET;
  delete process.env.AUTH_SECRET;
  delete process.env.AUTH_CASDOOR_ISSUER;
  delete process.env.AUTH_CASDOOR_ID;
  delete process.env.AUTH_CASDOOR_SECRET;
  delete process.env.SOCIAL_DEFAULT_CHANNEL_VISIBILITY;
  delete process.env.SOCIAL_DEFAULT_NOTIFICATION_LEVEL;
  delete process.env.SOCIAL_PUBLIC_CHANNEL_JOIN_POLICY;
  delete process.env.EMAIL_REPLY_ENABLED;
  delete process.env.EMAIL_REPLY_CONSENT_GRANTED;
  delete process.env.EMAIL_REPLY_SMTP_HOST;
  delete process.env.EMAIL_REPLY_SMTP_PORT;
  delete process.env.EMAIL_REPLY_SMTP_USERNAME;
  delete process.env.EMAIL_REPLY_SMTP_USER;
  delete process.env.EMAIL_REPLY_SMTP_PASSWORD;
  delete process.env.EMAIL_REPLY_FROM;
  delete process.env.EMAIL_REPLY_FROM_ADDRESS;
  delete process.env.EMAIL_REPLY_SMTP_USE_TLS;
  delete process.env.EMAIL_REPLY_SMTP_USE_SSL;
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
    url: "http://social.test/api",
    nextUrl: new URL("http://social.test/api"),
    async json() {
      return body;
    },
    async text() {
      return body === undefined ? "" : JSON.stringify(body);
    },
  };
}

function urlRequest(url) {
  return {
    url,
    nextUrl: new URL(url),
    async json() {
      return {};
    },
    async text() {
      return "";
    },
  };
}

async function responseJson(response) {
  return response.json();
}

function createAuthMock(userId = "current-user", role) {
  return {
    async auth() {
      if (!userId) return null;
      return { user: { id: userId, name: "Current User", role } };
    },
  };
}

function createUnauthenticatedMocks() {
  return {
    "next/server": nextServerMock,
    "@/lib/session": createAuthMock(null),
    "@/lib/prisma": { prisma: {} },
    "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
  };
}

function loadWorkspacePoliciesModule() {
  const notificationPreferences = loadTsModule("src/lib/notificationPreferences.ts");
  const channelManagement = loadTsModule("src/lib/channelManagement.ts");

  return loadTsModule("src/lib/workspacePolicies.ts", {
    "@/lib/notificationPreferences": notificationPreferences,
    "@/lib/channelManagement": channelManagement,
  });
}

function loadMessageModerationModule() {
  const channelManagement = loadTsModule("src/lib/channelManagement.ts");

  return loadTsModule("src/lib/messageModeration.ts", {
    "@/lib/channelManagement": channelManagement,
  });
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

describe("Setup diagnostics", () => {
  function loadSetupDiagnosticsModule(prismaMock) {
    return loadTsModule("src/lib/setupDiagnostics.ts", {
      "@/lib/auth": {
        authOptions: {
          providers: [{ id: "casdoor" }],
        },
      },
      "@/lib/prisma": {
        prisma: prismaMock,
      },
    });
  }

  test("reports healthy checks when auth, database, and bridge are reachable", async () => {
    process.env.AUTH_SECRET = "auth-secret";
    process.env.AUTH_CASDOOR_ISSUER = "http://casdoor.test";
    process.env.AUTH_CASDOOR_ID = "client-id";
    process.env.AUTH_CASDOOR_SECRET = "client-secret";
    process.env.LIBRECHAT_AUTH_BRIDGE_URL = "http://bridge.test/session";
    process.env.LIBRECHAT_AUTH_BRIDGE_SECRET = "shared-secret";

    const bridgeRequests = [];
    globalThis.fetch = async (url, options) => {
      bridgeRequests.push({ url, options });
      return new Response("", { status: 401 });
    };
    const { runSetupDiagnostics } = loadSetupDiagnosticsModule({
      async $queryRaw() {
        return [{ users_table: "users", channels_table: "channels" }];
      },
    });

    const diagnostics = await runSetupDiagnostics({
      now: () => new Date("2026-05-04T12:00:00.000Z"),
    });

    assert.equal(diagnostics.status, "ok");
    assert.equal(diagnostics.generatedAt, "2026-05-04T12:00:00.000Z");
    assert.equal(bridgeRequests[0].url, "http://bridge.test/session");
    assert.equal(bridgeRequests[0].options.headers["x-librechat-social-secret"], "shared-secret");
    assert.deepEqual(
      diagnostics.checks.map((check) => [check.id, check.status]),
      [
        ["social-service", "ok"],
        ["social-auth-provider", "ok"],
        ["social-database", "ok"],
        ["librechat-auth-bridge", "ok"],
        ["host-health-check", "ok"],
      ],
    );
  });

  test("returns actionable errors for missing secrets and database tables", async () => {
    process.env.AUTH_CASDOOR_ISSUER = "http://casdoor.test";
    process.env.AUTH_CASDOOR_ID = "client-id";

    globalThis.fetch = async () => {
      throw new Error("bridge should not be called without a secret");
    };
    const { runSetupDiagnostics } = loadSetupDiagnosticsModule({
      async $queryRaw() {
        return [{ users_table: null, channels_table: null }];
      },
    });

    const diagnostics = await runSetupDiagnostics();
    const checksById = Object.fromEntries(diagnostics.checks.map((check) => [check.id, check]));

    assert.equal(diagnostics.status, "error");
    assert.equal(checksById["social-auth-provider"].status, "error");
    assert.match(checksById["social-auth-provider"].summary, /AUTH_SECRET/);
    assert.equal(checksById["social-database"].status, "error");
    assert.match(checksById["social-database"].summary, /core tables are missing/);
    assert.equal(checksById["librechat-auth-bridge"].status, "error");
    assert.match(checksById["librechat-auth-bridge"].summary, /LIBRECHAT_AUTH_BRIDGE_SECRET/);
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

describe("Unread message divider marker", () => {
  test("uses the later of read receipt time and join time", () => {
    const { resolveUnreadAfter } = loadTsModule("src/lib/unreadMarker.ts");

    assert.equal(
      resolveUnreadAfter(
        new Date("2026-05-02T12:00:00.000Z"),
        new Date("2026-05-01T12:00:00.000Z"),
      ),
      "2026-05-02T12:00:00.000Z",
    );
    assert.equal(
      resolveUnreadAfter(
        new Date("2026-05-01T12:00:00.000Z"),
        new Date("2026-05-02T12:00:00.000Z"),
      ),
      "2026-05-02T12:00:00.000Z",
    );
  });

  test("finds the first unread message from another user", () => {
    const { findFirstUnreadMessageId } = loadTsModule("src/lib/unreadMarker.ts");
    const messages = [
      {
        id: "read-teammate-message",
        createdAt: "2026-05-02T11:59:00.000Z",
        author: { id: "teammate" },
      },
      {
        id: "own-new-message",
        createdAt: "2026-05-02T12:01:00.000Z",
        author: { id: "current-user" },
      },
      {
        id: "first-unread-message",
        createdAt: "2026-05-02T12:02:00.000Z",
        author: { id: "teammate" },
      },
    ];

    assert.equal(
      findFirstUnreadMessageId(
        messages,
        "current-user",
        "2026-05-02T12:00:00.000Z",
      ),
      "first-unread-message",
    );
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

describe("Message deep links", () => {
  test("builds highlighted conversation hrefs for channel and DM messages", () => {
    const {
      getJumpToMessageLabel,
      getMessageHref,
    } = loadTsModule("src/lib/messageLinks.ts");

    assert.equal(
      getMessageHref({
        channelId: "channel-general",
        messageId: "message-1",
        channelType: "PUBLIC",
      }),
      "/channels/channel-general?message=message-1",
    );
    assert.equal(
      getMessageHref({
        channelId: "dm-channel-1",
        messageId: "message-2",
        channelType: "DM",
      }),
      "/dm/dm-channel-1?message=message-2",
    );
    assert.equal(getJumpToMessageLabel("mention"), "Jump to mention");
    assert.equal(getJumpToMessageLabel("saved"), "Jump to saved message");
    assert.equal(getJumpToMessageLabel("thread"), "Jump to thread");
    assert.equal(getJumpToMessageLabel("reaction"), "Jump to reacted message");
  });
});

describe("Activity inbox helpers", () => {
  test("merges mentions and saved items by newest activity", () => {
    const { mergeActivityItems } = loadTsModule("src/lib/activityItems.ts");

    const author = {
      id: "user-1",
      username: "joel",
      displayName: "Joel Zola",
      avatarUrl: null,
      isAgent: false,
    };
    const mentions = [
      {
        id: "mention-old",
        channelId: "channel-general",
        content: "@joel can you check this?",
        createdAt: "2026-05-01T10:00:00.000Z",
        href: "/channels/channel-general?message=mention-old",
        channelLabel: "#general",
        channelType: "PUBLIC",
        author,
      },
    ];
    const savedItems = [
      {
        id: "saved-new",
        messageId: "message-new",
        channelId: "dm-1",
        content: "Follow up here.",
        createdAt: "2026-05-01T09:00:00.000Z",
        savedAt: "2026-05-02T10:00:00.000Z",
        href: "/dm/dm-1?message=message-new",
        channelLabel: "Alex Rivera",
        channelType: "DM",
        author,
      },
    ];

    const items = mergeActivityItems({ mentions, savedItems });

    assert.equal(items.length, 2);
    assert.equal(items[0].id, "saved:saved-new");
    assert.equal(items[0].kind, "saved");
    assert.equal(items[1].id, "mention:mention-old");
    assert.equal(items[1].kind, "mention");
  });

  test("limits merged activity items after sorting", () => {
    const { mergeActivityItems } = loadTsModule("src/lib/activityItems.ts");

    const author = {
      id: "user-1",
      username: "joel",
      displayName: "Joel Zola",
      avatarUrl: null,
      isAgent: false,
    };
    const mentions = [
      {
        id: "mention-new",
        channelId: "channel-general",
        content: "@joel newer",
        createdAt: "2026-05-03T10:00:00.000Z",
        href: "/channels/channel-general?message=mention-new",
        channelLabel: "#general",
        channelType: "PUBLIC",
        author,
      },
      {
        id: "mention-old",
        channelId: "channel-general",
        content: "@joel older",
        createdAt: "2026-05-01T10:00:00.000Z",
        href: "/channels/channel-general?message=mention-old",
        channelLabel: "#general",
        channelType: "PUBLIC",
        author,
      },
    ];
    const savedItems = [
      {
        id: "saved-middle",
        messageId: "message-middle",
        channelId: "dm-1",
        content: "Follow up.",
        createdAt: "2026-05-02T08:00:00.000Z",
        savedAt: "2026-05-02T10:00:00.000Z",
        href: "/dm/dm-1?message=message-middle",
        channelLabel: "Alex Rivera",
        channelType: "DM",
        author,
      },
    ];

    const items = mergeActivityItems({ mentions, savedItems, limit: 2 });

    assert.deepEqual(
      items.map((item) => item.id),
      ["mention:mention-new", "saved:saved-middle"],
    );
  });

  test("counts and filters all supported activity types", () => {
    const { filterActivityItems, getActivityCounts, getUnreadActivityCounts } =
      loadTsModule("src/lib/activityItems.ts");

    const items = [
      { id: "mention:1", kind: "mention", isUnread: true },
      { id: "saved:1", kind: "saved" },
      { id: "thread:1", kind: "thread", isUnread: true },
      { id: "reaction:1", kind: "reaction", isUnread: true },
      { id: "reaction:2", kind: "reaction" },
    ];

    assert.deepEqual(getActivityCounts(items), {
      all: 5,
      mentions: 1,
      saved: 1,
      threads: 1,
      reactions: 2,
    });
    assert.deepEqual(
      filterActivityItems(items, "reactions").map((item) => item.id),
      ["reaction:1", "reaction:2"],
    );
    assert.deepEqual(
      filterActivityItems(items, "threads").map((item) => item.id),
      ["thread:1"],
    );
    assert.deepEqual(getUnreadActivityCounts(items), {
      all: 3,
      mentions: 1,
      saved: 0,
      threads: 1,
      reactions: 1,
    });
  });

  test("groups activity items into readable date sections", () => {
    const { getActivityDateGroupLabel, groupActivityItemsByDate } =
      loadTsModule("src/lib/activityItems.ts");
    const now = new Date(2026, 4, 4, 12);
    const items = [
      {
        id: "mention:today",
        kind: "mention",
        occurredAt: new Date(2026, 4, 4, 10).toISOString(),
      },
      {
        id: "saved:today",
        kind: "saved",
        occurredAt: new Date(2026, 4, 4, 9).toISOString(),
      },
      {
        id: "thread:yesterday",
        kind: "thread",
        occurredAt: new Date(2026, 4, 3, 9).toISOString(),
      },
      {
        id: "reaction:older",
        kind: "reaction",
        occurredAt: new Date(2026, 4, 1, 9).toISOString(),
      },
    ];

    assert.equal(
      getActivityDateGroupLabel(items[0].occurredAt, now),
      "Today",
    );
    assert.equal(
      getActivityDateGroupLabel(items[2].occurredAt, now),
      "Yesterday",
    );
    assert.equal(
      getActivityDateGroupLabel(items[3].occurredAt, now),
      "May 1, 2026",
    );

    const groups = groupActivityItemsByDate(items, now);

    assert.deepEqual(
      groups.map((group) => ({
        id: group.id,
        label: group.label,
        itemIds: group.items.map((item) => item.id),
      })),
      [
        {
          id: "2026-05-04",
          label: "Today",
          itemIds: ["mention:today", "saved:today"],
        },
        {
          id: "2026-05-03",
          label: "Yesterday",
          itemIds: ["thread:yesterday"],
        },
        {
          id: "2026-05-01",
          label: "May 1, 2026",
          itemIds: ["reaction:older"],
        },
      ],
    );
  });
});

describe("Activity read state API route", () => {
  test("marks activity as read for the signed-in user", async () => {
    const readAt = new Date("2026-05-04T16:30:00.000Z");
    let markedUserId;
    const { POST } = loadTsModule("src/app/api/activity/read/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("current-user"),
      "@/lib/activity": {
        async markActivityReadForUser(userId) {
          markedUserId = userId;
          return readAt;
        },
      },
    });

    const response = await POST();

    assert.equal(response.status, 200);
    assert.equal(markedUserId, "current-user");
    assert.deepEqual(await responseJson(response), {
      ok: true,
      readAt: "2026-05-04T16:30:00.000Z",
    });
  });

  test("rejects unauthenticated activity read updates", async () => {
    const { POST } = loadTsModule("src/app/api/activity/read/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(null),
      "@/lib/activity": {
        async markActivityReadForUser() {
          throw new Error("should not mark read");
        },
      },
    });

    const response = await POST();

    assert.equal(response.status, 401);
    assert.deepEqual(await responseJson(response), { error: "Unauthorized" });
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
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
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
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
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
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await POST(jsonRequest({ userId: "other-user" }));

    assert.equal(response.status, 201);
    assert.deepEqual(await responseJson(response), { channelId: "new-dm" });
    assert.equal(createArgs.data.type, "DM");
    assert.equal(createArgs.data.slug, "dm-current-user-other-user");
    assert.deepEqual(createArgs.data.members.create, [
      {
        userId: "current-user",
        role: "member",
        notificationLevel: "ALL",
        mutedAt: null,
      },
      {
        userId: "other-user",
        role: "member",
        notificationLevel: "ALL",
        mutedAt: null,
      },
    ]);
  });
});

describe("Channel management permissions", () => {
  test("normalizes manager roles and channel names", () => {
    const {
      canCreateWorkspaceChannels,
      canManageChannel,
      canRemoveChannelMemberRole,
      normalizeChannelDescription,
      normalizeChannelName,
      normalizeAssignableChannelRole,
      normalizeChannelVisibility,
    } = loadTsModule("src/lib/channelManagement.ts");

    assert.equal(canCreateWorkspaceChannels({ role: "ADMIN" }), true);
    assert.equal(canCreateWorkspaceChannels({ role: "USER" }), true);
    assert.equal(canCreateWorkspaceChannels(null), false);
    assert.equal(canManageChannel({ role: "USER" }, "owner"), true);
    assert.equal(canManageChannel({ role: "USER" }, "admin"), true);
    assert.equal(canManageChannel({ role: "USER" }, "member"), false);
    assert.equal(canRemoveChannelMemberRole("owner"), false);
    assert.equal(canRemoveChannelMemberRole("admin"), true);
    assert.equal(normalizeChannelName(" #Team Updates!! "), "team-updates");
    assert.equal(normalizeChannelDescription("  Planning channel  "), "Planning channel");
    assert.equal(normalizeChannelDescription("   "), null);
    assert.equal(normalizeAssignableChannelRole("ADMIN"), "admin");
    assert.equal(normalizeAssignableChannelRole("owner"), "member");
    assert.equal(normalizeChannelVisibility("PRIVATE"), "PRIVATE");
    assert.equal(normalizeChannelVisibility("GROUP_DM"), "PUBLIC");
  });
});

describe("Embedded navigation", () => {
  test("preserves embedded Messages context in local destinations", () => {
    const { withEmbedParam } = loadTsModule("src/lib/embeddedNavigation.ts");

    assert.equal(withEmbedParam("/channels", false), "/channels");
    assert.equal(
      withEmbedParam("/channels?create=true", true),
      "/channels?create=true&embed=true",
    );
    assert.equal(
      withEmbedParam("/channels?embed=true&create=true", true),
      "/channels?embed=true&create=true",
    );
    assert.equal(
      withEmbedParam("/dm/channel-1#latest", true),
      "/dm/channel-1?embed=true#latest",
    );
  });
});

describe("Workspace policies", () => {
  test("normalizes env-backed workspace defaults", () => {
    process.env.SOCIAL_DEFAULT_CHANNEL_VISIBILITY = "private";
    process.env.SOCIAL_DEFAULT_NOTIFICATION_LEVEL = "mentions";
    process.env.SOCIAL_PUBLIC_CHANNEL_JOIN_POLICY = "workspace-admins";
    const now = new Date("2026-05-04T10:00:00.000Z");
    const {
      canJoinPublicChannel,
      getDefaultChannelVisibility,
      getDefaultMembershipPreferences,
      getWorkspacePolicies,
    } = loadWorkspacePoliciesModule();

    assert.deepEqual(getWorkspacePolicies(), {
      defaultChannelVisibility: "PRIVATE",
      defaultNotificationLevel: "MENTIONS",
      publicChannelJoinPolicy: "WORKSPACE_ADMINS",
      privateChannelJoinPolicy: "INVITE_ONLY",
      channelCreationPolicy: "MEMBERS",
    });
    assert.equal(getDefaultChannelVisibility(undefined), "PRIVATE");
    assert.deepEqual(getDefaultMembershipPreferences(now), {
      notificationLevel: "MENTIONS",
      mutedAt: null,
    });
    assert.equal(canJoinPublicChannel({ role: "USER" }, null), false);
    assert.equal(canJoinPublicChannel({ role: "ADMIN" }, null), true);
    assert.equal(canJoinPublicChannel({ role: "USER" }, "member"), true);
  });

  test("uses muted timestamps for muted default memberships", () => {
    process.env.SOCIAL_DEFAULT_NOTIFICATION_LEVEL = "muted";
    const now = new Date("2026-05-04T11:00:00.000Z");
    const { getDefaultMembershipPreferences } = loadWorkspacePoliciesModule();

    assert.deepEqual(getDefaultMembershipPreferences(now), {
      notificationLevel: "MUTED",
      mutedAt: now,
    });
  });

  test("returns workspace policies for signed-in users", async () => {
    process.env.SOCIAL_DEFAULT_CHANNEL_VISIBILITY = "PRIVATE";
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const { GET } = loadTsModule("src/app/api/workspace/policies/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("admin-user", "ADMIN"),
      "@/lib/channelManagement": channelManagement,
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await GET();
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.equal(body.defaultChannelVisibility, "PRIVATE");
    assert.equal(body.defaultNotificationLevel, "ALL");
    assert.equal(body.channelCreationPolicy, "MEMBERS");
    assert.equal(body.canManage, true);
  });
});

describe("Channel API route", () => {
  const defaultChannelMocks = {
    DEFAULT_CHANNELS: [
      { slug: "announcements" },
      { slug: "general" },
      { slug: "help" },
    ],
    async ensureDefaultChannelsForUser() {},
  };

  test("creates channels for signed-in workspace members", async () => {
    let createArgs;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const { POST } = loadTsModule("src/app/api/channels/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("current-user", "USER"),
      "@/lib/prisma": {
        prisma: {
          channel: {
            async findUnique() {
              return null;
            },
            async create(args) {
              createArgs = args;
              return {
                id: "member-channel",
                name: args.data.name,
                slug: args.data.slug,
                description: args.data.description,
                type: args.data.type,
                iconEmoji: null,
                isDefault: false,
                isArchived: false,
                _count: { members: 1, messages: 0 },
              };
            },
          },
        },
      },
      "@/lib/defaultChannels": defaultChannelMocks,
      "@/lib/channelManagement": channelManagement,
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await POST(jsonRequest({ name: "team-updates" }));

    assert.equal(response.status, 201);
    assert.equal(createArgs.data.name, "team-updates");
    assert.deepEqual(createArgs.data.members.create, {
      userId: "current-user",
      role: "owner",
      notificationLevel: "ALL",
      mutedAt: null,
    });
    assert.deepEqual(await responseJson(response), {
      id: "member-channel",
      name: "team-updates",
      slug: "team-updates",
      description: null,
      type: "PUBLIC",
      iconEmoji: null,
      isDefault: false,
      isArchived: false,
      isMember: true,
      memberCount: 1,
      messageCount: 0,
      role: "owner",
      canCreate: true,
      canManage: true,
    });
  });

  test("creates a normalized channel for workspace admins", async () => {
    let createArgs;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const { POST } = loadTsModule("src/app/api/channels/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("admin-user", "ADMIN"),
      "@/lib/prisma": {
        prisma: {
          channel: {
            async findUnique({ where }) {
              assert.deepEqual(where, { slug: "team-updates" });
              return null;
            },
            async create(args) {
              createArgs = args;
              return {
                id: "channel-new",
                name: args.data.name,
                slug: args.data.slug,
                description: args.data.description,
                type: args.data.type,
                iconEmoji: null,
                isDefault: false,
                isArchived: false,
                _count: { members: 1, messages: 0 },
              };
            },
          },
        },
      },
      "@/lib/defaultChannels": defaultChannelMocks,
      "@/lib/channelManagement": channelManagement,
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await POST(jsonRequest({
      name: " #Team Updates!! ",
      description: "  Planning channel  ",
      type: "PRIVATE",
    }));

    assert.equal(response.status, 201);
    assert.equal(createArgs.data.name, "team-updates");
    assert.equal(createArgs.data.slug, "team-updates");
    assert.equal(createArgs.data.description, "Planning channel");
    assert.equal(createArgs.data.type, "PRIVATE");
    assert.deepEqual(createArgs.data.members.create, {
      userId: "admin-user",
      role: "owner",
      notificationLevel: "ALL",
      mutedAt: null,
    });
    assert.deepEqual(await responseJson(response), {
      id: "channel-new",
      name: "team-updates",
      slug: "team-updates",
      description: "Planning channel",
      type: "PRIVATE",
      iconEmoji: null,
      isDefault: false,
      isArchived: false,
      isMember: true,
      memberCount: 1,
      messageCount: 0,
      role: "owner",
      canCreate: true,
      canManage: true,
    });
  });

  test("uses the workspace default visibility when type is omitted", async () => {
    process.env.SOCIAL_DEFAULT_CHANNEL_VISIBILITY = "PRIVATE";
    let createArgs;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const { POST } = loadTsModule("src/app/api/channels/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("admin-user", "ADMIN"),
      "@/lib/prisma": {
        prisma: {
          channel: {
            async findUnique() {
              return null;
            },
            async create(args) {
              createArgs = args;
              return {
                id: "channel-private-default",
                name: args.data.name,
                slug: args.data.slug,
                description: args.data.description,
                type: args.data.type,
                iconEmoji: null,
                isDefault: false,
                isArchived: false,
                _count: { members: 1, messages: 0 },
              };
            },
          },
        },
      },
      "@/lib/defaultChannels": defaultChannelMocks,
      "@/lib/channelManagement": channelManagement,
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await POST(jsonRequest({ name: "Ops Planning" }));

    assert.equal(response.status, 201);
    assert.equal(createArgs.data.type, "PRIVATE");
  });

  test("allows channel owners to edit custom channels", async () => {
    let updateArgs;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const { PATCH } = loadTsModule("src/app/api/channels/[id]/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("owner-user", "USER"),
      "@/lib/prisma": {
        prisma: {
          channel: {
            async findUnique({ where }) {
              if (where.id === "channel-1") {
                return {
                  id: "channel-1",
                  name: "old-name",
                  slug: "old-name",
                  description: null,
                  type: "PUBLIC",
                  iconEmoji: null,
                  isArchived: false,
                  isDefault: false,
                  members: [{ role: "owner" }],
                  _count: { members: 3, messages: 9 },
                };
              }
              if (where.slug === "new-name") return null;
              return null;
            },
            async update(args) {
              updateArgs = args;
              return {
                id: "channel-1",
                name: args.data.name,
                slug: args.data.slug,
                description: args.data.description,
                type: args.data.type,
                iconEmoji: null,
                isDefault: false,
                isArchived: false,
                members: [{ role: "owner" }],
                _count: { members: 3, messages: 9 },
              };
            },
          },
        },
      },
      "@/lib/channelManagement": channelManagement,
    });

    const response = await PATCH(
      jsonRequest({ name: "new name", description: "Updated", type: "PRIVATE" }),
      { params: Promise.resolve({ id: "channel-1" }) },
    );

    assert.equal(response.status, 200);
    assert.deepEqual(updateArgs.data, {
      name: "new-name",
      slug: "new-name",
      description: "Updated",
      type: "PRIVATE",
    });
    assert.equal((await responseJson(response)).canManage, true);
  });

  test("blocks regular channel members from editing channels", async () => {
    let updateCalled = false;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const { PATCH } = loadTsModule("src/app/api/channels/[id]/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("member-user", "USER"),
      "@/lib/prisma": {
        prisma: {
          channel: {
            async findUnique() {
              return {
                id: "channel-1",
                type: "PUBLIC",
                isArchived: false,
                isDefault: false,
                members: [{ role: "member" }],
                _count: { members: 2, messages: 0 },
              };
            },
            async update() {
              updateCalled = true;
            },
          },
        },
      },
      "@/lib/channelManagement": channelManagement,
    });

    const response = await PATCH(
      jsonRequest({ name: "new name", type: "PUBLIC" }),
      { params: Promise.resolve({ id: "channel-1" }) },
    );

    assert.equal(response.status, 403);
    assert.deepEqual(await responseJson(response), {
      error: "Workspace admin access required",
    });
    assert.equal(updateCalled, false);
  });

  test("archives custom channels for channel managers", async () => {
    let updateArgs;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const { PATCH } = loadTsModule("src/app/api/channels/[id]/archive/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("owner-user", "USER"),
      "@/lib/prisma": {
        prisma: {
          channel: {
            async findUnique({ where }) {
              assert.deepEqual(where, { id: "channel-1" });
              return {
                id: "channel-1",
                name: "planning",
                slug: "planning",
                description: null,
                type: "PUBLIC",
                iconEmoji: null,
                isArchived: false,
                isDefault: false,
                members: [{ role: "owner" }],
                _count: { members: 3, messages: 4 },
              };
            },
            async update(args) {
              updateArgs = args;
              return {
                id: "channel-1",
                name: "planning",
                slug: "planning",
                description: null,
                type: "PUBLIC",
                iconEmoji: null,
                isArchived: args.data.isArchived,
                isDefault: false,
                members: [{ role: "owner" }],
                _count: { members: 3, messages: 4 },
              };
            },
          },
        },
      },
      "@/lib/channelManagement": channelManagement,
    });

    const response = await PATCH(
      jsonRequest({ archived: true }),
      { params: Promise.resolve({ id: "channel-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.deepEqual(updateArgs.data, { isArchived: true });
    assert.equal(body.isArchived, true);
    assert.equal(body.canManage, true);
  });

  test("restores archived channels for workspace admins", async () => {
    let updateArgs;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const { PATCH } = loadTsModule("src/app/api/channels/[id]/archive/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("admin-user", "ADMIN"),
      "@/lib/prisma": {
        prisma: {
          channel: {
            async findUnique() {
              return {
                id: "channel-1",
                name: "planning",
                slug: "planning",
                description: "Planning",
                type: "PRIVATE",
                iconEmoji: null,
                isArchived: true,
                isDefault: false,
                members: [],
                _count: { members: 2, messages: 7 },
              };
            },
            async update(args) {
              updateArgs = args;
              return {
                id: "channel-1",
                name: "planning",
                slug: "planning",
                description: "Planning",
                type: "PRIVATE",
                iconEmoji: null,
                isArchived: args.data.isArchived,
                isDefault: false,
                members: [],
                _count: { members: 2, messages: 7 },
              };
            },
          },
        },
      },
      "@/lib/channelManagement": channelManagement,
    });

    const response = await PATCH(
      jsonRequest({ archived: false }),
      { params: Promise.resolve({ id: "channel-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.deepEqual(updateArgs.data, { isArchived: false });
    assert.equal(body.isArchived, false);
    assert.equal(body.canCreate, true);
    assert.equal(body.canManage, true);
  });

  test("does not archive default channels", async () => {
    let updateCalled = false;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const { PATCH } = loadTsModule("src/app/api/channels/[id]/archive/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("admin-user", "ADMIN"),
      "@/lib/prisma": {
        prisma: {
          channel: {
            async findUnique() {
              return {
                id: "channel-general",
                type: "PUBLIC",
                isDefault: true,
                members: [{ role: "owner" }],
                _count: { members: 3, messages: 0 },
              };
            },
            async update() {
              updateCalled = true;
            },
          },
        },
      },
      "@/lib/channelManagement": channelManagement,
    });

    const response = await PATCH(
      jsonRequest({ archived: true }),
      { params: Promise.resolve({ id: "channel-general" }) },
    );

    assert.equal(response.status, 400);
    assert.deepEqual(await responseJson(response), {
      error: "Default channels cannot be archived",
    });
    assert.equal(updateCalled, false);
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
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await POST(jsonRequest({}), { params: Promise.resolve({ id: "channel-1" }) });

    assert.equal(response.status, 200);
    assert.deepEqual(await responseJson(response), {
      channelId: "channel-1",
      isMember: true,
      role: "member",
    });
    assert.equal(upsertArgs.create.role, "member");
    assert.equal(upsertArgs.create.notificationLevel, "ALL");
    assert.equal(upsertArgs.create.mutedAt, null);
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
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await POST(jsonRequest({}), { params: Promise.resolve({ id: "private-1" }) });

    assert.equal(response.status, 403);
    assert.deepEqual(await responseJson(response), {
      error: "Private channels require an invitation",
    });
    assert.equal(upsertCalled, false);
  });

  test("enforces workspace public channel join policy", async () => {
    process.env.SOCIAL_PUBLIC_CHANNEL_JOIN_POLICY = "WORKSPACE_ADMINS";
    let upsertCalled = false;
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
        async upsert() {
          upsertCalled = true;
          return { role: "member" };
        },
      },
    };
    const { POST } = loadTsModule("src/app/api/channels/[id]/membership/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("member-user", "USER"),
      "@/lib/prisma": { prisma },
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await POST(jsonRequest({}), { params: Promise.resolve({ id: "channel-1" }) });

    assert.equal(response.status, 403);
    assert.deepEqual(await responseJson(response), {
      error: "Workspace admin access required to join public channels",
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
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
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
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await DELETE(jsonRequest({}), { params: Promise.resolve({ id: "channel-general" }) });

    assert.equal(response.status, 400);
    assert.deepEqual(await responseJson(response), {
      error: "Default channels cannot be left",
    });
    assert.equal(deleteCalled, false);
  });
});

describe("Channel members API route", () => {
  const memberUser = (id, role, displayName = "Team Member") => ({
    id: `${id}-membership`,
    channelId: "channel-1",
    userId: id,
    role,
    joinedAt: new Date("2026-05-04T10:00:00.000Z"),
    user: {
      id,
      username: id,
      displayName,
      avatarUrl: null,
      isAgent: false,
      status: "offline",
    },
  });

  test("lists channel members for channel managers", async () => {
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const prisma = {
      channel: {
        async findUnique({ where, select }) {
          assert.deepEqual(where, { id: "channel-1" });
          assert.equal(select.members.where.userId, "owner-user");
          return {
            id: "channel-1",
            type: "PRIVATE",
            isArchived: false,
            members: [{ role: "owner" }],
          };
        },
      },
      channelMember: {
        async findMany({ where }) {
          assert.deepEqual(where, { channelId: "channel-1" });
          return [
            memberUser("member-user", "member", "Member User"),
            memberUser("owner-user", "owner", "Owner User"),
          ];
        },
      },
    };
    const { GET } = loadTsModule("src/app/api/channels/[id]/members/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("owner-user", "USER"),
      "@/lib/prisma": { prisma },
      "@/lib/channelManagement": channelManagement,
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await GET(jsonRequest({}), { params: Promise.resolve({ id: "channel-1" }) });
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.equal(body.canManage, true);
    assert.equal(body.members[0].role, "owner");
    assert.equal(body.members[1].role, "member");
    assert.equal(body.members[0].joinedAt, "2026-05-04T10:00:00.000Z");
  });

  test("adds teammates to a channel for managers", async () => {
    let createArgs;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const prisma = {
      channel: {
        async findUnique() {
          return {
            id: "channel-1",
            type: "PRIVATE",
            isArchived: false,
            members: [{ role: "admin" }],
          };
        },
      },
      user: {
        async findUnique({ where }) {
          assert.deepEqual(where, { id: "target-user" });
          return {
            id: "target-user",
            username: "target",
            displayName: "Target User",
            avatarUrl: null,
            isAgent: false,
            status: "offline",
          };
        },
      },
      channelMember: {
        async findUnique() {
          return null;
        },
        async create(args) {
          createArgs = args;
          return memberUser(args.data.userId, args.data.role, "Target User");
        },
      },
    };
    const { POST } = loadTsModule("src/app/api/channels/[id]/members/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("admin-user", "USER"),
      "@/lib/prisma": { prisma },
      "@/lib/channelManagement": channelManagement,
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await POST(
      jsonRequest({ userId: "target-user", role: "owner" }),
      { params: Promise.resolve({ id: "channel-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 201);
    assert.deepEqual(createArgs.data, {
      channelId: "channel-1",
      userId: "target-user",
      role: "member",
      notificationLevel: "ALL",
      mutedAt: null,
    });
    assert.equal(body.userId, "target-user");
    assert.equal(body.role, "member");
  });

  test("blocks regular members from adding teammates", async () => {
    let userLookupCalled = false;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const prisma = {
      channel: {
        async findUnique() {
          return {
            id: "channel-1",
            type: "PUBLIC",
            isArchived: false,
            members: [{ role: "member" }],
          };
        },
      },
      user: {
        async findUnique() {
          userLookupCalled = true;
        },
      },
    };
    const { POST } = loadTsModule("src/app/api/channels/[id]/members/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("member-user", "USER"),
      "@/lib/prisma": { prisma },
      "@/lib/channelManagement": channelManagement,
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    });

    const response = await POST(
      jsonRequest({ userId: "target-user" }),
      { params: Promise.resolve({ id: "channel-1" }) },
    );

    assert.equal(response.status, 403);
    assert.deepEqual(await responseJson(response), {
      error: "Channel admin access required",
    });
    assert.equal(userLookupCalled, false);
  });

  test("updates member roles for channel managers", async () => {
    let updateArgs;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const prisma = {
      channel: {
        async findUnique() {
          return {
            id: "channel-1",
            type: "PUBLIC",
            isArchived: false,
            isDefault: false,
            members: [{ role: "owner" }],
          };
        },
      },
      channelMember: {
        async findUnique() {
          return memberUser("target-user", "member", "Target User");
        },
        async update(args) {
          updateArgs = args;
          return memberUser("target-user", args.data.role, "Target User");
        },
      },
    };
    const { PATCH } = loadTsModule("src/app/api/channels/[id]/members/[userId]/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("owner-user", "USER"),
      "@/lib/prisma": { prisma },
      "@/lib/channelManagement": channelManagement,
    });

    const response = await PATCH(
      jsonRequest({ role: "admin" }),
      { params: Promise.resolve({ id: "channel-1", userId: "target-user" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.deepEqual(updateArgs.where, {
      channelId_userId: {
        channelId: "channel-1",
        userId: "target-user",
      },
    });
    assert.deepEqual(updateArgs.data, { role: "admin" });
    assert.equal(body.role, "admin");
  });

  test("does not remove channel owners", async () => {
    let deleteCalled = false;
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    const prisma = {
      channel: {
        async findUnique() {
          return {
            id: "channel-1",
            type: "PUBLIC",
            isArchived: false,
            isDefault: false,
            members: [{ role: "admin" }],
          };
        },
      },
      channelMember: {
        async findUnique() {
          return memberUser("owner-user", "owner", "Owner User");
        },
        async deleteMany() {
          deleteCalled = true;
        },
      },
    };
    const { DELETE } = loadTsModule("src/app/api/channels/[id]/members/[userId]/route.ts", {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("admin-user", "USER"),
      "@/lib/prisma": { prisma },
      "@/lib/channelManagement": channelManagement,
    });

    const response = await DELETE(
      jsonRequest({}),
      { params: Promise.resolve({ id: "channel-1", userId: "owner-user" }) },
    );

    assert.equal(response.status, 400);
    assert.deepEqual(await responseJson(response), {
      error: "Channel owner cannot be removed",
    });
    assert.equal(deleteCalled, false);
  });
});

describe("Channel messages API route", () => {
  function messageRecord(id, content, createdAt) {
    return {
      id,
      channelId: "channel-1",
      content,
      createdAt: new Date(createdAt),
      isEdited: false,
      isPinned: false,
      parentId: null,
      metadata: null,
      author: {
        id: `author-${id}`,
        username: `author-${id}`,
        displayName: `Author ${id}`,
        avatarUrl: null,
        isAgent: false,
      },
      reactions: [],
      savedItems: [],
      attachments: [],
      replies: [],
      _count: { replies: 0 },
    };
  }

  function messagesRouteMocks(prisma) {
    return {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock("current-user", "USER"),
      "@/lib/prisma": { prisma },
      "@/lib/nanobot": { async invokeAgentStreaming() {} },
      "@/lib/socketServer": { getIO() { return null; } },
      "@/lib/messageFormat": loadTsModule("src/lib/messageFormat.ts"),
    };
  }

  test("returns the latest message window in chronological display order", async () => {
    let findManyArgs;
    const prisma = {
      channelMember: {
        async findUnique() {
          return {
            role: "member",
            channel: { isArchived: false },
          };
        },
      },
      message: {
        async findMany(args) {
          findManyArgs = args;
          return [
            messageRecord("msg-3", "Newest", "2026-05-04T12:00:00.000Z"),
            messageRecord("msg-2", "Middle", "2026-05-04T11:00:00.000Z"),
            messageRecord("msg-1", "Oldest extra", "2026-05-04T10:00:00.000Z"),
          ];
        },
      },
    };
    const { GET } = loadTsModule(
      "src/app/api/channels/[id]/messages/route.ts",
      messagesRouteMocks(prisma),
    );

    const response = await GET(
      urlRequest("http://social.test/api/channels/channel-1/messages?limit=2"),
      { params: Promise.resolve({ id: "channel-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.deepEqual(findManyArgs.where, {
      channelId: "channel-1",
      deletedAt: null,
      parentId: null,
    });
    assert.deepEqual(findManyArgs.orderBy, { createdAt: "desc" });
    assert.equal(findManyArgs.take, 3);
    assert.equal(body.messages[0].id, "msg-2");
    assert.equal(body.messages[1].id, "msg-3");
    assert.equal(body.nextCursor, "msg-2");
  });

  test("uses the cursor to load older messages", async () => {
    let findManyArgs;
    const prisma = {
      channelMember: {
        async findUnique() {
          return {
            role: "member",
            channel: { isArchived: false },
          };
        },
      },
      message: {
        async findMany(args) {
          findManyArgs = args;
          return [
            messageRecord("msg-1", "Older", "2026-05-04T10:00:00.000Z"),
          ];
        },
      },
    };
    const { GET } = loadTsModule(
      "src/app/api/channels/[id]/messages/route.ts",
      messagesRouteMocks(prisma),
    );

    const response = await GET(
      urlRequest("http://social.test/api/channels/channel-1/messages?cursor=msg-2&limit=2"),
      { params: Promise.resolve({ id: "channel-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.deepEqual(findManyArgs.cursor, { id: "msg-2" });
    assert.equal(findManyArgs.skip, 1);
    assert.deepEqual(body.messages.map((message) => message.id), ["msg-1"]);
    assert.equal(body.nextCursor, null);
  });
});

describe("Email import API route", () => {
  function importedMessageRecord(args) {
    const createdAt = new Date("2026-05-04T12:00:00.000Z");
    return {
      id: "imported-message-1",
      channelId: args.data.channelId,
      authorId: args.data.authorId,
      content: args.data.content,
      parentId: null,
      isEdited: false,
      isPinned: false,
      metadata: args.data.metadata,
      createdAt,
      updatedAt: createdAt,
      deletedAt: null,
      author: {
        id: args.data.authorId,
        username: "current-user",
        displayName: "Current User",
        avatarUrl: null,
        isAgent: false,
      },
      attachments: [],
    };
  }

  function emailImportRouteMocks({ prisma, userId = "current-user", emitted = [] }) {
    return {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(userId, "USER"),
      "@/lib/prisma": { prisma },
      "@/lib/messageFormat": loadTsModule("src/lib/messageFormat.ts"),
      "@/lib/socketServer": {
        getIO() {
          return {
            to(room) {
              return {
                emit(event, payload) {
                  emitted.push({ room, event, payload });
                },
              };
            },
          };
        },
      },
      "@/lib/emailImport": loadTsModule("src/lib/emailImport.ts"),
      "@/lib/workspacePolicies": loadWorkspacePoliciesModule(),
    };
  }

  test("imports a visible email into a channel the user belongs to", async () => {
    const emitted = [];
    let membershipArgs;
    let createArgs;
    let updateArgs;
    const prisma = {
      channelMember: {
        async findUnique(args) {
          membershipArgs = args;
          return { channel: { isArchived: false } };
        },
      },
      message: {
        async create(args) {
          createArgs = args;
          return importedMessageRecord(args);
        },
      },
      channel: {
        async update(args) {
          updateArgs = args;
          return {};
        },
      },
    };
    const { POST } = loadTsModule(
      "src/app/api/email-import/route.ts",
      emailImportRouteMocks({ prisma, emitted }),
    );

    const response = await POST(jsonRequest({
      destination: { type: "channel", channelId: "channel-1" },
      email: {
        provider: "gmail",
        subject: "Grant update",
        from: { name: "Sender Name", email: "sender@example.org" },
        to: [{ email: "team@example.org" }],
        sentAt: "May 4, 2026, 9:30 AM",
        sourceUrl: "https://mail.google.com/mail/u/0/#inbox/message-1",
        bodyText: "Hello team,\n\nHere is the grant update.",
        bodyHtml: "<div>Hello team,<br><br>Here is the grant update.</div>",
        attachments: [{ name: "brief.pdf", url: "application/pdf:https://mail.google.test/brief:brief.pdf" }],
      },
    }));
    const body = await responseJson(response);

    assert.equal(response.status, 201);
    assert.deepEqual(membershipArgs.where, {
      channelId_userId: { channelId: "channel-1", userId: "current-user" },
    });
    assert.equal(createArgs.data.channelId, "channel-1");
    assert.equal(createArgs.data.authorId, "current-user");
    assert.match(createArgs.data.content, /Imported email: Grant update/);
    assert.match(createArgs.data.content, /From: Sender Name <sender@example.org>/);
    assert.match(createArgs.data.content, /Here is the grant update/);
	    assert.equal(createArgs.data.metadata.type, "email_import");
	    assert.equal(createArgs.data.metadata.email.provider, "gmail");
	    assert.equal(createArgs.data.metadata.email.subject, "Grant update");
	    assert.match(createArgs.data.metadata.email.bodyText, /Here is the grant update/);
	    assert.match(createArgs.data.metadata.email.bodyHtml, /Hello team/);
    assert.equal(createArgs.data.metadata.email.attachments[0].name, "brief.pdf");
    assert.equal(createArgs.data.metadata.email.bodyTruncated, false);
    assert.equal(createArgs.data.metadata.email.htmlTruncated, false);
    assert.deepEqual(updateArgs.where, { id: "channel-1" });
    assert.equal(body.channelId, "channel-1");
    assert.equal(body.message.id, "imported-message-1");
    assert.deepEqual(emitted.map(({ room, event }) => [room, event]), [
      ["channel:channel-1", "message:new"],
    ]);
	  });

	  test("auto-joins a visible public channel before importing into it", async () => {
	    let upsertArgs;
	    let messageCreateArgs;
	    const prisma = {
	      channelMember: {
	        async findUnique() {
	          return null;
	        },
	        async upsert(args) {
	          upsertArgs = args;
	          return { role: "member" };
	        },
	      },
	      channel: {
	        async findUnique() {
	          return { id: "public-channel-1", type: "PUBLIC", isArchived: false };
	        },
	        async update() {
	          return {};
	        },
	      },
	      message: {
	        async create(args) {
	          messageCreateArgs = args;
	          return importedMessageRecord(args);
	        },
	      },
	    };
	    const { POST } = loadTsModule(
	      "src/app/api/email-import/route.ts",
	      emailImportRouteMocks({ prisma }),
	    );

	    const response = await POST(jsonRequest({
	      destination: { type: "channel", channelId: "public-channel-1" },
	      email: {
	        subject: "Public channel follow-up",
	        bodyText: "Send this to the picked channel.",
	      },
	    }));
	    const body = await responseJson(response);

	    assert.equal(response.status, 201);
	    assert.deepEqual(upsertArgs.where, {
	      channelId_userId: {
	        channelId: "public-channel-1",
	        userId: "current-user",
	      },
	    });
	    assert.equal(upsertArgs.create.channelId, "public-channel-1");
	    assert.equal(upsertArgs.create.userId, "current-user");
	    assert.equal(messageCreateArgs.data.channelId, "public-channel-1");
	    assert.equal(body.channelId, "public-channel-1");
	  });

	  test("creates a DM destination when importing to a user with no existing DM", async () => {
    let createdChannelArgs;
    let messageCreateArgs;
    const prisma = {
      user: {
        async findUnique(args) {
          return { id: args.where.id };
        },
      },
      channel: {
        async findFirst() {
          return null;
        },
        async create(args) {
          createdChannelArgs = args;
          return { id: "dm-1" };
        },
        async update() {
          return {};
        },
      },
      message: {
        async create(args) {
          messageCreateArgs = args;
          return importedMessageRecord(args);
        },
      },
    };
    const { POST } = loadTsModule(
      "src/app/api/email-import/route.ts",
      emailImportRouteMocks({ prisma }),
    );

    const response = await POST(jsonRequest({
      destination: { type: "dm", userId: "agent-user" },
      email: {
        subject: "Follow up",
        bodyText: "Can you turn this into next steps?",
      },
    }));
    const body = await responseJson(response);

    assert.equal(response.status, 201);
    assert.equal(createdChannelArgs.data.type, "DM");
    assert.deepEqual(
      createdChannelArgs.data.members.create.map((member) => member.userId),
      ["current-user", "agent-user"],
    );
    assert.equal(messageCreateArgs.data.channelId, "dm-1");
    assert.equal(body.channelId, "dm-1");
  });

  test("rejects malformed email import payloads", async () => {
    const { POST } = loadTsModule(
      "src/app/api/email-import/route.ts",
      emailImportRouteMocks({ prisma: {} }),
    );

    const response = await POST(jsonRequest({
      destination: { type: "channel", channelId: "" },
      email: { subject: "Missing body shape" },
    }));
    const body = await responseJson(response);

    assert.equal(response.status, 400);
    assert.equal(body.error, "Invalid email import payload");
    assert.ok(Array.isArray(body.details));
  });
});

describe("Email reply API route", () => {
  function replyMessageRecord(args) {
    const createdAt = new Date("2026-05-04T12:05:00.000Z");
    return {
      id: "email-reply-message-1",
      channelId: args.data.channelId,
      authorId: args.data.authorId,
      content: args.data.content,
      parentId: args.data.parentId,
      isEdited: false,
      isPinned: false,
      metadata: args.data.metadata,
      createdAt,
      updatedAt: createdAt,
      deletedAt: null,
      author: {
        id: args.data.authorId,
        username: "current-user",
        displayName: "Current User",
        avatarUrl: null,
        isAgent: false,
      },
      attachments: [],
    };
  }

  function nodemailerMock(sentMail = [], transportConfigs = []) {
    return {
      createTransport(config) {
        transportConfigs.push(config);
        return {
          async sendMail(mail) {
            sentMail.push(mail);
            return { messageId: "<sent-reply@example.org>" };
          },
        };
      },
    };
  }

  function emailReplyRouteMocks({
    prisma,
    userId = "current-user",
    emitted = [],
    mailer = nodemailerMock(),
    fsMock = { existsSync: () => false, readFileSync: () => "" },
  }) {
    const emailReplyModule = loadTsModule("src/lib/emailReply.ts", {
      nodemailer: mailer,
      "node:fs": fsMock,
    });

    return {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(userId, "USER"),
      "@/lib/prisma": { prisma },
      "@/lib/messageFormat": loadTsModule("src/lib/messageFormat.ts"),
      "@/lib/socketServer": {
        getIO() {
          return {
            to(room) {
              return {
                emit(event, payload) {
                  emitted.push({ room, event, payload });
                },
              };
            },
          };
        },
      },
      "@/lib/emailReply": emailReplyModule,
    };
  }

  function importedEmailRecord(overrides = {}) {
    return {
      id: "imported-message-1",
      channelId: "channel-1",
      authorId: "current-user",
      deletedAt: null,
      metadata: {
        type: "email_import",
        email: {
	          provider: "gmail",
	          subject: "Grant update",
	          from: { name: "Sender Name", email: "sender@example.org" },
	          sentAt: "May 4, 2026, 9:30 AM",
	          messageId: "original@example.org",
	          sourceUrl: "https://mail.google.com/mail/u/0/#inbox/message-1",
	          bodyText: "Hello team,\n\nHere is the grant update.",
	        },
	      },
      channel: { isArchived: false },
      ...overrides,
    };
  }

  test("sends an SMTP reply and records it as a thread reply", async () => {
    process.env.EMAIL_REPLY_SMTP_HOST = "smtp.test";
    process.env.EMAIL_REPLY_SMTP_PORT = "2525";
    process.env.EMAIL_REPLY_SMTP_USERNAME = "nanobot@example.org";
    process.env.EMAIL_REPLY_SMTP_PASSWORD = "app-password";
    process.env.EMAIL_REPLY_FROM = "Nanobot <nanobot@example.org>";
    process.env.EMAIL_REPLY_CONSENT_GRANTED = "true";

    const emitted = [];
    const sentMail = [];
    const transportConfigs = [];
    const mailer = nodemailerMock(sentMail, transportConfigs);
    let createArgs;
    let updateArgs;
    const prisma = {
      message: {
        async findUnique() {
          return importedEmailRecord();
        },
        async create(args) {
          createArgs = args;
          return replyMessageRecord(args);
        },
      },
      channelMember: {
        async findUnique() {
          return { role: "member" };
        },
      },
      channel: {
        async update(args) {
          updateArgs = args;
          return {};
        },
      },
    };
    const { POST } = loadTsModule(
      "src/app/api/email-import/[messageId]/reply/route.ts",
      emailReplyRouteMocks({ prisma, emitted, mailer }),
    );

    const response = await POST(
      jsonRequest({ content: "Thanks, we will review this today." }),
      { params: Promise.resolve({ messageId: "imported-message-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 201);
    assert.equal(transportConfigs[0].host, "smtp.test");
    assert.equal(transportConfigs[0].port, 2525);
    assert.equal(sentMail[0].to, "\"Sender Name\" <sender@example.org>");
	    assert.equal(sentMail[0].subject, "Re: Grant update");
	    assert.match(sentMail[0].text, /^Thanks, we will review this today\./);
	    assert.match(sentMail[0].text, /On May 4, 2026, 9:30 AM, "Sender Name" <sender@example.org> wrote:/);
	    assert.match(sentMail[0].text, /> Here is the grant update\./);
	    assert.match(sentMail[0].html, /<div dir="ltr">Thanks, we will review this today\.<\/div>/);
	    assert.match(sentMail[0].html, /gmail_quote/);
	    assert.match(sentMail[0].html, /Here is the grant update\./);
	    assert.equal(sentMail[0].inReplyTo, "<original@example.org>");
    assert.equal(createArgs.data.channelId, "channel-1");
    assert.equal(createArgs.data.parentId, "imported-message-1");
    assert.equal(createArgs.data.metadata.type, "email_reply");
    assert.equal(createArgs.data.metadata.emailReply.to.email, "sender@example.org");
    assert.equal(createArgs.data.metadata.emailReply.subject, "Re: Grant update");
    assert.deepEqual(updateArgs.where, { id: "channel-1" });
    assert.equal(body.message.id, "email-reply-message-1");
    assert.equal(body.message.parentId, "imported-message-1");
    assert.deepEqual(emitted.map(({ room, event }) => [room, event]), [
      ["channel:channel-1", "message:new"],
    ]);
  });

  test("rejects email replies when SMTP is not configured", async () => {
    const prisma = {
      message: {
        async findUnique() {
          return importedEmailRecord();
        },
      },
      channelMember: {
        async findUnique() {
          return { role: "member" };
        },
      },
    };
    const { POST } = loadTsModule(
      "src/app/api/email-import/[messageId]/reply/route.ts",
      emailReplyRouteMocks({ prisma }),
    );

    const response = await POST(
      jsonRequest({ content: "Can you send the deck?" }),
      { params: Promise.resolve({ messageId: "imported-message-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 503);
    assert.equal(body.error, "SMTP email replies are not configured.");
  });

  test("rejects imported emails without a replyable sender", async () => {
    const prisma = {
      message: {
        async findUnique() {
          return importedEmailRecord({
            metadata: {
              type: "email_import",
              email: { subject: "No sender" },
            },
          });
        },
      },
      channelMember: {
        async findUnique() {
          return { role: "member" };
        },
      },
    };
    const { POST } = loadTsModule(
      "src/app/api/email-import/[messageId]/reply/route.ts",
      emailReplyRouteMocks({ prisma }),
    );

    const response = await POST(
      jsonRequest({ content: "Following up." }),
      { params: Promise.resolve({ messageId: "imported-message-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 400);
    assert.equal(body.error, "Imported email does not have a replyable sender");
  });
});

describe("Message moderation API route", () => {
  function socketMock(emitted) {
    return {
      getIO() {
        return {
          to(room) {
            return {
              emit(event, payload) {
                emitted.push({ room, event, payload });
              },
            };
          },
        };
      },
    };
  }

  function messageRouteMocks({
    prisma,
    userId = "current-user",
    role = "USER",
    emitted = [],
  }) {
    return {
      "next/server": nextServerMock,
      "@/lib/session": createAuthMock(userId, role),
      "@/lib/prisma": { prisma },
      "@/lib/socketServer": socketMock(emitted),
      "@/lib/messageModeration": loadMessageModerationModule(),
    };
  }

  test("records an audit entry when authors remove their own messages", async () => {
    const emitted = [];
    let updateArgs;
    const prisma = {
      message: {
        async findUnique() {
          return {
            id: "message-1",
            channelId: "channel-1",
            authorId: "current-user",
            deletedAt: null,
            metadata: { source: "voice" },
            channel: {
              isArchived: false,
              members: [{ role: "member" }],
            },
          };
        },
        async update(args) {
          updateArgs = args;
          return { id: "message-1" };
        },
      },
    };
    const { DELETE } = loadTsModule(
      "src/app/api/channels/[id]/messages/[messageId]/route.ts",
      messageRouteMocks({ prisma, emitted }),
    );

    const response = await DELETE(
      jsonRequest({}),
      { params: Promise.resolve({ id: "channel-1", messageId: "message-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.deepEqual(updateArgs.where, { id: "message-1" });
    assert.ok(updateArgs.data.deletedAt instanceof Date);
    assert.equal(updateArgs.data.metadata.source, "voice");
    assert.equal(updateArgs.data.metadata.deletionAudit.actorId, "current-user");
    assert.equal(updateArgs.data.metadata.deletionAudit.actorName, "Current User");
    assert.equal(updateArgs.data.metadata.deletionAudit.mode, "author");
    assert.equal(updateArgs.data.metadata.deletionAudit.reason, null);
    assert.equal(body.audit.mode, "author");
    assert.deepEqual(emitted, [
      {
        room: "channel:channel-1",
        event: "message:delete",
        payload: {
          id: "message-1",
          channelId: "channel-1",
          deletedById: "current-user",
          mode: "author",
        },
      },
    ]);
  });

  test("allows channel managers to remove messages with a reason", async () => {
    let updateArgs;
    const prisma = {
      message: {
        async findUnique() {
          return {
            id: "message-2",
            channelId: "channel-1",
            authorId: "author-user",
            deletedAt: null,
            metadata: null,
            channel: {
              isArchived: false,
              members: [{ role: "admin" }],
            },
          };
        },
        async update(args) {
          updateArgs = args;
          return { id: "message-2" };
        },
      },
    };
    const { DELETE } = loadTsModule(
      "src/app/api/channels/[id]/messages/[messageId]/route.ts",
      messageRouteMocks({ prisma, userId: "manager-user" }),
    );

    const response = await DELETE(
      jsonRequest({ reason: "  Off topic  " }),
      { params: Promise.resolve({ id: "channel-1", messageId: "message-2" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.equal(updateArgs.data.metadata.deletionAudit.actorId, "manager-user");
    assert.equal(updateArgs.data.metadata.deletionAudit.mode, "moderator");
    assert.equal(updateArgs.data.metadata.deletionAudit.reason, "Off topic");
    assert.equal(body.audit.reason, "Off topic");
  });

  test("blocks regular members from removing other teammates' messages", async () => {
    let updateCalled = false;
    const prisma = {
      message: {
        async findUnique() {
          return {
            id: "message-3",
            channelId: "channel-1",
            authorId: "author-user",
            deletedAt: null,
            metadata: null,
            channel: {
              isArchived: false,
              members: [{ role: "member" }],
            },
          };
        },
        async update() {
          updateCalled = true;
        },
      },
    };
    const { DELETE } = loadTsModule(
      "src/app/api/channels/[id]/messages/[messageId]/route.ts",
      messageRouteMocks({ prisma, userId: "member-user" }),
    );

    const response = await DELETE(
      jsonRequest({ reason: "not allowed" }),
      { params: Promise.resolve({ id: "channel-1", messageId: "message-3" }) },
    );

    assert.equal(response.status, 403);
    assert.deepEqual(await responseJson(response), {
      error: "Channel admin access required",
    });
    assert.equal(updateCalled, false);
  });

  test("lists removed messages for channel managers", async () => {
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    let findManyArgs;
    const prisma = {
      channel: {
        async findUnique() {
          return { id: "channel-1", members: [{ role: "owner" }] };
        },
      },
      message: {
        async findMany(args) {
          findManyArgs = args;
          return [
            {
              id: "message-4",
              channelId: "channel-1",
              content: "Removed content",
              createdAt: new Date("2026-05-04T10:00:00.000Z"),
              deletedAt: new Date("2026-05-04T11:00:00.000Z"),
              metadata: {
                deletionAudit: {
                  actorId: "manager-user",
                  actorName: "Manager User",
                  mode: "moderator",
                  reason: "Cleanup",
                  deletedAt: "2026-05-04T11:00:00.000Z",
                },
              },
              author: {
                id: "author-user",
                username: "author",
                displayName: "Author User",
                avatarUrl: null,
                isAgent: false,
              },
            },
          ];
        },
      },
    };
    const { GET } = loadTsModule(
      "src/app/api/channels/[id]/messages/deleted/route.ts",
      {
        "next/server": nextServerMock,
        "@/lib/session": createAuthMock("owner-user", "USER"),
        "@/lib/prisma": { prisma },
        "@/lib/channelManagement": channelManagement,
        "@/lib/messageModeration": loadMessageModerationModule(),
      },
    );

    const response = await GET(
      urlRequest("http://social.test/api/channels/channel-1/messages/deleted?limit=2"),
      { params: Promise.resolve({ id: "channel-1" }) },
    );
    const body = await responseJson(response);

    assert.equal(response.status, 200);
    assert.deepEqual(findManyArgs.where, {
      channelId: "channel-1",
      deletedAt: { not: null },
    });
    assert.deepEqual(findManyArgs.orderBy, { deletedAt: "desc" });
    assert.equal(findManyArgs.take, 2);
    assert.equal(body.deletedMessages[0].id, "message-4");
    assert.equal(body.deletedMessages[0].removedBy.displayName, "Manager User");
    assert.equal(body.deletedMessages[0].reason, "Cleanup");
    assert.equal(body.deletedMessages[0].removalMode, "moderator");
  });

  test("blocks regular members from the removed-message audit", async () => {
    const channelManagement = loadTsModule("src/lib/channelManagement.ts");
    let findManyCalled = false;
    const prisma = {
      channel: {
        async findUnique() {
          return { id: "channel-1", members: [{ role: "member" }] };
        },
      },
      message: {
        async findMany() {
          findManyCalled = true;
          return [];
        },
      },
    };
    const { GET } = loadTsModule(
      "src/app/api/channels/[id]/messages/deleted/route.ts",
      {
        "next/server": nextServerMock,
        "@/lib/session": createAuthMock("member-user", "USER"),
        "@/lib/prisma": { prisma },
        "@/lib/channelManagement": channelManagement,
        "@/lib/messageModeration": loadMessageModerationModule(),
      },
    );

    const response = await GET(
      urlRequest("http://social.test/api/channels/channel-1/messages/deleted"),
      { params: Promise.resolve({ id: "channel-1" }) },
    );

    assert.equal(response.status, 403);
    assert.deepEqual(await responseJson(response), {
      error: "Channel admin access required",
    });
    assert.equal(findManyCalled, false);
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
          role: "ADMIN",
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
    assert.equal(session.user.role, "ADMIN");
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
