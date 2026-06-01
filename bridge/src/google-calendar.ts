import { randomBytes } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { dirname } from "node:path";
import { IncomingMessage, ServerResponse } from "node:http";

const CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.readonly";
const EMAIL_SCOPE = "https://www.googleapis.com/auth/userinfo.email";
const OPENID_SCOPE = "openid";

const AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth";
const TOKEN_URL = "https://oauth2.googleapis.com/token";
const USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo";
const CALENDAR_API = "https://www.googleapis.com/calendar/v3";

const pendingStates = new Set<string>();

interface GoogleClientConfig {
  client_id: string;
  client_secret: string;
  redirect_uri: string;
}

interface GoogleToken {
  access_token?: string;
  refresh_token?: string;
  scope?: string;
  token_type?: string;
  id_token?: string;
  expiry_date?: number;
  expires_in?: number;
}

interface GoogleEventDate {
  date?: string;
  dateTime?: string;
  timeZone?: string;
}

interface GoogleEvent {
  id?: string;
  summary?: string;
  description?: string;
  location?: string;
  htmlLink?: string;
  status?: string;
  start?: GoogleEventDate;
  end?: GoogleEventDate;
  attendees?: Array<{ email?: string; responseStatus?: string }>;
  organizer?: { email?: string; displayName?: string };
}

type LocalSummaryProvider = () => Promise<unknown>;

function jsonResp(res: ServerResponse, status: number, data: unknown) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
  });
  res.end(JSON.stringify(data));
}

function htmlResp(res: ServerResponse, status: number, html: string) {
  res.writeHead(status, {
    "Content-Type": "text/html; charset=utf-8",
    "Cache-Control": "no-store",
  });
  res.end(html);
}

function redirectResp(res: ServerResponse, location: string) {
  res.writeHead(302, {
    Location: location,
    "Cache-Control": "no-store",
  });
  res.end();
}

function wantsHtml(req: IncomingMessage, url: URL): boolean {
  if (url.searchParams.get("format") === "json") return false;
  const accept = String(req.headers.accept || "");
  return accept.includes("text/html");
}

function tokenPath(): string {
  return (
    process.env.GOOGLE_CALENDAR_TOKEN_PATH ||
    "/root/.nanobot/google-calendar-token.json"
  );
}

function credentialPath(): string {
  return (
    process.env.GOOGLE_OAUTH_CREDENTIALS ||
    process.env.GOOGLE_CALENDAR_CREDENTIALS ||
    "/root/.nanobot/gcp-oauth.keys.json"
  );
}

function getRedirectUri(req: IncomingMessage): string {
  if (process.env.GOOGLE_CALENDAR_REDIRECT_URI) {
    return process.env.GOOGLE_CALENDAR_REDIRECT_URI;
  }

  const host =
    String(req.headers["x-forwarded-host"] || req.headers.host || "").trim() ||
    "localhost:3050";
  const proto =
    String(req.headers["x-forwarded-proto"] || "").trim() ||
    (host.startsWith("localhost") || host.startsWith("127.0.0.1")
      ? "http"
      : "https");

  return `${proto}://${host}/calendar/oauth/callback`;
}

function readJson<T>(path: string): T | null {
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, "utf-8")) as T;
  } catch {
    return null;
  }
}

function writeToken(token: GoogleToken) {
  const path = tokenPath();
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(token, null, 2), { mode: 0o600 });
}

function loadClientConfig(req: IncomingMessage): GoogleClientConfig | null {
  const envClientId = process.env.GOOGLE_CLIENT_ID;
  const envClientSecret = process.env.GOOGLE_CLIENT_SECRET;
  if (envClientId && envClientSecret) {
    return {
      client_id: envClientId,
      client_secret: envClientSecret,
      redirect_uri: getRedirectUri(req),
    };
  }

  const raw = readJson<Record<string, any>>(credentialPath());
  const config = raw?.installed || raw?.web || raw;
  if (!config?.client_id || !config?.client_secret) return null;

  return {
    client_id: config.client_id,
    client_secret: config.client_secret,
    redirect_uri: getRedirectUri(req),
  };
}

function getToken(): GoogleToken | null {
  return readJson<GoogleToken>(tokenPath());
}

function authConfigured(req: IncomingMessage) {
  const config = loadClientConfig(req);
  return {
    configured: !!config,
    credentialsPath: credentialPath(),
    redirectUri: getRedirectUri(req),
  };
}

function makeAuthUrl(client: GoogleClientConfig): string {
  const state = randomBytes(24).toString("hex");
  pendingStates.add(state);
  setTimeout(() => pendingStates.delete(state), 10 * 60 * 1000).unref();

  const params = new URLSearchParams({
    client_id: client.client_id,
    redirect_uri: client.redirect_uri,
    response_type: "code",
    scope: [OPENID_SCOPE, EMAIL_SCOPE, CALENDAR_SCOPE].join(" "),
    access_type: "offline",
    include_granted_scopes: "true",
    prompt: "consent",
    state,
  });

  return `${AUTH_URL}?${params.toString()}`;
}

async function postToken(params: URLSearchParams): Promise<GoogleToken> {
  const resp = await fetch(TOKEN_URL, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: params.toString(),
  });

  const data = (await resp.json().catch(() => ({}))) as any;
  if (!resp.ok) {
    const detail = data.error_description || data.error || "token request failed";
    throw new Error(detail);
  }

  return data as GoogleToken;
}

async function exchangeCode(
  client: GoogleClientConfig,
  code: string
): Promise<GoogleToken> {
  const token = await postToken(
    new URLSearchParams({
      code,
      client_id: client.client_id,
      client_secret: client.client_secret,
      redirect_uri: client.redirect_uri,
      grant_type: "authorization_code",
    })
  );

  token.expiry_date = Date.now() + Number(token.expires_in || 3600) * 1000;
  writeToken(token);
  return token;
}

async function refreshToken(
  client: GoogleClientConfig,
  token: GoogleToken
): Promise<GoogleToken> {
  if (!token.refresh_token) {
    throw new Error("Google Calendar needs to be reconnected.");
  }

  const refreshed = await postToken(
    new URLSearchParams({
      refresh_token: token.refresh_token,
      client_id: client.client_id,
      client_secret: client.client_secret,
      grant_type: "refresh_token",
    })
  );

  const merged = {
    ...token,
    ...refreshed,
    refresh_token: refreshed.refresh_token || token.refresh_token,
    expiry_date: Date.now() + Number(refreshed.expires_in || 3600) * 1000,
  };
  writeToken(merged);
  return merged;
}

async function validToken(client: GoogleClientConfig): Promise<GoogleToken | null> {
  const token = getToken();
  if (!token?.access_token) return null;

  const expiresAt = Number(token.expiry_date || 0);
  if (expiresAt && expiresAt > Date.now() + 60_000) return token;

  return refreshToken(client, token);
}

async function googleGet<T>(pathOrUrl: string, token: GoogleToken): Promise<T> {
  const url = pathOrUrl.startsWith("https://")
    ? pathOrUrl
    : `${CALENDAR_API}${pathOrUrl}`;
  const resp = await fetch(url, {
    headers: { Authorization: `Bearer ${token.access_token}` },
  });
  const data = (await resp.json().catch(() => ({}))) as any;
  if (!resp.ok) {
    const detail = data.error?.message || data.error_description || "Google API request failed";
    throw new Error(detail);
  }
  return data as T;
}

function addDays(date: Date, days: number): Date {
  const next = new Date(date);
  next.setDate(next.getDate() + days);
  return next;
}

function normalizeEvent(event: GoogleEvent) {
  return {
    id: event.id,
    title: event.summary || "(No title)",
    location: event.location || "",
    htmlLink: event.htmlLink || "",
    status: event.status || "",
    start: event.start?.dateTime || event.start?.date || null,
    end: event.end?.dateTime || event.end?.date || null,
    allDay: !!event.start?.date,
    timeZone: event.start?.timeZone || event.end?.timeZone || "",
    organizer: event.organizer?.displayName || event.organizer?.email || "",
    attendeeCount: Array.isArray(event.attendees) ? event.attendees.length : 0,
  };
}

async function getGoogleCalendarData(
  req: IncomingMessage,
  daysAhead: number,
  range?: { timeMin: string; timeMax: string }
) {
  const setup = authConfigured(req);
  const client = loadClientConfig(req);
  if (!client) {
    return {
      connected: false,
      configured: false,
      connectUrl: null,
      redirectUri: setup.redirectUri,
      credentialsPath: setup.credentialsPath,
      error: "Google OAuth credentials were not found.",
      events: [],
    };
  }

  let token: GoogleToken | null = null;
  try {
    token = await validToken(client);
  } catch (err: any) {
    return {
      connected: false,
      configured: true,
      connectUrl: "/calendar/connect",
      redirectUri: client.redirect_uri,
      error: err.message || "Google Calendar needs to be reconnected.",
      events: [],
    };
  }

  if (!token?.access_token) {
    return {
      connected: false,
      configured: true,
      connectUrl: "/calendar/connect",
      redirectUri: client.redirect_uri,
      error: null,
      events: [],
    };
  }

  const now = new Date();
  const timeMin = range?.timeMin || now.toISOString();
  const timeMax = range?.timeMax || addDays(now, daysAhead).toISOString();
  const params = new URLSearchParams({
    timeMin,
    timeMax,
    maxResults: "80",
    singleEvents: "true",
    orderBy: "startTime",
  });

  const [userinfo, eventsResp, calendarsResp] = await Promise.all([
    googleGet<{ email?: string; name?: string }>(USERINFO_URL, token).catch(
      () => ({} as { email?: string; name?: string })
    ),
    googleGet<{ items?: GoogleEvent[] }>(
      `/calendars/primary/events?${params.toString()}`,
      token
    ),
    googleGet<{ items?: Array<{ id?: string; summary?: string; primary?: boolean }> }>(
      "/users/me/calendarList?minAccessRole=reader",
      token
    ).catch(() => ({ items: [] })),
  ]);

  return {
    connected: true,
    configured: true,
    connectUrl: "/calendar/connect",
    disconnectUrl: "/calendar/disconnect",
    redirectUri: client.redirect_uri,
    account: {
      email: userinfo.email || "",
      name: userinfo.name || "",
    },
    calendars: (calendarsResp.items || []).map((calendar) => ({
      id: calendar.id,
      summary: calendar.summary,
      primary: !!calendar.primary,
    })),
    events: (eventsResp.items || []).map(normalizeEvent),
    range: { timeMin, timeMax, daysAhead },
    syncedAt: new Date().toISOString(),
  };
}

const LOCAL_CALENDARS = [
  {
    id: "cal-personal",
    user_id: "demo-user",
    name: "Personal",
    color: "#FFD700",
    visibility: "private",
    is_default: true,
    is_external: false,
    course_id: null,
    description: null,
  },
  {
    id: "cal-community",
    user_id: "demo-user",
    name: "Community Events",
    color: "#22c55e",
    visibility: "shared",
    is_default: false,
    is_external: false,
    course_id: null,
    description: null,
  },
  {
    id: "cal-work",
    user_id: "demo-user",
    name: "Work",
    color: "#3b82f6",
    visibility: "private",
    is_default: false,
    is_external: false,
    course_id: null,
    description: null,
  },
];

function googleCalendarColor(index: number): string {
  const colors = ["#0f766e", "#8b5cf6", "#ef4444", "#0891b2", "#f59e0b"];
  return colors[index % colors.length];
}

function toStreetVoicesCalendar(calendar: any, index: number, userId: string) {
  return {
    id: `google:${calendar.id}`,
    user_id: userId,
    name: calendar.summary || calendar.id || "Google Calendar",
    color: googleCalendarColor(index),
    visibility: "private",
    is_default: false,
    is_external: true,
    external_provider: "google",
    external_id: calendar.id,
    course_id: null,
    description: null,
  };
}

function toStreetVoicesEvent(event: any, calendarId: string, color: string) {
  return {
    id: `google:${event.id}`,
    user_id: "demo-user",
    calendar_id: calendarId,
    title: event.title || "(No title)",
    description: event.htmlLink || "",
    start_at: event.start,
    end_at: event.end || event.start,
    all_day: !!event.allDay,
    color,
    location: event.location || "",
    rrule: null,
    status: event.status || "confirmed",
    external_provider: "google",
    external_id: event.id,
    html_link: event.htmlLink || "",
  };
}

async function handleOriginalCalendarApi(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL
): Promise<boolean> {
  if (url.pathname === "/api/calendar/calendars") {
    if (req.method !== "GET") return false;

    const userId = url.searchParams.get("user_id") || "demo-user";
    const google = await getGoogleCalendarData(req, 30);
    const googleCalendars = google.connected
      ? (google.calendars || []).map((calendar: any, index: number) =>
          toStreetVoicesCalendar(calendar, index, userId)
        )
      : [];

    jsonResp(res, 200, [
      ...LOCAL_CALENDARS.map((calendar) => ({ ...calendar, user_id: userId })),
      ...googleCalendars,
    ]);
    return true;
  }

  if (url.pathname === "/api/calendar/events") {
    if (req.method !== "GET") return false;

    const start = url.searchParams.get("start") || new Date().toISOString();
    const end =
      url.searchParams.get("end") ||
      addDays(new Date(start), 30).toISOString();
    const google = await getGoogleCalendarData(req, 30, {
      timeMin: new Date(start).toISOString(),
      timeMax: new Date(end).toISOString(),
    });
    const primaryCalendar =
      (google.calendars || []).find((calendar: any) => calendar.primary) ||
      (google.calendars || [])[0] ||
      { id: "primary", summary: "Google Calendar" };
    const calendarId = `google:${primaryCalendar.id}`;
    const color = googleCalendarColor(0);
    const events = google.connected
      ? (google.events || []).map((event: any) =>
          toStreetVoicesEvent(event, calendarId, color)
        )
      : [];

    jsonResp(res, 200, {
      events,
      tasks: [],
      total: events.length,
    });
    return true;
  }

  return false;
}

function calendarPage(): string {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Calendar - Street Voices</title>
  <style>
    :root {
      --bg: #f6f7f8;
      --panel: #ffffff;
      --ink: #17212b;
      --muted: #5f6b76;
      --line: #dbe1e6;
      --teal: #0f766e;
      --teal-soft: #dff3f0;
      --gold: #9a6700;
      --gold-soft: #fff1c2;
      --danger: #a13d3d;
      --shadow: 0 1px 2px rgba(17, 24, 39, 0.06);
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    .shell {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 48px;
    }
    .topline {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .lede {
      margin: 7px 0 0;
      color: var(--muted);
      font-size: 14px;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    button, .button, select {
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 8px;
      min-height: 38px;
      padding: 0 12px;
      font: inherit;
      font-size: 14px;
      cursor: pointer;
      box-shadow: var(--shadow);
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      white-space: nowrap;
    }
    .button.primary, button.primary {
      background: var(--teal);
      border-color: var(--teal);
      color: #fff;
    }
    .button.danger {
      color: var(--danger);
    }
    .status {
      display: flex;
      align-items: center;
      gap: 10px;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 12px 14px;
      margin-bottom: 16px;
      box-shadow: var(--shadow);
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--gold);
      flex: 0 0 auto;
    }
    .dot.connected { background: var(--teal); }
    .status strong { font-size: 14px; }
    .status span { color: var(--muted); font-size: 13px; }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.85fr);
      gap: 16px;
      align-items: start;
    }
    .section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }
    h2 {
      margin: 0;
      font-size: 15px;
      line-height: 1.3;
      letter-spacing: 0;
    }
    .count {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .list { display: grid; }
    .day {
      padding: 11px 16px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      border-top: 1px solid var(--line);
      background: #f8fafb;
    }
    .day:first-child { border-top: 0; }
    .event, .issue, .goal {
      display: grid;
      grid-template-columns: 88px minmax(0, 1fr);
      gap: 14px;
      padding: 13px 16px;
      border-top: 1px solid var(--line);
    }
    .event:first-child, .issue:first-child, .goal:first-child { border-top: 0; }
    .time {
      color: var(--teal);
      font-weight: 700;
      font-size: 13px;
    }
    .title {
      font-weight: 700;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .meta {
      margin-top: 4px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 3px 8px;
      background: var(--teal-soft);
      color: #075e57;
      font-size: 12px;
      font-weight: 700;
      margin-right: 6px;
    }
    .pill.gold {
      background: var(--gold-soft);
      color: var(--gold);
    }
    .empty {
      padding: 34px 18px;
      color: var(--muted);
      text-align: center;
      font-size: 14px;
    }
    .connect {
      padding: 28px;
      display: grid;
      gap: 14px;
      justify-items: start;
    }
    .connect p {
      margin: 0;
      max-width: 620px;
      color: var(--muted);
      line-height: 1.5;
    }
    .error {
      color: var(--danger);
      background: #fff0f0;
      border: 1px solid #f1c4c4;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
    }
    @media (max-width: 840px) {
      .shell { width: min(100vw - 20px, 720px); padding-top: 16px; }
      .topline { display: grid; }
      .actions { justify-content: flex-start; }
      .grid { grid-template-columns: 1fr; }
      .event, .issue, .goal { grid-template-columns: 72px minmax(0, 1fr); }
    }
  </style>
</head>
<body>
  <main class="shell">
    <div class="topline">
      <div>
        <h1>Calendar</h1>
        <p class="lede">Google Calendar and Nanobot work in one local view.</p>
      </div>
      <div class="actions">
        <select id="days" aria-label="Date range">
          <option value="7">7 days</option>
          <option value="14">14 days</option>
          <option value="30" selected>30 days</option>
          <option value="60">60 days</option>
        </select>
        <button id="refresh" type="button">Refresh</button>
        <a id="disconnect" class="button danger" href="/calendar/disconnect" hidden>Disconnect</a>
      </div>
    </div>

    <div id="status" class="status">
      <span id="status-dot" class="dot"></span>
      <div>
        <strong id="status-title">Checking calendar connection</strong><br>
        <span id="status-detail">One moment.</span>
      </div>
    </div>

    <section id="connect-panel" class="section connect" hidden>
      <h2>Connect Google Calendar</h2>
      <p>This uses read-only Google Calendar access and stores the OAuth token on this machine under Nanobot's local config directory.</p>
      <a id="connect-link" class="button primary" href="/calendar/connect">Connect Google Calendar</a>
      <div id="connect-error" class="error" hidden></div>
    </section>

    <div id="calendar-grid" class="grid" hidden>
      <section class="section">
        <div class="section-head">
          <h2>Google Calendar</h2>
          <span id="event-count" class="count"></span>
        </div>
        <div id="events" class="list"></div>
      </section>

      <section class="section">
        <div class="section-head">
          <h2>Nanobot Timeline</h2>
          <span id="local-count" class="count"></span>
        </div>
        <div id="local" class="list"></div>
      </section>
    </div>
  </main>
  <script src="/unified-nav.js?v=85"></script>
  <script>
    const els = {
      days: document.getElementById('days'),
      refresh: document.getElementById('refresh'),
      disconnect: document.getElementById('disconnect'),
      status: document.getElementById('status'),
      dot: document.getElementById('status-dot'),
      statusTitle: document.getElementById('status-title'),
      statusDetail: document.getElementById('status-detail'),
      connectPanel: document.getElementById('connect-panel'),
      connectLink: document.getElementById('connect-link'),
      connectError: document.getElementById('connect-error'),
      grid: document.getElementById('calendar-grid'),
      events: document.getElementById('events'),
      local: document.getElementById('local'),
      eventCount: document.getElementById('event-count'),
      localCount: document.getElementById('local-count')
    };

    function text(value) {
      return String(value == null ? '' : value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
    }

    function parseDate(value) {
      return value ? new Date(value) : null;
    }

    function dayLabel(value) {
      const date = parseDate(value);
      if (!date || Number.isNaN(date.getTime())) return 'No date';
      return date.toLocaleDateString([], { weekday: 'long', month: 'short', day: 'numeric' });
    }

    function timeLabel(event) {
      const start = parseDate(event.start);
      if (!start || Number.isNaN(start.getTime())) return '';
      if (event.allDay) return 'All day';
      return start.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
    }

    function setConnected(data) {
      els.dot.classList.add('connected');
      els.statusTitle.textContent = data.account && data.account.email ? data.account.email : 'Google Calendar connected';
      els.statusDetail.textContent = 'Synced ' + new Date(data.syncedAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
      els.disconnect.hidden = false;
      els.connectPanel.hidden = true;
      els.grid.hidden = false;
    }

    function setDisconnected(data) {
      els.dot.classList.remove('connected');
      els.statusTitle.textContent = data.configured === false ? 'Google Calendar is not configured' : 'Google Calendar is not connected';
      els.statusDetail.textContent = data.redirectUri ? 'Redirect URI: ' + data.redirectUri : 'Ready to connect.';
      els.connectPanel.hidden = false;
      els.grid.hidden = true;
      els.disconnect.hidden = true;
      els.connectLink.href = data.connectUrl || '/calendar/connect';
      if (data.error) {
        els.connectError.hidden = false;
        els.connectError.textContent = data.error;
      } else {
        els.connectError.hidden = true;
      }
    }

    function renderEvents(events) {
      els.eventCount.textContent = events.length + (events.length === 1 ? ' event' : ' events');
      if (!events.length) {
        els.events.innerHTML = '<div class="empty">No Google Calendar events in this range.</div>';
        return;
      }

      let html = '';
      let currentDay = '';
      for (const event of events) {
        const day = dayLabel(event.start);
        if (day !== currentDay) {
          currentDay = day;
          html += '<div class="day">' + text(day) + '</div>';
        }
        const meta = [event.location, event.organizer, event.attendeeCount ? event.attendeeCount + ' attendees' : '']
          .filter(Boolean)
          .join(' · ');
        html += '<div class="event">';
        html += '<div class="time">' + text(timeLabel(event)) + '</div>';
        html += '<div><div class="title">';
        if (event.htmlLink) {
          html += '<a href="' + text(event.htmlLink) + '" target="_blank" rel="noreferrer">' + text(event.title) + '</a>';
        } else {
          html += text(event.title);
        }
        html += '</div>';
        if (meta) html += '<div class="meta">' + text(meta) + '</div>';
        html += '</div></div>';
      }
      els.events.innerHTML = html;
    }

    function renderLocal(local) {
      const active = Array.isArray(local && local.active) ? local.active.slice(0, 18) : [];
      const goals = Array.isArray(local && local.goals) ? local.goals.slice(0, 6) : [];
      const total = active.length + goals.length;
      els.localCount.textContent = total + (total === 1 ? ' item' : ' items');
      if (!total) {
        els.local.innerHTML = '<div class="empty">No active Nanobot timeline items.</div>';
        return;
      }

      let html = '';
      if (local && local.project) {
        const target = local.project.targetDate ? new Date(local.project.targetDate).toLocaleDateString() : 'No target date';
        html += '<div class="goal"><div class="time">Project</div><div><div class="title">' + text(local.project.name || 'Nanobot') + '</div><div class="meta">Target: ' + text(target) + '</div></div></div>';
      }
      for (const issue of active) {
        html += '<div class="issue"><div class="time">' + text(issue.identifier || '') + '</div><div><div class="title">' + text(issue.title || '') + '</div><div class="meta"><span class="pill">' + text(issue.status || 'active') + '</span><span class="pill gold">' + text(issue.priority || 'normal') + '</span>' + text(issue.assignee || 'unassigned') + '</div></div></div>';
      }
      for (const goal of goals) {
        html += '<div class="goal"><div class="time">Goal</div><div><div class="title">' + text(goal.title || '') + '</div><div class="meta">' + text(goal.level || '') + '</div></div></div>';
      }
      els.local.innerHTML = html;
    }

    async function load() {
      els.refresh.disabled = true;
      try {
        const res = await fetch('/calendar/data?days=' + encodeURIComponent(els.days.value), {
          headers: { Accept: 'application/json' }
        });
        const data = await res.json();
        if (!data.google || !data.google.connected) {
          setDisconnected(data.google || {});
          renderLocal(data.local || {});
          return;
        }
        setConnected(data.google);
        renderEvents(data.google.events || []);
        renderLocal(data.local || {});
      } catch (err) {
        setDisconnected({ configured: true, error: err.message || 'Calendar failed to load.' });
      } finally {
        els.refresh.disabled = false;
      }
    }

    els.refresh.addEventListener('click', load);
    els.days.addEventListener('change', load);
    load();
  </script>
</body>
</html>`;
}

export async function handleGoogleCalendarRoute(
  req: IncomingMessage,
  res: ServerResponse,
  url: URL,
  getLocalSummary: LocalSummaryProvider
): Promise<void> {
  if (await handleOriginalCalendarApi(req, res, url)) return;

  if (url.pathname === "/api/integrations/google/status") {
    if (req.method !== "GET") return jsonResp(res, 405, { error: "Method not allowed" });
    const google = await getGoogleCalendarData(req, 7);
    jsonResp(res, 200, {
      connected: google.connected,
      configured: google.configured,
      account: google.account || null,
      calendars: google.calendars || [],
      redirectUri: google.redirectUri,
      error: google.error || null,
    });
    return;
  }

  if (url.pathname === "/api/integrations/google/connect") {
    if (req.method !== "GET") return jsonResp(res, 405, { error: "Method not allowed" });
    const client = loadClientConfig(req);
    if (!client) {
      return jsonResp(res, 500, {
        error: "Google OAuth credentials were not found.",
        credentialsPath: credentialPath(),
        redirectUri: getRedirectUri(req),
      });
    }
    redirectResp(res, makeAuthUrl(client));
    return;
  }

  if (url.pathname === "/api/integrations/google/sync") {
    if (req.method !== "POST") return jsonResp(res, 405, { error: "Method not allowed" });
    const google = await getGoogleCalendarData(req, 30);
    jsonResp(res, google.connected ? 200 : 401, {
      connected: google.connected,
      events: google.events || [],
      calendars: google.calendars || [],
      syncedAt: google.syncedAt || null,
      error: google.error || null,
    });
    return;
  }

  if (url.pathname === "/calendar/connect") {
    if (req.method !== "GET") return jsonResp(res, 405, { error: "Method not allowed" });
    const client = loadClientConfig(req);
    if (!client) {
      return htmlResp(
        res,
        500,
        "<!doctype html><title>Google Calendar</title><p>Google OAuth credentials were not found.</p>"
      );
    }
    redirectResp(res, makeAuthUrl(client));
    return;
  }

  if (url.pathname === "/calendar/oauth/callback") {
    if (req.method !== "GET") return jsonResp(res, 405, { error: "Method not allowed" });
    const error = url.searchParams.get("error");
    if (error) {
      return redirectResp(res, `/calendar?error=${encodeURIComponent(error)}`);
    }

    const state = url.searchParams.get("state") || "";
    if (!pendingStates.has(state)) {
      return htmlResp(
        res,
        400,
        "<!doctype html><title>Google Calendar</title><p>OAuth state expired. Please reconnect Google Calendar.</p>"
      );
    }
    pendingStates.delete(state);

    const code = url.searchParams.get("code");
    const client = loadClientConfig(req);
    if (!code || !client) {
      return htmlResp(
        res,
        400,
        "<!doctype html><title>Google Calendar</title><p>OAuth callback was missing required data.</p>"
      );
    }

    try {
      await exchangeCode(client, code);
      redirectResp(res, "/calendar?connected=1");
    } catch (err: any) {
      htmlResp(
        res,
        500,
        `<!doctype html><title>Google Calendar</title><p>Google Calendar connection failed: ${String(err.message || err)}</p>`
      );
    }
    return;
  }

  if (url.pathname === "/calendar/disconnect") {
    if (req.method !== "GET" && req.method !== "POST") {
      return jsonResp(res, 405, { error: "Method not allowed" });
    }
    if (existsSync(tokenPath())) rmSync(tokenPath(), { force: true });
    redirectResp(res, "/calendar?disconnected=1");
    return;
  }

  if (url.pathname === "/calendar/data") {
    if (req.method !== "GET") return jsonResp(res, 405, { error: "Method not allowed" });
    const days = Math.min(90, Math.max(1, Number(url.searchParams.get("days") || 30)));
    const [google, local] = await Promise.all([
      getGoogleCalendarData(req, days),
      getLocalSummary().catch((err: any) => ({ error: err.message || "Local calendar failed" })),
    ]);
    jsonResp(res, 200, { google, local });
    return;
  }

  if (url.pathname === "/calendar") {
    if (req.method !== "GET") return jsonResp(res, 405, { error: "Method not allowed" });
    if (wantsHtml(req, url)) {
      htmlResp(res, 200, calendarPage());
      return;
    }
    jsonResp(res, 200, await getLocalSummary());
    return;
  }

  jsonResp(res, 404, { error: "Not found" });
}
