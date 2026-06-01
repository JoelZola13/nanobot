const DEFAULT_SETTINGS = {
  baseUrl: "http://localhost:3180/social",
  destinationType: "channel",
  destinationId: "",
  destinationLabel: "",
  documentsUrl: "http://localhost:3180/documents",
  documentsUserId: "",
  openDocumentAfterClip: true,
};

function normalizeMessagesBaseUrl(value) {
  if (!value) throw new Error("Messages URL is required.");
  const url = new URL(value);
  const path = url.pathname.replace(/\/+$/, "");
  return `${url.origin}${path && path !== "/" ? path : "/social"}`;
}

function normalizeDocumentsUrl(value) {
  if (!value) throw new Error("Documents URL is required.");
  const url = new URL(value);
  const path = url.pathname.replace(/\/+$/, "");
  return `${url.origin}${path && path !== "/" ? path : "/documents"}`;
}

async function getSettings() {
  return chrome.storage.sync.get(DEFAULT_SETTINGS);
}

async function saveSettings(settings) {
  const nextSettings = {
    ...DEFAULT_SETTINGS,
    ...settings,
    baseUrl: settings.baseUrl || DEFAULT_SETTINGS.baseUrl,
    documentsUrl: settings.documentsUrl || DEFAULT_SETTINGS.documentsUrl,
    documentsUserId: settings.documentsUserId ?? DEFAULT_SETTINGS.documentsUserId,
    openDocumentAfterClip: settings.openDocumentAfterClip !== false,
  };
  await chrome.storage.sync.set(nextSettings);
  return nextSettings;
}

function destinationFromSelection(selection) {
  if (!selection?.destinationId) {
    throw new Error("Choose a Messages destination first.");
  }

  if (selection.destinationType === "dm") {
    return { type: "dm", userId: selection.destinationId };
  }

  return { type: "channel", channelId: selection.destinationId };
}

async function readError(response) {
  const text = await response.text().catch(() => "");
  if (!text) return response.statusText || "Request failed";

  try {
    const json = JSON.parse(text);
    return json.error || text;
  } catch {
    return text;
  }
}

function truncateText(value, maxLength = 100_000) {
  const text = String(value || "").replace(/\u0000/g, "").trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 24).trim()}\n\n[Clip truncated]`;
}

async function importEmail(email, destinationSelection) {
  const settings = await getSettings();
  const baseUrl = normalizeMessagesBaseUrl(settings.baseUrl);
  const destination = destinationSelection
    ? destinationFromSelection(destinationSelection)
    : destinationFromSelection(settings);
  const response = await fetch(`${baseUrl}/api/email-import`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      destination,
      email,
    }),
  });

  if (!response.ok) {
    if (response.status === 401) {
      throw new Error("Sign in to Messages in this browser, then try again.");
    }
    throw new Error(await readError(response));
  }

  return response.json();
}

async function fetchJson(path) {
  const settings = await getSettings();
  const baseUrl = normalizeMessagesBaseUrl(settings.baseUrl);
  const response = await fetch(`${baseUrl}${path}`, {
    credentials: "include",
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    if (response.status === 401) {
      throw new Error("Sign in to Messages in this browser, then try again.");
    }
    throw new Error(await readError(response));
  }

  return response.json();
}

function channelLabel(channel) {
  return channel.name || channel.slug || "Untitled channel";
}

async function searchDestinations({ destinationType, query }) {
  const q = (query || "").trim().toLowerCase();

  if (destinationType === "dm") {
    if (q.length < 2) return [];
    const users = await fetchJson(`/api/users/search?q=${encodeURIComponent(q)}`);
    return users.map((user) => ({
      id: user.id,
      label: user.displayName || user.username,
      detail: user.username ? `@${user.username}` : user.isAgent ? "Agent" : "Person",
    }));
  }

  const channels = await fetchJson("/api/channels");
  return channels
    .filter((channel) => {
      if (!q) return true;
      return [channel.name, channel.slug, channel.description]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(q));
    })
    .slice(0, 20)
    .map((channel) => ({
      id: channel.id,
      label: channelLabel(channel),
      detail: channel.type === "PRIVATE" ? "Private channel" : "Channel",
    }));
}

function capturePageClipScript() {
  const MAX_TEXT_CHARS = 150_000;
  const MAX_BLOCKS = 260;
  const MAX_TABLES = 8;
  const MAX_TABLE_ROWS = 80;
  const MAX_TABLE_COLUMNS = 12;
  const MAX_LINKS = 60;
  const MAX_IMAGES = 24;
  const BOILERPLATE_SELECTOR = [
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "iframe",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "input",
    "select",
    "textarea",
    "[role='navigation']",
    "[role='banner']",
    "[role='contentinfo']",
    "[aria-hidden='true']",
    ".advertisement",
    ".ads",
    ".ad",
    ".cookie",
    ".cookie-banner",
    ".newsletter",
    ".subscribe",
    ".modal",
    ".popup",
  ].join(",");

  function cleanText(value) {
    return String(value || "")
      .replace(/\u0000/g, "")
      .replace(/[ \t]+/g, " ")
      .replace(/\s+\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  function compactText(value, maxLength = 1_000) {
    const text = cleanText(value);
    return text.length > maxLength ? `${text.slice(0, maxLength - 16).trim()}...` : text;
  }

  function meta(...names) {
    const targetNames = names.map((name) => String(name).toLowerCase());
    for (const node of document.querySelectorAll("meta")) {
      const key = String(
        node.getAttribute("name") ||
          node.getAttribute("property") ||
          node.getAttribute("itemprop") ||
          "",
      ).toLowerCase();
      if (targetNames.includes(key)) {
        const value = cleanText(node.getAttribute("content") || "");
        if (value) return value;
      }
    }
    return "";
  }

  function visible(element) {
    if (!(element instanceof Element)) return false;
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
  }

  function absoluteUrl(value) {
    try {
      return new URL(value, window.location.href).href;
    } catch {
      return "";
    }
  }

  function nodeLooksBoilerplate(node, root) {
    const closest = node.closest?.(BOILERPLATE_SELECTOR);
    return Boolean(closest && closest !== root && !closest.contains(root));
  }

  function scoreRoot(element) {
    const text = cleanText(element.innerText || element.textContent);
    if (text.length < 80) return 0;
    const linkTextLength = Array.from(element.querySelectorAll("a"))
      .map((link) => cleanText(link.innerText || link.textContent).length)
      .reduce((sum, length) => sum + length, 0);
    const linkDensity = text.length ? Math.min(1, linkTextLength / text.length) : 0;
    const paragraphs = element.querySelectorAll("p").length;
    const headings = element.querySelectorAll("h1,h2,h3,h4").length;
    const tables = element.querySelectorAll("table").length;
    return text.length * (1 - linkDensity * 0.65) + paragraphs * 120 + headings * 60 + tables * 240;
  }

  function bestContentRoot() {
    const selectors = [
      "article",
      "[itemtype*='Article']",
      "main",
      "[role='main']",
      "#content",
      ".content",
      ".article",
      ".post",
      ".entry-content",
      ".post-content",
      ".story-body",
      "[data-testid*='article']",
      "[data-component*='article']",
    ].join(",");
    const roots = Array.from(document.querySelectorAll(selectors))
      .filter(visible)
      .sort((a, b) => scoreRoot(b) - scoreRoot(a));
    return roots[0] || document.body;
  }

  function unique(values, keyFn = (value) => value) {
    const seen = new Set();
    const result = [];
    for (const value of values) {
      const key = keyFn(value);
      if (!key || seen.has(key)) continue;
      seen.add(key);
      result.push(value);
    }
    return result;
  }

  function extractHeadings(root) {
    return unique(
      Array.from(root.querySelectorAll("h1,h2,h3,h4"))
        .filter((heading) => visible(heading) && !nodeLooksBoilerplate(heading, root))
        .map((heading) => ({
          level: Number(heading.tagName.replace(/\D/g, "")) || 2,
          text: compactText(heading.innerText || heading.textContent, 500),
        }))
        .filter((heading) => heading.text),
      (heading) => `${heading.level}:${heading.text}`,
    ).slice(0, 80);
  }

  function tableRows(table) {
    return Array.from(table.rows)
      .filter(visible)
      .map((row) =>
        Array.from(row.cells)
          .slice(0, MAX_TABLE_COLUMNS)
          .map((cell) => compactText(cell.innerText || cell.textContent, 1_000)),
      )
      .filter((row) => row.some(Boolean));
  }

  function extractTable(table, index) {
    const rows = tableRows(table).slice(0, MAX_TABLE_ROWS);
    if (!rows.length) return null;
    const firstRow = Array.from(table.rows)[0];
    const hasHeader =
      Boolean(table.querySelector("thead th")) ||
      Boolean(firstRow?.querySelector?.("th")) ||
      rows[0].every((cell) => cell.length < 90);
    const caption = compactText(table.caption?.innerText || table.closest("figure")?.querySelector("figcaption")?.innerText, 500);
    const headers = hasHeader ? rows[0] : [];
    const bodyRows = hasHeader ? rows.slice(1) : rows;
    const columnCount = Math.max(headers.length, ...bodyRows.map((row) => row.length), 1);
    return {
      index,
      caption,
      headers: headers.slice(0, columnCount),
      rows: bodyRows.map((row) => row.slice(0, columnCount)),
      rowCount: rows.length,
      columnCount,
      truncated: tableRows(table).length > MAX_TABLE_ROWS,
    };
  }

  function extractTables(root) {
    const candidateTables = Array.from(root.querySelectorAll("table")).filter(
      (table) => visible(table) && !nodeLooksBoilerplate(table, root),
    );
    const tableElements = [];
    const tables = [];
    for (const tableElement of candidateTables.slice(0, MAX_TABLES)) {
      const table = extractTable(tableElement, tables.length);
      if (!table) continue;
      tableElements.push(tableElement);
      tables.push(table);
    }
    return { tableElements, tables };
  }

  function extractBlocks(root, tableIndexByElement) {
    const selector = "h1,h2,h3,h4,p,li,blockquote,pre,table";
    const blocks = [];
    const seen = new Set();

    for (const node of Array.from(root.querySelectorAll(selector))) {
      if (blocks.length >= MAX_BLOCKS) break;
      if (!visible(node) || nodeLooksBoilerplate(node, root)) continue;
      const tagName = node.tagName.toLowerCase();
      const closestTable = node.closest("table");
      if (closestTable && tagName !== "table") continue;
      if (tagName !== "li" && node.closest("li")) continue;

      if (tagName === "table") {
        const tableIndex = tableIndexByElement.get(node);
        if (typeof tableIndex === "number") {
          blocks.push({ type: "tableRef", tableIndex });
        }
        continue;
      }

      const text = compactText(node.innerText || node.textContent, tagName === "pre" ? 8_000 : 2_000);
      if (!text || (text.length < 2 && !/^h[1-4]$/.test(tagName))) continue;
      const key = `${tagName}:${text.toLowerCase()}`;
      if (seen.has(key)) continue;
      seen.add(key);

      if (/^h[1-4]$/.test(tagName)) {
        blocks.push({ type: "heading", level: Number(tagName.slice(1)), text });
      } else if (tagName === "li") {
        blocks.push({
          type: "listItem",
          ordered: node.parentElement?.tagName?.toLowerCase() === "ol",
          text,
        });
      } else if (tagName === "blockquote") {
        blocks.push({ type: "blockquote", text });
      } else if (tagName === "pre") {
        blocks.push({ type: "codeBlock", text });
      } else {
        blocks.push({ type: "paragraph", text });
      }
    }

    if (!blocks.length) {
      const fallback = cleanText(root.innerText || document.body.innerText);
      for (const paragraph of fallback.split(/\n{2,}/).slice(0, 120)) {
        const text = compactText(paragraph, 2_000);
        if (text) blocks.push({ type: "paragraph", text });
      }
    }

    return blocks;
  }

  function blockText(block, tables) {
    if (block.type === "tableRef") {
      const table = tables[block.tableIndex];
      if (!table) return "";
      const rows = [table.headers, ...table.rows].filter((row) => row?.length);
      return rows.map((row) => row.join(" | ")).join("\n");
    }
    return block.text || "";
  }

  function parseName(value) {
    if (!value) return "";
    if (typeof value === "string") return compactText(value, 500);
    if (Array.isArray(value)) return value.map(parseName).filter(Boolean).join(", ");
    if (typeof value === "object") return compactText(value.name || value.headline || value.title || "", 500);
    return "";
  }

  function flattenJsonLd(value, result = []) {
    if (!value) return result;
    if (Array.isArray(value)) {
      value.forEach((item) => flattenJsonLd(item, result));
      return result;
    }
    if (typeof value === "object") {
      if (Array.isArray(value["@graph"])) {
        flattenJsonLd(value["@graph"], result);
      }
      result.push(value);
    }
    return result;
  }

  function simplifyStructuredData(item) {
    const type = Array.isArray(item["@type"]) ? item["@type"].join(", ") : item["@type"] || item.type || "";
    const fields = {
      headline: parseName(item.headline || item.name),
      description: compactText(item.description || item.abstract, 1_000),
      author: parseName(item.author || item.creator),
      publisher: parseName(item.publisher),
      datePublished: compactText(item.datePublished || item.dateCreated || item.uploadDate, 200),
      dateModified: compactText(item.dateModified, 200),
      keywords: Array.isArray(item.keywords) ? item.keywords.join(", ") : compactText(item.keywords, 500),
    };
    const hasUsefulField = Object.values(fields).some(Boolean);
    if (!type && !hasUsefulField) return null;
    return { type: compactText(type, 200), ...fields };
  }

  function extractStructuredData() {
    const items = [];
    for (const script of document.querySelectorAll('script[type="application/ld+json"]')) {
      try {
        const parsed = JSON.parse(script.textContent || "");
        for (const item of flattenJsonLd(parsed)) {
          const simplified = simplifyStructuredData(item);
          if (simplified) items.push(simplified);
        }
      } catch {
        // Ignore malformed site-provided JSON-LD.
      }
    }
    return unique(items, (item) => `${item.type}:${item.headline || item.description}`).slice(0, 10);
  }

  function extractLinks(root) {
    return unique(
      Array.from(root.querySelectorAll("a[href]"))
        .filter((link) => visible(link) && !nodeLooksBoilerplate(link, root))
        .map((link) => ({
          text: compactText(link.innerText || link.textContent || link.href, 300),
          url: absoluteUrl(link.getAttribute("href") || ""),
          context: compactText(link.closest("p,li,section,article")?.innerText || "", 500),
        }))
        .filter((link) => link.url && link.text && !link.url.startsWith("javascript:")),
      (link) => link.url,
    ).slice(0, MAX_LINKS);
  }

  function extractImages(root) {
    return unique(
      Array.from(root.querySelectorAll("img[src],picture img[src]"))
        .filter((image) => visible(image) && !nodeLooksBoilerplate(image, root))
        .map((image) => ({
          alt: compactText(image.alt || image.getAttribute("aria-label") || "", 300),
          title: compactText(image.title || "", 300),
          url: absoluteUrl(image.currentSrc || image.src || image.getAttribute("src") || ""),
          caption: compactText(image.closest("figure")?.querySelector("figcaption")?.innerText || "", 500),
          width: image.naturalWidth || image.width || 0,
          height: image.naturalHeight || image.height || 0,
        }))
        .filter((image) => image.url),
      (image) => image.url,
    ).slice(0, MAX_IMAGES);
  }

  const root = bestContentRoot();
  const { tableElements, tables } = extractTables(root);
  const tableIndexByElement = new Map(tableElements.map((table, index) => [table, index]));
  const contentBlocks = extractBlocks(root, tableIndexByElement);
  const selectedText = cleanText(window.getSelection()?.toString() || "");
  const title =
    cleanText(meta("og:title", "twitter:title")) ||
    cleanText(document.querySelector("h1")?.innerText) ||
    cleanText(document.title) ||
    "Untitled page";
  const canonicalUrl = absoluteUrl(document.querySelector('link[rel="canonical"]')?.getAttribute("href") || "");
  const headings = extractHeadings(root);
  const links = extractLinks(root);
  const images = extractImages(root);
  const structuredData = extractStructuredData();
  const contentText = selectedText || contentBlocks.map((block) => blockText(block, tables)).filter(Boolean).join("\n\n");
  const truncatedText =
    contentText.length > MAX_TEXT_CHARS
      ? `${contentText.slice(0, MAX_TEXT_CHARS).trim()}\n\n[Clip truncated]`
      : contentText;
  const rootText = cleanText(root.innerText || root.textContent);
  const hostname = window.location.hostname.replace(/^www\./, "");

  return {
    title,
    url: window.location.href,
    canonicalUrl,
    description: cleanText(meta("description", "og:description", "twitter:description")),
    siteName: cleanText(meta("og:site_name", "application-name") || hostname),
    author: cleanText(meta("author", "article:author") || document.querySelector('[rel="author"]')?.textContent),
    publishedAt: cleanText(meta("article:published_time", "date", "pubdate", "datePublished")),
    modifiedAt: cleanText(meta("article:modified_time", "dateModified")),
    language: document.documentElement.lang || "",
    contentType: tables.length ? "table-page" : structuredData.some((item) => /article|posting|news/i.test(item.type)) ? "article" : "web-page",
    rootSelector: root.tagName ? root.tagName.toLowerCase() : "body",
    selectedText,
    text: truncatedText,
    contentBlocks,
    headings,
    tables,
    links,
    images,
    structuredData,
    stats: {
      textCharacters: rootText.length,
      capturedCharacters: truncatedText.length,
      words: truncatedText.split(/\s+/).filter(Boolean).length,
      headings: headings.length,
      tables: tables.length,
      links: links.length,
      images: images.length,
    },
    capturedAt: new Date().toISOString(),
  };
}

function readDocumentsUserIdScript() {
  return window.localStorage.getItem("streetbot:user-id") || "";
}

async function captureActivePageClip() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error("Open a web page before saving it to Documents.");
  if (!/^https?:\/\//i.test(tab.url || "")) {
    throw new Error("Open a regular http or https web page before saving it to Documents.");
  }

  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: capturePageClipScript,
    });
    if (!injection?.result?.text && !injection?.result?.title) {
      throw new Error("I could not read enough content from this page.");
    }
    return injection.result;
  } catch (error) {
    throw new Error(error?.message || "This page cannot be clipped by the browser extension.");
  }
}

async function detectDocumentsUserId(documentsUrl) {
  const url = new URL(documentsUrl);
  const tabs = await chrome.tabs.query({ url: `${url.protocol}//${url.hostname}/*` }).catch(() => []);
  const preferredTabs = tabs
    .filter((tab) => tab.id && /^https?:\/\//i.test(tab.url || "") && new URL(tab.url).origin === url.origin)
    .sort((a, b) => {
      const aIsDocuments = new URL(a.url).pathname.startsWith("/documents") ? 0 : 1;
      const bIsDocuments = new URL(b.url).pathname.startsWith("/documents") ? 0 : 1;
      return aIsDocuments - bIsDocuments;
    });

  for (const tab of preferredTabs) {
    try {
      const [injection] = await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: readDocumentsUserIdScript,
      });
      const detected = String(injection?.result || "").trim();
      if (detected && detected !== "anonymous") return detected;
    } catch {
      // Keep trying other local Nanobot tabs.
    }
  }

  return "";
}

async function fetchAuthenticatedDocumentsUser(documentsOrigin) {
  try {
    const response = await fetch(`${documentsOrigin}/api/auth/refresh`, {
      method: "POST",
      credentials: "include",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return null;

    const text = await response.text();
    if (!text || !text.trim().startsWith("{")) return null;

    const data = JSON.parse(text);
    const user = data?.user || null;
    const userId = String(user?._id || user?.id || user?.userId || "").trim();
    if (!userId || userId === "anonymous") return null;

    return { userId, user, token: data?.token || "", source: "session" };
  } catch {
    return null;
  }
}

async function resolveDocumentsUser(settings, documentsUrl) {
  const manualUserId = String(settings.documentsUserId || "").trim();
  if (manualUserId) return { userId: manualUserId, source: "manual" };

  const documentsOrigin = new URL(documentsUrl).origin;
  const sessionUser = await fetchAuthenticatedDocumentsUser(documentsOrigin);
  if (sessionUser?.userId) return sessionUser;

  const detectedUserId = await detectDocumentsUserId(documentsUrl);
  if (detectedUserId) return { userId: detectedUserId, source: "localStorage" };

  throw new Error(
    "Open Documents and sign in in this browser, then try again. You can also enter a User ID in the extension popup.",
  );
}

async function resolveDocumentsLocation(documentsOrigin, userId) {
  try {
    const response = await fetch(
      `${documentsOrigin}/sbapi/api/document-workspaces?user_id=${encodeURIComponent(userId)}`,
      {
        credentials: "include",
        headers: { Accept: "application/json" },
      },
    );
    if (!response.ok) return {};

    const workspaces = await response.json();
    if (!Array.isArray(workspaces) || !workspaces.length) return {};

    const workspace = workspaces.find((item) => item?.is_default) || workspaces[0];
    return workspace?.id ? { workspaceId: workspace.id, workspaceName: workspace.name || "" } : {};
  } catch {
    return {};
  }
}

function rememberDocumentsClipScript({ userId, documentId }) {
  const id = String(documentId || "").trim();
  const owner = String(userId || "").trim();
  if (!id || !owner) return { ok: false, error: "Missing document or user id." };

  try {
    window.localStorage.setItem("streetbot:user-id", owner);
    const key = `streetbot:created-document-ids:${owner}`;
    let existing = [];
    try {
      const parsed = JSON.parse(window.localStorage.getItem(key) || "[]");
      existing = Array.isArray(parsed) ? parsed : [];
    } catch {
      existing = [];
    }
    const next = [id, ...existing.filter((value) => value !== id)].slice(0, 100);
    window.localStorage.setItem(key, JSON.stringify(next));
    window.dispatchEvent(
      new CustomEvent("nanobot:document-clipped", {
        detail: { userId: owner, documentId: id },
      }),
    );
    return { ok: true, key, count: next.length };
  } catch (error) {
    return { ok: false, error: error?.message || String(error) };
  }
}

async function findDocumentsTabs(documentsUrl) {
  const url = new URL(documentsUrl);
  const tabs = await chrome.tabs.query({ url: `${url.protocol}//${url.hostname}/*` }).catch(() => []);
  return tabs.filter((tab) => {
    if (!tab.id || !/^https?:\/\//i.test(tab.url || "")) return false;
    try {
      const tabUrl = new URL(tab.url);
      return tabUrl.origin === url.origin && tabUrl.pathname.startsWith("/documents");
    } catch {
      return false;
    }
  });
}

function waitForTabComplete(tabId, timeoutMs = 15_000) {
  return new Promise((resolve) => {
    let settled = false;
    let timer = null;

    const finish = (value) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      chrome.tabs.onUpdated.removeListener(listener);
      resolve(value);
    };

    const listener = (updatedTabId, changeInfo) => {
      if (updatedTabId === tabId && changeInfo.status === "complete") {
        finish(true);
      }
    };

    chrome.tabs.onUpdated.addListener(listener);
    timer = setTimeout(() => finish(false), timeoutMs);
    chrome.tabs.get(tabId).then((tab) => {
      if (tab?.status === "complete") finish(true);
    }).catch(() => finish(false));
  });
}

async function rememberDocumentInTab(tabId, userId, documentId) {
  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId },
      func: rememberDocumentsClipScript,
      args: [{ userId, documentId }],
    });
    return injection?.result || null;
  } catch {
    return null;
  }
}

async function openOrRememberSavedDocument(documentsUrl, documentId, userId, shouldOpen) {
  const openUrl = documentId
    ? `${documentsUrl}?documentId=${encodeURIComponent(documentId)}`
    : documentsUrl;

  if (shouldOpen && documentId) {
    const tab = await chrome.tabs.create({ url: openUrl });
    if (tab?.id) {
      await waitForTabComplete(tab.id);
      await rememberDocumentInTab(tab.id, userId, documentId);
    }
    return openUrl;
  }

  const [existingTab] = await findDocumentsTabs(documentsUrl);
  if (existingTab?.id && documentId) {
    await rememberDocumentInTab(existingTab.id, userId, documentId);
  }
  return openUrl;
}

function tiptapText(value, maxLength = 20_000) {
  const text = truncateText(value, maxLength).replace(/\s+/g, " ").trim();
  return text ? { type: "text", text } : null;
}

function tiptapPlainText(value, maxLength = 20_000) {
  const text = truncateText(value, maxLength).replace(/\r\n/g, "\n").trim();
  return text ? { type: "text", text } : null;
}

function tiptapParagraph(value) {
  const content = tiptapText(value);
  return content ? { type: "paragraph", content: [content] } : null;
}

function tiptapHeading(value, level = 2) {
  const content = tiptapText(value);
  return content
    ? { type: "heading", attrs: { level }, content: [content] }
    : null;
}

function tiptapBulletList(items) {
  const content = items
    .map((item) => tiptapText(item))
    .filter(Boolean)
    .map((item) => ({ type: "listItem", content: [{ type: "paragraph", content: [item] }] }));
  return content.length ? { type: "bulletList", content } : null;
}

function tiptapOrderedList(items) {
  const content = items
    .map((item) => tiptapText(item))
    .filter(Boolean)
    .map((item) => ({ type: "listItem", content: [{ type: "paragraph", content: [item] }] }));
  return content.length ? { type: "orderedList", attrs: { start: 1 }, content } : null;
}

function tiptapBlockquote(value) {
  const paragraph = tiptapParagraph(value);
  return paragraph ? { type: "blockquote", content: [paragraph] } : null;
}

function tiptapCodeBlock(value) {
  const content = tiptapPlainText(value, 20_000);
  return content ? { type: "codeBlock", attrs: { language: "plaintext" }, content: [content] } : null;
}

function tiptapTableCell(value, type = "tableCell") {
  return {
    type,
    attrs: { colspan: 1, rowspan: 1, colwidth: null },
    content: [tiptapParagraph(value) || { type: "paragraph" }],
  };
}

function normalizeTableRow(row, width) {
  const normalized = Array.isArray(row) ? row.slice(0, width) : [];
  while (normalized.length < width) normalized.push("");
  return normalized;
}

function tiptapTable(table) {
  const header = Array.isArray(table?.headers) ? table.headers : [];
  const rows = Array.isArray(table?.rows) ? table.rows : [];
  const allRows = [header, ...rows].filter((row) => Array.isArray(row) && row.some(Boolean));
  if (!allRows.length) return null;

  const width = Math.max(...allRows.map((row) => row.length), 1);
  const hasHeader = header.some(Boolean);
  return {
    type: "table",
    content: allRows.slice(0, 80).map((row, rowIndex) => ({
      type: "tableRow",
      content: normalizeTableRow(row, width).map((cell) =>
        tiptapTableCell(cell, hasHeader && rowIndex === 0 ? "tableHeader" : "tableCell"),
      ),
    })),
  };
}

function textParagraphs(value) {
  return truncateText(value, 140_000)
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.replace(/\s+/g, " ").trim())
    .filter(Boolean)
    .slice(0, 160)
    .map(tiptapParagraph)
    .filter(Boolean);
}

function contentBlocksToTiptap(blocks, fallbackText) {
  const nodes = [];
  let pendingList = [];
  let pendingListOrdered = false;

  const flushList = () => {
    if (!pendingList.length) return;
    nodes.push(pendingListOrdered ? tiptapOrderedList(pendingList) : tiptapBulletList(pendingList));
    pendingList = [];
  };

  for (const block of (Array.isArray(blocks) ? blocks : []).slice(0, 240)) {
    if (!block?.type) continue;
    if (block.type === "tableRef") continue;

    if (block.type === "listItem") {
      const isOrdered = Boolean(block.ordered);
      if (pendingList.length && pendingListOrdered !== isOrdered) flushList();
      pendingListOrdered = isOrdered;
      pendingList.push(block.text || "");
      continue;
    }

    flushList();
    if (block.type === "heading") {
      nodes.push(tiptapHeading(block.text, Math.max(2, Math.min(6, block.level || 2))));
    } else if (block.type === "blockquote") {
      nodes.push(tiptapBlockquote(block.text));
    } else if (block.type === "codeBlock") {
      nodes.push(tiptapCodeBlock(block.text));
    } else {
      nodes.push(tiptapParagraph(block.text));
    }
  }
  flushList();

  const filtered = nodes.filter(Boolean);
  return filtered.length ? filtered : textParagraphs(fallbackText || "");
}

function tableMarkdown(table) {
  const header = Array.isArray(table?.headers) ? table.headers : [];
  const rows = Array.isArray(table?.rows) ? table.rows : [];
  const allRows = [header, ...rows].filter((row) => Array.isArray(row) && row.some(Boolean));
  if (!allRows.length) return "";
  const width = Math.max(...allRows.map((row) => row.length), 1);
  const escapeCell = (value) => String(value || "").replace(/\|/g, "\\|").replace(/\n/g, " ").trim();
  const normalizedRows = allRows.map((row) => normalizeTableRow(row, width).map(escapeCell));
  const hasHeader = header.some(Boolean);
  const lines = [];
  if (hasHeader) {
    lines.push(`| ${normalizedRows[0].join(" | ")} |`);
    lines.push(`| ${normalizedRows[0].map(() => "---").join(" | ")} |`);
    normalizedRows.slice(1).forEach((row) => lines.push(`| ${row.join(" | ")} |`));
  } else {
    normalizedRows.forEach((row) => lines.push(row.join(" | ")));
  }
  return lines.join("\n");
}

function formatStructuredDataItem(item) {
  const parts = [
    item?.type ? `Type: ${item.type}` : "",
    item?.headline ? `Title: ${item.headline}` : "",
    item?.author ? `Author: ${item.author}` : "",
    item?.publisher ? `Publisher: ${item.publisher}` : "",
    item?.datePublished ? `Published: ${item.datePublished}` : "",
    item?.dateModified ? `Modified: ${item.dateModified}` : "",
    item?.description ? `Description: ${item.description}` : "",
    item?.keywords ? `Keywords: ${item.keywords}` : "",
  ].filter(Boolean);
  return parts.join(" | ");
}

function sourceHostname(value) {
  try {
    return new URL(value).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function buildDocumentPayload(clip, owner = {}) {
  const title = truncateText(clip?.title || "Web clip", 150);
  const sourceUrl = truncateText(clip?.canonicalUrl || clip?.url || "", 2_000);
  const capturedAt = clip?.capturedAt || new Date().toISOString();
  const tables = Array.isArray(clip?.tables) ? clip.tables : [];
  const structuredData = Array.isArray(clip?.structuredData) ? clip.structuredData : [];
  const links = Array.isArray(clip?.links) ? clip.links : [];
  const images = Array.isArray(clip?.images) ? clip.images : [];
  const headings = Array.isArray(clip?.headings) ? clip.headings : [];
  const hostname = sourceHostname(sourceUrl || clip?.url || "");
  const clipDetails = [
    sourceUrl ? `Source: ${sourceUrl}` : "",
    clip?.siteName ? `Site: ${clip.siteName}` : "",
    clip?.description ? `Description: ${clip.description}` : "",
    clip?.author ? `Author: ${clip.author}` : "",
    clip?.publishedAt ? `Published: ${clip.publishedAt}` : "",
    clip?.modifiedAt ? `Modified: ${clip.modifiedAt}` : "",
    owner.workspaceName ? `Workspace: ${owner.workspaceName}` : "",
    clip?.contentType ? `Detected type: ${clip.contentType}` : "",
    `Clipped: ${capturedAt}`,
    `Capture mode: ${clip?.selectedText ? "Selected text" : "Full page"}`,
    `Captured: ${clip?.stats?.words || 0} words, ${tables.length} table${tables.length === 1 ? "" : "s"}, ${links.length} link${links.length === 1 ? "" : "s"}`,
  ].filter(Boolean);

  const tableNodes = tables.flatMap((table, index) => {
    const nodes = [
      tiptapHeading(table.caption || `Table ${index + 1}`, 3),
      tiptapTable(table),
      table.truncated ? tiptapParagraph("Table rows were truncated to keep the document responsive.") : null,
    ];
    return nodes.filter(Boolean);
  });

  const paragraphs = [
    tiptapHeading(title, 1),
    tiptapHeading("Clip Details", 2),
    tiptapBulletList(clipDetails),
    structuredData.length ? tiptapHeading("Structured Data", 2) : null,
    structuredData.length ? tiptapBulletList(structuredData.map(formatStructuredDataItem)) : null,
    headings.length ? tiptapHeading("Outline", 2) : null,
    headings.length
      ? tiptapBulletList(headings.map((heading) => `${"#".repeat(Math.max(1, Math.min(6, heading.level || 2)))} ${heading.text}`))
      : null,
    tiptapHeading(clip?.selectedText ? "Selected Text" : "Page Text", 2),
    ...(clip?.selectedText
      ? textParagraphs(clip.selectedText)
      : contentBlocksToTiptap(clip?.contentBlocks, clip?.text || "")),
    tableNodes.length ? tiptapHeading("Tables", 2) : null,
    ...tableNodes,
    links.length ? tiptapHeading("Links", 2) : null,
    links.length
      ? tiptapBulletList(links.map((link) => `${link.text} - ${link.url}${link.context ? ` (${link.context})` : ""}`))
      : null,
    images.length ? tiptapHeading("Images", 2) : null,
    images.length
      ? tiptapBulletList(images.map((image) => `${image.alt || image.caption || image.title || "Image"} - ${image.url}`))
      : null,
  ].filter(Boolean);

  const contentText = [
    title,
    `Clip Details:\n${clipDetails.map((item) => `- ${item}`).join("\n")}`,
    structuredData.length ? `Structured Data:\n${structuredData.map((item) => `- ${formatStructuredDataItem(item)}`).join("\n")}` : "",
    headings.length ? `Outline:\n${headings.map((heading) => `- ${heading.text}`).join("\n")}` : "",
    clip?.text || "",
    tables.length
      ? `Tables:\n${tables.map((table, index) => `${table.caption || `Table ${index + 1}`}\n${tableMarkdown(table)}`).join("\n\n")}`
      : "",
    links.length ? `Links:\n${links.map((link) => `- ${link.text}: ${link.url}`).join("\n")}` : "",
    images.length ? `Images:\n${images.map((image) => `- ${image.alt || image.caption || image.title || "Image"}: ${image.url}`).join("\n")}` : "",
  ].filter(Boolean).join("\n\n");

  return {
    title,
    document_type: "general",
    ...(owner.workspaceId ? { workspace_id: owner.workspaceId } : {}),
    ...(owner.folderId ? { folder_id: owner.folderId } : {}),
    content: { type: "doc", content: paragraphs.length ? paragraphs : [tiptapParagraph("No readable page text was found.")] },
    content_text: truncateText(contentText, 100_000),
    metadata: {
      nanobot_web_clip: {
        type: "web_clip",
        scraper_version: "2026-05-05.2",
        page_title: title,
        source_url: clip?.url || "",
        canonical_url: clip?.canonicalUrl || "",
        site_name: clip?.siteName || "",
        description: clip?.description || "",
        author: clip?.author || "",
        published_at: clip?.publishedAt || "",
        modified_at: clip?.modifiedAt || "",
        language: clip?.language || "",
        content_type: clip?.contentType || "",
        clipped_at: capturedAt,
        used_selection: Boolean(clip?.selectedText),
        saved_user_id: owner.userId || "",
        saved_user_source: owner.source || "",
        workspace_id: owner.workspaceId || "",
        workspace_name: owner.workspaceName || "",
        folder_id: owner.folderId || "",
        headings: headings.slice(0, 80),
        structured_data: structuredData.slice(0, 10),
        tables: tables.slice(0, 8),
        links: links.slice(0, 60),
        images: images.slice(0, 24),
        stats: clip?.stats || {},
      },
    },
    tags: ["web-clip", hostname].filter(Boolean).slice(0, 6),
  };
}

async function savePageClip(clip, options = {}) {
  if (!clip) throw new Error("No page clip was captured.");
  const settings = await getSettings();
  const documentsUrl = normalizeDocumentsUrl(settings.documentsUrl);
  const documentsOrigin = new URL(documentsUrl).origin;
  const owner = await resolveDocumentsUser(settings, documentsUrl);
  const location = await resolveDocumentsLocation(documentsOrigin, owner.userId);
  const documentOwner = { ...owner, ...location };
  const response = await fetch(
    `${documentsOrigin}/sbapi/api/documents?user_id=${encodeURIComponent(owner.userId)}`,
    {
      method: "POST",
      credentials: "include",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(buildDocumentPayload(clip, documentOwner)),
    },
  );

  if (!response.ok) {
    throw new Error(await readError(response));
  }

  const document = await response.json();
  const shouldOpen = options.openAfterSave !== false && settings.openDocumentAfterClip !== false;
  const openUrl = await openOrRememberSavedDocument(documentsUrl, document?.id, owner.userId, shouldOpen);

  return { document, openUrl, userId: owner.userId, userSource: owner.source, workspaceId: location.workspaceId || "" };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  const run = async () => {
    if (message?.type === "emailToMessages.getSettings") return getSettings();
    if (message?.type === "emailToMessages.saveSettings") {
      return saveSettings(message.settings || {});
    }
    if (message?.type === "emailToMessages.searchDestinations") {
      return searchDestinations(message);
    }
    if (message?.type === "emailToMessages.importEmail") {
      return importEmail(message.email, message.destination);
    }
    if (message?.type === "emailToMessages.capturePageClip") {
      return captureActivePageClip();
    }
    if (message?.type === "emailToMessages.savePageClip") {
      return savePageClip(message.clip, { openAfterSave: message.openAfterSave });
    }
    throw new Error("Unknown extension message.");
  };

  run()
    .then((result) => sendResponse({ ok: true, result }))
    .catch((error) => sendResponse({ ok: false, error: error.message || String(error) }));

  return true;
});
