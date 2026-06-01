(function () {
  const ACTION_CLASS = "nanobot-email-action";
  const SENT_CLASS = "nanobot-email-action-sent";
  const ERROR_CLASS = "nanobot-email-action-error";
  const WORKING_CLASS = "nanobot-email-action-working";
  const NOTICE_CLASS = "nanobot-email-notice";
  const STATUS_RESET_MS = 2600;
  let closeActiveDialog = null;

  function visible(element) {
    if (!element) return false;
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden";
  }

  function text(element) {
    return (element?.innerText || element?.textContent || "")
      .replace(/\s+\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();
  }

  function firstVisible(selectors, root = document) {
    const selectorList = Array.isArray(selectors) ? selectors : [selectors];
    for (const selector of selectorList) {
      const elements = Array.from(root.querySelectorAll(selector));
      const match = elements.find(visible);
      if (match) return match;
    }
    return null;
  }

  function pageProvider() {
    const host = window.location.hostname;
    if (host.includes("mail.google.com")) return "gmail";
    if (host.includes("outlook.")) return "outlook";
    return "unknown";
  }

  function selectedText() {
    return (window.getSelection()?.toString() || "").trim();
  }

  function cleanTitle(value) {
    return (value || "")
      .replace(/\s+-\s+Gmail$/, "")
      .replace(/\s+-\s+Outlook$/, "")
      .trim();
  }

  function parseAddressFromText(value) {
    const trimmed = (value || "").trim();
    if (!trimmed) return undefined;

    const angleMatch = trimmed.match(/^(.*?)\s*<([^>]+)>$/);
    if (angleMatch) {
      return {
        name: angleMatch[1].trim() || undefined,
        email: angleMatch[2].trim() || undefined,
      };
    }

    const emailMatch = trimmed.match(/[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/i);
    if (emailMatch) {
      return {
        name: trimmed.replace(emailMatch[0], "").replace(/[<>]/g, "").trim() || undefined,
        email: emailMatch[0],
      };
    }

    return { name: trimmed };
  }

  function parseAddress(element) {
    if (!element) return undefined;
    const email = element.getAttribute("email") || element.getAttribute("data-email");
    const label =
      element.getAttribute("name") ||
      element.getAttribute("aria-label") ||
      element.getAttribute("title") ||
      text(element);
    const parsed = parseAddressFromText(label);

    return {
      ...(parsed?.name ? { name: parsed.name } : {}),
      ...(email || parsed?.email ? { email: email || parsed?.email } : {}),
    };
  }

  function uniqueBy(items, keyFn) {
    const seen = new Set();
    return items.filter((item) => {
      const key = keyFn(item);
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  function parseAddressList(root, selectors) {
    const candidates = selectors.flatMap((selector) => Array.from(root.querySelectorAll(selector)));
    return uniqueBy(candidates.map(parseAddress).filter(Boolean), (address) => {
      return `${address.email || ""}|${address.name || ""}`;
    });
  }

  function extractGmailAttachments(root) {
    const downloadNodes = Array.from(root.querySelectorAll("[download_url]"));
    return uniqueBy(downloadNodes.map((node) => {
      const raw = node.getAttribute("download_url") || "";
      const parts = raw.split(":");
      const name = parts.length >= 3 ? parts[2] : text(node) || node.getAttribute("aria-label");
      return {
        name: name || undefined,
        url: raw || undefined,
        mimeType: parts.length >= 2 ? parts[0] : undefined,
      };
    }), (attachment) => attachment.url || attachment.name);
  }

  function extractOutlookAttachments(root) {
    const nodes = Array.from(root.querySelectorAll("a[download], a[href], [role='listitem']"));
    return uniqueBy(nodes.map((node) => {
      const label = node.getAttribute("aria-label") || node.getAttribute("title") || text(node);
      if (!label || !/\.[a-z0-9]{2,5}/i.test(label)) return null;
      return {
        name: label.replace(/\s+/g, " ").trim(),
        url: node.href || undefined,
      };
    }).filter(Boolean), (attachment) => attachment.url || attachment.name);
  }

  function getGmailMessageRoots() {
    return Array.from(document.querySelectorAll(".adn.ads, [data-message-id]"))
      .filter((root) => visible(root) && firstVisible([".a3s.aiL", ".a3s"], root));
  }

  function getCurrentGmailRoot(preferredRoot) {
    if (preferredRoot && document.contains(preferredRoot)) return preferredRoot;
    const roots = getGmailMessageRoots();
    return roots[roots.length - 1] || document.querySelector('[role="main"]') || document.body;
  }

  function captureGmail(preferredRoot) {
    const main = document.querySelector('[role="main"]') || document.body;
    const root = getCurrentGmailRoot(preferredRoot);
    const bodyEl =
      firstVisible([".a3s.aiL", ".a3s"], root) ||
      firstVisible([".a3s.aiL", ".a3s"], main);
    const subjectEl = firstVisible(
      ['h2[data-thread-perm-id]', "h2.hP", '[data-thread-perm-id]', '[role="main"] h2'],
      main,
    );
    const senderEl =
      firstVisible([".gD[email]", "[email].gD", "[email]", ".go"], root) ||
      firstVisible([".gD[email]", "[email].gD", "[email]", ".go"], main);
    const dateEl =
      firstVisible([".g3[title]", "[data-tooltip][title]", "[title].g3"], root) ||
      firstVisible([".g3[title]", "[data-tooltip][title]", "[title].g3"], main);
    const explicitSelection = selectedText();

    return {
      provider: "gmail",
      subject: text(subjectEl) || cleanTitle(document.title),
      from: parseAddress(senderEl),
      to: parseAddressList(root, [".g2[email]", "[name='to'] [email]", "[data-hovercard-id][email]"]),
      cc: parseAddressList(root, ["[name='cc'] [email]"]),
      sentAt:
        dateEl?.getAttribute("title") ||
        dateEl?.getAttribute("data-tooltip") ||
        text(dateEl) ||
        undefined,
      sourceUrl: window.location.href,
      messageId:
        root.getAttribute("data-message-id") ||
        firstVisible("[data-legacy-message-id]", root)?.getAttribute("data-legacy-message-id") ||
        window.location.hash ||
        undefined,
      bodyText: explicitSelection || text(bodyEl),
      bodyHtml: bodyEl?.innerHTML || undefined,
      attachments: extractGmailAttachments(root),
    };
  }

  function getOutlookRoot() {
    const body = firstVisible([
      '[aria-label="Message body"]',
      '[data-testid="message-body"]',
      '[role="document"]',
      '[role="main"] [dir="ltr"]',
    ]);
    return body?.closest('[role="main"]') || document.querySelector('[role="main"]') || document.body;
  }

  function captureOutlook() {
    const root = getOutlookRoot();
    const subjectEl = firstVisible(
      [
        '[data-testid="message-subject"]',
        '[role="heading"][aria-level="2"]',
        '[aria-label^="Subject"]',
        "h1",
        "h2",
      ],
      root,
    );
    const senderEl = firstVisible(
      [
        '[data-testid="message-header"] [title*="@"]',
        '[aria-label*="From"] [title]',
        '[title*="@"]',
      ],
      root,
    );
    const bodyEl = firstVisible(
      [
        '[aria-label="Message body"]',
        '[data-testid="message-body"]',
        '[role="document"]',
        '[role="main"] [dir="ltr"]',
      ],
      root,
    );
    const dateEl = firstVisible(
      ['[data-testid="SentReceivedSavedTime"]', "time", '[aria-label*="Sent"]'],
      root,
    );
    const explicitSelection = selectedText();

    return {
      provider: "outlook",
      subject: text(subjectEl) || cleanTitle(document.title),
      from: parseAddress(senderEl) || parseAddressFromText(text(senderEl)),
      sentAt: dateEl?.getAttribute("datetime") || dateEl?.getAttribute("title") || text(dateEl),
      sourceUrl: window.location.href,
      bodyText: explicitSelection || text(bodyEl),
      bodyHtml: bodyEl?.innerHTML || undefined,
      attachments: extractOutlookAttachments(root),
    };
  }

  function fallbackCapture() {
    return {
      provider: pageProvider(),
      subject: cleanTitle(document.title) || "(no subject)",
      sourceUrl: window.location.href,
      bodyText: selectedText(),
      attachments: [],
    };
  }

  function captureEmail(preferredRoot) {
    const provider = pageProvider();
    const email = provider === "gmail"
      ? captureGmail(preferredRoot)
      : provider === "outlook"
        ? captureOutlook()
        : fallbackCapture();

    if (!email.bodyText) email.bodyText = selectedText();
    return {
      ...email,
      subject: email.subject || "(no subject)",
      bodyText: email.bodyText || "",
      attachments: email.attachments || [],
    };
  }

  function request(message) {
    return chrome.runtime.sendMessage(message).then((response) => {
      if (!response?.ok) throw new Error(response?.error || "Extension request failed.");
      return response.result;
    });
  }

  function createIcon() {
    const icon = document.createElement("span");
    icon.className = "nanobot-email-action__icon";
    icon.textContent = "N";
    return icon;
  }

  function createActionButton(root) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = ACTION_CLASS;
    button.title = "Choose a Messages channel or DM for this email";
    button.append(createIcon(), document.createTextNode("Choose destination"));
    button.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      openDestinationDialog(button, root);
    });
    return button;
  }

  function setButtonStatus(button, label, className) {
    const icon = button.querySelector(".nanobot-email-action__icon") || createIcon();
    button.replaceChildren(icon, document.createTextNode(label));
    button.classList.remove(SENT_CLASS, ERROR_CLASS, WORKING_CLASS);
    if (className) button.classList.add(className);
  }

  function resetButton(button) {
    setButtonStatus(button, "Choose destination");
    button.title = "Choose a Messages channel or DM for this email";
  }

  function showNotice(anchor, message, type = "success") {
    const root = anchor.closest(".adn, [data-message-id], [role='main']") || document.body;
    root.querySelector(`.${NOTICE_CLASS}`)?.remove();
    const notice = document.createElement("div");
    notice.className = `${NOTICE_CLASS} ${NOTICE_CLASS}--${type}`;
    notice.textContent = message;
    anchor.insertAdjacentElement("afterend", notice);
    window.setTimeout(() => notice.remove(), STATUS_RESET_MS);
  }

  function destinationForSelection(selection) {
    return {
      destinationType: selection.type,
      destinationId: selection.id,
      destinationLabel: selection.label,
    };
  }

  function closeDestinationDialog() {
    closeActiveDialog?.();
    closeActiveDialog = null;
  }

  function openDestinationDialog(button, root) {
    closeDestinationDialog();

    const email = captureEmail(root);
    const previousStatus = button.textContent;
    setButtonStatus(button, "Choose destination", WORKING_CLASS);

    const backdrop = document.createElement("div");
    backdrop.className = "nanobot-email-dialog-backdrop";

    const dialog = document.createElement("div");
    dialog.className = "nanobot-email-dialog";
    dialog.setAttribute("role", "dialog");
    dialog.setAttribute("aria-modal", "true");
    dialog.setAttribute("aria-label", "Send email to Messages");

    const title = document.createElement("div");
    title.className = "nanobot-email-dialog__title";
    title.textContent = "Choose where to send this email";

    const summary = document.createElement("div");
    summary.className = "nanobot-email-dialog__summary";
    summary.textContent = email.subject || "(no subject)";

    const typeLabel = document.createElement("label");
    typeLabel.className = "nanobot-email-dialog__field";
    const typeText = document.createElement("span");
    typeText.textContent = "Destination type";
    const typeInput = document.createElement("select");
    const channelOption = document.createElement("option");
    channelOption.value = "channel";
    channelOption.textContent = "Channel";
    const dmOption = document.createElement("option");
    dmOption.value = "dm";
    dmOption.textContent = "Direct message";
    typeInput.append(channelOption, dmOption);
    typeLabel.append(typeText, typeInput);

    const searchLabel = document.createElement("label");
    searchLabel.className = "nanobot-email-dialog__field";
    const searchText = document.createElement("span");
    searchText.textContent = "Search destination";
    const searchInput = document.createElement("input");
    searchInput.type = "search";
    searchInput.autocomplete = "off";
    searchLabel.append(searchText, searchInput);

    const selectedEl = document.createElement("div");
    selectedEl.className = "nanobot-email-dialog__selected";

    const resultsEl = document.createElement("div");
    resultsEl.className = "nanobot-email-dialog__results";

    const statusEl = document.createElement("div");
    statusEl.className = "nanobot-email-dialog__status";
    statusEl.setAttribute("role", "status");

    const actions = document.createElement("div");
    actions.className = "nanobot-email-dialog__actions";
    const cancelButton = document.createElement("button");
    cancelButton.type = "button";
    cancelButton.className = "nanobot-email-dialog__cancel";
    cancelButton.textContent = "Cancel";
    const sendButton = document.createElement("button");
    sendButton.type = "button";
    sendButton.className = "nanobot-email-dialog__send";
    sendButton.textContent = "Send";
    actions.append(cancelButton, sendButton);

    dialog.append(
      title,
      summary,
      typeLabel,
      searchLabel,
      selectedEl,
      resultsEl,
      statusEl,
      actions,
    );
    backdrop.appendChild(dialog);
    document.documentElement.appendChild(backdrop);

    let settings = {};
    let selected = null;
    let searchTimer = null;
    let closed = false;

    function setStatus(message, type = "") {
      statusEl.textContent = message || "";
      statusEl.className = `nanobot-email-dialog__status ${type ? `nanobot-email-dialog__status--${type}` : ""}`;
    }

    function updateSelected() {
      selectedEl.textContent = selected
        ? `Selected: ${selected.label}`
        : "Pick a channel or DM from the list below";
      sendButton.disabled = !selected;
    }

    function renderResults(results) {
      resultsEl.replaceChildren();
      for (const result of results) {
        const resultButton = document.createElement("button");
        resultButton.type = "button";
        resultButton.className = "nanobot-email-dialog__result";
        const label = document.createElement("strong");
        label.textContent = result.label;
        const detail = document.createElement("span");
        detail.textContent = result.detail || result.id;
        resultButton.append(label, detail);
        resultButton.addEventListener("click", () => {
          selected = {
            type: typeInput.value,
            id: result.id,
            label: result.label,
          };
          searchInput.value = result.label;
          resultsEl.replaceChildren();
          setStatus("");
          updateSelected();
        });
        resultsEl.appendChild(resultButton);
      }
    }

    async function runSearch() {
      const query = searchInput.value.trim();
      if (typeInput.value === "dm" && query.length < 2) {
        renderResults([]);
        setStatus("Type at least 2 letters for direct messages.");
        return;
      }

      setStatus("Searching...");
      try {
        const results = await request({
          type: "emailToMessages.searchDestinations",
          destinationType: typeInput.value,
          query,
        });
        renderResults(results);
        setStatus(results.length ? "" : "No matching destinations.");
      } catch (error) {
        renderResults([]);
        setStatus(error.message, "error");
      }
    }

    function scheduleSearch() {
      window.clearTimeout(searchTimer);
      searchTimer = window.setTimeout(runSearch, 200);
    }

    async function handleSend() {
      if (!selected || sendButton.disabled) return;

      sendButton.disabled = true;
      cancelButton.disabled = true;
      setButtonStatus(button, "Sending...", WORKING_CLASS);
      setStatus("Sending...");

      try {
        const destination = destinationForSelection(selected);
        const result = await request({
          type: "emailToMessages.importEmail",
          email,
          destination,
        });
        request({
          type: "emailToMessages.saveSettings",
          settings: {
            ...settings,
            destinationType: selected.type,
            destinationId: selected.id,
            destinationLabel: selected.label,
          },
        }).catch(() => {});
        setButtonStatus(button, "Sent", SENT_CLASS);
        button.title = "Email sent to Messages";
        showNotice(button, `Email sent to ${selected.label}.`, "success");
        closeDestinationDialog();
        window.setTimeout(() => resetButton(button), STATUS_RESET_MS);
        return result;
      } catch (error) {
        setButtonStatus(button, "Could not send", ERROR_CLASS);
        button.title = error.message || "Could not send email to Messages";
        setStatus(button.title, "error");
        showNotice(button, button.title, "error");
        sendButton.disabled = false;
        cancelButton.disabled = false;
        window.setTimeout(() => resetButton(button), STATUS_RESET_MS);
      }
    }

    closeActiveDialog = () => {
      if (closed) return;
      closed = true;
      window.clearTimeout(searchTimer);
      document.removeEventListener("keydown", handleKeydown);
      backdrop.remove();
      if (previousStatus && button.isConnected && !button.classList.contains(SENT_CLASS)) {
        resetButton(button);
      }
    };

    function handleKeydown(event) {
      if (event.key === "Escape") closeDestinationDialog();
    }

    cancelButton.addEventListener("click", closeDestinationDialog);
    sendButton.addEventListener("click", handleSend);
    typeInput.addEventListener("change", () => {
      selected = null;
      searchInput.value = "";
      searchInput.placeholder = typeInput.value === "dm" ? "Search people or agents" : "Search channels";
      updateSelected();
      runSearch();
    });
    searchInput.addEventListener("input", () => {
      selected = null;
      updateSelected();
      scheduleSearch();
    });
    backdrop.addEventListener("click", (event) => {
      if (event.target === backdrop) closeDestinationDialog();
    });
    document.addEventListener("keydown", handleKeydown);

    request({ type: "emailToMessages.getSettings" })
      .then((loadedSettings) => {
        settings = loadedSettings || {};
        typeInput.value = settings.destinationType || "channel";
        searchInput.placeholder = typeInput.value === "dm" ? "Search people or agents" : "Search channels";
        searchInput.value = "";
        updateSelected();
        searchInput.focus();
        return runSearch();
      })
      .catch((error) => setStatus(error.message, "error"));
  }

  function ensureGmailButtons() {
    const roots = getGmailMessageRoots();
    for (const root of roots) {
      if (root.querySelector(`.${ACTION_CLASS}`)) continue;

      const anchor =
        firstVisible([".gH.acX", ".gE.iv", ".ha"], root) ||
        firstVisible([".a3s.aiL", ".a3s"], root);
      if (!anchor) continue;

      const row = document.createElement("div");
      row.className = "nanobot-email-action-row";
      row.appendChild(createActionButton(root));
      anchor.insertAdjacentElement("afterend", row);
    }
  }

  function ensureOutlookButton() {
    const root = getOutlookRoot();
    if (!root || root.querySelector(`.${ACTION_CLASS}`)) return;

    const subjectEl = firstVisible(
      [
        '[data-testid="message-subject"]',
        '[role="heading"][aria-level="2"]',
        '[aria-label^="Subject"]',
        "h1",
        "h2",
      ],
      root,
    );
    const bodyEl = firstVisible(
      [
        '[aria-label="Message body"]',
        '[data-testid="message-body"]',
        '[role="document"]',
      ],
      root,
    );
    const anchor = subjectEl || bodyEl;
    if (!anchor) return;

    const bar = document.createElement("div");
    bar.className = "nanobot-email-action-row";
    bar.appendChild(createActionButton(root));
    anchor.insertAdjacentElement("afterend", bar);
  }

  function ensureProviderButtons() {
    const provider = pageProvider();
    if (provider === "gmail") ensureGmailButtons();
    if (provider === "outlook") ensureOutlookButton();
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type !== "emailToMessages.capture") return false;
    sendResponse({ email: captureEmail() });
    return true;
  });

  ensureProviderButtons();
  const observer = new MutationObserver(() => ensureProviderButtons());
  observer.observe(document.documentElement, { childList: true, subtree: true });
})();
