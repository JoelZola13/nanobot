const baseUrlInput = document.getElementById("baseUrl");
const destinationTypeInput = document.getElementById("destinationType");
const destinationSearchInput = document.getElementById("destinationSearch");
const selectedDestination = document.getElementById("selectedDestination");
const resultsEl = document.getElementById("results");
const sendButton = document.getElementById("sendButton");
const documentsUrlInput = document.getElementById("documentsUrl");
const documentsUserIdInput = document.getElementById("documentsUserId");
const openDocumentAfterClipInput = document.getElementById("openDocumentAfterClip");
const clipButton = document.getElementById("clipButton");
const statusEl = document.getElementById("status");

let settings = {};
let searchTimer = null;

function request(message) {
  return chrome.runtime.sendMessage(message).then((response) => {
    if (!response?.ok) throw new Error(response?.error || "Extension request failed.");
    return response.result;
  });
}

function setStatus(message, type = "") {
  statusEl.textContent = message || "";
  statusEl.className = type;
}

function updateSelectedDestination() {
  selectedDestination.textContent = settings.destinationLabel
    ? `Selected: ${settings.destinationLabel}`
    : "No destination selected";
}

function clearSelectedDestination() {
  settings = {
    ...settings,
    destinationId: "",
    destinationLabel: "",
  };
  updateSelectedDestination();
  request({
    type: "emailToMessages.saveSettings",
    settings,
  }).catch(() => {});
}

async function save(partial) {
  settings = await request({
    type: "emailToMessages.saveSettings",
    settings: { ...settings, ...partial },
  });
  updateSelectedDestination();
}

function renderResults(results) {
  resultsEl.replaceChildren();

  for (const result of results) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "result";
    button.innerHTML = `<strong></strong><span></span>`;
    button.querySelector("strong").textContent = result.label;
    button.querySelector("span").textContent = result.detail || result.id;
    button.addEventListener("click", async () => {
      await save({
        destinationType: destinationTypeInput.value,
        destinationId: result.id,
        destinationLabel: result.label,
      });
      destinationSearchInput.value = result.label;
      resultsEl.replaceChildren();
      setStatus("Destination saved.", "success");
    });
    resultsEl.appendChild(button);
  }
}

async function runSearch() {
  const query = destinationSearchInput.value.trim();
  setStatus(query ? "Searching..." : "");

  try {
    await save({
      baseUrl: baseUrlInput.value.trim(),
      destinationType: destinationTypeInput.value,
    });
    const results = await request({
      type: "emailToMessages.searchDestinations",
      destinationType: destinationTypeInput.value,
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
  searchTimer = window.setTimeout(runSearch, 250);
}

async function captureCurrentTabEmail() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error("No active mail tab found.");
  const response = await chrome.tabs.sendMessage(tab.id, {
    type: "emailToMessages.capture",
  });
  if (!response?.email) throw new Error("Open a Gmail or Outlook email first.");
  return response.email;
}

async function sendCurrentEmail() {
  sendButton.disabled = true;
  setStatus("Checking destination...");

  try {
    await save({
      baseUrl: baseUrlInput.value.trim(),
      destinationType: destinationTypeInput.value,
    });
    const typedDestination = destinationSearchInput.value.trim();
    if (
      !settings.destinationId ||
      !settings.destinationLabel ||
      typedDestination !== settings.destinationLabel
    ) {
      throw new Error("Pick a channel or DM from the search results before sending.");
    }
    const email = await captureCurrentTabEmail();
    setStatus("Sending to Messages...");
    await request({
      type: "emailToMessages.importEmail",
      email,
      destination: {
        destinationType: settings.destinationType,
        destinationId: settings.destinationId,
        destinationLabel: settings.destinationLabel,
      },
    });
    setStatus(`Sent to ${settings.destinationLabel}.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    sendButton.disabled = false;
  }
}

async function saveClipSettings() {
  settings = await request({
    type: "emailToMessages.saveSettings",
    settings: {
      ...settings,
      documentsUrl: documentsUrlInput.value.trim(),
      documentsUserId: documentsUserIdInput.value.trim(),
      openDocumentAfterClip: openDocumentAfterClipInput.checked,
    },
  });
}

async function saveCurrentPageToDocuments() {
  clipButton.disabled = true;
  setStatus("Scraping page...");

  try {
    await saveClipSettings();
    const clip = await request({ type: "emailToMessages.capturePageClip" });
    setStatus("Saving to Documents...");
    const result = await request({
      type: "emailToMessages.savePageClip",
      clip,
      openAfterSave: openDocumentAfterClipInput.checked,
    });
    const title = result?.document?.title || clip.title || "web clip";
    const owner = result?.userSource ? ` (${result.userSource})` : "";
    setStatus(`Saved "${title}" to Documents${owner}.`, "success");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    clipButton.disabled = false;
  }
}

async function init() {
  settings = await request({ type: "emailToMessages.getSettings" });
  baseUrlInput.value = settings.baseUrl || "";
  documentsUrlInput.value = settings.documentsUrl || "";
  documentsUserIdInput.value = settings.documentsUserId || "";
  openDocumentAfterClipInput.checked = settings.openDocumentAfterClip !== false;
  destinationTypeInput.value = settings.destinationType || "channel";
  destinationSearchInput.placeholder =
    destinationTypeInput.value === "dm" ? "Search people or agents" : "Search channels";
  if (settings.destinationLabel) destinationSearchInput.value = settings.destinationLabel;
  updateSelectedDestination();
}

baseUrlInput.addEventListener("change", () => save({ baseUrl: baseUrlInput.value.trim() }));
destinationTypeInput.addEventListener("change", async () => {
  await save({
    destinationType: destinationTypeInput.value,
    destinationId: "",
    destinationLabel: "",
  });
  destinationSearchInput.value = "";
  destinationSearchInput.placeholder =
    destinationTypeInput.value === "dm" ? "Search people or agents" : "Search channels";
  resultsEl.replaceChildren();
});
destinationSearchInput.addEventListener("input", () => {
  clearSelectedDestination();
  scheduleSearch();
});
sendButton.addEventListener("click", sendCurrentEmail);
documentsUrlInput.addEventListener("change", saveClipSettings);
documentsUserIdInput.addEventListener("change", saveClipSettings);
openDocumentAfterClipInput.addEventListener("change", saveClipSettings);
clipButton.addEventListener("click", saveCurrentPageToDocuments);

init().catch((error) => setStatus(error.message, "error"));
