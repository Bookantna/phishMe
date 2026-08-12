import { RESCAN_PAGE, tabStateKey } from "./protocol.js";

const heading = document.querySelector("#status-heading");
const detail = document.querySelector("#status-detail");
const scoreDetails = document.querySelector("#score-details");
const scoreValue = document.querySelector("#score-value");
const thresholdValue = document.querySelector("#threshold-value");
const rescan = document.querySelector("#rescan");
const rescanStatus = document.querySelector("#rescan-status");
let activeTab;

initialize().catch(() => renderUnavailable("This page cannot be scanned."));
rescan.addEventListener("click", rescanActiveTab);

async function initialize() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  activeTab = tab;
  if (!isScannableTab(tab)) {
    renderUnavailable("This page cannot be scanned. phishMe runs only on ordinary HTTP(S) pages.");
    return;
  }
  await renderStoredResult();
}

async function renderStoredResult() {
  if (!isScannableTab(activeTab)) return;
  const key = tabStateKey(activeTab.id);
  const stored = await chrome.storage.session.get(key);
  const result = stored[key];
  if (!result || result.url !== activeTab.url) {
    renderUnavailable("This page has not been scanned yet.", true);
    return;
  }
  renderResult(result);
}

async function rescanActiveTab() {
  if (!isScannableTab(activeTab)) return;
  rescan.disabled = true;
  rescanStatus.textContent = "Scanning…";
  try {
    const result = await chrome.tabs.sendMessage(activeTab.id, { type: RESCAN_PAGE });
    if (result?.url === activeTab.url) renderResult(result);
    else await renderStoredResult();
    rescanStatus.textContent = "Scan complete.";
  } catch {
    renderUnavailable("The page could not be scanned. Reload it and try again.", true);
    rescanStatus.textContent = "Scan unavailable.";
  } finally {
    rescan.disabled = false;
  }
}

function renderResult(result) {
  scoreDetails.hidden = true;
  rescan.hidden = false;
  if (result.status === "possible-phishing") {
    heading.textContent = "Possible phishing";
    detail.textContent = "The experimental phishMe model detected a phishing signal. Verify the address before entering sensitive information.";
    document.body.dataset.state = "warning";
  } else if (result.status === "no-signal") {
    heading.textContent = "No phishing signal detected";
    detail.textContent = "The model did not cross its warning threshold. This does not prove the page is legitimate.";
    document.body.dataset.state = "clear";
  } else {
    renderUnavailable(result.error || "The scan is unavailable.", true);
    return;
  }
  scoreValue.textContent = formatPercent(result.probability);
  thresholdValue.textContent = formatPercent(result.threshold);
  scoreDetails.hidden = false;
}

function renderUnavailable(message, canRescan = false) {
  heading.textContent = "Scan unavailable";
  detail.textContent = String(message).slice(0, 240);
  scoreDetails.hidden = true;
  rescan.hidden = !canRescan;
  document.body.dataset.state = "unavailable";
}

function isScannableTab(tab) {
  if (!Number.isInteger(tab?.id) || typeof tab.url !== "string") return false;
  try {
    const protocol = new URL(tab.url).protocol;
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}
