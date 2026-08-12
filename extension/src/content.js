import { extractDomFeatures } from "../../web/scorer.js";
import { createPageRecord, shouldRenderWarning } from "./content-core.js";
import { RESCAN_PAGE, SCAN_PAGE } from "./protocol.js";

const HOST_ATTRIBUTE = "data-phishme-warning-host";
let inFlight;

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== RESCAN_PAGE) return false;
  scanCurrentPage().then(sendResponse, () => sendResponse({ status: "unavailable" }));
  return true;
});

scanCurrentPage().catch(() => removeWarning());

async function scanCurrentPage() {
  if (inFlight) return inFlight;
  inFlight = performScan().finally(() => {
    inFlight = undefined;
  });
  return inFlight;
}

async function performScan() {
  const record = createPageRecord({
    documentRef: document,
    href: location.href,
    extractFeatures: extractDomFeatures,
  });
  const result = await chrome.runtime.sendMessage({ type: SCAN_PAGE, record });
  renderResult(result);
  return result;
}

function renderResult(result) {
  removeWarning();
  if (!shouldRenderWarning(result, location.href)) return;

  const host = document.createElement("div");
  host.setAttribute(HOST_ATTRIBUTE, "");
  for (const [property, value] of [
    ["all", "initial"],
    ["position", "fixed"],
    ["inset", "0 0 auto 0"],
    ["z-index", "2147483646"],
    ["display", "block"],
    ["pointer-events", "none"],
  ]) {
    host.style.setProperty(property, value, "important");
  }
  const shadow = host.attachShadow({ mode: "closed" });
  const style = document.createElement("style");
  style.textContent = `
    :host { all: initial; color-scheme: light; }
    .warning { pointer-events: auto; box-sizing: border-box; display: grid; grid-template-columns: auto 1fr auto;
      align-items: start; gap: 12px; width: min(760px, calc(100vw - 24px)); margin: 12px auto;
      border: 2px solid #f79009; border-radius: 12px; background: #fffaeb; color: #3b2f14;
      box-shadow: 0 8px 28px rgba(16, 24, 40, .24); padding: 14px 16px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 14px; line-height: 1.45; }
    .mark { display: grid; place-items: center; width: 28px; height: 28px; border-radius: 50%;
      background: #b42318; color: white; font-weight: 800; font-size: 18px; }
    h2 { margin: 0 0 4px; color: #7a271a; font: 700 16px/1.25 inherit; }
    p { margin: 0; } .detail { margin-top: 4px; color: #664d03; font-size: 12px; }
    button { min-width: 44px; min-height: 44px; border: 1px solid #d0d5dd; border-radius: 8px;
      background: white; color: #344054; padding: 8px 12px; font: 600 13px/1.2 inherit; cursor: pointer; }
    button:hover { background: #f9fafb; } button:focus-visible { outline: 3px solid #2e90fa; outline-offset: 2px; }
    @media (max-width: 560px) { .warning { grid-template-columns: auto 1fr; }
      button { grid-column: 1 / -1; width: 100%; } }
    @media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
  `;
  const warning = document.createElement("section");
  warning.className = "warning";
  warning.setAttribute("role", "alert");
  warning.setAttribute("aria-labelledby", "phishme-warning-title");

  const mark = document.createElement("span");
  mark.className = "mark";
  mark.setAttribute("aria-hidden", "true");
  mark.textContent = "!";

  const copy = document.createElement("div");
  const heading = document.createElement("h2");
  heading.id = "phishme-warning-title";
  heading.textContent = "Possible phishing page";
  const body = document.createElement("p");
  body.textContent = "phishMe's experimental local model flagged this page. Avoid entering passwords or payment details until you verify the address.";
  const detail = document.createElement("p");
  detail.className = "detail";
  detail.textContent = "This warning can be wrong. Chrome's built-in protections remain important.";
  copy.append(heading, body, detail);

  const dismiss = document.createElement("button");
  dismiss.type = "button";
  dismiss.setAttribute("data-phishme-dismiss", "");
  dismiss.textContent = "Dismiss warning";
  dismiss.addEventListener("click", () => host.remove());

  warning.append(mark, copy, dismiss);
  shadow.append(style, warning);
  (document.documentElement || document).append(host);
}

function removeWarning() {
  document.querySelector(`[${HOST_ATTRIBUTE}]`)?.remove();
}
