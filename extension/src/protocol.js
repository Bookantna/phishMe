export const SCAN_PAGE = "phishme.scan-page";
export const RESCAN_PAGE = "phishme.rescan-page";
export const MAX_URL_CODE_POINTS = 16_384;
export const MAX_TITLE_CODE_POINTS = 4_096;
export const MAX_DOM_KEYS = 64;

const MESSAGE_KEYS = Object.freeze(["record", "type"]);
const RECORD_KEYS = Object.freeze(["dom", "title", "url"]);
const UNSAFE_KEYS = new Set(["__proto__", "constructor", "prototype"]);

export function validateScanMessage(message, allowedDomFeatures) {
  const allowedDomKeys = new Set(allowedDomFeatures ?? []);
  requireRecord(message, "message");
  requireExactKeys(message, MESSAGE_KEYS, "message shape mismatch");
  if (message.type !== SCAN_PAGE) throw new Error("unsupported message type");

  const record = message.record;
  requireRecord(record, "record");
  requireExactKeys(record, RECORD_KEYS, "record shape mismatch");
  const url = requireBoundedText(record.url, MAX_URL_CODE_POINTS, "URL");
  const title = requireBoundedText(record.title, MAX_TITLE_CODE_POINTS, "title");
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error("URL must be a valid HTTP(S) URL");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("URL must use HTTP or HTTPS");
  }

  requireRecord(record.dom, "DOM features");
  const entries = Object.entries(record.dom);
  if (entries.length > MAX_DOM_KEYS) throw new Error("DOM features must contain at most 64 keys");
  const dom = Object.create(null);
  for (const [key, value] of entries) {
    if (!key || UNSAFE_KEYS.has(key) || key.includes("\0")) throw new Error("unsafe DOM key");
    if (!allowedDomKeys.has(key)) throw new Error("DOM feature key is not allowlisted");
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error("DOM feature values must be finite numbers");
    }
    dom[key] = value;
  }

  return Object.freeze({ url, title, dom: Object.freeze(dom) });
}

export function validateSender(sender, recordUrl, extensionId) {
  const context = validateSenderContext(sender, extensionId);
  if (context.url !== recordUrl) throw new Error("sender URL does not match record URL");
  return context.tabId;
}

export function validateSenderContext(sender, extensionId) {
  requireRecord(sender, "sender");
  if (sender.id !== extensionId) throw new Error("message must come from this extension");
  if (sender.frameId !== 0) throw new Error("message must come from the top frame");
  const tabId = sender.tab?.id;
  if (!Number.isInteger(tabId) || tabId < 0) throw new Error("sender must have a valid tab id");
  const url = requireBoundedText(sender.url, MAX_URL_CODE_POINTS, "sender URL");
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    throw new Error("sender URL must be valid HTTP(S)");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error("sender URL must use HTTP or HTTPS");
  }
  return Object.freeze({ tabId, url });
}

export function tabStateKey(tabId) {
  if (!Number.isInteger(tabId) || tabId < 0) throw new Error("tab id must be a non-negative integer");
  return `tab:${tabId}`;
}

function requireRecord(value, label) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object`);
  }
}

function requireExactKeys(value, expected, errorMessage) {
  const keys = Object.keys(value).sort();
  if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
    throw new Error(errorMessage);
  }
}

function requireBoundedText(value, limit, label) {
  if (typeof value !== "string") throw new Error(`${label} must be text`);
  if (Array.from(value).length > limit) throw new Error(`${label} exceeds supported length`);
  return value;
}
