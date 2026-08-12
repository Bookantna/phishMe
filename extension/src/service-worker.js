import { createScanHandler } from "./background-core.js";
import { SCAN_PAGE } from "./protocol.js";
import { scoreRecord } from "../../web/scorer.js";

const MODEL_URL = chrome.runtime.getURL("model.json");
const BUILD_INFO_URL = chrome.runtime.getURL("build-info.json");
let resourcesPromise;

const handleScan = createScanHandler({
  extensionId: chrome.runtime.id,
  now: () => new Date().toISOString(),
  loadModel: async () => (await loadResources()).model,
  modelSha256: async () => (await loadResources()).sha256,
  score: scoreRecord,
  store: async (key, result) => chrome.storage.session.set({ [key]: result }),
  setAction: async (tabId, badge) => {
    await Promise.all([
      chrome.action.setBadgeText({ tabId, text: badge.text }),
      chrome.action.setBadgeBackgroundColor({ tabId, color: badge.color }),
      chrome.action.setTitle({ tabId, title: badge.title }),
    ]);
  },
  isCurrent: async (tabId, url) => {
    try {
      return (await chrome.tabs.get(tabId)).url === url;
    } catch {
      return false;
    }
  },
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== SCAN_PAGE) return false;
  handleScan(message, sender).then(
    (result) => sendResponse(result),
    () => sendResponse({
      schema: "phishme-extension-result-v1",
      status: "unavailable",
      label: null,
      probability: null,
      threshold: null,
      url: typeof message?.record?.url === "string" ? message.record.url : "",
      scannedAt: new Date().toISOString(),
      featureVersion: null,
      model: null,
      error: "Scan request rejected",
    }),
  );
  return true;
});

chrome.tabs?.onUpdated?.addListener((tabId, changeInfo) => {
  if (changeInfo.status !== "loading") return;
  Promise.all([
    chrome.storage.session.remove(`tab:${tabId}`),
    chrome.action.setBadgeText({ tabId, text: "" }),
    chrome.action.setTitle({ tabId, title: "phishMe: scan unavailable" }),
  ]).catch(() => {});
});

chrome.tabs?.onRemoved?.addListener((tabId) => {
  chrome.storage.session.remove(`tab:${tabId}`).catch(() => {});
});

function loadResources() {
  if (!resourcesPromise) {
    resourcesPromise = Promise.all([fetchPackaged(MODEL_URL), fetchPackaged(BUILD_INFO_URL)])
      .then(async ([modelBytes, buildInfoBytes]) => {
        const modelText = new TextDecoder("utf-8", { fatal: true }).decode(modelBytes);
        const buildInfoText = new TextDecoder("utf-8", { fatal: true }).decode(buildInfoBytes);
        const buildInfo = JSON.parse(buildInfoText);
        if (buildInfo?.schema !== "phishme-extension-build-v1") throw new Error("invalid build metadata");
        if (!/^[0-9a-f]{64}$/.test(buildInfo.model_sha256)) throw new Error("invalid model hash metadata");
        const sha256 = await sha256Hex(modelBytes);
        if (sha256 !== buildInfo.model_sha256) throw new Error("packaged model hash mismatch");
        const model = JSON.parse(modelText);
        scoreRecord(model, { url: "", title: "", dom: {} });
        return Object.freeze({ model, sha256 });
      })
      .catch((error) => {
        resourcesPromise = undefined;
        throw error;
      });
  }
  return resourcesPromise;
}

async function fetchPackaged(url) {
  const parsed = new URL(url);
  if (parsed.protocol !== "chrome-extension:" || parsed.host !== chrome.runtime.id) {
    throw new Error("refusing non-packaged resource");
  }
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error("packaged resource unavailable");
  return new Uint8Array(await response.arrayBuffer());
}

async function sha256Hex(bytes) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}
