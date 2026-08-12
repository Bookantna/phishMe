import { DOM_NUMERIC_FEATURES } from "../../web/scorer.js";
import { tabStateKey, validateScanMessage, validateSender, validateSenderContext } from "./protocol.js";
import {
  badgeForResult,
  createDetectionResult,
  createUnavailableResult,
} from "./result.js";

export function createScanHandler({
  extensionId,
  now,
  loadModel,
  modelSha256,
  score,
  store,
  setAction,
  isCurrent,
}) {
  return async function handleScan(message, sender) {
    const scannedAt = now();
    let record;
    let tabId;
    try {
      record = validateScanMessage(message, DOM_NUMERIC_FEATURES);
      tabId = validateSender(sender, record.url, extensionId);
    } catch {
      const context = validateSenderContext(sender, extensionId);
      const rejected = createUnavailableResult(context.url, "Scan request rejected", scannedAt);
      if (await isCurrent(context.tabId, context.url)) {
        await store(tabStateKey(context.tabId), rejected);
        await setAction(context.tabId, badgeForResult(rejected));
      }
      return rejected;
    }
    let result;

    try {
      const [model, sha256] = await Promise.all([loadModel(), modelSha256()]);
      const scored = score(model, record);
      result = createDetectionResult({
        score: scored,
        threshold: model.threshold,
        url: record.url,
        scannedAt,
        metadata: model.metadata,
        sha256,
      });
    } catch {
      result = createUnavailableResult(record.url, "Model or scoring unavailable", scannedAt);
    }

    if (!(await isCurrent(tabId, record.url))) {
      return createUnavailableResult(record.url, "Page changed before scan completed", scannedAt);
    }
    try {
      await store(tabStateKey(tabId), result);
      await setAction(tabId, badgeForResult(result));
    } catch {
      const unavailable = createUnavailableResult(record.url, "Page closed before scan state was updated", scannedAt);
      try {
        await store(tabStateKey(tabId), unavailable);
      } catch {
        // The tab/session may already be gone; there is no remaining state to repair.
      }
      return unavailable;
    }
    return result;
  };
}
