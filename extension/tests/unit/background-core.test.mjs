import assert from "node:assert/strict";
import test from "node:test";

import { createScanHandler } from "../../src/background-core.js";
import { SCAN_PAGE } from "../../src/protocol.js";

function scanMessage() {
  return {
    type: SCAN_PAGE,
    record: { url: "https://example.test/", title: "Login", dom: { has_password_field: 1 } },
  };
}

function sender() {
  return { id: "extension-id", frameId: 0, url: "https://example.test/", tab: { id: 7 } };
}

function dependencies(overrides = {}) {
  const calls = [];
  return {
    calls,
    options: {
      extensionId: "extension-id",
      now: () => "2026-08-12T00:00:00.000Z",
      loadModel: async () => ({ threshold: 0.33, metadata: { dataset: "phishpedia", variant: "linear" } }),
      modelSha256: async () => "a".repeat(64),
      score: () => ({ label: 1, probability: 0.9, featureVersion: "phishme-features-v1" }),
      store: async (key, value) => calls.push(["store", key, value.status]),
      setAction: async (tabId, badge) => calls.push(["badge", tabId, badge.text]),
      isCurrent: async () => true,
      ...overrides,
    },
  };
}

test("scan handler validates, scores, stores, and badges a positive page", async () => {
  const { calls, options } = dependencies();
  const result = await createScanHandler(options)(scanMessage(), sender());
  assert.equal(result.status, "possible-phishing");
  assert.deepEqual(calls, [["store", "tab:7", "possible-phishing"], ["badge", 7, "!"]]);
});

test("scan handler stores a no-signal result and OK badge", async () => {
  const { calls, options } = dependencies({
    score: () => ({ label: 0, probability: 0.1, featureVersion: "phishme-features-v1" }),
  });
  const result = await createScanHandler(options)(scanMessage(), sender());
  assert.equal(result.status, "no-signal");
  assert.deepEqual(calls.at(-1), ["badge", 7, "OK"]);
});

test("invalid page messages fail closed without invoking the scorer", async () => {
  let scored = false;
  const { calls, options } = dependencies({ score: () => { scored = true; } });
  const message = scanMessage();
  message.record.html = "forbidden";
  const result = await createScanHandler(options)(message, sender());
  assert.equal(result.status, "unavailable");
  assert.equal(scored, false);
  assert.deepEqual(calls, [["store", "tab:7", "unavailable"], ["badge", 7, ""]]);
});

test("a completed scan cannot overwrite state after its tab navigates", async () => {
  const { calls, options } = dependencies({ isCurrent: async () => false });
  const result = await createScanHandler(options)(scanMessage(), sender());
  assert.equal(result.status, "unavailable");
  assert.equal(result.error, "Page changed before scan completed");
  assert.deepEqual(calls, []);
});

test("badge update failures replace an already-stored result with unavailable", async () => {
  const calls = [];
  const { options } = dependencies({
    store: async (key, value) => calls.push(["store", key, value.status]),
    setAction: async () => { throw new Error("tab closed"); },
  });
  const result = await createScanHandler(options)(scanMessage(), sender());
  assert.equal(result.status, "unavailable");
  assert.equal(result.error, "Page closed before scan state was updated");
  assert.deepEqual(calls, [
    ["store", "tab:7", "possible-phishing"],
    ["store", "tab:7", "unavailable"],
  ]);
});

test("model errors become stored unavailable results with a cleared badge", async () => {
  const { calls, options } = dependencies({
    loadModel: async () => { throw new Error(`model failed ${"secret".repeat(100)}`); },
  });
  const result = await createScanHandler(options)(scanMessage(), sender());
  assert.equal(result.status, "unavailable");
  assert.ok(result.error.length <= 160);
  assert.deepEqual(calls, [["store", "tab:7", "unavailable"], ["badge", 7, ""]]);
});

test("result URL always comes from the validated page record", async () => {
  const { options } = dependencies();
  const result = await createScanHandler(options)(scanMessage(), sender());
  assert.equal(result.url, "https://example.test/");
});
