import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_DOM_KEYS,
  MAX_TITLE_CODE_POINTS,
  MAX_URL_CODE_POINTS,
  RESCAN_PAGE,
  SCAN_PAGE,
  tabStateKey,
  validateScanMessage as validateProtocolScanMessage,
  validateSender,
} from "../../src/protocol.js";
import { DOM_NUMERIC_FEATURES } from "../../../web/scorer.js";

function validateScanMessage(message) {
  return validateProtocolScanMessage(message, DOM_NUMERIC_FEATURES);
}

function validMessage() {
  return {
    type: SCAN_PAGE,
    record: {
      url: "https://example.test/login",
      title: "Sign in",
      dom: { has_password_field: 1, image_count: 2 },
    },
  };
}

test("validateScanMessage accepts the exact bounded page-record shape", () => {
  const record = validateScanMessage(validMessage());
  assert.equal(record.url, "https://example.test/login");
  assert.equal(record.title, "Sign in");
  assert.deepEqual({ ...record.dom }, { has_password_field: 1, image_count: 2 });
  assert.equal(Object.getPrototypeOf(record.dom), null);
});

test("validateScanMessage rejects extra top-level and record keys", () => {
  assert.throws(() => validateScanMessage({ ...validMessage(), tabId: 7 }), /message shape/);
  const message = validMessage();
  message.record.html = "<p>no</p>";
  assert.throws(() => validateScanMessage(message), /record shape/);
});

test("validateScanMessage rejects non-http URLs and oversized text", () => {
  for (const url of ["file:///tmp/x", "chrome://version", "not a URL"] ) {
    const message = validMessage();
    message.record.url = url;
    assert.throws(() => validateScanMessage(message), /http/i);
  }
  const longUrl = validMessage();
  longUrl.record.url = `https://example.test/${"x".repeat(MAX_URL_CODE_POINTS)}`;
  assert.throws(() => validateScanMessage(longUrl), /URL/);
  const longTitle = validMessage();
  longTitle.record.title = "𠮷".repeat(MAX_TITLE_CODE_POINTS + 1);
  assert.throws(() => validateScanMessage(longTitle), /title/);
});

test("validateScanMessage rejects non-finite or oversized DOM maps", () => {
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY, true, "1", null]) {
    const message = validMessage();
    message.record.dom = { image_count: value };
    assert.throws(() => validateScanMessage(message), /finite number/);
  }
  const message = validMessage();
  message.record.dom = Object.fromEntries(Array.from({ length: MAX_DOM_KEYS + 1 }, (_, i) => [`x${i}`, i]));
  assert.throws(() => validateScanMessage(message), /64/);
});

test("validateScanMessage rejects unsafe and non-allowlisted DOM keys", () => {
  for (const key of ["", "__proto__", "constructor", "prototype", "bad\0key"]) {
    const message = validMessage();
    message.record.dom = Object.create(null);
    Object.defineProperty(message.record.dom, key, { value: 1, enumerable: true });
    assert.throws(() => validateScanMessage(message), /DOM key/);
  }
  const unknown = validMessage();
  unknown.record.dom = { arbitrary_feature: 1 };
  assert.throws(() => validateScanMessage(unknown), /allowlisted/);
  for (const dom of [null, [], "dom"]) {
    const message = validMessage();
    message.record.dom = dom;
    assert.throws(() => validateScanMessage(message), /DOM/);
  }
});

test("validateSender requires this extension, a top frame, a tab, and the exact URL", () => {
  const sender = { id: "extension-id", frameId: 0, url: "https://example.test/login", tab: { id: 4 } };
  assert.equal(validateSender(sender, sender.url, "extension-id"), 4);
  assert.throws(() => validateSender({ ...sender, id: "other" }, sender.url, "extension-id"), /extension/);
  assert.throws(() => validateSender({ ...sender, frameId: 1 }, sender.url, "extension-id"), /top frame/);
  assert.throws(() => validateSender({ ...sender, tab: {} }, sender.url, "extension-id"), /tab/);
  assert.throws(() => validateSender({ ...sender, url: "https://other.test/" }, sender.url, "extension-id"), /sender URL/);
});

test("tabStateKey accepts only non-negative integer tab IDs", () => {
  assert.equal(tabStateKey(7), "tab:7");
  for (const id of [-1, 1.5, "7", true]) assert.throws(() => tabStateKey(id), /tab/);
  assert.equal(RESCAN_PAGE, "phishme.rescan-page");
});
