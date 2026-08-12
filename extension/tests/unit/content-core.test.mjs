import assert from "node:assert/strict";
import test from "node:test";

import { createPageRecord, shouldRenderWarning } from "../../src/content-core.js";

test("createPageRecord sends URL title and numeric DOM features only", () => {
  const documentRef = { title: "Sign in", documentElement: { outerHTML: "<secret>" } };
  const record = createPageRecord({
    documentRef,
    href: "https://example.test/login",
    extractFeatures: () => Object.freeze({ has_password_field: 1, image_count: 2 }),
  });
  assert.deepEqual(record, {
    url: "https://example.test/login",
    title: "Sign in",
    dom: { has_password_field: 1, image_count: 2 },
  });
  assert.doesNotMatch(JSON.stringify(record), /secret/);
  assert.equal(Object.isFrozen(record), true);
  assert.equal(Object.isFrozen(record.dom), true);
});

test("createPageRecord rejects non-numeric extracted features", () => {
  assert.throws(() => createPageRecord({
    documentRef: { title: "x" },
    href: "https://example.test/",
    extractFeatures: () => ({ password: "secret" }),
  }), /finite number/);
});

test("createPageRecord propagates extraction failures without partial output", () => {
  assert.throws(() => createPageRecord({
    documentRef: { title: "x" },
    href: "https://example.test/",
    extractFeatures: () => { throw new Error("extraction failed"); },
  }), /extraction failed/);
});

test("shouldRenderWarning only accepts a current positive result", () => {
  assert.equal(shouldRenderWarning({ status: "possible-phishing", url: "https://example.test/" }, "https://example.test/"), true);
  assert.equal(shouldRenderWarning({ status: "possible-phishing", url: "https://old.test/" }, "https://example.test/"), false);
  assert.equal(shouldRenderWarning({ status: "no-signal", url: "https://example.test/" }, "https://example.test/"), false);
  assert.equal(shouldRenderWarning(null, "https://example.test/"), false);
});
