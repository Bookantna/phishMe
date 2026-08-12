import assert from "node:assert/strict";
import test from "node:test";

import {
  RESULT_SCHEMA,
  badgeForResult,
  createDetectionResult,
  createUnavailableResult,
} from "../../src/result.js";

const base = {
  score: { label: 1, probability: 0.91, featureVersion: "phishme-features-v1" },
  threshold: 0.33,
  url: "https://example.test/",
  scannedAt: "2026-08-12T00:00:00.000Z",
  metadata: { dataset: "phishpedia", variant: "linear", ignored: "value" },
  sha256: "a".repeat(64),
};

test("positive labels become immutable possible-phishing results", () => {
  const result = createDetectionResult(base);
  assert.equal(result.schema, RESULT_SCHEMA);
  assert.equal(result.status, "possible-phishing");
  assert.deepEqual(result.model, { dataset: "phishpedia", variant: "linear", sha256: "a".repeat(64) });
  assert.equal(Object.isFrozen(result), true);
  assert.deepEqual(badgeForResult(result), {
    text: "!",
    color: "#b42318",
    title: "phishMe: possible phishing detected",
  });
});

test("negative labels become no-signal rather than safe", () => {
  const result = createDetectionResult({
    ...base,
    score: { ...base.score, label: 0, probability: 0.1 },
  });
  assert.equal(result.status, "no-signal");
  assert.doesNotMatch(JSON.stringify(result), /\bsafe\b/i);
  assert.deepEqual(badgeForResult(result), {
    text: "OK",
    color: "#16794b",
    title: "phishMe: no phishing signal detected",
  });
});

test("unavailable results expose a bounded error and clear the badge", () => {
  const result = createUnavailableResult(
    "https://example.test/",
    "x".repeat(500),
    "2026-08-12T00:00:00.000Z",
  );
  assert.equal(result.status, "unavailable");
  assert.equal(result.probability, null);
  assert.equal(result.error.length, 160);
  assert.deepEqual(badgeForResult(result), {
    text: "",
    color: "#667085",
    title: "phishMe: scan unavailable",
  });
});

test("detection results reject invalid scores and model identity", () => {
  for (const probability of [-0.1, 1.1, Number.NaN]) {
    assert.throws(() => createDetectionResult({ ...base, score: { ...base.score, probability } }), /probability/);
  }
  for (const threshold of [-0.1, 1.1, Number.NaN]) {
    assert.throws(() => createDetectionResult({ ...base, threshold }), /threshold/);
  }
  assert.throws(() => createDetectionResult({ ...base, score: { ...base.score, label: 2 } }), /label/);
  assert.throws(() => createDetectionResult({ ...base, sha256: "bad" }), /sha256/);
  assert.throws(() => createDetectionResult({ ...base, metadata: { dataset: {}, variant: "linear" } }), /metadata/);
});
