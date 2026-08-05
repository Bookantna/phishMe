import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function importScorer() {
  const source = await readFile(new URL("./scorer.js", import.meta.url), "utf8");
  const url = `data:text/javascript;base64,${Buffer.from(source, "utf8").toString("base64")}`;
  return import(url);
}

const scorerPromise = importScorer();
const fixtureArgs = process.argv.slice(2);

function fixtureArgsAvailable() {
  return fixtureArgs.length === 3 && fixtureArgs.every((value) => value && !value.startsWith("--"));
}

test("fnv1a32 matches ASCII and non-ASCII golden vectors", async () => {
  const { fnv1a32 } = await scorerPromise;

  assert.equal(fnv1a32("hello"), 0x4f9f2cab);
  assert.equal(fnv1a32("é"), 0x1e9de8c1);
  assert.equal(fnv1a32("𠮷"), 0xc130251a);
});

test("vectorizeRecord uses binary buckets for collisions", async () => {
  const { scoreRecord, vectorizeRecord } = await scorerPromise;
  const model = {
    schema: "phishme-model-v1",
    feature_version: "phishme-features-v1",
    hash: { name: "fnv1a-32", dimension: 1, ngram_min: 3, ngram_max: 5 },
    include_dom: false,
    numeric_features: [],
    weights: [0.75],
    intercept: 0,
    threshold: 0.5,
    metadata: {},
  };
  const record = { url: "https://aaaa.example/aaaa", title: "aaaa" };

  const vector = vectorizeRecord(model, record);
  const result = scoreRecord(model, record);

  assert.deepEqual(vector.hashIndices, [0]);
  assert.equal(vector.numericValues.length, 0);
  assert.ok(Math.abs(result.probability - 0.679178699175393) <= 1e-12);
});

test("scoreRecord iterates astral n-grams as Unicode code points", async () => {
  const { fnv1a32, scoreRecord, vectorizeRecord } = await scorerPromise;
  const model = {
    schema: "phishme-model-v1",
    feature_version: "phishme-features-v1",
    hash: { name: "fnv1a-32", dimension: 128, ngram_min: 3, ngram_max: 5 },
    include_dom: false,
    numeric_features: [],
    weights: Array(128).fill(0),
    intercept: -0.125,
    threshold: 0.5,
    metadata: {},
  };
  const record = { url: "https://example.com/𠮷abc", title: "Login 𠮷abc" };
  for (const token of ["url:𠮷ab", "title:𠮷ab"]) {
    model.weights[fnv1a32(token) % model.hash.dimension] = 0.5;
  }

  const vector = vectorizeRecord(model, record);
  const result = scoreRecord(model, record);

  assert.ok(vector.hashIndices.includes(fnv1a32("url:𠮷ab") % model.hash.dimension));
  assert.ok(vector.hashIndices.includes(fnv1a32("title:𠮷ab") % model.hash.dimension));
  assert.ok(result.probability > 0.5);
});

test("extractDomFeatures emits frozen allowlisted DOM-only numbers", async () => {
  const { extractDomFeatures } = await scorerPromise;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = () => {
    throw new Error("network access is not allowed");
  };
  const documentRef = {
    documentElement: {
      outerHTML: "<html>\n<body><p>Copyright bank payment bitcoin</p></body>\n</html>",
    },
    body: {
      textContent: "Copyright bank payment bitcoin",
    },
    querySelector() {
      return null;
    },
    querySelectorAll() {
      return [];
    },
  };

  try {
    const features = extractDomFeatures(documentRef, "https://example.com/login");

    assert.equal(Object.isFrozen(features), true);
    assert.equal(features.url_length, undefined);
    assert.equal(features.is_https, undefined);
    assert.equal(typeof features.html_line_count, "number");
    assert.equal(typeof features.has_password_field, "number");
    assert.equal(features.mentions_bank, 1);
    assert.equal(features.mentions_pay, 1);
    assert.equal(features.mentions_crypto, 1);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("extractDomFeatures classifies references with frozen multi-label suffixes", async () => {
  const { extractDomFeatures } = await scorerPromise;
  const links = [
    {
      getAttribute(name) {
        return name === "href" ? "https://cdn.example.co.nz/assets/logo.png" : null;
      },
    },
    {
      getAttribute(name) {
        return name === "href" ? "https://evil.co.nz/login" : null;
      },
    },
  ];
  const documentRef = {
    documentElement: {
      outerHTML: '<html><body><a href="https://cdn.example.co.nz/assets/logo.png"></a></body></html>',
    },
    body: {
      textContent: "",
    },
    querySelector() {
      return null;
    },
    querySelectorAll(selector) {
      return selector === "a[href]" ? links : [];
    },
  };

  const features = extractDomFeatures(documentRef, "https://login.shop.example.co.nz/account");

  assert.equal(features.self_ref_count, 1);
  assert.equal(features.external_ref_count, 1);
});

test("scoreRecord rejects malformed model artifacts", async () => {
  const { scoreRecord } = await scorerPromise;
  const baseModel = () => ({
    schema: "phishme-model-v1",
    feature_version: "phishme-features-v1",
    hash: { name: "fnv1a-32", dimension: 1, ngram_min: 3, ngram_max: 5 },
    include_dom: false,
    numeric_features: [],
    weights: [0],
    intercept: 0,
    threshold: 0.5,
    metadata: {},
  });

  assert.throws(() => scoreRecord({ ...baseModel(), schema: "wrong" }, {}), /schema/);
  assert.throws(
    () => scoreRecord({ ...baseModel(), feature_version: "wrong" }, {}),
    /feature version/,
  );
  assert.throws(
    () => scoreRecord({ ...baseModel(), hash: { ...baseModel().hash, name: "murmur" } }, {}),
    /hash name/,
  );
  assert.throws(() => scoreRecord({ ...baseModel(), weights: [0, 0] }, {}), /dimension/);
  assert.throws(() => scoreRecord({ ...baseModel(), weights: [Number.NaN] }, {}), /finite/);
  assert.throws(
    () => scoreRecord({ ...baseModel(), metadata: JSON.parse('{"__proto__":{"polluted":true}}') }, {}),
    /metadata/,
  );
});

test("scoreRecord caches an immutable validated artifact snapshot", async () => {
  const { scoreRecord } = await scorerPromise;
  const model = {
    schema: "phishme-model-v1",
    feature_version: "phishme-features-v1",
    hash: { name: "fnv1a-32", dimension: 1, ngram_min: 3, ngram_max: 5 },
    include_dom: false,
    numeric_features: [],
    weights: [0.75],
    intercept: 0,
    threshold: 0.5,
    metadata: { dataset_revision: "cache" },
  };
  const record = { url: "https://aaaa.example/aaaa", title: "aaaa" };

  const first = scoreRecord(model, record);
  model.schema = "wrong";
  model.hash.name = "murmur";
  model.hash.dimension = 2;
  model.numeric_features.push("url_length");
  model.weights[0] = 100;
  model.metadata.constructor = { polluted: true };
  const second = scoreRecord(model, record);

  assert.equal(second.featureVersion, first.featureVersion);
  assert.equal(second.probability, first.probability);
  assert.equal(second.label, first.label);
});

if (fixtureArgsAvailable()) {
  test("fixture probability matches Python reference", async () => {
    const { scoreRecord } = await scorerPromise;
    const [modelPath, recordPath, scorePath] = fixtureArgs;
    const [model, record, pythonText] = await Promise.all([
      readFile(modelPath, "utf8").then(JSON.parse),
      readFile(recordPath, "utf8").then(JSON.parse),
      readFile(scorePath, "utf8"),
    ]);

    const result = scoreRecord(model, record);
    const pythonProbability = Number(pythonText.trim());

    assert.ok(Number.isFinite(result.probability));
    assert.ok(
      Math.abs(result.probability - pythonProbability) <= 1e-6,
      `JS ${result.probability} != Python ${pythonProbability}`,
    );
  });
}
