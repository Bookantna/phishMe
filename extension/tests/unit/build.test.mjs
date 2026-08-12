import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import test from "node:test";

import { buildExtension } from "../../scripts/build.mjs";
import { createTestModel } from "../helpers/test-model.mjs";

const extensionDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

async function readManifest() {
  return JSON.parse(await readFile(path.join(extensionDir, "manifest.json"), "utf8"));
}

test("source manifest uses MV3 with only the required automatic-scan permissions", async () => {
  const manifest = await readManifest();
  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.background.type, "module");
  assert.deepEqual(manifest.permissions, ["storage"]);
  assert.deepEqual(manifest.host_permissions, ["http://*/*", "https://*/*"]);
  assert.equal(manifest.content_scripts[0].all_frames, false);
  assert.equal(manifest.content_scripts[0].run_at, "document_idle");
  assert.ok(!Object.hasOwn(manifest, "web_accessible_resources"));
  const serialized = JSON.stringify(manifest);
  for (const forbidden of [
    '"tabs"', '"webRequest"', '"declarativeNetRequest"', '"notifications"',
    '"nativeMessaging"', '"<all_urls>"',
  ]) {
    assert.ok(!serialized.includes(forbidden), `manifest includes forbidden permission ${forbidden}`);
  }
});

test("source icons are valid PNGs with their declared dimensions", async () => {
  for (const size of [16, 32, 48, 128]) {
    const bytes = await readFile(path.join(extensionDir, "static", "icons", `icon${size}.png`));
    assert.deepEqual([...bytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
    assert.equal(bytes.readUInt32BE(16), size);
    assert.equal(bytes.readUInt32BE(20), size);
  }
});

test("extension documentation covers installation, privacy, permissions, and limitations", async () => {
  const documentation = await readFile(path.join(extensionDir, "README.md"), "utf8");
  for (const required of [
    "npm run build:extension",
    "chrome://extensions",
    "Load unpacked",
    "extension/dist",
    "http://*/*",
    "https://*/*",
    "No browsing data is transmitted",
    "experimental",
    "false positive",
    "not a substitute for Chrome Safe Browsing",
    "hybrid",
    "linear",
  ]) {
    assert.ok(documentation.toLowerCase().includes(required.toLowerCase()), `documentation is missing ${required}`);
  }
});

test("buildExtension refuses destructive output paths", async () => {
  for (const outDir of [
    path.resolve(extensionDir, "..", "unsafe-dist"),
    path.join(extensionDir, "src"),
    path.join(extensionDir, "tests"),
    path.join(extensionDir, "README.md"),
    path.join(extensionDir, "arbitrary-output"),
  ]) {
    await assert.rejects(
      buildExtension({
        modelPath: path.join(extensionDir, "tests", "fixtures", "missing-model.json"),
        outDir,
      }),
      /extension directory|extension\/dist|test sandbox/,
    );
  }
});

test("buildExtension creates a self-contained package and auditable model metadata", async () => {
  const sandbox = await mkdtemp(path.join(extensionDir, ".test-build-"));
  const modelPath = path.join(sandbox, "model.json");
  const outDir = path.join(sandbox, "dist");
  try {
    await writeFile(modelPath, JSON.stringify(createTestModel()));
    const summary = await buildExtension({ modelPath, outDir });
    for (const name of [
      "manifest.json", "service-worker.js", "content.js", "content.css",
      "popup.html", "popup.js", "popup.css", "model.json", "build-info.json",
      "icons/icon16.png", "icons/icon32.png", "icons/icon48.png", "icons/icon128.png",
    ]) {
      assert.ok((await stat(path.join(outDir, name))).isFile(), `missing ${name}`);
    }
    const serviceWorker = await readFile(path.join(outDir, "service-worker.js"), "utf8");
    const content = await readFile(path.join(outDir, "content.js"), "utf8");
    assert.doesNotMatch(serviceWorker, /\.\.\/\.\.\/web\/scorer/);
    assert.doesNotMatch(content, /^\s*import\s/m);
    const buildInfo = JSON.parse(await readFile(path.join(outDir, "build-info.json"), "utf8"));
    assert.equal(buildInfo.model_sha256, summary.modelSha256);
    assert.equal(buildInfo.schema, "phishme-extension-build-v1");
    assert.equal(buildInfo.model_metadata.variant, "linear");
    assert.ok(!JSON.stringify(buildInfo).includes(sandbox));
  } finally {
    await rm(sandbox, { recursive: true, force: true });
  }
});
