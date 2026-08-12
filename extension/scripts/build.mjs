import { createHash } from "node:crypto";
import {
  cp,
  mkdir,
  readFile,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { build } from "esbuild";
import { scoreRecord } from "../../web/scorer.js";

const scriptPath = fileURLToPath(import.meta.url);
const extensionDir = path.resolve(path.dirname(scriptPath), "..");
const MAX_PACKAGE_BYTES = 25 * 1024 * 1024;
const FORBIDDEN_JS = [
  { pattern: /https?:\/\//i, label: "remote URL" },
  { pattern: /\beval\s*\(/, label: "eval" },
  { pattern: /\bnew\s+Function\b/, label: "new Function" },
];

export async function buildExtension({ modelPath, outDir }) {
  const sourceModel = path.resolve(String(modelPath));
  const destination = path.resolve(String(outDir));
  requireSafeOutputDirectory(destination);
  const modelBytes = await readFile(sourceModel);
  if (modelBytes.byteLength > MAX_PACKAGE_BYTES) throw new Error("model exceeds the 25 MB budget");

  let model;
  try {
    model = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(modelBytes));
  } catch (error) {
    throw new Error(`model is not valid UTF-8 JSON: ${error.message}`);
  }
  scoreRecord(model, { url: "", title: "", dom: {} });
  if (model.include_dom !== true) throw new Error("extension requires an include_dom model");
  if (model.metadata?.variant !== "linear") throw new Error("extension requires the linear model variant");

  const modelSha256 = createHash("sha256").update(modelBytes).digest("hex");
  const temporary = `${destination}.tmp-${process.pid}-${Date.now()}`;
  requireWithinExtension(temporary, "temporary output directory");
  await rm(temporary, { recursive: true, force: true });
  await mkdir(temporary, { recursive: true });

  try {
    await Promise.all([
      bundle("service-worker.js", "esm", temporary),
      bundle("content.js", "iife", temporary),
      bundle("popup.js", "iife", temporary),
      copyStaticFiles(temporary),
    ]);
    await writeFile(path.join(temporary, "model.json"), modelBytes);
    const buildInfo = {
      schema: "phishme-extension-build-v1",
      feature_version: model.feature_version,
      model_bytes: modelBytes.byteLength,
      model_metadata: {
        dataset: model.metadata?.dataset ?? "unknown",
        variant: model.metadata.variant,
      },
      model_sha256: modelSha256,
      threshold: model.threshold,
    };
    await writeFile(path.join(temporary, "build-info.json"), stableJson(buildInfo));
    await validateGeneratedJavaScript(temporary);
    await JSON.parse(await readFile(path.join(temporary, "manifest.json"), "utf8"));
    await rm(destination, { recursive: true, force: true });
    await rename(temporary, destination);
  } catch (error) {
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }

  return Object.freeze({
    outDir: destination,
    modelBytes: modelBytes.byteLength,
    modelSha256,
    extensionBytes: await directoryBytes(destination),
  });
}

async function bundle(entry, format, outputDir) {
  await build({
    entryPoints: [path.join(extensionDir, "src", entry)],
    outfile: path.join(outputDir, entry),
    bundle: true,
    format,
    platform: "browser",
    target: "chrome114",
    sourcemap: false,
    legalComments: "none",
    charset: "utf8",
    logLevel: "silent",
  });
}

async function copyStaticFiles(outputDir) {
  await cp(path.join(extensionDir, "manifest.json"), path.join(outputDir, "manifest.json"));
  for (const name of ["content.css", "popup.css", "popup.html"]) {
    await cp(path.join(extensionDir, "static", name), path.join(outputDir, name));
  }
  await cp(path.join(extensionDir, "static", "icons"), path.join(outputDir, "icons"), { recursive: true });
}

async function validateGeneratedJavaScript(outputDir) {
  for (const name of ["service-worker.js", "content.js", "popup.js"]) {
    const source = await readFile(path.join(outputDir, name), "utf8");
    for (const forbidden of FORBIDDEN_JS) {
      if (forbidden.pattern.test(source)) throw new Error(`${name} contains forbidden ${forbidden.label}`);
    }
  }
}

async function directoryBytes(directory) {
  let total = 0;
  const entries = await import("node:fs/promises").then(({ readdir }) => readdir(directory, { withFileTypes: true }));
  for (const entry of entries) {
    const target = path.join(directory, entry.name);
    total += entry.isDirectory() ? await directoryBytes(target) : (await stat(target)).size;
  }
  return total;
}

function requireSafeOutputDirectory(target) {
  requireWithinExtension(target, "output directory");
  const relative = path.relative(extensionDir, target);
  const [topLevel] = relative.split(path.sep);
  const generatedRoot = topLevel === "dist" || /^\.test-(?:build|e2e)-/.test(topLevel);
  if (!generatedRoot) {
    throw new Error("output directory must be extension/dist or a generated test sandbox");
  }
}

function requireWithinExtension(target, label) {
  const relative = path.relative(extensionDir, target);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error(`${label} must be a child of the extension directory`);
  }
}

function stableJson(value) {
  return `${JSON.stringify(sortObject(value))}\n`;
}

function sortObject(value) {
  if (Array.isArray(value)) return value.map(sortObject);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sortObject(value[key])]));
  }
  return value;
}

function parseArguments(argv) {
  const output = {};
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!value || (flag !== "--model" && flag !== "--outdir")) {
      throw new Error("usage: node extension/scripts/build.mjs --model PATH --outdir extension/dist");
    }
    output[flag.slice(2)] = value;
  }
  if (!output.model || !output.outdir) {
    throw new Error("usage: node extension/scripts/build.mjs --model PATH --outdir extension/dist");
  }
  return output;
}

if (path.resolve(process.argv[1] || "") === scriptPath) {
  try {
    const args = parseArguments(process.argv.slice(2));
    const summary = await buildExtension({ modelPath: args.model, outDir: args.outdir });
    process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`phishMe extension build failed: ${error.message}\n`);
    process.exitCode = 1;
  }
}
