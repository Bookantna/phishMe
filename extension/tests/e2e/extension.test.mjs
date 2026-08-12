import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import test, { after, before } from "node:test";

import puppeteer from "puppeteer-core";

import { buildExtension } from "../../scripts/build.mjs";
import { findChrome } from "../helpers/chrome-path.mjs";
import { startFixtureServer } from "../helpers/local-server.mjs";
import { createTestModel } from "../helpers/test-model.mjs";

const extensionDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const fixturesDir = path.join(extensionDir, "tests", "fixtures");
let sandbox;
let server;
let browser;
let extension;

before(async () => {
  sandbox = await mkdtemp(path.join(extensionDir, ".test-e2e-"));
  const modelPath = path.join(sandbox, "model.json");
  const distPath = path.join(sandbox, "dist");
  await writeFile(modelPath, JSON.stringify(createTestModel()));
  await buildExtension({ modelPath, outDir: distPath });
  server = await startFixtureServer(fixturesDir);
  browser = await puppeteer.launch({
    executablePath: await findChrome(),
    pipe: true,
    headless: true,
    enableExtensions: [distPath],
  });
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline && !extension) {
    const extensions = await browser.extensions();
    extension = [...extensions.values()].find((item) => item.name.startsWith("phishMe"));
    if (!extension) await new Promise((resolve) => setTimeout(resolve, 50));
  }
  assert.ok(extension?.enabled, "phishMe extension did not load");
});

after(async () => {
  await browser?.close();
  await server?.close();
  await rm(sandbox, { recursive: true, force: true });
});

test("positive fixture produces a visible warning, badge, storage, and popup state", async () => {
  const page = await browser.newPage();
  const externalRequests = [];
  page.on("request", (request) => {
    const url = request.url();
    if (!url.startsWith(server.baseUrl) && !url.startsWith("chrome-extension://")) externalRequests.push(url);
  });
  const workerNetwork = await waitForWorker(extension);
  await workerNetwork.client.send("Network.enable");
  workerNetwork.client.on("Network.requestWillBeSent", ({ request }) => {
    const url = request.url;
    if (
      !url.startsWith(server.baseUrl) &&
      !url.startsWith("chrome-extension://") &&
      url !== "chrome://newtab/" &&
      url !== "about:blank"
    ) {
      externalRequests.push(url);
    }
  });
  await page.goto(`${server.baseUrl}/phishing.html`, { waitUntil: "networkidle0" });
  await page.waitForSelector("[data-phishme-warning-host]", { visible: true, timeout: 15_000 });

  const hostStyle = await page.$eval("[data-phishme-warning-host]", (host) => ({
    display: getComputedStyle(host).display,
    position: getComputedStyle(host).position,
  }));
  assert.equal(hostStyle.display, "block");
  assert.equal(hostStyle.position, "fixed");

  const worker = await waitForWorker(extension);
  const tabState = await worker.evaluate(async () => {
    const values = await chrome.storage.session.get(null);
    const [key, value] = Object.entries(values).find(([name]) => name.startsWith("tab:"));
    const tabId = Number(key.slice(4));
    return {
      key,
      result: value,
      badge: await chrome.action.getBadgeText({ tabId }),
      stored: JSON.stringify(value),
    };
  });
  assert.equal(tabState.result.status, "possible-phishing");
  assert.equal(tabState.badge, "!");
  assert.doesNotMatch(tabState.stored, /outerHTML|do-not-collect|<form|<body/i);

  await extension.triggerAction(page);
  const popup = await waitForPopup(extension);
  await popup.waitForSelector("body[data-state=warning]");
  assert.equal(await popup.$eval("#status-heading", (node) => node.textContent), "Possible phishing");
  assert.match(await popup.$eval("#disclaimer", (node) => node.textContent), /not a substitute for Chrome Safe Browsing/i);
  assert.equal(await popup.$eval("#score-value", (node) => node.textContent), "99.3%");
  assert.equal(await popup.$eval("#threshold-value", (node) => node.textContent), "50.0%");
  await popup.close();

  const dismissSelector = "aria/Dismiss warning";
  const buttonText = await page.$eval(dismissSelector, (button) => button.textContent);
  assert.equal(buttonText, "Dismiss warning");
  await page.click(dismissSelector);
  await page.waitForFunction(() => !document.querySelector("[data-phishme-warning-host]"));
  assert.ok(await page.$("form input[type=password]"), "dismissal removed page-owned form content");

  assert.deepEqual([...new Set(externalRequests)], []);
  await workerNetwork.client.send("Network.disable");
  await page.close();
});

test("negative fixture produces no banner, an OK badge, and honest popup wording", async () => {
  const page = await browser.newPage();
  await page.goto(`${server.baseUrl}/benign.html`, { waitUntil: "networkidle0" });
  await waitForStoredResult(extension, page.url(), "no-signal");
  assert.equal(await page.$("[data-phishme-warning-host]"), null);

  const worker = await waitForWorker(extension);
  const state = await worker.evaluate(async (url) => {
    const values = await chrome.storage.session.get(null);
    const [key, value] = Object.entries(values).find(([, result]) => result.url === url);
    return { result: value, badge: await chrome.action.getBadgeText({ tabId: Number(key.slice(4)) }) };
  }, page.url());
  assert.equal(state.result.status, "no-signal");
  assert.equal(state.badge, "OK");

  await extension.triggerAction(page);
  const popup = await waitForPopup(extension);
  await popup.waitForSelector("body[data-state=clear]");
  const popupText = await popup.$eval("body", (node) => node.textContent);
  assert.match(popupText, /No phishing signal detected/);
  assert.doesNotMatch(popupText, /\bthis page is safe\b/i);
  await popup.close();
  await page.close();
});

test("service worker restart preserves popup state and manual rescanning", async () => {
  const page = await browser.newPage();
  await page.goto(`${server.baseUrl}/benign.html`, { waitUntil: "networkidle0" });
  await waitForStoredResult(extension, page.url(), "no-signal");
  await (await waitForWorker(extension)).close();

  await extension.triggerAction(page);
  let popup = await waitForPopup(extension);
  await popup.waitForSelector("body[data-state=clear]");
  assert.equal(await popup.$eval("#status-heading", (node) => node.textContent), "No phishing signal detected");
  await popup.close();

  await page.evaluate(() => {
    const input = document.createElement("input");
    input.type = "password";
    document.body.append(input);
  });
  await extension.triggerAction(page);
  popup = await waitForPopup(extension);
  await popup.click("#rescan");
  await popup.waitForSelector("body[data-state=warning]");
  await popup.close();
  await page.close();
});

test("manual rescan updates the current tab and restricted pages remain neutral", async () => {
  const page = await browser.newPage();
  await page.goto(`${server.baseUrl}/benign.html`, { waitUntil: "networkidle0" });
  await waitForStoredResult(extension, page.url(), "no-signal");
  await page.evaluate(() => {
    const input = document.createElement("input");
    input.type = "password";
    document.body.append(input);
  });

  await extension.triggerAction(page);
  let popup = await waitForPopup(extension);
  await popup.click("#rescan");
  await popup.waitForSelector("body[data-state=warning]");
  assert.equal(await popup.$eval("#status-heading", (node) => node.textContent), "Possible phishing");
  await popup.close();
  await page.waitForSelector("[data-phishme-warning-host]", { visible: true });

  await page.goto("chrome://version");
  await waitForBadge(extension, page, "");
  await extension.triggerAction(page);
  popup = await waitForPopup(extension);
  await popup.waitForSelector("body[data-state=unavailable]");
  assert.match(await popup.$eval("#status-detail", (node) => node.textContent), /cannot be scanned/i);
  await popup.close();
  await page.close();
});

async function waitForBadge(targetExtension, _page, expectedText) {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    const worker = await waitForWorker(targetExtension);
    const text = await worker.evaluate(async () => {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      return Number.isInteger(tab?.id) ? chrome.action.getBadgeText({ tabId: tab.id }) : null;
    });
    if (text === expectedText) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`badge did not become ${JSON.stringify(expectedText)}`);
}

async function waitForStoredResult(targetExtension, url, status) {
  const deadline = Date.now() + 15_000;
  while (Date.now() < deadline) {
    const worker = await waitForWorker(targetExtension);
    const found = await worker.evaluate(async ({ expectedUrl, expectedStatus }) => {
      const values = await chrome.storage.session.get(null);
      return Object.values(values).some((value) => value.url === expectedUrl && value.status === expectedStatus);
    }, { expectedUrl: url, expectedStatus: status });
    if (found) return;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error(`stored ${status} result did not appear for ${url}`);
}

async function waitForWorker(targetExtension) {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    const workers = await targetExtension.workers();
    if (workers.length) return workers[0];
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("extension service worker did not start");
}

async function waitForPopup(targetExtension) {
  const deadline = Date.now() + 10_000;
  while (Date.now() < deadline) {
    const pages = await targetExtension.pages();
    const popup = pages.find((page) => page.url().endsWith("popup.html"));
    if (popup) return popup;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("extension popup did not open");
}
