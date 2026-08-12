import { access } from "node:fs/promises";
import path from "node:path";

export async function findChrome() {
  const candidates = [
    process.env.CHROME_BIN,
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "Google/Chrome/Application/chrome.exe"),
  ].filter(Boolean).map(normalizeMsysPath);
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {
      // Try the next known installation path.
    }
  }
  throw new Error("Google Chrome not found; set CHROME_BIN to chrome.exe");
}

function normalizeMsysPath(value) {
  const match = /^\/([a-zA-Z])\/(.*)$/.exec(value);
  return match ? `${match[1].toUpperCase()}:/${match[2]}` : value;
}
