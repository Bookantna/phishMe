# phishMe Chrome Extension

The phishMe Chrome extension is an experimental Manifest V3 client for the project's trained linear phishing model. It evaluates ordinary HTTP(S) pages locally using the same feature and scoring implementation as `web/scorer.js`.

## Build with the reviewed trained model

From `C:\Users\User\codespace\phishMe` in Git Bash:

```bash
npm install
npm test
npm run build:extension -- \
  --model 'D:/phishme-dataset/artifacts/phishpedia-full-v3-reviewed/model-linear.json' \
  --outdir extension/dist
```

The reviewed model currently has SHA-256 `91602262a2dae69238c22cf91f9b27efe4ce0b81ec106a6d5be914d0f771443a`. The build validates the strict `phishme-model-v1` schema, requires an `include_dom` linear model, writes auditable `build-info.json`, and produces a self-contained unpacked extension under `extension/dist`.

`extension/dist` includes a local model artifact and is intentionally git-ignored. Do not commit it.

## Install in Google Chrome

1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Select **Load unpacked**.
4. Choose `C:\Users\User\codespace\phishMe\extension\dist`.
5. Pin phishMe from Chrome's extensions menu if desired.
6. Visit an HTTP(S) page and open the phishMe toolbar action to see the latest local result.

A positive classification shows a red `!` badge and a dismissible warning banner. A negative classification shows an `OK` badge and the phrase **No phishing signal detected**. It does not claim that the page is safe.

## Permissions and privacy

The extension requests:

- `storage`: saves the latest per-tab result in `chrome.storage.session` so the popup still works after the Manifest V3 service worker goes idle.
- `http://*/*` and `https://*/*`: required for automatic top-frame DOM feature extraction on ordinary web pages.

It does not request tabs, browsing history, web-request interception, downloads, notifications, native messaging, or remote code permissions.

All model evaluation is local. No browsing data is transmitted. The extension does not message or store raw HTML, visible body text, form values, passwords, or payment details. The page content script sends only the current URL, document title, and allowlisted numeric DOM features to the extension service worker. Session results are removed when the extension/browser session ends.

Chrome internal pages, Chrome Web Store pages, file URLs, and other restricted pages cannot be scanned and show a neutral unavailable state.

## Important limitations

This is an **experimental research model**, not a security boundary and not a substitute for Chrome Safe Browsing.

The main project evaluation found that models with near-perfect PhishPedia validation scores generalized poorly to the sampled PhreshPhish data. False-positive rates were high. Consequently:

- warnings can be false positives;
- **No phishing signal detected** does not prove that a page is legitimate;
- the extension does not block navigation or form submission;
- keep Chrome Safe Browsing enabled;
- independently verify suspicious addresses before entering credentials or payment details.

The reviewed V3 run selected the hybrid variant as its primary in-domain research model. This extension deliberately packages the trained **linear** variant because it is the variant exported in the audited browser-compatible `phishme-model-v1` format. It must not be described as deploying the hybrid model.

## Development and verification

```bash
# JavaScript scorer parity and extension unit tests
npm test

# Real installed-Chrome extension tests
CHROME_BIN='/c/Program Files/Google/Chrome/Application/chrome.exe' \
  npm run test:extension:e2e

# Existing Python quality gates
python -m pytest -q
ruff check .

# Rebuild with another compatible trained linear artifact
npm run build:extension -- --model PATH/TO/model-linear.json --outdir extension/dist
```

The real-Chrome suite loads a temporary unpacked extension with a deterministic fixture model. It verifies positive and negative behavior, the banner, per-tab badges, session storage, popup wording, service-worker restart recovery, manual rescanning, restricted-page handling, and the absence of external network requests.
