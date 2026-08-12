const SCHEMA = "phishme-model-v1";
const FEATURE_VERSION = "phishme-features-v1";
const HASH_NAME = "fnv1a-32";
const NGRAM_MIN = 3;
const NGRAM_MAX = 5;

const COUNT_FEATURES = [
  "url_length",
  "domain_length",
  "tld_length",
  "subdomain_count",
  "letter_count",
  "digit_count",
  "obfuscated_count",
  "query_count",
  "equals_count",
  "ampersand_count",
  "special_count",
  "html_line_count",
  "largest_line_length",
  "iframe_count",
  "image_count",
  "css_count",
  "javascript_count",
  "self_ref_count",
  "empty_ref_count",
  "external_ref_count",
];
const RATIO_FEATURES = ["letter_ratio", "digit_ratio", "obfuscated_ratio", "special_ratio"];
const BOOLEAN_FEATURES = [
  "is_domain_ip",
  "is_https",
  "has_title",
  "has_favicon",
  "is_responsive",
  "has_description",
  "external_form_submit",
  "has_social",
  "has_submit_button",
  "has_hidden_fields",
  "has_password_field",
  "mentions_bank",
  "mentions_pay",
  "mentions_crypto",
  "has_copyright",
];
const NUMERIC_FEATURES = Object.freeze([...COUNT_FEATURES, ...RATIO_FEATURES, ...BOOLEAN_FEATURES]);
const URL_NUMERIC_FEATURES = Object.freeze([
  "url_length",
  "domain_length",
  "tld_length",
  "subdomain_count",
  "letter_count",
  "digit_count",
  "obfuscated_count",
  "query_count",
  "equals_count",
  "ampersand_count",
  "special_count",
  "letter_ratio",
  "digit_ratio",
  "obfuscated_ratio",
  "special_ratio",
  "is_domain_ip",
  "is_https",
]);
const DOM_NUMERIC_FEATURES = Object.freeze(
  NUMERIC_FEATURES.filter((name) => !URL_NUMERIC_FEATURES.includes(name)),
);
const COUNT_FEATURE_SET = new Set(COUNT_FEATURES);
const RATIO_FEATURE_SET = new Set(RATIO_FEATURES);
const URL_NUMERIC_FEATURE_SET = new Set(URL_NUMERIC_FEATURES);
const DOM_NUMERIC_FEATURE_SET = new Set(DOM_NUMERIC_FEATURES);
const UNSAFE_METADATA_KEYS = new Set(["__proto__", "constructor", "prototype"]);
const TOP_LEVEL_KEYS = [
  "feature_version",
  "hash",
  "include_dom",
  "intercept",
  "metadata",
  "numeric_features",
  "schema",
  "threshold",
  "weights",
];
const HASH_KEYS = ["dimension", "name", "ngram_max", "ngram_min"];
const FROZEN_SINGLE_LABEL_SUFFIXES = new Set([
  "ai",
  "app",
  "au",
  "biz",
  "br",
  "cn",
  "co",
  "com",
  "dev",
  "edu",
  "gov",
  "info",
  "int",
  "io",
  "jp",
  "kr",
  "mil",
  "mx",
  "net",
  "nz",
  "org",
  "pl",
  "sa",
  "sg",
  "th",
  "tr",
  "uk",
  "us",
  "za",
]);
const FROZEN_MULTI_LABEL_SUFFIXES = Object.freeze([
  "ac.th",
  "ac.uk",
  "co.jp",
  "co.kr",
  "co.nz",
  "co.th",
  "co.uk",
  "co.za",
  "com.au",
  "com.br",
  "com.cn",
  "com.mx",
  "com.pl",
  "com.sa",
  "com.sg",
  "com.tr",
  "edu.au",
  "go.th",
  "gov.au",
  "gov.uk",
  "ltd.uk",
  "me.uk",
  "ne.jp",
  "net.au",
  "net.nz",
  "or.jp",
  "or.th",
  "org.nz",
  "org.uk",
]);
const FROZEN_MULTI_LABEL_SUFFIX_LABELS = Object.freeze(
  FROZEN_MULTI_LABEL_SUFFIXES.map((suffix) => Object.freeze(suffix.split("."))).sort(
    (left, right) => right.length - left.length || left.join(".").localeCompare(right.join(".")),
  ),
);
const SOCIAL_DOMAINS = [
  "facebook.com",
  "instagram.com",
  "linkedin.com",
  "pinterest.com",
  "reddit.com",
  "t.co",
  "t.me",
  "telegram.me",
  "tiktok.com",
  "twitter.com",
  "wa.me",
  "whatsapp.com",
  "x.com",
  "youtube.com",
];
const EMPTY_REF_SCHEMES = ["javascript:", "mailto:", "tel:", "data:", "about:"];
const BANK_TERMS = ["bank", "banking"];
const PAY_TERMS = ["pay", "payment", "paypal", "checkout"];
const CRYPTO_TERMS = ["crypto", "bitcoin", "btc", "ethereum", "wallet", "blockchain", "usdt"];
const IPV4_COMPONENT_RE = /^(0|[1-9][0-9]{0,2})$/;
const IPV6_GROUP_RE = /^[0-9A-Fa-f]{1,4}$/;
const MODEL_CACHE = new WeakMap();

export function fnv1a32(text) {
  let value = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(asText(text))) {
    value ^= byte;
    value = Math.imul(value, 0x01000193) >>> 0;
  }
  return value >>> 0;
}

export function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

export function vectorizeRecord(model, record) {
  const artifact = validateModel(model);
  const source = isRecord(record) ? record : {};
  const seen = new Set();
  for (const [namespace, raw] of [
    ["url", source.url],
    ["title", source.title ?? ""],
  ]) {
    const characters = Array.from(asText(raw).normalize("NFKC").toLowerCase());
    const prefix = `${namespace}:`;
    for (let size = artifact.hash.ngram_min; size <= artifact.hash.ngram_max; size += 1) {
      for (let start = 0; start + size <= characters.length; start += 1) {
        seen.add(fnv1a32(`${prefix}${characters.slice(start, start + size).join("")}`) % artifact.hash.dimension);
      }
    }
  }

  const numericValues = [];
  if (artifact.include_dom) {
    const dom = isRecord(source.dom) ? source.dom : {};
    const urlValues = urlNumericFeatures(source.url);
    for (const name of artifact.numeric_features) {
      const raw = URL_NUMERIC_FEATURE_SET.has(name)
        ? urlValues[name]
        : DOM_NUMERIC_FEATURE_SET.has(name)
          ? dom[name]
          : 0;
      numericValues.push(transformNumeric(name, raw));
    }
  }

  return Object.freeze({
    hashIndices: Object.freeze(Array.from(seen).sort((left, right) => left - right)),
    numericValues: Object.freeze(numericValues),
  });
}

export function scoreRecord(model, record) {
  const artifact = validateModel(model);
  const vector = vectorizeRecord(artifact, record);
  let logit = artifact.intercept;

  for (const index of vector.hashIndices) {
    logit += artifact.weights[index];
  }
  if (artifact.include_dom) {
    for (let offset = 0; offset < artifact.numeric_features.length; offset += 1) {
      logit += artifact.weights[artifact.hash.dimension + offset] * vector.numericValues[offset];
    }
  }

  const probability = sigmoid(logit);
  return Object.freeze({
    probability,
    label: Number(probability >= artifact.threshold),
    featureVersion: artifact.feature_version,
  });
}

export function extractDomFeatures(documentRef, locationHref, model = undefined) {
  const featureNames = domFeatureNamesForModel(model);
  const values = Object.fromEntries(featureNames.map((name) => [name, 0]));
  const doc = documentRef;
  if (!doc || typeof doc.querySelectorAll !== "function") {
    return Object.freeze(values);
  }

  const htmlText = asText(doc.documentElement?.outerHTML ?? "");
  const lines = htmlText ? htmlText.split(/\r\n|\r|\n/) : [];
  const visibleText = normalizeVisibleText(doc);
  const referenceCounts = referenceCountsForDocument(doc, asText(locationHref));

  assignIfAllowed(values, "html_line_count", lines.length);
  assignIfAllowed(values, "largest_line_length", lines.reduce((maximum, line) => Math.max(maximum, Array.from(line).length), 0));
  assignIfAllowed(values, "iframe_count", queryCount(doc, "iframe"));
  assignIfAllowed(values, "image_count", queryCount(doc, "img"));
  assignIfAllowed(values, "css_count", cssCount(doc));
  assignIfAllowed(values, "javascript_count", queryCount(doc, "script"));
  assignIfAllowed(values, "self_ref_count", referenceCounts.self);
  assignIfAllowed(values, "empty_ref_count", referenceCounts.empty);
  assignIfAllowed(values, "external_ref_count", referenceCounts.external);
  assignIfAllowed(values, "has_title", titleText(doc) ? 1 : 0);
  assignIfAllowed(values, "has_favicon", hasFavicon(doc) ? 1 : 0);
  assignIfAllowed(values, "is_responsive", queryCount(doc, "meta[name]") && hasMetaName(doc, "viewport") ? 1 : 0);
  assignIfAllowed(values, "has_description", hasDescription(doc) ? 1 : 0);
  assignIfAllowed(values, "external_form_submit", hasExternalFormSubmit(doc, asText(locationHref)) ? 1 : 0);
  assignIfAllowed(values, "has_social", hasSocial(doc) ? 1 : 0);
  assignIfAllowed(values, "has_submit_button", hasSubmitButton(doc) ? 1 : 0);
  assignIfAllowed(values, "has_hidden_fields", hasInputType(doc, "hidden") ? 1 : 0);
  assignIfAllowed(values, "has_password_field", hasInputType(doc, "password") ? 1 : 0);
  assignIfAllowed(values, "mentions_bank", containsTerm(visibleText, BANK_TERMS) ? 1 : 0);
  assignIfAllowed(values, "mentions_pay", containsTerm(visibleText, PAY_TERMS) ? 1 : 0);
  assignIfAllowed(values, "mentions_crypto", containsTerm(visibleText, CRYPTO_TERMS) ? 1 : 0);
  assignIfAllowed(values, "has_copyright", visibleText.includes("copyright") || visibleText.includes("\u00a9") || visibleText.includes("(c)") ? 1 : 0);

  return Object.freeze(values);
}

function validateModel(model) {
  if (!isRecord(model)) throw new Error("model artifact must be an object");
  const cached = MODEL_CACHE.get(model);
  if (cached) return cached;

  assertExactKeys(model, TOP_LEVEL_KEYS, "model schema");
  if (model.schema !== SCHEMA) throw new Error("unsupported model schema");
  if (model.feature_version !== FEATURE_VERSION) throw new Error("unsupported feature version");
  if (typeof model.include_dom !== "boolean") throw new Error("include_dom must be boolean");
  const hash = validateHash(model.hash);
  const numericFeatures = validateNumericFeatures(model.include_dom, model.numeric_features);
  const weights = validateWeights(model.weights, hash.dimension + numericFeatures.length);
  const intercept = model.intercept;
  const threshold = model.threshold;
  if (!isFiniteNumber(intercept)) throw new Error("intercept must be finite");
  if (!isFiniteNumber(threshold) || threshold < 0 || threshold > 1) {
    throw new Error("threshold must be finite and between 0 and 1");
  }
  const metadata = sanitizeMetadata(model.metadata);
  const artifact = Object.freeze({
    schema: SCHEMA,
    feature_version: FEATURE_VERSION,
    hash,
    include_dom: model.include_dom,
    numeric_features: numericFeatures,
    weights,
    intercept,
    threshold,
    metadata,
  });

  MODEL_CACHE.set(model, artifact);
  MODEL_CACHE.set(artifact, artifact);
  return artifact;
}

function validateHash(value) {
  if (!isRecord(value)) throw new Error("hash must be an object");
  assertExactKeys(value, HASH_KEYS, "hash");
  if (value.name !== HASH_NAME) throw new Error("hash name mismatch");
  if (!Number.isInteger(value.dimension) || value.dimension <= 0) {
    throw new Error("hash dimension must be a positive integer");
  }
  if (value.ngram_min !== NGRAM_MIN || value.ngram_max !== NGRAM_MAX) {
    throw new Error("hash ngram range mismatch");
  }
  return Object.freeze({
    name: HASH_NAME,
    dimension: value.dimension,
    ngram_min: NGRAM_MIN,
    ngram_max: NGRAM_MAX,
  });
}

function validateNumericFeatures(includeDom, value) {
  if (!Array.isArray(value)) throw new Error("numeric_features must be an array");
  const expected = includeDom ? NUMERIC_FEATURES : [];
  if (value.length !== expected.length) throw new Error("numeric_features order mismatch");
  for (let index = 0; index < expected.length; index += 1) {
    if (value[index] !== expected[index]) throw new Error("numeric_features order mismatch");
  }
  return Object.freeze([...expected]);
}

function validateWeights(value, expectedWidth) {
  if (!Array.isArray(value)) throw new Error("weights must be an array");
  if (value.length !== expectedWidth) throw new Error("model dimension mismatch");
  return Object.freeze(
    value.map((item) => {
      if (!isFiniteNumber(item)) throw new Error("weights must contain only finite numbers");
      return item;
    }),
  );
}

function sanitizeMetadata(value, depth = 0) {
  if (depth > 12) throw new Error("metadata is too deeply nested");
  if (!isRecord(value)) throw new Error("metadata must be an object");
  const output = {};
  for (const [key, item] of Object.entries(value)) {
    if (UNSAFE_METADATA_KEYS.has(key) || key.includes("\u0000")) {
      throw new Error("metadata contains unsafe key");
    }
    output[key] = sanitizeMetadataValue(item, depth + 1);
  }
  return Object.freeze(output);
}

function sanitizeMetadataValue(value, depth) {
  if (depth > 12) throw new Error("metadata is too deeply nested");
  if (value === null || typeof value === "string" || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("metadata numbers must be finite");
    return value;
  }
  if (Array.isArray(value)) {
    return Object.freeze(value.map((item) => sanitizeMetadataValue(item, depth + 1)));
  }
  return sanitizeMetadata(value, depth);
}

function transformNumeric(name, raw) {
  const value = finiteOrZero(raw);
  if (COUNT_FEATURE_SET.has(name)) return Math.log1p(Math.max(0, value));
  if (RATIO_FEATURE_SET.has(name)) return Math.min(1, Math.max(0, value));
  return value ? 1 : 0;
}

function urlNumericFeatures(url) {
  const raw = asText(url);
  const parts = splitUrl(raw);
  const host = parts.hostname.toLowerCase();
  const urlLength = Array.from(raw).length;
  let letterCount = 0;
  let digitCount = 0;
  let specialCount = 0;

  for (const character of Array.from(raw)) {
    if (isLetter(character)) {
      letterCount += 1;
    }
    if (isDigit(character)) {
      digitCount += 1;
    }
    if (!isAlphaNumeric(character)) {
      specialCount += 1;
    }
  }

  const [suffix, subdomainCount] = suffixAndSubdomainCount(host);
  const denominator = urlLength || 1;
  const obfuscatedCount = (raw.match(/%[0-9A-Fa-f]{2}/g) || []).length;
  return {
    url_length: urlLength,
    domain_length: Array.from(host).length,
    tld_length: Array.from(suffix).length,
    subdomain_count: subdomainCount,
    letter_count: letterCount,
    digit_count: digitCount,
    obfuscated_count: obfuscatedCount,
    query_count: countOccurrences(raw, "?"),
    equals_count: countOccurrences(raw, "="),
    ampersand_count: countOccurrences(raw, "&"),
    special_count: specialCount,
    letter_ratio: letterCount / denominator,
    digit_ratio: digitCount / denominator,
    obfuscated_ratio: obfuscatedCount / denominator,
    special_ratio: specialCount / denominator,
    is_domain_ip: isIpAddress(host) ? 1 : 0,
    is_https: parts.scheme === "https" ? 1 : 0,
  };
}

function splitUrl(raw) {
  const text = asText(raw);
  let scheme = "";
  let rest = text;
  const schemeIndex = text.indexOf("://");
  if (schemeIndex >= 0) {
    scheme = text.slice(0, schemeIndex).toLowerCase();
    rest = text.slice(schemeIndex + 3);
  } else if (text.startsWith("//")) {
    rest = text.slice(2);
  }

  const authorityEndCandidates = ["/", "?", "#"]
    .map((marker) => rest.indexOf(marker))
    .filter((index) => index >= 0);
  const authorityEnd = authorityEndCandidates.length ? Math.min(...authorityEndCandidates) : rest.length;
  let authority = rest.slice(0, authorityEnd);
  const atIndex = authority.lastIndexOf("@");
  if (atIndex >= 0) authority = authority.slice(atIndex + 1);

  let hostname = authority;
  if (hostname.startsWith("[")) {
    const bracketIndex = hostname.indexOf("]");
    hostname = bracketIndex >= 0 ? hostname.slice(1, bracketIndex) : hostname.slice(1);
  } else {
    const colonIndex = hostname.lastIndexOf(":");
    if (colonIndex >= 0) hostname = hostname.slice(0, colonIndex);
  }
  return { scheme, hostname };
}

function suffixAndSubdomainCount(host) {
  if (!host || isIpAddress(host)) return ["", 0];
  const labels = host.split(".").filter(Boolean);
  if (labels.length <= 1) return ["", 0];

  const multiLabelSuffix = matchingFrozenMultiLabelSuffix(labels);
  if (multiLabelSuffix) {
    return [multiLabelSuffix, Math.max(0, labels.length - multiLabelSuffix.split(".").length - 1)];
  }

  const suffix = labels[labels.length - 1];
  const knownSuffix = FROZEN_SINGLE_LABEL_SUFFIXES.has(suffix);
  const subdomainCount = knownSuffix ? Math.max(0, labels.length - 2) : Math.max(0, labels.length - 1);
  return [suffix, subdomainCount];
}

function matchingFrozenMultiLabelSuffix(labels) {
  for (const suffixLabels of FROZEN_MULTI_LABEL_SUFFIX_LABELS) {
    if (
      labels.length > suffixLabels.length &&
      suffixLabels.every((label, index) => labels[labels.length - suffixLabels.length + index] === label)
    ) {
      return suffixLabels.join(".");
    }
  }
  return "";
}

function domFeatureNamesForModel(model) {
  if (model === undefined) return DOM_NUMERIC_FEATURES;
  const artifact = validateModel(model);
  return Object.freeze(artifact.numeric_features.filter((name) => DOM_NUMERIC_FEATURE_SET.has(name)));
}

function assignIfAllowed(values, name, value) {
  if (Object.hasOwn(values, name)) values[name] = Number(value) || 0;
}

function queryCount(doc, selector) {
  try {
    return doc.querySelectorAll(selector).length;
  } catch {
    return 0;
  }
}

function titleText(doc) {
  return collapseWhitespace(doc.querySelector("title")?.textContent ?? "");
}

function cssCount(doc) {
  let count = queryCount(doc, "style");
  for (const link of doc.querySelectorAll("link")) {
    const rel = asText(link.getAttribute("rel")).toLowerCase().split(/\s+/);
    if (rel.includes("stylesheet")) count += 1;
  }
  return count;
}

function hasFavicon(doc) {
  for (const link of doc.querySelectorAll("link")) {
    const rel = asText(link.getAttribute("rel")).toLowerCase();
    const href = asText(link.getAttribute("href")).toLowerCase();
    if (rel.split(/\s+/).includes("icon") || rel.includes("shortcut icon") || href.endsWith("favicon.ico")) {
      return true;
    }
  }
  return false;
}

function hasMetaName(doc, expected) {
  for (const meta of doc.querySelectorAll("meta[name]")) {
    if (asText(meta.getAttribute("name")).toLowerCase() === expected) return true;
  }
  return false;
}

function hasDescription(doc) {
  for (const meta of doc.querySelectorAll("meta[name], meta[property]")) {
    const name = asText(meta.getAttribute("name")).toLowerCase();
    const property = asText(meta.getAttribute("property")).toLowerCase();
    const content = asText(meta.getAttribute("content")).trim();
    if (content && (name === "description" || property === "description" || property === "og:description")) {
      return true;
    }
  }
  return false;
}

function hasExternalFormSubmit(doc, baseUrl) {
  for (const form of doc.querySelectorAll("form[action]")) {
    if (classifyReference(baseUrl, form.getAttribute("action")) === "external") return true;
  }
  return false;
}

function hasSocial(doc) {
  for (const node of doc.querySelectorAll("[href]")) {
    const href = asText(node.getAttribute("href")).toLowerCase();
    if (SOCIAL_DOMAINS.some((domain) => href.includes(domain))) return true;
  }
  return false;
}

function hasSubmitButton(doc) {
  for (const input of doc.querySelectorAll("input[type]")) {
    if (asText(input.getAttribute("type")).toLowerCase() === "submit") return true;
  }
  for (const button of doc.querySelectorAll("button")) {
    const type = asText(button.getAttribute("type")).toLowerCase();
    if (!type || type === "submit") return true;
  }
  return false;
}

function hasInputType(doc, expected) {
  for (const input of doc.querySelectorAll("input[type]")) {
    if (asText(input.getAttribute("type")).toLowerCase() === expected) return true;
  }
  return false;
}

function referenceCountsForDocument(doc, baseUrl) {
  const counts = { self: 0, empty: 0, external: 0 };
  for (const [selector, attribute] of [
    ["a[href]", "href"],
    ["link[href]", "href"],
    ["script[src]", "src"],
    ["img[src]", "src"],
    ["iframe[src]", "src"],
    ["form[action]", "action"],
  ]) {
    for (const node of doc.querySelectorAll(selector)) {
      counts[classifyReference(baseUrl, node.getAttribute(attribute))] += 1;
    }
  }
  return counts;
}

function classifyReference(baseUrl, reference) {
  const value = asText(reference).trim();
  const lower = value.toLowerCase();
  if (!value || value.startsWith("#") || EMPTY_REF_SCHEMES.some((scheme) => lower.startsWith(scheme))) {
    return "empty";
  }
  const target = resolveUrl(baseUrl, value);
  const targetDomain = registrableDomainFromUrl(target);
  const baseDomain = registrableDomainFromUrl(baseUrl);
  if (targetDomain && baseDomain && targetDomain === baseDomain) return "self";
  return "external";
}

function resolveUrl(baseUrl, reference) {
  const value = asText(reference).trim();
  if (value.includes("://") || value.startsWith("//")) return value;
  return asText(baseUrl);
}

function registrableDomainFromUrl(url) {
  const host = splitUrl(asText(url)).hostname.toLowerCase();
  if (!host || isIpAddress(host)) return host;
  const [suffix, subdomainCount] = suffixAndSubdomainCount(host);
  const labels = host.split(".").filter(Boolean);
  if (!suffix) return host;
  const suffixLabels = suffix.split(".").length;
  const domainStart = Math.max(0, labels.length - suffixLabels - 1);
  if (subdomainCount >= labels.length - 1) return host;
  return labels.slice(domainStart).join(".");
}

function normalizeVisibleText(doc) {
  const root = doc.body ?? doc.documentElement;
  return asText(root?.textContent ?? "").normalize("NFKC").toLowerCase();
}

function collapseWhitespace(value) {
  return asText(value).split(/\s+/).filter(Boolean).join(" ");
}

function containsTerm(text, terms) {
  return terms.some((term) => text.includes(term));
}

function assertExactKeys(value, expected, label) {
  const keys = Object.keys(value).sort();
  if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
    throw new Error(`${label} mismatch`);
  }
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function finiteOrZero(value) {
  const number = Number(value || 0);
  return Number.isFinite(number) ? number : 0;
}

function asText(value) {
  if (value == null) return "";
  if (typeof value === "number" && !Number.isFinite(value)) return "";
  return String(value);
}

function countOccurrences(text, marker) {
  return text.split(marker).length - 1;
}

function isLetter(character) {
  return /^\p{L}$/u.test(character);
}

function isDigit(character) {
  return /^\p{N}$/u.test(character);
}

function isAlphaNumeric(character) {
  return /^[\p{L}\p{N}]$/u.test(character);
}

function isIpAddress(host) {
  const text = asText(host);
  return isIpv4Address(text) || isIpv6Address(text);
}

function isIpv4Address(host) {
  const pieces = host.split(".");
  if (pieces.length !== 4) return false;
  return pieces.every((piece) => IPV4_COMPONENT_RE.test(piece) && Number(piece) <= 255);
}

function isIpv6Address(host) {
  if (!host.includes(":")) return false;
  if (/[^0-9A-Fa-f:.]/.test(host)) return false;

  let text = host;
  if (text.includes(".")) {
    const lastColon = text.lastIndexOf(":");
    if (lastColon < 0 || !isIpv4Address(text.slice(lastColon + 1))) return false;
    text = `${text.slice(0, lastColon)}:0:0`;
  }

  if (countOccurrences(text, "::") > 1) return false;
  if (text.includes("::")) {
    const [left, right] = text.split("::");
    const leftGroups = left ? left.split(":") : [];
    const rightGroups = right ? right.split(":") : [];
    if (!areIpv6Groups(leftGroups) || !areIpv6Groups(rightGroups)) return false;
    return leftGroups.length + rightGroups.length < 8;
  }

  const groups = text.split(":");
  return areIpv6Groups(groups) && groups.length === 8;
}

function areIpv6Groups(groups) {
  return groups.every((group) => IPV6_GROUP_RE.test(group));
}

export { FEATURE_VERSION, DOM_NUMERIC_FEATURES, NUMERIC_FEATURES };
