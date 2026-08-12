export const RESULT_SCHEMA = "phishme-extension-result-v1";

const BADGES = Object.freeze({
  "possible-phishing": Object.freeze({
    text: "!",
    color: "#b42318",
    title: "phishMe: possible phishing detected",
  }),
  "no-signal": Object.freeze({
    text: "OK",
    color: "#16794b",
    title: "phishMe: no phishing signal detected",
  }),
  unavailable: Object.freeze({
    text: "",
    color: "#667085",
    title: "phishMe: scan unavailable",
  }),
});

export function createDetectionResult({ score, threshold, url, scannedAt, metadata, sha256 }) {
  if (score === null || typeof score !== "object") throw new Error("score must be an object");
  if (score.label !== 0 && score.label !== 1) throw new Error("label must be 0 or 1");
  requireProbability(score.probability, "probability");
  requireProbability(threshold, "threshold");
  if (typeof score.featureVersion !== "string" || !score.featureVersion) {
    throw new Error("feature version must be text");
  }
  if (typeof url !== "string" || typeof scannedAt !== "string") throw new Error("result text fields are invalid");
  if (typeof sha256 !== "string" || !/^[0-9a-f]{64}$/.test(sha256)) throw new Error("model sha256 is invalid");

  const model = Object.freeze({
    dataset: safeMetadataText(metadata, "dataset"),
    variant: safeMetadataText(metadata, "variant"),
    sha256,
  });
  return Object.freeze({
    schema: RESULT_SCHEMA,
    status: score.label === 1 ? "possible-phishing" : "no-signal",
    label: score.label,
    probability: score.probability,
    threshold,
    url,
    scannedAt,
    featureVersion: score.featureVersion,
    model,
    error: null,
  });
}

export function createUnavailableResult(url, error, scannedAt = new Date().toISOString()) {
  return Object.freeze({
    schema: RESULT_SCHEMA,
    status: "unavailable",
    label: null,
    probability: null,
    threshold: null,
    url: typeof url === "string" ? url : "",
    scannedAt: typeof scannedAt === "string" ? scannedAt : "",
    featureVersion: null,
    model: null,
    error: String(error || "Scan unavailable").slice(0, 160),
  });
}

export function badgeForResult(result) {
  const badge = BADGES[result?.status] ?? BADGES.unavailable;
  return { ...badge };
}

function requireProbability(value, label) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 1) {
    throw new Error(`${label} must be finite and between 0 and 1`);
  }
}

function safeMetadataText(metadata, key) {
  if (metadata === null || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error("model metadata is invalid");
  }
  const value = metadata[key] ?? "unknown";
  if (typeof value !== "string" || !value || value.length > 80) {
    throw new Error("model metadata must contain bounded text values");
  }
  return value;
}
