export function createPageRecord({ documentRef, href, extractFeatures }) {
  const extracted = extractFeatures(documentRef, href);
  if (extracted === null || typeof extracted !== "object" || Array.isArray(extracted)) {
    throw new Error("extracted DOM features must be an object");
  }
  const dom = {};
  for (const [key, value] of Object.entries(extracted)) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error("extracted DOM features must contain finite numbers");
    }
    dom[key] = value;
  }
  return Object.freeze({
    url: String(href),
    title: String(documentRef?.title ?? ""),
    dom: Object.freeze(dom),
  });
}

export function shouldRenderWarning(result, currentUrl) {
  return Boolean(
    result &&
      result.status === "possible-phishing" &&
      typeof currentUrl === "string" &&
      result.url === currentUrl,
  );
}
