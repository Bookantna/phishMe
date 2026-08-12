import { DOM_NUMERIC_FEATURES, NUMERIC_FEATURES } from "../../../web/scorer.js";

export function createTestModel({ passwordWeight = 10, intercept = -5 } = {}) {
  const weights = Array(1 + NUMERIC_FEATURES.length).fill(0);
  const passwordIndex = NUMERIC_FEATURES.indexOf("has_password_field");
  if (passwordIndex < 0 || !DOM_NUMERIC_FEATURES.includes("has_password_field")) {
    throw new Error("scorer contract is missing has_password_field");
  }
  weights[1 + passwordIndex] = passwordWeight;
  return {
    schema: "phishme-model-v1",
    feature_version: "phishme-features-v1",
    hash: { name: "fnv1a-32", dimension: 1, ngram_min: 3, ngram_max: 5 },
    include_dom: true,
    numeric_features: [...NUMERIC_FEATURES],
    weights,
    intercept,
    threshold: 0.5,
    metadata: { dataset: "test-fixture", variant: "linear" },
  };
}
