import assert from "node:assert/strict";
import test from "node:test";

import { DOM_NUMERIC_FEATURES, NUMERIC_FEATURES } from "../../../web/scorer.js";

test("scorer exports its frozen numeric feature contracts", () => {
  assert.equal(Object.isFrozen(DOM_NUMERIC_FEATURES), true);
  assert.equal(Object.isFrozen(NUMERIC_FEATURES), true);
  assert.ok(DOM_NUMERIC_FEATURES.includes("has_password_field"));
  assert.ok(NUMERIC_FEATURES.includes("is_https"));
  assert.ok(!DOM_NUMERIC_FEATURES.includes("url_length"));
});
