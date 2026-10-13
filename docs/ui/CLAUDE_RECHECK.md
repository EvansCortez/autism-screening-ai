# Claude Code Review Record

CLI: 2.1.119. Runtime models: claude-sonnet-4-6.
Access: captured Figma reference PNGs, not live Figma MCP. Source edits were disabled.

The original reviewer output below is preserved verbatim, including hypotheses and
misinterpretations. See research/TASKS.md for verified triage and corrections.
Denied tool calls: 0. No billing/session metadata is published.

---

All three test runs complete. Here is the focused re-check result.

---

## Re-check results

### Test outcomes (actual)

| Command | Actual |
|---|---|
| `pytest tests/test_main.py tests/test_ui.py -q` | **28 passed** in 2.37 s |
| `node --test ui-tests/browser.test.cjs` | **10 passed** in 14.6 s |
| `wait_for_server.py` | exit 0 — "FastAPI app and saved model are ready." |

---

### Fix verification

**F1 — age strict mode** (`src/main.py:87`)
`age: float = Field(strict=True, ge=18)` — confirmed present. Pydantic v2 `strict=True` on `float` accepts JSON integers and floats, rejects strings and booleans. The original lax field is gone.

**F8 — observed_rule_label** (`src/static/app.js:187–192`)
`normalizeResult` computes `ruleLabel` locally, then line 189 throws `"invalid-response"` if the API-supplied `observed_rule_label` disagrees with the score-derived value. Line 192 uses `?? ruleLabel` only when the field is absent — the API value is never silently overwritten. Browser test "contradictory API rule label fails visibly instead of being silently overwritten" passed.

**F4 — primary button wording** (`app.js:212`)
`byId("submit-report").textContent = modelOnly ? "Generate demo summary" : "Generating summary..."` — when model-only is active, the primary button stays at its idle label ("Generate demo summary"); it only shows the ellipsis form when it is the active action.

**F7 — Figma data attributes** (`index.html`)
`grep data-node-id` returns zero matches. Attributes removed.

**F9 — external reference links** (`index.html:51`)
Both links carry `target="_blank" rel="noreferrer noopener"` with visible "(new tab)" label text. State-loss on click is resolved.

**F2 — CI readiness** (`.github/workflows/ui.yml:38`, `ui-tests/wait_for_server.py`)
The workflow has a distinct "Wait for app and saved model" step calling `wait_for_server.py` between the app start and `npm test`. The script polls `/health`, requires `model_loaded: true`, enforces a 60 s hard deadline, and prints a diagnostic on timeout. Server log is uploaded as an artifact on failure only. The browser test's internal polling and the CI gate are now separate layers; the earlier concern that only the internal hook existed is resolved.

**F6 — aria helper association** (`app.js`)
Both `aria-errormessage` and `aria-describedby` are retained with `describedby` as the well-supported fallback — matches the documented decision in TASKS.md.

**Figma context** (no code change warranted)
TASKS.md records that the original reviewer's suggestion to reduce mobile `h2` to something smaller than 26 px was not supported by the Figma design context: nodes 2:470/2:495 specify 26 px. The welcome node 4:237 design-references section is present in the implementation. Both corrections are recorded; no visual change is needed.

---

### Residual findings

**None.** Every F1–F9 item is either confirmed fixed, confirmed intentionally retained with documented rationale, or classified as a non-defect per the Figma design spec. The test suite expanded from 8 to 10 browser tests post-fix (the new mismatch test and one additional responsive test), and all 10 pass.
