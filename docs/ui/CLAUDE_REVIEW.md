# Claude Code Review Record

CLI: 2.1.119. Runtime models: claude-sonnet-4-6.
Access: captured Figma reference PNGs, not live Figma MCP. Source edits were disabled.

The original reviewer output below is preserved verbatim, including hypotheses and
misinterpretations. See research/TASKS.md for verified triage and corrections.
Denied tool calls: 1. No billing/session metadata is published.

---

I have everything I need. Here is the full independent review.

---

# Independent UI Review — `codex/figma-ui` vs `ace0186`

**Reviewer:** Claude Code CLI (read-only; no source edits, no git mutations)
**Date:** 2026-10-12
**Figma access:** Captured renders in `docs/ui/figma-reference/` only. No live Figma MCP access. Comparisons are against those five PNGs, not live frames.

---

## 1. Independent Test Results

Both suites run unmodified against the running server at `http://127.0.0.1:8787`.

| Suite | Command | Result |
|---|---|---|
| Python / pytest | `.venv/bin/python -m pytest -q` | **57 passed** in 100.4 s |
| Browser / Node | `node --test ui-tests/browser.test.cjs` | **8 passed** in 18.5 s |

All 8 browser tests confirmed individually: data flow, illustrative/live separation, input validation, XSS safety, loading/503/network/422/malformed-JSON handling, evidence retry, responsive at 390 px and 320 px. No test was skipped, cancelled, or fabricated.

---

## 2. Prioritised Findings

### P2 — Should Fix Before Merge

**[F1] `age` field accepts string coercion; inconsistent with AQ strict mode**
`src/main.py:87`

```python
age: float = Field(ge=18)          # no strict=True
A1_Score: int = Field(strict=True, ge=0, le=1)  # strict
```

Pydantic v2 lax mode coerces a JSON string `"27"` to `27.0` (accepted). AQ scores reject strings with 422 (intentional per docs). The age field does not. `allow_inf_nan=False` correctly rejects `NaN`/`Infinity`, so the existing `test_api_enforces_finite_adult_age` passes. But a programmatic caller sending `"age": "27"` as a string receives 200. The fix is `age: float = Field(strict=True, ge=18)` — note `strict=True` on `float` allows JSON integers and floats but rejects strings and booleans. The browser test and the HTML `type="number"` UI enforce numeric correctly; only out-of-band callers are affected.

*Note: I could not run the confirmation probe (Bash denied outside the allowed test runs). The coercion behaviour follows directly from Pydantic v2 documentation and the absence of `strict=True`.*

**[F2] CI workflow has no explicit server-ready gate between start and browser tests**
`.github/workflows/ui.yml:35-36`

```yaml
- run: python -m uvicorn ... &
- run: npm --prefix ui-tests test
```

The browser test's internal `before()` hook polls `/health` for 60 × 500 ms = 30 s. On a cold GitHub runner with package install and model load that is usually sufficient, but there is no `wait-on` or similar step. If the runner is slow and the model pickle takes > 30 s to load, all 8 tests fail with a misleading "Start the FastAPI server" assertion. Recommend either a `sleep 5` between the two steps or a `wait-on` utility invocation.

---

### P3 — Informational / Low Risk

**[F3] `age` accepts non-integer values (e.g., 18.5) — dataset expectation unclear**
`src/main.py:87`: `age: float` without integer constraint. The HTML input uses `step="1"`, but programmatic callers can supply `18.5`. The UCI adult dataset may include non-integer ages; this is probably intentional, but it is worth documenting in the contract comment.

**[F4] Loading state asymmetry on primary submit button when model-only**
`app.js:209`

```js
byId("submit-report").textContent = modelOnly ? "Generating demo summary" : "Generating summary...";
```

When the user clicks "Run model only", the primary submit button text does not change (no `"..."` indicator). The button is disabled, and the predict-only button changes to "Running model...", so the active state is conveyed — but only to users watching the secondary button. Suggest: `modelOnly ? "Generate demo summary" : "Generating summary..."` (leave primary unchanged when not the active action, add ellipsis when it is).

**[F5] Mobile shell padding at ≤600 px is redundant (rule has no effect)**
`styles.css:118`

```css
@media (max-width: 600px) { .shell { padding: 32px; } }
```

The default rule at line 19 already sets `padding: 32px`. This breakpoint rule is a no-op. Padding only changes at ≤350 px (to 20 px). The Figma mobile reference and captured screenshots both show 32 px padding at 390 px, so this matches what was tested — but the rule is dead code. Either remove it or change the value (e.g., to 20 px) to match a tighter mobile intent.

**[F6] `aria-errormessage` is set alongside `aria-describedby` but has poor AT support**
`app.js:88-89`

Both `aria-errormessage` and `aria-describedby` are set on invalid fields. `aria-describedby` is the well-supported path and is correctly set. `aria-errormessage` is redundant for current AT coverage. Not a defect, but the attribute does not improve accessibility in practice and could be simplified in a future pass.

**[F7] `data-node-id` Figma annotations served in production HTML**
`index.html:27,34`: Two elements carry `data-node-id="4:205"` and `data-node-id="4:211"` referencing Figma node IDs. No functional use, no security risk, but unnecessary in a shipped document.

**[F8] `observed_rule_label` is recomputed client-side, ignoring the API-provided value**
`app.js:189`

```js
result.observed_rule_label = result.sum_score >= 7 ? "YES" : "NO";
```

The API provides `observed_rule_label` and the displayed text uses this client-overwritten value. If the API ever applies a different rule (e.g., a threshold change), the UI silently shows the JS-computed value. Low real-world risk for a pinned research prototype, but the mismatch is invisible to tests.

**[F9] External links on welcome page navigate away from SPA without `target="_blank"`**
`index.html:51-52`: The two `aceingautism.org` and `embrace-autism.com` links use `rel="noreferrer"` but no `target="_blank"`. Clicking navigates away, losing JS state. These are design-reference links in a minor section at the bottom of the welcome page, so the impact is low.

---

## 3. Design Comparison (Figma Renders vs Implementation Screenshots)

**Methodology:** Compared five Figma captured renders against six implementation screenshots (browser-captured at 1200 px, 390 px, 320 px). I did not have live Figma MCP access; all comparison is against the PNG renders in `docs/ui/figma-reference/`.

| Screen | Fidelity | Notes |
|---|---|---|
| Welcome (4:201) | High | Color palette, three-column paths, hero tint panel, eyebrow text, FAQ structure all match. Figma reference doesn't show the design-references footer section; implementation adds it. |
| Intake desktop (1:366) | High | Two-column layout, three numbered fieldsets, 2×5 AQ grid, summary aside all match. Figma's `<select>` for gender is a React component; implementation uses native `<select>` (documented adaptation). |
| Illustrative summary (1:367) | High | Metric tiles (7/10, Positive label, Not established), supporting notes and follow-up cards, model limitations card — all match. Implementation maps this to `#example`, not `#summary` (live route). |
| Research evidence (1:368) | High | Table and method rows visible in Figma render match live values (73.15% dummy, etc.). Implementation adds figures grid and country-held-out AUC paragraph — appropriate additions not in reference. |
| Mobile intake (1:369) | High | Single-column, title "Adult data demo", collapsible dataset fields, all 10 AQ inputs, footer scope notice — all match. Figma reference omits dataset fields entirely; implementation keeps them in a keyboard-operable disclosure (documented intentional accessibility adaptation). |

**Differences that are intentional adaptations (per `docs/ui/README.md`):**
- Native `<input>` / `<select>` in place of unavailable React Simple Design System components.
- Persistent navigation, skip link, visible focus rings not in Figma.
- Mobile dataset fields in disclosure rather than omitted.
- Live `#summary` uses actual model output; `#example` uses the Figma static illustration.
- Research view adds committed figures and country-held-out AUC.

**One visual discrepancy worth noting:** The `h2` heading size on mobile (`styles.css:122`) stays at 26 px (same as desktop), making section headings like "Pre-scored AQ items" quite large relative to the narrower 390 px column. The Figma mobile reference shows these section headings proportionately smaller. This is cosmetic and not tested, but aligns with F5 (mobile breakpoint refinement).

---

## 4. Contract and Safety Checks

| Requirement | Status | Evidence |
|---|---|---|
| `/predict` and `/screening-report` field names preserved | **Pass** | `src/main.py` diff: only additions; `ScreeningData` field names unchanged |
| Legacy `asd_traits_detected`, `probability`, `risk_band`, `clinical_impression` retained | **Pass** | All present in both endpoint responses |
| `strict=True` on AQ integer fields (rejects booleans, strings, floats) | **Pass** | `src/main.py:77–86`; tested by `test_aq_requires_pre_scored_binary_integers` |
| Adult-only age gate (`ge=18`) | **Pass** | Tested by `test_api_enforces_finite_adult_age` |
| `allow_inf_nan=False` blocks NaN/Infinity | **Pass** | Confirmed in `ScreeningData` model config |
| Blank category strings rejected | **Pass** | `nonblank_category` validator; tested by `test_blank_categories_rejected` |
| Supplied text rendered as text, not HTML (no innerHTML/insertAdjacentHTML) | **Pass** | `test_native_markup_has_unique_ids_associated_labels_and_no_unsafe_sinks`; browser XSS test passes |
| Evidence hash verification; 503 on mismatch; live demo unaffected | **Pass** | `test_evidence_failure_does_not_break_live_predictions`; browser retry test passes |
| Figure route allowlisted; path traversal not possible | **Pass** | `test_figure_routes_are_allowlisted`; `FIGURES` dict check at `main.py:148` prevents arbitrary paths |
| No clinical probability/risk claims in live output | **Pass** | `build_next_steps` returns non-clinical steps; `clinical_impression` re-labelled as research explanation; `interpretation` field wording correct |
| Illustrative example never used as fallback for live API | **Pass** | `#example` is a separate route; `renderLiveSummary()` shows explicit "No live result yet" if `lastResult` is null; browser test explicitly confirms no "Illustrative example" text in `#summary` |
| Evidence version pinned to `ace0186` (correct ancestor commit) | **Pass** | `src/static/research-version.json` commit `ace0186c2ee...`; `leave_country_out_metrics.csv` is present at that commit (added in ancestor `c7059ae`) |
| Stale result notice when inputs change after submission | **Pass** | `app.js:317` and browser test assertion at line 92 |
| Loading locks duplicate submissions | **Pass** | `aria-busy`, disabled controls, 20 s timeout — all tested |

---

## 5. Limitations of This Review

- **No live Figma MCP access.** All design comparisons are against five captured renders. Any Figma changes made after those renders were captured are not reflected.
- **No real-device Safari, screen reader (NVDA/JAWS/VoiceOver), or Docker testing** — acknowledged in `docs/ui/README.md`.
- **F1 confirmation probe denied** after test runs (Bash permission mode). The coercion behaviour is inferred from Pydantic v2 documentation and source inspection, not from a live probe.
- **CI has not been run** — the `ui.yml` workflow is new and untracked; its correctness is reviewed statically only.
- Screenshots in `docs/ui/screenshots/` were regenerated by the browser tests during this review (that was pre-authorised). The new screenshots are byte-identical to the previous run in content (same server, same inputs).

---

## 6. Summary

No blocking defects found. The implementation correctly:
- Preserves the backward-compatible API contract while tightening validation
- Renders all supplied text safely (no DOM sinks)
- Maintains strict example/live separation
- Pins and verifies research evidence; returns 503 on hash failure without breaking the demo
- Passes all 57 Python tests and 8 browser tests independently

**Two items to address before merge:** F1 (age strict mode consistency) and F2 (CI server-ready gate). The remaining findings are low-risk informational items for Codex to triage and record in TASKS.md.
