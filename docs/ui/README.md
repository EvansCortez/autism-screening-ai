# ASD-Predict Figma UI

Implementation branch: `codex/figma-ui`, based on `ace0186` from the still-open research
PR #1. The UI PR targets `research/label-circularity` so its diff excludes the research work.
Merge/rebase onto main only after the lead decides the research PR's disposition.

## Design And Intentional Adaptations

[Figma file](https://www.figma.com/design/tscUSSRj9GJCheoEn8bZFu), accessed through the
Figma integration on 2026-10-12. Native reference PNGs are in `figma-reference/`:

| Screen | Node | Reference |
|---|---|---|
| Welcome | 4:201 | welcome.png |
| Desktop intake | 1:366 | intake.png |
| Illustrative summary | 1:367 | summary.png |
| Research evidence | 1:368 | research.png |
| Mobile intake | 1:369 | mobile.png |

References are review artifacts, never application assets. The Figma frames contain text,
panels and form controls, not bitmap/SVG illustration assets. Connected Simple Design System
InputField/Button snippets provide no imports and there are no corresponding components in
this vanilla repository; native HTML controls replace the unavailable React components.

Preserved design: Libre Baskerville, warm #f4efe7 background, #fffaf2 panels, #e4f1ec tint,
teal accents, 20px panel corners, 32px outer spacing, 44px desktop/26px mobile headings.
Font files are self-hosted; no third-party font request is required.

Intentional functional/accessibility differences:

- Persistent navigation, skip link, visible focus and status/error messages; darker muted
  text and input borders for contrast; minimum 48px controls.
- Mobile retains every API input, including the fields omitted in the reference, in
  keyboard-operable disclosures with explicit editable fictional defaults. Errors open
  any disclosure containing the invalid field. Optional notes remain accessible.
- A live summary uses actual model output, not the example's positive label. The observed
  >=7 rule is separately identified; it is not silently substituted for the model.
- The research view adds original committed figures and country-held-out AUC. No metrics
  come from the illustrative design, and evidence failures never display invented values.

## Contract And Scientific Wording

`/predict` and `/screening-report` preserve request field names and supported numeric
0/1 inputs. Validation is intentionally tightened: age must be finite and >=18, category
strings must be nonblank, AQ values must be JSON integers 0/1 (not booleans or strings),
and ages must be JSON numbers, not numeric strings. The API preserves fractional numeric
ages for compatibility; the UI uses whole-year increments.
The prior API accepted non-adult ages and coerced raw strings/booleans, inconsistent with
the adult pre-scored workflow. Tests cover these integration fixes.

Responses add `dataset_label`, `observed_rule_label`, `model_label_score` and interpretation.
Legacy `asd_traits_detected`, `probability`, `risk_band` and `clinical_impression` keys remain
for compatibility. The UI does not present those names as clinical findings. The legacy
impression text is now a research explanation; generated next steps no longer recommend
healthcare actions based on model scores. Notes are supplied context and never enter the model.

The trained pickle, dataset, training pipeline, experiment code, locks and all research
outputs are unchanged. Research evidence is pinned to
`ace0186c2ee3c6a8f40d0b20803acc0a59be84c7` in `src/static/research-version.json`.
`/research-evidence` computes displayed means from the committed CSV, verifies hashes for
the metrics, settings, audit, country-held-out results and six PNG figures. Missing or
changed evidence returns 503; live prediction remains separate. Figure routes are allowlisted.
To intentionally select a different committed snapshot (not to rerun experiments):

```bash
python scripts/build_ui_evidence.py --commit EXISTING_RESEARCH_COMMIT
```

The Dockerfile includes these read-only outputs; no Git runtime is required for serving.

## Run And Verify

With the existing pinned Python environment:

```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8787
```

Open `http://127.0.0.1:8787/`. Hash routes: `#welcome`, `#intake`, `#summary`, `#example`,
`#research`, `#about`. Reload/back navigation work; live results intentionally stay in
memory only and are lost on a full reload. No participant fields are stored in browser storage.

In another terminal (Node >=20 required for UI checks only):

```bash
python -m pytest -q
npm ci --prefix ui-tests --ignore-scripts
npm --prefix ui-tests run install-browser
npm --prefix ui-tests test
```

Optional: set `UI_BASE_URL` for another local port and `UI_BROWSER_PATH` for an existing
Chromium executable. The recorded local check used Chrome 154.0.8037.98 and Playwright 1.62.1;
`screenshots/validation.json` records the environment. CI uses Playwright's installed Chromium.
The browser suite writes fictional-data screenshots in `screenshots/` and tests both APIs,
validation/focus, safe supplied-text rendering, loading, 503, 422, malformed JSON, network
failure, evidence retry, stale results, and 390/320px layouts. Research image loads are checked.

Claude Code independently passed 57 Python tests and 8 browser tests on the pre-fix snapshot;
its complete review and access limitations are in [CLAUDE_REVIEW.md](CLAUDE_REVIEW.md).
Codex addressed/triaged every finding and passed the expanded 60 Python /10 browser checks
plus explicit app/model readiness. Screenshots were inspected against the Figma renders.
Claude's focused re-check independently passed28 final UI/API tests,10 browser tests and
readiness, with no residual findings ([CLAUDE_RECHECK.md](CLAUDE_RECHECK.md)).
Final evidence is recorded in `research/TASKS.md`; an implementer's checks alone are not
independent review. CI includes a model-ready gate and uploads
server logs on failure, rather than relying on an arbitrary startup sleep.

## Limitations

Not clinical validation, not a production/private health service. Real-device Safari,
screen readers and Docker build were not tested in this pass. The API still retains explicitly
deprecated response aliases. Original participant identities/label provenance are unresolved.
Prefilled categories are fictional demo inputs, not assumptions about a real participant.

User/project ownership of the design is assumed. UCI-derived figures retain their existing
research attribution; the font's OFL license is in `src/static/fonts/LICENSE`.
