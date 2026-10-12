# Record of AI assistance

Purpose: a factual log of what AI tools did on this project, what a human decided, and how
AI output was checked. It is a working record, not a disclosure statement. Check your
university's and the target venue's policy on AI use before writing the paper's disclosure.

Status: **partial.** Claude Code's section is written from this conversation. Codex's and the
project lead's sections must be completed by them; I only list what is observable in the
repository and in the project lead's instructions. Do not treat the Codex section as
authoritative until Codex has confirmed it.

## Tools

| Tool | Model identifiers seen in this work | Role |
|---|---|---|
| Claude Code (Anthropic) | Sonnet 5 (commit trailer 2026-09-17); Sonnet 5.5 (session setting 2026-10-11) | Review, literature verification, statistical review, independent reproduction, coordination documents |
| Codex (OpenAI) | not recorded here | Implementation of experiments, pinned environment, one-command reproduction, explanatory analyses (per the project lead) |

## Claude Code activity

| Date | What it did | Who decided / approved | How it was checked |
|---|---|---|---|
| 2026-09-17 | Proposed improvements; added cross-validation and a baseline to `src/train.py`, `tests/test_train.py`, a CI workflow and a Dockerfile (commit `74f03d1`, trailer "Co-Authored-By: Claude Sonnet 5") | Project lead chose the three areas | Tests ran (10 passed). **Docker build was never run** (Docker not installed). |
| 2026-09-17 to 2026-10-11 | Investigated why accuracy was 100%: found the label equals `sum_score >= 7` on all 704 rows; ran ablations | Project lead asked for the review | Scripts run against the data; later reproduced in a clean environment |
| 2026-10-11 | Independent methodology review saved as `research/REVIEW.md` (13 findings) | Project lead requested | Every number came from a script I ran; scripts are in a scratch directory and **not in the repository** |
| 2026-10-11 | Literature verification (`research/RELATED_WORK.md`): downloaded and read in full Bone 2015, Rasul 2023, ASD-Bench 2026, Bengio and Grandvalet 2004, Varoquaux 2017 preprint; partially read Kapoor and Narayanan; abstracts only for Allison 2012 and Booth 2013; metadata only for others. Read both ARC AQ-10 forms. | Project lead requested | Access level recorded for each source; paywalled papers were not accessed |
| 2026-10-11 | Uncertainty review: simulation with known truth, whole-procedure bootstrap, country-cluster bootstrap, nested threshold check | Project lead requested | Re-run in a clean environment; outputs identical |
| 2026-10-11 | Independent reproduction of Codex's final analysis in two clean Python 3.14.5 environments, plus a from-scratch re-implementation of the results grid | Project lead requested | Outputs compared bit-for-bit; counts matched exactly; floats to 1e-14 |
| 2026-10-11 | Created `TASKS.md`, this file, `METHODS_GUIDE.md`, `PAPER_OUTLINE.md` | Project lead requested | Numbers traced to repository outputs or marked as scratch-derived |
| 2026-10-11 to 2026-10-12 | **Implemented tasks the project lead had assigned to Codex** (X7 figures and tables, X8 uncertainty reporting, X9 reproducibility fixes, X10 residual-signal analysis) after Codex made no repository change for 27 minutes: new `src/uncertainty.py` and `src/report_assets.py`, two small edits to `src/evaluate.py`, pinned matplotlib, extended `scripts/reproduce.py`, 11 new tests, updates to `FINDINGS.md`, `STUDY.md`, `CIRCULARITY.md`, README. The claim was logged in `TASKS.md` before work started. | **Not a decision of the project lead.** It departs from the assigned division of labor; the project lead should review and may reject it. | Full pipeline regenerated in a fresh Python 3.14.5 environment (34 tests pass); existing result tables unchanged; every number added to `FINDINGS.md` checked by script against the outputs. **The implementer and the verifier are the same agent, so this is not independent verification.** |
| 2026-10-12 | Committed the work on branch `research/label-circularity`, pushed it, and opened a pull request (merge left to the project lead) | Project lead requested | Clean-clone reproduction of the committed branch (see `TASKS.md` C5) |

## Codex activity (to be completed by Codex)

Observed in the repository, attributed to Codex by the project lead's description. Codex should
confirm, add dates, and summarize its prompts and the decisions it made.

- `src/evaluate.py`, `src/circularity_audit.py`, `scripts/reproduce.py`, `tests/test_evaluate.py`,
  `tests/test_circularity_audit.py`
- `notebooks/02_*` (now archived), `notebooks/03_reproducible_evaluation.ipynb`,
  `notebooks/04_label_circularity_audit.ipynb`
- `requirements.lock`, `.python-version`, `research/protocol.json`, `research/datasets/*.json`
- `research/STUDY.md`, `research/FINDINGS.md`, `research/CIRCULARITY.md`, `research/README.md`
- `research/evaluation/`, `research/circularity/`, `research/sources/` (UCI archives), README rewrite

Not yet recorded: Codex's model name and version, session dates, which of its design choices were
the project lead's instructions and which were its own.

## Project lead decisions (to be completed by the project lead)

Record, in your own words: the research question and who proposed it; the instruction to treat
results as screening-label agreement only; decisions on dataset choice; what you verified
yourself; which AI suggestions you rejected or changed; and what you wrote without AI help.

## Errors made by AI during this project, and their correction

Recording these is part of being able to defend the work.

| Date | Error | Correction |
|---|---|---|
| 2026-09-17 | I wrote in the README and commit message that the perfect score was "by the AQ-10 tool's design, not a code bug". I had not verified this. | Corrected on 2026-10-11 after the data showed the label equals `sum_score >= 7`, which differs from the published cut-off of 6. `train.py:47` re-adds the column it drops. |
| 2026-10-11 | I recommended exact binomial intervals "wherever errors are zero" | Narrowed after testing the assumptions (`REVIEW.md` F6): suitable for a single holdout or a rule fixed in advance, not for pooled cross-validation predictions |
| 2026-10-11 | I suggested the depth cap of 4 explained weak tree results | A follow-up showed leaf size also matters and neither alone reaches 1.0; the hypothesis was incomplete |
| 2026-10-11 | My sibling-dataset script had a download bug; I used a separate ad hoc command | Re-run from a saved script; the result is unchanged |
| 2026-10-11 | During testing I briefly modified, then restored, a data file in a scratch copy, and left a temp file in `/tmp` | Restored and deleted; the repository was never touched |
| 2026-10-11 | I reported the repeat-to-repeat SD as "0.41 and 0.58 of the binomial standard error" (in `REVIEW.md`, `METHODS_GUIDE.md`, `TASKS.md`). My scratch script built folds from single rows instead of the study's matching-predictor groups. | Found on 2026-10-12 when comparing with the in-repo output. Correct values: 0.49 and 1.00 of the binomial SE; 0.34 and 0.64 of the whole-procedure bootstrap SE. All three documents corrected. |
| 2026-10-12 | I wrote bootstrap standard errors truncated instead of rounded (0.0175 and 0.0191; correct 0.0176 and 0.0192) in `FINDINGS.md` | Caught by the claims-checking script before commit; corrected |
| 2026-10-12 | A figure title claimed that removing the sum column "changes little", and another claimed a tree "cannot" find the boundary from items alone | Both over-claimed (forests and trees do change; the tree result depends on leaf size). Retitled neutrally before the figures were committed. |

## What AI did not do

Collect data; contact the dataset authors; obtain clinical validation; read paywalled papers
(Allison 2012, Booth 2013, Thabtah 2017 and 2018, Kaufman 2012, Nadeau and Bengio 2003); run
the Docker image; test on Linux; verify claims about NICE guideline CG142 (the page returned
HTTP 403).
