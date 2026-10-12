# Paper outline: evidence map

Working title (tentative): *Reconstructing a score threshold: what machine-learning accuracy on
the UCI autism screening datasets measures.*

Every claim below points to the evidence for it and a status. Tags as in
[METHODS_GUIDE.md](METHODS_GUIDE.md): **[repo]** reproducible in the repository (verification
status is recorded in `TASKS.md`); **[scratch]** computed in Claude Code scratch scripts, reproduced in a clean
environment but not yet in the repository; **[source]** read from a cited source;
**[hypothesis]**; **MISSING** = evidence does not exist yet.

Framing rules: performance is *screening-label agreement*; no claim of clinically validated
diagnostic accuracy; no explanation of demographic associations; novelty is not assumed.

## Abstract (skeleton)

- Problem: ML papers on public autism-screening data report near-perfect accuracy; the label's
  origin is undocumented.
- Finding: on UCI Adult (704 rows) and Child (292 rows) the label is exactly "item total >= 7"
  (0 disagreements; the published referral cut-off is 6). **[repo]**
- Consequence: models that agree with the label have reproduced a counting rule; removing the
  total column leaves the same information in the items. **[repo]**
- Residual: with all questionnaire information removed, AUC about 0.74 (adult), mostly from
  country and ethnicity fields. **[repo]**
- Contribution: a reproducible audit protocol and its results. Not a diagnostic model.

## 1. Introduction

| Claim | Evidence | Status |
|---|---|---|
| The AQ-10 is a brief referral aid, with a published cut-point of 6 | ARC adult and child forms ("6 or above, consider referring") | **[source]** read in full |
| Case-control validations exist for the AQ-10 | Allison 2012 (sensitivity 0.88, specificity 0.91 at 6, adults); Booth 2013 | **[source]** abstracts only |
| ML papers on these datasets report near-perfect accuracy | Rasul 2023 (child 100%, adult 97.14%); ASD-Bench 2026 (10 of 17 models perfect on adults) | **[source]** full text of both preprints |
| The dataset's label definition is undocumented | UCI page and description file | **[source]** read; confirmed absent |
| Research question and three contributions (audit protocol; decomposition; open pipeline) | `STUDY.md`, `CIRCULARITY.md` | **[repo]** |

## 2. Related work

Groups, with access level in [RELATED_WORK.md](RELATED_WORK.md):

1. Instrument validation: Allison 2012, Booth 2013 (abstracts only), Baron-Cohen 2001 (metadata).
2. ML on these datasets: Rasul 2023, ASD-Bench 2026 (full text). **MISSING:** a systematic
   search; I read two of many papers.
3. Circularity and leakage: Bone 2015 (full text; ADOS and ADI-R), Kapoor and Narayanan 2023
   (partial), Kaufman 2012 (second-hand).
4. Cross-validation uncertainty: Bengio and Grandvalet 2004, Varoquaux 2018 preprint (full text);
   Nadeau and Bengio 2003 not accessed.

**MISSING before submission:** Thabtah 2017 and 2018 (the dataset's own papers; may define the
label); Kaufman 2012; Nadeau and Bengio 2003; any prior statement that these labels are
threshold-derived. Do not state novelty until these are read.

## 3. Data and provenance

| Claim | Evidence | Status |
|---|---|---|
| Adult: 704 rows, 515 NO / 189 YES, CC BY 4.0, DOI 10.24432/C5F019 | UCI page; `audit.json` | **[repo]**, **[source]** |
| Local CSV equals the original archive cell for cell | `audit.json` `source_verification` | **[repo]** |
| `result` equals the item sum on all rows | `audit.json` | **[repo]** |
| Quality issues: 95 missing ethnicity and relation, 2 missing ages, 21 flagged ages, 5 duplicate rows | `audit.json` | **[repo]** |
| Child replication: 292 rows, 151 / 141, ARC child form ages 4-11 | `research/circularity/child`, form read | **[repo]**, **[source]** |
| Item direction encoding is inferred, not documented | `STUDY.md`; wording differs across papers | **[source]**, limitation |
| Adolescent dataset (UCI 420, 104 rows) | not downloaded | **MISSING** |

## 4. Methods

1. **Label audit:** label-by-score table, purity, conditional entropy, threshold scan 0-11.
   `src/circularity_audit.py`, `src/evaluate.py` **[repo]**
2. **Models and feature sets:** majority dummy; fixed rules (6 and 7); logistic regression,
   decision tree, random forest on four feature sets (with sum, without sum, items only,
   non-questionnaire). **[repo]**
3. **Validation:** five grouped folds, seeds 42, 137, 2026; imputation, scaling, encoding and
   oversampling inside training folds; no tuning on validation labels. **[repo]**
4. **Why removing the sum removes nothing:** the sum is a linear function of the items; ablation
   to non-questionnaire features. `METHODS_GUIDE.md` section 4 **[repo]** numbers
5. **Explanatory analyses:** coefficient analysis; tree-depth comparison. **[repo]**
6. **Sensitivity cohort:** ages 18-100, first record per matching group, 677 rows. **[repo]**
7. **Uncertainty:** report counts; label every interval with what it conditions on; do not treat
   cross-validation predictions as independent. Whole-procedure and cluster-aware intervals:
   **[repo]** (`research/uncertainty/`)
8. **Reproducibility:** pinned `requirements.lock`, `scripts/reproduce.py`, input hashes.
   **[repo]**; Claude verified a clean clone of `0a5cd7c` (task C5); subsequent Codex changes
   require a new committed-checkout reproduction.

## 5. Results

| ID | Result | Evidence file | Status |
|---|---|---|---|
| R1 | Label equals `sum >= 7`; only cutoff 7 has zero disagreements; 6 disagrees on 70 rows, 8 on 57 | `label_by_score.csv`, `threshold_scan.csv` | **[repo]** |
| R2 | Accuracy: dummy 0.7315; rule 6: 0.9006; rule 7: 1.0000; logistic with sum 0.9995, without sum 0.9991, items only 1.0000 | `summary.csv` | **[repo]** |
| R3 | Trees and forests without the sum: 0.8778 and 0.9536; depth sweep plateaus at 0.8963 (leaf size 10) | `tree_depth_summary.csv` | **[repo]** |
| R3b | Leaf size also matters (unlimited depth, leaf 1: 0.9332) | scratch | **[scratch]** |
| R4 | Raw item coefficients all positive, 3.74-4.39 | `questionnaire_coefficient_summary.csv` | **[repo]** |
| R5 | Non-questionnaire: accuracy 0.6804, AUC 0.7405 (adult); child AUC 0.5834 | `summary.csv`, `circularity/*/summary.csv` | **[repo]** |
| R6 | Residual signal associated with country and ethnicity fields; exploratory label-shuffle null using the same three-repeat statistic as the observed AUC 0.7405 | `uncertainty/residual_decomposition.csv`, `permutation_null.csv`, `permutation_draws.csv` | **[repo]**; exchangeability unverified |
| R7 | Child dataset has the same threshold structure (cutoff 7; 40 disagreements for 6) | `research/circularity/child` | **[repo]** |
| R8 | Sensitivity cohort: rule 7 and items-only perfect; with-sum 0.9995, without-sum 0.9966 | `summary.csv` | **[repo]** |
| R9 | Whole-procedure bootstrap (SE ratios 1.45 and 1.56), country clustering (3.7-3.8 for non-questionnaire models), nested cut-off (7 in 15 of 15 folds) | `uncertainty/*.csv` | **[repo]** |
| R9b | Exact intervals cover 88-91% at nominal 95% in a known-truth simulation | scratch | **[scratch]** |
| R10 | Subgroup counts with Wilson intervals, descriptive only | `uncertainty/subgroup_counts.csv` | **[repo]** |
| R11 | Country-held-out residual logistic mean AUC 0.6865 versus grouped five-fold 0.7405; different validation tasks, no causal interpretation | `uncertainty/leave_country_out_metrics.csv`, `report/tables/table8_country_held_out.csv` | **[repo]**; new Codex analysis, independent review pending |

**Figures and tables (task X7):** generated in `research/report/` (six figures as SVG, PDF and PNG;
eight tables as CSV, Markdown and LaTeX): label by score; cut-off scan; accuracy by model and feature set;
item weights; tree depth; residual signal; country-held-out validation. Codex inspected the six PNG figures
independently of Claude. The project lead should review venue-specific table layout before submission.

## 6. Discussion

- Interpretation: agreement with the label measures reproduction of a counting rule.
  **Supported inference** that labels were generated from the score; **not confirmed**.
- Relation to prior work: Bone 2015 (same circularity, other instruments) **[source]**.
  ASD-Bench's flat adult feature importance is what equal item weights would produce; whether
  that explains their result is a **[hypothesis]** about data we have not examined.
- Practical checklist for reporting ML on instrument-derived labels: label purity given the
  candidate score; threshold scan; parity with the fixed rule; ablation with a model-class
  caveat; grouped validation. (Mapping to Kapoor and Narayanan's taxonomy is ours: closest to
  L2, with L1.4, L3.2, L3.3.)
- What a valid diagnostic study needs: independently assessed outcomes, participant-level
  splits, an external cohort, adequate subgroup sizes. **MISSING:** no suitable dataset
  identified.

## 7. Limitations

Undocumented label; app-collected self-selected sample; no participant identifiers; no
independently documented diagnoses or clinical external validation; inferred item encoding; same-machine
reproduction only; single dataset family; subgroup sizes too small for performance
comparisons (only White-European has 30 or more positives); no systematic literature search.

## 8. Reproducibility, data and AI-use statement

- Code, pins, protocol, hashes: `README.md`, `research/protocol.json`. Data licence: CC BY 4.0
  (attribution required).
- AI-use statement: **MISSING.** Draft from `AI_ASSISTANCE.md` after Codex and the project lead
  complete their sections, and check the venue policy.

## 9. Evidence still needed (summary)

1. Label-generation procedure (dataset author or Thabtah 2017/2018).
2. Independent review (Codex or the project lead) of the figures, tables and the new uncertainty module, which
   were implemented and checked by Claude Code only.
3. A simulation of interval coverage that lives in the repository (it is still scratch-only).
4. Clean-checkout verification of the committed branch (C5).
5. Adolescent dataset check.
6. Systematic literature search; unread papers listed in section 2.
7. Independently assessed outcome data for any diagnostic claim.
8. Linux or cross-platform reproduction.
9. AI-use disclosure per venue policy.
