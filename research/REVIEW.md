# Independent review: ASD-Predict research methodology

Historical snapshot: references to preliminary paths and unpinned dependencies
describe the reviewed state. Current reproduction instructions are in STUDY.md;
preliminary artifacts now reside in `research/archive/preliminary/`. The current status of
each open item, and what has been independently verified since, is tracked in
[TASKS.md](TASKS.md); the plain-language method explanation is in
[METHODS_GUIDE.md](METHODS_GUIDE.md).

Reviewer: Claude (independent of Codex). Review date: 2026-10-11.
State reviewed: branch `add-cv-tests-ci` at commit `74f03d1`, plus Codex's uncommitted
files (`src/evaluate.py`, `tests/test_evaluate.py`, `research/`, notebooks 02 and 03).
Dataset: `data/train.csv`, sha256 `657021cd...0060d`. Source archive:
`research/sources/uci_adult_426.zip`, sha256 `c57b2986...9d6d09`.

This review did not modify the repository. The README and Codex's documents were not
treated as evidence.

## How to read this document

Each finding has three parts:

- **Confirmed evidence**: something I computed or read directly this session. The
  method is stated so you can reproduce it.
- **Possible explanations (untested)**: hypotheses that fit the evidence but were not
  tested. Do not state these as findings in the paper.
- **Actions**: a checklist. Items marked **[paper]** affect what you can claim.

Priority: **P0** blocks any clinical-sounding claim, **P1** affects methodology,
**P2** is code, scope or hygiene.

Limitation of this document: the numbers below came from ad-hoc scripts in a session
scratchpad, not from scripts stored in the repo. See F13 for the action that fixes this.

## Summary

- Codex's experimental code is sound and reproducible. I reran six experiments with my
  own implementation and got identical accuracies. I recomputed all 90 method-run
  accuracies from `predictions.csv`. My independent fold assignments matched
  `fold_assignment.csv` for seeds 42, 137 and 2026.
- The dataset label is exactly `sum_score >= 7`. Models that score about 100% are
  reconstructing a questionnaire rule. They are not predicting independently
  established diagnoses.
- The only non-circular question left in this data is what the non-questionnaire
  columns carry. That signal is mostly country and ethnicity structure.

---

## P0 findings

### F1. The target is undocumented and equals `sum_score >= 7`

**Location:** `data/train.csv` column `Class/ASD`; `research/STUDY.md`.

**Confirmed evidence**

- The UCI page (accessed 2026-10-11) does not state how `Class/ASD` was defined or
  derived, and does not say labels are clinically confirmed. Its "Additional Variable
  Information" reads "See attached variables' description file."
- The attached description (`.docx` inside the source archive) has an attribute table
  of 20 rows with no class column, and describes the screening score as "computed in an
  automated manner". The ARFF has 21 attributes including `Class/ASD`.
- Crosstab of `sum_score` against label: scores 0-6 are all `NO` (515 rows), scores
  7-10 are all `YES` (189 rows). Zero disagreements with the rule `sum_score >= 7`.
- The ARC AQ-10 form (read directly) says: "If the individual scores 6 or above,
  consider referring them for a specialist diagnostic assessment." Rule `>= 6`
  disagrees with the labels on 70 rows, all with score 6, all labeled `NO`.
- The local CSV equals the ARFF cell for cell and in row order (I re-parsed the ARFF).
  The ARFF relation name contains `NumericToNominal`, so it is a Weka-processed file,
  not a raw export.

**Possible explanations (untested)**

- The source app labeled `YES` when the score was greater than 6, which would be an
  off-by-one against the ARC form's "6 or above". I could not verify this.
- Labels came from some other process that happens to coincide with the threshold on
  all 704 rows. This is unlikely given zero exceptions, but the sources do not rule it
  out.
- Nothing in the sources supports or contradicts clinician confirmation. Absence of
  documentation is not evidence either way.

**Actions**

- [ ] **[paper]** Describe the target as "the dataset's screening label, which this
  study found to equal `sum_score >= 7`", never as a diagnosis or ground truth.
- [ ] **[paper]** Report both thresholds (dataset `>= 7`, ARC referral `>= 6`) and say
  the disagreement does not show either rule is clinically wrong.
- [ ] Add a data-audit test asserting zero disagreements with `>= 7`, so a changed
  dataset is noticed.
- [ ] Decide whether to ask the dataset author how labels were generated. The contact
  address is in the description file inside the archive. This is your decision.

### F2. Headline artifacts still present 100% as model performance

**Location:** `src/train.py:32`, `src/train.py:47`, `src/train.py:82-98`;
`models/metrics.json`; README section "Model Performance (Current Artifacts)".

**Confirmed evidence**

- `DROP_COLUMNS` removes `result` (`train.py:32`), but `train.py:47` adds `sum_score`,
  which equals `result` on all 704 rows (Codex's audit and my own check agree).
- `cross_validate_pipeline` (`train.py:82-98`) uses ungrouped `StratifiedKFold`.
- `models/metrics.json` reports accuracy 1.0, CV accuracy 1.0 and a majority baseline
  0.7305. It contains no `>= 7` rule baseline.
- Cross-validation cannot detect this problem: it measures generalization to unseen
  rows, and the label is a function of the features for every row.

**Possible explanations (untested)**

- `sum_score` may have been added as ordinary feature engineering without noticing that
  it restores the dropped column. This is a guess about intent.

**Actions**

- [ ] Add the `sum_score >= 7` rule accuracy to `metrics.json` next to the model's.
- [ ] Reorder the README so the caveat comes before the number.
- [ ] Decide whether the serving model keeps `sum_score`. Dropping it does not remove
  the circularity (see F3), so the README should say the model reproduces a rule either
  way.
- [ ] **[paper]** Do not cite the `train.py` CV result as validation.

### F3. The fitted model is an unweighted sum

**Location:** results of a descriptive fit; `src/evaluate.py` feature set
`questionnaire_only`.

**Confirmed evidence**

- Near-unregularised logistic regression (`C=1e6`) on the ten items, fit on all rows,
  gives coefficients 10.22 to 10.93 (coefficient of variation 2.2%), intercept -69.18,
  implied cut on the sum 6.5.
- Item prevalence of `1` ranges from 0.28 (A6) to 0.72 (A1). All ten items correlate
  positively with the label (0.24 to 0.64). Inter-item correlations: mean 0.187, minimum
  -0.042.

**Possible explanations (untested)**

- The positive inter-item correlations are consistent with items that are already
  scored in the trait direction, as `research/STUDY.md` assumes. They do not prove the
  encoding. The exact app encoding remains a provenance question.

**Actions**

- [ ] **[paper]** Report the coefficient vector and implied cut as direct evidence that
  the model learned the sum.
- [ ] Add a small coefficient table to the evaluation outputs so it is reproducible.
- [ ] State in Limitations that item encoding is inferred, not documented.

---

## P1 findings

### F4. Non-questionnaire signal is driven by country and ethnicity structure

**Location:** `src/evaluate.py` feature set `non_questionnaire`;
`research/FINDINGS.md` table row "Logistic non-questionnaire".

**Confirmed evidence** (pooled out-of-fold ROC-AUC, logistic regression with fold-local
oversampling, 3 seeds, grouped folds)

| Predictors | AUC |
|---|---|
| age + gender + jaundice | 0.579 |
| ... + country_of_res | 0.743 |
| ... + ethnicity | 0.713 |
| ... + relation | 0.612 |
| ethnicity only | 0.701 |
| "ethnicity unknown" flag only | 0.549 |
| Codex `non_questionnaire` set | 0.741 |
| ... plus family history (`austim`) | 0.742 |

- 95 rows have both ethnicity and relation missing; the two sets coincide exactly. The
  rows are not contiguous (positions 4 to 701). Of these 95, 42 are from Jordan and 15
  from the United Arab Emirates.
- Mean AQ sum is 3.58 in that group versus 5.08 elsewhere. Label YES rate is 9.5%
  versus 29.6%.
- YES rate by ethnicity: White-European 46.8%, Asian 13.0%, Middle Eastern 8.7%,
  Black 41.9%, South Asian 8.3%.

**Possible explanations (untested)**

- A different app version, language version or recruitment channel in some countries.
- Self-selection into the app differing by country.
- The questionnaire functioning differently across cultures or translations.
- Genuine prevalence differences.
- Missing ethnicity and relation meaning "question not asked" rather than "unknown".

None of these were tested. They cannot be separated with this dataset.

**Actions**

- [ ] **[paper]** Report that the residual signal is carried mainly by country and
  ethnicity, and do not describe it as demographic prediction of autism.
- [ ] Run leave-one-country-group-out validation to test whether the signal transfers
  across sites. Not yet done.
- [ ] Add a label-shuffle null for the non-questionnaire AUC and a grouped bootstrap
  confidence interval on it.
- [ ] Justify in Methods why `austim`, `used_app_before` and `age_desc` are excluded.
  Adding `austim` changed AUC by about 0.001, so the exclusion did not matter here.

### F5. The "without sum" results for trees reflect model capacity, not information

**Location:** `src/evaluate.py:76` (`max_depth=4`, `min_samples_leaf=10`);
`research/FINDINGS.md` lines 27-39.

**Confirmed evidence**

- Accuracy (primary, mean of 3 seeds): logistic without sum 0.9991, tree 0.8778, forest
  0.9536. Tree and forest with sum are 1.0000.
- All three models see the same ten items. Only the logistic model can represent an
  unweighted sum cheaply (see F3).
- Seed-42 re-runs by me matched Codex exactly: tree 0.8920, forest 0.9446, logistic
  0.9986.

**Possible explanations (untested)**

- A depth-4 tree cannot express "at least seven of ten items" without many splits, and
  oversampled duplicate rows interact with `min_samples_leaf`. No depth sweep was run.

**Actions**

- [ ] Run a depth and leaf-size sweep before interpreting the tree and forest rows.
- [ ] **[paper]** Word this as "removing `sum_score` hurts only models that cannot
  compute a sum", not as "the sum is informative".
- [ ] Keep the sentence in `research/STUDY.md` that individual items retain the sum's
  information.

### F6. Uncertainty reporting

**Location:** `research/FINDINGS.md` lines 47-54;
`src/evaluate.py:84-95` (`bootstrap_intervals`).

**Revision note.** An earlier version of this finding recommended exact binomial bounds
"wherever errors are zero". After checking the assumptions (below) that advice was too
broad. Whether an exact binomial interval is appropriate depends on what produced the
number. Literature behind this finding is summarized in [RELATED_WORK.md](RELATED_WORK.md).

**What an exact binomial interval assumes:** (A1) the classifier or rule is fixed before
the test rows are seen, (A2) test rows are independent draws from the population of
interest, (A3) nothing is selected using the same rows, and (A4) every row is scored by the
same classifier. Only a single holdout or a pre-specified rule meets A1 and A4 by
construction.

**Confirmed evidence**

- For perfect predictions the group bootstrap returns [1, 1].
- Exact (Clopper-Pearson) 95% upper bounds on the error rate for zero errors: 0.52% for
  n=704, 0.65% for n=563, 2.58% for n=141 (the original holdout). These are
  mathematically correct only under A1-A4.
- The "+/-" values in the results table are standard deviations over three seeds that
  reuse the same 704 rows. They are not confidence intervals. **Correction (2026-10-12):**
  the first version of this review said they are "0.41 and 0.58 of the binomial standard
  error". That used a scratch script that built folds from single rows instead of the
  study's matching-predictor groups. With the study's folds (reproducible output
  `research/uncertainty/whole_procedure_bootstrap.csv`) the repeat SD is 0.0086 for the
  non-questionnaire logistic model (0.49 of the binomial SE 0.0176) and 0.0123 for the tree
  without the sum (1.00 of its binomial SE 0.0123), and 0.34 and 0.64 of the whole-procedure
  bootstrap SE (0.0254 and 0.0192). Other scratch numbers below (bootstrap SE 0.0247 and
  0.0195, ratios 1.41 and 1.57) are close to the reproducible ones (ratios 1.45 and 1.56).
- **Pooled out-of-fold predictions violate A1 and A4.** They come from five different
  models trained on overlapping data. Bengio and Grandvalet (2004, read in full) prove no
  universal unbiased variance estimator of K-fold CV exists and that ignoring the
  correlation between errors can underestimate variance by about the size of the variance;
  they also say repeating K-fold with new splits does not give independent replicates.
  Varoquaux (2018 preprint, read in full) calls the binomial a best-case lower bound.
- **Simulation with known truth (my check).** Population: the 704 real item vectors;
  label `sum >= 7` flipped with probability eps; logistic regression; 5-fold CV; the
  estimand is the expected accuracy of the fold models on fresh rows. Rows are independent
  and the model is stable, which is the best case for the binomial. Results:

  | n | eps | runs | true spread / binomial SE | coverage of the exact 95% interval |
  |---|---|---|---|---|
  | 150 | 0.05 | 500 | 1.23 | 0.900 |
  | 150 | 0.15 | 500 | 1.24 | 0.896 |
  | 704 | 0.05 | 300 | 1.28 | 0.877 |
  | 704 | 0.15 | 300 | 1.18 | 0.907 |

  A Wilson interval did about the same (0.87-0.90). So even in the best case the interval
  is roughly 20-30% too narrow and covers about 88-91%, not 95%.
- **Zero-error regime in the same simulation.** With a deterministic label (eps = 0) at
  n=704, 52% of runs had zero out-of-fold errors. In all of them the fold models' true
  error (about 0.13%) was below the exact upper bound (0.52%). With zero errors the
  bound was conservative in this setting. This does not show it holds with dependence,
  clustering or a different label process.
- **Whole-procedure bootstrap on the real data (my check).** Resampling rows, keeping
  copies of a row in one fold, and repeating grouped 5-fold CV: standard error of pooled
  accuracy was 0.0247 for the non-questionnaire logistic model (binomial SE 0.0176, ratio
  1.41) and 0.0195 for the tree without `sum_score` (binomial 0.0124, ratio 1.57). A
  bootstrap sample holds only about 63% distinct rows, which inflates these ratios, so
  treat them as upper-leaning.
- **Clustering by country (my check, seed-42 out-of-fold predictions).** For the
  non-questionnaire logistic model, a country-cluster bootstrap standard error was 0.0669
  against 0.0174 at row level (ratio 3.8, design effect 14.7, implied intraclass
  correlation 0.23) because its predictions depend on country. For the tree and forest
  without `sum_score` the ratios were 0.73 and 1.15, and for the fixed `>= 6` rule 1.24.
  There are 67 countries, 55 with fewer than 10 rows, so these estimates are rough.
- **The `>= 7` rule is not a cross-validation output and was found post hoc.** Choosing
  the threshold inside each training fold gave 7 in all 15 folds (3 seeds) and 0/704
  out-of-fold errors, so selection did not create the zero. Adjusting for 11 candidate
  thresholds raises the one-sided 95% upper bound from 0.42% to 0.76%.
- **The `>= 6` rule was fixed in advance (ARC form).** Exact interval for its agreement
  with the labels: 634/704 = 0.9006, [0.8760, 0.9217]. Codex's conditional bootstrap gives
  [0.8796, 0.9220], essentially the same; a country-cluster bootstrap gives
  [0.8760, 0.9299].
- **The near-perfect logistic results rest on one row.** With and without `sum_score`, the
  only misclassified row is row 37 (score 6, label NO, age 34, New Zealand): wrong in one
  seed (with the sum) or two seeds (without). The "mean over seeds" averages a single
  borderline case.

**Possible explanations (untested)**

- The simulation undercoverage likely comes from correlated errors between fold models,
  as Bengio and Grandvalet's theory predicts, but I did not decompose the variance into
  their three components.
- The country clustering of correctness probably reflects the model using country as a
  predictor (see F4). I did not test other cluster definitions such as recruitment batch.
- Participant-level dependence beyond identical rows is unknown (see F7) and would add
  further dependence to every interval above.

**Actions**

- [ ] State next to every interval what it conditions on: fitted model, test rows treated
  as independent, rule fixed in advance or not.
- [ ] **Single holdout (n=141):** an exact interval is acceptable, described as conditional
  on the fitted model and on independent test rows.
- [ ] **Rules fixed in advance (`>= 6`):** report the exact or Wilson interval, plus the
  country-cluster interval as a sensitivity check.
- [ ] **Post-hoc rule (`>= 7`):** report "0 of 704 rows disagree" as a descriptive fact
  about this file and the nested-threshold result. If a bound is wanted, give the
  threshold-adjusted one and state it applies only to new rows from the same labeling
  process.
- [ ] **Trained models in cross-validation:** do not present an exact or Wilson interval on
  pooled out-of-fold predictions as a confidence interval. Use a bootstrap of the whole
  pipeline with fold-safe grouping, state its 63%-distinct-rows caveat, and for models
  that use country or ethnicity add a cluster-aware version.
- [ ] Relabel "+/- SD over seeds" as split-to-split variability, or remove it.
- [ ] Add ROC-AUC intervals from the same whole-procedure bootstrap and a label-shuffle
  null for the non-questionnaire AUC (Varoquaux reports permutation tests give good
  control). Neither is implemented yet.
- [ ] For near-perfect models, report the number and identity of distinct rows
  misclassified (row 37) instead of a seed-averaged accuracy.
- [ ] **[paper]** Never write "100% accurate" alone. Write that models reproduce the
  dataset's own `sum >= 7` rule on all 704 rows, give the count, and state what any bound
  is conditional on (F1).
- [ ] If you add paired model comparisons, read Nadeau and Bengio (2003) and Dietterich
  (1998) first. I have not read either.

### F7. Duplicate grouping is correct but immaterial

**Location:** `src/evaluate.py:58-61` (`predictor_groups`).

**Confirmed evidence**

- 5 duplicate full rows, 6 duplicate predictor rows (698 distinct predictor groups).
  My independently built groups gave the same fold assignments as Codex's.
- 2 predictor groups cross the original 563/141 holdout.
- Non-questionnaire AUC is 0.7405 with grouped folds and 0.7374 with plain stratified
  folds, so grouping does not change conclusions.

**Possible explanations (untested)**

- Duplicates could be repeated submissions by one person or coincidentally identical
  responses. Without participant identifiers this cannot be determined, and one person
  with slightly different answers would not be grouped.

**Actions**

- [ ] Keep the grouping, but describe it as a precaution, not a major leakage fix.
- [ ] **[paper]** State in Limitations that participant identity is unknown.

### F8. Preliminary and final results coexist and disagree

**Location:** `research/results/`, `research/README.md`,
`notebooks/02_model_experiments.ipynb`, `notebooks/README.md`.

**Confirmed evidence**

- The preliminary run (no oversampling, different age handling) reports demographics
  logistic accuracy 71.88%. The final run reports 68.04% (primary) for the same feature
  idea. Codex documents that these must not be merged.
- `notebooks/README.md` lists `03_screening_workflow_logic.ipynb`, which does not exist.
  Notebook 03 is `03_reproducible_evaluation.ipynb`.
- Neither notebook has stored outputs, so results exist only after running them.

**Possible explanations (untested)**

- None.

**Actions**

- [ ] Archive or delete `research/results/` and notebook 02, or label them clearly as
  superseded.
- [ ] Fix the notebook list in `notebooks/README.md`.
- [ ] Make sure every number in the paper traces to `research/evaluation/`.

### F9. Checks that are not yet in the evaluation

**Location:** `src/evaluate.py` as a whole.

**Confirmed evidence**

- The final evaluation has no label-shuffle null, no leave-country-out validation, no
  depth sweep, no coefficient report and no AUC confidence intervals. Calibration
  appears only in the preliminary `research/results/calibration.csv`, not in the final
  protocol, although models trained with oversampling do not produce calibrated
  probabilities (`research/STUDY.md` says so).

**Possible explanations (untested)**

- Not applicable.

**Actions**

- [ ] Add the null, leave-country-out and depth-sweep analyses (F4, F5).
- [ ] Either add calibration to the final evaluation or state that probabilities are
  not evaluated and should not be interpreted.
- [ ] Add exact bounds and AUC intervals (F6).

---

## Subgroup analysis

### F10. Subgroup comparisons are only feasible descriptively

**Location:** dataset columns `gender`, `ethnicity`, `age`, `relation`,
`country_of_res`.

**Confirmed evidence**

- Performance disparities cannot be compared: the `>= 7` rule has zero errors in every
  subgroup by construction.

| Group | n (YES) | 95% CI half-width on YES rate |
|---|---|---|
| female / male | 337 (103) / 367 (86) | about 0.05 / 0.04 |
| White-European | 233 (109) | 0.064 |
| Asian / Middle Eastern | 123 (16) / 92 (8) | about 0.06 |
| Black / Latino / Hispanic | 43 (18) / 20 (10) / 13 (5) | 0.14 / 0.20 / 0.23 |
| South Asian / Pasifika / Turkish | 36 (3) / 12 (1) / 6 (1) | 0.10 / 0.17 / 0.27 |
| age 18-24 / 25-34 / 35-49 / 50+ | 269 / 234 / 146 / 34 | 0.05 / 0.06 / 0.08 / 0.16 |
| relation: Self / Parent | 522 / 50 | 0.04 / 0.12 |

- Only White-European has 30 or more YES rows. 67 countries appear; 6 have at least 30
  rows and 55 have fewer than 10.
- Minimum detectable difference in rates (alpha 0.05, 80% power, base rate 0.27): male
  vs female about 0.095; White-European vs Asian 0.142; vs Black 0.220; vs Latino 0.317.
- YES rate: female 30.6%, male 23.4% (chi-square p = 0.033, uncorrected, one of many
  possible comparisons). Mean AQ sum is 4.99 (female) vs 4.77 (male), Mann-Whitney p = 0.32.
- Gender takes only the values `f` and `m`. Ethnicity categories overlap (Hispanic and
  Latino are separate; `Others` and `others` differ by case; one raw value is
  `Middle Eastern ` with a trailing space; `Turkish` is a nationality).

**Possible explanations (untested)**

- The gender difference in YES rate with no significant difference in mean score may
  come from the threshold amplifying a small shift. This was not tested.
- Group differences could reflect prevalence, self-selection, recruitment site or
  differential item functioning. These are confounded here.

**Actions**

- [ ] **[paper]** Report only descriptive subgroup tables with Wilson intervals. Make no
  fairness or prevalence claims.
- [ ] Fix the ethnicity category mapping and the minimum group size before looking at
  results, and report the mapping.
- [ ] Treat country as a site variable, not a demographic one.
- [ ] State the multiple-comparisons caveat next to any p-value.
- [ ] Do not attempt age 50+, fine ethnicity or intersectional comparisons.

---

## P2 findings

### F11. Training/serving skew and scope in the web app

**Location:** `src/main.py:202-207`, `src/main.py:210-215`, `src/main.py:240`,
`src/main.py:250`; `src/static/index.html:449-451`, `455`, `467`, `576`;
`src/train.py:41`, `src/train.py:44`.

**Confirmed evidence**

- `normalize_payload` lowercases only gender and jaundice. Training maps `others` to
  `Others` and `?` to `Unknown`.
- Ethnicity and relation are free-text inputs (`index.html:455`, `467`). The gender
  dropdown offers `other` (`index.html:451`), which never appears in the training data.
- The AQ-item inputs are raw `0`/`1` selects (`index.html:576`) with no item wording or
  reverse-keying instructions. The ARC form scores agree for items 1, 7, 8, 10 and
  disagree for items 2, 3, 4, 5, 6, 9.
- `train.py:44` silently converts invalid AQ values to 0. `evaluate.py` rejects them.
- Risk bands (0.45 and 0.8) are not derived from any validation. The `>= 7` trigger at
  `main.py:240` is the dataset rule. The report text at `main.py:250` refers to "home or
  school settings" although the model is trained on adults.

**Possible explanations (untested)**

- Unseen categories are ignored by the encoder, so predictions may change little
  because the items determine the label. I did not test the effect on predictions.

**Actions**

- [ ] Keep the web app out of the paper's claims.
- [ ] If the app stays: use dropdowns limited to the training vocabulary, normalize
  categories the same way as training, show item wording with the scoring rule, and
  reject invalid AQ values.
- [ ] Keep the adult-only and not-a-diagnosis disclaimers prominent.

### F12. Data-quality issues

**Location:** `data/train.csv` columns `age`, `age_desc`, `ethnicity`.

**Confirmed evidence**

- `age_desc` is the single value "18 and more", yet 18 rows have age under 18 and one
  row has age 383. Two ages are missing. Together these are the 21 flagged rows.
- Ethnicity and relation each have 95 `?` markers.

**Possible explanations (untested)**

- The age 383 may be a data-entry error. The true value is unknown and must not be
  guessed.

**Actions**

- [ ] Keep the primary analysis (reproduces the existing model) and the 677-row
  sensitivity cohort. Report both.
- [ ] State the 18-100 age range as a convention, not a correction.

### F13. Repository hygiene and reproducibility

**Location:** repository root; `research/evaluation/`; `requirements.txt`.

**Confirmed evidence**

- Codex's work is untracked on `add-cv-tests-ci`. `research/evaluation/predictions.csv`
  is about 4.1 MB. A third-party zip is stored in `research/sources/`.
- `requirements.txt` is unpinned. Results were produced on Python 3.14.5 with
  scikit-learn 1.8.0 (`research/results/provenance.json`); CI runs Python 3.11.
- The UCI page lists the license as CC BY 4.0.
- The numbers in this review came from scratchpad scripts that are not in the repo.

**Possible explanations (untested)**

- Results may differ slightly across Python or scikit-learn versions. I did not test
  other versions.

**Actions**

- [ ] Commit the research files in logical pieces and create the PR when ready.
- [ ] Add a tracked script that reproduces the review's numbers (conditional label
  purity, exact bounds, coefficient fit, feature-set AUC table, subgroup counts).
- [ ] Keep attribution for the CC BY 4.0 dataset in the README and paper.
- [ ] Decide whether the large CSV and the zip belong in git or in release artifacts.
- [ ] Pin versions or record `research/evaluation/environment.txt` as the reference.

---

## Research question and contribution

**Question:** How much of the apparent predictive performance of ML models on the UCI
Adult Autism Screening dataset is explained by reconstructing the AQ-10 score
threshold, and what signal remains when questionnaire information is removed?

**Defensible contribution**

- [ ] A verified provenance audit (F1).
- [ ] A decomposition: rule equals label, models add nothing, residual signal is mostly
  country and ethnicity structure (F1, F3, F4).
- [ ] A short reusable circularity checklist: label purity given a candidate summary
  score, a threshold scan, parity with the rule, and ablation with the model-class
  caveat (F5).
- [ ] Optional: repeat the full checks on the UCI child (id 419, 292 instances) and
  adolescent (id 420, 104 instances) datasets. In a scratch check (not saved in the repo)
  I found the child label is exactly `sum >= 7` as well (0 disagreements; 40 at threshold
  6, 44 at 8). The adolescent file could not be downloaded from the URL pattern I tried, so
  it is unchecked.

**Novelty:** modest and not established. Two papers I read in full bear on it (see
[RELATED_WORK.md](RELATED_WORK.md)). ASD-Bench (arXiv:2605.11091) states that its labels
are questionnaire-derived and that results are not clinically validated, but attributes
its perfect adult scores to small sample size and limited feature diversity; it does not
test whether the label is a threshold on the score. Bone et al. (2015) argue that
predicting an instrument's own algorithmic diagnosis from its item codes is near-trivial
for ADOS and ADI-R. The circularity concept is therefore not new; quantifying it on the
UCI AQ-10 data is the narrow contribution. I read only two of the many ML papers that
use these datasets, so do not claim to be first.

## What the paper can and cannot claim

| Supported by the evidence | Not supported |
|---|---|
| Labels equal `sum_score >= 7` on all 704 rows | Models predict autism or diagnosis |
| Models match the rule and cannot exceed it here | Clinical screening utility |
| The non-questionnaire AUC is about 0.74 and is carried mainly by country and ethnicity | Demographics predict autism |
| Subgroup rates with intervals, as descriptions | Fairness conclusions or prevalence differences |

A valid diagnostic study would need independently assessed outcomes, participant-level
splits and an external cohort. No suitable dataset was identified in this review.

## Paper outline

- [ ] **Introduction:** AQ-10 as a referral aid; ML papers on screening datasets report
  very high accuracy; the question of what the labels encode.
- [ ] **Related work:** AQ-10 validation (Allison 2012; Booth 2013), the dataset
  documents, leakage methodology (Kaufman 2012; Kapoor and Narayanan 2023), and recent
  benchmarks on the same kind of data.
- [ ] **Methods:** provenance verification, label audit, grouped repeated folds, model
  and feature-set grid, exact bounds, subgroup protocol with pre-specified categories.
- [ ] **Results:** crosstab, parity with the rule, coefficient evidence, feature-set
  ablation, country decomposition, subgroup tables with intervals.
- [ ] **Discussion:** why high accuracy here is not diagnostic evidence; what a valid
  test needs.
- [ ] **Limitations:** undocumented label, app-collected self-selected sample, no
  participant identifiers, no independent diagnoses, no external validation, inferred
  item encoding, unknown label-generation procedure.

## References

A per-paper comparison with access levels (full text, abstract only, metadata only) is in
[RELATED_WORK.md](RELATED_WORK.md). The list below is the earlier, shorter verification
record; where the two differ, RELATED_WORK.md is more recent.

**Verified in this review** (metadata via Crossref, PubMed, the UCI page or arXiv;
full text not read unless stated)

- Allison C, Auyeung B, Baron-Cohen S (2012). *J Am Acad Child Adolesc Psychiatry*
  51(2):202-212.e7. doi:10.1016/j.jaac.2011.11.003. Abstract: at cut-point 6 on the
  adult AQ-10, sensitivity 0.88, specificity 0.91, PPV 0.85; 449 adult cases and 838
  adult controls, cases recruited from a volunteer database.
- Booth T, Murray AL, McKenzie K, Kuenssberg R, O'Donnell M, Burnett H (2013).
  *J Autism Dev Disord* 43(12):2997-3000. doi:10.1007/s10803-013-1844-5. Abstract: 149
  ASD cases and 134 controls; AQ-10 performed similarly to the AQ-50.
- Baron-Cohen S, Wheelwright S, Skinner R, Martin J, Clubley E (2001). *J Autism Dev
  Disord* 31(1):5-17. doi:10.1023/A:1005653411471.
- Thabtah F (2017). Proc. 1st Int. Conf. on Medical and Health Informatics, pp. 1-6.
  doi:10.1145/3107514.3107515. Crossref lists the short title "Autism Spectrum Disorder
  Screening"; the dataset description gives a longer title with a subtitle.
- Thabtah F. Machine learning in autistic spectrum disorder behavioral research: a
  review and ways forward. *Inform Health Soc Care* 44(3):278-297.
  doi:10.1080/17538157.2017.1399132 (Crossref dates it 2018).
- Thabtah F. Autism Screening Adult. UCI Machine Learning Repository, donated
  2017-12-23, CC BY 4.0. doi:10.24432/C5F019. Related: Children doi:10.24432/C5659W;
  Adolescent doi:10.24432/C5V89T.
- Kaufman S, Rosset S, Perlich C, Stitelman O (2012). Leakage in data mining. *ACM
  TKDD* 6(4). doi:10.1145/2382577.2382579.
- Kapoor S, Narayanan A (2023). Leakage and the reproducibility crisis in
  machine-learning-based science. *Patterns* 4(9):100804.
  doi:10.1016/j.patter.2023.100804.
- Autism Research Centre AQ-10 adult form (read in full): scoring keys and the "6 or
  above" referral wording, key reference Allison et al. 2012.
- Singh S, Shaikh H, Raghuwanshi K, Bulia K (2026). ASD-Bench. arXiv:2605.11091.
  Abstract page only.
- Rasul RA et al. (2023). An Evaluation of Machine Learning Approaches for Early
  Diagnosis of Autism Spectrum Disorder. arXiv:2309.11646. Abstract page only; it does
  not discuss leakage on that page.

**Not verified. Do not cite until checked**

- NICE guideline CG142: the page returned HTTP 403. The ARC form states that the AQ-10
  is recommended in CG142; attribute that statement to the form.
- A thesis from a Chilean university said to have removed a leaking variable: it appeared
  only in a search summary that mixed in its own inference, and I could not confirm it.
- The README's references (Zhang 2025, PMC12036476; Duke Psychiatry 2023): not checked
  in this review.
- The full text of every paper above: I read abstracts and metadata only.

## Not checked in this review

- Codex's bootstrap intervals were not recomputed.
- The 677-row sensitivity cohort was checked only for row count and accuracy
  consistency with `pooled_metrics.csv`.
- Per-fold metrics were not re-derived.
- Reproducibility across Python and scikit-learn versions was not tested.
- No analysis was run on the UCI child or adolescent datasets.
