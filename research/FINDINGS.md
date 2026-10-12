# Results of the reproducible evaluation

These numbers were computed with `src.evaluate` on the verified 704-row CSV.
The source archive's raw cells and row order match the local CSV exactly.
See `evaluation/settings.json` for seeds, software, hashes, and parameters.

## Audit findings: confirmed

- 515 NO and 189 YES labels; all labels match AQ sum >=7.
- The recorded result equals the sum of the ten item scores in all records.
- Five duplicate full rows and six duplicate app-predictor rows.
- Two predictor groups cross the original 563/141 holdout split. Overlap counts
  in the original training CV are 1, 2, 0, 4, and 1.
- The original holdout yields 103 true negatives, 38 true positives, and no
  errors for both the serving logistic model and the >=7 rule.
- Preprocessing and oversampling are fitted inside training folds, not before
  splitting. Deterministic row-wise summation is not preprocessing leakage.
- 95 missing markers each in ethnicity and relation, two missing ages, and 21
  missing/out-of-range ages under the specified 18-100 audit convention.

## Primary grouped evaluation

Means of pooled out-of-fold metrics across seeds 42, 137, and 2026.
The +/- quantity below is the sample standard deviation across three repeats;
it is not a confidence interval. All 704 records are evaluated once per repeat.

| Method | Accuracy mean +/- SD | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Majority dummy | 0.7315 +/- 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| ARC AQ >=6 rule | 0.9006 +/- 0.0000 | 0.7297 | 1.0000 | 0.8438 | 1.0000 |
| Observed label rule AQ >=7 | 1.0000 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic combined with sum | 0.9995 +/- 0.0008 | 0.9982 | 1.0000 | 0.9991 | 1.0000 |
| Logistic combined without sum | 0.9991 +/- 0.0008 | 0.9965 | 1.0000 | 0.9982 | 1.0000 |
| Logistic questionnaire only | 1.0000 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Logistic non-questionnaire | 0.6804 +/- 0.0086 | 0.4434 | 0.7443 | 0.5556 | 0.7405 |
| Tree combined with sum | 1.0000 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Tree combined without sum | 0.8778 +/- 0.0123 | 0.7216 | 0.8889 | 0.7964 | 0.9430 |
| Forest combined with sum | 1.0000 +/- 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Forest combined without sum | 0.9536 +/- 0.0110 | 0.8770 | 0.9630 | 0.9178 | 0.9925 |

All remaining model/feature combinations, per-fold metrics, and confusion counts
are in `evaluation/pooled_metrics.csv` and `evaluation/fold_metrics.csv`.
For seed 42, the >=6 rule has TN=445, FP=70, FN=0, TP=189; the >=7 rule and
logistic-with-sum have TN=515, FP=0, FN=0, TP=189. Rule AUCs use continuous
sum scores, not binary decisions, explaining why both can have AUC 1.0.

## Uncertainty and sensitivity

For seed 42, 1,000 predictor-group bootstrap resamples produce these conditional
95% accuracy intervals: AQ >=6 [0.8796, 0.9220], logistic without sum
[0.9957, 1.0000], and non-questionnaire logistic [0.6567, 0.7232].
Perfect first-seed reconstructions yield [1,1]; this boundary limitation does
not establish perfect generalization. Bootstrap predictions are fixed rather
than retrained. ROC-AUC variability is reported across repeats/folds.

**How to read these intervals.** Cross-validation predictions are not independent
observations: the five fold models share training data and the same rows recur across
repeats. A binomial or conditional-bootstrap interval on pooled out-of-fold predictions
omits retraining variability and is not a validated interval for generalization performance.
The checks below come from `python -m src.uncertainty` (`research/uncertainty/`);
`interval_table.csv` lists every interval with what it depends on.

- **Counts first.** The `>= 7` rule disagrees with the labels on 0 of 704 rows. That is a
  fact about this file, not an estimate of future agreement. The cut-off chosen inside each
  training fold was 7 in all 15 folds, with 0 out-of-fold errors in every repeat
  (`nested_threshold.csv`). If a bound is wanted, the exact one-sided 95% upper bound on the
  disagreement rate is 0.78% after adjusting for the 12 scanned cut-offs. This assumes
  independent rows from the same labeling process; that assumption is not established.
- **Exact intervals where their assumptions can hold.** Original 141-row holdout, 141/141
  agreement for the serving model: Clopper-Pearson [0.9742, 1.0000], conditional on the
  fitted model and independent test rows. The post-hoc `>= 7` rule also agrees on 141/141,
  but that holdout was used to discover the threshold, so no unadjusted interval is given.
  The documented `>= 6` rule, fixed before looking at the
  data, on all rows: 634/704 = 0.9006, [0.8760, 0.9217] (Wilson [0.8762, 0.9205]).
- **Whole-procedure bootstrap** (rows resampled, grouped cross-validation rerun, 200 draws;
  resamples hold about 63% distinct rows, changing effective training size; coverage and the
  direction of bias in this spread are unverified). Standard
  error of pooled accuracy: non-questionnaire logistic 0.0254 against a binomial 0.0176
  (ratio 1.45); tree without the sum 0.0192 against 0.0123 (ratio 1.56). The repeat SD in the
  main table (0.0086 and 0.0123) is split-to-split variability, not uncertainty about a
  population.
- **Country clustering** (seed-42 predictions). Standard errors from resampling whole
  countries are 3.7-3.8 times the row-level ones for the non-questionnaire models and 3.2
  times for the majority-class baseline, but 0.7-1.2 times for the other models. Models that
  use country or ethnicity make errors that cluster by country, and the label rate itself
  differs by country. With 67 countries of very unequal size these ratios are rough.
- **The near-perfect logistic results rest on one row.** Row 37 (total 6, label NO) is the
  only record misclassified by logistic regression with and without the sum: in one repeat
  with the sum, in two repeats without it (`predictions.csv`).

The age/duplicate sensitivity cohort has 677 records. The >=7 rule and
questionnaire-only logistic model remain perfect; combined-with-sum logistic
averages 0.9995, combined-without-sum logistic 0.9966. Cleaning does not remove
label circularity. The primary cohort retains age anomalies to reproduce the
serving model's handling; the sensitivity cohort excludes them explicitly.

## Signal left when questionnaire information is removed

With all ten items, their sum and the recorded result removed, only age and five categorical
fields remain: accuracy 0.6804 against a 0.7315 majority baseline, ROC-AUC 0.7405 (main
table). The exploratory label-shuffle analysis uses 200 shuffles, each evaluated with the
same three repeat seeds as the observed mean AUC. Its per-shuffle values and summary are
saved in `permutation_draws.csv` and `permutation_null.csv`. This checks association under
unrestricted row-label exchangeability, not its cause. Duplicate records and unknown
recruitment dependencies mean exchangeability is unverified, so its p-value is not a
clinically validated or guaranteed-calibrated significance result.
The corrected null has mean 0.4986, 95th percentile 0.5516, and range 0.4274-0.5771.
All 200 draws are below the observed 0.7405; the add-one Monte Carlo value is 1/201
(0.00498), subject to the exchangeability limitation above.

**Country-held-out robustness.** Training the non-questionnaire logistic model on all
other countries and pooling predictions gives ROC-AUC 0.6816, 0.6885 and 0.6892 across
the three seeds (mean 0.6865), versus 0.7405 in grouped five-fold validation. Mean
accuracy is 0.6771. Every row is predicted once per seed with its recorded country
absent from training (`leave_country_out_*.csv`, report table 8). This is a different
validation task, not evidence of a causal demographic mechanism or clinical transfer.
Small single-class countries have undefined within-country AUC.

Decomposition by predictor group (`residual_decomposition.csv`; logistic regression, mean
pooled out-of-fold ROC-AUC over three repeats):

| Predictors | ROC-AUC |
| --- | --- |
| age + gender + jaundice | 0.579 |
| age + gender + jaundice + country | 0.743 |
| age + gender + jaundice + ethnicity | 0.713 |
| age + gender + jaundice + relation | 0.612 |
| ethnicity only | 0.701 |
| "ethnicity unknown" flag only | 0.549 |
| main set (all six fields) | 0.741 |
| main set + family history | 0.742 |
| main set, no duplicate grouping | 0.737 |

**Confirmed:** country and ethnicity carry most of the signal; family history adds almost
nothing; duplicate grouping changes little. 95 rows are missing both ethnicity and relation;
42 of them are from Jordan and 15 from the United Arab Emirates, their label rate is 9.5%
against 29.6% elsewhere, and their mean item total is 3.58 against 5.08 (`missing_block.csv`).

**Not established:** why. Different app versions or recruitment channels, translation or
cultural differences in how items are answered, self-selection, real prevalence differences,
or missing fields meaning "not asked" are hypotheses. The dataset cannot separate them, and
this study makes no claim about demographic causes. `subgroup_counts.csv` gives descriptive
counts with Wilson intervals (sampling uncertainty only, categories as recorded); among
ethnicity categories only White-European has 30 or more positive labels, so subgroup
performance comparisons are not supported.

Publication figures and tables built from these outputs are in `research/report/`.

## Interpretation and unresolved evidence

### Explanatory results

The full-data threshold scan over integer cutoffs 0-11 has exactly one cutoff
with no label disagreements: >=7. Cutoff >=6 disagrees on 70 score-6 records;
>=8 disagrees on 57 score-7 records. This describes the label structure rather
than selecting a new clinical referral threshold.

| AQ score | NO | YES |
| --- | --- | --- |
| 0 | 14 | 0 |
| 1 | 33 | 0 |
| 2 | 74 | 0 |
| 3 | 110 | 0 |
| 4 | 131 | 0 |
| 5 | 83 | 0 |
| 6 | 70 | 0 |
| 7 | 0 | 57 |
| 8 | 0 | 55 |
| 9 | 0 | 47 |
| 10 | 0 | 30 |

Questionnaire-only logistic coefficients from 15 primary training-fold fits
are all positive. Mean raw 0-to-1 item coefficients range from 3.7385 to 4.3945.
Their similar magnitude is consistent with approximating an additive score
boundary, not discovering distinct clinical importance. Standardized coefficients,
raw coefficients, scaler parameters, odds ratios, and raw intercepts are exported.
Training-fold variation is not an independent confidence interval.

| Maximum tree depth | Questionnaire-only accuracy mean +/- SD | Combined-with-sum accuracy |
| --- | --- | --- |
| 1 | 0.8220 +/- 0.0207 | 1.0000 |
| 2 | 0.8641 +/- 0.0033 | 1.0000 |
| 3 | 0.8338 +/- 0.0071 | 1.0000 |
| 4 | 0.8802 +/- 0.0100 | 1.0000 |
| 6 | 0.8944 +/- 0.0059 | 1.0000 |
| 8 | 0.8963 +/- 0.0057 | 1.0000 |
| 10 | 0.8963 +/- 0.0057 | 1.0000 |
| Unlimited | 0.8963 +/- 0.0057 | 1.0000 |

These are pooled out-of-fold means across the same three seeds, with oversampling
inside training folds. All 15 combined-with-sum stumps split on sum_score at
6.5 in original units. Questionnaire-only stumps split on A9 (13 folds) or
A5 (2 folds). Minimum leaf size is fixed at 10. Greedy splits and this constraint
explain why extra depth need not monotonically improve validation agreement;
the comparison is descriptive and no depth is selected as a winner.

No evaluated model beats the deterministic >=7 rule on these labels. Individual
items retain the information in the sum, so removing only sum_score does not
make the target independent. The >6 rule is an empirical reconstruction of this
dataset, whereas the ARC form uses >=6 for referral consideration. These are
different tasks; disagreement cannot establish that the clinical referral rule
is wrong. No clinical diagnosis probabilities are estimated by this evaluation.

The source documentation verifies a screening dataset and automatically computed
scores; direct threshold generation of the labels remains an inference until
the original collection procedure is confirmed. Participant identities,
independent diagnostic outcomes, recruitment details, and external validation
remain missing. Those gaps limit clinical claims and the novelty of a paper.
See `STUDY.md` for the proposed research question and evidence requirements.
