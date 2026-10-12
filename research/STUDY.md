# Adult screening-label reconstruction study

## Research question

How do simple AQ-score rules and conventional classifiers compare in reproducing
the UCI adult screening labels, and how does performance change when score-derived
information is removed and matching records are kept in the same validation fold?

This is a reproducibility and methodology study. It is not a diagnostic-validation
study. A negative result (no improvement over a score rule) is informative.

## Source and label evidence

Sources accessed 2026-10-11:

1. [UCI dataset record](https://archive.ics.uci.edu/dataset/426/autism+screening+adult):
   Fadi Thabtah, 2017, 704 instances, DOI [10.24432/C5F019](https://doi.org/10.24432/C5F019),
   CC BY 4.0. The archive has 21 attributes including the target; the webpage's
   feature count should not be interpreted as 21 independent predictors.
2. [Original UCI archive](https://archive.ics.uci.edu/static/public/426/autism+screening+adult.zip),
   retained as `research/sources/uci_adult_426.zip`. The ARFF declares `Class/ASD`
   with values NO/YES. Its accompanying description identifies ten AQ-10 adult
   behavioral items and calls the screening score automatically computed.
   It references ASDTests. The accessible description does not explain a
   clinician-confirmation procedure for the target or explicitly specify its
   threshold. Therefore do not assert that it contains verified diagnoses.
3. [Autism Research Centre adult AQ-10 form](https://www.autismresearchcentre.com/content/uploads/2024/11/AQ10.pdf):
   a referral guide whose scoring instructions use 6 or above. Item scores
   are keyed: agreement scores some items, disagreement scores others.
   The repository's binary A-score columns must be treated as already-scored
   items, not generic yes/no answers. The source table calls them answer codes;
   the exact app encoding remains a provenance question.

The empirical rule `sum_score >= 7` is separate from the documented ARC referral
rule `sum_score >= 6`. Both are compared against the repository's labels. A
disagreement between a referral rule and these labels is not evidence that the
referral rule is clinically wrong. ROC-AUC for both rules uses the continuous
AQ sum, so their AUCs are identical even though binary decisions differ.

## Existing pipeline audit

`src/train.py` loads the CSV, renames two misspelled fields, replaces categorical
missing markers with Unknown, converts the ten items to integers, and adds their
sum. It removes ID, age description, prior app use, family-history field
`austim`, recorded result, and target from predictors. It makes an 80/20 stratified
random split with seed 42. Five-fold stratified CV is performed within training.
Each model imputes numeric medians, scales numeric features, encodes categories,
oversamples training records, and fits logistic regression. The trained serving
pipeline and metrics are then saved.

Confirmed from code: learned preprocessing and oversampling run inside each
training fold; the test set is not used for fitting. Deterministic row-wise
cleaning and score summation before splitting do not use test distribution
statistics. Target and recorded result are not directly passed to the estimator.
Invalid AQ values are silently converted to zero in the serving loader; the new
evaluation rejects these inputs before accepting results.

Confirmed from data and reproduced splitting: the local labels match sum >=7
exactly. Matching predictor records cross the original random holdout and CV
boundaries; see `evaluation/audit.json` for counts and the original holdout
comparison. Removing sum_score alone leaves the same information in ten items.
Perfect agreement therefore does not establish independent clinical prediction.

Hypothesis: the labels were generated directly from a screening-score threshold.
The exact empirical match supports this, but the accessible source documentation
does not prove the collection procedure. Duplicates might be repeated submissions;
without participant IDs, matching rows cannot establish participant identity.

`src/main.py` serves the saved pipeline with the same engineered sum. Its probability,
arbitrary risk-band thresholds, keyword flags, and referral text have not been
clinically validated. Oversampling changes class prevalence, so classifier
probabilities also require separate calibration evaluation. This study leaves
the serving artifacts intact and does not endorse their clinical interpretation.

## Evaluation protocol

Primary analysis retains the existing loader's age and missing-category handling
to reproduce the existing model. A separate sensitivity cohort keeps ages 18-100
and the first record per matching predictor group. This age range is an explicit
research convention, not a correction of known ages or a clinical eligibility rule.

All methods share five StratifiedGroupKFold folds, repeated with seeds 42, 137,
and 2026. Grouping uses the app's predictor values, excluding labels. The same
person with different predictor records may still cross folds; participant IDs
would be required to prevent that. Preprocessing and RandomOverSampler are
fitted inside each training fold. No tuning or model selection uses the validation
labels. Fold assignments and row indices are exported.

Methods: majority DummyClassifier; fixed AQ >=6 and >=7 rules; logistic regression
(the exact serving pipeline for combined-with-sum at seed 42); shallow decision
tree; small random forest. Four feature sets: combined with sum, combined without
sum, questionnaire items only, and age/categorical predictors only. The last set
removes all questionnaire-score information. No claim of demographic causation
or fairness is made. Settings and model parameters are saved in JSON.

Precision, recall, specificity, F1, accuracy, ROC-AUC, and confusion counts are
reported per fold and as pooled out-of-fold results per repeat. Repeat and fold
standard deviations quantify observed variation, not independent confidence
intervals. Constant dummy rankings have AUC 0.5 when both classes are present.

95% percentile bootstrap intervals resample whole predictor groups using the
first seed's fixed out-of-fold predictions. They reflect conditional sampling
uncertainty, not retraining variability, population transportability, or diagnosis
uncertainty. Perfect reconstruction can yield [1,1] bootstrap intervals: this is
a limitation of the conditional bootstrap at the boundary, not proof of perfect
future performance. ROC-AUC uncertainty is summarized by repeat/fold SD only.

**Cross-validation predictions are not independent observations.** The five fold models share
training data, the same rows recur across repeats, and Bengio and Grandvalet (2004) prove that no
unbiased estimator of the variance of K-fold cross-validation exists. An exact or Wilson interval
on pooled out-of-fold predictions therefore understates uncertainty. In a simulation with known
truth (independent rows, stable model) exact intervals covered 88-91% at a nominal 95%. The
`uncertainty/` analyses (`python -m src.uncertainty`) report, with what each depends on:

- counts first (for the `>= 7` rule, zero disagreements is a fact about this file);
- exact and Wilson intervals only where their assumptions can hold (a single holdout, or the
  documented `>= 6` rule fixed before looking at the data), and a zero-error bound adjusted for the
  number of scanned cut-offs;
- a nested check that the cut-off chosen inside each training fold is always 7;
- a bootstrap of the whole cross-validation procedure for the imperfect models (resamples hold about
  63% distinct rows, so the spread is somewhat overstated);
- a country-cluster bootstrap, because errors of models that use country or ethnicity cluster by country;
- a label-shuffle null for the non-questionnaire ROC-AUC, which tests for any label signal, not its cause;
- a decomposition of that signal by predictor group, and descriptive subgroup counts with Wilson
  intervals (sampling uncertainty only; categories as recorded).

"+/- SD over repeats" is split-to-split variability, not a confidence interval.

## Reproduction

### Explanatory analyses

The final runner also exports `label_by_score.csv` and `threshold_scan.csv` for
both cohorts. Thresholds 0-11 cover every possible integer cutoff, including
all-positive and all-negative decisions. This full-data scan describes the
observed score-label relationship; choosing a clinical threshold from it would
be inappropriate. Rule ROC-AUC always uses the continuous sum and therefore
does not change across cutoffs.

`questionnaire_logistic_coefficients.csv` records training-fold weights, scaler
means/scales, and raw-unit intercepts for every seed and cohort. A standardized
weight is divided by its training-fold scaler scale to recover the effect of
a one-point binary item change. The transformed intercept is also saved.
Summary mean/SD/min/max describe variation, not inferential uncertainty or
clinical importance. Oversampling and regularization affect these coefficients.

`tree_depth_metrics.csv`, `tree_depth_fold_metrics.csv`, and `tree_depth_summary.csv`
compare depths 1, 2, 3, 4, 6, 8, 10, and unlimited using the same grouped folds.
Questionnaire-only features are compared with combined features including sum.
Minimum leaf size remains 10, so deeper trees may still be constrained. Fold
outputs include training accuracy, validation metrics, learned depth, number of
leaves, root feature, and numeric root threshold mapped back to original units.
Depths are recorded in `protocol.json` and `settings.json`; no winning depth is
selected or promoted to the main evaluation.

From the repository root with the project virtual environment active:

```bash
python3.14 scripts/reproduce.py
```

Install Python 3.14.5 first. The command creates or uses `.venv`, installs the
root `requirements.lock`, checks dependencies and source-input hashes, runs tests,
and executes the seeds/folds/bootstrap count in `research/protocol.json`.
Default output: `research/evaluation/`. Serving artifacts are not changed.
The reproduction command also runs the reusable adult/child label-circularity
audits listed in `protocol.json`, saving separate results to `research/circularity/`.
See `CIRCULARITY.md` for source evidence and that audit's distinct missing-value
handling. The second dataset tests the audit methodology, not adult-to-child
prediction or independent clinical validation.
It then runs `python -m src.uncertainty` (outputs in `research/uncertainty/`) and
`python -m src.report_assets` (publication tables and figures in `research/report/`; the plotting
library is pinned in `requirements.lock`; with the same package versions on the same machine,
regenerated files were byte-identical, but identity across operating systems is untested).
`evaluation/environment.txt` captures installed versions; `requirements.lock`
is the maintained pin file. Existing virtual environments using another Python
version are rejected rather than replaced.

For custom exploratory runs after setup, invoke `python -m src.evaluate` with
`--output`, `--seeds`, `--folds`, or `--bootstrap`. Such runs use a different
protocol and should have a separate output directory. The source archive is
optional for custom runs but mandatory in the final reproduction command.

The older notebook and results in `research/archive/preliminary/` are preliminary,
unoversampled experiments with different age handling. Do not merge those numbers
with this protocol. `03_reproducible_evaluation.ipynb` is the entry point for this
study and uses the same runner as the command above.

## Evidence needed for a paper

A credible methodological paper needs a clear contribution beyond trying more
models: independently checkable provenance, complete reproducibility, discussion
of label circularity and duplicate handling, related-work comparisons, and preferably
confirmation on another appropriately sourced dataset. The findings may support
a student research report; publication depends on novelty and the venue.

Claims about adult clinical screening utility require independently assessed
diagnostic outcomes, participant-level splitting, an external cohort, relevant
population and recruitment details, adequate sample sizes, calibration, and
subgroup error estimates with uncertainty. Neither this dataset's screening labels
nor comparisons against them supply those outcomes. Exact label-generation and
item-encoding details should be sought from the dataset author before stronger
claims are made.
