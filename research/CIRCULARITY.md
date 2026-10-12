# Reusable label-circularity audit

This audit implements the methodological checklist proposed in the repository's
independent `REVIEW.md`: verify provenance, measure label purity by score, scan
thresholds, compare a fixed documented rule, and remove all questionnaire inputs.
It does not establish how labels were collected or validate a diagnostic model.

## Run

After installing the pinned project environment, from the repository root:

```bash
python -m src.circularity_audit --config research/datasets/adult.json --output research/circularity/adult
python -m src.circularity_audit --config research/datasets/child.json --output research/circularity/child
```

The full `python3.14 scripts/reproduce.py` command now runs both configured audits
after the final adult experiments. Output paths are separate. Neither the adult
serving model nor the adult dataset is replaced or used to predict child outcomes.

## What is reusable

The JSON config supplies input format/path/checksum, target and label mapping,
scored item names and allowed numeric values, recorded score column, candidate
thresholds, matching-record group columns, residual features, seeds/folds,
documented fixed rule, and population/source/label evidence. CSV and a named ARFF
member inside a ZIP are supported. The audit uses an unweighted sum of already
scored items; raw answers needing reverse coding, weighted instruments, multiclass
targets, and missing item scores must be prepared and documented separately.

For another dataset, create a new config based on these examples, independently
inspect its source documentation, and record the correct checksum. Do not assume
that item numbering, encoding, referral cutoff, or label meaning transfers between
instruments. A URL in a config is a declared reference, not automatic verification
of its claims. The runner checks bytes, schema, and values; source interpretation
requires researcher review. Every configured feature must be available in the input.

The audit rejects missing documentation, changed input bytes, unknown labels,
missing/invalid item values, explicit target or recorded-score predictors, and
questionnaire items in the residual feature set. It does not automatically detect
all derived proxies in a new dataset. Inspect residual columns for other summaries
or post-outcome variables before making independence claims.

## Outputs and interpretation

- `report.json`: class/duplicate counts, recorded-score agreement, age flags,
  empirical score-group purity, conditional label entropy, mixed-label scores,
  descriptive best thresholds, source declarations, and limitations.
- `label_by_score.csv` and `threshold_scan.csv`: complete descriptive tables.
- `pooled_metrics.csv`, `fold_metrics.csv`, and `summary.csv`: grouped validation
  of majority dummy, fixed documented rule, score-only logistic, questionnaire
  logistic, and residual-feature logistic models.
- `fold_assignment.csv` and `predictions.csv`: reproducible row-level evaluation.
- `settings.json`: complete config, source/config/code checksums, software versions,
  seeds, folds, and logistic parameters.

Conditional entropy is H(label | observed score), in bits. Zero means no label
variation within observed score groups. Purity is the proportion of rows matching
each score group's most frequent label. Purity can be misleading for singleton
score groups, which are counted explicitly. Pure groups need not form a threshold:
the audit tests threshold agreement separately. Neither statistic proves that the
score caused or generated the labels.

Threshold scans use all labels descriptively. The best scanned cutoff is not
treated as an independently validated rule. Only the pre-documented referral
rule is included in shared grouped folds. Its AUC uses continuous sum scores,
so perfect AUC may coexist with binary-rule disagreements.

All models share grouped folds over three seeds. Matching-record groups are built
without targets; they do not identify participants. Learned imputation, scaling,
encoding, and oversampling remain inside training folds. Residual categorical
missing values use fold-local most-frequent imputation, unlike the earlier adult
experiment's fixed Unknown category. The audit also keeps spelling variants such as
`others` and `Others` as separate categories, whereas the main evaluation merges them
(`src/train.py`); exact replication of the audit's residual-model counts requires keeping
them separate. These new audit model metrics therefore
belong to a separate protocol and must not silently replace the final adult table.
Fold/repeat SD is descriptive variability, not an independent confidence interval.

## Second dataset evidence

[UCI dataset 419](https://doi.org/10.24432/C5659W), Fadi Thabtah (2017), is a
292-record child screening dataset licensed CC BY 4.0. The original archive is
retained in `sources/uci_child_419.zip`; the audit reads its ARFF directly rather
than using an undocumented CSV mirror. Its ARFF identifies `Class/ASD` as NO/YES
and `age_desc` as 4-11 years. The attached description identifies AQ-10-Child and
automatically computed scores but does not explicitly explain target generation
or diagnostic confirmation. Its abstract contains adult wording as well, a source
documentation inconsistency recorded in the config.

[ARC child AQ-10 form](https://www.autismresearchcentre.com/content/uploads/2024/11/AQ10-Child.pdf)
specifies ages 4-11 and referral consideration at 6 or above. As with adults, the
binary dataset items are treated as scored codes; their exact app encoding remains
unverified. This audit of a related screening dataset is replication of a
methodological observation, not an independent clinical cohort or transfer study.

## Computed results

| Finding | Adult UCI 426 | Child UCI 419 |
| --- | --- | --- |
| Records | 704 | 292 |
| Negative / positive labels | 515 / 189 | 151 / 141 |
| Conditional label entropy by sum | 0 bits | 0 bits |
| Empirical score-group purity | 1.0 | 1.0 |
| Descriptive cutoff with zero disagreements | >=7 | >=7 |
| Recorded-score versus item-sum disagreements | 0 | 0 |
| Documented >=6 rule disagreements | 70 | 40 |
| Duplicate full records | 5 | 2 |
| Matching-predictor duplicate rows | 6 | 3 |
| Singleton score groups | 0 | 1 |

Both score-only and questionnaire-only logistic models achieved perfect label
agreement in these audits. Child residual logistic ROC-AUC averaged 0.5834
(repeat SD 0.0112), compared with adult residual AUC 0.7340 (SD 0.0028) under
this audit's missing-value handling. These are descriptive results about labels
in these samples, not causal demographic effects or clinical accuracy.

Confirmed: the observed tables and model results. Supported inference: direct
score-label generation is plausible. Still unknown: original label-generation
procedure, clinician confirmation, participant identity, recruitment effects,
and diagnostic generalization. Ask the dataset author about these gaps before
stronger claims. The audit also needs testing on data with independently assessed
outcomes to distinguish score dependence from genuine screening utility.
