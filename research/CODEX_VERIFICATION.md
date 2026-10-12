# Codex verification and corrections

Reviewed Claude's handoff at commit `24e84a4` on 2026-10-12, after the project lead
confirmed Claude had finished editing. Codex acquired the shared lock before changes.
This record supplements, rather than rewrites, Claude's historical `REVIEW.md`.

## Confirmed checks

- All 38 tests passed after four regression tests were added to Claude's 34-test snapshot.
- The adult and child circularity outputs and primary evaluation outputs regenerated
  byte-for-byte unchanged. The adult label equals the item total >= 7 on all 704 records;
  the documented >= 6 referral rule disagrees on 70 records.
- `scripts/verify_outputs.py` independently computes accuracy, precision, recall, F1,
  ROC-AUC and confusion counts with sklearn from saved predictions, not from the study's
  metric helpers. It also checks raw totals, source hashes and the saved null summary.
- Code inspection confirms whole-procedure bootstrap copies keep their original predictor
  group, so copies cannot cross folds. Preprocessing and oversampling occur inside each fit.
- Country-cluster bootstrap resamples fixed predictions, not newly fitted models. It is a
  sensitivity analysis conditional on those predictions, not complete training uncertainty.
- Codex viewed all six PNG figures. Labels and plotted values were readable. Figure 6
  required the range correction below. Venue-specific LaTeX table layout remains untested.
- No serving-model, API, training or static-app file changed in this pass. The existing PR
  includes an earlier training-CV commit, explicitly disclosed in its description.
- GitHub reported CI success for `24e84a4`. This tested dependencies and unit tests on Linux;
  it did not run the full research reproduction or Docker.

## Defects corrected

1. **Permutation statistic mismatch.** The observed statistic averaged three repeated
   out-of-fold AUCs; each null draw previously used only one repeat. Null draws now average
   the identical seeds, with individual values saved in `permutation_draws.csv`.
2. **Post-hoc threshold intervals.** An unadjusted interval was shown for the >= 7 rule on
   the original holdout even though those labels were used in threshold discovery. It is
   now a descriptive count. The full-data pre-specified-rule bound was removed; the
   12-candidate-adjusted bound remains explicitly conditional on independent rows.
3. **Unsupported certainty about bootstrap bias.** About 63% distinct rows in a bootstrap
   resample does not establish that its spread necessarily overstates uncertainty here.
   Coverage and direction of bias remain unverified.
4. **Figure range.** The label-shuffle band was mean-to-maximum, not the advertised range.
   It now spans minimum-to-maximum and the axis shows both ends.
5. **Evidence wording.** `repo` tags no longer automatically imply independent verification.
   Absence of independently documented diagnoses is not asserted as proof that no diagnosis
   was ever made. The methods guide and paper outline now preserve that distinction.

## Added robustness check

`leave_country_out` trains non-questionnaire logistic regression on all other recorded
countries, fits preprocessing and oversampling on training rows, and predicts the held-out
country. Outputs include predictions, pooled metrics by seed, and country-level confusion
counts. Single-class countries have undefined AUC. The training/test country separation has
a regression test. This is an internal geographic check, not external clinical validation.

## Limits and next handoff

Unrestricted label shuffling assumes exchangeability. Duplicate records and undocumented
recruitment dependencies make that assumption uncertain; the permutation result is
exploratory association evidence. No causal explanation of country or ethnicity is supplied.
The label-generation procedure, item encoding, novelty and clinical outcome validity remain
unresolved. Scratch-only interval coverage simulations are not independently reproduced here.

Codex independently reviewed Claude's implementation, then authored the corrections above.
Claude or the project lead should independently inspect those corrections and the new
country-held-out analysis. A successful regeneration or regression test is not itself an
independent scientific review. Final run and committed-checkout evidence are recorded in
`TASKS.md`.
