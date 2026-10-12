import json

import numpy as np
import pandas as pd
import pytest

from src.train import AQ_COLUMNS, CATEGORICAL_COLUMNS
from src.uncertainty import (
    cluster_bootstrap,
    exact_interval,
    interval_table,
    leave_country_out,
    missing_block_table,
    nested_threshold,
    permutation_null,
    subgroup_table,
    whole_procedure_bootstrap,
    wilson_interval,
    zero_error_upper_bound,
)


def synthetic_frame(rows=160, seed=0, rule=None):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.integers(0, 2, rows) for c in AQ_COLUMNS})
    frame["sum_score"] = frame[AQ_COLUMNS].sum(axis=1)
    frame["age"] = rng.integers(18, 60, rows).astype(float)
    for column in CATEGORICAL_COLUMNS:
        frame[column] = rng.choice(["a", "b", "c"], rows)
    frame["ethnicity"] = rng.choice(["a", "b", "Unknown"], rows)
    frame["relation"] = np.where(frame["ethnicity"] == "Unknown", "Unknown", "Self")
    frame["austim"] = rng.choice(["no", "yes"], rows)
    threshold = 5 if rule is None else rule
    y = (frame["sum_score"] >= threshold).astype(int).to_numpy()
    frame["Class/ASD"] = np.where(y == 1, "YES", "NO")
    groups = pd.factorize(pd.Series(list(zip(*[frame[c] for c in AQ_COLUMNS + ["age"] + CATEGORICAL_COLUMNS]))))[0]
    return frame, y, groups


def test_zero_error_bounds_match_closed_forms_and_adjustment_is_conservative():
    assert zero_error_upper_bound(704, two_sided=True) == pytest.approx(1 - 0.025 ** (1 / 704))
    assert round(zero_error_upper_bound(704, two_sided=True), 4) == 0.0052
    assert round(zero_error_upper_bound(141, two_sided=True), 4) == 0.0258
    assert zero_error_upper_bound(704, candidates=12) > zero_error_upper_bound(704, candidates=1)


def test_exact_and_wilson_intervals_bracket_the_estimate():
    for correct, total in [(0, 50), (25, 50), (634, 704), (704, 704)]:
        for method in (exact_interval, wilson_interval):
            low, high = method(correct, total)
            assert 0 <= low <= correct / total <= high <= 1
    low, _ = exact_interval(704, 704)
    assert low == pytest.approx(0.025 ** (1 / 704))


def test_cluster_bootstrap_matches_binomial_for_singletons_and_detects_clustering():
    rng = np.random.default_rng(1)
    singles = rng.random(2000) < 0.7
    binomial = np.sqrt(singles.mean() * (1 - singles.mean()) / 2000)
    result = cluster_bootstrap(singles, np.arange(2000), 2000, 3)
    assert result["se"] == pytest.approx(binomial, rel=0.15)
    clustered = np.repeat(rng.random(40) < 0.6, 25)
    result = cluster_bootstrap(clustered, np.repeat(np.arange(40), 25), 2000, 3)
    assert result["se"] > 3 * np.sqrt(clustered.mean() * (1 - clustered.mean()) / 1000)


def test_whole_procedure_bootstrap_is_reproducible_and_resamples_about_two_thirds():
    frame, y, groups = synthetic_frame()
    first = whole_procedure_bootstrap(frame, y, groups, "logistic", "combined_without_sum", 3, 11)
    second = whole_procedure_bootstrap(frame, y, groups, "logistic", "combined_without_sum", 3, 11)
    pd.testing.assert_frame_equal(first, second)
    assert first.distinct_row_fraction.between(0.5, 0.75).all()
    assert first.accuracy.between(0, 1).all()


def test_permutation_null_centers_near_chance_for_uninformative_features():
    frame, y, groups = synthetic_frame(rows=200, seed=5)
    values = permutation_null(frame, y, groups, "logistic", "non_questionnaire", 12, 4)
    assert len(values) == 12 and abs(values.mean() - 0.5) < 0.1


def test_permutation_null_uses_all_observed_repeat_seeds(monkeypatch):
    frame, y, groups = synthetic_frame()
    calls = []

    def fake_oof(frame, labels, groups, kind, feature_set, seed):
        calls.append((seed, labels.copy()))
        return labels, labels.astype(float) if seed == 42 else 1 - labels.astype(float)

    monkeypatch.setattr("src.uncertainty.pooled_oof", fake_oof)
    values = permutation_null(frame, y, groups, "logistic", "non_questionnaire", 2, 4, [42, 137])
    assert list(values) == [0.5, 0.5]
    assert [seed for seed, _ in calls] == [42, 137, 42, 137]
    np.testing.assert_array_equal(calls[0][1], calls[1][1])


def test_posthoc_threshold_has_no_unadjusted_interval():
    frame, y, _ = synthetic_frame(rule=7)
    counts = dict(tn=100, fp=0, fn=0, tp=41)
    audit = {"legacy_holdout_comparison": {name: counts for name in ["logistic", "aq_ge6", "aq_ge7"]}}
    table = interval_table(audit, frame, y, list(range(12)))
    holdout = table[table.quantity == "Rule total >= 7, original holdout"].iloc[0]
    assert holdout.method == "count only" and pd.isna(holdout.lower)
    assert not table.quantity.str.contains("single pre-specified").any()


def test_leave_country_out_keeps_countries_out_of_training(monkeypatch):
    frame, y, _ = synthetic_frame(rows=160)
    country_indices = []

    class FakeModel:
        def fit(self, training, labels):
            self.countries = set(training.country_of_res)
            return self

        def predict(self, testing):
            assert self.countries.isdisjoint(set(testing.country_of_res))
            country_indices.extend(testing.index.tolist())
            return np.zeros(len(testing), dtype=int)

        def predict_proba(self, testing):
            return np.tile([0.6, 0.4], (len(testing), 1))

    monkeypatch.setattr("src.uncertainty.make_model", lambda *args: FakeModel())
    metrics, countries, predictions = leave_country_out(frame, y, [42])
    assert sorted(country_indices) == list(range(len(frame)))
    assert metrics.iloc[0].n == len(frame) and countries.n.sum() == len(frame)
    assert not predictions.row_index.duplicated().any()


def test_nested_threshold_recovers_an_exact_rule_without_looking_at_validation_rows():
    frame, y, groups = synthetic_frame(rows=300, seed=7, rule=6)
    result = nested_threshold(frame, y, groups, [1, 2], list(range(12)))
    assert (result.out_of_fold_errors == 0).all()
    assert all(set(json.loads(row)) == {6} for row in result.thresholds_chosen_per_fold)


def test_descriptive_tables_report_counts_and_do_not_infer_causes():
    frame, y, _ = synthetic_frame(rows=120, seed=2)
    block = missing_block_table(frame, y)
    assert block.n.sum() == len(frame)
    table = subgroup_table(frame, y)
    assert (table.n >= table.label_yes).all()
    assert table.wilson_lower.le(table.label_yes_rate).all() and table.wilson_upper.ge(table.label_yes_rate).all()
    assert table.note.str.contains("descriptive").all()
