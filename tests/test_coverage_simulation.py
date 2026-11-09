import numpy as np
import pandas as pd
import pytest

from src.coverage_simulation import simulate


def toy_items():
    rng = np.random.default_rng(0)
    return (rng.random((300, 10)) < 0.5).astype(int)


def small_run(seed=1):
    return simulate(toy_items(), seed=seed, configurations=[(80, 12)], flips=[0.0, 0.1], fresh_rows=800)


def test_simulation_is_reproducible_and_reports_valid_proportions():
    first, second = small_run(), small_run()
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2 and (first.simulated_datasets == 12).all()
    for column in ["exact_coverage_of_expected_accuracy", "exact_coverage_of_each_dataset_fold_accuracy",
                   "wilson_coverage_of_expected_accuracy", "fraction_with_zero_errors"]:
        assert first[column].between(0, 1).all()
    assert first.mean_cv_accuracy.between(0.5, 1).all()


def test_label_noise_lowers_accuracy_and_noise_free_case_is_well_above_chance():
    result = small_run().set_index("flip_probability")
    assert result.loc[0.0, "mean_cv_accuracy"] > result.loc[0.1, "mean_cv_accuracy"]
    assert result.loc[0.0, "expected_fold_model_accuracy"] > result.loc[0.1, "expected_fold_model_accuracy"] + 0.03
    assert result.loc[0.0, "expected_fold_model_accuracy"] > 0.8


def test_different_seeds_give_different_draws():
    assert not small_run(1).mean_cv_accuracy.equals(small_run(2).mean_cv_accuracy)
