from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.evaluate import (
    FEATURES, bootstrap_intervals, compare_tree_depths, installed_packages, make_model, measure,
    predictor_groups, questionnaire_coefficients, repository_path, score_diagnostics,
)
from src.train import AQ_COLUMNS, CATEGORICAL_COLUMNS, ROOT_DIR


def test_duplicate_groups_ignore_target_and_stay_in_one_fold():
    records = []
    for i in range(20):
        row = {**{c: int(i % 2) for c in AQ_COLUMNS}, "age": 20 + i,
               **{c: "same" for c in CATEGORICAL_COLUMNS}, "sum_score": 10 * (i % 2)}
        records.extend([row, row.copy()])
    X = pd.DataFrame(records)
    groups = predictor_groups(X)
    y = X[AQ_COLUMNS[0]].to_numpy()
    assert groups[0] == groups[1]
    for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, groups):
        assert set(groups[train]).isdisjoint(groups[test])


def test_imputer_and_oversampler_use_only_training_rows():
    X = pd.DataFrame({c: [0, 0, 0, 1] for c in AQ_COLUMNS})
    X["age"] = [20., 22., np.nan, 30.]
    X["sum_score"] = X[AQ_COLUMNS].sum(axis=1)
    for c in CATEGORICAL_COLUMNS:
        X[c] = "training"
    model = make_model("logistic", *FEATURES["combined_without_sum"], 42)
    model.fit(X, [0, 0, 0, 1])
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_[-1] == 22
    indices = model.named_steps["oversample"].sample_indices_.copy()
    test = X.iloc[:1].copy()
    test["age"] = 99999
    test["gender"] = "unseen"
    model.predict(test)
    assert imputer.statistics_[-1] == 22
    np.testing.assert_array_equal(indices, model.named_steps["oversample"].sample_indices_)
    assert indices.max() < len(X)


def test_confusion_metrics_and_group_uncertainty():
    result = measure([0, 0, 1, 1], [0, 1, 0, 1], [0., .8, .2, 1.])
    assert result["tn"] == result["fp"] == result["fn"] == result["tp"] == 1
    assert result["precision"] == result["recall"] == result["f1"] == .5
    first = bootstrap_intervals([0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], 100, 42)
    assert first == bootstrap_intervals([0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], 100, 42)
    assert 0 <= first["accuracy"][0] <= first["accuracy"][1] <= 1


def test_feature_ablation_excludes_all_score_information():
    numeric, categorical = FEATURES["non_questionnaire"]
    assert not set(AQ_COLUMNS + ["sum_score", "result", "Class/ASD"]) & set(numeric + categorical)
    assert "sum_score" not in FEATURES["combined_without_sum"][0]


def test_score_scan_distinguishes_referral_rule_from_label_rule():
    frame = pd.DataFrame({"sum_score": range(11), "Class/ASD": ["NO"] * 7 + ["YES"] * 4})
    table, scan = score_diagnostics(frame)
    assert table.loc[table.aq_score == 6, "NO"].item() == 1
    assert table.loc[table.aq_score == 7, "positive_fraction"].item() == 1
    assert scan.loc[scan.threshold == 6, "disagreements"].item() == 1
    assert scan.loc[scan.threshold == 7, "disagreements"].item() == 0
    assert scan.loc[scan.threshold == 0, "tn"].item() == 0
    assert scan.loc[scan.threshold == 11, "recall"].item() == 0


def test_raw_coefficients_reproduce_standardized_model_logits():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.integers(0, 2, (80, 10)), columns=AQ_COLUMNS)
    y = (X.sum(axis=1) >= 6).astype(int)
    model = make_model("logistic", *FEATURES["questionnaire_only"], 42).fit(X, y)
    coefficients = questionnaire_coefficients(model)
    beta = np.array([row["raw_item_coefficient"] for row in coefficients])
    reconstructed = X.to_numpy() @ beta + coefficients[0]["raw_intercept"]
    np.testing.assert_allclose(reconstructed, model.decision_function(X), atol=1e-12)


def test_depth_comparison_scores_only_the_supplied_validation_records():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.integers(0, 2, (80, 10)), columns=AQ_COLUMNS)
    X["sum_score"] = X[AQ_COLUMNS].sum(axis=1)
    X["age"] = 30
    for col in CATEGORICAL_COLUMNS:
        X[col] = "same"
    y = (X.sum_score >= 6).astype(int).to_numpy()
    splits = [(np.arange(40), np.arange(40, 80)), (np.arange(40, 80), np.arange(40))]
    pooled, folds = compare_tree_depths(X, y, splits, 42, [1, 4])
    assert len(pooled) == 4
    assert len(folds) == 8
    for row in pooled:
        assert sum(row[key] for key in ["tn", "fp", "fn", "tp"]) == 80
    for row in folds:
        assert row["learned_depth"] <= int(row["max_depth"])
        assert sum(row[key] for key in ["tn", "fp", "fn", "tp"]) == 40


def test_archive_path_does_not_depend_on_how_the_run_was_launched():
    assert repository_path(ROOT_DIR / "research/sources/uci_adult_426.zip") == "research/sources/uci_adult_426.zip"
    assert repository_path(Path("/somewhere/else/archive.zip")) == "archive.zip"


def test_environment_capture_leaves_out_pip():
    packages = installed_packages()
    assert packages and not any(name.lower().startswith("pip==") for name in packages)
