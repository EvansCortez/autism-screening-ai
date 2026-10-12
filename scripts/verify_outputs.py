"""Check saved study outputs from predictions and raw data, independently of metric helpers."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol = json.loads((ROOT / "research/protocol.json").read_text())
    for path, expected in protocol["inputs"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path

    evaluation = ROOT / "research/evaluation"
    predictions = pd.read_csv(evaluation / "predictions.csv")
    metrics = pd.read_csv(evaluation / "pooled_metrics.csv")
    keys = ["population", "method", "seed"]
    assert not predictions.duplicated(keys + ["row_index"]).any()
    assert len(predictions.groupby(keys)) == len(metrics)
    for key, group in predictions.groupby(keys):
        row = metrics.set_index(keys).loc[key]
        y, pred = group.target, group.prediction
        computed = dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                        recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                        roc_auc=roc_auc_score(y, group.ranking_score))
        for name, value in computed.items():
            np.testing.assert_allclose(row[name], value, rtol=0, atol=1e-12, err_msg=str(key))
        np.testing.assert_array_equal(row[["tn", "fp", "fn", "tp"]].to_numpy(), confusion_matrix(y, pred, labels=[0, 1]).ravel())

    raw = pd.read_csv(ROOT / "data/train.csv")
    items = [f"A{i}_Score" for i in range(1, 11)]
    total = raw[items].sum(axis=1)
    np.testing.assert_array_equal(total.to_numpy(), raw.result.to_numpy())
    np.testing.assert_array_equal((total >= 7).to_numpy(), (raw["Class/ASD"] == "YES").to_numpy())
    assert int(((total >= 6) != (raw["Class/ASD"] == "YES")).sum()) == 70

    uncertainty = ROOT / "research/uncertainty"
    draws = pd.read_csv(uncertainty / "permutation_draws.csv").mean_repeated_oof_auc
    null = pd.read_csv(uncertainty / "permutation_null.csv").iloc[0]
    for column, value in [("null_mean", draws.mean()), ("null_min", draws.min()), ("null_max", draws.max()),
                          ("null_95th_percentile", draws.quantile(0.95))]:
        np.testing.assert_allclose(null[column], value, rtol=0, atol=1e-12)
    np.testing.assert_allclose(null.p_value_at_least_observed,
                               (1 + (draws >= null.observed_auc_mean_over_repeats).sum()) / (len(draws) + 1))
    observed = metrics.query("population == 'primary' and method == 'logistic/non_questionnaire'").roc_auc.mean()
    np.testing.assert_allclose(null.observed_auc_mean_over_repeats, observed, rtol=0, atol=1e-12)
    settings = json.loads((uncertainty / "settings.json").read_text())
    assert settings["code_sha256"] == hashlib.sha256((ROOT / "src/uncertainty.py").read_bytes()).hexdigest()
    assert settings["evaluation_predictions_sha256"] == hashlib.sha256((evaluation / "predictions.csv").read_bytes()).hexdigest()
    country_predictions = pd.read_csv(uncertainty / "leave_country_out_predictions.csv")
    country_metrics = pd.read_csv(uncertainty / "leave_country_out_metrics.csv").set_index("seed")
    for seed, group in country_predictions.groupby("seed"):
        assert len(group) == len(raw) and not group.row_index.duplicated().any()
        np.testing.assert_array_equal(group.sort_values("row_index").country, raw["contry_of_res"].replace({"?": "Unknown", "others": "Others"}))
        np.testing.assert_allclose(country_metrics.loc[seed, "roc_auc"], roc_auc_score(group.target, group.ranking_score), atol=1e-12, rtol=0)
        np.testing.assert_allclose(country_metrics.loc[seed, "accuracy"], accuracy_score(group.target, group.prediction), atol=1e-12, rtol=0)
    print(f"Verified {len(metrics)} pooled metric rows, raw score-label agreement, permutation summaries, country-held-out metrics and provenance hashes.")


if __name__ == "__main__":
    main()
