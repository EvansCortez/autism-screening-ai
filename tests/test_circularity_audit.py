import json

import numpy as np
import pandas as pd
import pytest

from src.circularity_audit import ROOT, load_configured_data, prepare, run_audit, score_dependence


def generic_config():
    return {"target_column": "outcome", "item_columns": ["item_x", "item_y"],
            "label_mapping": {"negative": 0, "positive": 1}, "allowed_item_values": [0, 1],
            "recorded_score_column": "total", "group_columns": ["item_x", "item_y"],
            "residual_features": {"numeric": [], "categorical": []}}


def test_score_purity_does_not_imply_a_threshold_or_clinical_provenance():
    result, _, _ = score_dependence([0, 1, 2], [0, 1, 0], [0, 1, 2, 3])
    assert result["empirical_score_group_purity"] == 1
    assert result["conditional_label_entropy_bits"] == 0
    assert result["singleton_score_groups"] == 3
    assert result["exact_threshold_agreement_observed"] is False


def test_mixed_labels_have_positive_entropy_and_imperfect_purity():
    result, _, _ = score_dependence([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 2])
    assert result["conditional_label_entropy_bits"] == 1
    assert result["empirical_score_group_purity"] == .5
    assert result["mixed_label_scores"] == 2


def test_generic_column_names_and_conflicting_duplicates_are_not_dropped():
    frame = pd.DataFrame({"item_x": [0, 0, 1, 1], "item_y": [0, 0, 1, 1],
                          "outcome": ["negative", "positive", "positive", "positive"],
                          "total": [0, 0, 2, 2]})
    X, y, groups = prepare(frame, generic_config())
    assert len(X) == 4
    assert groups[0] == groups[1]
    assert y[0] != y[1]
    np.testing.assert_array_equal(X.audit_sum, [0, 0, 2, 2])
    frame.loc[0, "item_x"] = np.nan
    with pytest.raises(ValueError, match="Missing or invalid"):
        prepare(frame, generic_config())


def test_rejects_target_leakage_and_score_information_in_residual_features():
    frame = pd.DataFrame({"item_x": [0, 1], "item_y": [0, 1],
                          "outcome": ["negative", "positive"], "total": [0, 2]})
    config = generic_config()
    config["residual_features"]["numeric"] = ["item_x"]
    with pytest.raises(ValueError, match="Residual features"):
        prepare(frame, config)
    config["residual_features"]["numeric"] = ["outcome"]
    with pytest.raises(ValueError, match="Target/recorded score"):
        prepare(frame, config)


def test_rejects_changed_source_and_missing_documentation(tmp_path):
    config = json.loads((ROOT / "research/datasets/child.json").read_text())
    config["input"]["sha256"] = "invalid"
    with pytest.raises(ValueError, match="checksum"):
        load_configured_data(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"documentation": {}}))
    with pytest.raises(ValueError, match="Document the dataset"):
        run_audit(path, tmp_path / "results")


def test_verified_child_archive_contract():
    config = json.loads((ROOT / "research/datasets/child.json").read_text())
    frame, digest = load_configured_data(config)
    assert digest == config["input"]["sha256"]
    X, y, groups = prepare(frame, config)
    assert len(frame) == 292
    assert int(y.sum()) == 141
    assert len(np.unique(groups)) == 289
    assert np.array_equal((X.audit_sum >= 7).astype(int), y)
    assert int(((X.audit_sum >= 6).astype(int) != y).sum()) == 40
