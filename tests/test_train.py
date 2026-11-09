import json

import pandas as pd
import pytest

from src.train import (
    AQ_COLUMNS,
    CATEGORICAL_COLUMNS,
    build_pipeline,
    load_dataset,
    train_model,
)


def make_raw_dataset(rows: int = 50) -> pd.DataFrame:
    genders = ["m", "f"]
    ethnicities = ["White-European", "Asian", "?"]
    countries = ["United States", "Spain", "others"]
    records = []
    for i in range(rows):
        is_yes = i < 15
        records.append(
            {
                **{column: 1 if is_yes else 0 for column in AQ_COLUMNS},
                "age": 20 + (i % 30),
                "gender": genders[i % len(genders)],
                "ethnicity": ethnicities[i % len(ethnicities)],
                "jundice": "yes" if i % 4 == 0 else "no",
                "austim": "no",
                "contry_of_res": countries[i % len(countries)],
                "used_app_before": "no",
                "result": 7 if is_yes else 1,
                "age_desc": "18 and more",
                "relation": "Self",
                "ID": i,
                "Class/ASD": "YES" if is_yes else "NO",
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "train.csv"
    make_raw_dataset().to_csv(path, index=False)
    return path


def test_load_dataset_renames_and_engineers_columns(raw_csv):
    df = load_dataset(raw_csv)
    assert "jaundice" in df.columns
    assert "country_of_res" in df.columns
    assert "jundice" not in df.columns
    assert "sum_score" in df.columns
    assert df.loc[0, "sum_score"] == df.loc[0, AQ_COLUMNS].sum()
    assert "Unknown" in df["ethnicity"].values
    assert "Others" in df["country_of_res"].values


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.csv")


def test_build_pipeline_has_expected_steps():
    pipeline = build_pipeline()
    step_names = [name for name, _ in pipeline.steps]
    assert step_names == ["preprocess", "oversample", "classifier"]
    preprocessor = pipeline.named_steps["preprocess"]
    transformer_columns = {name: columns for name, _, columns in preprocessor.transformers}
    assert transformer_columns["categorical"] == CATEGORICAL_COLUMNS


def test_train_model_writes_artifacts_with_cv_and_baseline(raw_csv, tmp_path):
    model_path = tmp_path / "model.pkl"
    metrics_path = tmp_path / "metrics.json"

    metrics = train_model(data_path=raw_csv, model_path=model_path, metrics_path=metrics_path)

    assert model_path.exists()
    assert metrics_path.exists()

    saved_metrics = json.loads(metrics_path.read_text())
    assert saved_metrics == metrics

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["baseline_accuracy"] <= 1
    assert metrics["training_rows"] + metrics["test_rows"] == 50

    cv = metrics["cross_validation"]
    assert cv["folds"] >= 2
    assert 0 <= cv["accuracy_mean"] <= 1
    assert 0 <= cv["f1_mean"] <= 1
