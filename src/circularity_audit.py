"""Configurable audit of score/label dependence; empirical agreement is not provenance."""

import argparse
import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path

import imblearn
import numpy as np
import pandas as pd
import scipy
import sklearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from scipy.io import arff
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluate import measure

ROOT = Path(__file__).resolve().parents[1]


def load_configured_data(config):
    source = config["input"]
    path = ROOT / source["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError("Input checksum differs from the documented dataset version")
    if source["format"] == "csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    elif source["format"] == "zip_arff":
        with zipfile.ZipFile(path) as archive:
            data, _ = arff.loadarff(io.StringIO(archive.read(source["member"]).decode()))
        frame = pd.DataFrame(data)
        for column in frame:
            frame[column] = frame[column].map(lambda x: x.decode() if isinstance(x, bytes) else x)
    else:
        raise ValueError("Supported formats are csv and zip_arff")
    return frame, digest


def prepare(frame, config):
    target = config["target_column"]
    items = config["item_columns"]
    numeric = config["residual_features"]["numeric"]
    categorical = config["residual_features"]["categorical"]
    group_columns = config["group_columns"]
    recorded = config.get("recorded_score_column")
    if not items or len(set(items)) != len(items):
        raise ValueError("Items must be nonempty and distinct")
    forbidden = {target, recorded, "audit_sum"} - {None}
    if forbidden.intersection(items + numeric + categorical + group_columns):
        raise ValueError("Target/recorded score cannot be predictors or grouping columns")
    if set(items).intersection(numeric + categorical):
        raise ValueError("Residual features must exclude all questionnaire items")
    if not set(items).issubset(group_columns):
        raise ValueError("Grouping must include the questionnaire items")
    required = set(items + numeric + categorical + group_columns + [target])
    if recorded:
        required.add(recorded)
    if not required.issubset(frame):
        raise ValueError(f"Missing configured columns: {sorted(required - set(frame))}")
    labels = frame[target].astype(str).str.strip()
    mapping = config["label_mapping"]
    if set(mapping.values()) != {0, 1} or not labels.isin(mapping).all():
        raise ValueError("Target must have explicitly documented binary label mapping")
    if len(frame) == 0 or set(labels.map(mapping)) != {0, 1}:
        raise ValueError("Audit requires nonempty data containing both mapped classes")
    if not np.isfinite(np.asarray(config["allowed_item_values"], dtype=float)).all():
        raise ValueError("Allowed scored-item values must be finite")
    item_data = frame[items].apply(pd.to_numeric, errors="coerce")
    if not item_data.isin(config["allowed_item_values"]).all().all():
        raise ValueError("Missing or invalid scored items: resolve before computing a sum")
    X = frame.drop(columns=[target]).copy()
    X[items] = item_data
    for column in numeric:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in categorical:
        X[column] = X[column].astype(str).str.strip().replace({"?": np.nan, "": np.nan, "nan": np.nan})
    X["audit_sum"] = item_data.sum(axis=1)
    # Participant identity is not inferred; these are only matching-record groups.
    groups = pd.factorize(pd.util.hash_pandas_object(X[group_columns], index=False))[0]
    return X, labels.map(mapping).to_numpy(dtype=int), groups


def score_dependence(score, target, thresholds):
    values = np.asarray(target, dtype=int)
    counts = pd.crosstab(pd.Series(np.asarray(score), name="score"), pd.Series(values))
    counts = counts.reindex(columns=[0, 1], fill_value=0).rename(columns={0: "negative", 1: "positive"})
    totals = counts.sum(axis=1)
    probabilities = counts.div(totals, axis=0).to_numpy()
    logs = np.zeros_like(probabilities)
    np.log2(probabilities, out=logs, where=probabilities > 0)
    entropy = float(np.average(-(probabilities * logs).sum(axis=1), weights=totals))
    purity = float(counts.max(axis=1).sum() / totals.sum())
    counts["total"] = totals
    counts["positive_fraction"] = counts.positive / totals
    scan = pd.DataFrame([dict(threshold=float(t), disagreements=int(((np.asarray(score) >= t) != values).sum()),
                              **measure(values, (np.asarray(score) >= t).astype(int), score)) for t in thresholds])
    minimum = int(scan.disagreements.min())
    summary = dict(conditional_label_entropy_bits=entropy, empirical_score_group_purity=purity,
                   mixed_label_scores=int(((counts.negative > 0) & (counts.positive > 0)).sum()),
                   singleton_score_groups=int((totals == 1).sum()), observed_score_groups=len(counts),
                   minimum_threshold_disagreements=minimum,
                   best_descriptive_thresholds=scan.loc[scan.disagreements == minimum, "threshold"].tolist(),
                   exact_threshold_agreement_observed=minimum == 0)
    return summary, counts.reset_index(), scan


def logistic_model(numeric, categorical, seed):
    transforms = []
    if numeric:
        transforms.append(("numeric", SkPipeline([("impute", SimpleImputer(strategy="median")),
                                                  ("scale", StandardScaler())]), numeric))
    if categorical:
        transforms.append(("categorical", SkPipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                                       ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    return Pipeline([("preprocess", ColumnTransformer(transforms)),
                     ("oversample", RandomOverSampler(random_state=seed)),
                     ("classifier", LogisticRegression(max_iter=1000, random_state=seed))])


def model_comparison(X, y, groups, config):
    evaluation = config["evaluation"]
    numeric = config["residual_features"]["numeric"]
    categorical = config["residual_features"]["categorical"]
    methods = ["dummy", "documented_score_rule", "score_only_logistic", "questionnaire_logistic"]
    if numeric or categorical:
        methods.append("residual_logistic")
    pooled, per_fold, assignments, predictions = [], [], [], []
    for seed in evaluation["seeds"]:
        folds = evaluation["folds"]
        if folds < 2 or len(np.unique(groups)) < folds:
            raise ValueError("Not enough matching-record groups for configured validation")
        splits = list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups))
        for fold, (train, test) in enumerate(splits):
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("Matching records cross fold boundary")
            if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                raise ValueError("Both classes are required in every fold; reduce fold count")
            assignments.extend(dict(seed=seed, fold=fold, row_index=int(i), group=int(groups[i])) for i in test)
        for method in methods:
            pred, ranking = np.zeros(len(X), dtype=int), np.zeros(len(X))
            for fold, (train, test) in enumerate(splits):
                if method == "documented_score_rule":
                    ranking[test] = X.iloc[test].audit_sum
                    pred[test] = (ranking[test] >= config["documented_rule"]["threshold"]).astype(int)
                else:
                    if method == "dummy":
                        estimator = DummyClassifier(strategy="most_frequent")
                    elif method == "score_only_logistic":
                        estimator = logistic_model(["audit_sum"], [], seed)
                    elif method == "questionnaire_logistic":
                        estimator = logistic_model(config["item_columns"], [], seed)
                    else:
                        estimator = logistic_model(numeric, categorical, seed)
                    estimator.fit(X.iloc[train], y[train])
                    pred[test] = estimator.predict(X.iloc[test])
                    ranking[test] = estimator.predict_proba(X.iloc[test])[:, 1]
                per_fold.append(dict(seed=seed, fold=fold, method=method, **measure(y[test], pred[test], ranking[test])))
            pooled.append(dict(seed=seed, method=method, **measure(y, pred, ranking)))
            predictions.extend(dict(seed=seed, method=method, row_index=int(i), label=int(y[i]),
                                    prediction=int(pred[i]), ranking_score=float(ranking[i])) for i in range(len(X)))
    return dict(pooled_metrics=pd.DataFrame(pooled), fold_metrics=pd.DataFrame(per_fold),
                fold_assignment=pd.DataFrame(assignments), predictions=pd.DataFrame(predictions))


def run_audit(config_path, output):
    config = json.loads(config_path.read_text())
    documentation = config.get("documentation", {})
    for field in ["source_url", "citation", "population", "label_definition_status", "label_evidence", "limitations"]:
        if not documentation.get(field):
            raise ValueError(f"Document the dataset before auditing: missing {field}")
    if not config.get("documented_rule", {}).get("source_url"):
        raise ValueError("Fixed score rule needs a documented source URL")
    if not config.get("thresholds") or not all(np.isfinite(config["thresholds"])):
        raise ValueError("Provide finite descriptive threshold candidates")
    frame, digest = load_configured_data(config)
    X, y, groups = prepare(frame, config)
    dependence, table, scan = score_dependence(X.audit_sum, y, config["thresholds"])
    recorded = config.get("recorded_score_column")
    duplicates = pd.DataFrame({"group": groups, "label": y}).groupby("group").label.nunique()
    report = {
        "dataset": config["name"], "rows": len(frame),
        "class_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "duplicate_full_rows": int(frame.duplicated().sum()),
        "duplicate_group_rows": int(len(groups) - len(np.unique(groups))),
        "conflicting_matching_record_groups": int(duplicates.gt(1).sum()),
        "recorded_score_disagreements": int((pd.to_numeric(frame[recorded], errors="coerce") != X.audit_sum).sum()) if recorded else None,
        "score_dependence": dependence, "declared_source_documentation": documentation,
        "confirmed": "Observed counts, score agreement, group overlap checks, and model metrics only",
        "not_established": "Label-generation procedure, clinical validity, participant identity, or external clinical generalization",
        "interpretation": "Exact empirical agreement is evidence of score-label dependence, not proof of how labels were collected. High purity in small or singleton score groups may be misleading.",
        "uncertainty": "Repeated/fold metric standard deviations are descriptive; they are not independent confidence intervals",
    }
    age = config.get("age_audit")
    if age:
        values = pd.to_numeric(frame[age["column"]], errors="coerce")
        report["age_missing_or_outside_configured_range"] = int((~values.between(age["minimum"], age["maximum"])).sum())
        report["age_range_status"] = age["range_status"]
    results = model_comparison(X, y, groups, config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    table.to_csv(output / "label_by_score.csv", index=False)
    scan.to_csv(output / "threshold_scan.csv", index=False)
    for name, values in results.items():
        values.to_csv(output / f"{name}.csv", index=False)
    results["pooled_metrics"].groupby("method")[["accuracy", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"]).to_csv(output / "summary.csv")
    settings = dict(config=config, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    input_sha256=digest, audit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    metric_code_sha256=hashlib.sha256((ROOT / "src/evaluate.py").read_bytes()).hexdigest(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, imblearn=imblearn.__version__),
                    logistic_parameters=logistic_model(["audit_sum"], [], config["evaluation"]["seeds"][0]).named_steps["classifier"].get_params())
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    print(json.dumps({"dataset": config["name"], "rows": len(frame), **dependence}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.config, args.output)


if __name__ == "__main__":
    main()
