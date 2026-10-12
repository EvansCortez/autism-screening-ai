"""Research evaluation of UCI screening labels; does not export serving models."""

import argparse
import csv
import hashlib
import io
from importlib.metadata import distributions
import json
import platform
import zipfile
from pathlib import Path

import imblearn
import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.train import AQ_COLUMNS, CATEGORICAL_COLUMNS, DATA_PATH, ROOT_DIR, build_pipeline, load_dataset

FEATURES = {
    "combined_with_sum": (AQ_COLUMNS + ["age", "sum_score"], CATEGORICAL_COLUMNS),
    "combined_without_sum": (AQ_COLUMNS + ["age"], CATEGORICAL_COLUMNS),
    "questionnaire_only": (AQ_COLUMNS, []),
    "non_questionnaire": (["age"], CATEGORICAL_COLUMNS),
}
METRICS = ["accuracy", "precision", "recall", "specificity", "f1"]
TREE_DEPTHS = (1, 2, 3, 4, 6, 8, 10, None)


def repository_path(path):
    """Repository-relative form of a path, so outputs do not depend on how a run was launched."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return resolved.name


def installed_packages():
    """Installed distributions, excluding pip: tooling that varies between environments, not a dependency."""
    return sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()
                  if d.metadata["Name"] and d.metadata["Name"].lower() != "pip")


def score_diagnostics(frame):
    """Descriptive score/label relationships; not threshold selection for diagnosis."""
    target = frame["Class/ASD"].map({"NO": 0, "YES": 1}).to_numpy()
    table = pd.crosstab(frame["sum_score"], frame["Class/ASD"])
    table = table.reindex(index=range(11), columns=["NO", "YES"], fill_value=0)
    table.index.name = "aq_score"
    table["total"] = table["NO"] + table["YES"]
    table["positive_fraction"] = table["YES"].div(table["total"].replace(0, np.nan))
    scan = []
    for threshold in range(12):
        pred = (frame["sum_score"].to_numpy() >= threshold).astype(int)
        scan.append(dict(threshold=threshold, disagreements=int((pred != target).sum()),
                         **measure(target, pred, frame["sum_score"].to_numpy())))
    return table.reset_index(), pd.DataFrame(scan)


def questionnaire_coefficients(model):
    """Map standardized coefficients back to a one-point change in binary items."""
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    if len(scaler.scale_) != len(AQ_COLUMNS):
        raise ValueError("Coefficient explanation requires questionnaire-only features")
    beta = classifier.coef_[0]
    raw_beta = beta / scaler.scale_
    raw_intercept = float(classifier.intercept_[0] - np.dot(raw_beta, scaler.mean_))
    return [dict(item=item, standardized_coefficient=float(beta[i]),
                 raw_item_coefficient=float(raw_beta[i]),
                 one_point_odds_ratio=float(np.exp(raw_beta[i])),
                 scaler_mean=float(scaler.mean_[i]), scaler_scale=float(scaler.scale_[i]),
                 raw_intercept=raw_intercept)
            for i, item in enumerate(AQ_COLUMNS)]


def compare_tree_depths(X, y, splits, seed, depths):
    pooled, per_fold = [], []
    for feature_set in ["questionnaire_only", "combined_with_sum"]:
        numeric, categorical = FEATURES[feature_set]
        for depth in depths:
            name = "unlimited" if depth is None else str(depth)
            pred, ranking = np.zeros(len(X), dtype=int), np.zeros(len(X))
            for fold, (train, test) in enumerate(splits):
                model = make_model("tree", numeric, categorical, seed)
                model.set_params(classifier__max_depth=depth)
                model.fit(X.iloc[train], y[train])
                pred[test] = model.predict(X.iloc[test])
                ranking[test] = model.predict_proba(X.iloc[test])[:, 1]
                classifier = model.named_steps["classifier"]
                root_index = int(classifier.tree_.feature[0])
                root_name, raw_threshold = "leaf", None
                if root_index >= 0:
                    root_name = str(model.named_steps["preprocess"].get_feature_names_out()[root_index])
                    if root_index < len(numeric):
                        scaler = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
                        raw_threshold = float(classifier.tree_.threshold[0] * scaler.scale_[root_index] + scaler.mean_[root_index])
                train_pred = model.predict(X.iloc[train])
                per_fold.append(dict(feature_set=feature_set, max_depth=name, fold=fold,
                                     learned_depth=classifier.get_depth(), leaves=classifier.get_n_leaves(),
                                     root_feature=root_name, root_threshold_raw=raw_threshold,
                                     training_accuracy=float(np.mean(train_pred == y[train])),
                                     **measure(y[test], pred[test], ranking[test])))
            pooled.append(dict(feature_set=feature_set, max_depth=name, **measure(y, pred, ranking)))
    return pooled, per_fold


def metric_values(counts):
    """Works on one confusion vector or many bootstrap vectors."""
    tn, fp, fn, tp = np.asarray(counts, dtype=float).T
    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return dict(accuracy=ratio(tp + tn, tp + tn + fp + fn),
                precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn),
                specificity=ratio(tn, tn + fp), f1=ratio(2 * tp, 2 * tp + fp + fn))


def measure(y, pred, ranking):
    counts = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {**{k: float(v) for k, v in metric_values(counts).items()},
            "roc_auc": float(roc_auc_score(y, ranking)) if len(np.unique(y)) == 2 else None,
            **dict(zip(["tn", "fp", "fn", "tp"], map(int, counts)))}


def predictor_groups(X):
    # Hash predictors only. Target labels never influence group construction.
    columns = AQ_COLUMNS + ["age"] + CATEGORICAL_COLUMNS
    return pd.factorize(pd.util.hash_pandas_object(X[columns], index=False))[0]


def make_model(kind, numeric, categorical, seed):
    transforms = []
    if numeric:
        transforms.append(("numeric", SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())]), numeric))
    if categorical:
        transforms.append(("categorical", SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    estimators = {
        "logistic": LogisticRegression(max_iter=1000, random_state=seed),
        "tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=seed),
        "forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=seed, n_jobs=1),
    }
    return Pipeline([("preprocess", ColumnTransformer(transforms)),
                     ("oversample", RandomOverSampler(random_state=seed)),
                     ("classifier", estimators[kind])])


def bootstrap_intervals(y, pred, groups, draws, seed):
    """Resample entire predictor groups, conditional on fixed OOF predictions."""
    unique, inverse = np.unique(groups, return_inverse=True)
    cells = np.asarray(y, dtype=int) * 2 + np.asarray(pred, dtype=int)
    per_group = np.zeros((len(unique), 4), dtype=int)
    np.add.at(per_group, (inverse, cells), 1)
    rng = np.random.default_rng(seed)
    samples = per_group[rng.integers(0, len(unique), size=(draws, len(unique)))].sum(axis=1)
    # Skip resamples lacking either class; precision uses zero when no positives predicted.
    valid = ((samples[:, 0] + samples[:, 1]) > 0) & ((samples[:, 2] + samples[:, 3]) > 0)
    return {key: tuple(map(float, np.quantile(value, [.025, .975])))
            for key, value in metric_values(samples[valid]).items()}


def audit(data_path, df, source_archive):
    raw = pd.read_csv(data_path, dtype=str, keep_default_na=False)
    aq = raw[AQ_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not aq.isin([0, 1]).all().all():
        raise ValueError("Resolve invalid or missing AQ item scores before evaluation")
    if not raw["Class/ASD"].isin(["NO", "YES"]).all():
        raise ValueError("Unexpected target labels")
    y = df["Class/ASD"].map({"NO": 0, "YES": 1})
    X = df[AQ_COLUMNS + ["age", "sum_score"] + CATEGORICAL_COLUMNS]
    groups = predictor_groups(X)
    train, test = train_test_split(np.arange(len(df)), test_size=.2, stratify=y, random_state=42)
    model = build_pipeline().fit(X.iloc[train], y.iloc[train])
    overlaps = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X.iloc[train], y.iloc[train]):
        overlaps.append(len(set(groups[train[a]]) & set(groups[train[b]])))
    report = {
        "rows": len(raw), "class_counts": raw["Class/ASD"].value_counts().to_dict(),
        "missing_markers": {c: int(raw[c].str.strip().isin(["", "?"]).sum()) for c in raw},
        "duplicate_full_rows": int(raw.duplicated().sum()),
        "duplicate_predictor_rows": int(len(groups) - len(np.unique(groups))),
        "age_outside_18_100_or_missing": int((~df.age.between(18, 100)).sum()),
        "result_sum_disagreements": int((pd.to_numeric(raw.result) != aq.sum(axis=1)).sum()),
        "rule_ge7_label_disagreements": int(((df.sum_score >= 7).astype(int) != y).sum()),
        "rule_ge6_label_disagreements": int(((df.sum_score >= 6).astype(int) != y).sum()),
        "legacy_holdout_predictor_overlap": len(set(groups[train]) & set(groups[test])),
        "legacy_cv_predictor_overlap": overlaps,
        "legacy_holdout_comparison": {
            "logistic": measure(y.iloc[test], model.predict(X.iloc[test]), model.predict_proba(X.iloc[test])[:, 1]),
            "aq_ge7": measure(y.iloc[test], (X.iloc[test].sum_score >= 7).astype(int), X.iloc[test].sum_score),
            "aq_ge6": measure(y.iloc[test], (X.iloc[test].sum_score >= 6).astype(int), X.iloc[test].sum_score)},
    }
    if source_archive:
        with zipfile.ZipFile(source_archive) as z:
            content = z.read("Autism-Adult-Data.arff").decode()
            names = [line.split()[1] for line in content.splitlines() if line.lower().startswith("@attribute")]
            source_rows = list(csv.reader(io.StringIO(content.split("@data", 1)[1].strip()), quotechar="'", skipinitialspace=True))
            original = pd.DataFrame(source_rows, columns=names)
            report["source_verification"] = {
                "archive_sha256": hashlib.sha256(Path(source_archive).read_bytes()).hexdigest(),
                "all_raw_cells_and_row_order_match": original.equals(raw),
                "original_rows": len(original), "archive": repository_path(source_archive),
            }
    return report


def evaluate(data_path, output, seeds, folds, draws, source_archive=None, tree_depths=TREE_DEPTHS):
    df = load_dataset(data_path)
    audit_report = audit(data_path, df, source_archive)
    output.mkdir(parents=True, exist_ok=True)
    (output / "environment.txt").write_text("\n".join(installed_packages()) + "\n")
    (output / "audit.json").write_text(json.dumps(audit_report, indent=2))
    settings = {
        "question": "Do models outperform a score rule in reproducing adult screening labels, and how does performance change without score information?",
        "seeds": seeds, "folds": folds, "bootstrap_draws": draws, "bootstrap_seed": 9102,
        "feature_sets": FEATURES, "aq_rules": {"aq_ge6": "ARC adult AQ-10 referral threshold", "aq_ge7": "Empirically observed local label reconstruction rule"},
        "model_parameters": {k: make_model(k, *FEATURES["combined_with_sum"], seeds[0]).named_steps["classifier"].get_params() for k in ["logistic", "tree", "forest"]},
        "oversampler": "RandomOverSampler inside each training fold; random_state equals split seed",
        "split": "StratifiedGroupKFold; identical app predictors grouped; same folds for every method",
        "scoring_threshold": .5, "precision_zero_division": 0,
        "explanatory_analyses": {
            "score_thresholds": list(range(12)), "tree_depths": tree_depths,
            "tree_feature_sets": ["questionnaire_only", "combined_with_sum"],
            "tree_min_samples_leaf": 10,
            "coefficients": "Questionnaire-only logistic; training-fold fits; standardized and raw binary-item units",
            "interpretation": "Descriptive threshold scan and depth comparison; no model or clinical threshold selected from validation results",
        },
        "sensitivity": "Retain ages 18-100 inclusive and first record per identical predictor group",
        "uncertainty": "Repeat/fold SD plus 95% group percentile bootstrap conditional on first-seed OOF predictions; no retraining in bootstrap; ROC-AUC repeat/fold SD only",
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "train_code_sha256": hashlib.sha256((ROOT_DIR / "src/train.py").read_bytes()).hexdigest(),
        "versions": dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, imblearn=imblearn.__version__),
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2))
    pooled, fold_rows, oof_rows, assignments, intervals = [], [], [], [], []
    labels_by_score, threshold_rows, coefficient_rows, depth_rows, depth_fold_rows = [], [], [], [], []
    groups = predictor_groups(df)
    cleaned = df.loc[df.age.between(18, 100)].copy()
    cleaned = cleaned.loc[~pd.Series(predictor_groups(cleaned), index=cleaned.index).duplicated()]
    for population, frame in [("primary", df), ("age_and_duplicate_sensitivity", cleaned)]:
        label_table, scan = score_diagnostics(frame)
        labels_by_score.extend(dict(population=population, **row) for row in label_table.to_dict("records"))
        threshold_rows.extend(dict(population=population, **row) for row in scan.to_dict("records"))
        X = frame[AQ_COLUMNS + ["age", "sum_score"] + CATEGORICAL_COLUMNS]
        y = frame["Class/ASD"].map({"NO": 0, "YES": 1}).to_numpy()
        groups = predictor_groups(X)
        for seed in seeds:
            splits = list(StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(X, y, groups))
            for fold, (train, test) in enumerate(splits):
                if set(groups[train]) & set(groups[test]):
                    raise AssertionError("Predictor groups overlap")
                if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
                    raise ValueError("Every fold must contain both classes; reduce folds")
                assignments.extend(dict(population=population, seed=seed, fold=fold, row_index=int(frame.index[i]), group=int(groups[i])) for i in test)
            methods = ["dummy", "aq_ge6", "aq_ge7"] + [f"{kind}/{feat}" for feat in FEATURES for kind in ["logistic", "tree", "forest"]]
            for method in methods:
                pred, ranking = np.zeros(len(X), dtype=int), np.zeros(len(X))
                for fold, (train, test) in enumerate(splits):
                    if method.startswith("aq_"):
                        threshold = int(method[-1])
                        pred[test] = (X.iloc[test].sum_score >= threshold).astype(int)
                        ranking[test] = X.iloc[test].sum_score
                    else:
                        if method == "dummy":
                            model = DummyClassifier(strategy="most_frequent", random_state=seed)
                        else:
                            kind, feat = method.split("/")
                            # Exact serving pipeline for the original feature set and seed.
                            model = build_pipeline() if kind == "logistic" and feat == "combined_with_sum" and seed == 42 else make_model(kind, *FEATURES[feat], seed)
                        model.fit(X.iloc[train], y[train])
                        if method == "logistic/questionnaire_only":
                            coefficient_rows.extend(dict(population=population, seed=seed, fold=fold, **row)
                                                    for row in questionnaire_coefficients(model))
                        pred[test] = model.predict(X.iloc[test])
                        ranking[test] = model.predict_proba(X.iloc[test])[:, 1]
                    fold_rows.append(dict(population=population, seed=seed, method=method, fold=fold, **measure(y[test], pred[test], ranking[test])))
                pooled.append(dict(population=population, seed=seed, method=method, **measure(y, pred, ranking)))
                oof_rows.extend(dict(population=population, seed=seed, method=method, row_index=int(frame.index[i]), target=int(y[i]), prediction=int(pred[i]), ranking_score=float(ranking[i])) for i in range(len(X)))
                if seed == seeds[0]:
                    ci = bootstrap_intervals(y, pred, groups, draws, 9102)
                    intervals.extend(dict(population=population, method=method, metric=k, lower=v[0], upper=v[1]) for k, v in ci.items())
            depth_results, depth_folds = compare_tree_depths(X, y, splits, seed, tree_depths)
            depth_rows.extend(dict(population=population, seed=seed, **row) for row in depth_results)
            depth_fold_rows.extend(dict(population=population, seed=seed, **row) for row in depth_folds)
            print(f"Finished {population}, seed {seed}, {len(frame)} rows", flush=True)
    tables = dict(pooled_metrics=pooled, fold_metrics=fold_rows, predictions=oof_rows, fold_assignment=assignments, conditional_bootstrap=intervals)
    for name, records in tables.items():
        pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)
    explanations = dict(label_by_score=labels_by_score, threshold_scan=threshold_rows,
                        questionnaire_logistic_coefficients=coefficient_rows,
                        tree_depth_metrics=depth_rows, tree_depth_fold_metrics=depth_fold_rows)
    for name, records in explanations.items():
        pd.DataFrame(records).to_csv(output / f"{name}.csv", index=False)
    pd.DataFrame(coefficient_rows).groupby(["population", "item"])[
        ["standardized_coefficient", "raw_item_coefficient", "one_point_odds_ratio"]
    ].agg(["mean", "std", "min", "max"]).to_csv(output / "questionnaire_coefficient_summary.csv")
    pd.DataFrame(depth_rows).groupby(["population", "feature_set", "max_depth"])[
        METRICS + ["roc_auc"]
    ].agg(["mean", "std"]).to_csv(output / "tree_depth_summary.csv")
    summary = pd.DataFrame(pooled).groupby(["population", "method"])[METRICS + ["roc_auc"]].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    print(summary.round(4).to_string())
    return audit_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "research/evaluation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--tree-depths", nargs="+", default=["1", "2", "3", "4", "6", "8", "10", "none"])
    args = parser.parse_args()
    if args.folds < 2 or args.bootstrap < 100 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use at least two folds, 100 bootstrap draws, and distinct seeds")
    try:
        depths = [None if value.lower() == "none" else int(value) for value in args.tree_depths]
        if any(value is not None and value < 1 for value in depths) or len(set(depths)) != len(depths):
            raise ValueError
    except ValueError:
        parser.error("Tree depths must be distinct positive integers or 'none'")
    evaluate(args.data, args.output, args.seeds, args.folds, args.bootstrap, args.source_archive, depths)


if __name__ == "__main__":
    main()
