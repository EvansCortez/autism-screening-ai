"""Uncertainty and residual-signal analyses for the adult screening-label study.

These analyses accompany src/evaluate.py without changing it. They report what the data can and cannot
support: a count of disagreements is a fact about this file; intervals describe sampling noise only under
stated assumptions; cross-validation predictions are not independent observations, so treating them as
independent does not give a validated generalization interval.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

from src.evaluate import FEATURES, make_model, measure, predictor_groups
from src.train import AQ_COLUMNS, CATEGORICAL_COLUMNS, DATA_PATH, ROOT_DIR, load_dataset

ALPHA = 0.05


def exact_interval(correct, total, alpha=ALPHA):
    """Clopper-Pearson interval for a proportion; valid only for independent trials with a fixed rule."""
    lower = 0.0 if correct == 0 else float(stats.beta.ppf(alpha / 2, correct, total - correct + 1))
    upper = 1.0 if correct == total else float(stats.beta.ppf(1 - alpha / 2, correct + 1, total - correct))
    return lower, upper


def wilson_interval(successes, total, z=1.959963984540054):
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == total else min(1.0, center + half)
    return float(lower), float(upper)


def zero_error_upper_bound(total, alpha=ALPHA, candidates=1, two_sided=False):
    """Upper bound on the error rate after zero errors in `total` independent trials.

    `candidates` applies a Bonferroni adjustment when the rule was chosen among several candidates using
    the same rows.
    """
    level = alpha / (2 if two_sided else 1) / candidates
    return float(1 - level ** (1 / total))


def cluster_bootstrap(correct, clusters, draws, seed):
    """Standard error of accuracy when whole clusters (for example countries) are resampled."""
    correct = np.asarray(correct, dtype=float)
    labels, inverse = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    sums = np.bincount(inverse, weights=correct, minlength=len(labels))
    counts = np.bincount(inverse, minlength=len(labels)).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(labels), size=(draws, len(labels)))
    accuracy = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return dict(se=float(accuracy.std(ddof=1)), lower=float(np.quantile(accuracy, ALPHA / 2)),
                upper=float(np.quantile(accuracy, 1 - ALPHA / 2)), clusters=len(labels))


def pooled_oof(frame, y, groups, kind, feature_set, seed, splits=None, extra=None):
    numeric, categorical = FEATURES[feature_set] if extra is None else extra
    prediction, ranking = np.zeros(len(frame), dtype=int), np.zeros(len(frame))
    if splits is None:
        splits = StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(frame, y, groups)
    for train, test in splits:
        model = make_model(kind, numeric, categorical, seed).fit(frame.iloc[train], y[train])
        prediction[test] = model.predict(frame.iloc[test])
        ranking[test] = model.predict_proba(frame.iloc[test])[:, 1]
    return prediction, ranking


def whole_procedure_bootstrap(frame, y, groups, kind, feature_set, draws, seed):
    """Resample rows, rerun grouped cross-validation on each resample, and record pooled accuracy and AUC.

    Copies of a row (and rows with identical predictors) stay in one fold. A bootstrap sample contains only
    about 63% distinct rows. This changes the effective training sample; coverage and the direction
    of any bias in the spread have not been established for this procedure.
    """
    rows = []
    for draw in range(draws):
        rng = np.random.default_rng([seed, draw])
        pick = rng.integers(0, len(frame), len(frame))
        sample, labels, sample_groups = frame.iloc[pick].reset_index(drop=True), y[pick], groups[pick]
        prediction, ranking = pooled_oof(sample, labels, sample_groups, kind, feature_set, draw)
        rows.append(dict(draw=draw, accuracy=float((prediction == labels).mean()), roc_auc=float(roc_auc_score(labels, ranking)),
                         distinct_row_fraction=float(len(np.unique(pick)) / len(frame))))
    return pd.DataFrame(rows)


def permutation_null(frame, y, groups, kind, feature_set, draws, seed, repeat_seeds=(42, 137, 2026)):
    """Mean repeated OOF AUC per shuffle, using the same statistic as the observed result.

    An unrestricted shuffle assumes exchangeable row labels. Duplicates and unobserved recruitment
    dependencies make this an exploratory null, not a guarantee of calibrated inference.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        shuffled = rng.permutation(y)
        aucs = []
        for repeat_seed in repeat_seeds:
            _, ranking = pooled_oof(frame, shuffled, groups, kind, feature_set, repeat_seed)
            aucs.append(float(roc_auc_score(shuffled, ranking)))
        values.append(float(np.mean(aucs)))
    return np.asarray(values)


def nested_threshold(frame, y, groups, seeds, thresholds):
    rows = []
    score = frame["sum_score"].to_numpy()
    for seed in seeds:
        chosen, prediction = [], np.zeros(len(frame), dtype=int)
        for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(frame, y, groups):
            accuracy = [float(((score[train] >= t).astype(int) == y[train]).mean()) for t in thresholds]
            best = int(thresholds[int(np.argmax(accuracy))])
            chosen.append(best)
            prediction[test] = (score[test] >= best).astype(int)
        rows.append(dict(seed=seed, thresholds_chosen_per_fold=json.dumps(chosen), out_of_fold_errors=int((prediction != y).sum()), rows=len(frame)))
    return pd.DataFrame(rows)


def interval_table(audit, frame, y, thresholds):
    """Intervals with their conditions stated next to them."""
    rows = []
    holdout = audit["legacy_holdout_comparison"]
    for name, key in [("Serving logistic model, original 141-row holdout", "logistic"), ("Rule total >= 7, original holdout", "aq_ge7"), ("Rule total >= 6, original holdout", "aq_ge6")]:
        counts = holdout[key]
        n = counts["tn"] + counts["fp"] + counts["fn"] + counts["tp"]
        correct = counts["tn"] + counts["tp"]
        low, high = exact_interval(correct, n)
        rows.append(dict(quantity=name, correct=correct, n=n, estimate=correct / n, method="Clopper-Pearson", lower=low, upper=high,
                         conditional_on="the fitted model (or rule) and independent test rows", status="acceptable if rows are independent"))
        if key == "aq_ge7":
            rows[-1].update(method="count only", lower=np.nan, upper=np.nan,
                            conditional_on="this file; threshold found using these labels",
                            status="post-hoc rule: original holdout is not independent of threshold discovery")
    n = len(frame)
    correct6 = int(((frame.sum_score >= 6).astype(int).to_numpy() == y).sum())
    low, high = exact_interval(correct6, n)
    rows.append(dict(quantity="Documented rule total >= 6, all 704 rows", correct=correct6, n=n, estimate=correct6 / n, method="Clopper-Pearson", lower=low, upper=high,
                     conditional_on="a rule fixed before looking at the data and independent rows", status="acceptable if rows are independent"))
    low, high = wilson_interval(correct6, n)
    rows.append(dict(quantity="Documented rule total >= 6, all 704 rows", correct=correct6, n=n, estimate=correct6 / n, method="Wilson", lower=low, upper=high,
                     conditional_on="a rule fixed before looking at the data and independent rows", status="acceptable if rows are independent"))
    correct7 = int(((frame.sum_score >= 7).astype(int).to_numpy() == y).sum())
    errors7 = n - correct7
    rows.append(dict(quantity="Rule total >= 7 (found after looking at these labels), all 704 rows", correct=correct7, n=n, estimate=correct7 / n, method="count only",
                     lower=np.nan, upper=np.nan, conditional_on="this file", status=f"descriptive fact: {errors7} disagreements; not an estimate of future agreement"))
    if errors7 == 0:
        for label, candidates, two_sided in [(f"one-sided 95%, adjusted for {len(thresholds)} scanned cut-offs", len(thresholds), False)]:
            rows.append(dict(quantity=f"Upper bound on disagreement rate for rule >= 7 ({label})", correct=correct7, n=n, estimate=0.0, method="exact, zero errors",
                             lower=0.0, upper=zero_error_upper_bound(n, candidates=candidates, two_sided=two_sided),
                             conditional_on="independent rows from the same labeling process", status="conditional; not clinical accuracy"))
    return pd.DataFrame(rows)


def cluster_table(predictions, frame, draws, seed, population="primary", repeat_seed=None):
    repeat_seed = repeat_seed if repeat_seed is not None else int(predictions.seed.min())
    country = frame["country_of_res"].to_numpy()
    rows = []
    for method, group in predictions[(predictions.population == population) & (predictions.seed == repeat_seed)].groupby("method"):
        group = group.sort_values("row_index")
        correct = (group.prediction.to_numpy() == group.target.to_numpy()).astype(float)
        accuracy = float(correct.mean())
        row_se = float(np.sqrt(accuracy * (1 - accuracy) / len(correct)))
        result = cluster_bootstrap(correct, country[group.row_index.to_numpy()], draws, seed)
        rows.append(dict(method=method, repeat_seed=repeat_seed, accuracy=accuracy, row_level_se=row_se, country_cluster_se=result["se"],
                         se_ratio=result["se"] / row_se if row_se > 0 else np.nan, countries=result["clusters"],
                         cluster_lower=result["lower"], cluster_upper=result["upper"]))
    return pd.DataFrame(rows)


def residual_decomposition(frame, y, groups, seeds):
    frame = frame.copy()
    frame["ethnicity_unknown"] = (frame["ethnicity"] == "Unknown").astype(int)
    frame["family_history"] = (frame["austim"] == "yes").astype(int)
    sets = {
        "age + gender + jaundice": (["age"], ["gender", "jaundice"]),
        "age + gender + jaundice + country": (["age"], ["gender", "jaundice", "country_of_res"]),
        "age + gender + jaundice + ethnicity": (["age"], ["gender", "jaundice", "ethnicity"]),
        "age + gender + jaundice + relation": (["age"], ["gender", "jaundice", "relation"]),
        "ethnicity only": ([], ["ethnicity"]),
        "'ethnicity unknown' flag only": (["ethnicity_unknown"], []),
        "main set: age, gender, jaundice, country, ethnicity, relation": FEATURES["non_questionnaire"],
        "main set + family history": (["age", "family_history"], CATEGORICAL_COLUMNS),
    }
    rows = []
    for name, extra in sets.items():
        aucs, plain = [], []
        for seed in seeds:
            _, ranking = pooled_oof(frame, y, groups, "logistic", "non_questionnaire", seed, extra=extra)
            aucs.append(float(roc_auc_score(y, ranking)))
        rows.append(dict(feature_set=name, roc_auc_mean=float(np.mean(aucs)), roc_auc_sd_over_repeats=float(np.std(aucs, ddof=1)), repeats=len(seeds)))
    grouped, ungrouped = [], []
    for seed in seeds:
        _, ranking = pooled_oof(frame, y, groups, "logistic", "non_questionnaire", seed)
        grouped.append(float(roc_auc_score(y, ranking)))
        _, ranking = pooled_oof(frame, y, groups, "logistic", "non_questionnaire", seed, splits=StratifiedKFold(5, shuffle=True, random_state=seed).split(frame, y))
        ungrouped.append(float(roc_auc_score(y, ranking)))
    rows.append(dict(feature_set="main set, no duplicate grouping", roc_auc_mean=float(np.mean(ungrouped)),
                     roc_auc_sd_over_repeats=float(np.std(ungrouped, ddof=1)), repeats=len(seeds)))
    return pd.DataFrame(rows)


def leave_country_out(frame, y, seeds):
    """Train on all other recorded countries; pool predictions once per row per seed.

    Small countries may contain only one class, so country-specific AUC can be undefined.
    The pooled AUC is not a country-level average or an external clinical validation.
    """
    countries = frame["country_of_res"].to_numpy()
    splits = list(LeaveOneGroupOut().split(frame, y, countries))
    numeric, categorical = FEATURES["non_questionnaire"]
    pooled, per_country, predictions = [], [], []
    for seed in seeds:
        prediction, ranking = np.zeros(len(frame), dtype=int), np.zeros(len(frame))
        for train, test in splits:
            if len(np.unique(y[train])) != 2:
                raise ValueError("Country-held-out training set requires both classes")
            model = make_model("logistic", numeric, categorical, seed).fit(frame.iloc[train], y[train])
            prediction[test] = model.predict(frame.iloc[test])
            ranking[test] = model.predict_proba(frame.iloc[test])[:, 1]
            per_country.append(dict(seed=seed, country=countries[test[0]], n=len(test),
                                    **measure(y[test], prediction[test], ranking[test])))
        pooled.append(dict(seed=seed, n=len(frame), countries=len(splits), **measure(y, prediction, ranking)))
        predictions.extend(dict(seed=seed, row_index=i, country=countries[i], target=int(y[i]),
                                prediction=int(prediction[i]), ranking_score=float(ranking[i]))
                           for i in range(len(frame)))
    return pd.DataFrame(pooled), pd.DataFrame(per_country), pd.DataFrame(predictions)


def missing_block_table(frame, y):
    missing = ((frame["ethnicity"] == "Unknown") & (frame["relation"] == "Unknown")).to_numpy()
    rows = []
    for name, mask in [("ethnicity and relation both missing", missing), ("all other rows", ~missing)]:
        top = frame.loc[mask, "country_of_res"].value_counts().head(3)
        rows.append(dict(group=name, n=int(mask.sum()), label_yes_rate=float(y[mask].mean()), mean_item_total=float(frame.loc[mask, "sum_score"].mean()),
                         median_age=float(frame.loc[mask, "age"].median()), top_countries="; ".join(f"{k} ({v})" for k, v in top.items())))
    return pd.DataFrame(rows)


def subgroup_table(frame, y):
    frame = frame.assign(label=y, age_band=pd.cut(frame["age"].where(frame["age"].between(18, 100)), [17, 24, 34, 49, 100], labels=["18-24", "25-34", "35-49", "50+"]))
    rows = []
    for variable in ["gender", "age_band", "ethnicity", "relation", "jaundice", "austim"]:
        for level, group in frame.groupby(variable, observed=True):
            low, high = wilson_interval(int(group.label.sum()), len(group))
            rows.append(dict(variable=variable, level=str(level), n=len(group), label_yes=int(group.label.sum()), label_yes_rate=float(group.label.mean()),
                             wilson_lower=low, wilson_upper=high, note="descriptive; sampling uncertainty only; categories as recorded"))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--evaluation", type=Path, default=ROOT_DIR / "research/evaluation")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "research/uncertainty")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 137, 2026])
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--permutation-draws", type=int, default=200)
    parser.add_argument("--cluster-draws", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=9102)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(args.data)
    y = frame["Class/ASD"].map({"NO": 0, "YES": 1}).to_numpy()
    groups = predictor_groups(frame)
    audit = json.loads((args.evaluation / "audit.json").read_text())
    predictions = pd.read_csv(args.evaluation / "predictions.csv")
    thresholds = sorted(int(t) for t in pd.read_csv(args.evaluation / "threshold_scan.csv").threshold.unique())

    interval_table(audit, frame, y, thresholds).to_csv(args.output / "interval_table.csv", index=False)
    nested_threshold(frame, y, groups, args.seeds, thresholds).to_csv(args.output / "nested_threshold.csv", index=False)
    cluster_table(predictions, frame, args.cluster_draws, args.bootstrap_seed).to_csv(args.output / "country_cluster_bootstrap.csv", index=False)

    boot_rows = []
    for kind, feature_set in [("logistic", "non_questionnaire"), ("tree", "combined_without_sum")]:
        draws = whole_procedure_bootstrap(frame, y, groups, kind, feature_set, args.bootstrap_draws, args.bootstrap_seed)
        accuracy = []
        for seed in args.seeds:
            prediction, _ = pooled_oof(frame, y, groups, kind, feature_set, seed)
            accuracy.append(float((prediction == y).mean()))
        pooled = float(np.mean(accuracy))
        binomial_se = float(np.sqrt(pooled * (1 - pooled) / len(frame)))
        boot_rows.append(dict(method=f"{kind}/{feature_set}", draws=len(draws), accuracy_original_mean_over_repeats=pooled,
                              repeat_sd=float(np.std(accuracy, ddof=1)), binomial_se=binomial_se, bootstrap_se=float(draws.accuracy.std(ddof=1)),
                              se_ratio=float(draws.accuracy.std(ddof=1) / binomial_se), bootstrap_lower=float(draws.accuracy.quantile(ALPHA / 2)),
                              bootstrap_upper=float(draws.accuracy.quantile(1 - ALPHA / 2)), auc_se=float(draws.roc_auc.std(ddof=1)),
                              distinct_row_fraction=float(draws.distinct_row_fraction.mean()),
                              caveat="resamples hold ~63% distinct rows; coverage and direction of bias are unverified; no interval here treats cross-validation predictions as independent"))
    pd.DataFrame(boot_rows).to_csv(args.output / "whole_procedure_bootstrap.csv", index=False)

    observed = []
    for seed in args.seeds:
        _, ranking = pooled_oof(frame, y, groups, "logistic", "non_questionnaire", seed)
        observed.append(float(roc_auc_score(y, ranking)))
    null = permutation_null(frame, y, groups, "logistic", "non_questionnaire", args.permutation_draws, args.bootstrap_seed, args.seeds)
    pd.DataFrame({"draw": np.arange(len(null)), "mean_repeated_oof_auc": null}).to_csv(args.output / "permutation_draws.csv", index=False)
    pd.DataFrame([dict(method="logistic/non_questionnaire", observed_auc_mean_over_repeats=float(np.mean(observed)), null_draws=len(null), null_mean=float(null.mean()),
                       null_95th_percentile=float(np.quantile(null, 0.95)), null_min=float(null.min()), null_max=float(null.max()),
                       p_value_at_least_observed=float((1 + (null >= np.mean(observed)).sum()) / (len(null) + 1)),
                       note="observed and null use mean OOF AUC over identical repeat seeds; unrestricted row-label shuffling assumes exchangeability, unverified with duplicates/recruitment dependence; exploratory label association, not causes")]).to_csv(args.output / "permutation_null.csv", index=False)

    residual_decomposition(frame, y, groups, args.seeds).to_csv(args.output / "residual_decomposition.csv", index=False)
    country_metrics, country_details, country_predictions = leave_country_out(frame, y, args.seeds)
    country_metrics.to_csv(args.output / "leave_country_out_metrics.csv", index=False)
    country_details.to_csv(args.output / "leave_country_out_by_country.csv", index=False)
    country_predictions.to_csv(args.output / "leave_country_out_predictions.csv", index=False)
    missing_block_table(frame, y).to_csv(args.output / "missing_block.csv", index=False)
    subgroup_table(frame, y).to_csv(args.output / "subgroup_counts.csv", index=False)

    settings = dict(seeds=args.seeds, bootstrap_draws=args.bootstrap_draws, permutation_draws=args.permutation_draws, cluster_draws=args.cluster_draws,
                    bootstrap_seed=args.bootstrap_seed, thresholds=thresholds, input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    evaluation_predictions_sha256=hashlib.sha256((args.evaluation / "predictions.csv").read_bytes()).hexdigest(),
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
                    interpretation="Descriptive uncertainty and robustness analyses. Not clinical validation. Repeated cross-validation predictions are not independent observations.")
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2))
    print("wrote analyses to", args.output)


if __name__ == "__main__":
    main()
