"""Simulation with known truth: do exact intervals on pooled out-of-fold accuracy cover cross-validated performance?

The population is the empirical distribution of the ten item scores in the adult data. The label is
1[item total >= 7] flipped with probability `eps`, so the true error of the best rule is known. A logistic
regression is fitted on the ten items with five-fold cross-validation. For each simulated dataset we form
the exact (Clopper-Pearson) and Wilson intervals as if the pooled out-of-fold predictions were independent
trials, and ask how often they contain the expected accuracy of the fold models on fresh rows (the quantity
cross-validation estimates; Bengio and Grandvalet 2004). Rows are independent and the model is stable, the
most favorable case for a binomial interval, so results here are a best case, not a guarantee.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.train import AQ_COLUMNS, DATA_PATH, ROOT_DIR
from src.uncertainty import exact_interval, wilson_interval

CONFIGURATIONS = [(150, 500), (704, 300)]
FLIP_PROBABILITIES = [0.0, 0.05, 0.15]
FRESH_ROWS = 20000


def simulate(items, seed=2026, configurations=CONFIGURATIONS, flips=FLIP_PROBABILITIES, fresh_rows=FRESH_ROWS):
    rng = np.random.default_rng(seed)

    def draw(n, eps):
        X = items[rng.integers(0, len(items), n)]
        y = (X.sum(axis=1) >= 7).astype(int)
        return X, np.where(rng.random(n) < eps, 1 - y, y)

    rows = []
    for n, repeats in configurations:
        for eps in flips:
            X_test, y_test = draw(fresh_rows, eps)
            accuracy, fold_truth, intervals = [], [], []
            for repeat in range(repeats):
                X, y = draw(n, eps)
                prediction, truth = np.zeros(n, dtype=int), []
                for train, test in StratifiedKFold(5, shuffle=True, random_state=repeat).split(X, y):
                    model = LogisticRegression(max_iter=2000).fit(X[train], y[train])
                    prediction[test] = model.predict(X[test])
                    truth.append(float((model.predict(X_test) == y_test).mean()))
                correct = int((prediction == y).sum())
                accuracy.append(correct / n)
                fold_truth.append(float(np.mean(truth)))
                intervals.append((correct, exact_interval(correct, n), wilson_interval(correct, n)))
            accuracy, fold_truth = np.asarray(accuracy), np.asarray(fold_truth)
            expected = float(fold_truth.mean())
            binomial_se = float(np.mean(np.sqrt(accuracy * (1 - accuracy) / n)))
            zero = [i for i, (correct, _, _) in enumerate(intervals) if correct == n]
            rows.append(dict(
                n=n, flip_probability=eps, simulated_datasets=repeats,
                mean_cv_accuracy=float(accuracy.mean()), expected_fold_model_accuracy=expected,
                empirical_sd_of_cv_accuracy=float(accuracy.std(ddof=1)), mean_binomial_se=binomial_se,
                spread_ratio=float(accuracy.std(ddof=1) / binomial_se) if binomial_se > 0 else np.nan,
                exact_coverage_of_expected_accuracy=float(np.mean([lo <= expected <= hi for _, (lo, hi), _ in intervals])),
                exact_coverage_of_each_dataset_fold_accuracy=float(np.mean([lo <= t <= hi for (_, (lo, hi), _), t in zip(intervals, fold_truth)])),
                wilson_coverage_of_expected_accuracy=float(np.mean([lo <= expected <= hi for _, _, (lo, hi) in intervals])),
                fraction_with_zero_errors=len(zero) / repeats,
                zero_error_bound_exceeds_true_error=float(np.mean([(1 - fold_truth[i]) <= intervals[i][1][1] for i in zero])) if zero else np.nan))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "research/uncertainty")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    items = pd.read_csv(args.data)[AQ_COLUMNS].astype(int).to_numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    simulate(items, args.seed).to_csv(args.output / "interval_coverage_simulation.csv", index=False)
    (args.output / "interval_coverage_simulation_settings.json").write_text(json.dumps(
        dict(seed=args.seed, configurations=CONFIGURATIONS, flip_probabilities=FLIP_PROBABILITIES, fresh_rows=FRESH_ROWS,
             versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
             interpretation="Best case for a binomial interval: independent rows, stable model, known truth. Not a guarantee for real data."), indent=2))
    print("wrote", args.output / "interval_coverage_simulation.csv")


if __name__ == "__main__":
    main()
