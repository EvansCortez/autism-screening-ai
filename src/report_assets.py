"""Publication tables and figures from the saved evaluation and circularity outputs.

Reads CSV outputs only; recomputes nothing statistical. Deterministic settings so that regenerated files
can be compared byte for byte. Figures show screening-label agreement, not diagnostic accuracy.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
           "gray": "#7F7F7F", "sky": "#56B4E9", "purple": "#CC79A7"}
MODEL_STYLE = {"logistic": ("o", PALETTE["blue"], "Logistic regression"),
               "tree": ("s", PALETTE["orange"], "Decision tree (depth 4)"),
               "forest": ("^", PALETTE["green"], "Random forest")}
FEATURE_LABELS = {"combined_with_sum": "Items + sum\n+ demographics", "combined_without_sum": "Items\n+ demographics",
                  "questionnaire_only": "Items only", "non_questionnaire": "Age and categories\nonly (no items)"}
SAVE = {"svg": dict(metadata={"Date": None}), "pdf": dict(metadata={"CreationDate": None, "Producer": None, "Creator": None}),
        "png": dict(dpi=200, metadata={"Software": None})}


def configure():
    plt.rcParams.update({"svg.hashsalt": "asd-predict", "svg.fonttype": "path", "pdf.fonttype": 42,
                         "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9,
                         "figure.dpi": 100, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.25})


def save(fig, directory, name):
    for extension, options in SAVE.items():
        fig.savefig(directory / f"{name}.{extension}", format=extension, **options)
    plt.close(fig)


def write_table(frame, directory, name, caption, floatfmt="{:.4f}"):
    frame.to_csv(directory / f"{name}.csv", index=False)
    text = frame.copy()
    for column in text.columns:
        if pd.api.types.is_float_dtype(text[column]):
            text[column] = text[column].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    header = "| " + " | ".join(map(str, text.columns)) + " |\n| " + " | ".join("---" for _ in text.columns) + " |\n"
    body = "".join("| " + " | ".join(map(str, row)) + " |\n" for row in text.itertuples(index=False))
    (directory / f"{name}.md").write_text(f"**{caption}**\n\n{header}{body}")
    (directory / f"{name}.tex").write_text(text.to_latex(index=False, escape=True, caption=caption, label=f"tab:{name}"))


def load(evaluation, circularity, uncertainty=None):
    data = dict(
        summary=pd.read_csv(evaluation / "summary.csv", header=[0, 1], index_col=[0, 1]),
        pooled=pd.read_csv(evaluation / "pooled_metrics.csv"),
        scan_adult=pd.read_csv(evaluation / "threshold_scan.csv").query("population == 'primary'"),
        label_adult=pd.read_csv(evaluation / "label_by_score.csv").query("population == 'primary'"),
        coefficients=pd.read_csv(evaluation / "questionnaire_logistic_coefficients.csv").query("population == 'primary'"),
        depth=pd.read_csv(evaluation / "tree_depth_metrics.csv").query("population == 'primary'"),
        scan_child=pd.read_csv(circularity / "child" / "threshold_scan.csv"),
        label_child=pd.read_csv(circularity / "child" / "label_by_score.csv"),
    )
    if uncertainty is not None:
        for name in ["interval_table", "whole_procedure_bootstrap", "country_cluster_bootstrap", "permutation_null", "residual_decomposition", "leave_country_out_metrics"]:
            data[name] = pd.read_csv(uncertainty / f"{name}.csv")
    return data


def figure_label_by_score(data, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    panels = [("Adult (UCI 426, n = 704)", data["label_adult"].rename(columns={"aq_score": "score", "NO": "negative", "YES": "positive"})),
              ("Child (UCI 419, n = 292)", data["label_child"])]
    for ax, (title, table) in zip(axes, panels):
        ax.bar(table.score, table.negative, color=PALETTE["sky"], label="Label NO")
        ax.bar(table.score, table.positive, bottom=table.negative, color=PALETTE["red"], label="Label YES")
        ax.axvline(5.5, color="black", linestyle=":", linewidth=1)
        ax.axvline(6.5, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(range(11))
        ax.set_xlabel("Questionnaire total (sum of ten item scores)")
        ax.set_title(title)
    axes[0].set_ylabel("Number of records")
    for ax in axes:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    axes[0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.55))
    for ax in axes:
        top = ax.get_ylim()[1]
        ax.text(5.4, top * 0.98, "documented referral\ncut-off (total >= 6)", ha="right", va="top", fontsize=7)
        ax.text(6.6, top * 0.98, "label boundary\n(total >= 7)", ha="left", va="top", fontsize=7)
    fig.suptitle("Every score has exactly one label; the boundary is at 7", y=1.02, fontsize=10)
    fig.tight_layout()
    save(fig, directory, "fig1_label_by_score")


def figure_threshold_scan(data, directory):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, (title, scan) in zip(axes, [("Adult (n = 704)", data["scan_adult"]), ("Child (n = 292)", data["scan_child"])]):
        ax.plot(scan.threshold, scan.disagreements, marker="o", color=PALETTE["blue"])
        for row, (cutoff, color, name) in enumerate([(6, PALETTE["orange"], "t = 6 (documented referral cut-off)"), (7, PALETTE["red"], "t = 7")]):
            value = int(scan.loc[scan.threshold == cutoff, "disagreements"].iloc[0])
            ax.scatter([cutoff], [value], s=70, color=color, zorder=3)
            ax.text(0.03, 0.97 - 0.09 * row, f"\u25cf {name}: {value} records", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=color)
        ax.set_ylim(-0.03 * scan.disagreements.max(), scan.disagreements.max() * 1.4)
        ax.set_xticks(range(12))
        ax.set_xlabel("Cut-off t (predict YES when total >= t)")
        ax.set_title(title)
    axes[0].set_ylabel("Records where rule disagrees with label")
    fig.suptitle("Only the cut-off at 7 reproduces the labels", y=1.02, fontsize=10)
    fig.tight_layout()
    save(fig, directory, "fig2_threshold_scan")


def figure_accuracy(data, directory):
    pooled = data["pooled"].query("population == 'primary'")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    features = list(FEATURE_LABELS)
    offsets = {"logistic": -0.22, "tree": 0.0, "forest": 0.22}
    for kind, (marker, color, label) in MODEL_STYLE.items():
        means = []
        for index, feature in enumerate(features):
            values = pooled[pooled.method == f"{kind}/{feature}"].accuracy.to_numpy()
            x = index + offsets[kind]
            ax.scatter([x] * len(values), values, marker=marker, s=16, color=color, alpha=0.45)
            means.append(values.mean())
        ax.scatter([i + offsets[kind] for i in range(len(features))], means, marker=marker, s=60, color=color, edgecolor="black", label=label, zorder=3)
    baselines = {"Majority class": pooled[pooled.method == "dummy"].accuracy.mean(), "Rule: total >= 6": pooled[pooled.method == "aq_ge6"].accuracy.mean(),
                 "Rule: total >= 7": pooled[pooled.method == "aq_ge7"].accuracy.mean()}
    for (name, value), style in zip(baselines.items(), [":", "-.", "--"]):
        ax.axhline(value, color=PALETTE["gray"], linestyle=style, linewidth=1)
        ax.text(len(features) - 0.45, value + 0.004, name, ha="right", va="bottom", fontsize=7, color="black")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([FEATURE_LABELS[f] for f in features])
    ax.set_ylim(0.6, 1.03)
    ax.set_ylabel("Pooled out-of-fold accuracy\n(agreement with the dataset label)")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title("Agreement with the dataset label, by model and feature set")
    fig.text(0.01, -0.04, "Small dots: three repeats of the same data (seeds 42, 137, 2026); large markers: their mean. Not confidence intervals.", fontsize=7)
    fig.tight_layout()
    save(fig, directory, "fig3_accuracy_by_features")


def figure_coefficients(data, directory):
    table = data["coefficients"]
    items = [f"A{i}" for i in range(1, 11)]
    grouped = table.groupby("item").raw_item_coefficient.agg(["mean", "min", "max"]).reindex([f"{i}_Score" for i in items])
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(items))
    ax.vlines(x, grouped["min"], grouped["max"], color=PALETTE["gray"], linewidth=2)
    ax.scatter(x, grouped["mean"], color=PALETTE["blue"], zorder=3, s=40)
    ax.set_xticks(x)
    ax.set_xticklabels(items)
    ax.set_ylim(0, grouped["max"].max() * 1.15)
    ax.set_xlabel("Questionnaire item")
    ax.set_ylabel("Weight per item point\n(log-odds, 0/1 item units)")
    ax.set_title("Items-only logistic weights are all positive and similar in size")
    fig.text(0.01, -0.04, "Dot: mean over 15 training-fold fits; bar: minimum to maximum. Descriptive; not a confidence interval and not clinical importance.", fontsize=7)
    fig.tight_layout()
    save(fig, directory, "fig4_item_weights")


def figure_tree_depth(data, directory):
    depth = data["depth"]
    order = ["1", "2", "3", "4", "6", "8", "10", "unlimited"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for feature_set, color, label in [("questionnaire_only", PALETTE["orange"], "Items only"), ("combined_with_sum", PALETTE["blue"], "Items + sum + demographics")]:
        sub = depth[depth.feature_set == feature_set]
        means = []
        for index, name in enumerate(order):
            values = sub[sub.max_depth.astype(str) == name].accuracy.to_numpy()
            ax.scatter([index] * len(values), values, color=color, alpha=0.4, s=14)
            means.append(values.mean())
        ax.plot(range(len(order)), means, marker="o", color=color, label=label)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Maximum tree depth (minimum leaf size fixed at 10)")
    ax.set_ylabel("Pooled out-of-fold accuracy")
    ax.set_ylim(0.78, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Decision-tree agreement by maximum depth, with and without the total as an input")
    fig.text(0.01, -0.04, "Small dots: repeats (seeds); line: mean. Descriptive; no depth is selected.", fontsize=7)
    fig.tight_layout()
    save(fig, directory, "fig5_tree_depth")


def figure_residual_signal(data, directory):
    table = data["residual_decomposition"].copy()
    null = data["permutation_null"].iloc[0]
    table["label"] = table.feature_set.str.replace("main set: age, gender, jaundice, country, ethnicity, relation", "main set (all six fields)", regex=False).str.replace("age + gender + jaundice", "age+gender+jaundice", regex=False)
    table = table.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh(table.label, table.roc_auc_mean, color=PALETTE["sky"], edgecolor="black", linewidth=0.5)
    ax.axvspan(null.null_min, null.null_max, color=PALETTE["gray"], alpha=0.25, label=f"shuffled-label range ({null.null_draws:.0f} shuffles)")
    ax.axvline(0.5, color="black", linewidth=0.8)
    ax.set_xlim(min(0.4, null.null_min - 0.01), max(0.8, null.null_max + 0.01))
    ax.set_xlabel("ROC-AUC, pooled out-of-fold (mean of 3 repeats)")
    ax.set_title("Signal left after removing all questionnaire items")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.text(0.01, -0.05, "Country and ethnicity fields carry most of it; why is not established. Descriptive: not a causal or clinical result.", fontsize=7)
    fig.tight_layout()
    save(fig, directory, "fig6_residual_signal")


def build_tables(data, directory):
    summary = data["summary"].loc["primary"]
    names = {"dummy": "Majority class", "aq_ge6": "Rule: total >= 6 (documented referral cut-off)", "aq_ge7": "Rule: total >= 7 (matches labels)",
             "logistic/combined_with_sum": "Logistic, items + sum + demographics", "logistic/combined_without_sum": "Logistic, items + demographics",
             "logistic/questionnaire_only": "Logistic, items only", "logistic/non_questionnaire": "Logistic, age and categories only",
             "tree/combined_with_sum": "Tree, items + sum + demographics", "tree/combined_without_sum": "Tree, items + demographics",
             "tree/questionnaire_only": "Tree, items only", "tree/non_questionnaire": "Tree, age and categories only",
             "forest/combined_with_sum": "Forest, items + sum + demographics", "forest/combined_without_sum": "Forest, items + demographics",
             "forest/questionnaire_only": "Forest, items only", "forest/non_questionnaire": "Forest, age and categories only"}
    rows = []
    for method, label in names.items():
        rows.append({"Method and features": label, "Accuracy (mean of 3 repeats)": summary.loc[method, ("accuracy", "mean")],
                     "Repeat SD (not a CI)": summary.loc[method, ("accuracy", "std")], "Precision": summary.loc[method, ("precision", "mean")],
                     "Recall": summary.loc[method, ("recall", "mean")], "F1": summary.loc[method, ("f1", "mean")], "ROC-AUC": summary.loc[method, ("roc_auc", "mean")]})
    write_table(pd.DataFrame(rows), directory, "table1_main_results", "Screening-label agreement, primary cohort (n = 704), grouped five-fold validation")
    a = data["label_adult"].rename(columns={"aq_score": "Total score", "NO": "Adult NO", "YES": "Adult YES"})[["Total score", "Adult NO", "Adult YES"]]
    c = data["label_child"].rename(columns={"score": "Total score", "negative": "Child NO", "positive": "Child YES"})[["Total score", "Child NO", "Child YES"]]
    write_table(a.merge(c, on="Total score"), directory, "table2_label_by_score", "Label counts by questionnaire total")
    scan = data["scan_adult"][["threshold", "disagreements"]].rename(columns={"disagreements": "Adult disagreements"}).merge(
        data["scan_child"][["threshold", "disagreements"]].rename(columns={"disagreements": "Child disagreements"}), on="threshold")
    scan["threshold"] = scan["threshold"].astype(int)
    write_table(scan.rename(columns={"threshold": "Cut-off t (total >= t)"}), directory, "table3_threshold_scan", "Records where the rule 'total >= t' disagrees with the label")
    depth = data["depth"].groupby(["feature_set", "max_depth"]).accuracy.mean().unstack(0).reset_index()
    order = {"1": 0, "2": 1, "3": 2, "4": 3, "6": 4, "8": 5, "10": 6, "unlimited": 7}
    depth["o"] = depth.max_depth.astype(str).map(order)
    depth = depth.sort_values("o").drop(columns="o").rename(columns={"max_depth": "Maximum depth", "questionnaire_only": "Items only", "combined_with_sum": "Items + sum + demographics"})
    write_table(depth, directory, "table4_tree_depth", "Tree accuracy by maximum depth (minimum leaf size 10; mean of 3 repeats)")
    if "interval_table" in data:
        intervals = data["interval_table"].rename(columns={"quantity": "Quantity", "correct": "Correct", "n": "n", "estimate": "Estimate", "method": "Method",
                                                           "lower": "Lower", "upper": "Upper", "conditional_on": "Conditional on", "status": "Status"})
        write_table(intervals, directory, "table5_intervals", "Intervals and counts with the conditions they depend on")
        boot = data["whole_procedure_bootstrap"][["method", "draws", "binomial_se", "bootstrap_se", "se_ratio", "repeat_sd", "distinct_row_fraction"]]
        cluster = data["country_cluster_bootstrap"][["method", "row_level_se", "country_cluster_se", "se_ratio"]].rename(columns={"se_ratio": "country_se_ratio"})
        write_table(boot.merge(cluster, on="method", how="outer").rename(columns={"se_ratio": "bootstrap_se_ratio"}), directory, "table6_uncertainty_checks",
                    "Whole-procedure bootstrap and country-cluster checks against the binomial standard error")
        write_table(data["residual_decomposition"].rename(columns={"feature_set": "Predictors (logistic regression)", "roc_auc_mean": "ROC-AUC mean", "roc_auc_sd_over_repeats": "Repeat SD (not a CI)"}),
                    directory, "table7_residual_signal", "Residual signal without questionnaire items")
        write_table(data["leave_country_out_metrics"], directory, "table8_country_held_out",
                    "Non-questionnaire logistic regression: pooled leave-one-country-out agreement by seed; not clinical external validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--circularity", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configure()
    data = load(args.evaluation, args.circularity, args.uncertainty)
    (args.output / "figures").mkdir(parents=True, exist_ok=True)
    (args.output / "tables").mkdir(parents=True, exist_ok=True)
    figure_label_by_score(data, args.output / "figures")
    figure_threshold_scan(data, args.output / "figures")
    figure_accuracy(data, args.output / "figures")
    figure_coefficients(data, args.output / "figures")
    figure_tree_depth(data, args.output / "figures")
    if "residual_decomposition" in data:
        figure_residual_signal(data, args.output / "figures")
    build_tables(data, args.output / "tables")
    print("wrote", len(list((args.output / "figures").iterdir())), "figure files and", len(list((args.output / "tables").iterdir())), "table files")


if __name__ == "__main__":
    main()
