"""Read-only UI adapter for hash-verified, committed experiment outputs."""

import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "src/static/research-version.json"
METHODS = {
    "dummy": "Majority dummy",
    "aq_ge6": "Documented AQ >= 6 rule",
    "aq_ge7": "Observed AQ >= 7 rule",
    "logistic/combined_with_sum": "Logistic regression with sum",
    "logistic/combined_without_sum": "Logistic regression without sum",
    "logistic/questionnaire_only": "Questionnaire-only logistic regression",
    "logistic/non_questionnaire": "Non-questionnaire logistic regression",
}
FIGURES = {
    "fig1_label_by_score": ("Labels by questionnaire total", "Adult and child label counts: totals below 7 have label NO; totals of 7 or more have label YES. The documented referral threshold is 6, not 7."),
    "fig2_threshold_scan": ("Threshold scan", "Only a threshold of 7 matches every observed adult and child label; a threshold of 6 disagrees on 70 adult and 40 child records."),
    "fig3_accuracy_by_features": ("Models and feature sets", "Saved grouped out-of-fold screening-label agreement for logistic regression, trees and forests. Removing all items lowers agreement; removing the sum alone retains item information."),
    "fig4_item_weights": ("Questionnaire-only coefficients", "All ten raw-unit logistic item weights are positive and similar in size. The ranges describe training-fold variation, not clinical importance or confidence intervals."),
    "fig5_tree_depth": ("Tree depth comparison", "Trees supplied with the sum reach perfect label agreement even at depth one. Item-only trees are constrained by depth and a fixed minimum leaf size of ten."),
    "fig6_residual_signal": ("Residual signal", "Non-questionnaire AUC varies by predictor set. Country and ethnicity fields carry much of the observed association; its cause is not established. The shaded band is the exploratory shuffled-label range."),
}


def verified_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Pinned research evidence missing or changed: {relative}")
    return manifest


def read_rows(relative):
    with (ROOT / relative).open(newline="") as source:
        return list(csv.DictReader(source))


def evidence_payload():
    manifest = verified_manifest()
    settings = json.loads((ROOT / "research/evaluation/settings.json").read_text())
    audit = json.loads((ROOT / "research/evaluation/audit.json").read_text())
    saved = read_rows("research/evaluation/pooled_metrics.csv")
    metrics = []
    for method, name in METHODS.items():
        rows = [r for r in saved if r["population"] == "primary" and r["method"] == method]
        if sorted(int(r["seed"]) for r in rows) != sorted(settings["seeds"]):
            raise ValueError("Saved evaluation seeds do not match settings")
        metrics.append(dict(method=method, name=name, **{
            key: fmean(float(r[key]) for r in rows)
            for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        }))
    country = read_rows("research/uncertainty/leave_country_out_metrics.csv")
    commit = manifest["commit"]
    return {
        "version": commit,
        "source_url": f"https://github.com/EvansCortez/autism-screening-ai/tree/{commit}/research",
        "files": manifest["files"],
        "rows": audit["rows"], "seeds": settings["seeds"], "folds": settings["folds"],
        "rule_disagreements": {"ge6": audit["rule_ge6_label_disagreements"], "ge7": audit["rule_ge7_label_disagreements"]},
        "metrics": metrics,
        "country_held_out_auc": fmean(float(r["roc_auc"]) for r in country),
        "figures": [dict(id=name, title=title, description=description,
                         url=f"/research-figures/{name}.png") for name, (title, description) in FIGURES.items()],
        "interpretation": "Agreement with dataset screening labels, not clinical diagnostic accuracy. Repeats are not independent studies.",
    }
