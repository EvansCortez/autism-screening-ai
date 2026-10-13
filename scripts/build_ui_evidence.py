"""Pin the UI's evidence to files verified against a specified existing Git commit."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["research/evaluation/pooled_metrics.csv", "research/evaluation/settings.json",
         "research/evaluation/audit.json", "research/uncertainty/leave_country_out_metrics.csv"]
FILES += [f"research/report/figures/{name}.png" for name in ["fig1_label_by_score", "fig2_threshold_scan",
          "fig3_accuracy_by_features", "fig4_item_weights", "fig5_tree_depth", "fig6_residual_signal"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", f"{args.commit}^{{commit}}"], cwd=ROOT, text=True).strip()
    hashes = {}
    for relative in FILES:
        committed = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if (ROOT / relative).read_bytes() != committed:
            raise SystemExit(f"Working output does not equal committed evidence: {relative}")
        hashes[relative] = hashlib.sha256(committed).hexdigest()
    destination = ROOT / "src/static/research-version.json"
    destination.write_text(json.dumps(dict(commit=commit, files=hashes), indent=2) + "\n")
    print(f"Pinned {len(hashes)} evidence files to {commit}")


if __name__ == "__main__":
    main()
