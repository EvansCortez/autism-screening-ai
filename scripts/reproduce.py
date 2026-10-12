"""Install the locked environment, test, and reproduce the final study."""

import hashlib
import json
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def main():
    required = (ROOT / ".python-version").read_text().strip()
    actual = ".".join(map(str, sys.version_info[:3]))
    if actual != required:
        raise SystemExit(f"Use Python {required}; this interpreter is {actual}.")
    protocol = json.loads((ROOT / "research/protocol.json").read_text())
    for relative, expected in protocol["inputs"].items():
        path = ROOT / relative
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Input missing or changed: {relative}. See research/STUDY.md.")
    env = ROOT / ".venv"
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    version = subprocess.check_output(
        [str(python), "-c", "import platform; print(platform.python_version())"], text=True
    ).strip()
    if version != required:
        raise SystemExit(f"Existing .venv uses {version}; create a Python {required} environment first.")
    run(python, "-m", "pip", "install", "-r", "requirements.txt")
    run(python, "-m", "pip", "check")
    run(python, "-m", "pytest", "-q")
    run(python, "-u", "-m", "src.evaluate", "--source-archive", "research/sources/uci_adult_426.zip",
        "--seeds", *protocol["seeds"], "--folds", protocol["folds"],
        "--bootstrap", protocol["bootstrap_draws"], "--output", "research/evaluation",
        "--tree-depths", *["none" if d is None else str(d) for d in protocol["tree_depths"]])
    for config_path in protocol["circularity_configs"]:
        name = Path(config_path).stem
        run(python, "-m", "src.circularity_audit", "--config", config_path,
            "--output", f"research/circularity/{name}")
    uncertainty = protocol["uncertainty"]
    run(python, "-m", "src.uncertainty", "--evaluation", "research/evaluation", "--output", "research/uncertainty",
        "--seeds", *protocol["seeds"], "--bootstrap-draws", uncertainty["bootstrap_draws"],
        "--permutation-draws", uncertainty["permutation_draws"], "--cluster-draws", uncertainty["cluster_draws"],
        "--bootstrap-seed", uncertainty["bootstrap_seed"])
    run(python, "-m", "src.report_assets", "--evaluation", "research/evaluation", "--circularity", "research/circularity",
        "--uncertainty", "research/uncertainty", "--output", "research/report")
    run(python, "scripts/verify_outputs.py")
    print("Final experiments, circularity audits, uncertainty analyses, and report assets reproduced.", flush=True)


if __name__ == "__main__":
    main()
