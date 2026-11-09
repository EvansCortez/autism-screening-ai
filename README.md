# ASD-Predict

An undergraduate research project studying machine learning for adult autism
screening-label reconstruction, with a separate FastAPI demonstration app.

The local UCI labels match AQ sum >=7 exactly. The documented ARC AQ-10 referral
guide uses >=6. Neither agreement with these labels nor the demo's probability
output establishes clinical diagnostic accuracy.

## Reproduce the final study

Install Python **3.14.5**, then run this single command from the repository root:

```bash
python3.14 scripts/reproduce.py
```

The script checks Python and input checksums, creates `.venv` if needed, installs
the exact dependencies in `requirements.lock`, checks dependency compatibility,
runs the tests, and executes the final experiments. It needs package-index access
if required packages are not already installed. An existing `.venv` must also
use Python 3.14.5; the script will not replace a different environment.

Results are regenerated in `research/evaluation/`, `research/circularity/`,
`research/uncertainty/` and `research/report/` (publication figures and tables).
Serving artifacts in `models/` are not overwritten. Experiment seeds and input checksums live in
`research/protocol.json`. Environment capture is in the generated
`research/evaluation/environment.txt`; the maintained dependency lock is
`requirements.lock`. Platform-specific dependencies have environment markers;
numerical identity across operating systems is not guaranteed.

## Research protocol and results

- [Study](research/STUDY.md): source evidence, confirmed findings versus hypotheses,
  methods, uncertainty limitations, and evidence needed for publication.
- [Findings](research/FINDINGS.md): computed results with interpretation.
- [Reusable circularity audit](research/CIRCULARITY.md): configurable source/schema
  checks and score-label dependence analysis, validated on adult and child UCI data.
  The reproduction command runs both audits into `research/circularity/`.
- [Final notebook](notebooks/03_reproducible_evaluation.ipynb): interactive access
  to the same evaluation runner.
- `src/evaluate.py`: repeated grouped validation of dummy, AQ >=6 / >=7 rules,
  logistic regression, shallow decision tree, and random forest.
- Four feature sets: combined with sum, combined without sum, questionnaire only,
  and non-questionnaire predictors. Preprocessing and oversampling stay inside
  training folds. An age/duplicate sensitivity cohort is evaluated separately.
- Explanatory outputs: a score-threshold scan, label-versus-score counts,
  training-fold questionnaire logistic coefficients, and a tree-depth comparison.
  The final notebook displays these alongside the main results.
- `src/uncertainty.py`: checks on how far standard binomial intervals can be trusted for
  cross-validated results (whole-procedure and country-cluster bootstraps, a label-shuffle
  null), a nested check of the cut-off, a decomposition of the signal left without
  questionnaire items, and descriptive subgroup counts.
- `src/report_assets.py`: publication figures and tables built from the saved CSV outputs.
- Research coordination and writing aids are in [research/](research/README.md): shared task
  list, methods guide, paper outline, literature comparison, and a record of AI assistance.

The original dataset is verified against the retained UCI archive. Original
random splits allow duplicate predictor overlap; the final experiments keep
matching predictor records in the same fold. Participant identities remain unknown.

The preliminary notebook and its outputs are preserved under
`research/archive/preliminary/`. They use a different protocol and are excluded
from final results. `research/REVIEW.md` is a historical independent review.

## Use the demo app

After environment setup:

```bash
source .venv/bin/activate
uvicorn src.main:app --reload
```

Open `http://127.0.0.1:8000/` for the demo and `/docs` for the API schema.
Endpoints: `GET /`, `GET /api`, `GET /health`, `POST /predict`,
and `POST /screening-report`. Reports accept `subject_name`, adult profile fields,
ten already-scored binary AQ items, observation notes, and follow-up fields.

The questionnaire columns are keyed scores, not generic yes/no answers.
The report's keyword flags, risk bands, and probability wording are unvalidated
demonstration behavior. Observation notes do not change the trained classifier.

## Legacy serving-model training

```bash
python -m src.train
```

This command replaces `models/autism_model.pkl` and `models/metrics.json` using
the original random-split protocol. Those artifacts describe the serving demo,
not the final research evaluation. Their 100% holdout/CV agreement and 73.05%
majority baseline should not be mixed with the final grouped experiment results.
The legacy pipeline does not save an AQ baseline and permits duplicate overlap.

## Tests and notebooks

```bash
python -m pytest -q
jupyter lab
```

See [notebooks/README.md](notebooks/README.md). The reproduction command already
runs tests; Jupyter is optional for inspecting results.

## Docker and CI

```bash
docker build -t asd-predict .
docker run -p 8000:8000 asd-predict
```

Docker serves the committed demo model and does not run research experiments.
CI uses the Python version in `.python-version` and the pinned dependency lock
to run tests. Docker uses the same Python patch version. These configuration
changes do not imply that a remote CI run or image build has been verified.

## Sources and scope

- [UCI Adult Autism Screening dataset](https://doi.org/10.24432/C5F019):
  Fadi Thabtah, 2017; CC BY 4.0.
- [ARC adult AQ-10 form](https://www.autismresearchcentre.com/content/uploads/2024/11/AQ10.pdf).

The accessible source documentation describes screening data and an automated
score; it does not establish independently assessed diagnoses. Exact label
agreement is confirmed, while the original label-generation procedure remains
unverified. This educational project has not been clinically validated.
