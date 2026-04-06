# Notebooks

This folder is for research, exploratory analysis, and model iteration before ideas are promoted into `src/`.

## Suggested Workflow

1. Use notebooks to inspect data, test hypotheses, and compare modeling ideas.
2. Keep notebooks focused on one question each.
3. Move stable logic into `src/train.py` or other Python modules once it becomes part of the app.

## Recommended Notebook Order

- `01_data_exploration.ipynb`
  Explore the AQ dataset, missing values, class balance, and feature distributions.

- `02_model_experiments.ipynb`
  Compare classifiers, thresholds, calibration, and cross-validation results.

- `03_screening_workflow_logic.ipynb`
  Experiment with risk bands, follow-up triggers, report logic, and triage rules.

## Current Helpers

- `utils.py`
  Shared loading and summary utilities so notebooks stay cleaner and more consistent.

## Notes

- The current production app uses the adult AQ-10 dataset in `data/train.csv`.
- Notebook experiments should be treated as exploratory until integrated into `src/`.
- Avoid putting large generated outputs or secrets into this folder.
