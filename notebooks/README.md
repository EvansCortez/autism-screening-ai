# Notebooks

## Active notebooks

- `01_data_exploration.ipynb`: introductory exploration with `utils.py`.
- `03_reproducible_evaluation.ipynb`: final research protocol using `src.evaluate`.
- `04_label_circularity_audit.ipynb`: reusable configured audit of adult and child
  UCI screening labels. See `research/CIRCULARITY.md` for source evidence and how
  to configure another dataset. The child data do not validate the adult model.

Read [the study](../research/STUDY.md) before interpreting results.
The introductory notebook uses legacy cleaning helpers; its summaries are not
the final experiment audit. The final runner preserves the serving model's
handling in its primary cohort and separately evaluates cleaning sensitivity.

## Reproduction

From the repository root, with Python 3.14.5 installed:

```bash
python3.14 scripts/reproduce.py
.venv/bin/jupyter lab
```

The first command installs pinned dependencies, runs tests, and generates final
results. The second opens the optional interactive workspace. Select the project's
Python kernel. The final notebook uses the same seeds, folds, and bootstrap count
from `research/protocol.json`.

## Archived exploration

`02_model_experiments.ipynb` and its outputs are in
`research/archive/preliminary/`. Its unoversampled models and different age
handling make those results unsuitable for mixing with the final study.
No screening-workflow notebook is currently implemented.

Use public dataset or synthetic records in notebooks. Move stable logic into
tested modules before integrating it into the app.
