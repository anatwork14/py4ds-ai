# Historical coursework archive

These materials were part of the original assignment repository and have been moved here **without changing their Git blob content**.

```text
legacy/
├── dataset_metadata.json
├── features/                 # Historical saved .npy arrays
├── modules/                  # Original helper scripts
├── notebooks/main.ipynb      # Original output-heavy notebook
├── report.pdf                # Original submission PDF (outdated scores)
└── requirements.txt          # Original unpinned dependency list
```

**Not the maintained pipeline.** The up-to-date implementation lives in `../src/py4ds_ai/` and `../scripts/`, with dependencies in `../pyproject.toml` and `../uv.lock`.

The old notebook, PDF and arrays contain historical results from the earlier workflow. Their scientific provenance and discrepancies are discussed in [the repository audit](../docs/REPO_AUDIT.md) and [experiment log](../docs/EXPERIMENT_STATUS.md).

**Warning:** legacy helper scripts can change, move or delete source directories and may use the held-out test in ways prohibited by the newer validation protocol. They are archived for inspection, not recommended for execution.

No raw Kaggle images or new saved model checkpoints were added by this reorganization.
