# Local data workspace

This directory intentionally contains **no dataset images or committed splits**. Dataset licenses and the provenance of the historical experiment must be respected.

The source dataset is [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). The package's data tools handle both flat and nested `seg_train`, `seg_test` and `seg_pred` directories without moving the source images.

Typical local-only layout:

```text
data/
├── raw/
│   └── extracted/                # User-managed Kaggle dataset
│       ├── seg_train/...
│       ├── seg_test/...
│       └── seg_pred/...
├── manifests/
│   └── seed-42-phash-reviewed/   # Historical reviewed split; not in Git
├── interim/                     # Optional scratch data
└── processed/                   # Optional generated features
```

These directories are ignored by Git. The exact reviewed manifest and split hashes are documented in [PHASE1_DATA_AUDIT](../docs/PHASE1_DATA_AUDIT.md). The tracked review ledger is in [configs](../configs/README.md).

**Do not regenerate or alter the historical reviewed manifest** for the completed final report; and do not score the held-out test a second time. New experiments would require a separate protocol and output namespace.
