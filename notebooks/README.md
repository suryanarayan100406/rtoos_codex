# Cloud bundle runners

## Kaggle GPU quality notebook

Import [`kaggle_quality.ipynb`](kaggle_quality.ipynb) into Kaggle, select a GPU,
attach the source ZIP, real mission descriptor/video/GPS and verified local
checkpoint registry, then edit its configuration cell and run the cells in order.
See the [Kaggle runbook](../docs/build/KAGGLE.md) for complete setup, resource
controls, downloads and resume instructions.

Create the source attachment from the repository root:

```bash
python scripts/package_kaggle.py
```

The notebook uses Python 3.11/3.12 and pinned application dependencies in a separate
virtual environment. If Kaggle's host kernel is newer, setup provisions Python 3.12
with uv before creating that environment. Dependency setup may use the network;
reconstruction does not download models. Separate RAM/VRAM settings reject ceilings above 30 decimal
GB, select one physical GPU and use lower measured stop budgets. Stage isolation,
allocator limits and sampled telemetry do not establish a hard instantaneous cap.
No mission performance or trained-checkpoint quality benchmark is claimed.

The supplied sample is packaged as `build/aukerman-kaggle-input.zip`. Upload that
ZIP as a Kaggle Dataset named `aukerman-input`; Kaggle may expose it as the
individual descriptor, MP4, SRT and provenance files, which the notebook also
accepts directly. A separate model Dataset with the registered Depth Anything V2
Small checkpoint is required before the full runner can execute depth and
downstream reconstruction.
The ready-to-upload model package is `build/drishti-models.zip`; upload it as a
Dataset named `drishti-models`, and the notebook extracts and verifies it.

## Other cloud bundle transfers

Cloud operators can also install the same Python package, provision verified
models, mount/upload a complete bundle and invoke the CLI for selected stages:

```bash
drishti doctor
drishti resume /workspace/runs/<run_id> --upto s7_dense
```

Zip/transfer the complete bundle, run `drishti verify`, and continue locally.
The runner records the actual cloud host environment; an advisory placement
label is not proof that remote execution happened. Record the actual environment
and run output in the validation ledger.
