# Utility scripts

`register_model.py` hashes a user-provided local checkpoint directory/file and
updates `configs/models.yaml` only after the same registry gate validates its
immutable revision, license family and inventory. It never downloads data.

`blender_export.py` is called only as a separate Blender headless process for
FBX. It is not imported by the Python pipeline.

`package_kaggle.py` uses only the Python standard library to build the source
attachment at `build/drishti-kaggle-source.zip`. It packages the Python application,
configurations, notebook and runbooks atomically, excluding mission data/weights.

`kaggle_run.py --settings <yaml>` invokes the typed GPU runner with stage-isolated
workers, measured memory stop budgets, partial-safe reporting and output archival.
Use `configs/kaggle.yaml.example` and the [Kaggle runbook](../docs/build/KAGGLE.md).
