# Deployment and bundle transfer

## Local offline ground station

Install the pinned Python extras and provision models/tools on the machine before
disconnecting it. Run `drishti doctor` and save its JSON output with the run. No
run-time call downloads models, geoid grids, imagery, basemaps, telemetry or
services. Start `python -m server.app` only on the local trusted interface.

The browser build is a static asset served by FastAPI. Cesium assets are copied
into `viewer/dist`; the viewer has no Ion token, CDN, remote imagery, geocoder or
terrain service. A fully offline model viewer can therefore operate after the
pipeline is finished, subject to real successful exports.

## Container build

The following Dockerfiles are intentionally small and install the pinned package;
bind-mount the input, model and run directories. A container cannot create a
checkpoint or fabricate a missing external tool.

```bash
docker build -f docker/local.Dockerfile -t drishti-local .
docker run --rm -it \
  -v "$PWD/data:/opt/drishti/data:ro" \
  -v "$PWD/models:/opt/drishti/models:ro" \
  -v "$PWD/runs:/opt/drishti/runs" \
  drishti-local drishti doctor
```

`docker/cloud.Dockerfile` is a CUDA/T4-oriented image for a manually selected
stage or the full bundle run. The core CLI executes on the invoking host; an
environment label is not a remote scheduler. Cloud work is durable: run against
a copied bundle, verify it, transfer it back and resume the downstream stages.
No cloud is mandatory for bundle semantics, but
neural checkpoints and GPU resources may be required by the selected profile.

## Kaggle GPU quality runner

Use the shipped [`kaggle_quality.ipynb`](../../notebooks/kaggle_quality.ipynb)
and [Kaggle runbook](KAGGLE.md). The notebook installs the pinned runtime and
invokes each core stage in a separate worker on the selected host/GPU. Typed
settings enforce separate maximum 30 decimal GB RAM/VRAM configured ceilings,
lower measured stop budgets, one physical GPU, sampled process-tree monitoring
and partial-safe report/archive recovery. Sampled telemetry and allocator limits
are not a hard instantaneous cap. Actual mission/GPU acceptance remains pending.

## Portable local↔cloud transfer

1. Create/run through the desired stage with `run.copy_inputs=true`.
2. Stop after a complete stage; do not copy a directory while its manifest is
   being written.
3. Transfer the entire bundle plus the exact model-cache registry entries and
   geoid files referenced by the manifest.
4. Run `drishti verify <bundle>` before continuing.
5. Run `drishti resume <bundle> --upto ...` or finish the full DAG.

The manifest records paths relative to the bundle where portability matters.
Model caches are not copied into the bundle by default; their verified registry
and hash inventory must be provisioned separately. A changed checkpoint or code
digest invalidates freshness. Never delete stale products until the new stage is
verified.

## Python and memory discipline

Python 3.12 is the tested environment. Open3D is a large optional install. T4
and local RAM availability must be measured on the actual machine. Use `fast` or
explicit chunk/tile limits for limited memory; use `max` only after checking
available memory. The report records the actual wall time and host environment;
the official <15-minute/10-minute-video value is not asserted without a measured
run on appropriate hardware.
