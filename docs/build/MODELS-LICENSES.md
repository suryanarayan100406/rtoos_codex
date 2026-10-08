# Offline checkpoints and license ledger

The registry starts as `models: {}` because no checkpoint, immutable revision,
or real hash was supplied. **No runtime download occurs.** Download/provision
weights separately, retaining their exact model card/license and immutable source
revision, then register every local file.

## Registry shape

Each name maps to `path`, `revision`, `license`, `files`, and optional `source`,
`runtime`, `output_domain`/OpenVINO preprocessing. Paths are relative to
`models.cache_dir` unless absolute. Revision must be an actual 40–64 hex commit
or content ID; each file has its measured SHA-256. Symlinks, unhashed additional
files, changed/missing files, and unknown specifications are rejected.

Use the registration helper after local provisioning:

```bash
python scripts/register_model.py --name depth_anything_v2_small \
  --path /absolute/path/to/local-small --revision ACTUAL_IMMUTABLE_COMMIT \
  --license Apache-2.0 --source depth-anything/Depth-Anything-V2-Small-hf \
  --runtime torch --output-domain relative_inverse

python scripts/register_model.py --name rtdetr \
  --path /absolute/path/to/local-rtdetr --revision ACTUAL_IMMUTABLE_COMMIT \
  --license Apache-2.0 --runtime torch
drishti doctor
```

The helper calculates real hashes, validates through the same license/inventory
gate, and writes the registry atomically. It cannot establish that a user-supplied
license declaration is legally correct; retain upstream evidence for that exact
revision. Placeholder revisions in examples are deliberately invalid.

Torch adapters use `local_files_only=True`, trusted built-in model classes, and
no remote code. Depth must declare `output_domain: relative_inverse` for the
implemented fitting method. A metric/range output must not be relabeled inverse.

## OpenVINO depth

Register a verified local XML/BIN pair with `runtime: openvino`, `model_file`,
`weights_file`, optional `output_name`, and explicit `preprocessing`:
layout `NCHW|NHWC`, color `RGB`, scale/mean/std, resize `stretch|aspect_multiple`,
and required dimensions/multiple. Every file including converter metadata must
be hashed. Set `depth.backend=openvino` and `compute.openvino_device=CPU|GPU|NPU`.
Conversion does not change source checkpoint licensing. Local IR adapter execution
is tested; trained-model aerial quality and target-device acceleration require
the actual approved model and hardware.

## Semantic checkpoints

Optional semantic inference uses the same exact-file license/inventory gate. No
semantic checkpoint is bundled or approved by model-family name. Provision a
local Transformers segmentation directory with its actual model configuration,
processor configuration and weights, then register it:

```bash
python scripts/register_model.py --name aerial_semantics \
  --path /absolute/path/to/local-semantic-model --revision ACTUAL_IMMUTABLE_COMMIT \
  --license Apache-2.0 --runtime torch --output-domain semantic_logits
```

The license and revision above are example arguments; use verified terms and the
actual immutable revision for your supplied checkpoint. The `neural` extra
provides Torch/Transformers. Configure the source-channel/class mapping in
`semantics.classes` as described in [INPUTS-CONFIG](INPUTS-CONFIG.md#optional-semantic-overlays).
The loader uses built-in Transformers classes, local files only and no remote
code. Loading/detecting a checkpoint is separate from recorded frame inference.
Resize-only processors are supported; cropped/padded predictions are not silently
stretched into camera geometry. Registry contents participate in resume identity
even after unavailable inference.

## Checkpoint decisions

| Family/variant | Shipped policy |
|---|---|
| Depth Anything V2 **Small** | Apache-2.0 candidate; exact revision/files required |
| V2 Base/Large/Giant | CC-BY-NC, rejected even if mislabeled |
| RT-DETR | Apache candidate; exact weights/license evidence required |
| MapAnything | Only specifically Apache variant eligible; not integrated |
| Pi3/Pi3X | Restricted weights, rejected; permissive code is insufficient |
| DA3 | Variant-specific licensing, no blanket family approval; not integrated |
| Metric3D | Verify exact checkpoint terms before adoption; not integrated |
| DUSt3R/MASt3R/UniDepth-V2/CUT3R/restricted VGGT | Reference-only; gate rejects known restricted identities |
| Transformers semantic segmentation | Optional local adapter implemented; exact checkpoint license/revision/mapping required; no trained-checkpoint run verified |
| SAM2/ByteTrack/RAFT | Planned masking enhancements, no shipped checkpoint execution |

The gate allowlist is Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause. It blocks known
restricted names/source identities in addition to declared license checks.

## Tool/library ledger

Python application dependencies are pinned directly in `pyproject.toml`; viewer
dependencies and transitive resolutions are in its lockfile. COLMAP/GTSAM are
BSD-family, Open3D/trimesh/FastAPI/React/NumPy are permissive, Cesium Apache-2.0,
Potree BSD-family, PROJ/GDAL/rasterio permissive. Preserve upstream notices when
redistributing. PyAV uses FFmpeg; FFmpeg's actual build options affect licensing.
This project does not claim every external binary build is permissive merely
from its executable name.

**Blender is the explicit GPL external-tool exception** prescribed by the docs:
it runs a separate headless process, and DRISHTI does not import/link `bpy`.
External-tool distribution obligations still apply. Blender, geoid grids, optional
converters and model weights are not bundled in the source checkout.
