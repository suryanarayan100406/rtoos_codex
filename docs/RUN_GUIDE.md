# DRISHTI — How to Run It (Step by Step, in Detail)

> **Historical source document:** referenced bundles/results are not present in
> this checkout. Current commands and validation are documented in
> [build/QUICKSTART.md](build/QUICKSTART.md) and [build/STATUS.md](build/STATUS.md).

This is the **hands-on walkthrough**: how to install it, how to point it at your own footage,
**exactly how frames get linked to telemetry and to each other**, how to run it end-to-end (locally
and on a free cloud GPU), and how to view and share the result.

It is a companion to [`GUIDE.md`](GUIDE.md) (the code-accurate reference + "how it works" internals)
and [`../AGENTS.md`](../AGENTS.md) (project status + decision log). Where this guide states a number or a
flag, it matches the shipped code as of this run.

> **What DRISHTI is, in one line:** you hand it *one pre-recorded drone video plus its GPS/flight
> telemetry*, and it produces *one accurate, georeferenced 3D model* + standard deliverables — entirely
> offline, on the ground. No drone, no live link, no internet required to run it.

---

## 0. The 60-second version

```bash
# 1. install (Python 3.11+; one-time)
python -m venv .venv && .venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -e ".[video,poses,geo,recon,depth,telem]"

# 2. check what this machine can actually do
drishti doctor

# 3. make a real test dataset (downloads real aerial imagery + real GPS)
python scripts/make_sample_dataset.py --dataset brighton_beach

# 4. run the whole pipeline
drishti run --dataset configs/datasets/brighton_beach.yaml --profile balanced

# 5. read the truth about the run, then open the report
drishti inspect runs/<run_id>
start runs/<run_id>/report/report.html
```

Everything below is the same thing, explained so you can do it with **your own footage** and understand
what each step is doing.

---

## 1. What you must provide (the input contract)

DRISHTI's entire input is **a video file + a telemetry track**, described by a tiny YAML file called a
**dataset descriptor**. That's it.

| Input | Required? | What it is |
|-------|-----------|-----------|
| **Video** | **yes** | The recorded clip, e.g. `flight.MP4` (1080p or 4K). Decoded with PyAV, or OpenCV as a fallback. |
| **Telemetry** | **yes** | A GPS/flight track that is **time-aligned to the video**. One of four formats: `dji_srt`, `csv`, `mavlink`, `exif`. |
| IMU / baro / intrinsics / RTK | optional | Constrain or improve later stages if present. |
| Ground check-points | optional | A CSV of surveyed points — this is what turns the report's accuracy from *UNVALIDATED* into a **measured** elevation error. |

Two things to internalize now, because they decide whether you get a good model:

1. **Georeferencing comes from the GPS in your telemetry** — not from the video. No telemetry, no
   metric/georeferenced output.
2. **The photogrammetry quality comes from the video's overlap.** The frames must see the same ground
   from many angles. A lawnmower/grid flight with **~70–80% forward and side overlap** is the target. A
   single fast fly-through with little overlap will reconstruct patchily (you'll see this as low
   `dsm_coverage` in the report).

---

## 2. Prerequisites & install

### 2.1 Core vs. heavy extras

The **core** (`pip install -e .`) is tiny — Pydantic, Typer, Rich, NumPy — and gives you a working
`drishti` CLI immediately, but the heavy stages will honestly report themselves as *blocked* until you
install their extras. Python **3.11+** is required.

Install the extras for the stages you want to run **in this environment**:

| Extra | Installs | Unlocks |
|-------|----------|---------|
| `video` | `av`, `opencv-python-headless` | S0 ingest, S1 frame-QA, S3/S4 I/O |
| `poses` | `pycolmap` | S2 camera poses (SfM) |
| `geo` | `pyproj`, `rasterio`, `shapely` | S2 projection, S9 GeoTIFFs, report validation |
| `recon` | `open3d`, `trimesh`, `pygltflib` | S7 dense, S8 mesh, S10 export |
| `depth` | `torch`, `openvino`, `onnx` | S3 masking, S4 metric depth |
| `telem` | `pymavlink`, `piexif` | MAVLink/CSV/EXIF telemetry parsing |
| `server` | `fastapi`, `uvicorn` | the API server + web viewer |

**Full CPU-runnable local install (no GPU needed):**

```bash
pip install -e ".[video,poses,geo,recon,depth,telem]"
```

### 2.2 Three real gotchas that aren't obvious (learned the hard way)

- **Two libraries the neural + LAS stages need are in *no* extra.** Install them explicitly or S3/S4 and
  S10-LAS will fail:
  ```bash
  pip install "transformers>=4.45" timm safetensors "laspy[lazrs]>=2.5"
  ```
- **Open3D has no wheel for Python 3.13** (it caps at cp312). Colab/Kaggle currently run 3.13, so a plain
  `pip install -e ".[recon]"` will *abort the whole install* there and take the `drishti` CLI down with
  it. On those machines **use the notebooks** (`notebooks/drishti_colab_full.ipynb`), which build an
  isolated Python 3.12 env with `uv`. On your own machine, just use Python ≤3.12.
- **FBX export needs Blender ≥4.0 on PATH.** Everything else exports without it.

### 2.3 Always run the doctor first

```bash
drishti doctor
```

This is the single most useful command. It prints, for *this* machine: CPU/RAM, whether CUDA and
OpenVINO are present, which optional modules import, **per-stage readiness** (can it run here, or will it
fail loudly), and which external tools (COLMAP, GLOMAP, Blender, PDAL, …) are detected. Run it before
every fresh environment.

---

## 3. Import your own data (writing the dataset descriptor)

### 3.1 Where files go

Convention (not enforced, but tidy):

```
data/<mission>/flight.MP4          # your video   (data/ is git-ignored)
data/<mission>/flight.SRT          # your telemetry
configs/datasets/<mission>.yaml    # the descriptor that points at them
```

Paths in the descriptor are resolved **relative to the repo root** (or you can use absolute paths).

### 3.2 The descriptor schema (every field)

This is the complete contract (`src/drishti/config/models.py::DatasetDescriptor`):

```yaml
name: my_mission                     # (required) a label for the run
video: data/my_mission/flight.MP4    # (required) path to the video

telemetry:                           # (required)
  format: dji_srt                    #   dji_srt | csv | mavlink | exif
  path: data/my_mission/flight.SRT   #   path to the telemetry file
  csv:                               #   ONLY for format: csv — column name mapping
    time: timestamp                  #     (defaults shown; change to match your CSV headers)
    lat: latitude
    lon: longitude
    alt: abs_alt
    yaw: null                        #     optional attitude columns
    pitch: null
    roll: null

optional:                            # all optional; leave null if you don't have them
  imu: null
  baro: null
  intrinsics: null                   #   a known camera calibration
  rtk: null                          #   RTK/PPK corrected track

crs:                                 # coordinate reference system (optional block)
  mode: derive_from_gps              #   derive_from_gps (default) | epsg
  epsg: null                         #   set an int (e.g. 32643) AND mode: epsg to force a CRS

report:                              # optional
  check_points: null                 #   path to a surveyed-points CSV to validate elevation accuracy
```

> **Note (code-accurate):** to *force* a CRS, the value is `mode: epsg` with an `epsg:` number — not
> `mode: set`. If you leave the whole `crs` block out, DRISHTI derives the correct UTM zone from your GPS
> automatically (see §4.4), which is what you want 95% of the time.

### 3.3 Worked example — DJI footage

DJI drones write a `.SRT` subtitle sidecar next to the video with bracketed GPS fields
(`[latitude: ..] [longitude: ..] [abs_alt: ..]`). That's the `dji_srt` format — the easiest case:

```yaml
name: backyard_survey
video: data/backyard_survey/DJI_0042.MP4
telemetry:
  format: dji_srt
  path: data/backyard_survey/DJI_0042.SRT
```

That is a complete, runnable descriptor. Everything else defaults sensibly.

### 3.4 Worked example — a generic flight log (CSV)

If your telemetry is a CSV exported from any flight-log tool, use `format: csv` and map your column
headers. Say your CSV has columns `t_sec, lat_deg, lon_deg, alt_m`:

```yaml
name: field_mapping
video: data/field_mapping/flight.mp4
telemetry:
  format: csv
  path: data/field_mapping/telemetry.csv
  csv:
    time: t_sec
    lat: lat_deg
    lon: lon_deg
    alt: alt_m
```

The `time` column must be on the **same clock as the video** (see §4.2). If it's an absolute wall-clock
timestamp and the video starts at 0, use `ingest.time_offset_s` to line them up (§4.2).

### 3.5 The other two telemetry formats

- **`mavlink`** — ArduPilot/PX4 `.tlog` or `.bin` logs, parsed via `pymavlink` (needs the `telem` extra).
  `path:` points at the log; no column mapping needed.
- **`exif`** — for an **image sequence** rather than a video: GPS is read from each image's EXIF. Useful
  when your "video" is really a folder of geotagged stills.

---

## 4. How frames are linked (the part people ask about)

This is the heart of the pipeline. There are **three distinct linkages**, and each is a real stage with
knobs you can turn.

### 4.1 Link #1 — video → frames (sampling), in **S0 ingest**

S0 decodes the clip and **samples frames at a fixed rate**, `ingest.target_fps` (default **4.0 fps**).
Each sampled frame keeps its **presentation timestamp (PTS)** from the video — that timestamp is the key
used for everything downstream.

Knobs (`ingest.*`):

| Key | Default | Effect |
|-----|---------|--------|
| `ingest.target_fps` | `4.0` | Frames sampled per second of video. Higher = more frames = more overlap but slower. 2–4 fps is typical for aerial. |
| `ingest.max_frames` | `0` (all) | Hard cap on total frames — handy for a quick smoke test. |
| `ingest.time_offset_s` | `0.0` | Shifts telemetry relative to the video (see next). |

Example: `--set ingest.target_fps=2 --set ingest.max_frames=200` for a fast trial.

### 4.2 Link #2 — frames → telemetry (georeference), also in **S0 ingest**

This is the "linking frames to GPS" step. S0 parses the telemetry track, **time-aligns it to the video
timeline, and interpolates the GPS track to each frame's timestamp**, attaching an interpolated
`lat/lon/alt` to every sampled frame. It also **derives the CRS** from the GPS here (§4.4).

The linkage is therefore **temporal**: frame at t=12.5 s gets the GPS position the track reports at
t=12.5 s (interpolated between logged samples). For this to be correct:

- Your telemetry's clock must correspond to the video's clock. DJI SRT is already frame-aligned. A CSV
  with wall-clock timestamps may be offset — **correct it with** `--set ingest.time_offset_s=<seconds>`
  (positive or negative).
- S0's confidence is literally the **temporal coverage**:
  `min(1.0, telemetry_duration / video_duration)`. If your telemetry only covers half the video, S0 tells
  you (confidence ≈ 0.5) instead of pretending.

You can sanity-check this immediately by running only S0 (see §6.4) and reading `s0_ingest/frames.json`.

### 4.3 Link #3 — frames → each other (geometry), in **S1** then **S2**

This is what actually builds the 3D structure — connecting frames by *what they see in common*.

**S1 frame-QA + keyframe selection** (`frameqa.*`): scores every frame for sharpness (variance of
Laplacian) and exposure, drops the unusable ones, then **selects keyframes** so consecutive keyframes
have enough baseline to triangulate. Method `frameqa.keyframe.method`:

| Method | What it does | Key knob |
|--------|--------------|----------|
| `parallax` (default) | Keeps a frame once real feature displacement (ORB) exceeds a threshold | `frameqa.keyframe.min_parallax_px` (default 8.0) |
| `feature_disp` | Similar, displacement-based | same |
| `fixed_stride` | Every Nth frame, regardless of motion | `frameqa.keyframe.stride` (default 5) |

Cap the count with `frameqa.keyframe.max_keyframes` if a stage is too slow.

**S2 poses — the actual frame-to-frame linking** (`poses.*`): this is COLMAP Structure-from-Motion. It
extracts features in each keyframe, **matches them across frames**, and solves for camera intrinsics +
per-frame 6-DoF poses in a visual coordinate frame. A **Sim(3)** (Umeyama scale+rotation+translation)
fit then anchors that visual frame onto your GPS track in the projected CRS — that's what makes the whole
reconstruction metric and georeferenced. The fit RMSE (`spine.json → fit_rmse_m`) is the accuracy proxy.

The single most important knob here is **the matcher**, `poses.matcher`:

| Value | When to use |
|-------|-------------|
| `sequential` (default) | Video where consecutive frames are neighbors and the path never revisits — a single linear sweep. |
| **`exhaustive`** | **Aerial lawnmower/grid surveys.** Frames on adjacent strips see the same ground but are far apart in time; only all-pairs matching links them. |
| `vocab_tree` | Large sets where exhaustive is too slow. |

> **Proven on this project:** switching an aerial grid survey from `sequential` to `exhaustive` dropped
> the georeferencing fit RMSE from **26.3 m → 6.5 m**. If you fly a grid, set it:
> ```bash
> drishti run -d configs/datasets/<mission>.yaml --profile balanced --set poses.matcher=exhaustive
> ```

Other S2 knobs: `poses.self_calibrate_intrinsics` (default on — recovers focal length/distortion),
`poses.min_track_len` (default 3), `poses.use_glomap`.

**Practical takeaway for good linking:** fly with high overlap, keep frames sharp (avoid motion blur →
S1 will drop blurry frames), and use `exhaustive` for grids. Everything downstream (depth, dense cloud,
mesh) is only as good as the poses S2 recovers.

### 4.4 CRS derivation (automatic)

With the default `crs.mode = derive_from_gps`, S0 takes the **median lon/lat** of your track, computes
the **UTM zone**, and picks the EPSG code (`32600 + zone` north, `32700 + zone` south). You do nothing;
the result is recorded in the manifest. All geometry then lives in that metric, projected CRS.

---

## 5. Don't have footage yet? Generate a real test dataset

DRISHTI ships **no data** (honesty policy), but `scripts/make_sample_dataset.py` will assemble a real
`video + telemetry + descriptor` for you:

```bash
python scripts/make_sample_dataset.py --list                    # see the registry
python scripts/make_sample_dataset.py --dataset brighton_beach  # ~62 MB, 18 imgs — fast smoke test
python scripts/make_sample_dataset.py --dataset aukerman        # ~543 MB, real buildings (house + barn)
python scripts/make_sample_dataset.py --synthetic               # rendered city w/ ground-truth (no network)
```

Each writes `data/<name>/` **and** a matching `configs/datasets/<name>.yaml`, so you can run it straight
away. The `--dataset` modes use **real** source imagery + **real** per-image GPS EXIF (only the playback
*timing* is synthesized); `--synthetic` is a labelled procedural scene with ground-truth you can check
accuracy against. Big sets like `aukerman` are best fetched **on the cloud T4**, where the heavy run
happens (the notebook has a cell for it).

---

## 6. Run the pipeline

### 6.1 The one command

```bash
drishti run --dataset configs/datasets/<mission>.yaml --profile balanced
```

`--dataset/-d` is **required** (there is no positional form). Full `run` flags:

| Flag | Meaning |
|------|---------|
| `-d, --dataset PATH` | **(required)** the descriptor YAML. |
| `-p, --profile NAME` | `fast` \| `balanced` (default) \| `max`. |
| `-o, --output-root DIR` | where the bundle is created (default `runs/`). |
| `--run-id ID` | force a run id (default: timestamp + suffix). |
| `--only STAGE` | run just one stage (its deps must already be done). |
| `--upto STAGE` | run from the start through this stage. |
| `--force` | re-run even if inputs/params are unchanged. |
| `-s, --set k=v` | override **any** config key; repeatable. |

### 6.2 Profiles — and the one that crashes

- **`fast`** — quick smoke-test settings.
- **`balanced`** — the default; runs the whole pipeline end-to-end on the implemented path. **Use this.**
- **`max`** — highest quality, but it selects `dense.method=gaussian` + `mesh.method=2dgs`, which this
  build does **not** implement — S7/S8 will **raise loudly**. If you want `max`'s other quality knobs,
  override the two methods to the implemented path:
  ```bash
  drishti run -d configs/datasets/<mission>.yaml --profile max \
    --set dense.method=tsdf --set mesh.method=poisson
  ```

### 6.3 Local vs. free cloud T4 (the recommended split)

The dev target is an Intel Core Ultra 7 155H **with no discrete GPU**. The neural stages (S3/S4/S7) want
CUDA, so the intended workflow splits the run — and because the bundle is **content-hashed and
resumable**, finished stages are simply skipped when you continue elsewhere.

| Tier | Stages | Why |
|------|--------|-----|
| **Cloud T4** (Colab/Kaggle) | S0 → **S7** | S3/S4/S7 need CUDA; S2's COLMAP benefits from the T4's faster CPU. |
| **Ground station** (your CPU) | S8 → report | Poisson meshing, rasterization, export, report are CPU work. |

**On the T4** (stop after dense):
```bash
drishti run --dataset "$MISSION" --profile max \
  --set dense.method=tsdf --set mesh.method=poisson --upto s7_dense
# then inspect + verify + zip the bundle and download it
```
**On the ground station** (unzip into `runs/`, then continue — GPU stages are already `done`):
```bash
drishti resume runs/<run_id>          # picks up at s8_mesh, reusing the recorded config
```

**Prefer one environment?** `notebooks/drishti_colab_full.ipynb` runs **all 11 stages** on a T4
(`balanced`), and then you only *view/serve* the finished bundle locally — no heavy deps on your machine.
That's the path this project's `aukerman` model was produced with.

### 6.4 Smoke-test first

Confirm your inputs parse and the CRS derives before committing to a full run:

```bash
drishti run -d configs/datasets/<mission>.yaml --upto s0_ingest
drishti inspect runs/<run_id>          # check the derived CRS + telemetry coverage
```

### 6.5 Reproduce **this exact model** yourself on a free cloud GPU (Kaggle, cell by cell)

The `aukerman` model in this repo wasn't hand-built — it came out of **one notebook,
`notebooks/drishti_colab_full.ipynb`, run top-to-bottom on a free Kaggle T4**. Below is that same run so
you can produce the bundle yourself (or run your own footage the identical way). Nothing here is faked: it
clones the real repo, installs the real stack, and runs the real `drishti` CLI — a missing dependency or a
bad input **fails loudly**.

> **Why Kaggle, not Colab's free tier?** The dense stage (S7) integrates depth out to `depth_trunc_m=150`
> at a **5 cm** voxel, and that working set needs **~30 GB RAM**. Kaggle's free tier gives ~30 GB; Colab's
> free tier gives ~13 GB and gets OOM-killed at S7. So use a **free Kaggle T4**, or a Colab **High-RAM**
> runtime. (Everything runs on the GPU; your own machine only *views* the result — no heavy deps needed.)

Run the notebook cells in order. What each does, and the one decision in each:

| Notebook § | Cell | What you do / what it does |
|-----------|------|----------------------------|
| **0** | `!nvidia-smi` | Kaggle: *New Notebook → Settings → Accelerator = **GPU T4 ×2**, Internet = **On*** (needed to clone + fetch model checkpoints). This cell must show a Tesla T4 — if not, fix the accelerator before spending time. |
| **1** | code | **Set `REPO_URL`** to your DRISHTI git URL (and `REPO_REF` to a branch/commit). Runs a clone into `/kaggle/working/drishti` and hard-resets to the latest commit (so a stale checkout can't silently resurrect a fixed bug). |
| **2** | code | **Install — just run it.** Because Kaggle is on Python 3.13 and Open3D ships no 3.13 wheel, the cell auto-builds an **isolated Python 3.12 env with `uv`**, installs core **first** (so the CLI always lands), then every extra, then the two gap packages (`transformers`, `laspy`). Ends with an `OK`/`MISSING` probe — everything should read **OK**. |
| **3** | code | *(optional)* Fetches portable **Blender 4.2** so the FBX deliverable is real. Skip it → FBX is recorded *skipped* (S10 **degraded**, not failed); every other format still exports. |
| **4** | `!drishti doctor` | **The honest gate.** Expect **cuda: yes** and every stage **ready**. Anything still `blocked` here *will* fail when reached — fix it before running. |
| **5** | Option **D** | **Set `DATASET = "aukerman"`** and run the Option-D cell. It calls `make_sample_dataset.py --dataset aukerman` (fetches ~543 MB of real, GPS-tagged aerial imagery), writes `configs/datasets/aukerman.yaml`, and sets `MISSION`. *(Options A/B = your own video; C = resume a partial bundle.)* |
| **6** | Block **6B** | **Run Block 6B** (the high-RAM one) — **not 6A.** It executes exactly the command below and tees the whole console to `logs/run_console.log`. |
| **7** | inspect/verify | `!drishti inspect "$BUNDLE"` + `!drishti verify "$BUNDLE"` — the per-stage truth table and a hash check of every output. |
| **8–9** | logs / export | Prints the report summary, then zips **full** (everything, reproducible) and **slim** (deliverables + report + logs + manifest). On Kaggle, grab `<run_id>_full.zip` from the file browser on the right (`/kaggle/working`). |

The one command that does the real work (Block 6B, verbatim):

```bash
drishti run --dataset <MISSION> --profile balanced --run-id colab-full-<timestamp> \
  --set poses.matcher=exhaustive \
  --set dense.depth_trunc_m=150
```

> **Those two `--set` flags are the whole story of a good aerial model — and both are forced by the data,
> not by taste:**
> - **`poses.matcher=exhaustive`** — lawnmower strips only loop-close under all-pairs matching; the default
>   `sequential` links only consecutive frames, so the trajectory drifts and the mesh smears. Measured on
>   aukerman: fit RMSE **26.3 → 6.5 m**, sparse points **15.9k → 34.9k**, **75/75** frames registered.
> - **`dense.depth_trunc_m=150`** — the cameras fly ~100–130 m AGL, so the ground sits ~100 m away; the
>   clamp must clear that (≈ 1.3× max AGL) or S7 **deletes the ground** and you get floating fragments.
>
> **If Block 6B dies with `[exit -9]`** (the OS OOM-killing S7, not a DRISHTI error): dense fusion is
> spatially **tiled**, so shrink the tile — add `--set dense.tile.tile_m=40` (more, smaller tiles = lower
> peak RAM). Voxel and `depth_trunc` stay put — **no resolution loss.** Do **not** fall back to 6A; its
> 40 m clamp deletes the ground. Completed stages are cached, so set `RESUME_BUNDLE = BUNDLE` and re-run.

The `<run_id>_full.zip` you download is exactly the shape of the `colab-full-20260910-212357` bundle in
this repo. Round-trip it back to your machine and you only *view* it (§8):

```bash
# on your machine — no GPU/heavy deps needed, just unzip and look
tar -xf colab-full-<timestamp>_full.zip -C runs/     # or: Expand-Archive in PowerShell
drishti inspect runs/colab-full-<timestamp>          # same manifest you saw on Kaggle
drishti verify  runs/colab-full-<timestamp>          # confirm the download is intact
#   then view it with §8 below (viewer.html / view_model.cmd)
```

### 6.6 Run it on YOUR OWN footage — no repo commit (notebook Option E)

Everything above reproduces the **aukerman** model. To run the same pipeline on **your own** recording —
without committing anything to the repo — use **Option E** in the notebook
(`notebooks/drishti_colab_full.ipynb`, §5 "Import"). It is an upload form for **every** input DRISHTI can
take, and it writes the dataset descriptor for you.

**What you provide** (one cell, top of Option E):

| Input | Required? | How to give it |
|-------|-----------|----------------|
| **video** | yes | `"upload"` → file picker (Colab), or a path to a file already on the machine |
| **telemetry** | yes | same; set `TELE_FORMAT` = `dji_srt` \| `csv` \| `mavlink` \| `exif` (CSV also takes a column map) |
| **imu / baro / intrinsics / rtk** | optional | `"upload"` to add one, or leave `""` to skip it |
| **check-points** | optional | surveyed control for a real accuracy check; else the report stays UNVALIDATED |

You also set the CRS (`derive_from_gps` or an explicit `epsg`) and the ingest knobs
(`TARGET_FPS`, `MAX_FRAMES`, `TIME_OFFSET_S`) right in the cell.

**Why there is no commit.** The cell stages your uploads under `data/<name>/` and writes the descriptor to
`data/<name>/<name>.yaml` there — a path already covered by `.gitignore` (`data/*` + every media
extension), so nothing you upload or generate is ever tracked by git. Large files are referenced **in place
by absolute path**, so they are not copied and the descriptor works regardless of the working directory.
The cell then validates that descriptor against the real `load_dataset` contract **before** any GPU time,
and exports `MISSION` (plus `DRISHTI__ingest__*` env overrides for the knobs) so the run cell (§6) works
**unchanged** — just run it.

- **Colab:** `"upload"` opens a picker; or mount Drive and pass a `/content/drive/...` path.
- **Kaggle:** there is no pop-up picker — use **＋ Add Input → Upload** and pass the
  `/kaggle/input/<your-dataset>/<file>` path (or drag into `/kaggle/working` and pass that path). The cell
  prints these exact instructions if a required upload is missing.

The output bundle has the identical shape to `colab-full-20260910-212357`; round-trip and view it exactly
as above.

---

## 7. Watch it, and read the outputs honestly

### 7.1 The commands that tell the truth

| Command | What it tells you |
|---------|-------------------|
| `drishti inspect runs/<id>` | Provenance, derived CRS, inputs table, and the **per-stage table**: status, environment, wall-seconds, measured confidence, degradation notes. This is ground truth, not a progress bar. |
| `drishti verify runs/<id>` | Re-hashes every completed stage's outputs vs. the manifest → `OK`/`MISMATCH`/`MISSING`. Exits non-zero if anything changed on disk. |
| `drishti resume runs/<id>` | Continue a partial/failed run. Reuses the recorded config (so it takes `--only`/`--upto`/`--force`, but **not** `--profile`). |
| `drishti stages` | The canonical stage order + each stage's dependencies. |
| `drishti doctor` | What can run in this environment. |

### 7.2 The bundle you get

```
runs/<run_id>/
  manifest.json     # the spine — provenance, per-stage records, timings, confidence (written after every stage)
  s0_ingest/        # telemetry.json, frames.json, decoded frames
  s1_frameqa/       # frameqa.json, keyframes.json
  s2_poses/         # poses.json, spine.json (fit_rmse_m), sparse_points.json, COLMAP workdir
  s3_masking/  s4_depth/  s6_global/     # masks, metric depth, global poses
  s7_dense/         # dense.ply (colored point cloud)
  s8_mesh/          # mesh.ply (vertex colors + normals)
  s9_geo/           # dsm.tif, dtm.tif, ortho.tif (georeferenced GeoTIFFs)
  s10_export/       # model.obj/.glb/.ply/.fbx, points.las/.laz, + exports.json
  report/           # report.json, report.html
```

(There is **no S5** — live fusion is folded into S7.)

### 7.3 The report, and the honesty model

`report/report.html` summarizes every stage's metrics, confidence, timing, and degradation. Two things
to understand so you read it correctly:

- **Confidence is a measured 0..1 value per stage**, not a guess and not a progress bar. A stage that
  falls back to a lesser method still produces a real result and records a `degraded_reason`, and its
  confidence is **capped** at the ceiling of the method that actually ran.
- **Accuracy is UNVALIDATED unless you supplied ground check-points.** If you did, the report shows
  **measured** `rmse_z_m` / `mae_z_m` / `bias_z_m` sampled from the DSM. DRISHTI never fabricates an
  accuracy number.

Also expect **partial coverage on real footage** that lacks full overlap: the report's `dsm_coverage`
tells you what fraction of the area was reconstructed. Low overlap → patchy model. This is reported, not
hidden.

---

## 8. View and share the finished model

### 8.0 The whole run on one page (results dashboard)

For a single view of *everything the run produced*, the bundle root ships an **`index.html`** results page
and a **`view_all.cmd`** launcher:

- **Double-click `view_all.cmd`** (in `runs/<run_id>/`) — it serves the bundle root and opens `index.html`,
  which shows, on one page: the **KPIs** (runtime, stage pass/degrade counts, GPS-fit RMSE, dense-point and
  mesh counts, CRS), the **3D model** (the viewer below, embedded), the **orthophoto / DSM / DTM** maps
  (click to enlarge), the **per-stage table** read live from `report.json` (status, environment, time,
  measured confidence, degradation reason), and a **deliverables** list that links each file present on disk
  and flags the rest as living in the bundle zip.

It reads the real numbers from `report.json` at load time — nothing is hardcoded — so it stays truthful to
the run. (The DSM/DTM/ortho thumbnails are pre-rendered PNGs in `previews/`, since browsers can't display
GeoTIFF directly.)

### 8.1 The local web viewer (3D model only, in this repo, no upload)

The finished GLB ships next to a self-contained viewer. In the export folder
(`runs/<run_id>/s10_export/`):

- **Double-click `view_model.cmd`** — it starts a tiny local web server and opens `viewer.html` in your
  browser, which **auto-loads `model.glb`** and renders it (orbit/pan/zoom, vertex colors, wireframe,
  reset-view, up-axis and background toggles, and a load progress bar for the ~350 MB file).
- Or open `viewer.html` and **drag `model.glb`** straight onto the window (works even with no server).

*(Why the little server? Browsers block a page opened from disk (`file://`) from fetching a local `.glb`.
The launcher serves the folder over `http://localhost:8000`, which is allowed.)*

> **Why `model.glb` looks sharp (the precision story).** glTF stores vertex positions as **float32**. The
> pipeline mesh is georeferenced in **UTM** (northing ~4.57e6); left at that magnitude, float32 resolves
> only ~**0.5 m** steps, which bakes visible stair-stepping into the file. So the shipped `model.glb` is
> **recentered to a local origin** (its coords sit within ±220 m of zero → float32 step ~**0.015 mm**), and
> `model.glb.README.txt` records the exact UTM offset to add back. The other exports —
> `model.gltf`/`.obj`/`.ply`, `points.las`/`.laz`, `s9_geo/*.tif` — are left **georeferenced in UTM** and
> are the ones to use for measurement/GIS. If you ever regenerate the display `.glb` from the full-precision
> `s8_mesh/mesh.ply` (which is `double`), recenter it **before** the float32 cast, not after.

### 8.2 In Blender

Open `model.blend` (double-click) for an instant, pre-framed view, or run the import script on the raw
`model.gltf`. The mesh is in **UTM coordinates** (easting ~4.4e5, northing ~4.6e6), so any importer must
**recenter it to the origin** or it appears "invisible" from float jitter — the provided scripts and the
web viewer already do this.

### 8.3 Send it to someone else

`model.glb` is a **single self-contained file** — the right thing to share. Upload it to a file-transfer
service (e.g. WeTransfer) and send the link; the recipient drags it onto
**https://gltf-viewer.donmccurdy.com** in any browser — no install. (Or they use the `viewer.html` +
`view_model.cmd` from this repo locally.)

> The repo's built-in Cesium web viewer (`viewer/`) renders **3D Tiles** only. This run has no tileset
> (the 3D-Tiles converter wasn't on PATH), so that globe view is empty for this bundle — use the GLB
> viewer above instead.

---

## 9. Troubleshooting (real, seen-in-practice issues)

| Symptom | Cause & fix |
|---------|-------------|
| `config error` / exit code 2 | The descriptor or a `--set` failed validation. The config model **forbids unknown keys** — check for a typo in a key name or descriptor field. |
| A stage in the pre-flight warning says it will "fail loudly" | Its extras aren't installed **here**. Install them (`pip install -e ".[recon,geo,depth]"`) or run that stage on the T4. `drishti doctor` shows exactly what's missing. |
| `--profile max` raises at `s7_dense`/`s8_mesh` | `max` picks unimplemented methods. Add `--set dense.method=tsdf --set mesh.method=poisson` (§6.2). |
| Install aborts on Colab/Kaggle | Python 3.13 has no Open3D wheel. Use the notebooks (they build a 3.12 env with `uv`). |
| S3/S4 fail importing `transformers`, or S10 LAS fails | Those libs are in no extra: `pip install "transformers>=4.45" timm safetensors "laspy[lazrs]>=2.5"`. |
| Georeferencing RMSE is huge on a grid flight | Wrong matcher. Add `--set poses.matcher=exhaustive` (§4.3). |
| Model looks patchy / low `dsm_coverage` | Not enough overlap in the footage — reconstruction is honest about what it could see. Fly higher-overlap next time, or raise `ingest.target_fps`. |
| Telemetry seems misaligned with the video | Use `--set ingest.time_offset_s=<seconds>` to shift the track (§4.2). |
| A run died partway | The bundle is intact and resumable: fix the cause, `drishti resume runs/<run_id>`. The real error string is in the manifest (`inspect` shows it). |
| Weird characters in the console on Windows | Ensure your terminal codepage is UTF-8. |

---

## 10. Copy-paste cheat sheet

```bash
# --- one-time setup (local, no GPU) ---
python -m venv .venv && .venv\Scripts\Activate.ps1
pip install -e ".[video,poses,geo,recon,depth,telem]"
pip install "transformers>=4.45" timm safetensors "laspy[lazrs]>=2.5"
drishti doctor

# --- a real test dataset ---
python scripts/make_sample_dataset.py --dataset brighton_beach

# --- run end to end (grid flight → add exhaustive matcher) ---
drishti run -d configs/datasets/brighton_beach.yaml --profile balanced
drishti run -d configs/datasets/<grid>.yaml       --profile balanced --set poses.matcher=exhaustive

# --- quick trial (fewer frames, first stage only) ---
drishti run -d configs/datasets/<m>.yaml --upto s0_ingest --set ingest.target_fps=2 --set ingest.max_frames=200

# --- inspect / verify / resume ---
drishti inspect runs/<run_id>
drishti verify  runs/<run_id>
drishti resume  runs/<run_id>

# --- cloud T4 split: GPU stages there, finish locally ---
#   (on T4)   drishti run -d "$MISSION" --profile max --set dense.method=tsdf --set mesh.method=poisson --upto s7_dense
#   (locally) drishti resume runs/<run_id>

# --- view the result ---
start runs/<run_id>/report/report.html
#   then: double-click runs/<run_id>/s10_export/view_model.cmd  (opens the 3D viewer)
```

---

*For the full stage-by-stage internals, the config precedence rules, the compute-placement policy, and
the list of what's detected-but-not-yet-wired, see [`GUIDE.md`](GUIDE.md). For project status and the
decision log, see [`../AGENTS.md`](../AGENTS.md).*
