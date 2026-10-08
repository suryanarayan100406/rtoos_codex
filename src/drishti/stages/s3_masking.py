"""Offline dynamic exclusion and optional, explicitly mapped semantic inference."""

from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from numpy.typing import NDArray

from drishti.bundle import Bundle
from drishti.config.models import Config
from drishti.errors import DependencyError, InputError, ReconstructionError
from drishti.models.inference import LocalDetector, model_record, resolve_model
from drishti.models.semantics import ConfidenceMap, LabelMap, LocalSegmenter
from drishti.recon.depth import image_domain
from drishti.runtime import Runtime
from drishti.stages.base import StageResult

SEMANTIC_CONFIDENCE_METHOD = "winning_source_class_softmax_probability_uncalibrated"


def _semantic_descriptor(cfg: Config) -> dict[str, Any]:
    enabled = cfg.semantics.model is not None
    return {
        "status": "unavailable" if enabled else "disabled",
        "reason": "Semantic inference has not executed for this invocation" if enabled
        else "No semantic model configured",
        "classes": [item.model_dump(mode="json") for item in cfg.semantics.classes],
        "nodata": 0,
        "model": None,
        "source_labels": None,
        "confidence_method": SEMANTIC_CONFIDENCE_METHOD,
        "frames": [],
        "failures": [],
    }


def _excluded_semantics(
    labels: LabelMap,
    probability: ConfidenceMap,
    excluded: NDArray[np.bool_],
    cfg: Config,
) -> tuple[LabelMap, ConfidenceMap]:
    """Validate the inference boundary; never resize a misaligned categorical map."""
    if not isinstance(labels, np.ndarray) or labels.dtype != np.uint16 or labels.shape != excluded.shape:
        raise InputError("Semantic labels must be camera-shaped uint16")
    if (not isinstance(probability, np.ndarray) or probability.dtype != np.float32
            or probability.shape != excluded.shape):
        raise InputError("Semantic probabilities must be camera-shaped float32")
    if not set(np.unique(labels)).issubset({0, *(item.id for item in cfg.semantics.classes)}):
        raise InputError("Semantic inference emitted a class absent from the configured mapping")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise InputError("Semantic probabilities must be finite and in [0, 1]")
    labels, probability = labels.copy(), probability.copy()
    invalid = excluded | (labels == 0) | (probability.astype(np.float64) < cfg.semantics.min_probability)
    labels[invalid] = 0
    probability[invalid] = 0
    return labels, probability


def _write_semantic_maps(
    directory: Path, frame_id: str, labels: LabelMap, probability: ConfidenceMap,
) -> tuple[Path, Path]:
    """Publish a complete pair, removing both if either write/publication fails."""
    path = directory / f"{frame_id}.labels.npy"
    confidence_path = directory / f"{frame_id}.confidence.npy"
    try:
        with TemporaryDirectory(prefix=".semantic-", dir=directory) as temporary:
            staging = Path(temporary)
            np.save(staging / path.name, labels, allow_pickle=False)
            np.save(staging / confidence_path.name, probability, allow_pickle=False)
            (staging / path.name).replace(path)
            (staging / confidence_path.name).replace(confidence_path)
    except Exception:
        path.unlink(missing_ok=True)
        confidence_path.unlink(missing_ok=True)
        raise
    return path, confidence_path


def _clear_semantic_products(directory: Path) -> None:
    """Remove only generated maps and interrupted staging directories from S3."""
    if directory.is_symlink():
        raise InputError("Semantic output directory must not be a symlink")
    if directory.exists():
        for path in directory.glob("*.npy"):
            path.unlink()
        for path in directory.glob(".semantic-*"):
            if path.is_dir():
                rmtree(path)


def _run_semantics(
    bundle: Bundle, cfg: Config, runtime: Runtime, cameras: list[dict[str, Any]],
    masks: list[dict[str, Any]], descriptor: dict[str, Any], relative: str,
    directory: Path,
) -> None:
    """Optional inference cannot discard or interrupt the completed dynamic masks.

    ``frames`` contains only published pairs. ``failures`` has no artifact paths.
    Model provenance counts validated frame inference, independently of map IO.
    Each descriptor update is atomic and contains only this invocation's products.
    """
    from PIL import Image

    try:
        if cfg.semantics.model is None:
            return
        if not cameras:
            descriptor["reason"] = "No registered cameras available for semantic inference"
            return
        directory.mkdir(exist_ok=True)
        spec = resolve_model(cfg, cfg.semantics.model)
        segmenter = LocalSegmenter(spec, cfg.semantics.classes, runtime.device)
    except Exception as exc:
        descriptor.update(status="unavailable", reason=f"Semantic inference unavailable: {type(exc).__name__}: {exc}")
        bundle.write_json(relative, descriptor)
        return

    for camera, mask in zip(cameras, masks, strict=True):
        frame_id = camera["frame_id"]
        try:
            domain = image_domain(camera)
            with Image.open(bundle.path(camera["image_path"])) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(bundle.path(mask["path"])) as image:
                excluded = np.asarray(image.convert("L")) > 0
            shape = camera["height"], camera["width"]
            if rgb.shape[:2] != shape or excluded.shape != shape:
                raise InputError("Semantic image/mask dimensions disagree with final camera")
            labels, probability = segmenter.predict(rgb, cfg.semantics.min_probability)
            labels, probability = _excluded_semantics(labels, probability, excluded, cfg)
            if descriptor["model"] is None:
                descriptor["model"] = model_record(spec, "transformers", segmenter.device) | {"inference_frames": 0}
                descriptor["source_labels"] = dict(segmenter.source_labels)
            descriptor["model"]["inference_frames"] += 1
            path, confidence_path = _write_semantic_maps(directory, frame_id, labels, probability)
            descriptor["frames"].append({
                "frame_id": frame_id, "path": path.relative_to(bundle.root).as_posix(),
                "confidence_path": confidence_path.relative_to(bundle.root).as_posix(),
                "height": shape[0], "width": shape[1], "image_domain": domain,
                "labeled_pixels": int(np.count_nonzero(labels)),
            })
        except Exception as exc:
            descriptor["failures"].append({"frame_id": frame_id, "reason": f"{type(exc).__name__}: {exc}"})
        complete = len(descriptor["frames"]) + len(descriptor["failures"]) == len(cameras)
        descriptor["status"] = (
            "ok" if complete and not descriptor["failures"] else
            "partial" if descriptor["frames"] else "unavailable"
        )
        failures = "; ".join(f"{item['frame_id']}: {item['reason']}" for item in descriptor["failures"])
        descriptor["reason"] = failures or (None if complete else "Semantic inference is incomplete")
        bundle.write_json(relative, descriptor)


def motion_mask(current, other, cfg, seed=0):
    """Reject ego-motion using a RANSAC epipolar or planar model, then dense flow.

    A failed geometric fit returns no mask and an explicit unknown status. Zero
    residuals are only called static after a supported geometric fit actually ran.
    """
    import cv2

    if current.shape != other.shape:
        return None, {"status": "unknown", "reason": "Pair image dimensions differ"}
    cv2.setRNGSeed(int(seed))
    first = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    second = cv2.cvtColor(other, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(first, maxCorners=cfg.motion_features,
                                     qualityLevel=cfg.motion_quality, minDistance=cfg.motion_spacing_px)
    if corners is None or len(corners) < cfg.min_motion_matches:
        return None, {"status": "unknown", "reason": "Insufficient optical-flow features"}
    # Runtime accepts None for uninitialized output arrays; these two OpenCV
    # callable stubs incorrectly require an ndarray. Keep the workaround local.
    lk: Any = cv2.calcOpticalFlowPyrLK
    farneback: Any = cv2.calcOpticalFlowFarneback
    target, status, _ = lk(
        first, second, corners, None, winSize=(cfg.flow_window_px, cfg.flow_window_px),
        maxLevel=cfg.flow_levels
    )
    reverse, back_status, _ = lk(
        second, first, target, None, winSize=(cfg.flow_window_px, cfg.flow_window_px),
        maxLevel=cfg.flow_levels
    )
    good = status.ravel().astype(bool) & back_status.ravel().astype(bool)
    good &= np.linalg.norm(reverse.reshape(-1, 2) - corners.reshape(-1, 2), axis=1) <= cfg.flow_fb_threshold_px
    p, q = corners.reshape(-1, 2)[good], target.reshape(-1, 2)[good]
    if len(p) < cfg.min_motion_matches:
        return None, {"status": "unknown", "matches": len(p), "reason": "Insufficient reliable flow matches"}
    H, hi = cv2.findHomography(p, q, cv2.RANSAC, cfg.motion_threshold_px,
                              maxIters=cfg.motion_ransac_iterations, confidence=cfg.motion_ransac_confidence)
    F, fi = cv2.findFundamentalMat(p, q, cv2.FM_RANSAC, cfg.motion_threshold_px,
                                 cfg.motion_ransac_confidence, cfg.motion_ransac_iterations)
    candidates = []
    for name, model, inliers in (("homography", H, hi), ("epipolar", F, fi)):
        if model is not None and model.shape == (3, 3) and np.isfinite(model).all() and inliers is not None:
            support = int(np.count_nonzero(inliers))
            if support >= cfg.min_motion_matches and support / len(p) >= cfg.motion_min_inlier_fraction:
                candidates.append((support, name, model))
    if not candidates:
        return None, {"status": "unknown", "matches": len(p), "reason": "No supported ego-motion model"}
    # Prefer a well-supported planar model; otherwise use general epipolar geometry.
    homographies = [c for c in candidates if c[1] == "homography"]
    support, name, model = max(homographies or candidates, key=lambda c: c[0])
    flow = farneback(
        first, second, None, cfg.flow_pyr_scale, cfg.flow_levels, cfg.flow_window_px,
        cfg.flow_iterations, cfg.flow_poly_n, cfg.flow_poly_sigma, 0
    )
    backward = farneback(
        second, first, None, cfg.flow_pyr_scale, cfg.flow_levels, cfg.flow_window_px,
        cfg.flow_iterations, cfg.flow_poly_n, cfg.flow_poly_sigma, 0
    )
    y, x = np.indices(first.shape, dtype=np.float32)
    qx, qy = x + flow[..., 0], y + flow[..., 1]
    back = cv2.remap(backward, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    reliable = np.linalg.norm(flow + back, axis=2) <= cfg.flow_fb_threshold_px
    reliable &= (qx >= 0) & (qx < first.shape[1]-1) & (qy >= 0) & (qy < first.shape[0]-1)
    if name == "homography":
        denominator = model[2, 0]*x + model[2, 1]*y + model[2, 2]
        reliable &= np.abs(denominator) > np.finfo(float).eps
        with np.errstate(divide="ignore", invalid="ignore"):
            hx = (model[0, 0]*x + model[0, 1]*y + model[0, 2])/denominator
            hy = (model[1, 0]*x + model[1, 1]*y + model[1, 2])/denominator
        residual = np.hypot(qx-hx, qy-hy)
    else:
        lx = model[0, 0]*x + model[0, 1]*y + model[0, 2]
        ly = model[1, 0]*x + model[1, 1]*y + model[1, 2]
        lc = model[2, 0]*x + model[2, 1]*y + model[2, 2]
        norm = np.hypot(lx, ly)
        reliable &= norm > np.finfo(float).eps
        residual = np.abs(lx*qx + ly*qy + lc)/np.maximum(norm, np.finfo(float).eps)
    # Unreliable flow is unknown/excluded, rather than evidence for static geometry.
    mask = ~reliable | (residual > cfg.motion_threshold_px)
    return mask, {"status": "ok", "method": f"farneback_{name}_residual",
                  "matches": len(p), "inlier_matches": support,
                  "reliable_fraction": float(np.mean(reliable)),
                  "unknown_flow_fraction": float(np.mean(~reliable))}


def detection_mask(shape, detections, dynamic_classes, dilation_px):
    """Conservative clipped boxes, honestly distinct from segmentation refinement."""
    import cv2

    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    selected = []
    classes = {c.lower() for c in dynamic_classes}
    for item in detections:
        if item["class"] not in classes:
            continue
        box = np.asarray(item["box"], float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise InputError("Detector emitted an invalid box")
        x0, y0 = np.floor(box[:2]).astype(int)
        x1, y1 = np.ceil(box[2:]).astype(int)
        x0, x1 = np.clip([x0, x1], 0, w)
        y0, y1 = np.clip([y0, y1], 0, h)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 255
            selected.append(item)
    if dilation_px:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilation_px+1, 2*dilation_px+1))
        mask = cv2.dilate(mask, kernel).astype(np.uint8)
    return mask.astype(bool), selected


class S3Masking:
    name = "s3_masking"
    requires = ("s1_frameqa", "s6_global")
    config_sections = ("masking", "semantics", "models", "compute", "run")

    def run(self, bundle: Bundle, cfg: Config, runtime: Runtime) -> StageResult:
        from PIL import Image

        folder = bundle.stage_dir(self.name)
        masks_dir = folder / "masks"
        masks_dir.mkdir(exist_ok=True)
        semantic_relative = (folder / "semantics.json").relative_to(bundle.root).as_posix()
        semantics = _semantic_descriptor(cfg)
        # Reset descriptors before required-input/model work can fail, so a previous
        # invocation's successes never become this invocation's partial products.
        bundle.write_json(semantic_relative, semantics)
        semantic_maps = folder / "semantics"
        semantic_reset_ok = True
        try:
            _clear_semantic_products(semantic_maps)
        except Exception as exc:
            semantic_reset_ok = False
            semantics["reason"] = f"Semantic output cleanup failed: {type(exc).__name__}: {exc}"
            bundle.write_json(semantic_relative, semantics)
        relative = (folder / "masks.json").relative_to(bundle.root).as_posix()
        masking: dict[str, Any] = {"frames": [], "semantic_labels": semantic_relative,
                   "semantic_status": semantics["status"],
                   "mask_domain": "255=exclude; includes unknown optical-flow support"}
        bundle.write_json(relative, masking)
        for path in masks_dir.glob("*.png"):
            path.unlink()
        poses = bundle.read_json(bundle.artifact("s6_global", "poses").relative_to(bundle.root).as_posix())
        cameras: list[dict[str, Any]] = poses["cameras"]
        frame_ids = [camera["frame_id"] for camera in cameras]
        if (any(not isinstance(fid, str) or not fid.strip() or fid in {".", ".."}
                or "/" in fid or "\\" in fid for fid in frame_ids)
                or len(set(frame_ids)) != len(frame_ids)):
            raise InputError("Registered camera frame IDs must be unique safe filename components")
        detector: LocalDetector | None = None
        records: list[dict[str, Any]] = []
        reasons: list[str] = []
        detector_spec: dict[str, Any] | None = None
        if cfg.masking.model:
            try:
                spec = resolve_model(cfg, cfg.masking.model)
                detector = LocalDetector(spec, runtime.device)
                detector_spec = spec
            except (DependencyError, InputError, OSError, ValueError, RuntimeError) as exc:
                reasons.append(f"RT-DETR unavailable: {exc}")
        if detector is None and not cfg.masking.motion_fallback:
            raise ReconstructionError("No dynamic masking method is available; enable motion fallback or supply RT-DETR")
        frames: list[dict[str, Any]] = []
        inference_frames = 0
        for index, camera in enumerate(cameras):
            with Image.open(bundle.path(camera["image_path"])) as image:
                rgb = np.asarray(image.convert("RGB"))
            if rgb.shape[:2] != (camera["height"], camera["width"]):
                raise InputError("Camera dimensions disagree with observed image")
            mask, detections, methods = np.zeros(rgb.shape[:2], bool), [], []
            detector_ok = False
            if detector is not None:
                try:
                    mask, detections = detection_mask(rgb.shape[:2], detector.detect(rgb, cfg.masking.threshold),
                                                       cfg.masking.dynamic_classes, cfg.masking.dilation_px)
                    methods.append("rtdetr_conservative_dilated_boxes")
                    detector_ok = True
                    inference_frames += 1
                except (RuntimeError, ValueError) as exc:
                    reasons.append(f"{camera['frame_id']}: detector failed: {exc}")
            motion: dict[str, Any] = {"status": "not_requested"}
            if cfg.masking.motion_fallback and (not detector_ok or cfg.masking.motion_with_detector):
                if len(cameras) > 1:
                    neighbor = cameras[index-1 if index else 1]
                    with Image.open(bundle.path(neighbor["image_path"])) as image:
                        other = np.asarray(image.convert("RGB"))
                    flow_mask, motion = motion_mask(rgb, other, cfg.masking, cfg.run.seed)
                    if flow_mask is not None:
                        exclusion_fraction = float(np.mean(flow_mask))
                        if exclusion_fraction > cfg.masking.max_motion_exclusion_fraction:
                            motion = {**motion, "status": "unknown",
                                      "reason": (f"Motion exclusion fraction {exclusion_fraction:.3f} exceeds "
                                                  f"configured maximum {cfg.masking.max_motion_exclusion_fraction:.3f}")}
                            flow_mask = None
                        else:
                            mask |= flow_mask
                            methods.append(motion["method"])
                else:
                    motion = {"status": "unknown", "reason": "No neighboring registered frame"}
            unknown = not detector_ok and motion["status"] != "ok"
            if unknown:
                if cfg.masking.motion_unknown_policy == "retain":
                    mask[:] = False
                    methods.append("unknown_pixels_retained")
                    reasons.append(f"{camera['frame_id']}: masking unknown; pixels retained with unknown support")
                else:
                    mask[:] = True
                    methods.append("unknown_all_excluded")
                    reasons.append(f"{camera['frame_id']}: masking unknown; all pixels excluded")
            path = masks_dir / f"{camera['frame_id']}.png"
            Image.fromarray(mask.astype(np.uint8)*255).save(path)
            support_scores = [float(np.mean([d["score"] for d in detections]))] if detections else []
            if motion["status"] == "ok":
                support_scores.append(motion["inlier_matches"]/motion["matches"]*motion["reliable_fraction"])
            frames.append({"frame_id": camera["frame_id"], "path": path.relative_to(bundle.root).as_posix(),
                           "method": "+".join(methods), "dynamic_fraction": None if unknown else float(np.mean(mask)),
                           "exclusion_fraction": float(np.mean(mask)), "unknown": unknown,
                           "confidence": min(support_scores) if support_scores else None,
                            "confidence_method": "native_selected_box_score_and_geometric_fit_flow_support",
                            "boxes": detections, "motion": motion})
            masking["frames"] = frames
            bundle.write_json(relative, masking)
        if detector_spec is not None and inference_frames:
            records.append(model_record(detector_spec, "transformers", runtime.device)
                           | {"inference_frames": inference_frames})
        # Release the detection model before allocating the separate optional model.
        detector = None
        if semantic_reset_ok:
            _run_semantics(bundle, cfg, runtime, cameras, frames, semantics, semantic_relative, semantic_maps)
        bundle.write_json(semantic_relative, semantics)
        masking["semantic_status"] = semantics["status"]
        bundle.write_json(relative, masking)
        if semantics["model"] is not None:
            records.append(semantics["model"])
        outputs = {"masks": relative, "images": masks_dir.relative_to(bundle.root).as_posix(),
                   "semantics": semantic_relative}
        if semantics["frames"]:
            outputs["semantic_maps"] = semantic_maps.relative_to(bundle.root).as_posix()
        confidence = [f["confidence"] for f in frames if f["confidence"] is not None]
        return StageResult(outputs,
                            metrics={"frames": len(frames), "unknown_frames": sum(f["unknown"] for f in frames),
                                     "methods": sorted({f["method"] for f in frames}),
                                     "semantic_status": semantics["status"],
                                     "semantic_frames": len(semantics["frames"]),
                                     "semantic_failed_frames": len(semantics["failures"]),
                                     "semantic_inference_frames": semantics["model"]["inference_frames"]
                                     if semantics["model"] else 0},
                           confidence=float(np.mean(confidence)) if confidence else None,
                           degraded_reason="; ".join(dict.fromkeys(reasons)) or None,
                           models=records)


STAGE = S3Masking()
