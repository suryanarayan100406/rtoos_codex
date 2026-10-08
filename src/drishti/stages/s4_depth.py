"""Local inverse-relative inference aligned with sparse points and final S6 poses."""

from pathlib import Path
from typing import Any

import numpy as np

from drishti.errors import DependencyError, InputError, ReconstructionError
from drishti.models.inference import LocalDepth, model_record, resolve_model
from drishti.recon.depth import align_inverse_depth, image_domain, sparse_anchors
from drishti.stages.base import StageResult


def pose_sparse_source(bundle, poses):
    """Select the sparse revision belonging to the final cameras, failing closed.

    Old bundles without a sparse revision can retain the original S2 fallback.
    A missing declared/recorded S6 revision must never silently use stale points.
    """
    if "sparse_path" in poses:
        relative = poses["sparse_path"]
        if not isinstance(relative, str) or not relative:
            raise InputError("Final poses must declare a valid bundle-relative sparse_path")
        path = (bundle.root / relative).resolve()
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.is_relative_to(bundle.root):
            raise InputError("Final-pose sparse_path must be confined to the bundle")
        if not path.is_file():
            raise InputError(f"Final-pose sparse revision is missing: {relative}")
        return path, "final_pose_declared_sparse", None
    try:
        return bundle.artifact("s6_global", "sparse"), "s6_recorded_sparse", None
    except InputError:
        recorded = bundle.manifest.get("stages", {}).get("s6_global", {}).get("outputs", {})
        if "sparse" in recorded:
            raise
    if poses.get("pose_revision") in {
        "s6_global_visual_GNSS_refined", "s6_joint_reprojection_GNSS_bundle_adjusted",
    }:
        raise InputError("Refined S6 poses require their matching sparse revision; rerun S6")
    return bundle.artifact("s2_poses", "sparse"), "legacy_s2_sparse_fallback", (
        "Legacy poses have no final sparse revision; using the S2 sparse fallback"
    )


class S4Depth:
    name = "s4_depth"
    requires = ("s2_poses", "s6_global", "s3_masking")
    config_sections = ("depth", "models", "compute")

    def run(self, bundle, cfg, runtime):
        from PIL import Image

        folder = bundle.stage_dir(self.name)
        maps = folder / "maps"
        maps.mkdir(exist_ok=True)
        poses_path = bundle.artifact("s6_global", "poses")
        poses = bundle.read_json(poses_path.relative_to(bundle.root).as_posix())
        sparse_path, sparse_source, sparse_warning = pose_sparse_source(bundle, poses)
        masking = bundle.read_json(bundle.artifact("s3_masking", "masks").relative_to(bundle.root).as_posix())
        masks = {f["frame_id"]: f for f in masking["frames"]}
        with np.load(sparse_path, allow_pickle=False) as data:
            points = np.asarray(data["points"], np.float64)
            sparse_confidence = np.array(data["confidence"], copy=True) if "confidence" in data else np.ones(len(points))
            inferred = data["inferred"] if "inferred" in data else np.zeros(len(points), bool)
            sparse_confidence[inferred] = 0
        reasons: list[str] = []
        records: list[dict[str, Any]] = []
        model = None
        model_spec = None
        if sparse_warning:
            reasons.append(sparse_warning)
        model_failure = None
        try:
            model_spec = resolve_model(cfg, cfg.depth.model)
            model = LocalDepth(model_spec, cfg.depth.backend, runtime.device, cfg.compute.openvino_device,
                               cfg.depth.input_long_side)
        except (DependencyError, InputError, OSError, ValueError, RuntimeError) as exc:
            model_failure = f"Depth inference unavailable: {exc}"
            reasons.append(model_failure)
        frames, total_confidence, total_valid = [], 0., 0
        inference_frames = 0
        for camera in poses["cameras"]:
            fid = camera["frame_id"]
            if fid not in masks:
                raise InputError(f"Missing exclusion mask for registered frame {fid}")
            rgb = np.asarray(Image.open(bundle.root / camera["image_path"]).convert("RGB"))
            excluded = np.asarray(Image.open(bundle.root / masks[fid]["path"]).convert("L")) > 0
            if excluded.shape != rgb.shape[:2] or excluded.shape != (camera["height"], camera["width"]):
                raise InputError(f"Image/depth/mask domain mismatch for {fid}")
            depth, confidence = np.zeros(excluded.shape, np.float32), np.zeros(excluded.shape, np.float32)
            entry = {"frame_id": fid, "metric": False, "anchor_count": 0, "rmse_m": None,
                     "pose_source": poses_path.relative_to(bundle.root).as_posix(),
                     "pose_revision": poses.get("pose_revision"),
                     "sparse_path": sparse_path.relative_to(bundle.root).as_posix(),
                     "sparse_source": sparse_source,
                     "image_domain": image_domain(camera),
                     "alignment": "inverse_metric_z = a * relative_inverse_prediction + b"}
            try:
                if model is None:
                    raise ReconstructionError(model_failure)
                if np.all(excluded):
                    raise ReconstructionError("All pixels excluded by masking")
                anchors = sparse_anchors(points, camera, excluded, sparse_confidence, cfg.depth.max_anchors,
                                         cfg.depth.max_anchor_reprojection_error_px)
                entry["anchor_count"] = len(anchors)
                prediction = model.predict(rgb)
                inference_frames += 1
                if not records:
                    records.append(model_record(model_spec, cfg.depth.backend, model.device)
                                   | {"inference_frames": 0})
                records[0]["inference_frames"] = inference_frames
                fit = align_inverse_depth(
                    prediction, anchors, excluded, min_anchors=cfg.depth.min_anchors,
                    robust_delta=cfg.depth.robust_delta, max_relative_residual=cfg.depth.max_relative_residual,
                    min_depth_m=cfg.depth.min_depth_m, max_depth_m=cfg.depth.max_depth_m,
                    support_radius_px=cfg.depth.support_radius_px, robust_iterations=cfg.depth.robust_iterations,
                    min_inlier_fraction=cfg.depth.min_inlier_fraction
                )
                depth, confidence = fit.depth, fit.confidence
                entry.update(metric=True, anchor_count=fit.anchor_count, inlier_count=fit.inlier_count,
                             rmse_m=fit.rmse_m, relative_rmse=fit.relative_rmse, a=fit.a, b=fit.b,
                             confidence_method="accepted_anchor_residual_and_pixel_distance_support",
                             anchor_method=f"{sparse_source}_tracks_reprojected_with_pixel_z_buffer"
                             if "observations" in camera else f"{sparse_source}_projection_with_pixel_z_buffer")
                valid = depth > 0
                total_confidence += float(np.sum(confidence[valid]))
                total_valid += int(np.count_nonzero(valid))
            except (ReconstructionError, RuntimeError, ValueError) as exc:
                entry["reason"] = str(exc)
                reasons.append(f"{fid}: {exc}")
            path, conf_path = maps / f"{fid}.npy", maps / f"{fid}_confidence.npy"
            np.save(path, depth, allow_pickle=False)
            np.save(conf_path, confidence, allow_pickle=False)
            entry.update(path=path.relative_to(bundle.root).as_posix(),
                         confidence_path=conf_path.relative_to(bundle.root).as_posix())
            frames.append(entry)
        relative = (folder / "depth.json").relative_to(bundle.root).as_posix()
        bundle.write_json(relative, {"frames": frames, "domain": "camera_z_metres",
                                     "pose_revision": poses.get("pose_revision"),
                                     "sparse_path": sparse_path.relative_to(bundle.root).as_posix(),
                                     "sparse_source": sparse_source,
                                     "relative_predictions_fused": False,
                                     "residual_is_independent_accuracy": False})
        if records:
            records[0]["inference_frames"] = inference_frames
        return StageResult({"depth": relative, "maps": maps.relative_to(bundle.root).as_posix()},
                           metrics={"metric_frames": sum(f["metric"] for f in frames), "frames": len(frames),
                                    "valid_pixels": total_valid},
                           confidence=total_confidence/total_valid if total_valid else None,
                           degraded_reason="; ".join(dict.fromkeys(reasons)) or None, models=records)


STAGE = S4Depth()
