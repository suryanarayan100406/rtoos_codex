"""SfM -> robust camera-centre Sim3 -> real visual+GNSS metric pose graph."""

import copy
from typing import Any

from drishti.errors import InputError, ReconstructionError
from drishti.geo.alignment import robust_similarity
from drishti.geo.crs import project_gps
from drishti.recon.sfm import extract_sparse, load_intrinsics, recompute_anchors, run_sfm
from drishti.recon.spine import optimize_pose_graph, refine_sparse_points
from drishti.stages.base import StageResult


def _transfer_barometer(cameras: list[dict[str, Any]], sources: list[dict[str, Any]],
                        *, id_key: str) -> None:
    """Copy recorded sensor provenance by identity, never camera/selection order."""
    by_id = {source[id_key]: source for source in sources}
    for camera in cameras:
        source = by_id.get(camera["frame_id"], {})
        for field in ("barometer", "barometer_reason"):
            if field in source:
                camera[field] = copy.deepcopy(source[field])


def _barometer_attempt(metadata: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    """Separate solver attempts from accepted factor use, including legacy metadata."""
    diagnostics = metadata.get("barometer")
    diagnostics = copy.deepcopy(diagnostics) if isinstance(diagnostics, dict) else None
    factors = diagnostics.get("factors_added", 0) if diagnostics else 0
    used = (metadata.get("status") == "optimized" and metadata.get("baro_used") is True
            and isinstance(factors, int) and factors > 0)
    reason = diagnostics.get("reason") if diagnostics else None
    if not used and reason is None:
        reason = ("barometer.enabled=false" if not enabled else
                  metadata.get("reason") or "Solver did not record accepted barometer factors")
    return {"status": metadata.get("status"), "method": metadata.get("method"),
            "optimizer_invoked": metadata.get("optimizer_invoked", False),
            "used": used, "reason": reason, "barometer": diagnostics}


def _barometer_summary(stage: str, selected: dict[str, Any], final_source: str,
                       attempts: dict[str, Any], *, enabled: bool,
                       inherited: dict[str, Any] | None = None) -> dict[str, Any]:
    """Report-consumable final-geometry provenance; stage freshness is added by reports.

    `used` describes the selected final source only. Precursor graph use remains in
    `attempts`, and retained S2 diagnostics in `inherited`, even when accepted BA
    changes the poses without barometer factors. No fit/model is manufactured for
    legacy metadata or input detection.
    """
    used = selected.get("used") is True
    summary = {"stage": stage, "method": selected.get("method"),
               "status": "used" if used else ("not_used" if enabled else "disabled"),
               "used": used, "reason": selected.get("reason"), "final_source": final_source,
               "barometer": copy.deepcopy(selected.get("barometer")), "attempts": attempts}
    if inherited is not None:
        summary["inherited"] = copy.deepcopy(inherited)
    return summary


def _update_barometer_optional(optional: dict[str, Any], summary: dict[str, Any]) -> None:
    """Keep input availability/alignment distinct from final factor use."""
    entry = copy.deepcopy(optional.get("barometer", optional.get("baro", {})))
    # Preserve the original input ledger when updating an S2 ledger again in S6.
    if "status" in entry:
        entry.setdefault("input_status", entry["status"])
    if "reason" in entry:
        entry.setdefault("input_reason", entry["reason"])
    entry.update(used=summary["used"], status=summary["status"], reason=summary["reason"],
                 stage=summary["stage"], final_source=summary["final_source"])
    optional["barometer"] = entry
    if "baro" in optional:
        optional["baro"] = copy.deepcopy(entry)


class S2Poses:
    name = "s2_poses"
    requires = ("s0_ingest", "s1_frameqa")
    config_sections = ("dataset", "poses", "spine", "crs", "run", "barometer")

    def run(self, bundle: Any, cfg: Any, runtime: Any) -> StageResult:
        import numpy as np

        directory = bundle.stage_dir(self.name)
        keyframes = bundle.read_json(str(bundle.artifact("s1_frameqa", "keyframes").relative_to(bundle.root)))
        frames = keyframes["frames"]
        requested_intrinsics = cfg.dataset.optional.intrinsics
        intrinsics_path = None
        intrinsics_status: dict[str, Any] = {
            "provided": requested_intrinsics is not None, "available": False, "used": False,
            "status": "absent" if requested_intrinsics is None else "provided_file_missing",
            "reason": None if requested_intrinsics is None else f"optional calibration file is missing: {requested_intrinsics}",
        }
        warnings: list[str] = []
        if requested_intrinsics is not None and requested_intrinsics.is_file():
            try:
                load_intrinsics(requested_intrinsics, frames[0]["width"], frames[0]["height"])
                intrinsics_path = requested_intrinsics
                intrinsics_status.update(available=True, status="validated_not_consumed", reason=None)
            except (InputError, OSError, ValueError) as exc:
                intrinsics_status.update(available=True, status="invalid_not_consumed", reason=str(exc))
        elif requested_intrinsics is not None:
            warnings.append(intrinsics_status["reason"])
        if intrinsics_status["reason"] and intrinsics_status["status"] == "invalid_not_consumed":
            warnings.append(intrinsics_status["reason"])
        try:
            reconstruction, undistorted_dir, sfm_metrics = run_sfm(
                frames, bundle.root, directory, cfg.poses, seed=cfg.run.seed, intrinsics_path=intrinsics_path)
        except (InputError, ReconstructionError) as exc:
            if intrinsics_path is None:
                raise
            warnings.append(f"Optional intrinsics were not usable by SfM; self-calibration fallback: {exc}")
            intrinsics_status.update(status="self_calibrated_fallback", used=False, reason=str(exc))
            reconstruction, undistorted_dir, sfm_metrics = run_sfm(
                frames, bundle.root, directory, cfg.poses, seed=cfg.run.seed, intrinsics_path=None)
        else:
            if intrinsics_path is not None:
                intrinsics_status.update(used=True, status="consumed_by_sfm",
                                         reason=None, sfm_method=sfm_metrics.get("intrinsics"))
        cameras, sparse = extract_sparse(reconstruction, frames, undistorted_dir, bundle.root,
                                         cfg.poses.min_track_length)
        _transfer_barometer(cameras, frames, id_key="id")
        crs = copy.deepcopy(bundle.manifest["crs"])
        gps, origin = project_gps([camera["gps_sample"] for camera in cameras], crs["epsg"],
                                  settings=cfg.crs, source_datum=crs["source_altitude_datum"])
        transform, alignment = robust_similarity([camera["center"] for camera in cameras], gps,
                                                  cfg.spine, seed=cfg.run.seed)
        for index, camera in enumerate(cameras):
            camera["center"] = transform.apply(camera["center"]).tolist()
            camera["R_cw"] = (np.asarray(camera["R_cw"], dtype=np.float64) @ transform.rotation.T).tolist()
            camera["gps"] = gps[index].tolist()
            camera["gps_inlier"] = alignment["inlier_mask"][index]
            camera["visual_reference"] = {"R_cw": camera["R_cw"], "center": camera["center"]}
            camera["pose_method"] = "robust_Sim3_camera_centre_alignment"
        sparse["points"] = transform.apply(sparse["points"]).astype(np.float64)
        settings = getattr(cfg, "barometer", None)
        enabled = settings is not None and settings.enabled
        kwargs: dict[str, Any] = {"barometer_settings": settings} if enabled else {}
        source_cameras = copy.deepcopy(cameras)
        cameras, optimization = optimize_pose_graph(cameras, cfg.spine, **kwargs)
        _transfer_barometer(cameras, source_cameras, id_key="frame_id")
        if optimization["status"] == "optimized":
            sparse, sparse_metrics = refine_sparse_points(cameras, sparse)
        else:
            sparse_metrics = {"sparse_refinement_method": "retained_Sim3_sparse_points"}
        anchor_metrics = recompute_anchors(cameras, sparse["points"])
        poses_path, sparse_path, spine_path = (f"{self.name}/{name}" for name in ("poses.json", "sparse.npz", "spine.json"))
        np.savez_compressed(bundle.root / sparse_path, **sparse)
        optional = copy.deepcopy(keyframes.get("capabilities", {}))
        optional.setdefault("intrinsics", {}).update(intrinsics_status)
        attempt = _barometer_attempt(optimization, enabled=enabled)
        fusion = _barometer_summary(self.name, attempt, "pose_graph",
                                   {"pose_graph": attempt, "bundle_adjustment": None}, enabled=enabled)
        _update_barometer_optional(optional, fusion)
        poses = {"frame": "projected_local", "origin": origin.tolist(), "crs": crs,
                 "cameras": cameras, "pose_revision": "s2_metric_spine", "sparse_path": sparse_path,
                 "optional_inputs": optional, "barometer_fusion": fusion,
                 "confidence_method": "sparse_track_support_and_reprojection_score"}
        bundle.write_json(poses_path, poses)
        spine = {"alignment": alignment, "optimization": optimization, "sfm": sfm_metrics,
                 "sparse": sparse_metrics, "anchors": anchor_metrics, "optional_inputs": optional,
                 "barometer_fusion": fusion,
                 "accuracy": "UNVALIDATED", "gnss_fit_is_independent_accuracy": False}
        bundle.write_json(spine_path, spine)
        warnings = warnings + list(alignment["warnings"])
        if optimization["status"] != "optimized":
            warnings.append(optimization["reason"])
        if len(cameras) != len(frames):
            warnings.append(f"SfM registered {len(cameras)} of {len(frames)} keyframes")
        if crs["vertical"] == "unknown":
            warnings.append("altitude datum unknown; vertical accuracy unvalidated")
        return StageResult(outputs={"poses": poses_path, "sparse": sparse_path, "spine": spine_path,
                                    "reconstruction": f"{self.name}/colmap"},
                           metrics={"sfm": sfm_metrics, "alignment": alignment, "optimization": optimization,
                                    "barometer_fusion": fusion, "optional_inputs": optional,
                                    "sparse_points": len(sparse["points"]), "anchors": anchor_metrics,
                                    "unregistered_frame_ids": sorted(set(f["id"] for f in frames) - set(c["frame_id"] for c in cameras))},
                           confidence=float(np.mean(sparse["confidence"])) if len(sparse["confidence"]) else None,
                           degraded_reason="; ".join(warnings) or None)


STAGE = S2Poses()
