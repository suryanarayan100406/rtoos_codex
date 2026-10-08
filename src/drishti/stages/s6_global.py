"""Global pose graph and joint calibrated visual/GNSS BA before metric depth."""

import copy
from typing import Any

from drishti.recon.bundle_adjustment import optimize_bundle
from drishti.recon.sfm import recompute_anchors
from drishti.recon.spine import optimize_pose_graph, refine_sparse_points
from drishti.stages.base import StageResult
from drishti.stages.s2_poses import (
    _barometer_attempt,
    _barometer_summary,
    _transfer_barometer,
    _update_barometer_optional,
)


class S6Global:
    name = "s6_global"
    requires = ("s2_poses",)
    config_sections = ("spine", "bundle_adjustment", "barometer")

    def run(self, bundle: Any, cfg: Any, runtime: Any) -> StageResult:
        import numpy as np

        directory = bundle.stage_dir(self.name)
        poses = bundle.read_json(str(bundle.artifact("s2_poses", "poses").relative_to(bundle.root)))
        source_cameras = copy.deepcopy(poses["cameras"])
        inherited = copy.deepcopy(poses.get("barometer_fusion"))
        settings = getattr(cfg, "barometer", None)
        enabled = settings is not None and settings.enabled
        kwargs: dict[str, Any] = {"barometer_settings": settings} if enabled else {}
        cameras, optimization = optimize_pose_graph(poses["cameras"], cfg.spine,
                                                   global_refinement=True, **kwargs)
        _transfer_barometer(cameras, source_cameras, id_key="frame_id")
        with np.load(bundle.artifact("s2_poses", "sparse"), allow_pickle=False) as data:
            sparse = {key: data[key] for key in data.files}
        if optimization["status"] == "optimized":
            sparse, sparse_metrics = refine_sparse_points(cameras, sparse)
            poses["pose_revision"] = "s6_global_visual_GNSS_refined"
        else:
            sparse_metrics = {"sparse_refinement_method": "retained_input_sparse_points"}
            poses["pose_revision"] = "s6_retained_s2_poses"
            for camera in cameras:
                camera["global_refinement_status"] = optimization["status"]
        cameras, sparse, adjustment = optimize_bundle(
            cameras, sparse, cfg.bundle_adjustment, cfg.spine, **kwargs,
        )
        _transfer_barometer(cameras, source_cameras, id_key="frame_id")
        warnings = []
        if adjustment["status"] == "optimized":
            poses["pose_revision"] = "s6_joint_reprojection_GNSS_bundle_adjusted"
            sparse_metrics = {
                "sparse_refinement_method": adjustment["method"],
                "jointly_optimized_points": adjustment["optimized_points"],
                "retained_sparse_points": adjustment["retained_points"],
                "point_confidence_method": adjustment["point_confidence_method"],
            }
        else:
            if optimization["status"] != "optimized":
                warnings.append(optimization["reason"])
            if cfg.bundle_adjustment.enabled and cfg.spine.optimize:
                warnings.append(adjustment["reason"])
        anchor_metrics = recompute_anchors(cameras, sparse["points"])
        poses["cameras"] = cameras
        poses["sparse_path"] = f"{self.name}/sparse.npz"
        poses["global_optimization"] = optimization
        poses["bundle_adjustment"] = adjustment
        graph_attempt = _barometer_attempt(optimization, enabled=enabled)
        ba_attempt = _barometer_attempt(adjustment, enabled=enabled)
        if adjustment["status"] == "optimized":
            selected, final_source = ba_attempt, "bundle_adjustment"
        elif optimization["status"] == "optimized":
            selected, final_source = graph_attempt, "pose_graph"
        else:
            selected = inherited if isinstance(inherited, dict) else {
                "used": False, "method": None, "barometer": None,
                "reason": "Retained S2 poses have no recorded barometer fusion provenance",
            }
            final_source = "retained_s2"
        fusion = _barometer_summary(self.name, selected, final_source,
                                   {"pose_graph": graph_attempt, "bundle_adjustment": ba_attempt},
                                   enabled=enabled, inherited=inherited)
        poses["barometer_fusion"] = fusion
        optional = poses.setdefault("optional_inputs", {})
        _update_barometer_optional(optional, fusion)
        np.savez_compressed(directory / "sparse.npz", **sparse)
        poses_path, optimization_path = f"{self.name}/poses.json", f"{self.name}/optimization.json"
        bundle.write_json(poses_path, poses)
        bundle.write_json(optimization_path, {**optimization, **anchor_metrics, **sparse_metrics,
                                              "bundle_adjustment": adjustment,
                                              "barometer_fusion": fusion, "optional_inputs": optional,
                                              "final_pose_revision": poses["pose_revision"]})
        return StageResult(outputs={"poses": poses_path, "optimization": optimization_path,
                                    "sparse": poses["sparse_path"]},
                            metrics={**optimization, **anchor_metrics, **sparse_metrics,
                                     "bundle_adjustment": adjustment,
                                     "barometer_fusion": fusion, "optional_inputs": optional},
                            confidence=None,
                            degraded_reason="; ".join(warnings) or None)


STAGE = S6Global()
