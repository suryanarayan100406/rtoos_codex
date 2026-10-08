"""Masked, confidence-filtered, metric-only, out-of-core point fusion."""

import tempfile

import numpy as np

from drishti.errors import InputError
from drishti.recon.depth import image_domain
from drishti.recon.fusion import TiledFusion, depth_observations
from drishti.stages.base import StageResult


class S7Dense:
    name = "s7_dense"
    requires = ("s4_depth", "s6_global", "s3_masking")
    config_sections = ("dense",)

    def run(self, bundle, cfg, runtime):
        from PIL import Image

        if cfg.dense.method != "point_fusion":
            raise InputError(f"Dense method {cfg.dense.method!r} is not implemented")
        folder = bundle.stage_dir(self.name)
        poses = bundle.read_json(bundle.artifact("s6_global", "poses").relative_to(bundle.root).as_posix())
        depth_index = bundle.read_json(bundle.artifact("s4_depth", "depth").relative_to(bundle.root).as_posix())
        masks_index = bundle.read_json(bundle.artifact("s3_masking", "masks").relative_to(bundle.root).as_posix())
        cameras = {c["frame_id"]: c for c in poses["cameras"]}
        masks = {f["frame_id"]: f for f in masks_index["frames"]}
        skipped, fused_frames, processed_metric_frames = [], 0, 0
        if depth_index.get("domain") != "camera_z_metres":
            raise InputError("Dense fusion requires depth in camera-Z metres")
        with tempfile.TemporaryDirectory(prefix="fusion-", dir=folder) as work:
            fusion = TiledFusion(work, cfg.dense.voxel_m, cfg.dense.tile_m, cfg.dense.max_points_per_chunk)
            for frame in depth_index["frames"]:
                fid = frame["frame_id"]
                if frame.get("metric") is not True:
                    skipped.append({"frame_id": fid, "reason": "unaligned/non-metric depth excluded"})
                    continue
                if fid not in cameras or fid not in masks:
                    raise InputError(f"Missing camera or mask for depth frame {fid}")
                camera = cameras[fid]
                if frame.get("image_domain", "distorted") != image_domain(camera):
                    raise InputError(f"Depth/image distortion domain mismatch for {fid}")
                depth = np.load(bundle.root/frame["path"], mmap_mode="r", allow_pickle=False)
                confidence = np.load(bundle.root/frame["confidence_path"], mmap_mode="r", allow_pickle=False)
                rgb = np.asarray(Image.open(bundle.root/camera["image_path"]).convert("RGB"))
                excluded = np.asarray(Image.open(bundle.root/masks[fid]["path"]).convert("L")) > 0
                before = fusion.input_count
                for observations in depth_observations(depth, confidence, rgb, excluded, camera, cfg.dense):
                    fusion.add(observations)
                processed_metric_frames += 1
                if fusion.input_count > before:
                    fused_frames += 1
                else:
                    skipped.append({"frame_id": fid,
                                    "reason": "no metric samples survived masking/confidence/normal filters"})
            arrays, metadata = fusion.finish(folder)
        metadata.update(frame="projected_local", origin=poses["origin"], crs=poses.get("crs"),
                        fused_frames=fused_frames, processed_metric_frames=processed_metric_frames,
                        skipped_frames=skipped, pose_revision=poses.get("pose_revision"),
                        confidence_method="weighted_mean_of_depth_anchor_support_scores",
                        arrays={name: (folder/"arrays"/f"{name}.npy").relative_to(bundle.root).as_posix()
                                for name in arrays},
                        inferred_method="depth observations are not hole-completion geometry")
        relative = (folder/"dense.json").relative_to(bundle.root).as_posix()
        bundle.write_json(relative, metadata)
        count = metadata["points"]
        # Chunked reduction also avoids faulting the entire mmap into memory at once.
        confidence_sum = sum(float(np.sum(arrays["confidence"][i:i+cfg.dense.max_points_per_chunk]))
                             for i in range(0, count, cfg.dense.max_points_per_chunk))
        reason = None
        if not count:
            reason = "No supported metric points; preserved empty measured cloud and depth failure metadata"
        elif skipped:
            reason = f"Excluded {len(skipped)} depth frames without accepted metric support"
        return StageResult({"cloud": (folder/"dense.npz").relative_to(bundle.root).as_posix(),
                            "ply": (folder/"dense.ply").relative_to(bundle.root).as_posix(),
                            "metadata": relative, "arrays": (folder/"arrays").relative_to(bundle.root).as_posix()},
                           metrics={k: v for k, v in metadata.items() if k not in {"arrays", "skipped_frames", "crs"}},
                           confidence=confidence_sum/count if count else None, degraded_reason=reason)


STAGE = S7Dense()
