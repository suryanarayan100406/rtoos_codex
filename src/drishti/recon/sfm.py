"""Actual CPU COLMAP execution and coherent undistorted sparse camera exports."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from drishti.errors import DependencyError, InputError, ReconstructionError


def load_intrinsics(path: Path, width: int, height: int) -> dict[str, Any]:
    """Validate a supplied COLMAP model/params or OpenCV K/distortion calibration."""
    import numpy as np

    try:
        if Path(path).suffix.lower() in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(Path(path).read_text())
        else:
            data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.get("width") != width or data.get("height") != height:
            raise ValueError("calibration width/height must exactly match decoded image size")
        if "params" in data:
            model = data["camera_model"]
            expected = {"SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4,
                        "RADIAL": 5, "OPENCV": 8, "FULL_OPENCV": 12, "OPENCV_FISHEYE": 8}
            params = np.asarray(data["params"], dtype=np.float64)
            if model not in expected or params.shape != (expected[model],):
                raise ValueError("unsupported COLMAP model or wrong parameter count")
            focal_count = 1 if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL") else 2
            if np.any(params[:focal_count] <= 0):
                raise ValueError("focal lengths must be positive")
        else:
            matrix = np.asarray(data["K"], dtype=np.float64)
            if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
                raise ValueError("K must be a finite 3x3 matrix")
            if not np.allclose(matrix[2], [0, 0, 1]) or matrix[0, 1] != 0 or matrix[1, 0] != 0:
                raise ValueError("K must have standard zero-skew OpenCV pinhole form")
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
                raise ValueError("focal lengths must be positive")
            distortion = np.asarray(data.get("distortion", []), dtype=np.float64)
            model = data.get("camera_model", "OPENCV" if len(distortion) == 4 else "FULL_OPENCV" if len(distortion) in (5, 8) else "PINHOLE")
            params_list = [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]
            if model == "PINHOLE" and len(distortion) == 0:
                pass
            elif model in ("OPENCV", "OPENCV_FISHEYE") and len(distortion) == 4:
                params_list.extend(distortion)
            elif model == "FULL_OPENCV" and len(distortion) in (5, 8):
                params_list.extend(np.pad(distortion, (0, 8 - len(distortion))))
            else:
                raise ValueError("K/distortion requires PINHOLE, OPENCV (4), FULL_OPENCV (5/8), or OPENCV_FISHEYE (4)")
            params = np.asarray(params_list, dtype=np.float64)
        if not np.isfinite(params).all():
            raise ValueError("all calibration parameters must be finite")
        return {"camera_model": model, "params": params.tolist(), "width": width, "height": height}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise InputError(f"Invalid camera intrinsics {path}: {exc}") from exc


def run_sfm(frames: list[dict[str, Any]], bundle_root: Path, stage_dir: Path,
            settings: Any, *, seed: int, intrinsics_path: Path | None = None) -> tuple[Any, Path, dict[str, Any]]:
    try:
        import pycolmap
    except ImportError as exc:
        raise DependencyError("Pose estimation requires pycolmap; no poses were estimated") from exc
    if len(frames) < settings.min_registered:
        raise ReconstructionError("Insufficient accepted keyframes for requested SfM registration minimum")
    shapes = {(frame["width"], frame["height"]) for frame in frames}
    if len(shapes) != 1:
        raise InputError("Video keyframes have mixed image sizes; supply one fixed-resolution recording")
    width, height = next(iter(shapes))
    calibration = load_intrinsics(intrinsics_path, width, height) if intrinsics_path else None
    glomap_executable = shutil.which("glomap") if settings.use_glomap else None
    if settings.use_glomap and glomap_executable is None:
        raise DependencyError("poses.use_glomap requested, but glomap executable is unavailable")
    colmap_dir = stage_dir / "colmap"
    # Execution-owned working products are reset on rerun, never reuse a stale DB.
    if colmap_dir.exists():
        shutil.rmtree(colmap_dir)
    images_dir = colmap_dir / "input"
    models_dir = colmap_dir / "models"
    images_dir.mkdir(parents=True)
    models_dir.mkdir()
    for frame in frames:
        shutil.copyfile(bundle_root / frame["path"], images_dir / f"{frame['id']}.png")
    database = colmap_dir / "database.db"
    pycolmap.set_random_seed(seed)
    reader = pycolmap.ImageReaderOptions()
    if calibration:
        reader.camera_params = ",".join(str(p) for p in calibration["params"])
    camera_model = calibration["camera_model"] if calibration else settings.camera_model
    reader.camera_model = camera_model
    extraction = pycolmap.SiftExtractionOptions()
    extraction.num_threads = 1
    matching = pycolmap.SiftMatchingOptions()
    matching.num_threads = 1
    mapping = pycolmap.IncrementalPipelineOptions()
    mapping.num_threads = 1
    mapping.mapper.num_threads = 1
    mapping.min_model_size = settings.min_registered
    if calibration:
        mapping.ba_refine_focal_length = False
        mapping.ba_refine_principal_point = False
        mapping.ba_refine_extra_params = False
        mapping.mapper.abs_pose_refine_focal_length = False
        mapping.mapper.abs_pose_refine_extra_params = False
    try:
        pycolmap.extract_features(str(database), str(images_dir),
                                  image_list=[f"{frame['id']}.png" for frame in frames],
                                  camera_mode=pycolmap.CameraMode.SINGLE, camera_model=camera_model,
                                  reader_options=reader, sift_options=extraction, device=pycolmap.Device.cpu)
        if settings.matcher == "sequential":
            sequential = pycolmap.SequentialMatchingOptions()
            sequential.overlap = settings.overlap
            sequential.quadratic_overlap = False
            sequential.loop_detection = False  # No offline vocabulary model was supplied.
            pycolmap.match_sequential(str(database), sift_options=matching,
                                      matching_options=sequential, device=pycolmap.Device.cpu)
        elif settings.matcher == "exhaustive":
            pycolmap.match_exhaustive(str(database), sift_options=matching, device=pycolmap.Device.cpu)
        else:
            raise InputError(f"Unsupported SfM matcher {settings.matcher}")
        if settings.use_glomap:
            assert glomap_executable is not None  # Validated before SfM execution.
            command = [glomap_executable, "mapper", "--database_path", str(database),
                       "--image_path", str(images_dir), "--output_path", str(models_dir)]
            process = subprocess.run(command, check=False, capture_output=True, text=True)
            (colmap_dir / "glomap.log").write_text(process.stdout + "\n" + process.stderr)
            if process.returncode:
                raise ReconstructionError(f"GLOMAP mapper failed (exit {process.returncode}); inspect colmap/glomap.log")
            model_paths = sorted(path for path in models_dir.iterdir()
                                 if path.is_dir() and ((path / "cameras.bin").exists() or (path / "cameras.txt").exists()))
            reconstructions = [pycolmap.Reconstruction(str(path)) for path in model_paths]
            method = "glomap_mapper_executed"
        else:
            results = pycolmap.incremental_mapping(str(database), str(images_dir), str(models_dir), options=mapping)
            reconstructions = list(results.values())
            method = "pycolmap_incremental_mapping"
        if not reconstructions:
            raise ReconstructionError("SfM produced no reconstruction; real database/matching products were retained")
        reconstruction = max(reconstructions, key=lambda model: model.num_reg_images())
        if reconstruction.num_reg_images() < settings.min_registered:
            raise ReconstructionError(f"SfM registered only {reconstruction.num_reg_images()} images")
        selected_dir = colmap_dir / "selected"
        selected_dir.mkdir()
        reconstruction.write(str(selected_dir))
        undistorted_dir = colmap_dir / "undistorted"
        pycolmap.undistort_images(str(undistorted_dir), str(selected_dir), str(images_dir), output_type="COLMAP")
        reconstruction = pycolmap.Reconstruction(str(undistorted_dir / "sparse"))
    except (InputError, DependencyError, ReconstructionError):
        raise
    except (RuntimeError, ValueError) as exc:
        raise ReconstructionError(f"COLMAP execution failed; retained working files at {colmap_dir}: {exc}") from exc
    metrics = {"method": method, "matcher": settings.matcher, "feature_backend": "CPU_SIFT",
               "registered_images": reconstruction.num_reg_images(), "input_images": len(frames),
               "models": len(reconstructions), "projection_domain": "undistorted_pinhole",
                "intrinsics": "provided_fixed" if calibration else "self_calibrated_COLMAP",
               "provided_calibration": calibration, "pycolmap_version": pycolmap.__version__,
               "options": {"reader": reader.todict(), "sift_extraction": extraction.todict(),
                           "sift_matching": matching.todict()}, "seed": seed}
    metrics["options"]["sift_extraction"]["normalization"] = str(extraction.normalization)
    return reconstruction, undistorted_dir, metrics


def extract_sparse(reconstruction: Any, frames: list[dict[str, Any]], undistorted_dir: Path,
                   bundle_root: Path, min_track_length: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    frame_by_name = {f"{frame['id']}.png": frame for frame in frames}
    cameras = []
    point_ids = sorted(point_id for point_id, point in reconstruction.points3D.items()
                       if point.track.length() >= min_track_length and np.isfinite(point.xyz).all()
                       and np.isfinite(point.error) and point.error >= 0)
    point_index = {point_id: index for index, point_id in enumerate(point_ids)}
    points = np.asarray([reconstruction.points3D[p].xyz for p in point_ids], dtype=np.float64).reshape(-1, 3)
    colors = np.asarray([reconstruction.points3D[p].color for p in point_ids], dtype=np.uint8).reshape(-1, 3)
    errors = np.asarray([reconstruction.points3D[p].error for p in point_ids], dtype=np.float64)
    support = np.asarray([reconstruction.points3D[p].track.length() for p in point_ids], dtype=np.int64)
    # Observable support/reprojection score, explicitly not a probability or accuracy.
    confidence = (1 / (1 + errors)) * (1 - 1 / support)
    for image in sorted(reconstruction.images.values(), key=lambda image: frame_by_name[image.name]["time_s"]):
        if not image.has_pose:
            continue
        frame = frame_by_name[image.name]
        camera = reconstruction.cameras[image.camera_id]
        if camera.model.name not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ReconstructionError("COLMAP undistortion did not produce pinhole images")
        observed = []
        for feature in image.points2D:
            if feature.has_point3D() and feature.point3D_id in point_index:
                observed.append({"point_index": point_index[feature.point3D_id], "uv": feature.xy.tolist()})
        cameras.append({"frame_id": frame["id"],
                        "image_path": (undistorted_dir / "images" / image.name).relative_to(bundle_root).as_posix(),
                        "source_image_path": frame["path"], "time_s": frame["time_s"],
                        "width": camera.width, "height": camera.height,
                        "K": camera.calibration_matrix().tolist(), "distortion": [],
                        "projection_domain": "undistorted_pinhole", "R_cw": image.cam_from_world.rotation.matrix().tolist(),
                        "center": np.asarray(image.projection_center()).reshape(3).tolist(),
                        "gps_sample": frame["gps"], "observations": observed, "anchors": []})
    if not len(points):
        raise ReconstructionError("SfM has no supported sparse points; inspect track lengths and registration")
    return cameras, {"points": points, "colors": colors, "confidence": confidence.astype(np.float64),
                     "inferred": np.zeros(len(points), dtype=bool), "reprojection_error_px": errors,
                     "track_length": support}


def recompute_anchors(cameras: list[dict[str, Any]], points: Any) -> dict[str, Any]:
    """Use measured feature positions and final metric camera-Z, with current residuals."""
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    count = 0
    residuals = []
    for camera in cameras:
        rotation = np.asarray(camera["R_cw"], dtype=np.float64)
        center = np.asarray(camera["center"], dtype=np.float64)
        matrix = np.asarray(camera["K"], dtype=np.float64)
        anchors = []
        for observation in camera.get("observations", []):
            point_index = int(observation["point_index"])
            if not 0 <= point_index < len(points):
                raise ReconstructionError("Camera observation points outside sparse array")
            point_camera = rotation @ (points[point_index] - center)
            if not np.isfinite(point_camera).all() or point_camera[2] <= 0:
                continue
            projected = matrix @ point_camera
            projected = projected[:2] / projected[2]
            uv = np.asarray(observation["uv"], dtype=np.float64)
            if not np.isfinite(uv).all() or not (0 <= uv[0] < camera["width"] and 0 <= uv[1] < camera["height"]):
                continue
            error = float(np.linalg.norm(projected - uv))
            anchors.append([float(uv[0]), float(uv[1]), float(point_camera[2]), error])
            residuals.append(error)
        camera["anchors"] = anchors
        count += len(anchors)
    return {"anchors": count, "anchor_reprojection_rmse_px": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
            "anchor_domain": "final_pose_metric_camera_z", "anchor_uv": "measured_undistorted_feature"}
