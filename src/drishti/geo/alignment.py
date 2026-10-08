"""Proper-rotation Umeyama and seeded RANSAC on camera centres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from drishti.errors import ReconstructionError


@dataclass
class Similarity:
    scale: float
    rotation: Any
    translation: Any

    def apply(self, points: Any) -> Any:
        import numpy as np

        return self.scale * (np.asarray(points, dtype=np.float64) @ self.rotation.T) + self.translation

    def as_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "rotation": self.rotation.tolist(),
                "translation": self.translation.tolist(), "mapping": "metric = scale * R * visual + t"}


def _paired(source: Any, target: Any) -> tuple[Any, Any]:
    import numpy as np

    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,) or target.shape != source.shape or len(source) < 3:
        raise ReconstructionError("Similarity requires at least three paired XYZ camera centres")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ReconstructionError("Similarity camera centres must be finite")
    return source, target


def umeyama(source: Any, target: Any) -> Similarity:
    import numpy as np

    source, target = _paired(source, target)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    x, y = source - source_mean, target - target_mean
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ReconstructionError("Collinear/stationary camera track leaves rotation unobservable; Sim3 rejected")
    variance = np.sum(x * x) / len(x)
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    correction = np.ones(3)
    correction[-1] = np.linalg.det(u @ vt)
    rotation = (u * correction) @ vt
    scale = float(np.dot(singular, correction) / variance)
    if not np.isfinite(scale) or scale <= 0:
        raise ReconstructionError("Similarity scale is invalid")
    return Similarity(scale, rotation, target_mean - scale * rotation @ source_mean)


def robust_similarity(source: Any, target: Any, settings: Any, *, seed: int) -> tuple[Similarity, dict[str, Any]]:
    import numpy as np

    source, target = _paired(source, target)
    baseline = float(np.max(np.linalg.norm(target[:, None] - target[None, :], axis=2)))
    if baseline < settings.min_baseline_m:
        raise ReconstructionError(f"GNSS baseline {baseline:.6g} m is insufficient to establish scale")
    rng = np.random.default_rng(seed)
    best = None
    best_score = (-1, float("-inf"))
    for _ in range(settings.ransac_iterations):
        indices = rng.choice(len(source), 3, replace=False)
        try:
            candidate = umeyama(source[indices], target[indices])
        except ReconstructionError:
            continue
        residuals = np.linalg.norm(candidate.apply(source) - target, axis=1)
        inliers = residuals <= settings.alignment_threshold_m
        score = (int(inliers.sum()), -float(np.median(residuals[inliers])) if inliers.any() else float("-inf"))
        if score > best_score:
            best, best_score = inliers, score
    if best is None or best.sum() < 3:
        raise ReconstructionError("RANSAC could not find three non-collinear visual/GNSS correspondences")
    transform = umeyama(source[best], target[best])
    residuals = np.linalg.norm(transform.apply(source) - target, axis=1)
    inliers = residuals <= settings.alignment_threshold_m
    if inliers.sum() < 3:
        raise ReconstructionError("Similarity refit lost sufficient GNSS support")
    transform = umeyama(source[inliers], target[inliers])
    residuals = np.linalg.norm(transform.apply(source) - target, axis=1)
    final_inliers = residuals <= settings.alignment_threshold_m
    if final_inliers.sum() < 3:
        raise ReconstructionError("Final similarity has fewer than three inlier correspondences")
    conditioning = {}
    warnings = []
    for label, points in (("visual", source[final_inliers]), ("gnss", target[final_inliers])):
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        ratios = singular / singular[0]
        conditioning[label] = {"singular_values": singular.tolist(), "rank_ratios": ratios.tolist(),
                               "rank": int(np.linalg.matrix_rank(points - points.mean(axis=0)))}
        if ratios[1] < settings.min_rank_ratio:
            warnings.append(f"{label}_track_nearly_collinear_rotation_weak")
        if ratios[2] < settings.min_rank_ratio:
            warnings.append(f"{label}_track_low_3d_rank_vertical_geometry_weak")
    metrics = {"method": "seeded_RANSAC_Umeyama_camera_centres", "transform": transform.as_dict(),
               "inlier_mask": final_inliers.tolist(), "correspondences": len(source),
               "inliers": int(final_inliers.sum()), "gnss_baseline_m": baseline,
               "residuals_m": residuals.tolist(),
               "gnss_fit_rmse_m": float(np.sqrt(np.mean(residuals[final_inliers] ** 2))),
               "fit_is_independent_accuracy": False, "conditioning": conditioning, "warnings": warnings}
    return transform, metrics
