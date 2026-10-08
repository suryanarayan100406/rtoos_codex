"""Camera-domain geometry and robust relative-inverse-to-metric calibration."""

from dataclasses import dataclass

import numpy as np

from drishti.errors import InputError, ReconstructionError


def image_domain(camera):
    domain = camera.get("image_domain")
    if domain is None:
        domain = "undistorted" if camera.get("projection_domain") == "undistorted_pinhole" else "distorted"
    if domain not in {"distorted", "undistorted"}:
        raise InputError(f"Unsupported camera image domain {domain!r}")
    return domain


def distortion(camera):
    d = np.asarray(camera.get("distortion", []), dtype=np.float64)
    if image_domain(camera) == "undistorted" or not np.any(d):
        return "pinhole", np.zeros(4)
    model = camera.get("distortion_model", camera.get("camera_model"))
    if model is None:
        model = {1: "SIMPLE_RADIAL", 2: "RADIAL", 4: "OPENCV", 5: "OPENCV", 8: "FULL_OPENCV"}.get(len(d))
    if model == "SIMPLE_RADIAL" and len(d) == 1:
        return "opencv", np.array([d[0], 0, 0, 0])
    if model == "RADIAL" and len(d) == 2:
        return "opencv", np.array([d[0], d[1], 0, 0])
    if model in {"OPENCV", "FULL_OPENCV"} and len(d) in {4, 5, 8}:
        return "opencv", d
    if model == "OPENCV_FISHEYE" and len(d) == 4:
        return "fisheye", d
    raise InputError(f"Unsupported or ambiguous camera distortion model {model!r}")


def pixel_rays(uv, camera):
    """Camera rays with Z=1, for the actual image's distortion domain."""
    import cv2

    uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    kind, d = distortion(camera)
    K = np.asarray(camera["K"], np.float64)
    if kind == "fisheye":
        xy = cv2.fisheye.undistortPoints(uv, K, d)
    else:
        xy = cv2.undistortPoints(uv, K, d)
    return np.column_stack((xy.reshape(-1, 2), np.ones(len(uv))))


def project_points(points, camera):
    import cv2

    points = np.asarray(points, np.float64)
    xyz = (points - np.asarray(camera["center"], np.float64)) @ np.asarray(camera["R_cw"], np.float64).T
    kind, d = distortion(camera)
    K = np.asarray(camera["K"], np.float64)
    if len(points) == 0:
        return np.empty((0, 2)), np.empty(0)
    if kind == "fisheye":
        uv, _ = cv2.fisheye.projectPoints(xyz.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, d)
    else:
        uv, _ = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, d)
    return uv.reshape(-1, 2), xyz[:, 2]


def sparse_anchors(points, camera, excluded, confidence=None, max_anchors=None,
                   max_reprojection_error_px=None):
    """Reproject pose-matched local points through final poses; never reuse stale Z anchors.

    A pixel z-buffer rejects hidden sparse samples. These are geometric projected
    anchors, not claims that feature reprojection errors have been measured.
    """
    uv, z = project_points(points, camera)
    h, w = excluded.shape
    ok = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
    ok &= (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    if confidence is not None:
        ok &= np.isfinite(confidence) & (np.asarray(confidence) > 0)
    if "observations" in camera:
        observed = np.zeros(len(points), bool)
        for observation in camera["observations"]:
            index = int(observation["point_index"])
            measured_uv = np.asarray(observation["uv"], np.float64)
            if not 0 <= index < len(points) or measured_uv.shape != (2,):
                raise InputError("Invalid sparse-track observation")
            error = np.linalg.norm(uv[index]-measured_uv)
            observed[index] = np.isfinite(error) and (
                max_reprojection_error_px is None or error <= max_reprojection_error_px
            )
        ok &= observed
    indices = np.flatnonzero(ok)
    ij = np.rint(uv[indices]).astype(int)
    indices = indices[~excluded[ij[:, 1], ij[:, 0]].astype(bool)]
    if len(indices):
        indices = indices[np.argsort(z[indices], kind="stable")]
        ij = np.rint(uv[indices]).astype(int)
        _, first = np.unique(ij[:, 1] * w + ij[:, 0], return_index=True)
        indices = indices[np.sort(first)]
    if max_anchors and len(indices) > max_anchors:
        # Deterministic spatially spread selection, rather than the nearest depths only.
        order = np.lexsort((uv[indices, 0], uv[indices, 1]))
        indices = indices[order[np.linspace(0, len(indices) - 1, max_anchors, dtype=int)]]
    return np.column_stack((uv[indices], z[indices]))


def bilinear(array, uv):
    uv = np.asarray(uv, float)
    x, y = uv.T
    h, w = array.shape[:2]
    if np.any((x < 0) | (x > w - 1) | (y < 0) | (y > h - 1)):
        raise InputError("Sampling coordinates outside image")
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    if array.ndim == 3:
        dx, dy = dx[:, None], dy[:, None]
    return ((1-dx)*(1-dy)*array[y0, x0] + dx*(1-dy)*array[y0, x1]
            + (1-dx)*dy*array[y1, x0] + dx*dy*array[y1, x1])


@dataclass
class DepthAlignment:
    depth: np.ndarray
    confidence: np.ndarray
    a: float
    b: float
    anchor_count: int
    inlier_count: int
    rmse_m: float
    relative_rmse: float


def align_inverse_depth(prediction, anchors, excluded, *, min_anchors=8,
                        robust_delta=1.345, max_relative_residual=0.3,
                        min_depth_m=0.1, max_depth_m=300.0,
                        support_radius_px=128.0, robust_iterations=30,
                        min_inlier_fraction=0.5):
    """Fit 1/Z=a*p+b with robust inverse residuals, THEN invert.

    Confidence is a residual/support score, not an accuracy probability. It falls
    with distance from accepted anchors and outside their prediction range.
    """
    from scipy.spatial import cKDTree

    prediction = np.asarray(prediction, np.float64)
    excluded = np.asarray(excluded, bool)
    anchors = np.asarray(anchors, np.float64).reshape(-1, 3)
    if prediction.shape != excluded.shape:
        raise InputError("Depth and exclusion mask shapes differ")
    h, w = prediction.shape
    uv, z = anchors[:, :2], anchors[:, 2]
    good = np.isfinite(anchors).all(axis=1) & (z >= min_depth_m) & (z <= max_depth_m)
    good &= (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    uv, z = uv[good], z[good]
    lower = np.floor(uv).astype(int)
    upper = np.minimum(lower+1, [w-1, h-1])
    good = ~(excluded[lower[:, 1], lower[:, 0]] | excluded[lower[:, 1], upper[:, 0]]
             | excluded[upper[:, 1], lower[:, 0]] | excluded[upper[:, 1], upper[:, 0]])
    uv, z = uv[good], z[good]
    p = bilinear(prediction, uv)
    good = np.isfinite(p)
    p, uv, z = p[good], uv[good], z[good]
    if len(z) < min_anchors or np.ptp(p) <= np.finfo(float).eps * max(1, np.max(np.abs(p))):
        raise ReconstructionError("Too few valid or insufficiently varied inverse-depth anchors")
    # Normalize p for numerical conditioning. Robustly fit inverse-Z residuals.
    # Do not weight the design by Z: grossly overestimated Z outliers would gain
    # high leverage and can trap a relative-residual IRLS initialization.
    offset, scale = float(np.median(p)), float(np.std(p))
    design = np.column_stack(((p - offset) / scale, np.ones(len(p))))
    A = design
    target = 1/z
    weights = np.ones(len(p))
    coefficients = np.linalg.lstsq(A, target, rcond=None)[0]
    for _ in range(robust_iterations):
        residual = A @ coefficients - target
        mad = np.median(np.abs(residual - np.median(residual)))
        cutoff = max(robust_delta * mad / 0.6744897501960817, np.finfo(float).eps)
        weights = np.minimum(1, cutoff / np.maximum(np.abs(residual), np.finfo(float).eps))
        new = np.linalg.lstsq(A * np.sqrt(weights[:, None]), target*np.sqrt(weights), rcond=None)[0]
        if np.allclose(new, coefficients, rtol=1e-9, atol=1e-12):
            coefficients = new
            break
        coefficients = new
    inverse = design @ coefficients
    predicted_z = np.divide(1., inverse, out=np.full_like(inverse, np.inf), where=inverse > 0)
    relative = np.abs(predicted_z - z) / z
    inlier = np.isfinite(relative) & (relative <= max_relative_residual)
    if np.count_nonzero(inlier) < min_anchors or np.mean(inlier) < min_inlier_fraction:
        raise ReconstructionError("Inverse-depth fit rejected: insufficient inlier anchor support")
    if np.linalg.matrix_rank(A[inlier]) < 2:
        raise ReconstructionError("Accepted inverse-depth anchors do not constrain scale and shift")
    coefficients = np.linalg.lstsq(A[inlier] * np.sqrt(weights[inlier, None]),
                                   target[inlier]*np.sqrt(weights[inlier]), rcond=None)[0]
    a = coefficients[0] / scale
    b = coefficients[1] - a * offset
    if not np.isfinite(a + b) or a <= 0:
        raise ReconstructionError("Inverse-depth fit has non-positive slope or invalid coefficients")
    fitted = 1 / (a * p[inlier] + b)
    residual = np.abs(fitted - z[inlier]) / z[inlier]
    relative_rmse = float(np.sqrt(np.mean(residual**2)))
    if relative_rmse > max_relative_residual:
        raise ReconstructionError("Inverse-depth anchor residual exceeds configured tolerance")
    inv_map = a * prediction + b
    depth = np.divide(1., inv_map, out=np.zeros_like(inv_map), where=inv_map > 0)
    valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m) & ~excluded
    depth[~valid] = 0
    tree = cKDTree(uv[inlier])
    confidence = np.zeros_like(depth, dtype=np.float32)
    # Row-wise querying bounds confidence interpolation memory for large images.
    for y in range(h):
        distance, index = tree.query(np.column_stack((np.arange(w), np.full(w, y))))
        score = np.exp(-distance / support_radius_px) * np.exp(-residual[index] / max_relative_residual)
        confidence[y] = score * np.mean(inlier)
    lower, upper = np.min(p[inlier]), np.max(p[inlier])
    extrapolation = np.maximum(lower - prediction, np.maximum(prediction - upper, 0))
    confidence *= np.exp(-extrapolation / (upper - lower + np.finfo(float).eps))
    confidence[~valid] = 0
    return DepthAlignment(depth.astype(np.float32), confidence, float(a), float(b), len(z),
                          int(np.count_nonzero(inlier)), float(np.sqrt(np.mean((fitted-z[inlier])**2))),
                          relative_rmse)
