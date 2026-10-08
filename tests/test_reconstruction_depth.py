"""Constructed math cases for isolated units, not real-data product acceptance."""

import numpy as np
import pytest

from drishti.errors import InputError, ReconstructionError
from drishti.recon.depth import align_inverse_depth, pixel_rays, project_points, sparse_anchors


def inverse_case():
    y, x = np.indices((40, 60))
    prediction = (1+x/20+y/40).astype(np.float32)
    uv = np.array([(x, y) for y in range(3, 38, 5) for x in range(3, 58, 6)])
    z = 1/(0.08*prediction[uv[:, 1], uv[:, 0]]+0.01)
    return prediction, np.column_stack((uv, z)), np.zeros(prediction.shape, bool)


def test_affine_inverse_fit_then_invert():
    prediction, anchors, mask = inverse_case()
    fit = align_inverse_depth(prediction, anchors, mask)
    np.testing.assert_allclose([fit.a, fit.b], [0.08, 0.01], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(fit.depth, 1/(0.08*prediction+0.01), rtol=1e-6)
    assert fit.rmse_m < 1e-5
    # A depth-linear fit after first inverting relative output is a different model.
    p = prediction[anchors[:, 1].astype(int), anchors[:, 0].astype(int)]
    wrong = np.polyfit(1/p, anchors[:, 2], 1)
    assert np.max(np.abs(np.polyval(wrong, 1/p)-anchors[:, 2])) > 0.01


def test_outlier_anchors_do_not_move_metric_surface():
    prediction, anchors, mask = inverse_case()
    anchors[::5, 2] *= 6
    fit = align_inverse_depth(prediction, anchors, mask)
    assert fit.inlier_count < fit.anchor_count
    np.testing.assert_allclose([fit.a, fit.b], [0.08, 0.01], rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(fit.depth, 1/(0.08*prediction+0.01), rtol=1e-4)


def test_dynamic_pixels_and_anchors_excluded_confidence_is_support():
    prediction, anchors, mask = inverse_case()
    mask[:10] = True
    fit = align_inverse_depth(prediction, anchors, mask, support_radius_px=5)
    assert np.all(fit.depth[:10] == 0)
    assert np.all(fit.confidence[:10] == 0)
    assert fit.anchor_count < len(anchors)
    assert 0 <= fit.confidence.min() <= fit.confidence.max() <= 1
    assert fit.confidence[13, 3] > fit.confidence[-1, -1]


def test_unanchored_or_unvarying_predictions_fail():
    prediction, anchors, mask = inverse_case()
    with pytest.raises(ReconstructionError, match="Too few"):
        align_inverse_depth(prediction, anchors[:3], mask)
    with pytest.raises(ReconstructionError, match="varied"):
        align_inverse_depth(np.ones_like(prediction), anchors, mask)
    with pytest.raises(ReconstructionError):
        align_inverse_depth(prediction, anchors, np.ones_like(mask))


@pytest.mark.parametrize("model,d", [("SIMPLE_RADIAL", [0.04]), ("RADIAL", [0.04, -0.01]),
                                    ("OPENCV", [0.04, -0.01, 0.002, -0.001]),
                                    ("OPENCV_FISHEYE", [0.04, -0.01, 0, 0])])
def test_distorted_camera_projection_roundtrip(model, d):
    camera = {"K": [[110, 0, 50], [0, 105, 45], [0, 0, 1]],
              "R_cw": [[0, -1, 0], [1, 0, 0], [0, 0, 1]], "center": [80, -7, 11],
              "camera_model": model, "distortion": d}
    uv = np.array([[14., 9.], [60., 56.], [88., 73.]])
    z = np.array([3., 4., 8.])
    points = pixel_rays(uv, camera)*z[:, None] @ np.array(camera["R_cw"]) + camera["center"]
    projected, actual_z = project_points(points, camera)
    np.testing.assert_allclose(projected, uv, atol=1e-5)
    np.testing.assert_allclose(actual_z, z)


def test_unsupported_distortion_is_explicit():
    with pytest.raises(InputError, match="Unsupported"):
        pixel_rays([[0, 0]], {"K": np.eye(3), "distortion": [0.1]*3, "camera_model": "UNKNOWN"})


def test_sparse_anchor_reprojection_uses_final_pose_and_track_associations():
    camera = {"K": [[10, 0, 10], [0, 10, 10], [0, 0, 1]], "R_cw": np.eye(3),
              "center": [1, 0, 0], "width": 30, "height": 30, "distortion": [],
              "anchors": [[0, 0, 999, 0]],
              "observations": [{"point_index": 0, "uv": [10, 10]},
                               {"point_index": 2, "uv": [1, 1]}]}
    points = np.array([[1, 0, 3], [1, 0, 6], [2, 0, 3]], np.float64)
    anchors = sparse_anchors(points, camera, np.zeros((30, 30), bool), max_reprojection_error_px=2)
    np.testing.assert_allclose(anchors, [[10, 10, 3]])


def test_sparse_pixel_z_buffer_and_exclusion():
    camera = {"K": [[10, 0, 10], [0, 10, 10], [0, 0, 1]], "R_cw": np.eye(3),
              "center": [0, 0, 0], "width": 30, "height": 30, "distortion": []}
    points = np.array([[0, 0, 3], [0, 0, 6], [1, 0, 3]], np.float64)
    mask = np.zeros((30, 30), bool)
    anchors = sparse_anchors(points, camera, mask)
    assert len(anchors) == 2
    assert anchors[0, 2] == 3
    mask[10, 10] = True
    assert len(sparse_anchors(points, camera, mask)) == 1
