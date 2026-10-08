"""PTS policy and COLMAP API/convention units, with no claimed real-flight solve."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from drishti.errors import InputError
from drishti.io.video import decode_video
from drishti.recon.sfm import extract_sparse
from drishti.stages.s1_frameqa import feature_displacement


def mock_pts_decoder(monkeypatch, pts_values):
    av = pytest.importorskip("av")
    image = pytest.importorskip("PIL.Image")
    # The decoder is isolated: supplied PTS records test policy, not video quality.
    stream = SimpleNamespace(start_time=100, time_base=Fraction(1, 10),
                             codec_context=SimpleNamespace(name="isolated_unit_decoder", thread_count=0))
    frames = [SimpleNamespace(pts=pts, time_base=Fraction(1, 10), width=8, height=8,
                              to_image=lambda: image.new("RGB", (8, 8))) for pts in pts_values]
    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, selected_stream):
            assert selected_stream is stream
            return iter(frames)
    monkeypatch.setattr(av, "open", lambda path: Container())
    return stream


def test_variable_pts_sampling_never_uses_record_index_clock(monkeypatch, tmp_path):
    stream = mock_pts_decoder(monkeypatch, [100, 103, 108, 112])
    frames, metadata = decode_video(tmp_path / "isolated_unit.mp4", tmp_path / "frames", tmp_path, target_fps=2)
    np.testing.assert_allclose([f["time_s"] for f in frames], [0, 0.8, 1.2])
    assert [f["pts"] for f in frames] == [100, 108, 112]
    assert metadata["sampling_dropped_frames"] == 1
    assert metadata["sampling_drops"][0]["frame_id"] == "f00000001"
    assert metadata["sampling_drops"][0]["reason"] == "target_fps_sampling"
    assert metadata["sampling_drops"][0]["time_s"] == pytest.approx(0.3)
    assert stream.codec_context.thread_count == 1 and stream.thread_type == "NONE"
    assert all((tmp_path / frame["path"]).is_file() for frame in frames)


def test_max_frame_limit_records_the_unprocessed_tail(monkeypatch, tmp_path):
    mock_pts_decoder(monkeypatch, [100, 103, 108, 112])
    frames, metadata = decode_video(tmp_path / "isolated_unit.mp4", tmp_path / "frames", tmp_path,
                                    target_fps=10, max_frames=2)
    assert len(frames) == 2
    assert metadata["decoded_frames"] == 4
    assert metadata["max_frames_dropped_frames"] == 2
    assert [item["reason"] for item in metadata["sampling_drops"]] == ["max_frames_limit", "max_frames_limit"]
    assert metadata["truncated_by_max_frames"]


@pytest.mark.parametrize("missing_metadata", ["start_time", "time_base"])
def test_missing_stream_clock_metadata_uses_first_valid_frame_pts(monkeypatch, tmp_path, missing_metadata):
    stream = mock_pts_decoder(monkeypatch, [100, 108])
    stream.start_time = 90
    setattr(stream, missing_metadata, None)
    frames, metadata = decode_video(tmp_path / "isolated_unit.mp4", tmp_path / "frames", tmp_path, target_fps=2)
    np.testing.assert_allclose([frame["time_s"] for frame in frames], [0, 0.8])
    assert metadata["pts_origin_s"] == 10
    assert metadata["first_decoded_pts_s"] == 10


@pytest.mark.parametrize("pts", [[100, None], [100, 100], [100, 99]])
def test_pts_absence_duplicates_and_reversal_fail(monkeypatch, tmp_path, pts):
    mock_pts_decoder(monkeypatch, pts)
    with pytest.raises(InputError, match="timestamp"):
        decode_video(tmp_path / "isolated_unit.mp4", tmp_path / "frames", tmp_path, target_fps=4)


def test_feature_displacement_and_grid_coverage_are_measured():
    cv2 = pytest.importorskip("cv2")
    previous = ([cv2.KeyPoint(10, 10, 1), cv2.KeyPoint(60, 60, 1)], np.zeros((2, 32), dtype=np.uint8), 100, 100)
    current = ([cv2.KeyPoint(20, 10, 1), cv2.KeyPoint(70, 60, 1)], np.zeros((2, 32), dtype=np.uint8), 100, 100)
    matcher = SimpleNamespace(match=lambda a, b: [cv2.DMatch(0, 0, 0), cv2.DMatch(1, 1, 0)])
    displacement, count, coverage = feature_displacement(previous, current, matcher)
    assert displacement == 10 and count == 2 and coverage == 2 / 16
    assert feature_displacement(previous, ([], None, 100, 100), matcher) == (None, 0, 0)


def test_actual_pycolmap_sparse_api_exports_world_to_camera_and_observations(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    # Manually constructed camera/track unit verifies pycolmap binding compatibility.
    reconstruction = pycolmap.Reconstruction()
    reconstruction.add_camera(pycolmap.Camera(model="PINHOLE", width=100, height=100,
                                              params=[100, 100, 50, 50], camera_id=1))
    frames = []
    track = pycolmap.Track()
    for index, (center, uv) in enumerate([([0, 0, 0], [60, 60]), ([2, 0, 0], [40, 60]), ([0, 2, 0], [60, 40])], 1):
        name = f"f{index}.png"
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), -np.asarray(center, dtype=float))
        reconstruction.add_image(pycolmap.Image(name=name, keypoints=np.array([uv], dtype=float),
                                                cam_from_world=pose, camera_id=1, id=index))
        track.add_element(index, 0)
        frames.append({"id": f"f{index}", "path": name, "time_s": float(index), "gps": None})
    point_id = reconstruction.add_point3D(np.array([1., 1., 10.]), track, np.array([1, 2, 3], dtype=np.uint8))
    reconstruction.points3D[point_id].error = 0.25
    cameras, sparse = extract_sparse(reconstruction, frames, tmp_path / "undistorted", tmp_path, 3)
    assert len(cameras) == 3 and sparse["points"].dtype == np.float64
    np.testing.assert_allclose(cameras[1]["center"], [2, 0, 0])
    np.testing.assert_allclose(cameras[1]["R_cw"], np.eye(3))
    assert cameras[0]["distortion"] == [] and cameras[0]["projection_domain"] == "undistorted_pinhole"
    assert cameras[0]["observations"] == [{"point_index": 0, "uv": [60., 60.]}]
    assert sparse["confidence"][0] == pytest.approx((1 - 1 / 3) / 1.25)
