"""Single-threaded PyAV decoding on presentation timestamps, never frame-rate clocks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from drishti.errors import DependencyError, InputError


def decode_video(path: Path, output_dir: Path, bundle_root: Path, *, target_fps: float,
                 max_frames: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import av
    except ImportError as exc:
        raise DependencyError("Video decoding requires PyAV (install the ingest extra)") from exc
    if not math.isfinite(target_fps) or target_fps <= 0 or max_frames < 0:
        raise InputError("target_fps must be positive and max_frames non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    decoded = 0
    sampled_out = 0
    sampling_drops = []
    max_frame_drops = 0
    first_pts_s = None
    previous_pts_s = None
    next_slot = 0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise InputError(f"Video {path} has no video stream")
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        stream.thread_type = "NONE"
        stream_time_base = stream.time_base
        clock_origin = (float(stream.start_time * stream_time_base)
                        if stream.start_time is not None and stream_time_base is not None else None)
        for decoded_frame in container.decode(stream):
            decoded += 1
            if decoded_frame.pts is None or decoded_frame.time_base is None:
                raise InputError(f"Decoded video frame {decoded} lacks presentation timestamp/time base")
            pts_s = float(decoded_frame.pts * decoded_frame.time_base)
            if not math.isfinite(pts_s):
                raise InputError("Video presentation timestamp is non-finite")
            if previous_pts_s is not None and pts_s <= previous_pts_s:
                raise InputError("Video presentation timestamps are duplicate/non-monotonic")
            previous_pts_s = pts_s
            if first_pts_s is None:
                first_pts_s = pts_s
            if clock_origin is None:
                clock_origin = pts_s
            time_s = pts_s - clock_origin
            slot = math.floor(time_s * target_fps + 1e-9)
            if slot < next_slot:
                sampled_out += 1
                sampling_drops.append({"frame_id": f"f{decoded - 1:08d}", "pts": int(decoded_frame.pts),
                                       "time_s": time_s, "reason": "target_fps_sampling"})
                continue
            if max_frames and len(frames) >= max_frames:
                max_frame_drops += 1
                sampling_drops.append({"frame_id": f"f{decoded - 1:08d}", "pts": int(decoded_frame.pts),
                                       "time_s": time_s, "reason": "max_frames_limit"})
                continue
            next_slot = slot + 1
            frame_id = f"f{decoded - 1:08d}"
            destination = output_dir / f"{frame_id}.png"
            decoded_frame.to_image().save(destination, format="PNG")
            frames.append({"id": frame_id, "path": destination.relative_to(bundle_root).as_posix(),
                           "time_s": time_s, "pts": int(decoded_frame.pts),
                           "time_base": str(decoded_frame.time_base), "width": decoded_frame.width,
                           "height": decoded_frame.height})
        metadata = {"codec": stream.codec_context.name, "time_base": str(stream.time_base),
                    "pts_origin_s": clock_origin, "first_decoded_pts_s": first_pts_s,
                    "clock": "clip_seconds_from_stream_start_pts_or_first_pts", "decoder": "pyav",
                    "decoder_threads": 1, "decoded_frames": decoded, "selected_frames": len(frames),
                     "sampling_dropped_frames": sampled_out, "sampling_drop_reason": "target_fps",
                     "sampling_drops": sampling_drops,
                     "max_frames_dropped_frames": max_frame_drops,
                     "truncated_by_max_frames": bool(max_frames and len(frames) >= max_frames)}
    if not frames:
        raise InputError("Video yielded no decodable timestamped frames")
    return frames, metadata
