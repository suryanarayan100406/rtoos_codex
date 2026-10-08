"""Mandatory input validation, recorded-clock alignment and optional-input ledger."""

from pathlib import Path
from typing import Any

from drishti.errors import InputError
from drishti.geo.crs import crs_metadata, derive_epsg, project_gps
from drishti.io.barometer import BarometerAligner, load_barometer
from drishti.io.telemetry import TelemetryInterpolator, load_telemetry, parse_csv
from drishti.io.video import decode_video
from drishti.stages.base import StageResult


class S0Ingest:
    name = "s0_ingest"
    requires = ()
    config_sections = ("dataset", "ingest", "crs", "barometer")

    def run(self, bundle: Any, cfg: Any, runtime: Any) -> StageResult:
        stage_dir = bundle.stage_dir(self.name)
        dataset = cfg.dataset
        barometer_path = f"{self.name}/barometer.json"
        barometer = load_barometer(dataset.optional.baro, cfg.barometer)
        # Retain a real optional-input descriptor even if mandatory ingest fails.
        bundle.write_json(barometer_path, barometer)
        telemetry = load_telemetry(dataset.telemetry)
        optional: dict[str, dict[str, Any]] = {}
        for name in ("imu", "baro", "intrinsics", "rtk"):
            path = getattr(dataset.optional, name)
            available = path is not None and Path(path).exists()
            optional[name] = {"provided": path is not None, "available": available, "used": False,
                              "status": "available_not_consumed" if available else "absent",
                              "reason": None if path is None or available else "provided_file_missing"}
        optional["baro"].update(parsed=barometer["status"] == "ok", parsing_status=barometer["status"],
                                parsing_reason=barometer["reason"], sample_count=len(barometer["samples"]))
        barometer_aligner = BarometerAligner(barometer["samples"], cfg.barometer) if barometer["status"] == "ok" else None
        if optional["rtk"]["available"]:
            try:
                corrected_samples = parse_csv(Path(dataset.optional.rtk), dataset.telemetry.csv)
                for sample in corrected_samples:
                    sample["source_time_s"] = sample["time_s"]
                    sample["time_s"] -= telemetry["clock"]["time_origin_s"]
                    sample["position_source"] = "rtk_corrected_csv"
                telemetry["rtk_samples"] = corrected_samples
                optional["rtk"].update(status="validated_corrected_positions_not_yet_used",
                                        reason="corrected geographic CSV; declared telemetry clock/datum")
            except (InputError, OSError) as exc:
                optional["rtk"].update(status="invalid_not_consumed", reason=str(exc))
        samples = telemetry["samples"]
        epsg = derive_epsg(samples, cfg.crs)
        # Validate the actual local transform, including a requested geoid grid,
        # before labeling the reconstruction's target vertical reference.
        project_gps(samples, epsg, settings=cfg.crs, source_datum=telemetry["altitude_datum"])
        bundle.manifest["crs"] = crs_metadata(epsg, samples, cfg.crs, telemetry["altitude_datum"])
        bundle.manifest["crs"]["logged_gps_projection_validated"] = True
        telemetry["clock"]["frame_time_offset_s"] = cfg.ingest.time_offset_s
        interpolator = TelemetryInterpolator(samples, cfg.ingest.max_telemetry_gap_s)
        rtk = TelemetryInterpolator(telemetry["rtk_samples"], cfg.ingest.max_telemetry_gap_s) if telemetry.get("rtk_samples") else None
        frames, video = decode_video(dataset.video, stage_dir / "frames", bundle.root,
                                     target_fps=cfg.ingest.target_fps, max_frames=cfg.ingest.max_frames)
        unsupported = 0
        rtk_frames = 0
        barometer_frames = 0
        for frame in frames:
            aligned_time = frame["time_s"] + cfg.ingest.time_offset_s
            frame["gps"], reason = interpolator.at(aligned_time)
            if rtk is not None:
                corrected_gps, _ = rtk.at(aligned_time)
                if corrected_gps is not None:
                    frame["gps"] = {**corrected_gps, "position_source": "rtk_corrected_csv"}
                    reason = None
                    rtk_frames += 1
            frame["telemetry_time_s"] = aligned_time
            frame["flags"] = [reason] if reason else []
            unsupported += reason is not None
            if barometer_aligner is None:
                frame["barometer"], frame["barometer_reason"] = None, barometer["reason"]
            else:
                frame["barometer"], frame["barometer_reason"] = barometer_aligner.at(frame["time_s"])
                barometer_frames += frame["barometer"] is not None
        if rtk is not None:
            optional["rtk"].update(used=rtk_frames > 0, aligned_frames=rtk_frames,
                                    status="consumed_corrected_positions" if rtk_frames else "valid_no_frame_time_overlap")
        barometer["aligned_frames"] = barometer_frames
        if barometer_aligner is not None and not barometer_frames:
            barometer.update(status="unavailable", reason="no_decoded_frame_has_a_bounded_barometer_sample")
        optional["baro"].update(used=False, aligned_frames=barometer_frames,
                                alignment_status="ok" if barometer_frames else barometer["status"],
                                status="aligned_not_fused" if barometer_frames else barometer["status"],
                                reason=barometer["reason"])
        telemetry["position_selection"] = "supported_RTK_corrected_positions_else_mandatory_GPS_without_extrapolation"
        frames_path = f"{self.name}/frames.json"
        telemetry_path = f"{self.name}/telemetry.json"
        bundle.write_json(barometer_path, barometer)
        bundle.write_json(telemetry_path, telemetry)
        bundle.write_json(frames_path, {"frames": frames, "video": video, "capabilities": optional})
        bundle.save()
        if unsupported == len(frames):
            raise InputError("No decoded frame overlaps supported telemetry; check time_origin_s/time_offset_s and gap limit")
        warnings = []
        if unsupported:
            warnings.append(f"{unsupported} frames have unsupported telemetry and will be dropped")
        if telemetry["altitude_datum"] == "unknown":
            warnings.append("absolute altitude datum is unknown; vertical accuracy unvalidated")
        return StageResult(outputs={"frames": frames_path, "telemetry": telemetry_path,
                                    "barometer": barometer_path,
                                    "images": f"{self.name}/frames"},
                           metrics={"frames": len(frames), "telemetry_samples": len(samples),
                                    "unsupported_telemetry_frames": unsupported, "optional_inputs": optional,
                                    "clock": telemetry["clock"]},
                           confidence=None, degraded_reason="; ".join(warnings) or None)


STAGE = S0Ingest()
