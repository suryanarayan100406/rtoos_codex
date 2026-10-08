"""S9: supported raster surfaces and observed-image orthorectification."""

from typing import Any

from drishti.errors import DrishtiError
from drishti.stages.base import StageResult


class S9Geo:
    name = "s9_geo"
    requires = ("s7_dense", "s8_mesh", "s6_global", "s3_masking")
    config_sections = ("geo", "dense", "crs", "semantics")

    def run(self, bundle, cfg, runtime):
        import numpy as np

        from drishti.geo.products import (
            georeference,
            orthoproject,
            rasterize_cloud,
            read_cloud,
            read_npz,
            validate_visibility_mesh,
            write_geotiff,
        )
        from drishti.geo.semantics import semantic_products

        directory = bundle.stage_dir(self.name)
        outputs: dict[str, str] = {}
        reasons: list[str] = []
        metadata: dict[str, Any] = {"status": "degraded", "georeference": None, "products": {}}
        poses: dict[str, Any] = {}
        products: dict[str, Any] | None = None
        ref: dict[str, Any] | None = None
        mesh: dict[str, Any] | None = None
        masks: dict[str, str] | None = None
        semantic_mask_error: str | None = None

        def relative(path):
            return path.relative_to(bundle.root).as_posix()

        def remove_raster(path):
            path.unlink(missing_ok=True)
            path.with_suffix(".tif.msk").unlink(missing_ok=True)

        def publish_raster(key, data, grid, ref, *, supported_cells=None, **kwargs):
            path = directory / f"{key}.tif"
            try:
                remove_raster(path)
                write_geotiff(path, data, grid, ref, **kwargs)
                outputs[key] = relative(path)
                record: dict[str, Any] = {"status": "ok", "path": outputs[key], "reason": None}
                if supported_cells is not None:
                    record["supported_cells"] = supported_cells
                metadata["products"][key] = record
            except Exception as exc:
                reason = f"{key} unavailable: {type(exc).__name__}: {exc}"
                metadata["products"][key] = {"status": "failed", "path": None, "reason": reason}
                reasons.append(reason)
                try:
                    remove_raster(path)
                except OSError:
                    pass

        try:
            source_poses = bundle.read_json(relative(bundle.artifact("s6_global", "poses")))
            if not isinstance(source_poses, dict):
                raise ValueError("Final poses descriptor must be an object")
            poses = source_poses
            ref = georeference(poses, bundle.manifest.get("crs"))
            metadata["georeference"] = ref
            cloud = read_cloud(bundle)
            products = rasterize_cloud(cloud, cfg.geo, chunk_size=cfg.dense.max_points_per_chunk)
            metadata.update(products["metadata"])
            grid = products["grid"]
            metadata["grid"] = {"width": grid.width, "height": grid.height,
                                "resolution_m": grid.resolution,
                                "transform": list(grid.transform(ref["origin"]))[:6]}
            try:
                mesh = validate_visibility_mesh(read_npz(bundle.artifact("s8_mesh", "mesh")))
            except (DrishtiError, KeyError, OSError, ValueError, TypeError, IndexError) as exc:
                reasons.append(f"Mesh visibility unavailable: {exc}")
            try:
                mask_meta = bundle.read_json(relative(bundle.artifact("s3_masking", "masks")))
                frames = mask_meta["frames"]
                if (not isinstance(frames, list)
                        or any(not isinstance(item, dict)
                               or not isinstance(item.get("frame_id"), str)
                               or not item["frame_id"].strip()
                               or not isinstance(item.get("path"), str) or not item["path"]
                               for item in frames)
                        or len({item["frame_id"] for item in frames}) != len(frames)):
                    raise ValueError("Dynamic mask frames must have unique IDs and actual paths")
                masks = {item["frame_id"]: item["path"] for item in frames}
            except (DrishtiError, KeyError, OSError, ValueError, TypeError) as exc:
                if "masks" in bundle.manifest.get("stages", {}).get("s3_masking", {}).get("outputs", {}):
                    semantic_mask_error = f"Recorded dynamic mask descriptor unavailable: {exc}"
            metadata["dynamic_mask_reason"] = semantic_mask_error
            for key in ("dsm", "dtm", "confidence"):
                values = products[key]
                valid = np.isfinite(values)
                absolute = values + ref["origin"][2] if key != "confidence" else values
                # Elevations stay float64; adding an absolute origin before float32
                # would otherwise lose small vertical differences at large datums.
                data = np.where(valid, absolute, cfg.geo.nodata)
                publish_raster(key, data, grid, ref, nodata=cfg.geo.nodata, valid=valid,
                               supported_cells=int(valid.sum()),
                               descriptions=("absolute_elevation_m" if key != "confidence"
                                             else "observed_point_mean_confidence",),
                               tags={"inferred_excluded": "true",
                                     "elevation_frame": "projected_absolute" if key != "confidence"
                                     else "dimensionless_confidence"})
                if not valid.any():
                    reasons.append(f"{key} contains no observed support")
            for key in ("support", "mask"):
                data = products[key].astype(np.uint32 if key == "support" else np.uint8)
                publish_raster(key, data, grid, ref, nodata=0,
                               descriptions=("observed_point_count" if key == "support"
                                             else "0_unsupported_1_observed_2_inferred_only",))
            path = directory / "ortho.tif"
            try:
                remove_raster(path)
                rgba, ortho_meta = orthoproject(
                    products, poses.get("cameras", []), bundle.root, mesh=mesh,
                    masks=masks, chunk_size=cfg.dense.max_points_per_chunk,
                )
                metadata["orthomosaic"] = ortho_meta
                if ortho_meta["textured_cells"]:
                    write_geotiff(path, np.moveaxis(rgba, -1, 0), grid, ref,
                                  nodata=None, valid=rgba[..., 3] > 0,
                                  descriptions=("red", "green", "blue", "alpha"))
                    import rasterio
                    from rasterio.enums import ColorInterp

                    with rasterio.open(path, "r+") as dst:
                        dst.colorinterp = (ColorInterp.red, ColorInterp.green,
                                           ColorInterp.blue, ColorInterp.alpha)
                    outputs["ortho"] = relative(path)
                    metadata["products"]["ortho"] = {"status": "ok", "path": outputs["ortho"],
                        "supported_cells": ortho_meta["textured_cells"], "reason": None}
                else:
                    reason = "No DSM cell could be textured from final observed images"
                    metadata["products"]["ortho"] = {"status": "unavailable", "path": None,
                                                      "supported_cells": 0, "reason": reason}
                    reasons.append(reason)
                if ortho_meta["cameras_skipped"]:
                    reasons.append("Some final cameras could not be projected; see metadata")
            except Exception as exc:
                reason = f"Orthomosaic unavailable: {type(exc).__name__}: {exc}"
                metadata["products"]["ortho"] = {"status": "failed", "path": None, "reason": reason}
                reasons.append(reason)
                try:
                    remove_raster(path)
                except OSError:
                    pass
        except (DrishtiError, ImportError, KeyError, OSError, ValueError, TypeError) as exc:
            reasons.append(f"Geospatial products unavailable: {exc}")
        semantic_outputs, semantic_metadata = semantic_products(
            bundle, cfg, products, poses.get("cameras", []), ref, mesh=mesh, masks=masks,
            mask_error=semantic_mask_error,
        )
        outputs.update(semantic_outputs)
        path = directory / "semantics.json"
        bundle.write_json(relative(path), semantic_metadata)
        outputs["semantic_metadata"] = relative(path)
        metadata["semantics"] = semantic_metadata
        metadata["status"] = "degraded" if reasons else "ok"
        metadata["reasons"] = reasons
        path = directory / "geo.json"
        bundle.write_json(relative(path), metadata)
        outputs["metadata"] = relative(path)
        return StageResult(outputs=outputs, metrics=metadata, confidence=None,
                           degraded_reason="; ".join(reasons) or None)


STAGE = S9Geo()
