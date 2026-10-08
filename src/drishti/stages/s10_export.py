"""S10: per-format exports; failures preserve the actual successful deliverables."""

from pathlib import Path
from typing import Any

from drishti.errors import DrishtiError
from drishti.stages.base import StageResult


class S10Export:
    name = "s10_export"
    requires = ("s7_dense", "s8_mesh", "s9_geo", "s6_global")
    config_sections = ("export", "crs", "dense")

    def run(self, bundle, cfg, runtime):
        import json
        import shutil

        from drishti.geo.products import georeference, read_cloud, read_npz
        from drishti.io.exporters import (
            ExportUnavailable,
            copy_geotiffs,
            copy_obj,
            export_fbx,
            export_gltf,
            export_las,
            export_optional,
            export_ply,
        )

        directory = bundle.stage_dir(self.name)
        outputs: dict[str, str] = {}
        formats: dict[str, dict[str, Any]] = {}
        optional_products: dict[str, dict[str, Any]] = {}
        reference: dict[str, Any] | None = None
        ref_reason = None

        def relative(path):
            return path.relative_to(bundle.root).as_posix()

        def artifact(stage, key):
            try:
                return bundle.artifact(stage, key)
            except (DrishtiError, KeyError, OSError, ValueError) as exc:
                raise ExportUnavailable(f"Missing {stage}/{key}: {exc}") from exc

        try:
            poses = bundle.read_json(relative(artifact("s6_global", "poses")))
            reference = georeference(poses, bundle.manifest.get("crs"))
        except (ImportError, ExportUnavailable, OSError, ValueError, TypeError) as exc:
            ref_reason = str(exc)
        georef: dict[str, Any] = {"epsg": reference.get("epsg") if reference else None,
                  "origin": reference.get("origin") if reference else None,
                  "vertical": reference.get("vertical") if reference else "unknown",
                  "glb_axes": "x_east_y_up_z_south",
                  "reference": reference, "reason": ref_reason}
        sidecar = directory / "georeference.json"
        bundle.write_json(relative(sidecar), georef)
        outputs["georeference"] = relative(sidecar)
        cloud = mesh = None
        chunk = cfg.dense.max_points_per_chunk

        def get_cloud():
            nonlocal cloud
            if cloud is None:
                try:
                    cloud = read_cloud(bundle)
                except DrishtiError as exc:
                    raise ExportUnavailable(str(exc)) from exc
            return cloud

        def get_mesh():
            nonlocal mesh
            if mesh is None:
                mesh = read_npz(artifact("s8_mesh", "mesh"))
            return mesh

        required = ("obj", "ply", "las", "geotiff", "glb", "gltf", "fbx")
        requested = list(dict.fromkeys([*required, *cfg.export.formats, *cfg.export.optional_formats]))
        for kind in requested:
            target = directory / f"model.{kind}"
            try:
                detail: dict[str, Any] = {}
                if kind == "ply":
                    target = directory / "cloud.ply"
                    detail = export_ply(target, get_cloud(), reference, chunk)
                elif kind in ("las", "laz"):
                    target = directory / f"cloud.{kind}"
                    if reference is None:
                        raise ExportUnavailable(ref_reason or "Projected georeference is absent")
                    detail = export_las(target, get_cloud(), reference,
                                        scale_m=cfg.export.las_scale_m, chunk_size=chunk,
                                        compress=kind == "laz")
                elif kind == "obj":
                    target = copy_obj(artifact("s8_mesh", "obj"), artifact("s8_mesh", "material"),
                                      artifact("s8_mesh", "texture"), directory / "obj")
                    outputs["obj_assets"] = relative(target.parent)
                elif kind in ("glb", "gltf"):
                    if kind == "gltf":
                        folder = directory / "gltf"
                        folder.mkdir(exist_ok=True)
                        target = folder / "model.gltf"
                    detail = export_gltf(target, get_mesh(), artifact("s8_mesh", "texture"), reference)
                    georef[f"{kind}_transform"] = detail
                    georef["glb_axes"] = {"gltf_to_local": detail["gltf_to_local"],
                                          "mapping": detail["axis_mapping"]}
                    if kind == "gltf":
                        outputs["gltf_assets"] = relative(target.parent)
                    outputs[f"{kind}_georeference"] = relative(target.with_suffix(
                        target.suffix + ".georeference.json"))
                elif kind == "geotiff":
                    mandatory_geo = ("dsm", "dtm", "ortho")
                    optional_geo = ("confidence", "support", "mask", "semantic", "semantic_confidence")
                    copied: dict[str, Path] = {}
                    invalid_geo: list[str] = []
                    for key in (*mandatory_geo, *optional_geo):
                        optional_key = f"geotiff_{key}"
                        destination = directory / "geotiff" / f"{key}.tif"
                        try:
                            # Never include a previous invocation's optional file in
                            # the registered assets directory when its source is absent.
                            destination.unlink(missing_ok=True)
                            destination.with_suffix(".tif.msk").unlink(missing_ok=True)
                            if reference is None:
                                raise ExportUnavailable(ref_reason or "Projected georeference is absent")
                            source = artifact("s9_geo", key)
                            # Older bundles may use external GDAL masks. Preserve
                            # those before the independent reader validates the copy.
                            source_mask = source.with_suffix(source.suffix + ".msk")
                            if source_mask.exists():
                                source_mask = bundle.path(relative(source_mask))
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(source_mask, destination.with_suffix(".tif.msk"))
                            product = copy_geotiffs({key: source}, directory / "geotiff", reference)
                            product_detail: dict[str, Any] = {"reader_checked": "rasterio"}
                            if key in ("semantic", "semantic_confidence"):
                                import numpy as np
                                import rasterio

                                from drishti.geo.semantics import (
                                    CONFIDENCE_METHOD,
                                    validate_descriptor,
                                )

                                with rasterio.open(product[key]) as raster:
                                    tags = raster.tags()
                                    classes = json.loads(tags.get("semantic_classes", "null"))
                                    # Reuse categorical identity checks without
                                    # claiming to execute or validate source inference.
                                    validate_descriptor({"status": "unavailable", "nodata": 0,
                                                         "classes": classes, "frames": []})
                                    method = json.loads(tags.get("confidence_method", "null"))
                                    if not classes or method != CONFIDENCE_METHOD:
                                        raise ValueError("Semantic GeoTIFF lacks class/probability provenance")
                                    dtype, nodata = (("uint16", 0) if key == "semantic"
                                                     else ("float32", -9999.))
                                    if raster.count != 1 or raster.dtypes != (dtype,) or raster.nodata != nodata:
                                        raise ValueError("Semantic GeoTIFF categorical/probability storage is invalid")
                                    class_ids = {item["id"] for item in classes}
                                    for _, window in raster.block_windows(1):
                                        values = raster.read(1, window=window)
                                        valid = raster.dataset_mask(window=window) > 0
                                        if key == "semantic":
                                            if (not set(np.unique(values[valid])).issubset(class_ids)
                                                    or (values[~valid] != 0).any()):
                                                raise ValueError("Semantic GeoTIFF labels disagree with mapping/nodata")
                                        elif (not np.isfinite(values[valid]).all()
                                              or ((values[valid] < 0) | (values[valid] > 1)).any()
                                              or (values[~valid] != -9999.).any()):
                                            raise ValueError("Semantic GeoTIFF probabilities disagree with domain/nodata")
                                    product_detail.update(classes=classes, confidence_method=method,
                                                          provenance=json.loads(tags.get("semantic_provenance", "null")),
                                                          model=json.loads(tags.get("model_provenance", "null")))
                            copied.update(product)
                            outputs[optional_key] = relative(product[key])
                            if key in optional_geo:
                                optional_products[optional_key] = {
                                    "status": "ok", "path": outputs[optional_key], "reason": None,
                                    "details": product_detail,
                                }
                        except Exception as exc:
                            # A failed optional reader/writer is recorded individually;
                            # only DSM/DTM/orthomosaic affect mandatory GeoTIFF status.
                            if key in optional_geo:
                                optional_products[optional_key] = {
                                    "status": "unavailable" if isinstance(exc, ExportUnavailable | ImportError) else "failed",
                                    "path": None, "reason": f"{type(exc).__name__}: {exc}",
                                }
                            elif not isinstance(exc, ExportUnavailable):
                                invalid_geo.append(f"{key}: {exc}")
                            try:
                                destination.unlink(missing_ok=True)
                                destination.with_suffix(".tif.msk").unlink(missing_ok=True)
                            except OSError:
                                pass
                    if copied:
                        outputs["geotiff_assets"] = relative(directory / "geotiff")
                    detail = {"products": {key: relative(path) for key, path in copied.items()},
                              "reader_checked": "rasterio", "optional_products": optional_products}
                    if invalid_geo:
                        raise ValueError("Invalid GeoTIFF products: " + "; ".join(invalid_geo))
                    missing_geo = [key for key in mandatory_geo if key not in copied]
                    if missing_geo:
                        raise ExportUnavailable("Missing mandatory GeoTIFF products: " + ", ".join(missing_geo))
                    target = directory / "geotiff"
                elif kind == "fbx":
                    source = copy_obj(artifact("s8_mesh", "obj"), artifact("s8_mesh", "material"),
                                      artifact("s8_mesh", "texture"), directory / "fbx_input")
                    detail = export_fbx(source, target, cfg.export.blender)
                    georef["fbx_transform"] = detail
                elif kind in ("3dtiles", "potree"):
                    if formats.get("las", {}).get("status") != "ok":
                        raise ExportUnavailable(f"{kind} requires a successful LAS export from this invocation")
                    if reference is None:
                        raise ExportUnavailable(ref_reason or "Projected georeference is absent")
                    target = export_optional(kind, bundle.path(formats["las"]["path"]),
                                             directory / kind, reference)
                    outputs[f"{kind}_assets"] = relative(target.parent)
                else:
                    raise ExportUnavailable(f"No exporter for requested format {kind}")
                formats[kind] = {"status": "ok", "path": relative(target), "reason": None,
                                 "details": detail, "georeference": relative(sidecar)}
                outputs[kind] = relative(target)
            except (ExportUnavailable, ImportError) as exc:
                formats[kind] = {"status": "unavailable", "path": None, "reason": str(exc),
                                 "details": detail}
            except Exception as exc:
                # Each adapter is isolated: one writer failure must not erase a
                # real successful output or prevent the failure-path report.
                formats[kind] = {"status": "failed", "path": None,
                                 "reason": f"{type(exc).__name__}: {exc}", "details": detail}
        bundle.write_json(relative(sidecar), georef)
        missing = [kind for kind in required if formats[kind]["status"] != "ok"]
        if reference is None:
            missing.append("georeference")
        manifest = {"formats": formats, "georeference": georef, "optional_products": optional_products,
                    "missing_mandatory": missing, "status": "degraded" if missing else "ok"}
        path = directory / "exports.json"
        bundle.write_json(relative(path), manifest)
        outputs["exports"] = relative(path)
        return StageResult(outputs=outputs, metrics={"formats": formats, "missing_mandatory": missing,
                                                   "optional_products": optional_products},
                           confidence=None,
                           degraded_reason="Missing mandatory exports: " + ", ".join(missing) if missing else None)


STAGE = S10Export()
