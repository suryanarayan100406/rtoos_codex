"""Poisson surface and real observed-image texturing, preserving partial products."""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from drishti.errors import DependencyError, InputError, ReconstructionError
from drishti.recon.meshing import poisson_mesh
from drishti.recon.texturing import best_view_texture, write_material, write_obj
from drishti.stages.base import StageResult


class TexturePublicationError(OSError):
    def __init__(self, message, rollback_failed=False):
        super().__init__(message)
        self.rollback_failed = rollback_failed


def publish_texture_set(work, destination):
    """Publish prepared files with rollback to the real partial products on error."""
    names = ("texture.png", "model.mtl", "model.obj", "mesh.npz")
    previous = work/"previous"
    previous.mkdir()
    for name in names:
        if (destination/name).exists():
            shutil.copy2(destination/name, previous/name)
    published = []
    try:
        for name in names:
            (work/name).replace(destination/name)
            published.append(name)
    except OSError as exc:
        failures = []
        for name in reversed(published):
            try:
                if (previous/name).exists():
                    (previous/name).replace(destination/name)
                else:
                    (destination/name).unlink()
            except OSError as restore_error:
                failures.append(f"{name}: {restore_error}")
        message = f"Texture publication failed: {exc}"
        if failures:
            message += "; partial product rollback failed: " + "; ".join(failures)
        raise TexturePublicationError(message, bool(failures)) from exc


class S8Mesh:
    name = "s8_mesh"
    requires = ("s7_dense", "s6_global", "s3_masking")
    config_sections = ("mesh", "dense")

    def run(self, bundle, cfg, runtime):
        from PIL import Image

        folder = bundle.stage_dir(self.name)
        def relative(path):
            return path.relative_to(bundle.root).as_posix()

        poses = bundle.read_json(relative(bundle.artifact("s6_global", "poses")))
        cloud_meta = bundle.read_json(relative(bundle.artifact("s7_dense", "metadata")))
        metadata = {"frame": "projected_local", "origin": poses["origin"], "crs": poses.get("crs"),
                    "source_cloud": relative(bundle.artifact("s7_dense", "cloud")),
                    "confidence_is_independent_accuracy": False}
        outputs = {"metadata": relative(folder/"mesh.json")}
        reasons = []
        if cfg.mesh.method != "poisson":
            raise InputError(f"Mesh algorithm {cfg.mesh.method!r} is not implemented")
        if cloud_meta.get("arrays"):
            arrays = {key: np.load(bundle.root/path, mmap_mode="r", allow_pickle=False)
                      for key, path in cloud_meta["arrays"].items()}
        else:
            raise InputError("Bounded meshing needs S7's mmap array sidecars; rerun S7")
        try:
            mesh, method_metadata = poisson_mesh(arrays, cfg.mesh, cfg.dense.max_points_per_chunk)
            metadata.update(method_metadata)
        except (DependencyError, ReconstructionError, RuntimeError) as exc:
            metadata.update(status="unavailable" if isinstance(exc, DependencyError) else "failed",
                            reason=str(exc), algorithm_executed=False if isinstance(exc, DependencyError) else None)
            bundle.write_json(outputs["metadata"], metadata)
            return StageResult(outputs, metrics=metadata, degraded_reason=str(exc))
        # Persist the real untextured surface before attempting optional texturing.
        np.savez(folder/"mesh.npz", **mesh)
        write_obj(folder/"model.obj", mesh["vertices"], mesh["faces"])
        outputs.update(mesh=relative(folder/"mesh.npz"), obj=relative(folder/"model.obj"))
        artifacts_valid = True
        if cfg.mesh.texture == "best_view":
            masks = bundle.read_json(relative(bundle.artifact("s3_masking", "masks")))
            mask_paths = {f["frame_id"]: bundle.root/f["path"] for f in masks["frames"]}
            try:
                result = best_view_texture(
                    mesh["vertices"], mesh["faces"], poses["cameras"],
                    lambda c: np.asarray(Image.open(bundle.root/c["image_path"]).convert("RGB")),
                    lambda c: np.asarray(Image.open(mask_paths[c["frame_id"]]).convert("L")) > 0,
                    cfg.mesh, mesh["inferred"]
                )
                metadata.update(result.metadata)
                if not result.metadata["observed_texels"]:
                    raise ReconstructionError("No visible unmasked supported image texels; preserved untextured Poisson mesh")
                mapped = {key: mesh[key][result.vertex_source] for key in ("colors", "confidence", "inferred")}
                textured_mesh = {"vertices": result.vertices, "faces": result.faces, "uv": result.uv,
                                 "source_views": result.source_views, **mapped}
                if len(result.vertices) > cfg.mesh.max_vertices:
                    raise ReconstructionError("UV-expanded mesh exceeds max_vertices")
                # Prepare the complete texture set before replacing real partial
                # products. Disk/inference errors during baking leave the saved
                # untextured surface and its metadata coherent.
                with tempfile.TemporaryDirectory(prefix="texture-", dir=folder) as temporary:
                    work = Path(temporary)
                    Image.fromarray(result.texture).save(work/"texture.png")
                    write_material(work/"model.mtl")
                    write_obj(work/"model.obj", result.vertices, result.faces, result.uv, "model.mtl")
                    np.savez(work/"mesh.npz", **textured_mesh)
                    publish_texture_set(work, folder)
                mesh = textured_mesh
                outputs.update(material=relative(folder/"model.mtl"), texture=relative(folder/"texture.png"))
                metadata["texture_status"] = "ok"
                if result.metadata["observed_triangle_texel_fraction"] < 1:
                    reasons.append("Some mesh texels lack visible observed support; uncovered texels retain alpha=0")
            except (InputError, ReconstructionError, RuntimeError, OSError, KeyError) as exc:
                metadata["texture_status"] = "failed"
                if isinstance(exc, TexturePublicationError) and exc.rollback_failed:
                    artifacts_valid = False
                    outputs.pop("mesh", None)
                    outputs.pop("obj", None)
                    metadata["geometry_status"] = "publication_rollback_failed; use the preserved source cloud"
                reasons.append(str(exc))
        else:
            reasons.append(f"Requested texturing method {cfg.mesh.texture!r} was not executed; preserved untextured mesh")
        metadata.update(status="degraded" if reasons else "ok",
                        vertices=len(mesh["vertices"]) if artifacts_valid else None,
                        faces=len(mesh["faces"]) if artifacts_valid else None,
                        algorithm_executed=True, reasons=reasons)
        bundle.write_json(outputs["metadata"], metadata)
        return StageResult(outputs, metrics=metadata,
                           confidence=float(np.mean(mesh["confidence"]))
                           if artifacts_valid and len(mesh["confidence"]) else None,
                           degraded_reason="; ".join(reasons) or None)


STAGE = S8Mesh()
