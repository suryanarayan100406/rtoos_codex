"""Real export adapters with reader checks and explicit coordinate sidecars."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


class ExportUnavailable(RuntimeError):
    """The actual dependency/input needed to produce an export is unavailable."""


def validate_mesh(mesh):
    import numpy as np

    vertices = np.asarray(mesh["vertices"])
    faces = np.asarray(mesh["faces"])
    if vertices.dtype != np.float64 or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("Mesh vertices must be Nx3 float64 local metres")
    if not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("Mesh has empty or nonfinite vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("Mesh must contain triangular faces")
    if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Mesh face indices are invalid")
    for key in ("confidence", "inferred"):
        if np.asarray(mesh[key]).shape != (len(vertices),):
            raise ValueError(f"Mesh {key} must contain one value per vertex")
    confidence = np.asarray(mesh["confidence"])
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Mesh confidence must be finite and in [0, 1]")
    if np.asarray(mesh["inferred"]).dtype != np.bool_:
        raise ValueError("Mesh inferred values must be booleans")
    return mesh


def export_ply(path: Path, cloud: dict, reference: dict | None, chunk_size: int):
    """Binary point PLY, double XYZ, actual confidence and inferred attributes."""
    import numpy as np

    from drishti.geo.products import validate_cloud

    cloud = validate_cloud(cloud, chunk_size=chunk_size)
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                      ("confidence", "<f4"), ("inferred", "u1")])
    header = ("ply\nformat binary_little_endian 1.0\n"
              "comment coordinates projected_local metres; see georeference.json\n"
              f"comment georeference {json.dumps(reference, separators=(',', ':'))}\n"
              f"element vertex {len(cloud['points'])}\n"
              "property double x\nproperty double y\nproperty double z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "property float confidence\nproperty uchar inferred\nend_header\n")
    if chunk_size <= 0:
        raise ValueError("Point export chunk size must be positive")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        for start in range(0, len(cloud["points"]), chunk_size):
            stop = start + chunk_size
            points, colors = cloud["points"][start:stop], cloud["colors"][start:stop]
            records = np.empty(len(points), dtype=dtype)
            for column, key in enumerate(("x", "y", "z")):
                records[key] = points[:, column]
            for column, key in enumerate(("red", "green", "blue")):
                records[key] = colors[:, column]
            records["confidence"] = cloud["confidence"][start:stop]
            records["inferred"] = cloud["inferred"][start:stop]
            stream.write(records.tobytes())
    # A standard independent reader is used when present; size is always checked.
    expected = len(header.encode("ascii")) + len(cloud["points"]) * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError("PLY record count/size mismatch")
    if importlib.util.find_spec("plyfile"):
        from plyfile import PlyData

        loaded = PlyData.read(path)
        if len(loaded["vertex"]) != len(cloud["points"]):
            raise ValueError("PLY reader returned a different point count")
    return {"points": len(cloud["points"]), "coordinate_frame": "projected_local",
            "xyz_storage": "float64", "attributes": ["confidence", "inferred"]}


def export_las(path: Path, cloud: dict, reference: dict, *, scale_m: float,
               chunk_size: int, compress=False):
    import laspy
    import numpy as np
    from pyproj import CRS

    from drishti.geo.products import validate_cloud

    cloud = validate_cloud(cloud, chunk_size=chunk_size)
    if not reference or not reference.get("wkt"):
        raise ExportUnavailable("LAS requires a validated projected CRS and origin")
    if not np.isfinite(scale_m) or scale_m <= 0 or chunk_size <= 0:
        raise ValueError("LAS scale and point chunk size must be positive")
    if compress and not laspy.LazBackend.detect_available():
        raise ExportUnavailable("No actual laspy LAZ compressor backend is installed")
    origin = np.asarray(reference["origin"], dtype=np.float64)
    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.full(3, scale_m, dtype=np.float64)
    header.offsets = origin.copy()
    header.add_crs(CRS.from_user_input(reference["wkt"]))
    header.add_extra_dims([
        laspy.ExtraBytesParams(name="confidence", type=np.float32,
                              description="Measured source confidence"),
        laspy.ExtraBytesParams(name="inferred", type=np.uint8,
                              description="1 inferred, 0 observed"),
    ])
    header.vlrs.append(laspy.VLR(
        user_id="DRISHTI", record_id=1, description="Datum and local-origin provenance",
        record_data=json.dumps(reference, separators=(",", ":")).encode("utf-8"),
    ))
    for start in range(0, len(cloud["points"]), chunk_size):
        if (np.abs(cloud["points"][start:start + chunk_size] / scale_m) > np.iinfo(np.int32).max - 1).any():
            raise ValueError("Local coordinate extent exceeds LAS int32 range at configured scale")
    with laspy.open(path, mode="w", header=header, do_compress=compress) as writer:
        for start in range(0, len(cloud["points"]), chunk_size):
            stop = start + chunk_size
            local = cloud["points"][start:stop]
            records = laspy.ScaleAwarePointRecord.zeros(len(local), header=header)
            # Global projected coordinates are float64 until integer quantization.
            absolute = local + origin
            records.x, records.y, records.z = absolute.T
            colors = cloud["colors"][start:stop].astype(np.uint16) * 257
            records.red, records.green, records.blue = colors.T
            records.confidence = cloud["confidence"][start:stop]
            records.inferred = cloud["inferred"][start:stop].astype(np.uint8)
            records.classification = np.ones(len(local), dtype=np.uint8)  # actually unclassified
            writer.write_points(records)
    with laspy.open(path) as reader:
        if reader.header.point_count != len(cloud["points"]):
            raise ValueError("LAS reader point count mismatch")
        if not reader.header.parse_crs().equals(CRS.from_user_input(reference["wkt"])):
            raise ValueError("LAS reader CRS mismatch")
        position = 0
        for records in reader.chunk_iterator(chunk_size):
            absolute = np.column_stack((records.x, records.y, records.z))
            expected = cloud["points"][position:position + len(records)] + origin
            tolerance = scale_m / 2 + np.spacing(np.abs(expected)).max()
            if not np.all(np.abs(absolute - expected) <= tolerance):
                raise ValueError("LAS coordinate quantization exceeds half the configured scale")
            position += len(records)
    return {"points": len(cloud["points"]), "scale_m": scale_m,
            "coordinate_frame": "projected_absolute", "reader_checked": "laspy"}


def copy_obj(obj: Path, material: Path, texture: Path, destination: Path):
    """Copy the observed-image OBJ material set, validating all file references."""
    if not all(path.is_file() and path.stat().st_size for path in (obj, material, texture)):
        raise ExportUnavailable("Textured OBJ, MTL, or observed-image texture is missing")
    destination.mkdir(parents=True, exist_ok=True)
    text = obj.read_text(encoding="utf-8")
    material_text = material.read_text(encoding="utf-8")
    materials = [line.split(maxsplit=1)[1] for line in text.splitlines()
                 if line.startswith("mtllib ")]
    textures = [line.split(maxsplit=1)[1] for line in material_text.splitlines()
                if line.startswith("map_Kd ")]
    if not materials or not textures:
        raise ValueError("OBJ material set does not reference a real diffuse image texture")
    for item in materials:
        if Path(item).name != material.name or Path(item).is_absolute() or ".." in Path(item).parts:
            raise ValueError("OBJ has unsupported or escaping material references")
    for item in textures:
        if Path(item).name != texture.name or Path(item).is_absolute() or ".." in Path(item).parts:
            raise ValueError("MTL has unsupported or escaping texture references")
    import trimesh

    loaded = trimesh.load(obj, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise ValueError("OBJ reader returned no actual mesh scene")
    meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
    if not any(len(item.faces) for item in meshes):
        raise ValueError("OBJ reader returned no actual triangular mesh")
    if not any(item.visual.kind == "texture" for item in meshes):
        raise ValueError("OBJ reader could not load the observed-image texture material")
    # Preserve names/relative paths exactly as the original set uses them.
    shutil.copy2(obj, destination / obj.name)
    for item in materials:
        target = destination / item
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(material, target)
        for image_name in textures:
            image_target = target.parent / image_name
            image_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(texture, image_target)
    return destination / obj.name


def recenter_mesh(mesh: dict, reference: dict | None):
    """Recenter in float64 *before* any glTF float32 serialization, then map axes."""
    import numpy as np

    validate_mesh(mesh)
    vertices = mesh["vertices"]
    center = vertices.min(axis=0) + (vertices.max(axis=0) - vertices.min(axis=0)) / 2
    local = vertices - center
    axes = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    converted = local @ axes.T
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = axes.T
    inverse[:3, 3] = center
    base_origin = np.asarray(reference["origin"]) if reference else None
    sidecar = {"source_frame": "projected_local", "source_local_center": center.tolist(),
               "projected_anchor": (base_origin + center).tolist() if reference else None,
               "axis_mapping": "glTF XYZ = local (X, Z, -Y); right-handed Y-up",
               "source_to_gltf_rotation": axes.tolist(),
               "gltf_to_local": inverse.ravel().tolist(),
               "inverse": "projected_XYZ = projected_anchor + glTF_XYZ @ source_to_gltf_rotation",
               "float32_max_rounding_error_m": float(np.max(np.abs(
                   converted.astype(np.float32).astype(np.float64) - converted))),
               "georeference": reference}
    return converted, sidecar


def export_gltf(path: Path, mesh: dict, texture: Path, reference: dict | None):
    import numpy as np
    import trimesh
    from PIL import Image

    converted, sidecar = recenter_mesh(mesh, reference)
    uv = np.asarray(mesh.get("uv"))
    if uv.shape != (len(converted), 2) or not np.isfinite(uv).all():
        raise ExportUnavailable("glTF requires actual observed-image UV coordinates")
    if not texture.is_file():
        raise ExportUnavailable("glTF requires the actual observed-image texture")
    with Image.open(texture) as image:
        has_alpha = "A" in image.getbands()
        image = image.convert("RGBA" if has_alpha else "RGB").copy()
    alpha_mode = "OPAQUE"
    alpha_cutoff = None
    if has_alpha:
        alpha = np.asarray(image.getchannel("A"))
        binary_alpha = bool(np.all((alpha == 0) | (alpha == 255)))
        alpha_mode = "MASK" if binary_alpha else "BLEND"
        alpha_cutoff = 0.5 if binary_alpha else None
    material_kwargs = {"baseColorTexture": image, "alphaMode": alpha_mode}
    if alpha_cutoff is not None:
        material_kwargs["alphaCutoff"] = alpha_cutoff
    material = trimesh.visual.material.PBRMaterial(**material_kwargs)
    visual = trimesh.visual.texture.TextureVisuals(uv=uv, material=material)
    geometry = trimesh.Trimesh(vertices=converted, faces=mesh["faces"], visual=visual,
                               process=False, validate=False)
    geometry.vertex_attributes["_CONFIDENCE"] = np.asarray(mesh["confidence"], dtype=np.float32)
    # uint16 is a valid glTF custom accessor; trimesh 4.6's uint8 accessor
    # lookup does not handle NumPy's endian-neutral '|u1' dtype spelling.
    geometry.vertex_attributes["_INFERRED"] = np.asarray(mesh["inferred"], dtype=np.uint16)
    scene = trimesh.Scene(geometry)
    sidecar["alpha_mode"] = alpha_mode
    if alpha_cutoff is not None:
        sidecar["alpha_cutoff"] = alpha_cutoff
    scene.metadata["drishti"] = sidecar
    if path.suffix == ".glb":
        path.write_bytes(trimesh.exchange.gltf.export_glb(scene, include_normals=True))
    else:
        files = trimesh.exchange.gltf.export_gltf(scene, include_normals=True)
        for filename, data in files.items():
            target = path if filename == "model.gltf" else path.parent / filename
            target.resolve().relative_to(path.parent.resolve())
            target.write_bytes(data)
        if not path.exists():
            raise ValueError("trimesh did not emit the requested glTF JSON")
    loaded = trimesh.load(path, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise ValueError("glTF reader returned no actual mesh scene")
    meshes = [item for item in loaded.geometry.values() if isinstance(item, trimesh.Trimesh)]
    if len(meshes) != len(loaded.geometry) or sum(len(item.vertices) for item in meshes) != len(converted):
        raise ValueError("glTF reader returned an empty mesh or changed vertex count")
    sidecar_path = path.with_suffix(path.suffix + ".georeference.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2, allow_nan=False), encoding="utf-8")
    return sidecar


def copy_geotiffs(sources: dict[str, Path], destination: Path, reference: dict):
    import rasterio
    from rasterio.crs import CRS

    destination.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for key, source in sources.items():
        target = destination / f"{key}.tif"
        shutil.copy2(source, target)
        with rasterio.open(target) as raster:
            if not raster.crs or raster.crs != CRS.from_wkt(reference["wkt"]):
                raise ValueError(f"{key}: GeoTIFF reader CRS mismatch")
            if raster.tags().get("vertical_datum") != str(reference["vertical"]):
                raise ValueError(f"{key}: GeoTIFF vertical datum provenance mismatch")
            # Read blocks rather than allocating the full raster merely to validate it.
            supported = False
            for _, window in raster.block_windows(1):
                if raster.dataset_mask(window=window).any():
                    supported = True
                    break
            if not supported and key in ("dsm", "dtm", "ortho"):
                raise ValueError(f"{key}: GeoTIFF has no supported pixels")
        outputs[key] = target
    return outputs


def run_tool(command: list[str], log: Path, output: Path):
    """Execute the installed tool and verify its output, not just its presence."""
    if output.is_file():
        output.unlink()  # Never accept an old product as output of this invocation.
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"External tool exited {result.returncode}; see {log.name}")
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"External tool did not create a nonempty {output.name}")


def export_fbx(obj: Path, path: Path, blender: str):
    executable = shutil.which(blender)
    if not executable:
        raise ExportUnavailable(f"Blender executable is unavailable: {blender}")
    candidates = (Path(__file__).resolve().parents[3] / "scripts" / "blender_export.py",
                  Path(sys.prefix) / "share" / "drishti" / "scripts" / "blender_export.py")
    script = next((path for path in candidates if path.is_file()), None)
    if script is None:
        raise ExportUnavailable("The isolated scripts/blender_export.py is not installed")
    run_tool([executable, "--background", "--factory-startup", "--python", str(script),
              "--", str(obj.resolve()), str(path.resolve())], path.with_suffix(".log"), path)
    with path.open("rb") as stream:
        if not stream.read(23).startswith(b"Kaydara FBX Binary"):
            raise ValueError("Blender output is not a binary FBX file")
    return {"method": "Blender headless isolated subprocess", "source_frame": "projected_local",
            "fbx_axes": "Y-up, -Z-forward; exporter transform baked by Blender"}


def export_optional(kind: str, las: Path, directory: Path, reference: dict):
    if not las.is_file() or not reference:
        raise ExportUnavailable(f"{kind} requires a real projected LAS input")
    if kind == "potree":
        executable = shutil.which("PotreeConverter") or shutil.which("potreeconverter")
        if not executable:
            raise ExportUnavailable("PotreeConverter executable is unavailable")
        output = directory / "metadata.json"
        run_tool([executable, str(las.resolve()), "-o", str(directory.resolve())],
                 directory.parent / "potree.log", output)
        metadata = json.loads(output.read_text(encoding="utf-8"))
        if not metadata.get("points") or not all((directory / file).is_file()
                                                for file in ("octree.bin", "hierarchy.bin")):
            raise ValueError("PotreeConverter did not emit a real point octree")
        return output
    if kind == "3dtiles":
        if reference.get("vertical") != "ellipsoidal":
            raise ExportUnavailable(
                "ECEF 3D Tiles require declared ellipsoidal heights; this adapter does not convert vertical datums"
            )
        if not importlib.util.find_spec("py3dtiles"):
            raise ExportUnavailable("py3dtiles is unavailable")
        output = directory / "tileset.json"
        # py3dtiles executes its converter/reprojection path in an actual subprocess.
        run_tool([sys.executable, "-m", "py3dtiles", "convert", str(las.resolve()),
                  "--out", str(directory.resolve()), "--srs_in", str(reference["epsg"]),
                  "--srs_out", "4978"], directory.parent / "3dtiles.log", output)
        tileset = json.loads(output.read_text(encoding="utf-8"))
        if not tileset.get("asset") or not tileset.get("root"):
            raise ValueError("py3dtiles did not emit a valid tileset structure")
        return output
    raise ExportUnavailable(f"No adapter implemented for optional format {kind}")
