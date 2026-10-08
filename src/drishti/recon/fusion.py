"""Bounded observation generation and disk-partitioned, external voxel reduction."""

import sqlite3
from pathlib import Path

import numpy as np

from drishti.errors import InputError
from drishti.io.pointcloud import create_point_arrays, save_pointcloud, write_ply
from drishti.recon.depth import pixel_rays

OBSERVATION = np.dtype([("point", "<f8", (3,)), ("color", "u1", (3,)),
                        ("normal", "<f8", (3,)), ("view", "<f8", (3,)),
                        ("confidence", "<f4"), ("inferred", "u1")])
AGGREGATE = np.dtype([("kx", "<i8"), ("ky", "<i8"), ("kz", "<i8"),
                      ("point", "<f8", (3,)), ("color", "<f8", (3,)),
                      ("normal", "<f8", (3,)), ("view", "<f8", (3,)), ("weight", "<f8"),
                      ("confidence", "<f8"), ("inferred", "u1"), ("observations", "<i8")])
KEY = np.dtype([("kx", "<i8"), ("ky", "<i8"), ("kz", "<i8")])
KEY_NAMES = ("kx", "ky", "kz")


def key_array(records):
    result = np.empty(len(records), KEY)
    for field in KEY_NAMES:
        result[field] = records[field]
    return result


def reduce_aggregates(records):
    if not len(records):
        return records
    records = records[np.argsort(key_array(records), kind="stable")]
    keys = key_array(records)
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
    result = np.empty(len(starts), AGGREGATE)
    for name in KEY_NAMES:
        result[name] = records[name][starts]
    for name in ("point", "color", "normal", "view", "weight", "confidence", "observations"):
        result[name] = np.add.reduceat(records[name], starts)
    result["inferred"] = np.maximum.reduceat(records["inferred"], starts)
    return result


def aggregate_observations(records, voxel_m):
    result = np.empty(len(records), AGGREGATE)
    keys = np.floor(records["point"]/voxel_m).astype(np.int64)
    for channel, name in enumerate(KEY_NAMES):
        result[name] = keys[:, channel]
    weight = records["confidence"].astype(np.float64)
    for name in ("point", "color", "normal", "view"):
        result[name] = records[name] * weight[:, None]
    result["weight"] = weight
    result["confidence"] = weight**2
    result["inferred"] = records["inferred"]
    result["observations"] = 1
    return reduce_aggregates(result)


def merge_runs(first, second, output, chunk_size):
    """Merge two globally sorted unique runs, without loading a tile into memory."""
    count = 0
    with Path(first).open("rb") as a, Path(second).open("rb") as b, Path(output).open("wb") as target:
        left, right = np.empty(0, AGGREGATE), np.empty(0, AGGREGATE)
        eof_a = eof_b = False
        while True:
            if not len(left) and not eof_a:
                left = np.fromfile(a, AGGREGATE, count=chunk_size)
                eof_a = not len(left)
            if not len(right) and not eof_b:
                right = np.fromfile(b, AGGREGATE, count=chunk_size)
                eof_b = not len(right)
            if not len(left) and not len(right):
                break
            if not len(left):
                reduced, right = right, np.empty(0, AGGREGATE)
            elif not len(right):
                reduced, left = left, np.empty(0, AGGREGATE)
            else:
                lk, rk = key_array(left), key_array(right)
                cutoff = lk[-1] if tuple(lk[-1]) < tuple(rk[-1]) else rk[-1]
                ai = np.searchsorted(lk, cutoff, side="right")
                bi = np.searchsorted(rk, cutoff, side="right")
                reduced = reduce_aggregates(np.concatenate((left[:ai], right[:bi])))
                left, right = left[ai:], right[bi:]
            reduced.tofile(target)
            count += len(reduced)
    return count


def depth_observations(depth, confidence, rgb, excluded, camera, cfg):
    """Yield bounded float64 XYZ/normal batches, excluding unsupported depths."""
    h, w = depth.shape
    if confidence.shape != (h, w) or excluded.shape != (h, w) or rgb.shape != (h, w, 3):
        raise InputError("Fusion image/depth/confidence/mask shape mismatch")
    stride = cfg.pixel_stride
    nx, ny = len(range(0, w, stride)), len(range(0, h, stride))
    R = np.asarray(camera["R_cw"], np.float64)
    center = np.asarray(camera["center"], np.float64)
    for begin in range(0, nx*ny, cfg.max_points_per_chunk):
        index = np.arange(begin, min(nx*ny, begin+cfg.max_points_per_chunk))
        x, y = (index % nx)*stride, (index // nx)*stride
        good = (x > 0) & (x < w-1) & (y > 0) & (y < h-1)
        x, y = x[good], y[good]
        z, c = depth[y, x], confidence[y, x]
        good = np.isfinite(z) & (z > 0) & (z <= cfg.max_depth_m)
        good &= np.isfinite(c) & (c >= cfg.min_confidence) & (c > 0) & ~excluded[y, x].astype(bool)
        x, y, z, c = x[good], y[good], z[good], c[good]
        if not len(x):
            continue
        uv = np.column_stack((x, y))
        xyz = pixel_rays(uv, camera) * z[:, None]
        neighbors, valid = [], np.ones(len(x), bool)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            zn = depth[y+dy, x+dx]
            cn = confidence[y+dy, x+dx]
            valid &= np.isfinite(zn) & (zn > 0) & (zn <= cfg.max_depth_m) & ~excluded[y+dy, x+dx].astype(bool)
            valid &= np.isfinite(cn) & (cn >= cfg.min_confidence) & (cn > 0)
            valid &= np.abs(zn-z)/z <= cfg.normal_max_relative_jump
            neighbors.append(pixel_rays(uv+np.array([dx, dy]), camera)*zn[:, None])
        normals = np.cross(neighbors[1]-neighbors[0], neighbors[3]-neighbors[2])
        norm = np.linalg.norm(normals, axis=1)
        valid &= np.isfinite(norm) & (norm > np.finfo(float).eps)
        normals /= np.maximum(norm[:, None], np.finfo(float).eps)
        normals[np.einsum("ij,ij->i", normals, -xyz) < 0] *= -1
        records = np.empty(int(np.count_nonzero(valid)), OBSERVATION)
        records["point"] = xyz[valid] @ R + center
        records["normal"] = normals[valid] @ R
        records["view"] = center-records["point"]
        records["view"] /= np.linalg.norm(records["view"], axis=1)[:, None]
        records["color"] = rgb[y[valid], x[valid]]
        records["confidence"] = c[valid]
        records["inferred"] = False
        if len(records):
            yield records


class TiledFusion:
    """Raw tiles + sorted runs + bounded pair merges + mmap artifact assembly.

    Even a tile larger than RAM is never loaded wholesale. A sqlite catalog
    avoids an unbounded Python tile dictionary; run compaction keeps O(log N)
    paths. The working-set bound is proportional to max_points_per_chunk,
    not scene size. Disk capacity and Poisson meshing have separate limits.
    """

    def __init__(self, directory, voxel_m, tile_m, max_points_per_chunk):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.voxel_m = float(voxel_m)
        self.tile_voxels = max(1, int(round(tile_m/voxel_m)))
        self.chunk_size = int(max_points_per_chunk)
        if self.voxel_m <= 0 or self.chunk_size < 4:
            raise InputError("Fusion needs a positive voxel size and chunk limit >=4")
        self.db = sqlite3.connect(self.directory / "catalog.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS tiles(name TEXT PRIMARY KEY, observations INTEGER, points INTEGER DEFAULT 0)")
        self.db.execute("DELETE FROM tiles")
        # Caller uses a fresh work directory for each execution.
        self.input_count = 0

    def add(self, records):
        if len(records) > self.chunk_size:
            raise InputError("Fusion input batch exceeds the configured bound")
        if not len(records):
            return
        if not all(np.isfinite(records[name]).all() for name in ("point", "normal", "view", "confidence")):
            raise InputError("Fusion observations contain non-finite geometry/support")
        if np.any(records["confidence"] <= 0) or np.any(records["confidence"] > 1):
            raise InputError("Fusion observations must be finite and supported")
        tile = np.floor_divide(np.floor(records["point"]/self.voxel_m).astype(np.int64), self.tile_voxels)
        order = np.lexsort((tile[:, 2], tile[:, 1], tile[:, 0]))
        tile, records = tile[order], records[order]
        boundaries = np.r_[0, np.flatnonzero(np.any(tile[1:] != tile[:-1], axis=1))+1, len(tile)]
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            name = "_".join(str(int(v)) for v in tile[begin])
            with (self.directory / f"{name}.raw").open("ab") as stream:
                records[begin:end].tofile(stream)
            self.db.execute("INSERT INTO tiles(name, observations) VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET observations=observations+excluded.observations", (name, int(end-begin)))
        self.db.commit()
        self.input_count += len(records)

    def _reduce_tile(self, name):
        levels: dict[int, Path] = {}
        counter = 0
        block_size = max(1, self.chunk_size//4)
        raw = self.directory / f"{name}.raw"
        with raw.open("rb") as stream:
            while True:
                records = np.fromfile(stream, OBSERVATION, count=block_size)
                if not len(records):
                    break
                current = self.directory / f"{name}.run{counter}"
                counter += 1
                aggregate_observations(records, self.voxel_m).tofile(current)
                level = 0
                while level in levels:
                    previous = levels.pop(level)
                    output = self.directory / f"{name}.run{counter}"
                    counter += 1
                    merge_runs(previous, current, output, block_size)
                    previous.unlink()
                    current.unlink()
                    current = output
                    level += 1
                levels[level] = current
        final = None
        for current in levels.values():
            if final is None:
                final = current
            else:
                output = self.directory / f"{name}.run{counter}"
                counter += 1
                merge_runs(final, current, output, block_size)
                final.unlink()
                current.unlink()
                final = output
        result = self.directory / f"{name}.reduced"
        count = 0
        if final is not None:
            final.rename(result)
            count = result.stat().st_size//AGGREGATE.itemsize
        raw.unlink()
        self.db.execute("UPDATE tiles SET points=? WHERE name=?", (count, name))
        return count

    def finish(self, output_dir):
        total = 0
        for (name,) in self.db.execute("SELECT name FROM tiles ORDER BY name"):
            total += self._reduce_tile(name)
        self.db.commit()
        arrays = create_point_arrays(Path(output_dir)/"arrays", total)
        offset = 0
        for name, _count in self.db.execute("SELECT name, points FROM tiles ORDER BY name"):
            with (self.directory/f"{name}.reduced").open("rb") as stream:
                while True:
                    block = np.fromfile(stream, AGGREGATE, count=max(1, self.chunk_size//4))
                    if not len(block):
                        break
                    end = offset+len(block)
                    weights = block["weight"]
                    arrays["points"][offset:end] = block["point"]/weights[:, None]
                    arrays["colors"][offset:end] = np.clip(np.rint(block["color"]/weights[:, None]), 0, 255).astype(np.uint8)
                    normals = block["normal"]
                    view = block["view"]
                    normals[np.einsum("ij,ij->i", normals, view) < 0] *= -1
                    lengths = np.linalg.norm(normals, axis=1)
                    arrays["normals"][offset:end] = normals/np.maximum(lengths[:, None], np.finfo(float).eps)
                    arrays["view_direction"][offset:end] = view/np.maximum(np.linalg.norm(view, axis=1)[:, None], np.finfo(float).eps)
                    arrays["confidence"][offset:end] = np.clip(block["confidence"]/weights, 0, 1)
                    arrays["inferred"][offset:end] = block["inferred"].astype(bool)
                    arrays["observations"][offset:end] = block["observations"]
                    offset = end
        for array in arrays.values():
            array.flush()
        save_pointcloud(Path(output_dir)/"dense.npz", arrays)
        write_ply(Path(output_dir)/"dense.ply", arrays, max(1, self.chunk_size//4))
        tile_count = self.db.execute("SELECT COUNT(*) FROM tiles").fetchone()[0]
        self.db.close()
        return arrays, {"points": total, "observations": self.input_count, "tiles": tile_count,
                        "method": "disk_partition_external_merge_voxel_weighted_point_fusion",
                        "working_set_limit_observations": self.chunk_size,
                        "effective_tile_m": self.tile_voxels*self.voxel_m,
                        "xyz_dtype": "float64", "normal_orientation": "toward_weighted_observing_camera_directions"}
