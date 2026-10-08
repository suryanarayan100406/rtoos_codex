"""Point artifacts with real confidence, provenance, normals, and float64 XYZ."""

from pathlib import Path

import numpy as np

from drishti.errors import InputError

POINT_DTYPES = {"points": (np.float64, (3,)), "colors": (np.uint8, (3,)),
                "normals": (np.float64, (3,)), "view_direction": (np.float64, (3,)),
                "confidence": (np.float32, ()),
                "inferred": (np.bool_, ()), "observations": (np.int64, ())}


def create_point_arrays(directory, count):
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    return {name: np.lib.format.open_memmap(directory / f"{name}.npy", mode="w+",
                                            dtype=dtype, shape=(count, *shape))
            for name, (dtype, shape) in POINT_DTYPES.items()}


def save_pointcloud(path, arrays):
    count = len(arrays["points"])
    if any(len(a) != count for a in arrays.values()):
        raise InputError("Pointcloud attribute lengths disagree")
    if arrays["points"].dtype != np.float64:
        raise InputError("Pointcloud XYZ must be float64 local metres")
    # NumPy writes NPY members in bounded buffers; mmap arrays aren't concatenated.
    np.savez(path, **arrays)


def write_ply(path, arrays, chunk_size=100000):
    """Binary PLY fields describe actual attributes, including inferred status."""
    count = len(arrays["points"])
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1"),
                      ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                      ("confidence", "<f4"), ("inferred", "u1"), ("observations", "<u4")])
    header = ["ply", "format binary_little_endian 1.0", "comment local-origin XYZ metres",
              f"element vertex {count}", "property double x", "property double y", "property double z",
              "property uchar red", "property uchar green", "property uchar blue", "property float nx",
              "property float ny", "property float nz", "property float confidence", "property uchar inferred",
              "property uint observations", "end_header", ""]
    with Path(path).open("wb") as stream:
        stream.write("\n".join(header).encode("ascii"))
        for begin in range(0, count, chunk_size):
            end = min(count, begin+chunk_size)
            block = np.empty(end-begin, dtype)
            for names, key in ((["x", "y", "z"], "points"), (["red", "green", "blue"], "colors"),
                               (["nx", "ny", "nz"], "normals")):
                for channel, name in enumerate(names):
                    block[name] = arrays[key][begin:end, channel]
            block["confidence"] = arrays["confidence"][begin:end]
            block["inferred"] = arrays["inferred"][begin:end]
            block["observations"] = arrays["observations"][begin:end] if "observations" in arrays else 1
            block.tofile(stream)
