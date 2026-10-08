"""Executed by Blender only; never imported by the DRISHTI process."""

import sys
from pathlib import Path


def main():
    import bpy

    args = sys.argv[sys.argv.index("--") + 1:]
    if len(args) != 2:
        raise ValueError("Expected input OBJ and output FBX paths after --")
    source, output = (Path(arg).resolve() for arg in args)
    if not source.is_file() or output.suffix.lower() != ".fbx":
        raise ValueError("A real OBJ input and .fbx output are required")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(source), forward_axis="Y", up_axis="Z")
    else:
        bpy.ops.import_scene.obj(filepath=str(source), axis_forward="Y", axis_up="Z")
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects or not any(len(obj.data.polygons) for obj in objects):
        raise RuntimeError("Blender imported no actual mesh geometry")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    result = bpy.ops.export_scene.fbx(
        filepath=str(output), use_selection=True, object_types={"MESH"},
        axis_forward="-Z", axis_up="Y", use_mesh_modifiers=True,
        bake_anim=False, path_mode="COPY", embed_textures=True,
    )
    if "FINISHED" not in result or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Blender failed to produce a nonempty FBX")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
