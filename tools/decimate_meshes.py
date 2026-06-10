"""Decimate the heavy robot meshes via Blender's Decimate modifier (Collapse).

GPU runs out of memory with the raw CATIA-derived meshes (chassis_only.stl
has ~1.5M verts). This collapses them to ~10% of the original face count,
preserving silhouette while massively reducing VRAM usage.

Run with:
    "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe" \\
        --background --python c:\\dev\\tetrabot_sim\\tools\\decimate_meshes.py
"""

import sys
from pathlib import Path

import bpy

VISUAL_DIR = Path(r"c:\dev\tetrabot_sim\assets\meshes\visual")

# (filename, target ratio) — keep wheels & lift more detailed than chassis
JOBS = [
    ("chassis_only.stl",   0.10),
    ("wheel.stl",          0.20),
    ("lift_central.stl",   0.25),
]


def decimate(stl_path: Path, ratio: float) -> bool:
    if not stl_path.exists():
        print(f"  SKIP: {stl_path.name} not found")
        return False

    bpy.ops.wm.read_factory_settings(use_empty=True)

    try:
        bpy.ops.wm.stl_import(filepath=str(stl_path))
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=str(stl_path))

    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not objs:
        print(f"  FAIL: no meshes after import")
        return False

    obj = objs[0]
    bpy.context.view_layer.objects.active = obj

    n_before = len(obj.data.polygons)

    mod = obj.modifiers.new(name="dec", type="DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = ratio
    bpy.ops.object.modifier_apply(modifier=mod.name)

    n_after = len(obj.data.polygons)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    try:
        bpy.ops.wm.stl_export(
            filepath=str(stl_path),
            export_selected_objects=True,
            ascii_format=False,
        )
    except AttributeError:
        bpy.ops.export_mesh.stl(
            filepath=str(stl_path),
            use_selection=True,
            ascii=False,
        )

    print(f"  {stl_path.name}: {n_before} -> {n_after} faces "
          f"({n_after / n_before * 100:.1f}%) "
          f"size {stl_path.stat().st_size // 1024} KB")
    return True


def main() -> int:
    ok = 0
    for name, ratio in JOBS:
        print(f"=== {name} (ratio {ratio})")
        if decimate(VISUAL_DIR / name, ratio):
            ok += 1
    print(f"\nDone: {ok}/{len(JOBS)}")
    return 0 if ok == len(JOBS) else 1


if __name__ == "__main__":
    sys.exit(main())
