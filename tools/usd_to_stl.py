"""Convert TETRABot CATIA-derived USDs to STL for URDF mesh referencing.

Runs headless in Blender. Expects Blender 4.0+ (uses new bpy.ops.wm.stl_export).

Usage (from any cwd):
    "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe" \\
        --background --python c:\\dev\\tetrabot_sim\\tools\\usd_to_stl.py
"""

import sys
from pathlib import Path

import bpy

SOURCE_DIR = Path(r"c:\Users\docht\OneDrive - TUM\Desktop\dominik-dev\docs\reference")
TARGET_DIR = Path(r"c:\dev\tetrabot_sim\assets\meshes\visual")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

JOBS = [
    ("TETRABot_Main_Frame.usd",                "main_frame.stl"),
    ("TETRABot_Mecanum.usd",                   "mecanum_wheel.stl"),
    ("TETRABot_E1_Cover_Assy_AllCATPart.usd",  "cover.stl"),
    ("TETRABot_Camera_dome.usd",               "camera_dome.stl"),
]


def convert(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"  SKIP: {src.name} not found at {src}")
        return False

    bpy.ops.wm.read_factory_settings(use_empty=True)

    try:
        bpy.ops.wm.usd_import(filepath=str(src))
    except Exception as e:
        print(f"  FAIL on USD import: {e}")
        return False

    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        print(f"  WARN: no mesh objects found")
        return False

    bpy.ops.object.select_all(action="DESELECT")
    for o in mesh_objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]

    try:
        bpy.ops.wm.stl_export(
            filepath=str(dst),
            export_selected_objects=True,
            ascii_format=False,
        )
    except AttributeError:
        bpy.ops.export_mesh.stl(
            filepath=str(dst),
            use_selection=True,
            ascii=False,
        )

    print(f"  OK: {len(mesh_objs)} mesh(es) -> {dst.name} ({dst.stat().st_size // 1024} KB)")
    return True


def main() -> int:
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}\n")

    ok_count = 0
    for src_name, dst_name in JOBS:
        print(f"=== {src_name}")
        if convert(SOURCE_DIR / src_name, TARGET_DIR / dst_name):
            ok_count += 1

    print(f"\nDone: {ok_count}/{len(JOBS)} converted.")
    return 0 if ok_count == len(JOBS) else 1


if __name__ == "__main__":
    sys.exit(main())
