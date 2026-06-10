"""Re-export the assembly USD WITHOUT the Mecanum drive.

Output: chassis_only.stl (Main_Frame + Cover + Camera_dome).
This becomes the chassis_link visual in URDF; the mecanum is then composed
from separate wheel.stl + lift_central.stl visuals on dedicated links.
"""

import sys
from pathlib import Path

import bpy

SOURCE = Path(r"c:\Users\docht\OneDrive - TUM\Desktop\dominik-dev\docs\reference\tetrabot_exported.usd")
TARGET = Path(r"c:\dev\tetrabot_sim\assets\meshes\visual\chassis_only.stl")
LOG_FILE = Path(r"c:\dev\tetrabot_sim\logs\extract_chassis_only.txt")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def emit(s=""):
    sys.__stdout__.write(s + "\n")
    sys.__stdout__.flush()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(s + "\n")


def bbox(o):
    from mathutils import Vector

    corners = [o.matrix_world @ Vector(v) for v in o.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return ((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs)))


def main() -> int:
    LOG_FILE.write_text("", encoding="utf-8")  # clear log

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=str(SOURCE))

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    emit(f"Imported {len(meshes)} meshes")

    # Identify mecanum: highest Y_max (CATIA Y-up, so highest Y_max is closest
    # to bottom of upright robot) AND most verts (rollers)
    mecanum = max(meshes, key=lambda o: (bbox(o)[1][1], len(o.data.vertices)))
    emit(f"Identified mecanum to delete: {mecanum.name}")

    # Delete it
    bpy.ops.object.select_all(action="DESELECT")
    mecanum.select_set(True)
    bpy.ops.object.delete()

    remaining = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    emit(f"Remaining meshes: {len(remaining)}")
    for m in remaining:
        bb = bbox(m)
        emit(f"  {m.name}: X[{bb[0][0]:+.2f},{bb[0][1]:+.2f}] "
             f"Y[{bb[1][0]:+.2f},{bb[1][1]:+.2f}] Z[{bb[2][0]:+.2f},{bb[2][1]:+.2f}]")

    # Export
    bpy.ops.object.select_all(action="DESELECT")
    for m in remaining:
        m.select_set(True)
    bpy.context.view_layer.objects.active = remaining[0]

    try:
        bpy.ops.wm.stl_export(
            filepath=str(TARGET),
            export_selected_objects=True,
            ascii_format=False,
        )
    except AttributeError:
        bpy.ops.export_mesh.stl(
            filepath=str(TARGET),
            use_selection=True,
            ascii=False,
        )

    emit(f"Wrote {TARGET.name} ({TARGET.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
