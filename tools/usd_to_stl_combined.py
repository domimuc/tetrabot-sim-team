"""Convert TETRABot assembly USD to ONE combined STL with BOTH mecanum drives.

The assembly USD only contains 1 Mecanum mesh; we duplicate it with a Y-mirror
(symmetric across chassis center) so both drive slots are filled.

Run with:
    "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe" \\
        --background --python c:\\dev\\tetrabot_sim\\tools\\usd_to_stl_combined.py
"""

import sys
from pathlib import Path

import bpy

SOURCE = Path(r"c:\Users\docht\OneDrive - TUM\Desktop\dominik-dev\docs\reference\tetrabot_exported.usd")
TARGET = Path(r"c:\dev\tetrabot_sim\assets\meshes\visual\tetrabot_full.stl")
TARGET.parent.mkdir(parents=True, exist_ok=True)


def bbox_dims(o):
    from mathutils import Vector
    corners = [o.matrix_world @ Vector(v) for v in o.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def main() -> int:
    if not SOURCE.exists():
        print(f"FAIL: {SOURCE} not found")
        return 1

    bpy.ops.wm.read_factory_settings(use_empty=True)

    print(f"Importing assembly: {SOURCE.name}")
    bpy.ops.wm.usd_import(filepath=str(SOURCE))

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    print(f"Imported {len(meshes)} mesh(es)\n")

    print("=== Bounding boxes (world space, meters) ===")
    mecanum_obj = None
    for o in meshes:
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox_dims(o)
        print(f"  {o.name:40s} "
              f"X[{xmin:+.3f}, {xmax:+.3f}] "
              f"Y[{ymin:+.3f}, {ymax:+.3f}] "
              f"Z[{zmin:+.3f}, {zmax:+.3f}]")
        if "mecanum" in o.name.lower():
            mecanum_obj = o

    if mecanum_obj is None:
        print("\nWARN: no mesh named *mecanum* — will export as is")
    else:
        print(f"\n=== Duplicating {mecanum_obj.name} ===")
        # Mirror across X=0 plane (heuristic — chassis symmetry plane)
        bpy.ops.object.select_all(action="DESELECT")
        mecanum_obj.select_set(True)
        bpy.context.view_layer.objects.active = mecanum_obj

        bpy.ops.object.duplicate()
        dup = bpy.context.active_object
        # Mirror in X (negate X scale, then Apply)
        dup.scale.x *= -1.0
        bpy.context.view_layer.objects.active = dup
        # bake the negative scale into the mesh
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        # flip normals (since negative scale inverts winding)
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.editmode_toggle()
        print(f"  Created mirrored duplicate: {dup.name}")

    # Export all meshes
    bpy.ops.object.select_all(action="DESELECT")
    final_meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in final_meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = final_meshes[0]
    print(f"\nExporting {len(final_meshes)} mesh(es) -> {TARGET.name}")

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

    print(f"OK: {TARGET.name} ({TARGET.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
