"""Convert environment STL files to USD for use as static scene props in Isaac Sim.

Run with:
    "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe" \\
        --background --python c:\\dev\\tetrabot_sim\\tools\\stl_to_usd.py
"""

import sys
from pathlib import Path

import bpy

ENV_DIR = Path(r"c:\dev\tetrabot_sim\assets\environment")

JOBS = [
    "A320_AIX.stl",
    "HATRACKS.stl",
    "Lavatory_AIX.stl",
]


def convert(stl_path: Path) -> bool:
    usd_path = stl_path.with_suffix(".usd")
    if not stl_path.exists():
        print(f"  SKIP: {stl_path.name} not found")
        return False

    bpy.ops.wm.read_factory_settings(use_empty=True)

    try:
        bpy.ops.wm.stl_import(filepath=str(stl_path))
    except AttributeError:
        bpy.ops.import_mesh.stl(filepath=str(stl_path))

    objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    print(f"  imported {len(objs)} mesh objects from {stl_path.name}")
    if not objs:
        return False

    # Use Blender's USD exporter
    bpy.ops.wm.usd_export(
        filepath=str(usd_path),
        selected_objects_only=False,
        export_animation=False,
    )
    print(f"  wrote {usd_path.name} ({usd_path.stat().st_size // 1024} KB)")
    return True


def main() -> int:
    print(f"Environment dir: {ENV_DIR}")
    ok = 0
    for name in JOBS:
        print(f"=== {name}")
        if convert(ENV_DIR / name):
            ok += 1
    print(f"\nDone: {ok}/{len(JOBS)} converted")
    return 0 if ok == len(JOBS) else 1


if __name__ == "__main__":
    sys.exit(main())
