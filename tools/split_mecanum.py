"""Split mecanum_wheel.stl into clean per-component STLs by X-coordinate clustering.

The mecanum mesh has wheel hubs at CATIA X=+/-0.68 with 12 rollers ringing each.
We cluster everything within +/- 0.45 of those X centers as "left wheel" / "right wheel",
and treat the remainder (centered around X=0) as the lift mechanism.

Outputs (all in CATIA-native frame, centered on their own pivot for URDF placement):
  wheel.stl              one full wheel (hub + 12 rollers + rim), origin at hub
  lift_central.stl       central lift column / spindle, origin at original CATIA origin
"""

import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "assets" / "meshes" / "visual" / "mecanum_wheel.stl"
TARGET_DIR = REPO / "assets" / "meshes" / "visual"

LEFT_WHEEL_X = -0.68
RIGHT_WHEEL_X = +0.68
WHEEL_X_HALF_WIDTH = 0.32  # generous: includes hub, rim, rollers


def main() -> int:
    print(f"Loading {SOURCE.name}")
    mesh = trimesh.load(str(SOURCE))
    print(f"  {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    components = mesh.split(only_watertight=False)
    print(f"  Split into {len(components)} components")

    left_parts: list[trimesh.Trimesh] = []
    right_parts: list[trimesh.Trimesh] = []
    central_parts: list[trimesh.Trimesh] = []

    for c in components:
        ctr = (c.bounds[0] + c.bounds[1]) / 2
        if abs(ctr[0] - LEFT_WHEEL_X) < WHEEL_X_HALF_WIDTH:
            left_parts.append(c)
        elif abs(ctr[0] - RIGHT_WHEEL_X) < WHEEL_X_HALF_WIDTH:
            right_parts.append(c)
        else:
            central_parts.append(c)

    print(f"\nClustering result:")
    print(f"  left wheel:   {len(left_parts)} parts")
    print(f"  right wheel:  {len(right_parts)} parts")
    print(f"  central lift: {len(central_parts)} parts")

    # Build the LEFT wheel mesh, recenter on its own hub
    left_combined = trimesh.util.concatenate(left_parts)
    left_center = (left_combined.bounds[0] + left_combined.bounds[1]) / 2
    left_combined.apply_translation(-left_center)
    out_wheel = TARGET_DIR / "wheel.stl"
    left_combined.export(str(out_wheel))
    print(f"\nWrote {out_wheel.name}: {out_wheel.stat().st_size // 1024} KB")
    print(f"  {len(left_combined.vertices)} verts, original hub at CATIA {tuple(left_center.round(3))}")

    # Lift central — keep at its native CATIA origin (no recenter)
    if central_parts:
        lift_combined = trimesh.util.concatenate(central_parts)
        out_lift = TARGET_DIR / "lift_central.stl"
        lift_combined.export(str(out_lift))
        print(f"\nWrote {out_lift.name}: {out_lift.stat().st_size // 1024} KB")
        lift_center = (lift_combined.bounds[0] + lift_combined.bounds[1]) / 2
        print(f"  {len(lift_combined.vertices)} verts, center at CATIA {tuple(lift_center.round(3))}")

    # Print URDF placement helpers (CATIA -> URDF after rpy=-1.5708 0 0):
    # CATIA X stays X, CATIA Y becomes -Z, CATIA Z becomes +Y.
    print("\n=== URDF placement (post rpy=-1.5708 0 0 rotation) ===")
    print("Wheel hub centers in URDF frame (relative to chassis_link):")
    for label, x in [("LEFT", LEFT_WHEEL_X), ("RIGHT", RIGHT_WHEEL_X)]:
        # CATIA Y of hub is +0.144, Z is 0. After rotation:
        # URDF X = X, Y = Z = 0, Z = -Y = -0.144
        print(f"  {label:5s}: URDF({x:+.3f}, {0:+.3f}, {-0.144:+.3f})")
    print("\nNote: this is for ONE axle. To get the SECOND axle, mirror in URDF Y")
    print("      (because the second axle would be on the other side of the chassis).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
