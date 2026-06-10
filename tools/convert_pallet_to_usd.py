"""Split Euro_Paletten_Test_2.stl into pallet + boxes and write each as USD.

Source STL (mm units) contains 9 connected components:
  part 0  -> the pallet (1040x2160x144 mm)
  part 1  -> box A     (756x540x393 mm)
  part 5  -> box B     (704x454x393 mm)
  parts 2,3,4,6,7,8 -> degenerate slivers (<10 verts, mesh artefacts) — dropped

This script writes three USD files into assets/environment/:
  euro_pallet.usd   — pallet only, scaled to metres
  pallet_box_a.usd  — first box
  pallet_box_b.usd  — second box

Each USD has a single root Xform "geom" containing a UsdGeom.Mesh.
The mesh is centred at the prim's origin (we re-centre vertices) so the
caller can place each prim at its desired world position cleanly.

Run via Isaac Sim's bundled python — but pxr is only importable AFTER
SimulationApp boots, so we bootstrap a headless SimApp first:
    "C:\\isaac-sim\\python.bat" tools\\convert_pallet_to_usd.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

# Bootstrap a headless Kit so pxr.UsdGeom is importable.
from isaacsim import SimulationApp
_kit = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402  (must come after SimApp init)

ASSETS = Path(r"c:\dev\tetrabot_sim\assets\environment")
SRC_STL = ASSETS / "Euro_Paletten_Test_2.stl"

# Map source-component-index -> output USD name. Anything not in this
# mapping is dropped (degenerate slivers).
COMPONENT_MAP = {
    0: "euro_pallet.usd",
    1: "pallet_box_a.usd",
    5: "pallet_box_b.usd",
}

MM_TO_M = 0.001


def write_mesh_usd(out_path: Path, vertices_m: np.ndarray, faces: np.ndarray) -> None:
    """Write a single-mesh USD with Y-up converted to Z-up.

    Vertices are expected in metres. We re-centre them around origin so that
    the resulting prim's xform translate places it intuitively.
    """
    # Recenter at origin (XY centre, Z floor at 0 so it stands on a plane).
    bounds_min = vertices_m.min(axis=0)
    bounds_max = vertices_m.max(axis=0)
    centre_xy = (bounds_min[:2] + bounds_max[:2]) * 0.5
    floor_z = bounds_min[2]
    offset = np.array([centre_xy[0], centre_xy[1], floor_z], dtype=np.float64)
    centred = vertices_m - offset

    # Create stage with Z-up convention to match Isaac Sim default.
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/geom")
    mesh = UsdGeom.Mesh.Define(stage, "/geom/Mesh")

    mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in centred])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))      # all triangles
    mesh.CreateFaceVertexIndicesAttr(faces.flatten().tolist())

    # Normals — recompute from triangulation for clean shading.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    # Mark the root xform as the default-prim so reference pickups work.
    stage.SetDefaultPrim(root.GetPrim())

    stage.GetRootLayer().Save()
    extents_m = bounds_max - bounds_min
    print(f"  wrote {out_path.name}: {len(centred):,} verts, "
          f"{len(faces):,} tris, extents={extents_m.round(3)} m")


def main() -> int:
    if not SRC_STL.exists():
        print(f"FAIL: source not found: {SRC_STL}")
        return 1
    print(f"Loading {SRC_STL.name} ({SRC_STL.stat().st_size / 1024:.0f} KB)")

    mesh = trimesh.load(SRC_STL, force="mesh")
    parts = mesh.split(only_watertight=False)
    print(f"  {len(parts)} connected components found")

    n_written = 0
    for idx, part in enumerate(parts):
        if idx not in COMPONENT_MAP:
            print(f"  drop component {idx} (verts={len(part.vertices)}, "
                  f"extents={part.extents.round(1)}) — not in map")
            continue
        out_name = COMPONENT_MAP[idx]
        out_path = ASSETS / out_name
        verts_m = np.asarray(part.vertices, dtype=np.float64) * MM_TO_M
        faces = np.asarray(part.faces, dtype=np.int32)
        write_mesh_usd(out_path, verts_m, faces)
        n_written += 1

    print(f"\nWrote {n_written}/{len(COMPONENT_MAP)} USD files into {ASSETS}")
    return 0 if n_written == len(COMPONENT_MAP) else 1


if __name__ == "__main__":
    rc = main()
    _kit.close()
    sys.exit(rc)
