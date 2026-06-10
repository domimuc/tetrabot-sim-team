"""Convert a STEP/STP CAD file to USD via Isaac Sim's bundled HOOPS Converter.

Usage:
    "C:\\isaac-sim\\python.bat" tools\\convert_step_to_usd.py [INPUT.stp [OUTPUT.usd]]

Defaults convert assets/environment/A30X_AllCATPart.stp ->
                  assets/environment/A30X_AllCATPart.usd

Why this script: Isaac Sim 5.0 ships the HOOPS Exchange CAD converter as
the extension `omni.kit.converter.hoops_core`. It can read STEP/STP, CATIA
V5/V6, NX, Parasolid, SolidWorks, Inventor, JT, Rhino, Creo, ACIS, IFC,
plus mesh formats. The extension exposes
`omni.converter.hoops.Converter(options).convert(in, out, args)` once
loaded — but the extension only loads after a Kit (SimulationApp)
process is up, so we bootstrap a headless SimApp first.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap headless Kit so the converter extension is available.
from isaacsim import SimulationApp
_kit = SimulationApp({"headless": True})

ASSETS = Path(r"c:\dev\tetrabot_sim\assets\environment")
DEFAULT_INPUT = ASSETS / "A30X_AllCATPart.stp"
DEFAULT_OUTPUT = ASSETS / "A30X_AllCATPart.usd"


def main() -> int:
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        input_path = DEFAULT_INPUT
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_path.with_suffix(".usd")

    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}")
        return 1
    print(f"Converting:")
    print(f"  in : {input_path}  ({input_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  out: {output_path}")

    # Enable the HOOPS converter extension. set_extension_enabled_immediate
    # registers the converter classes synchronously, but we still pump a few
    # update ticks to let DLLs resolve.
    import omni.kit.app
    mgr = omni.kit.app.get_app().get_extension_manager()
    ext = "omni.kit.converter.hoops_core"
    if not mgr.is_extension_enabled(ext):
        print(f"Enabling {ext}...")
        mgr.set_extension_enabled_immediate(ext, True)
        for _ in range(8):
            _kit.update()
    if not mgr.is_extension_enabled(ext):
        print(f"ERROR: {ext} did not enable")
        return 1

    # Now the converter is callable.
    import omni.converter.hoops as ohoops
    from omni.kit.converter.hoops_core import HoopsOptions

    options = HoopsOptions()
    # Defaults are sane for our use: meters per unit = 0.01 (cm), upAxis = Y.
    # We override upAxis to Z (Isaac Sim convention) so the cabin imports
    # the right way up out of the box.
    options.iUpAxis = 2          # 0=X, 1=Y, 2=Z
    options.dMetersPerUnit = 1.0  # treat input as metres directly

    converter = ohoops.Converter(options)
    result = converter.convert(str(input_path), str(output_path), {})
    print(f"Converter result: {result}")

    if not output_path.exists():
        print(f"ERROR: output USD not produced at {output_path}")
        return 1
    print(f"OK wrote {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    rc = main()
    _kit.close()
    sys.exit(rc)
