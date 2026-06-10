"""Create a referenceable wrapper for the A30X cabin USD.

Isaac Lab's AddReference command rejects A30X_AllCATPart.usd directly (CATIA
export trips _is_reference_valid) even though it has a valid defaultPrim and
loads via a raw AddReference. A thin wrapper — one Xform 'Cabin' (defaultPrim)
that references the cabin via a raw composition arc — references cleanly.
"""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom  # noqa: E402

ENV = "/workspace/tetrabot-sim/assets/environment"
WRAP = f"{ENV}/cabin_wrapped.usd"

w = Usd.Stage.CreateNew(WRAP)
cabin = UsdGeom.Xform.Define(w, "/Cabin")
w.SetDefaultPrim(cabin.GetPrim())
cabin.GetPrim().GetReferences().AddReference("./A30X_AllCATPart.usd")
w.GetRootLayer().Save()

wv = Usd.Stage.Open(WRAP)
wdp = wv.GetDefaultPrim()
nchild = len(list(cabin.GetPrim().GetChildren()))
print(f"WRAPPER_OK default={wdp.GetPath()} valid={bool(wdp and wdp.IsValid())} cabin_children={nchild}")

app.close()
