# tetrabot_sim — Project Notes

Internes Dev-Doku-Layer für mich selbst und alle, die später am Repo arbeiten.
Hier stehen die Sachen, die aus Code allein nicht ableitbar sind: gelockte
Werte mit Begründung, Tech-Debt, Roadmap, Never-Touch-Liste.

Für Onboarding zuerst [`README.md`](README.md) lesen.

## Standing Orders

### 1. `GROUND_PLANE_Z = 0.79` ist immutabel

Gelockt 2026-05-07. Definiert in `tools/launch.py`. Aligned mit der authored
Passenger-Floor-Höhe der A30X-Kabine. Jede Drift hier reißt G1-Spawn-Z,
Pallet-Aufstandshöhe und Cabin-Translate-Z mit. CLI-Flag `--cabin-floor-z`
bleibt für Diagnose-Tests (drop-mode, ground-shift) überschreibbar — der
Default in der Konstante darf nicht driften.

### 2. `docking_unit` Visual: NIE Box/Plate — IMMER CAD-Mesh

Gelockt 2026-05-11 nach mehreren Iterationen. Der `docking_unit`-Link in
`urdf/tetrabot.urdf` darf niemals eine `<box>` oder `<plane>` als visual
geometry bekommen. Erlaubt sind ausschließlich:
- echte CAD-STL-Meshes aus `assets/meshes/visual/` (`lift_central.stl`,
  `lift_axle.stl`, oder ein dedicated Engagement-Mesh), oder
- kein Visual (Link nur tf-Anchor) — die sichtbare Engagement-Geometrie kommt
  dann über `chassis_only.stl`.

Collision-Geometrie auf `docking_unit` ist davon unabhängig — Box-Collider ist
okay, Collider sind unsichtbar.

## Tech-Debt

### `_physics_view`-Workaround in `_ensure_art_initialized()`

Isaac Sim löscht das `_physics_view`-Attribute beim sim_view-Teardown statt es
auf `None` zu setzen — wir setzen es manuell vor Re-Init, damit
`art.initialize()` nicht intern `AttributeError` wirft. Mit zukünftigem
Isaac-Sim-Update vermutlich obsolet; bis dahin nicht entfernen.

### Cabin-Collision im Default OFF

Der CATIA-Export hat 53 Mesh-Prims mit Triangle-Collidern → PhysX-GPU-Solver-
Explosionen wenn Robots drin spawnen. Re-Enable über `--cabin-collision`, aber
dann **muss** `--cabin-floor-z` auf ~0.252 gesetzt werden (unter SPANTENMODELL-
Beams).

### Pallet-Boxen resetten sich nicht bei Skript-Start/Stop

Beim Re-Run der Demo (Stop → Play) bleiben Box-Posen vom letzten Lauf hängen.
Fix ist klein (Boxen explizit auf Spawn-Pose zurücksetzen vor jedem Szenario-
Restart), war zeitlich nicht mehr drin.

### Isaac Lab + AVP-Teleop (`g1_handover/`) nicht fertig integriert

Das `g1_handover/`-Paket registriert die Gym-Tasks `Isaac-PickTurn-G1-Box-Abs-v0`
und `Isaac-PickTurn-G1-CabinHandover-Abs-v0`, hat `record_handover.py` (phasen-
gated HDF5-Recording) und `stream_hold.py` (View-Harness). Scripted on-box läuft
und exportiert valide HDF5s. **Was fehlt**:

- Echte AVP-Hand-Tracking-Aufzeichnung — braucht CloudXR-Runtime in einem
  separaten Docker-Container auf RTX-PRO-6000-/L40-Klasse-GPU. Die bisherige
  Brev-Box konnte das nicht hosten (kein Docker, falsche GPU-Klasse).
- Hybrid-Integration: aufgezeichneten G1-Clip in die `tools/launch.py`-Demo
  re-injecten (state-getriggert) für einen durchgehenden „Bots → Übergabe →
  G1 hebt"-Lauf.
- Wrapper-Skripte für Isaac Labs `record_demos.py` / `replay_demos.py` die das
  `g1_handover`-Modul automatisch importieren.

Detail-Status zu jedem deferred Punkt steht in der Roadmap unten.

## Setup-Environments (Historie)

Das Repo lief über die Projektlaufzeit auf drei Maschinen-Typen — die Start-
Skripte tragen Spuren davon:

- **Phase 1, Windows 11 lokal** (Tag 1 bis ~Tag 10): Isaac Sim 5.0 unter
  `C:\isaac-sim\` auf RTX 5060 Ti. Start über `launch.bat` (setzt Vulkan-Layer-
  Disable, ROS2-Humble-Pfade, PYTHONEXE-Workaround für Kit-107.3-Driver-Issue).
  Reiner Standalone-Isaac-Sim, kein Isaac Lab.
- **Phase 2, NVIDIA Brev Cloud-Instanz** (ab ~Tag 10): Container-Isaac-Sim
  unter `/isaac-sim/`, persistentes `/workspace/`. Headless → WebRTC-Livestream
  in Browser. Hier kam Isaac Lab 2.3 für `g1_handover/` dazu.
- **Phase 3, Office-Ubuntu** (übergabebereit): gleiche Codebasis, Setup über
  `scripts/setup/setup_ubuntu.sh`.

Setup-Skripte in [`scripts/setup/`](scripts/setup/) verstehen alle drei
Varianten. Per-Maschine-Konfiguration meist via `ISAAC_SIM_PATH`-Env-Var.

## Roadmap (was als nächstes anzugehen wäre)

Sortiert nach Wert für die Demo, nicht nach Aufwand.

1. **Echte AVP-Aufzeichnung + Hybrid-Replay**. Inkrement-Plan:
   1. `record_handover.py` scripted → HDF5 → `replay_demos.py` rundläuft (auf
      jeder Maschine verifizierbar, billigste Validation).
   2. CloudXR-Runtime auf einer Team-Workstation aufsetzen (RTX PRO 6000 oder
      L40, Docker, NVIDIA Container Toolkit). NVIDIA-Anleitung:
      <https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html>.
   3. AVP-Client „Isaac XR Teleop Sample Client" + VPN/Netz aufbauen.
   4. Aufzeichnung via `--teleop_device handtracking`. Aufgenommene HDF5s sind
      die Basis für Isaac-Lab-Mimic und Imitation-Learning.
2. **Pallet-Pocket-Engagement realistisch** (siehe Known Limitations unten).
3. **Trainierte Policies**: `tools/train_g1_locomotion.py` ausfüllen
   (IsaacLab + RSL-RL + 9-dim Obs / 3-dim Act); exportieren als ONNX/
   TorchScript; via `--controller rl --rl-weights …` einbinden.
4. **Echte Kabinen-Collision + physikalisches Absetzen** statt Kinematic-Carry.
   Vorbedingung: Convex-Decomposition-Collider für die Kabine statt
   Triangle-Mesh (sonst PhysX-Explosionen).
5. **Pallet-Box-Reset** beim Re-Run (siehe Tech-Debt — klein).
6. **LFS-Migration**: aktuell 3 Files >50 MB
   (`TETRABot_Mecanum.stl` 56 MB, `TETRABot_Camera_dome.stl` 50 MB,
   `tetrabot_base.usd` 50 MB). Unter GitHubs 100-MB-Hard-Limit, aber LFS
   bevor das nächste große Asset reinkommt.
7. **CI / automatisierte Tests**: aktuell nur manueller headless Smoke-Test
   (`--headless --frames N`, exit 0). Für Produkt-Reife fehlen reproduzierbare
   Test-Gates.

## Known Limitations (deferred)

### Pallet-Pocket-Engagement nicht visuell realistisch

**Status**: Bots erreichen ihre Pickup-Positionen am Palettenrand, fahren aber
nicht in die Fork-Pockets der Europalette. Mehrere Iterationen versucht
(FilteredPairsAPI, Pickup-Position-Tuning ±0.55 vs ±0.69, docking_unit-Visual
mit verschiedenen CAD-Meshes, cover/chassis-Body-Split) — keine produzierte
das gewünschte „Vorsprung-fährt-in-Pocket-und-hebt"-Verhalten.

**Workaround**: Bots docken am Palettenrand, heben via Kinematic-Carry. Visuell
unbefriedigend, funktionell stabil.

**Was bei Re-Visit zu untersuchen wäre**: Vorsprung-Z-Höhe vs.
Pocket-Eingangs-Z-Höhe verifizieren; Approach-Vector mit niedrigerem
Lift-Down-Anfahrt-Z; `chassis_only.stl` in Fork-Vorsprung-Submesh splitten
(damit nur dieser eine flache Collision bekommt); oder parametrische
Pocket-Geometrie auf der Pallet-Seite statt V-HACD.

### G1 ist FULLY-KINEMATIC

G1-Wurzel ist `kinematic=True`, Arm-Animation läuft über manuelle FK auf den
USD-Link-Xforms (siehe `tools/g1_ik.py`). Keine Balance-Policy, kein Umfallen,
keine Bein-Bewegung. Ausreichend für die Übergabe-Demo, nicht ausreichend für
RL-Training. Für letzteres braucht's IsaacLab-G1-Loco-Manip-Env.

### Box `pallet_box_b` als Primitiv-Cuboid

Das in `g1_handover/`-Env zu greifende Paket ist ein Primitiv-Cuboid (sicherer
Rigid-Body), nicht das echte `pallet_box_a`-Asset. Letzteres ist ein reines
Mesh ohne RigidBodyAPI; für echten Physik-Carry müsste ein USD-Wrapper-
Konverter geschrieben werden (analog `cabin_wrapped.usd`).

## Never Touch ohne explizite Diskussion

- Wheel-Sphere-Collider (hart erarbeitete Stabilität).
- `lift_joint` Drive-Konfig (k=1000, d=50).
- `docking_unit` Geometrie (laufendes Issue, siehe oben).
- Pallet-/Box-Physics (verifiziert, jede Änderung riskiert PhysX-Explosionen).
- Cabin-Mesh.
