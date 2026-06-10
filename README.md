# tetrabot_sim — Multi-Robot Cabin-Assembly Demo (Isaac Sim 5.x)

Plain Isaac Sim Sandbox: **4 TETRABot Mecanum-Roboter transportieren kooperativ
eine Euro-Palette in eine A30X-Flugzeugkabine und übergeben eine Box an einen
Unitree-G1-Humanoid.** Gebaut für Capgemini-Engineering-Demo-Recordings und als
R&D-Spielwiese für Multi-Robot-Manipulation.

```
   ┌──────────────────────────────────────────────────────────┐
   │ (cabin interior, walls W/N/S, east open as entry)        │
   │                              ●  G1 humanoid              │
   │                            ╱    (right arm picks box)    │
   │  ┌──────────┐                                            │
   │  │  pallet  │── delivered                                │
   │  └──────────┘                                            │
   ╳════════════════════════════════════════════════════ open │
                                            ▲                 │
   ┌─┐ ┌─┐ ┌─┐ ┌─┐    transport east → west │ ← bots enter    │
   │0│ │1│ │2│ │3│    in row, then split    │   here           │
   └─┘ └─┘ └─┘ └─┘                          │                  │
   row spawn (X=+5.5, Y=∓1.5/∓0.5)                             │
   ────────────────────────────────────────────────────────────
                              ground plane (Z=0.79)
```

`cabin_assembly` ist eine 47.5-s-Choreographie in 11 Phasen (ROW → SPLIT → DOCK
→ LIFT_UP → ENTER_CABIN → RELEASE → RETREAT → G1_REACH → GRASP → LIFT → PLACE).
Jeder Lauf schreibt strukturierte Telemetrie (JSONL pro Snapshot) und eine
Synthetic-Data-Card nach `logs/`.

## Features

**Simulation**
- Multi-TETRABot-Spawn (N Robots aus einer URDF, `cabin_assembly` erzwingt ≥4)
- Mecanum-/Holonom-Antrieb über planare Dummy-Joints (Ridgeback-Pattern aus
  IsaacLab) — stabile holonome Fahrt ohne Rad-Boden-Reibungs-Numerik
- Waypoint-Choreographie mit Smoothstep-Easing (ruckelfreie Beschleunigung)
- Euro-Paletten-Pickup via Kinematic-Carry (kein FixedJoint — vermeidet
  PhysX-Impulse-Akkumulation)
- A30X-Kabine als CATIA-STEP-USD-Asset (Collision per Default OFF — Triangle-
  Mesh-Collider sind PhysX-instabil)
- Unitree-G1-Humanoid am Anlieferpunkt mit ikpy-basierter Arm-IK + scripted
  Keyframes, kinematic-Carry-Pickup

**Recording / Output**
- Telemetrie-JSONL: ein Snapshot pro Frame mit Roboter-/Objekt-/G1-State
- Synthetic-Data-Card pro Lauf (`logs/data_card_*.md`)
- Replicator-SDG (RGB + Depth + Instance + BBox) per `--sdg`
- ROS2-Topic-Publishing per `--ros` (gebundeltes Humble, kein separater Install)
- Cinematic-Kamera-Auto-Cycling (9 Anchor-Cams) opt-in per `--camera-cycle`
- Live-HUD-Overlay (GUI-Mode, 10 Hz Phase/Time/State)

**Betrieb / Tooling**
- WebRTC-Livestream für headless Cloud-Setups (`--livestream`, Browser-Viewer)
- Keyboard-Teleop für einen einzelnen Bot (WASD/QE/RF)
- Drop-Mode für Höhen-Kalibrierung (Objekte hoch spawnen, Aufschlag-Z loggen)
- Low-VRAM-Modus gegen `cudaErrorMemoryAllocation` auf ≤8 GB GPUs
- Asset-Konverter in `tools/`: STEP→USD, STL→USD, Pallet-V-HACD, URDF-Lint

**RL-Scaffold (nicht trainiert)**
- `--controller rl` routet die Chassis-Steuerung durch
  `tools/rl_policy.py::TetraLocomotionPolicy` — Stub heute, Inferenz-Plumbing
  (ONNX/TorchScript via `--rl-weights`) verdrahtet
- `tools/train_g1_locomotion.py` als IsaacLab-Trainings-Scaffold (TODO)

**Isaac-Lab-Teleop-Pfad (`g1_handover/`, deferred)**
- Gym-Tasks `Isaac-PickTurn-G1-Box-Abs-v0` und
  `Isaac-PickTurn-G1-CabinHandover-Abs-v0` für G1-Oberkörper-Teleop
- Apple-Vision-Pro-/OpenXR-Handtracking-Config vorhanden
- `record_handover.py` + `stream_hold.py` als Recording-/Stream-Harness
- **Status**: scripted on-box läuft, AVP-CloudXR-End-to-End nicht fertig
  integriert (Details in [PROJECT_NOTES.md](PROJECT_NOTES.md))

## Nutzungs-Historie

| Phase | Setup | Was lief |
|---|---|---|
| 1 | Windows 11 + lokal installiertes Isaac Sim unter `C:\isaac-sim\` (RTX 5060 Ti) | Standalone-Sim (kein Isaac Lab) — die ganze `cabin_assembly`-Choreo, G1-IK |
| 2 | NVIDIA Brev Cloud-Instanz, Container-Isaac-Sim unter `/isaac-sim/`, persistenter `/workspace/` | Headless + WebRTC-Livestream in Browser; Isaac Lab 2.3 für `g1_handover/` dazu |
| 3 | Office-Ubuntu (übergabebereit) | gleichbleibende Codebasis; Setup über `scripts/setup/setup_ubuntu.sh` |

Beim Wechsel zwischen Setups muss meistens nur `ISAAC_SIM_PATH` an die jeweilige
Installation gesetzt werden (`launch.sh`/`launch.bat` respektieren das).

## Prerequisites

| Komponente | Version | Quelle |
|---|---|---|
| Isaac Sim | 5.0 (Kit 107.3) | <https://docs.isaacsim.omniverse.nvidia.com/> |
| NVIDIA-Driver | ≥ 555.85 | `ubuntu-drivers autoinstall` bzw. NVIDIA-Installer |
| GPU | ≥ 8 GB VRAM (sonst `--low-vram`) | RTX 30-/40-/50er-Serie verifiziert |
| Python | 3.11 (gebundelt mit Isaac Sim) | kommt mit Isaac Sim |
| ikpy | 3.4.2 | wird vom Setup-Skript via pip installiert |
| Isaac Lab 2.3 | optional, nur für `g1_handover/` | <https://isaac-sim.github.io/IsaacLab/> |

## Setup (einmalig pro Maschine)

```bash
# Windows (Isaac Sim unter C:\isaac-sim\)
scripts\setup\setup_windows.bat

# Windows (Source-Build / andere Lokation)
set ISAAC_SIM_PATH=C:\dein\isaac-sim-pfad
scripts\setup\setup_windows.bat

# Ubuntu
bash scripts/setup/setup_ubuntu.sh                  # Basis
bash scripts/setup/setup_ubuntu.sh --with-isaaclab  # + Isaac Lab 2.3 klonen

# NVIDIA Brev Cloud-Instanz
bash scripts/setup/setup_brev.sh
```

Die Skripte prüfen Isaac-Sim-Pfad, installieren `ikpy` und machen einen
headless Smoke-Test. Details (was wird **nicht** gemacht, Troubleshooting) in
[`scripts/setup/README.md`](scripts/setup/README.md).

## Demo starten

```bash
# Windows
launch.bat --scenario cabin_assembly

# Linux / Brev
./launch.sh --scenario cabin_assembly
# bzw. (Brev mit WebRTC-Stream):
bash start_tetrabot.sh
```

GUI lädt ~25 s, dann **Play** drücken. Choreographie läuft 47.5 s und stoppt
automatisch. Stop + Play = Replay.

Headless für CI / Smoke-Tests:

```bash
./launch.sh --scenario cabin_assembly --headless --frames 3300
```

## CLI-Flags

| Flag | Default | Bedeutung |
|---|---|---|
| `--scenario {none,cabin_assembly}` | `none` | Skript-Szenario. `cabin_assembly` = volle Demo (erzwingt ≥4 Bots, Kabine, Palette, G1). |
| `--headless` | aus | Kein GUI/Fenster (CI / `--frames`-Läufe). |
| `--livestream` | aus | WebRTC-Stream in den Browser (für headless Cloud). |
| `--frames N` | `0` (endlos) | Nach N Frames automatisch stoppen. |
| `--num-tetrabots N` | `1` | Anzahl TETRABots. |
| `--scene {none,cabin}` | `none` | A30X-Kabine laden. |
| `--workpiece` | aus | Euro-Palette + 2 Boxen spawnen. |
| `--g1` / `--no-g1` | an für `cabin_assembly` | G1-Humanoid am Anlieferpunkt. |
| `--g1-usd PATH` | `assets/environment/g1_humanoid.usd` | G1-USD überschreiben. |
| `--g1-z-offset FLOAT` | `0.0` | G1-Spawn-Z feinjustieren (±5 cm). |
| `--keyboard` | aus | WASD/QE/RF-Steuerung eines Bots. |
| `--auto-demo` | aus | Gelenke sinusförmig animieren. |
| `--drop-mode` | aus | Objekte hoch spawnen, Aufprall-Z loggen. |
| `--camera-cycle` | aus | Cinematic-Kamera-Auto-Switching für Recordings. |
| `--cameras` | aus | D435i-Kamera an Bot 0 anhängen. |
| `--ros` | aus | Kamera als ROS2-Topics publizieren (impliziert `--cameras`). |
| `--sdg` | aus | Replicator-SDG (RGB+Depth+Instance+BBox), impliziert `--cameras`. |
| `--controller {hand,rl}` | `hand` | Chassis-Controller. `rl` = `tools/rl_policy.py`-Stub. |
| `--rl-weights PATH` | `None` | `.pt`/`.onnx`-Gewichte für `--controller rl`. |
| `--low-vram` | aus | Renderauflösung senken (gegen VRAM-OOM). |
| `--cabin-floor-z FLOAT` | `0.79` (locked) | Arbeitshöhe. **Standing Order — nicht ändern ohne PROJECT_NOTES.md zu lesen.** |
| `--cabin-translate-z FLOAT` | `0.79` | Kabinen-USD-Translate-Z. |
| `--cabin-collision` | aus | Collision auf alle 53 Kabinen-Meshes. Gefahr: PhysX-Explosion wenn `cabin-floor-z > 0.252`. |
| `--urdf PATH` | `urdf/tetrabot.urdf` | URDF überschreiben. |

## Architektur

```
tools/launch.py — ~3000-Zeilen Orchestrator
  ├ argparse + SimulationApp init (single-GPU pinned)
  ├ URDF-Importer für N TETRABots
  ├ Stage-Setup: ground, cabin, walls, pallet, boxes, G1
  ├ Scenario-State-Machine (cabin_assembly: 11 Phasen)
  ├ Per-Frame: P-Controller ODER tools/rl_policy.py stub
  │            + lift drive + kinematic-carry + G1 manual FK
  ├ Telemetrie-Snapshots → JSONL
  └ Data-Card-Markdown am Szenario-Ende
        ↑                  ↑
  tools/g1_ik.py     tools/rl_policy.py
  ikpy chain +       stub heute, trained-policy
  FK helpers         shape
```

**Konventionen:**
- Pallet/Box-Transport ist KINEMATIC-Carry, NICHT FixedJoint
- G1 ist FULLY-KINEMATIC (keine Balance-Policy, kein Umfallen)
- `cover_link`↔`pallet`-Collision FILTERED gegen Retreat-Drag
- `pallet`/`box` ↔ `/World/g1` Collision FILTERED gegen G1-Pickup-Explosions

## Projekt-Layout

```
tetrabot-sim-team/
├── README.md                          ← du bist hier
├── PROJECT_NOTES.md                   ← Standing Orders, Tech-Debt, Roadmap
├── launch.bat / launch.sh             ← Windows / Linux Launcher
├── start_tetrabot.sh                  ← Brev-Cloud-Bootstrap
├── pyproject.toml, .gitignore
│
├── tools/
│   ├── launch.py                      ← MAIN-Orchestrator
│   ├── cloud_start.sh                 ← == start_tetrabot.sh (für /workspace/)
│   ├── g1_ik.py                       ← ikpy IK + FK für G1 rechten Arm
│   ├── rl_policy.py                   ← RL-Controller-Stub
│   ├── train_g1_locomotion.py         ← RL-Trainings-Scaffold
│   ├── run_demo.py                    ← Stage-1 Watch-Demo
│   ├── validate_urdf.py               ← URDF-Lint
│   └── {stl,usd,step}_to_*.py         ← Asset-Konverter
│
├── g1_handover/                       ← Isaac Lab 2.3 G1-Teleop (deferred)
│   ├── __init__.py                    ← registriert die Gym-Tasks
│   ├── cabin_handover_env_cfg.py      ← volle Szene (Kabine + Bots + G1)
│   ├── pick_turn_env_cfg.py           ← einfachere Variante (Palette + Box + G1)
│   ├── mdp_terms.py                   ← box_picked_and_turned Success-Term
│   ├── record_handover.py             ← phasen-gated HDF5-Recording
│   ├── stream_hold.py                 ← View-/Verify-Harness
│   └── make_cabin_wrapper.py          ← cabin_wrapped.usd bauen
│
├── urdf/                              ← TETRABot URDF + Configurations
├── assets/                            ← USDs, STLs, G1-URDF
├── scripts/setup/                     ← Setup-Skripte für Windows/Ubuntu/Brev
├── config/                            ← FastDDS XML
├── docs/                              ← Capgemini-Slides (PDF)
└── logs/                              ← Runtime-Output (gitignored)
```

## Weiterführend

- [PROJECT_NOTES.md](PROJECT_NOTES.md) — Standing Orders (gelockte Werte mit
  Begründung), Tech-Debt, Known Limitations, Roadmap, Never-Touch-Liste.
- [scripts/setup/README.md](scripts/setup/README.md) — was die Setup-Skripte
  tun (und was nicht), Troubleshooting.
- [docs/slides/](docs/slides/) — Capgemini-CAD-Slides (Explosionszeichnung,
  Schwerpunkt-Analyse, Engineering-Horizons-Übersicht).
