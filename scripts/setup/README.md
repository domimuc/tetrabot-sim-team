# Setup-Skripte

Einmaliges Bootstrap pro Maschine. Prüft Isaac-Sim-Pfad, installiert `ikpy`,
macht einen kurzen headless Smoke-Test.

| Setup | Skript | Default-Pfad |
|---|---|---|
| Windows 10/11, Isaac Sim lokal | [`setup_windows.bat`](setup_windows.bat) | `C:\isaac-sim\` |
| Ubuntu 22.04+ Workstation | [`setup_ubuntu.sh`](setup_ubuntu.sh) | `~/isaac-sim`, `/opt/isaac-sim`, `~/.local/share/ov/pkg/isaac-sim-5.0.0` |
| NVIDIA Brev Cloud-Instanz | [`setup_brev.sh`](setup_brev.sh) | `/isaac-sim/` |

Andere Lokation? `ISAAC_SIM_PATH` setzen:

```bash
set ISAAC_SIM_PATH=C:\dein\pfad        # Windows
ISAAC_SIM_PATH=/opt/isaac-sim bash scripts/setup/setup_ubuntu.sh
```

`setup_ubuntu.sh --with-isaaclab` klont zusätzlich Isaac Lab 2.3 nach
`../isaaclab/` (für `g1_handover/`). Das Isaac-Lab-eigene `./isaaclab.sh
--install` muss danach manuell laufen.

## Was die Skripte NICHT machen

- **Isaac Sim 5.0 selbst installieren** — geht nur über NVIDIA's Omniverse
  Launcher bzw. `pip install isaacsim`. Siehe
  <https://docs.isaacsim.omniverse.nvidia.com/>.
- **NVIDIA-Driver installieren** — mindestens 555.85 für Kit 107.3.
  Ubuntu: `sudo ubuntu-drivers autoinstall && sudo reboot`.
- **ROS2 Humble installieren** — Isaac Sim bringt ein gebundeltes Humble unter
  `$ISAAC_SIM_PATH/exts/isaacsim.ros2.bridge/humble/` mit; `launch.sh` /
  `launch.bat` hängen das automatisch ein. System-Install (`apt install
  ros-humble-desktop`) optional, dann vor dem launch `setup.bash` sourcen.

## Troubleshooting

- **`ikpy`-Install scheitert**: das `--only-binary=:all:`-Flag verhindert
  Source-Build. Wenn doch versucht wird zu kompilieren, fehlt `build-essential`
  (Linux) bzw. die VS-C++-Toolchain (Windows). Notfalls ältere Version:
  `pip install --only-binary=:all: "ikpy<3.4"`.
- **`cudaErrorMemoryAllocation`**: VRAM zu knapp → `--low-vram`, oder andere
  GPU-Apps schließen.
- **Vulkan-Loader-Errors**: Implicit-Layer-Konflikt. Die Launcher setzen
  `VK_LOADER_LAYERS_DISABLE=~implicit~`. Wenn das nicht reicht, unter
  `~/.local/share/vulkan/implicit_layer.d/` (Linux) bzw. der Registry
  `HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers` (Windows) schauen.
- **Brev: Browser-Viewer schwarz**: Kit braucht 25–60 s bis der WebRTC-Encoder
  läuft. Erst `[INFO] Streaming server started` abwarten, dann Tab reloaden.
