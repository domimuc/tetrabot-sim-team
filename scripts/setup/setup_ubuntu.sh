#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# tetrabot_sim — Ubuntu one-shot Setup (Isaac Sim standalone)
#
# Getestet gegen Ubuntu 22.04 LTS. Sollte auf 24.04 ebenso funktionieren;
# wenn nicht, ist es entweder ein NVIDIA-Driver- oder ein Isaac-Sim-Versions-
# Problem (siehe unten), nicht dieses Skript.
#
# Was dieses Skript macht:
#   1. apt-Pakete: git, python3-pip, build-essential (für ikpy-Wheel-Fallback)
#   2. Sucht Isaac Sim in den üblichen Pfaden bzw. unter $ISAAC_SIM_PATH.
#   3. Installiert ikpy in Isaac Sim's bundled Python (für G1-Arm-IK).
#   4. Optional: klont Isaac Lab 2.3 nach ../isaaclab/ falls --with-isaaclab
#      gesetzt ist (für g1_handover/). Erfordert sudo nicht.
#   5. Legt logs/ an, macht einen kurzen headless Smoke-Test.
#
# Was dieses Skript NICHT macht:
#   - Es installiert Isaac Sim 5.0 nicht. Anleitung:
#       https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html
#     Empfohlen: tarball-Install nach ~/isaac-sim oder /opt/isaac-sim.
#   - Es installiert keine NVIDIA-Treiber. Mindestens 555.85 für Kit 107.3.
#       sudo ubuntu-drivers autoinstall
#       sudo reboot
#   - Es installiert ROS2 Humble nicht. Isaac Sim bringt eine gebundelte
#     Variante unter $ISAAC/exts/isaacsim.ros2.bridge/humble/ mit, die
#     launch.sh automatisch wired-up.
#
# Aufruf:
#   bash scripts/setup/setup_ubuntu.sh                # Basis-Setup
#   bash scripts/setup/setup_ubuntu.sh --with-isaaclab  # + Isaac Lab 2.3 klonen
#   ISAAC_SIM_PATH=/opt/isaac-sim bash scripts/setup/setup_ubuntu.sh
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WITH_ISAACLAB=0
for arg in "$@"; do
    case "$arg" in
        --with-isaaclab) WITH_ISAACLAB=1 ;;
        -h|--help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            echo "Unbekannte Option: $arg" >&2
            exit 1
            ;;
    esac
done

echo
echo "=== tetrabot_sim Ubuntu Setup ==="
echo "Repo: $REPO_ROOT"
echo

# --- Schritt 1: apt-Pakete -----------------------------------------------
echo "[1/5] apt-Pakete prüfen..."
NEED_APT=()
for pkg in git python3-pip build-essential; do
    if ! dpkg -s "$pkg" >/dev/null 2>&1; then
        NEED_APT+=("$pkg")
    fi
done
if [[ ${#NEED_APT[@]} -gt 0 ]]; then
    echo "      installiere: ${NEED_APT[*]}"
    sudo apt-get update
    sudo apt-get install -y "${NEED_APT[@]}"
else
    echo "      alle nötigen apt-Pakete vorhanden."
fi

# --- Schritt 2: Isaac Sim finden -----------------------------------------
echo
echo "[2/5] Isaac Sim suchen..."
ISAAC_CANDIDATES=(
    "${ISAAC_SIM_PATH:-}"
    "${HOME}/isaac-sim"
    "${HOME}/.local/share/ov/pkg/isaac-sim-5.0.0"
    "/opt/isaac-sim"
    "/isaac-sim"
)
ISAAC_ROOT=""
for cand in "${ISAAC_CANDIDATES[@]}"; do
    [[ -z "$cand" ]] && continue
    if [[ -x "$cand/python.sh" ]]; then
        ISAAC_ROOT="$cand"
        break
    fi
done

if [[ -z "$ISAAC_ROOT" ]]; then
    cat >&2 <<EOF
[FEHLER] Isaac Sim python.sh nicht gefunden.

Gesuchte Pfade:
$(printf '  - %s\n' "${ISAAC_CANDIDATES[@]}")

Lösung:
  1. Isaac Sim 5.0 installieren (tarball oder Launcher).
     Anleitung: https://docs.isaacsim.omniverse.nvidia.com/
  2. ODER ISAAC_SIM_PATH setzen, z.B.:
       ISAAC_SIM_PATH=/opt/isaac-sim bash scripts/setup/setup_ubuntu.sh
EOF
    exit 1
fi
echo "      [OK] gefunden unter: $ISAAC_ROOT"

# --- Schritt 3: ikpy in Isaac-Python installieren -----------------------
echo
echo "[3/5] ikpy in Isaac Sim's Python prüfen..."
if "$ISAAC_ROOT/python.sh" -c "import ikpy" >/dev/null 2>&1; then
    echo "      [OK] ikpy bereits vorhanden."
else
    echo "      installiere ikpy..."
    "$ISAAC_ROOT/python.sh" -m pip install --only-binary=:all: ikpy
fi

# --- Schritt 4: optional Isaac Lab klonen --------------------------------
if [[ "$WITH_ISAACLAB" == "1" ]]; then
    echo
    echo "[4/5] Isaac Lab 2.3 klonen..."
    ISAACLAB_DIR="$(dirname "$REPO_ROOT")/isaaclab"
    if [[ -d "$ISAACLAB_DIR/.git" ]]; then
        echo "      [OK] Isaac Lab schon vorhanden unter $ISAACLAB_DIR (überspringe Clone)."
    else
        git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
        echo "      [OK] geklont nach $ISAACLAB_DIR"
        echo "      Setup-Anleitung für Isaac Lab selbst: cd $ISAACLAB_DIR && ./isaaclab.sh --install"
        echo "      (lief nicht automatisch — das Isaac-Lab-eigene Setup ist nicht-trivial)"
    fi
else
    echo
    echo "[4/5] Isaac Lab übersprungen (kein --with-isaaclab)."
fi

# --- Schritt 5: logs/ + Smoke-Test --------------------------------------
echo
echo "[5/5] logs/ anlegen + Smoke-Test..."
mkdir -p "$REPO_ROOT/logs"

cd "$REPO_ROOT"
set +e
ISAAC_SIM_PATH="$ISAAC_ROOT" ./launch.sh --scenario cabin_assembly --headless --frames 300
RC=$?
set -e

echo
if [[ "$RC" == "0" ]]; then
    cat <<EOF
[OK] Smoke-Test grün. Setup fertig.

Nächste Schritte:
  - Vollen Demo-Lauf:   ./launch.sh --scenario cabin_assembly
  - Headless mit Telemetrie:   ./launch.sh --scenario cabin_assembly --headless --frames 3300
  - Doku:   README.md, docs/START_GUIDE.md

Falls Isaac Sim woanders liegt, beim launch.sh-Aufruf wieder
ISAAC_SIM_PATH=$ISAAC_ROOT setzen (oder in ~/.bashrc exportieren).
EOF
else
    cat >&2 <<EOF
[WARNUNG] Smoke-Test mit Exit-Code $RC beendet.
Letztes Log: $REPO_ROOT/logs/latest.log

Häufige Ursachen:
  - NVIDIA-Driver zu alt (mindestens 555.85 für Kit 107.3).
  - VRAM zu knapp: --low-vram probieren.
  - Vulkan-Layer-Konflikt (Steam, OBS, RenderDoc): launch.sh setzt
    VK_LOADER_LAYERS_DISABLE=~implicit~, sollte das aushebeln; wenn
    nicht, ImplicitLayer-Files unter ~/.local/share/vulkan/ prüfen.
EOF
fi

exit "$RC"
