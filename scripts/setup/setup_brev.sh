#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# tetrabot_sim — NVIDIA-Brev Setup (Cloud-Instanz mit gebundeltem Isaac Sim)
#
# Brev-Launchables haben Isaac Sim 5.x im Container unter /isaac-sim/ und
# ein persistentes Volume unter /workspace/. Dieses Skript ist dünn — es
# muss "nur" ikpy nachziehen (wird beim Image-Restart resettet) und das
# Repo nach /workspace/ legen.
#
# Was dieses Skript macht:
#   1. Repo nach /workspace/tetrabot-sim/ klonen (oder pullen, wenn schon da).
#   2. ikpy in /isaac-sim/python.sh installieren (idempotent).
#   3. Lokales Convenience-Skript /workspace/start_tetrabot.sh als Symlink
#      auf das Repo-Pendant anlegen, damit "bash /workspace/start_tetrabot.sh"
#      funktioniert wie historisch dokumentiert.
#
# Was dieses Skript NICHT macht:
#   - Setzt keinen WebRTC-Port frei — das macht Brev's Proxy-URL.
#   - Setzt keinen Stream — das macht launch.py mit --livestream.
#   - Installiert kein Isaac Lab — siehe Hinweise am Ende.
#
# Aufruf (direkt auf der Brev-Box):
#   bash <(curl -fsSL https://raw.githubusercontent.com/<repo>/scripts/setup/setup_brev.sh)
#   # oder, falls Repo schon da:
#   bash /workspace/tetrabot-sim/scripts/setup/setup_brev.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/domimuc/tetrabot-sim.git}"
REPO_DIR="/workspace/tetrabot-sim"
ISAAC_PY="/isaac-sim/python.sh"

echo
echo "=== tetrabot_sim Brev Setup ==="
echo "Repo URL:    $REPO_URL"
echo "Repo Dir:    $REPO_DIR"
echo "Isaac Sim:   /isaac-sim/ (Brev image default)"
echo

# --- Schritt 1: /workspace prüfen ---------------------------------------
if [[ ! -d /workspace ]]; then
    echo "[FEHLER] /workspace existiert nicht. Bist du sicher dass das eine Brev-Instanz ist?" >&2
    exit 1
fi

if [[ ! -x "$ISAAC_PY" ]]; then
    echo "[FEHLER] $ISAAC_PY nicht ausführbar. Brev-Image evtl. defekt oder anderer Pfad?" >&2
    echo "         Setze ggf. ISAAC_PY=/dein/pfad/python.sh vor dem Aufruf." >&2
    exit 1
fi

# --- Schritt 2: Repo klonen/pullen --------------------------------------
echo "[1/3] Repo nach $REPO_DIR ..."
if [[ ! -d "$REPO_DIR/.git" ]]; then
    git clone "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" pull --ff-only \
        || echo "      (pull übersprungen — lokale Änderungen oder offline. Aktueller Stand bleibt.)"
fi
echo "      [OK]"

# --- Schritt 3: ikpy ----------------------------------------------------
echo
echo "[2/3] ikpy installieren..."
if "$ISAAC_PY" -c "import ikpy" >/dev/null 2>&1; then
    echo "      [OK] ikpy bereits vorhanden."
else
    "$ISAAC_PY" -m pip install --only-binary=:all: ikpy
    echo "      [OK] ikpy installiert."
fi

# --- Schritt 4: Convenience-Symlink -------------------------------------
echo
echo "[3/3] /workspace/start_tetrabot.sh Symlink..."
if [[ -L /workspace/start_tetrabot.sh || -f /workspace/start_tetrabot.sh ]]; then
    rm -f /workspace/start_tetrabot.sh
fi
ln -s "$REPO_DIR/start_tetrabot.sh" /workspace/start_tetrabot.sh
echo "      [OK] -> $REPO_DIR/start_tetrabot.sh"

# --- Abschluss ----------------------------------------------------------
PROXY_HINT="${VSCODE_PROXY_URI:-https://<dein-host>.brevlab.com/proxy/{{port}}/}"
VIEWER_URL="$(printf '%s' "$PROXY_HINT" | sed 's/{{port}}/5173/')viewer/"

cat <<EOF

[OK] Brev-Setup fertig.

Demo starten:
  bash /workspace/start_tetrabot.sh                      # voll: cabin_assembly + Livestream
  bash /workspace/start_tetrabot.sh --no-camera-cycle    # ohne Kamera-Switch
  bash /workspace/start_tetrabot.sh --low-vram           # weniger VRAM

Viewer im Browser:
  $VIEWER_URL
  (warten bis Port 49100 lauscht und der Kit-Renderer hochgefahren ist)

Hinweis Isaac Lab 2.3:
  Für das g1_handover/-Paket brauchst du Isaac Lab 2.3 separat unter
  /workspace/isaaclab/. Brev-Images bringen das nicht mit. Anleitung:
    cd /workspace
    git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git isaaclab
    cd isaaclab && ./isaaclab.sh --install
  Das ist nicht trivial und nicht Teil dieses Setup-Skripts.

Doku: README.md, docs/START_GUIDE.md, docs/AVP_CLOUDXR_TELEOP.md
EOF
