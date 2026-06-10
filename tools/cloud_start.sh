#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Cloud bootstrap for tetrabot-sim on the headless NVIDIA-Brev Isaac Sim
# launchable. Survives an instance restart / container re-create:
#   1. ensures the repo exists in persistent /workspace (clone or pull),
#   2. reinstalls the `ikpy` dependency (the Isaac Sim pip site is reset to the
#      image on restart, so user-installed packages are lost),
#   3. stops any stale kit instance (avoids kvdb-lock + WebRTC-port contention),
#   4. launches the cabin_assembly demo with WebRTC livestream.
#
# After it prints "open the GUI at ...", open that URL in a browser logged into
# Brev and press Play in the streamed Isaac Sim toolbar.
#
# Usage:  bash /workspace/tetrabot-sim/tools/cloud_start.sh [extra launch.py flags]
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_URL="https://github.com/domimuc/tetrabot-sim.git"
REPO_DIR="/workspace/tetrabot-sim"
ISAAC_PY="/isaac-sim/python.sh"

# 1. Repo present in persistent /workspace?
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "[cloud_start] cloning $REPO_URL -> $REPO_DIR"
  git clone "$REPO_URL" "$REPO_DIR"
else
  echo "[cloud_start] repo present; pulling latest (ff-only)"
  git -C "$REPO_DIR" pull --ff-only || echo "[cloud_start] pull skipped (local changes or offline) — using current checkout"
fi

# 2. Ensure ikpy (G1 inverse-kinematics dep; cabin_assembly forces G1 on)
if ! "$ISAAC_PY" -c "import ikpy" >/dev/null 2>&1; then
  echo "[cloud_start] installing ikpy"
  "$ISAAC_PY" -m pip install --only-binary=:all: ikpy
fi

# 3. Kill any stale launch.py / kit instance
if pgrep -f "tools/launch.py" >/dev/null 2>&1; then
  echo "[cloud_start] stopping stale launch.py instance(s)"
  pkill -f "tools/launch.py" || true
  sleep 3
fi

# 4. Derive + print the viewer URL from Brev's proxy env var.
#    VSCODE_PROXY_URI is like https://<host>.brevlab.com/proxy/{{port}}/
VIEWER_URL="$(printf '%s' "${VSCODE_PROXY_URI:-}" | sed 's/{{port}}/5173/')viewer/"
echo "[cloud_start] ----------------------------------------------------------"
echo "[cloud_start] open the GUI at:  ${VIEWER_URL:-https://<your-brev-host>/proxy/5173/viewer/}"
echo "[cloud_start] (wait until the scene finished loading, reload the tab, then Play)"
echo "[cloud_start] ----------------------------------------------------------"

# 5. Launch (interactive: waits for Play, replayable, streams over WebRTC)
cd "$REPO_DIR"
exec "$ISAAC_PY" tools/launch.py --scenario cabin_assembly --livestream "$@"
