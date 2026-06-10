#!/usr/bin/env bash
# TETRABot launcher — Linux counterpart to launch.bat
#
# Usage:
#   ./launch.sh                              # interactive (no environment)
#   ./launch.sh --auto-demo                  # auto-animate joints
#   ./launch.sh --scene cabin                # interactive with A320 cabin
#   ./launch.sh --scene cabin --auto-demo    # cabin + auto-animate
#   ./launch.sh --headless --auto-demo --frames 600   # CI-style smoke test
#   ./launch.sh --scenario cabin_assembly --headless --frames 3300  # full demo headless
#
# Override Isaac Sim location:
#   ISAAC_SIM_PATH=/opt/isaac-sim ./launch.sh ...
#
# All Python output goes to logs/launch_YYYYMMDD_HHMMSS.log
# (and logs/latest.log which is overwritten each run).

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/tools/launch.py"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ----------------------------------------------------------------------
# Locate Isaac Sim install.
#
# Isaac Sim 5.0 on Linux ships either via the standalone tarball
# (extract anywhere, default ~/isaac-sim) or via pip install isaacsim
# into a venv. We support the tarball path here — for pip-installed
# Isaac, set ISAAC_SIM_PATH to the venv root so $ISAAC_SIM_PATH/python.sh
# resolves (or run `python tools/launch.py` directly inside the venv).
# ----------------------------------------------------------------------
ISAAC_CANDIDATES=(
    "${ISAAC_SIM_PATH:-}"
    "${HOME}/isaac-sim"
    "${HOME}/.local/share/ov/pkg/isaac-sim-5.0.0"
    "/opt/isaac-sim"
    "/isaac-sim"
)

ISAAC_PYTHON=""
for cand in "${ISAAC_CANDIDATES[@]}"; do
    [[ -z "${cand}" ]] && continue
    if [[ -x "${cand}/python.sh" ]]; then
        ISAAC_PYTHON="${cand}/python.sh"
        ISAAC_ROOT="${cand}"
        break
    fi
done

if [[ -z "${ISAAC_PYTHON}" ]]; then
    echo "ERROR: Isaac Sim python.sh not found." >&2
    echo "  Searched: ${ISAAC_CANDIDATES[*]}" >&2
    echo "  Set ISAAC_SIM_PATH=/path/to/isaac-sim and re-run, or install" >&2
    echo "  Isaac Sim 5.0 from https://docs.isaacsim.omniverse.nvidia.com/" >&2
    exit 1
fi

# ----------------------------------------------------------------------
# Disable Vulkan implicit layers (defense-in-depth — on Linux these are
# usually rare, but Steam/OBS/RenderDoc can still install layers under
# ~/.local/share/vulkan/implicit_layer.d that auto-inject into every
# Vulkan app and have caused PhysX/Hydra crashes on the Windows side).
# ----------------------------------------------------------------------
export VK_LOADER_LAYERS_DISABLE="${VK_LOADER_LAYERS_DISABLE:-~implicit~}"

# ----------------------------------------------------------------------
# Bundled ROS2 (Humble) shipped with Isaac Sim 5.0.
#
# Mirrors launch.bat: always wire up the bundled distro so the ROS2
# bridge extension can dlopen its libs. Harmless when --ros isn't
# requested. If the user has a system-wide ROS2 install (e.g. apt-
# installed /opt/ros/humble) they want to prefer, they can source its
# setup.bash before running ./launch.sh.
# ----------------------------------------------------------------------
: "${ROS_DISTRO:=humble}"
: "${RMW_IMPLEMENTATION:=rmw_fastrtps_cpp}"
export ROS_DISTRO RMW_IMPLEMENTATION

ROS_BRIDGE_LIB="${ISAAC_ROOT}/exts/isaacsim.ros2.bridge/humble/lib"
if [[ -d "${ROS_BRIDGE_LIB}" ]]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${ROS_BRIDGE_LIB}"
fi

echo "Launching TETRABot..."
echo "  Isaac:  ${ISAAC_ROOT}"
echo "  Script: ${SCRIPT}"
echo "  Logs:   ${LOG_DIR}/latest.log"
echo

# Pre-create logs/latest.log so headless redirection works even if the
# Python side crashes before logging is fully initialized.
: > "${LOG_DIR}/latest.log"

set +e
"${ISAAC_PYTHON}" "${SCRIPT}" "$@"
RC=$?
set -e

echo
echo "Exit code: ${RC}"
echo "Last log: ${LOG_DIR}/latest.log"
exit "${RC}"
