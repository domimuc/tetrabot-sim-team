@echo off
REM TETRABot launcher
REM Usage:
REM   launch.bat                              # interactive (no environment)
REM   launch.bat --auto-demo                  # auto-animate joints
REM   launch.bat --scene cabin                # interactive with A320 cabin
REM   launch.bat --scene cabin --auto-demo    # cabin + auto-animate
REM   launch.bat --scene cabin --cameras --auto-demo   # full demo with camera
REM   launch.bat --headless --auto-demo --frames 600   # CI-style smoke test
REM
REM All Python output goes to logs/launch_YYYYMMDD_HHMMSS.log
REM (and logs/latest.log which is overwritten each run).

setlocal

REM ----------------------------------------------------------------------
REM Disable all Vulkan implicit layers for this process only.
REM HKLM\SOFTWARE\Khronos\Vulkan\ImplicitLayers contains OBS, Overwolf (x2),
REM RTSS, Steam (x2); HKCU contains Epic Games Online Services (x2). These
REM DLLs auto-inject into every Vulkan app even if the host apps aren't
REM running. Crash backtrace shows graphics-hook64.dll (OBS) and
REM ow-graphics-hook64.dll (Overwolf) crashing inside vkGetInstanceProcAddr.
REM '~implicit~' is a Vulkan-Loader special token (Loader >= v1.3.234) that
REM disables every implicit layer. setlocal scopes this to one launch.
REM ----------------------------------------------------------------------
set "VK_LOADER_LAYERS_DISABLE=~implicit~"

REM ----------------------------------------------------------------------
REM Bypass NVIDIA's process-name-renaming workaround.
REM
REM Isaac Sim's python.bat copies python.exe -> kit\python\kit.exe so the
REM NV driver applies kit-specific RTX optimizations to the Python process.
REM Comment in python.bat explicitly states: "Later, when kit 107.0 uses
REM the Vulkan as a default, this workaround needs to be removed."  We are
REM on Kit 107.3 — the rename is stale and now harmful: Driver 596.36 sees
REM "kit.exe" process name, applies RTX-process optimizations, but the
REM scenedb plugin loaded inside that Python process can't use them and
REM crashes at carbOnPluginStartup+0x252db. Forcing PYTHONEXE to the real
REM python.exe sidesteps the driver name-based detection entirely.
REM ----------------------------------------------------------------------
set "PYTHONEXE=C:\isaac-sim\kit\python\python.exe"

REM ----------------------------------------------------------------------
REM Bundled ROS2 (humble) shipped with Isaac Sim 5.0.
REM
REM The isaacsim.ros2.bridge extension runs a subprocess
REM (isaacsim.ros2.bridge.check.exe) on startup to verify it can dlopen
REM the ROS2 libs. If ROS_DISTRO is unset or LD/PATH is missing the libs,
REM the extension silently self-disables and OmniGraph node creation fails
REM with "unrecognized type 'isaacsim.ros2.bridge.ROS2Context'".
REM
REM We always wire up the bundled distro here. This is harmless when
REM --ros isn't requested (extension simply isn't enabled), and self-
REM activates the bundled libs when --ros is used. If the user has a
REM system-wide ROS2 install they want to prefer, they can override
REM ROS_DISTRO before invoking launch.bat.
REM ----------------------------------------------------------------------
if not defined ROS_DISTRO set "ROS_DISTRO=humble"
if not defined RMW_IMPLEMENTATION set "RMW_IMPLEMENTATION=rmw_fastrtps_cpp"
if not defined ISAAC_SIM_PATH set "ISAAC_SIM_PATH=C:\isaac-sim"
set "PATH=%PATH%;%ISAAC_SIM_PATH%\exts\isaacsim.ros2.bridge\humble\lib"

REM ----------------------------------------------------------------------
REM FastDDS Transport: bewusst KEIN Override auf Sim-Seite.
REM
REM Empirisch: mit Sim auf Default (SHM+UDP) und WSL-Bridge auf UDPv4-only
REM funktioniert Discovery zuverlässig. Wenn beide Seiten UDPv4-only
REM erzwingen, hängt's. Vermutung: Default-FastDDS auf Sim-Seite hat
REM Multi-Transport-Discovery-Pakete die WSL-UDP-Listener aufpicken kann,
REM während UDPv4-only auf Sim-Seite einen Multicast-Edge-Case triggert.
REM
REM WSL-Seite setzt UDPv4 explicit im Wrapper (tools/run_foxglove_bridge.sh).
REM ----------------------------------------------------------------------

REM ----------------------------------------------------------------------
REM Isaac Sim location. Default: C:\isaac-sim\ (Omniverse Launcher install).
REM Override per Umgebungs-Variable (nuetzlich fuer Source-Builds / andere
REM Installations-Pfade):
REM   set ISAAC_SIM_PATH=C:\Users\you\isaacsim\_build\windows-x86_64\release
REM   launch.bat ...
REM ----------------------------------------------------------------------
if not defined ISAAC_SIM_PATH set "ISAAC_SIM_PATH=C:\isaac-sim"
set "ISAAC_PYTHON=%ISAAC_SIM_PATH%\python.bat"
set "SCRIPT_DIR=%~dp0"
set "SCRIPT=%SCRIPT_DIR%tools\launch.py"
set "LOG_DIR=%SCRIPT_DIR%logs"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%ISAAC_PYTHON%" (
    echo ERROR: Isaac Sim python not found at %ISAAC_PYTHON%
    echo Set ISAAC_SIM_PATH to your Isaac Sim root, e.g.:
    echo    set ISAAC_SIM_PATH=C:\Users\you\isaacsim\_build\windows-x86_64\release
    echo    launch.bat ...
    exit /b 1
)

echo Launching TETRABot...
echo   Script: %SCRIPT%
echo   Logs:   %LOG_DIR%\latest.log
echo.

call "%ISAAC_PYTHON%" "%SCRIPT%" %*

set RC=%ERRORLEVEL%
echo.
echo Exit code: %RC%
echo Last log: %LOG_DIR%\latest.log
exit /b %RC%
