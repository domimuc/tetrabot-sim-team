@echo off
REM ---------------------------------------------------------------------------
REM tetrabot_sim — Windows one-shot Setup (Isaac Sim standalone)
REM
REM Was dieses Skript macht:
REM   1. Prueft ob Isaac Sim 5.0 unter C:\isaac-sim\ liegt (Default).
REM      Falls woanders: ISAAC_SIM_PATH=C:\dein\pfad vor dem Aufruf setzen.
REM   2. Installiert ikpy in die Isaac-Sim-eigene Python-Umgebung
REM      (wird von g1_humanoid / cabin_assembly fuer die G1-Arm-IK gebraucht).
REM   3. Legt logs\ an.
REM   4. Macht einen kurzen headless Smoke-Test (3 Sekunden), damit man
REM      sofort weiss ob die Toolchain ueberhaupt startet.
REM
REM Was dieses Skript NICHT macht:
REM   - Es installiert Isaac Sim 5.0 nicht. Das geht ueber NVIDIA's
REM     Omniverse Launcher: https://docs.isaacsim.omniverse.nvidia.com/
REM   - Es installiert keine NVIDIA-GPU-Driver. Mindestens 555.85 fuer
REM     Kit 107.3 / Isaac Sim 5.0.
REM   - Es installiert nicht ROS2 Humble. Wenn --ros benoetigt wird,
REM     bringt Isaac Sim einen gebundelten Humble unter
REM     C:\isaac-sim\exts\isaacsim.ros2.bridge\humble\ mit, der von
REM     launch.bat automatisch eingehaengt wird.
REM
REM Aufruf:
REM   scripts\setup\setup_windows.bat
REM ---------------------------------------------------------------------------

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.." >nul
set "REPO_ROOT=%CD%"
popd >nul

if not defined ISAAC_SIM_PATH set "ISAAC_SIM_PATH=C:\isaac-sim"
set "ISAAC_PY=%ISAAC_SIM_PATH%\python.bat"

echo.
echo === tetrabot_sim Windows Setup ===
echo Repo:      %REPO_ROOT%
echo Isaac Sim: %ISAAC_SIM_PATH%
echo.

REM --- Schritt 1: Isaac Sim Pfad pruefen ---------------------------------
if not exist "%ISAAC_PY%" (
    echo [FEHLER] Isaac Sim python.bat nicht gefunden unter "%ISAAC_PY%".
    echo.
    echo Optionen:
    echo   a^) Isaac Sim 5.0 unter C:\isaac-sim\ installieren ^(Default-Pfad^), oder
    echo   b^) Pfad explizit setzen, z.B.:
    echo        set ISAAC_SIM_PATH=D:\nvidia\isaac-sim
    echo        scripts\setup\setup_windows.bat
    echo.
    exit /b 1
)
echo [OK] Isaac Sim python.bat gefunden.

REM --- Schritt 2: ikpy installieren ---------------------------------------
echo.
echo Pruefe ikpy in Isaac Sim's Python...
call "%ISAAC_PY%" -c "import ikpy" >nul 2>&1
if errorlevel 1 (
    echo [INFO] ikpy fehlt, installiere via pip...
    call "%ISAAC_PY%" -m pip install --only-binary=:all: ikpy
    if errorlevel 1 (
        echo [FEHLER] pip install ikpy fehlgeschlagen.
        exit /b 1
    )
    echo [OK] ikpy installiert.
) else (
    echo [OK] ikpy bereits vorhanden.
)

REM --- Schritt 3: logs/ anlegen -------------------------------------------
if not exist "%REPO_ROOT%\logs" (
    mkdir "%REPO_ROOT%\logs"
    echo [OK] logs\ angelegt.
)

REM --- Schritt 4: Smoke-Test ----------------------------------------------
echo.
echo Smoke-Test: headless cabin_assembly, 300 frames ^(~10s^)...
pushd "%REPO_ROOT%" >nul
call launch.bat --scenario cabin_assembly --headless --frames 300
set RC=%ERRORLEVEL%
popd >nul

echo.
if "%RC%"=="0" (
    echo [OK] Smoke-Test gruen. Setup fertig.
    echo.
    echo Naechster Schritt: launch.bat --scenario cabin_assembly
    echo Doku: README.md, docs\START_GUIDE.md
) else (
    echo [WARNUNG] Smoke-Test mit Exit-Code %RC% beendet.
    echo Letztes Log: %REPO_ROOT%\logs\latest.log
    echo Pruefe ob NVIDIA Driver / VRAM / Vulkan-Layer in Ordnung sind.
)

exit /b %RC%
