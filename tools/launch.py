"""TETRABot launcher — auto-import URDF, set up scene, run interactive Isaac Sim.

USAGE (called via launch.bat):
    launch.bat                  # interactive mode, GUI
    launch.bat --auto-demo      # auto-animate joints (sinusoidal)
    launch.bat --headless       # no GUI (only useful with --auto-demo)

DIRECT (without bat):
    "C:\\isaac-sim\\python.bat" tools\\launch.py [args]

Logs go to logs/launch.log. Errors get full tracebacks.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

# ---- Args & logging (BEFORE SimulationApp, since SimulationApp eats stdout) ----

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser(description="TETRABot Isaac Sim launcher")
parser.add_argument("--headless", action="store_true", help="No GUI (server mode)")
parser.add_argument(
    "--livestream",
    action="store_true",
    help="Enable WebRTC livestream (omni.kit.livestream.webrtc) so the scene "
         "can be viewed in a browser. Use on headless/cloud servers; pair with "
         "--headless (no local window).",
)
parser.add_argument("--auto-demo", action="store_true", help="Animate joints sinusoidally")
parser.add_argument(
    "--camera-cycle",
    action="store_true",
    help="Enable the scripted phase-anchored camera switching in cabin_assembly "
         "(9 cinematic anchor cams across the choreography). Default OFF — the "
         "freely-navigated viewport camera is kept on Play.",
)
parser.add_argument(
    "--no-camera-cycle",
    action="store_true",
    help="Deprecated no-op (camera cycling is OFF by default). Kept for backward "
         "compatibility with older launch commands.",
)
parser.add_argument(
    "--urdf",
    default=str(REPO_ROOT / "urdf" / "tetrabot.urdf"),
    help="Path to URDF file",
)
parser.add_argument(
    "--scene",
    choices=("none", "cabin"),
    default="none",
    help="Spawn an environment around the robot. 'cabin' loads A320_AIX.usd",
)
parser.add_argument(
    "--cameras",
    action="store_true",
    help="Add a Camera prim at the robot's sensorhead position. "
         "View via Window > Viewport > New Viewport > set Camera dropdown "
         "to /tetrabot/chassis_link/d435i_camera.",
)
parser.add_argument(
    "--keyboard",
    action="store_true",
    help="Drive the robot interactively with WASD (base x/y), Q/E (yaw), R/F "
         "(lift). Implies GUI mode and exclusive of --auto-demo.",
)
parser.add_argument(
    "--ros",
    action="store_true",
    help="Enable ROS2 bridge — publish D435i camera as ROS2 topics "
         "/tetrabot/camera/rgb, /depth, /camera_info. Implies --cameras. "
         "Requires ROS2 (Humble) to be installed on the system.",
)
parser.add_argument(
    "--sdg",
    action="store_true",
    help="Enable Synthetic Data Generation — write RGB+Depth+Instance+BBox2D "
         "frames to logs/sdg_<timestamp>/ via Replicator BasicWriter. "
         "Implies --cameras.",
)
parser.add_argument("--frames", type=int, default=0, help="Auto-stop after N frames (0 = run forever)")
parser.add_argument(
    "--controller",
    choices=("hand", "rl"),
    default="hand",
    help="TETRABot chassis controller. 'hand' = built-in P-controller "
         "(deterministic, used in scenario state-machine). 'rl' = route "
         "through tools/rl_policy.py.TetraLocomotionPolicy (currently a "
         "stub — same behaviour but exercises the inference plumbing for "
         "future trained-policy integration; see tools/train_g1_"
         "locomotion.py for the training scaffold).",
)
parser.add_argument(
    "--rl-weights",
    default=None,
    help="Path to .pt/.onnx weights for --controller=rl. If unset or "
         "missing, falls back to stub. Once train_g1_locomotion.py "
         "produces real weights, point this at e.g. models/tetrabot_"
         "locomotion_v1.pt.",
)
parser.add_argument(
    "--low-vram",
    action="store_true",
    help="Mitigation for GPU OOM (cudaErrorMemoryAllocation, "
         "HydraEngine::render failed). Reduces SimulationApp render "
         "resolution from the Kit default (1280x720) to 640x360, which "
         "cuts framebuffer + RT acceleration structure VRAM by ~75%. "
         "Default-off so demo-recording quality stays intact; Phase-B "
         "IK iteration recommends turning it on. Independent from the "
         "--cameras flag (you can combine: low-res cameras work).",
)
parser.add_argument(
    "--num-tetrabots",
    type=int,
    default=1,
    help="Number of TETRABot instances to spawn at /World/tetrabot_<i>. "
         "Default 1 (backwards-compat single-robot). Demo target = 4. "
         "With N>1, --cameras/--ros/--sdg currently only attach to robot_0.",
)
parser.add_argument(
    "--workpiece",
    action="store_true",
    help="Spawn a Euro-pallet with two cardboard boxes on top at scene "
         "centre as target for the multi-TETRABot pickup demo. The pallet "
         "is what TETRABots fixed-joint to and transport; the boxes ride "
         "along via friction and are picked up individually by the humanoid "
         "(G1) at the end of the run.",
)
parser.add_argument(
    "--scenario",
    choices=("none", "cabin_assembly"),
    default="none",
    help="Run a scripted scenario. 'cabin_assembly' = 4 TETRABots converge "
         "on the Euro-pallet, lift it cooperatively, transport to the "
         "delivery point, release. Implies --num-tetrabots>=4, --workpiece, "
         "and --scene cabin. Excludes --auto-demo / --keyboard.",
)
parser.add_argument(
    "--g1", dest="g1", action="store_true", default=None,
    help="Deploy a Unitree G1 humanoid at the pallet delivery point. After "
         "the bots release the pallet, G1 picks up box_a via a scripted "
         "kinematic lift. Default: ON when --scenario=cabin_assembly, "
         "OFF otherwise.",
)
parser.add_argument(
    "--no-g1", dest="g1", action="store_false",
    help="Suppress G1 deployment even in cabin_assembly scenario.",
)
parser.add_argument(
    "--g1-usd",
    default=str(REPO_ROOT / "assets" / "environment" / "g1_humanoid.usd"),
    help="Path to the G1 humanoid USD file (default: project-bundled "
         "g1_humanoid.usd, copied from dominik-dev/isaaclab_arena/assets).",
)
parser.add_argument(
    "--g1-z-offset",
    type=float,
    default=0.0,
    help="Fine-tune Z offset (m) added on top of the computed spawn-Z "
         "formula `GROUND_PLANE_Z + G1_FOOT_TO_PELVIS + epsilon`. Default "
         "0.0 = use the FK-computed standing-pose pelvis height (0.784m). "
         "Override if your USD pose differs from the homie default_angles.",
)
# =============================================================================
# GROUND_PLANE_Z — IMMUTABLE — locked 2026-05-07 per user directive.
# DO NOT CHANGE without explicit user instruction.
#
# This is the canonical world-Z of the working surface. The ground plane
# collider sits at this Z, and TETRABot wheel-bottoms rest on it via gravity.
# It was determined by visual alignment with the A30X cabin's passenger floor
# (Hauptkoerper passenger fixtures begin at z=0.792, SPANTENMODELL frame ribs
# end at z=0.895 — see inspect_cabin_floor.py output). Pallet+boxes settle
# onto this plane via physics.
#
# `--cabin-floor-z` CLI flag below remains overridable for diagnostics
# (drop-mode calibration, ground-plane shift tests). The constant here is
# the LOCKED value the project standardizes on.
# =============================================================================
GROUND_PLANE_Z = 0.79  # IMMUTABLE — locked 2026-05-07 per user directive.

parser.add_argument(
    "--cabin-floor-z",
    type=float,
    default=GROUND_PLANE_Z,
    help="World Z of the working-surface (m). Robots' wheel-bottom and "
         "the ground plane sit at this Z. Default = GROUND_PLANE_Z "
         "(locked 2026-05-07) which matches the visible cabin floor "
         "(Hauptkoerper passenger fixtures begin at z=0.792, "
         "SPANTENMODELL frame ribs end at z=0.895). This is safe ONLY "
         "because the cabin defaults to --no-cabin-collision: the "
         "cabin is visual-only and the ground plane is the sole "
         "collision surface, so robots can't penetrate the structural "
         "ribs. The constant is locked; this CLI override stays for "
         "diagnostic ground-shift tests only. If you turn collision "
         "back on (--cabin-collision), tune this down to ~0.252 to "
         "avoid PhysX solver explosions from initial inter-penetration "
         "with the ribs.",
)
parser.add_argument(
    "--cabin-translate-z",
    type=float,
    default=GROUND_PLANE_Z,
    help="Cabin USD translate Z (m). Default tracks GROUND_PLANE_Z so the "
         "cabin USD's local origin sits at ground-plane height — assuming "
         "the cabin model's interior floor is at local z=0. (Previous "
         "default 0.4 was tuned for an older ground-plane Z and got out "
         "of sync after the GROUND_PLANE_Z=0.79 standing-order lock; user "
         "report 2026-05-11: 'kabine ist viel zu tief im boden'.) "
         "Override this flag if the cabin model has a non-zero floor offset.",
)
parser.add_argument(
    "--cabin-collision",
    action="store_true",
    help="Apply MeshCollisionAPI to all 53 cabin meshes (CATIA export). "
         "OFF by default: the cabin is visual-only and physics happens "
         "only against the ground plane. ON treats every wall, frame "
         "rib, seat, and panel as a triangle-mesh collider — exact but "
         "FRAGILE. With ON the SPANTENMODELL ribs at z=0.493..0.895 "
         "explode the PhysX articulation solver if any robot/pallet "
         "spawns inside that volume (artiPropagate* kernels fail, "
         "cascading into CUDA OOM during pickup). Only enable if you "
         "are sure your --cabin-floor-z places everything safely "
         "below the rib volume (e.g. 0.252).",
)
parser.add_argument(
    "--drop-mode",
    action="store_true",
    help="Calibration: import robots with fix_base=False and spawn "
         "everything 1.5m above the cabin top so it all FALLS under "
         "gravity when Play is pressed. Use this to find the natural "
         "settled positions on the cabin floor — the post-fall world Z "
         "of each prim is logged every 60 frames so you can read it off "
         "and hard-code the right --cabin-floor-z. Skips --scenario.",
)
args = parser.parse_args()

# --ros and --sdg both need a camera prim, so imply --cameras.
if (args.ros or args.sdg) and not args.cameras:
    args.cameras = True

# --scenario cabin_assembly implies the full demo set.
if args.scenario == "cabin_assembly":
    args.num_tetrabots = max(args.num_tetrabots, 4)
    args.workpiece = True
    if args.scene == "none":
        args.scene = "cabin"
    if args.g1 is None:
        args.g1 = True   # default ON for cabin_assembly
elif args.g1 is None:
    args.g1 = False

# --drop-mode is calibration only: skip scenario, keep workpiece + cabin
# so the user sees the full set falling. Force into "interactive mode"
# (no auto loops) so timeline only advances on Play.
if args.drop_mode:
    args.scenario = "none"
    args.auto_demo = False
    args.keyboard = False
    args.workpiece = True
    if args.scene == "none":
        args.scene = "cabin"
    args.num_tetrabots = max(args.num_tetrabots, 4)

log_path = LOG_DIR / f"launch_{time.strftime('%Y%m%d_%H%M%S')}.log"

# Symlink/copy the latest log under a stable name for easy tailing
latest_log = LOG_DIR / "latest.log"
try:
    if latest_log.exists() or latest_log.is_symlink():
        latest_log.unlink()
except OSError:
    pass

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)5s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("tetrabot")
log.info("=" * 70)
log.info(f"TETRABot launcher starting | log file: {log_path}")
log.info(f"Args: {vars(args)}")

try:
    import shutil
    shutil.copyfile(log_path, latest_log)
except OSError:
    pass


# ---- Launch Isaac Sim (must come BEFORE any omni imports) ----

try:
    from isaacsim import SimulationApp
except ImportError as e:
    log.error("Cannot import isaacsim — are you running this via Isaac Sim's python.bat?")
    log.error(f"Tried import path: {sys.path}")
    raise

# Force single-GPU mode. The system has NVIDIA RTX 5060 Ti + AMD Radeon iGPU
# (mixed vendor). With multi_gpu=True (Isaac Sim default) the renderer tries
# to enumerate both, and rtx.scenedb.plugin crashes at carbOnPluginStartup
# during init. Pinning to GPU 0 (the NVIDIA card) avoids that.
# --livestream forces SimulationApp into windowless mode (no local OS window
# on a headless/cloud server) WITHOUT setting args.headless — so the
# interactive GUI code paths (wait-for-Play, replay, camera cycling, HUD)
# stay active and the demo is fully controllable from the browser viewer.
_sim_cfg = {
    "renderer": "RaytracedLighting",
    "headless": args.headless or args.livestream,
    "multi_gpu": False,
    "active_gpu": 0,
}
if args.livestream:
    # Match NVIDIA's official standalone livestream example 1:1
    # (standalone_examples/api/isaacsim.simulation_app/livestream.py):
    # the streamed framebuffer needs an explicit window size and the UI must
    # be SHOWN (hide_ui=False). Without these the base python experience
    # launches with hideUi=1 and no window size, so no composited frame is
    # produced to encode → the browser viewer stays on "waiting for stream".
    _sim_cfg["hide_ui"] = False
    _sim_cfg["window_width"] = 1920
    _sim_cfg["window_height"] = 1080
    _sim_cfg["display_options"] = 3286   # show default grid

    # CRITICAL: enable the livestream extension as a KIT LAUNCH ARGUMENT, not
    # via runtime enable_extension() after init. Isaac Lab's AppLauncher does
    # exactly this for `--livestream 2` (WebRTC private) and notes: "some of
    # the extensions only work when launched with the kit file". Runtime
    # enabling is why every earlier attempt connected but stayed on "waiting
    # for stream". argparse already ran above, so appending to sys.argv now is
    # safe; SimulationApp forwards these through to the kit process.
    sys.argv += ["--enable", "omni.services.livestream.nvcf"]
if args.low_vram:
    # Phase B-0 (2026-05-11) mitigation: drop framebuffer res from
    # Kit-default 1280x720 to 640x360 — opt-in, default-off so demo
    # recordings keep their quality. Reduces RT acceleration-structure
    # VRAM significantly so the cabin+TETRABots+Pallet+Boxes+G1 scene
    # fits in 16GB without "Failed to allocate ... LdrColor resource".
    _sim_cfg["width"]  = 640
    _sim_cfg["height"] = 360
kit = SimulationApp(_sim_cfg)
log.info(f"SimulationApp ready (headless={args.headless}, multi_gpu=False, "
         f"gpu=0, low_vram={args.low_vram}, "
         f"res={_sim_cfg.get('width', 'kit-default')}x"
         f"{_sim_cfg.get('height', 'kit-default')})")

if args.livestream:
    # omni.services.livestream.nvcf is already enabled via the --enable kit
    # launch arg above (the only reliable way — see comment there). Here we
    # only apply the runtime settings NVIDIA's standalone example sets after
    # init. drawMouse renders the cursor into the stream; ngx(DLSS) off avoids
    # an extra headless-encode hop.
    kit.set_setting("/app/window/drawMouse", True)
    kit.set_setting("/ngx/enabled", False)
    kit.update()
    log.info("NVCF livestream enabled via kit --enable arg — open ${HOST}/viewer/ "
             "in the browser, wait for the scene to finish loading, then RELOAD "
             "that tab and press Play")


def main() -> int:
    # ---- Now omni imports work ----
    import omni.kit.commands
    import omni.timeline
    import omni.usd
    from isaacsim.core.prims import Articulation

    # ---- Scene-floor reference (single source of truth) ----
    # All robots / workpiece use CABIN_FLOOR_Z as their world Z reference.
    # When --scene cabin: A30X_AllCATPart.usd is loaded at translate
    # (-3.81, 0, 0.15409), scale 0.001 (per user-provided transform). The
    # floor inside the cabin then sits at world Z = CABIN_FLOOR_Z. Tune
    # below if the cabin USD's floor doesn't quite match.
    # When --scene none: ground plane at world Z = 0.
    if args.scene == "cabin":
        CABIN_TRANSLATE_X = -3.81
        CABIN_TRANSLATE_Y = 0.0
        CABIN_TRANSLATE_Z = args.cabin_translate_z   # cabin USD translate Z
        CABIN_SCALE       = 0.001
        CABIN_FLOOR_Z     = args.cabin_floor_z       # robots/pallet/groundplane Z
    else:
        CABIN_TRANSLATE_X = 0.0
        CABIN_TRANSLATE_Y = 0.0
        CABIN_TRANSLATE_Z = 0.0
        CABIN_SCALE       = 1.0
        CABIN_FLOOR_Z     = 0.0       # ground plane level
    log.info(f"CABIN_TRANSLATE_Z = {CABIN_TRANSLATE_Z:.4f} m (cabin USD)")
    log.info(f"CABIN_FLOOR_Z     = {CABIN_FLOOR_Z:.4f} m (robots/pallet/groundplane)")
    from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdLux, UsdPhysics, UsdShade

    def Usd_prim_iter(root):
        """Recursive prim iterator covering root and all descendants."""
        yield root
        for child in root.GetAllChildren():
            yield from Usd_prim_iter(child)

    # ---- Verify URDF exists ----
    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        log.error(f"URDF not found: {urdf_path}")
        return 1
    log.info(f"URDF: {urdf_path} ({urdf_path.stat().st_size} bytes)")

    # ---- Build URDF import config ----
    log.info("Creating URDF import config")
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        log.error("URDFCreateImportConfig failed")
        return 1
    import_config.merge_fixed_joints = False  # keep camera frames as separate prims
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    # fix_base controls whether base_link is pinned to world (kinematic) or
    # gets a free 6-DoF joint (full rigid body). Default False since
    # 2026-05-07 so gravity acts on the whole articulation: bots fall and
    # land on the ground plane via wheel-sphere collision instead of
    # floating at their spawn position when the ground is removed/lowered.
    # base_link has 0.1 kg + diagonal inertia in the URDF so PhysX has a
    # well-defined free root. --drop-mode is now redundant (everything
    # falls by default) but kept for the high-spawn calibration test.
    import_config.fix_base = False
    import_config.distance_scale = 1.0
    import_config.self_collision = False
    import_config.create_physics_scene = False  # we create our own below

    # ---- Spawn N TETRABots at /World/tetrabot_<i> ----
    # Each spawn imports the same URDF to a unique dest_path with a unique
    # XY translate. The Z translate of -0.058 puts the wheels on the floor
    # (sphere-collider bottom at chassis-frame z=0.058 → world z=0).
    from pxr import UsdGeom
    stage = omni.usd.get_context().get_stage()

    def _spawn_positions(n: int) -> list[tuple[float, float]]:
        """Spread N robots across the cabin floor without colliding initially."""
        # Cabin-assembly scenario: spawn in a ROW EAST of the cabin so the
        # waypoint-follower demo can play row-approach → split → dock →
        # enter-cabin (westward) choreography. East-wall removed (see
        # wall_specs above) so the bots+pallet have a clear corridor west
        # into the cabin via the big fuselage opening at X+. Pallet's long
        # axis (Y, 1.2m) fits inside the cabin's Y-corridor (±1.8m) easily;
        # the Y=±1.8 passenger doors would have been too narrow.
        if args.scenario == "cabin_assembly" and n >= 4:
            row_x = +5.5
            row_ys = [-1.5, -0.5, +0.5, +1.5]
            return [(row_x, y) for y in row_ys[:n]]
        if n <= 1:
            return [(0.0, 0.0)]
        if n == 2:
            return [(-0.75, 0.0), (+0.75, 0.0)]
        if n == 3:
            return [(-1.5, 0.0), (0.0, 0.0), (+1.5, 0.0)]
        # 2x2 grid for n=4 (demo target).
        grid = [(-1.0, +1.0), (+1.0, +1.0), (-1.0, -1.0), (+1.0, -1.0)]
        if n <= 4:
            return grid[:n]
        # n > 4: 3x3-ish — not optimised, just don't crash
        out = []
        side = int(n ** 0.5) + 1
        for i in range(n):
            r, c = divmod(i, side)
            out.append(((c - (side - 1) / 2.0) * 1.5, (r - (side - 1) / 2.0) * 1.5))
        return out

    n_robots = max(1, args.num_tetrabots)
    positions = _spawn_positions(n_robots)
    log.info(f"Spawning {n_robots} TETRABot(s) at: {positions}")

    # NOTE on multi-instance URDF import:
    # URDFParseAndImportFile's `dest_path` arg is a *filesystem* path to a USD
    # file (the importer creates a new USD layer at that path), NOT a stage
    # prim path. Trying to use a stage path causes "Failed verification:
    # fileFormat" errors at SDF-Layer creation.
    #
    # MovePrim AFTER import also doesn't work cleanly: the PhysX articulation
    # is registered against the original prim path; moving the prim leaves
    # the physics binding stale -> Articulation.initialize hits
    # `physics_view._backend is None`.
    #
    # Cleanest approach: rewrite the URDF's <robot name="..."> attribute per
    # spawn so the importer derives a unique prim path automatically. URDF
    # mesh refs use relative paths (../assets/...), so we write the temp URDF
    # back into the same dir as the original to keep mesh resolution working.

    def _make_unique_urdf(src: Path, idx: int) -> Path:
        text = src.read_text(encoding="utf-8")
        new_name = f"tetrabot_{idx}"
        # Replace only the root <robot name="...">; mesh paths and link names
        # stay untouched so relative URLs and joint references still resolve.
        new_text, n = re.subn(
            r'(<robot\s+name=)"[^"]*"',
            f'\\1"{new_name}"',
            text,
            count=1,
        )
        if n != 1:
            raise RuntimeError(f"URDF root rename failed: {src} has no <robot name=...>")
        out = src.parent / f".__tetrabot_spawn_{idx}.urdf"
        out.write_text(new_text, encoding="utf-8")
        return out

    robot_paths: list[str] = []         # /tetrabot_<i>
    art_root_paths: list[str] = []      # importer-returned prim path
    tmp_urdfs: list[Path] = []          # cleanup list
    try:
        for i, (sx, sy) in enumerate(positions):
            tmp_urdf = _make_unique_urdf(urdf_path, i)
            tmp_urdfs.append(tmp_urdf)

            # Important: reset stage's default-prim before each import. The
            # URDF importer otherwise inserts the next robot UNDER the previous
            # default-prim, producing nested articulations
            # (/tetrabot_0/tetrabot_1/root_joint instead of /tetrabot_1/...).
            try:
                stage.ClearDefaultPrim()
            except Exception:
                pass  # API may not exist on all USD versions; non-fatal

            status, art_root_path = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(tmp_urdf),
                import_config=import_config,
                get_articulation_root=True,
            )
            if not status:
                log.error(f"  TETRABot {i}: URDFParseAndImportFile failed -> skipping")
                continue

            # Trust the importer's returned path — derive instance_root from
            # actual returned art_root_path (e.g. "/tetrabot_0/root_joint" ->
            # "/tetrabot_0"). Don't assume a name based on URDF root.
            if art_root_path.endswith("/root_joint"):
                instance_root = art_root_path[: -len("/root_joint")]
            else:
                instance_root = art_root_path.rsplit("/", 1)[0]
            log.info(f"  TETRABot {i}: imported at {instance_root}, art_root={art_root_path}")

            # Translate xform to spawn position. fix_base=False is now the
            # default (gravity-aware), so spawn-Z places the wheels just
            # at ground contact and gravity pulls the bot the last few
            # millimetres onto the surface. Z = cabin floor minus 0.058
            # so the wheel sphere colliders (bottom at chassis frame
            # z=0.058) start in light contact with the floor. --drop-mode
            # spawns high (3.0 m) for the multi-second free-fall test.
            if args.drop_mode:
                robot_z = 3.0
            else:
                robot_z = CABIN_FLOOR_Z - 0.058
            robot_top = stage.GetPrimAtPath(instance_root)
            if robot_top.IsValid():
                rx = UsdGeom.Xformable(robot_top)
                rx.ClearXformOpOrder()
                rx.AddTranslateOp().Set(Gf.Vec3d(sx, sy, robot_z))
                # Per-bot spawn yaw rotation (re-attempted 2026-05-11):
                # User report: bot_0 and bot_2 visually face the wrong
                # direction. They are on the West side (sx<0) — with
                # default yaw=0 the bot's "front face" (per chassis
                # mesh orientation) points away from the pallet at world
                # origin. Rotating these two bots 180° around the
                # vertical Z axis flips their forward direction so all
                # 4 bots face the pallet symmetrically.
                # Earlier attempt (2026-05-08) broke the velocity
                # controller — bot Z drifted below ground, X ran away.
                # That was with dynamic pallet + docking collision +
                # no chassis collision; the architecture has since
                # changed (kinematic pallet, no docking collision,
                # chassis collision restored), so the failure mode may
                # not reproduce. Telemetry will confirm.
                if sx < 0:
                    rx.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble
                                   ).Set(Gf.Quatd(0.0, 0.0, 0.0, 1.0))
                log.info(f"    placed at world (x={sx:.2f}, y={sy:.2f}, "
                         f"z={robot_z:.3f}, "
                         f"yaw={'180°' if sx < 0 else '0°'})")
            else:
                log.warning(f"  TETRABot {i}: {instance_root} prim invalid after import")

            robot_paths.append(instance_root)
            art_root_paths.append(art_root_path)
    finally:
        # Always clean up temp URDFs even if an import raised mid-loop.
        for tmp in tmp_urdfs:
            try:
                tmp.unlink()
            except OSError:
                pass

    if not robot_paths:
        log.error("No TETRABots could be spawned — aborting.")
        return 1
    log.info(f"Spawned {len(robot_paths)}/{n_robots} TETRABot(s) successfully")

    # World-Z-lock joint REMOVED (URDF v3 refactor 2026-05-08): The previous
    # Ridgeback planar-dummy pattern needed it as anchor for the planar drive
    # forces to translate into world-frame motion. With chassis merged into
    # base_link as the single 6-DoF articulation root and direct velocity
    # control via art.set_world_velocities(), no anchor is needed — gravity
    # plus wheel-ground contact handle Z stability, and the velocity setter
    # bypasses the constraint coupling that was capping X-tracking at ~25%.

    # Backwards-compat alias for code below that still references a single robot
    # (cameras, ROS, SDG, log messages). All those paths point at robot 0.
    prim_path = art_root_paths[0]

    # ---- Physics scene ----
    log.info("Setting up physics scene + ground plane + light")
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    physx_scene = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableStabilizationAttr(True)
    physx_scene.CreateSolverTypeAttr("TGS")

    # Ground plane: lifted to CABIN_FLOOR_Z so it acts as the cabin's
    # passenger-floor surface. ALWAYS present — even in --drop-mode.
    # The A30X cabin USD is a fuselage shell with 53 thin CATIA meshes
    # (frame ribs, panels, seats) which make terrible PhysX colliders;
    # by default --cabin-collision is OFF and the cabin is visual-only,
    # so the ground plane is the SOLE collision surface. TETRABots drive
    # on it like on a polished floor; pallet+boxes settle onto it.
    # Ground plane radius enlarged from 5.0 → 30.0 (2026-05-11) so the
    # waypoint-follower scenario can spawn bots+pallet OUTSIDE the cabin
    # (south of cabin, ~6m away from cabin edge) and have them drive into
    # the cabin. The previous 5m disc would have left bots floating off
    # the edge.
    PhysicsSchemaTools.addGroundPlane(
        stage, "/groundPlane", "Z", 30.0,
        Gf.Vec3f(0, 0, CABIN_FLOOR_Z), Gf.Vec3f(0.5)
    )
    log.info(f"  Ground plane at world z={CABIN_FLOOR_Z:.4f} "
             f"(sole collision surface unless --cabin-collision is set)")

    # Cinematic lighting (2026-05-13 polish-sprint): brighter + more
    # cabin-interior fill via rect lights. Default values were too dim
    # for demo recording (cabin interior looked grey/flat).
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distant_light.CreateIntensityAttr(5500)   # was 3000
    # Slight angle from above-and-behind for natural shadows.
    UsdGeom.Xformable(distant_light).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
    dome_light.CreateIntensityAttr(1800)      # was 800
    # Two interior rect lights at cabin ceiling height to brighten the
    # scene where it matters (where the pallet+G1+box action happens).
    # Cabin interior is around X[-3, 0], Y[-1.8, 1.8], Z[0.79..2.91].
    for name, x, y in (("CabinLight_N", 0.0, +1.2),
                       ("CabinLight_S", 0.0, -1.2)):
        rect = UsdLux.RectLight.Define(stage, Sdf.Path(f"/{name}"))
        rect.CreateIntensityAttr(8000)
        rect.CreateWidthAttr(2.0)
        rect.CreateHeightAttr(0.6)
        rxf = UsdGeom.Xformable(rect)
        rxf.ClearXformOpOrder()
        rxf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 2.7))
        rxf.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))  # face down

    # Cinematic cameras (2026-05-13 polish-sprint): pre-defined viewpoints
    # for demo recording. After Play starts, switch cameras via the
    # viewport's camera dropdown (top-left of viewport) — pick any of
    # /World/cameras/<name>. Default Perspective stays available too.
    cams_xform = stage.DefinePrim("/World/cameras", "Xform")
    cam_defs = [
        # (name, translate, rotateXYZ_deg, focal_length, description)
        ("cam_overview",
         Gf.Vec3d(+6.0, +5.0, +3.0), Gf.Vec3f(-65.0, 0.0, +135.0), 18.0,
         "wide isometric over scene"),
        ("cam_pallet_east",
         Gf.Vec3d(+8.0, +0.0, +1.8), Gf.Vec3f(-85.0, 0.0, +90.0), 24.0,
         "east-side: watch pallet transport in"),
        ("cam_g1_overshoulder",
         Gf.Vec3d(-2.4, -0.9, +1.6), Gf.Vec3f(-80.0, 0.0, +45.0), 28.0,
         "over G1's right shoulder during pickup"),
        ("cam_top_down",
         Gf.Vec3d(+1.0, +0.0, +5.5), Gf.Vec3f(-90.0, 0.0, +90.0), 14.0,
         "bird's-eye top-down of full choreography"),
        ("cam_inside_cabin",
         Gf.Vec3d(-2.8, -0.0, +1.6), Gf.Vec3f(-85.0, 0.0, +90.0), 22.0,
         "from inside cabin looking east (toward entry)"),
    ]
    for cname, pos, rot, focal, desc in cam_defs:
        cpath = f"/World/cameras/{cname}"
        cam = UsdGeom.Camera.Define(stage, cpath)
        cam.CreateFocalLengthAttr(focal)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(11.78)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
        cxf = UsdGeom.Xformable(cam)
        cxf.ClearXformOpOrder()
        cxf.AddTranslateOp().Set(pos)
        cxf.AddRotateXYZOp().Set(rot)
        cam.GetPrim().GetAttribute("purpose").Set("default")
    log.info(f"  Lighting + cameras: DistantLight=5500, DomeLight=1800, "
             f"2 RectLights @ ceiling, {len(cam_defs)} cinematic cameras "
             f"under /World/cameras/")

    # ---- Scene environment (optional) ----
    if args.scene == "cabin":
        # Switched from A320_AIX.usd to A30X_AllCATPart.usd (CATIA STEP-export
        # converted via tools/convert_step_to_usd.py). Transform is the
        # user-supplied set: translate (-3.81, 0, 0.15409), scale 0.001.
        cabin_usd = REPO_ROOT / "assets" / "environment" / "A30X_AllCATPart.usd"
        if cabin_usd.exists():
            log.info(f"Adding cabin scene from {cabin_usd}")
            cabin_prim = stage.DefinePrim("/World/cabin", "Xform")
            cabin_prim.GetReferences().AddReference(str(cabin_usd))
            from pxr import UsdGeom
            xformable = UsdGeom.Xformable(cabin_prim)
            xformable.ClearXformOpOrder()
            translate_op = xformable.AddTranslateOp()
            scale_op = xformable.AddScaleOp()
            # Per user screenshot: translate (-3.81, 0, 0.15409), scale 0.001.
            # CABIN_FLOOR_Z constant defined above mirrors this so robots and
            # the pallet land on the cabin's interior floor.
            translate_op.Set(Gf.Vec3f(CABIN_TRANSLATE_X, CABIN_TRANSLATE_Y, CABIN_TRANSLATE_Z))
            scale_op.Set(Gf.Vec3f(CABIN_SCALE, CABIN_SCALE, CABIN_SCALE))
            log.info(f"  Cabin reference attached at {cabin_prim.GetPath()}")

            # Force a stage update so bounds reflect the new transforms, then
            # verify what we've actually placed in the world.
            kit.update()
            cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
            world_bbox = cache.ComputeWorldBound(cabin_prim)
            r = world_bbox.ComputeAlignedRange()
            log.info(
                f"  Cabin world bbox: "
                f"X[{r.GetMin()[0]:+.2f},{r.GetMax()[0]:+.2f}] "
                f"Y[{r.GetMin()[1]:+.2f},{r.GetMax()[1]:+.2f}] "
                f"Z[{r.GetMin()[2]:+.2f},{r.GetMax()[2]:+.2f}]"
            )

            # ---- Cabin collision: opt-in via --cabin-collision ----
            # OFF by default: cabin is pure visual, ground plane handles all
            # physics. Robots drive on the ground plane like on a polished
            # floor; the cabin's frame ribs / seats / panels are decorative.
            # This is the game-engine pattern (CAD geometry → visual only,
            # physics → simple primitives) and avoids the PhysX articulation
            # solver explosions that occur when robot bodies spawn inside the
            # SPANTENMODELL rib volume (z=0.493..0.895). Re-enable only if
            # your --cabin-floor-z keeps everything below z=0.49.
            if args.cabin_collision:
                from pxr import UsdGeom as _UsdGeom
                colliders_added = 0
                for descendant in Usd_prim_iter(cabin_prim):
                    if descendant.IsA(_UsdGeom.Mesh):
                        UsdPhysics.CollisionAPI.Apply(descendant)
                        mca = UsdPhysics.MeshCollisionAPI.Apply(descendant)
                        mca.CreateApproximationAttr().Set("none")
                        colliders_added += 1
                log.info(f"  Cabin collision: triangle-mesh applied to "
                         f"{colliders_added} mesh prim(s); floor = ground "
                         f"plane at z={CABIN_FLOOR_Z:.3f}")
            else:
                log.info(f"  Cabin collision: DISABLED (visual only). "
                         f"Physics surface = ground plane at z={CABIN_FLOOR_Z:.3f}. "
                         f"Pass --cabin-collision to enable triangle-mesh "
                         f"collision on cabin meshes.")
                # Add 4 invisible box wall colliders at the cabin perimeter
                # so bots can't drive through the fuselage during transport.
                # Cabin world bbox is X[-3.27, +3.27] Y[-2.02, +2.02], so we
                # place walls slightly inside (X=±3.0, Y=±1.8) to give a
                # safe corridor that respects the curved fuselage shape.
                # Walls span from ground up to cabin top (z=0.79..2.91).
                # User report 2026-05-11: bots drove through wall during
                # transport. Game-engine pattern: simple primitive walls
                # instead of trying to use the messy CAD mesh collision.
                wall_root = "/World/cabin_walls"
                if not stage.GetPrimAtPath(wall_root).IsValid():
                    UsdGeom.Xform.Define(stage, wall_root)
                wall_z_center = (CABIN_FLOOR_Z + 2.91) / 2  # ~1.85
                wall_z_height = 2.91 - CABIN_FLOOR_Z         # ~2.12
                wall_thickness = 0.05  # 5cm thick invisible walls
                # 2026-05-11 v2: EAST wall REMOVED (was X=+3.0). The cabin's
                # passenger-door (Y=±1.8 sides) is too narrow for a Euro-
                # pallet; the only big enough opening is at one of the
                # fuselage ends. User chose the EAST end (X+ side) as the
                # entry. Walls north/south/west remain so bots can't wander
                # off the cabin corridor once inside.
                wall_specs = [
                    # (name, center_x, center_y, size_x, size_y)
                    ("wall_west",  -3.0,  0.0, wall_thickness, 3.6),
                    ("wall_south",  0.0, -1.8, 6.0, wall_thickness),
                    ("wall_north",  0.0, +1.8, 6.0, wall_thickness),
                ]
                for name, cx, cy, sx, sy in wall_specs:
                    wall_path = f"{wall_root}/{name}"
                    cube = UsdGeom.Cube.Define(stage, wall_path)
                    cube.CreateSizeAttr(1.0)  # unit cube; scale via xformOp
                    xf = UsdGeom.Xformable(cube)
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, wall_z_center))
                    xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, wall_z_height))
                    # Visual: hide the cube (invisible wall).
                    visibility = UsdGeom.Imageable(cube).GetVisibilityAttr()
                    if not visibility:
                        visibility = UsdGeom.Imageable(cube).CreateVisibilityAttr()
                    visibility.Set(UsdGeom.Tokens.invisible)
                    # Collision: enable.
                    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                log.info(f"  Cabin walls: 3 invisible box colliders added at "
                         f"X=-3.0 (east open), Y=±1.8, Z=[{CABIN_FLOOR_Z:.2f}..2.91]")
        else:
            log.warning(f"--scene cabin requested but {cabin_usd} not found")

    # ---- Optional: workpiece (Euro pallet + 2 boxes, target for pickup) ----
    if args.workpiece:
        # Geometry was extracted from Euro_Paletten_Test_2.stl via
        # tools/convert_pallet_to_usd.py — three USDs in assets/environment/.
        # Each USD has its mesh re-centred at xy=(0,0), bottom at z=0, so we
        # place each at its world position via xform translate.
        #
        # Original STL (mm) had pallet-centre at xy=(-529.9, 13.7) and the
        # two boxes at xy=(-564.6, -288.7) and (-546.3, +325.3). Here we
        # recompute relative offsets so the assembly is centred at the
        # scene origin (0, 0).
        env_dir = REPO_ROOT / "assets" / "environment"
        pallet_usd = env_dir / "euro_pallet.usd"
        box_a_usd = env_dir / "pallet_box_a.usd"
        box_b_usd = env_dir / "pallet_box_b.usd"
        for p in (pallet_usd, box_a_usd, box_b_usd):
            if not p.exists():
                log.error(
                    f"Workpiece USD missing: {p}. Run "
                    f"`\"C:\\isaac-sim\\python.bat\" tools\\convert_pallet_to_usd.py` first."
                )
                return 1

        # Heights / offsets derived from STL inspection (see convert script).
        PALLET_HEIGHT = 0.144   # m, pallet thickness incl. blocks
        BOX_DROP = 0.005        # tiny gap so boxes drop onto pallet (settle via physics)
        BOX_A_OFFSET = (-0.035, -0.302)   # XY relative to pallet centre, m
        BOX_B_OFFSET = (-0.016, +0.312)
        # Mass values: real Europalette ~25 kg; cardboard box w/ goods ~10 kg.
        PALLET_MASS = 25.0
        BOX_MASS = 10.0

        def _spawn_rigid_mesh(prim_path: str, ref_usd: Path, world_xyz, mass: float,
                              collision_approx: str) -> None:
            """Reference a USD mesh as a rigid body at world_xyz.

            For approx='convexDecomposition' we additionally apply the PhysX
            convex-decomposition quality knobs so V-HACD doesn't fill in thin
            cavities like the Euro-pallet's fork pockets. Defaults assume a
            solid prim where convexHull is exact (boxes); the pallet should
            override to convexDecomposition so the lift columns can actually
            slide into the pockets and engage the underside of the top deck.
            """
            xform = UsdGeom.Xform.Define(stage, prim_path)
            xform.GetPrim().GetReferences().AddReference(str(ref_usd))
            xf = UsdGeom.Xformable(xform)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*world_xyz))
            prim = xform.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(mass)
            UsdPhysics.CollisionAPI.Apply(prim)
            mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mca.CreateApproximationAttr().Set(collision_approx)
            if collision_approx == "convexDecomposition":
                # Default V-HACD voxel resolution (~64k) is too coarse for
                # ~25mm pallet stringers; bump to 500k and allow up to 64
                # hulls so the algorithm can carve out the two fork pockets
                # instead of merging them with the surrounding stringers.
                pxconv = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                pxconv.CreateMaxConvexHullsAttr(64)
                pxconv.CreateHullVertexLimitAttr(64)
                pxconv.CreateVoxelResolutionAttr(500000)
                pxconv.CreateErrorPercentageAttr(0.5)
                pxconv.CreateShrinkWrapAttr(True)

        # In --drop-mode spawn high above the cabin so everything falls; in
        # normal mode the pallet sits on CABIN_FLOOR_Z directly.
        if args.drop_mode:
            pallet_z = 2.5
            box_z = pallet_z + 0.30   # boxes drop onto pallet
            log.info(f"Spawning Euro-pallet + 2 boxes HIGH (drop-mode) "
                     f"pallet z={pallet_z}, box z={box_z}")
        else:
            pallet_z = CABIN_FLOOR_Z
            box_z = CABIN_FLOOR_Z + PALLET_HEIGHT + BOX_DROP
            log.info(f"Spawning Euro-pallet + 2 boxes at scene centre (floor z={CABIN_FLOOR_Z:.3f})")

        # XY position of pallet centre. cabin_assembly scenario spawns the
        # pallet OUTSIDE the cabin (east, ~3.5m east of scene origin) so the
        # bots in row can drive west, split around it, dock, and transport
        # it westward INTO the cabin via the open east fuselage end
        # (delivery centre at X=-1.0). All other modes keep the pallet at
        # the scene origin.
        if args.scenario == "cabin_assembly":
            pallet_xy = (+3.5, 0.0)
        else:
            pallet_xy = (0.0, 0.0)

        _spawn_rigid_mesh(
            "/World/pallet", pallet_usd, (pallet_xy[0], pallet_xy[1], pallet_z),
            mass=PALLET_MASS, collision_approx="convexDecomposition",
        )
        # Pallet starts kinematic (2026-05-08 review fix). Telemetry showed
        # the dynamic pallet drifted by 50cm in Y BEFORE any bot reached it
        # — likely from box-on-pallet settling friction asymmetry plus
        # convex-decomposed pallet hulls extending into wheel contact zone.
        # Kinematic pallet stays put until PICKUP, when it's already
        # kinematic anyway for the leader-follower carry. RELEASE then
        # toggles back to dynamic and gravity takes over.
        # Drop-mode keeps the pallet dynamic so the original calibration
        # still works (pallet falls onto ground from spawn-Z=2.5m).
        if not args.drop_mode:
            pallet_prim = stage.GetPrimAtPath("/World/pallet")
            rb = UsdPhysics.RigidBodyAPI(pallet_prim)
            attr = rb.GetKinematicEnabledAttr()
            if not attr:
                attr = rb.CreateKinematicEnabledAttr()
            attr.Set(True)
            log.info(f"  pallet at ({pallet_xy[0]:+.3f}, {pallet_xy[1]:+.3f}, "
                     f"{pallet_z:.3f}), mass={PALLET_MASS}kg "
                     f"(collision=convexDecomposition, KINEMATIC at spawn — "
                     f"won't drift before pickup)")
        else:
            log.info(f"  pallet at ({pallet_xy[0]:+.3f}, {pallet_xy[1]:+.3f}, "
                     f"{pallet_z:.3f}), mass={PALLET_MASS}kg "
                     f"(collision=convexDecomposition, dynamic for drop-mode)")

        # ---- Filter cover_link <-> pallet (2026-05-11 v2) -------------
        # cover_link's box collider Z-range (0.16m thick at z=0.24 in
        # chassis frame ≈ world Z 0.90..1.06) overlaps the pallet Z-range
        # (0.79..0.93) by ~30mm at the bot's standing height. During the
        # carry the pallet is kinematic — cover-pallet contact would
        # contribute nothing useful — but after RELEASE the pallet is
        # dynamic again and the cover-pallet contact dragged the pallet
        # north by 0.7m during sidestep + east by 5m during retreat.
        # FilteredPairsAPI on each bot's cover_link relative to
        # /World/pallet eliminates that contact entirely. Boxes-on-pallet
        # remain unfiltered so they still rest on the pallet.
        from pxr import Sdf
        for robot_path in robot_paths:
            cover_prim = stage.GetPrimAtPath(f"{robot_path}/cover_link")
            if not cover_prim.IsValid():
                continue
            fpapi = UsdPhysics.FilteredPairsAPI.Apply(cover_prim)
            rel = fpapi.CreateFilteredPairsRel()
            rel.AddTarget(Sdf.Path("/World/pallet"))
        log.info(f"  cover_link<->pallet collision filter: applied to "
                 f"{len(robot_paths)} bots (no cover-drag on dynamic pallet)")

        box_a_world = (pallet_xy[0] + BOX_A_OFFSET[0],
                       pallet_xy[1] + BOX_A_OFFSET[1])
        box_b_world = (pallet_xy[0] + BOX_B_OFFSET[0],
                       pallet_xy[1] + BOX_B_OFFSET[1])
        _spawn_rigid_mesh(
            "/World/box_a", box_a_usd,
            (box_a_world[0], box_a_world[1], box_z),
            mass=BOX_MASS, collision_approx="convexHull",
        )
        log.info(f"  box_a at ({box_a_world[0]:+.3f}, {box_a_world[1]:+.3f}, "
                 f"{box_z:.3f}), mass={BOX_MASS}kg")

        _spawn_rigid_mesh(
            "/World/box_b", box_b_usd,
            (box_b_world[0], box_b_world[1], box_z),
            mass=BOX_MASS, collision_approx="convexHull",
        )
        log.info(f"  box_b at ({box_b_world[0]:+.3f}, {box_b_world[1]:+.3f}, "
                 f"{box_z:.3f}), mass={BOX_MASS}kg")

    # ---- Optional: G1 humanoid at the pallet delivery point -----------
    # Spawns a Unitree G1 as a Xform reference to the bundled
    # g1_humanoid.usd. Positioned 0.7m WEST of the delivery centre,
    # facing east toward the pallet.
    #
    # Physics strategy (v6, after v5's hybrid failed): G1 is FULLY
    # KINEMATIC — every RigidBodyAPI is set to kinematic, every
    # CollisionAPI is disabled, the ArticulationRootAPI is disabled.
    # G1 stands as a static visual prop in the authored standing pose.
    # Justification:
    #   - No locomotion/balance policy is available locally (Homie ONNX
    #     needs Nucleus access). Without one, dynamic G1 falls or wobbles.
    #   - v5 attempted "kinematic pelvis only + dynamic limbs + PD drives"
    #     but the user reports: upper body falls forward, feet wiggle,
    #     pallet explodes when contacting G1's collision shapes.
    #   - Going fully kinematic eliminates ALL three failure modes at
    #     once: no falling (every body is pose-driven), no wiggle (no
    #     dynamics on joints), no explosion (no collision).
    #   - The interaction with tetrabots is preserved through the
    #     scripted box-A pickup (kinematic-set trajectory): box visibly
    #     lifts from the delivered pallet to G1 chest height while G1
    #     stands at the delivery point.
    #
    # Z-spawn (Phase A, 2026-05-11): USD local origin is at the PELVIS
    # (Unitree convention — Arena's spezi env also uses pelvis-Z=0.89
    # against a Z=0 floor). Spawn Z must put the pelvis high enough so
    # the feet sit on the ground plane in the default standing pose.
    #
    # G1_FOOT_TO_PELVIS computed via forward kinematics from Unitree's
    # official g1_body29_hand14.urdf joint origins composed with the
    # default standing-pose angles from
    # offline-dev/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/
    # config/g1_homie_v2.yaml (left/right leg: hip_pitch=-0.1,
    # hip_roll=0, hip_yaw=0, knee=+0.3, ankle_pitch=-0.2, ankle_roll=0;
    # waist=0,0,0). Result: foot-bottom is 0.784 m below the pelvis in
    # the world frame for this pose. Cross-check with Unitree's
    # published 1.32 m total standing height (pelvis-to-foot in the
    # 0.65-0.80 m range): 0.784 fits.
    G1_FOOT_TO_PELVIS = 0.784   # m, source: FK from official URDF+yaml
    G1_FOOT_EPSILON   = 0.005   # m, small gap so feet don't clip the plane
    # G1 stands FAR ENOUGH WEST of pallet that the pallet visually does
    # not pass through G1's body (user report 2026-05-12: "pallet fährt
    # durch G1, palette oder G1 nicht rigid"). Geometry:
    #   Pallet at delivery (-1.0, 0) with half-X=0.4 -> X-range (-1.4, -0.6)
    #   G1 body half-X ≈ 0.15 -> for east-edge < pallet west-edge:
    #     G1_X + 0.15 < -1.4  =>  G1_X < -1.55
    # G1 at (-1.5, 0) gives 5cm clearance. Slightly tight; using -1.55
    # leaves 10cm safety margin. (Old value -1.3 had 0.24m overlap with
    # the pallet, causing the visual through-pass.)
    # Trade-off: arm reach to box is ~0.74m (over the ~0.6m comfortable
    # arm length), so the IK solver returns at-limit poses and the box
    # carries with a small offset. Acceptable for the demo — G1+pallet
    # geometric clearance > IK reach perfection.
    G1_DELIVERY_OFFSET_X = -0.55   # G1 stands this far WEST of pallet centre
    G1_PRIM_PATH = "/World/g1"
    # Forward-declare arm-link prim dict so the IK-setup block (later in
    # main()) can populate it. Helper definitions further below close
    # over this name; without the forward-decl Python treats it as an
    # uninitialised local at the spawn-block usage site.
    g1_arm_link_prims: dict[str, "Usd.Prim"] = {}
    g1_left_arm_link_prims: dict[str, "Usd.Prim"] = {}
    g1_head_prim: list = [None]   # [Usd.Prim] or [None]
    g1_world_xy = (None, None)
    # g1_joint_paths is populated only when arm-animation is in use; v6
    # leaves it empty (G1 fully kinematic, no joint drives). The helpers
    # _g1_set_joint_target_rad / _g1_arm_animate check for emptiness and
    # no-op so they remain harmless dormant code for future iterations.
    g1_joint_paths: dict[str, str] = {}
    if args.g1 and args.scenario == "cabin_assembly":
        g1_usd_path = Path(args.g1_usd)
        if not g1_usd_path.exists():
            log.warning(f"--g1 requested but USD not found at {g1_usd_path}. "
                        f"Skipping G1 deployment.")
        else:
            # Delivery centre is (-1.0, 0). G1 stands at (-1.7, 0).
            g1_xy = (-1.0 + G1_DELIVERY_OFFSET_X, 0.0)
            # Spawn-Z formula (Phase A):
            #   g1_z = GROUND_PLANE_Z + foot_to_pelvis + epsilon + user_offset
            # With defaults this yields 0.79 + 0.784 + 0.005 + 0.0 = 1.579.
            # GROUND_PLANE_Z is IMMUTABLE (Standing Order) — the fix lives
            # on the G1 side, not the ground plane.
            g1_z  = (CABIN_FLOOR_Z + G1_FOOT_TO_PELVIS + G1_FOOT_EPSILON
                     + args.g1_z_offset)
            g1_world_xy = g1_xy
            log.info(f"Spawning G1 humanoid from {g1_usd_path}")
            g1_prim = stage.DefinePrim(G1_PRIM_PATH, "Xform")
            g1_prim.GetReferences().AddReference(str(g1_usd_path))
            g1_xf = UsdGeom.Xformable(g1_prim)
            g1_xf.ClearXformOpOrder()
            g1_xf.AddTranslateOp().Set(Gf.Vec3d(g1_xy[0], g1_xy[1], g1_z))
            # Face EAST toward pallet (yaw=0).
            g1_xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

            # Full-kinematic + no-collision pass over the entire G1 prim
            # hierarchy. Every RigidBodyAPI -> kinematic, every
            # CollisionAPI -> disabled, ArticulationRootAPI -> disabled.
            # Also strip PhysxMimicJointAPI (used on hand-finger mimic
            # joints like L_thumb_distal_joint): without an active
            # articulation, PhysX logs "failed to find internal joint
            # object for PhysxMimicJointAPI ... ensure prim is part of
            # an articulation". Removing the API silences the error.
            from pxr import PhysxSchema as _PhSchema
            n_kin = 0
            n_col = 0
            n_art = 0
            n_mimic = 0
            for p in Usd_prim_iter(g1_prim):
                if p.HasAPI(UsdPhysics.RigidBodyAPI):
                    rb = UsdPhysics.RigidBodyAPI(p)
                    kattr = rb.GetKinematicEnabledAttr()
                    if not kattr:
                        kattr = rb.CreateKinematicEnabledAttr()
                    kattr.Set(True)
                    n_kin += 1
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    col = UsdPhysics.CollisionAPI(p)
                    cattr = col.GetCollisionEnabledAttr()
                    if not cattr:
                        cattr = col.CreateCollisionEnabledAttr()
                    cattr.Set(False)
                    n_col += 1
                if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                    en = p.GetAttribute("physics:articulationEnabled")
                    if not en:
                        en = p.CreateAttribute("physics:articulationEnabled",
                                               Sdf.ValueTypeNames.Bool)
                    en.Set(False)
                    n_art += 1
                # Strip MimicJointAPI (PhysX needs articulation to resolve).
                # MimicJointAPI is a MULTI-APPLY schema (multiple instances
                # per prim, e.g., for L_thumb / L_index linkage groups).
                # Use GetAppliedSchemas() to find instance names like
                # "PhysxMimicJointAPI:left_thumb" and remove each.
                try:
                    applied = list(p.GetAppliedSchemas())
                except Exception:
                    applied = []
                for schema_name in applied:
                    if schema_name.startswith("PhysxMimicJointAPI"):
                        try:
                            p.RemoveAppliedSchema(schema_name)
                            n_mimic += 1
                        except Exception:
                            pass
            log.info(f"  G1 at ({g1_xy[0]:+.3f}, {g1_xy[1]:+.3f}, {g1_z:.3f}), "
                     f"facing east toward delivered pallet")
            log.info(f"  G1 fully kinematic: {n_kin} RB->kinematic, "
                     f"{n_col} collision->off, {n_art} articulation->off, "
                     f"{n_mimic} mimic-joints stripped "
                     f"(static visual; no fall, no explosion, no mimic warnings)")

            # Belt-and-braces: filter pallet+boxes against /World/g1 so
            # NOTHING in the bot/payload set can collide with G1 even if
            # the USD has collision shapes the explicit disable above
            # didn't reach (collision can live on Mesh sub-prims, on
            # PhysxCollisionAPI, etc. — easier to filter from the other
            # side). User report 2026-05-11: pallet+boxes "exploded" when
            # contacting G1; this guarantees the contact never registers.
            for owner_path in ("/World/pallet", "/World/box_a", "/World/box_b"):
                owner_prim = stage.GetPrimAtPath(owner_path)
                if not owner_prim.IsValid():
                    continue
                fpapi = UsdPhysics.FilteredPairsAPI.Apply(owner_prim)
                rel = fpapi.CreateFilteredPairsRel()
                rel.AddTarget(Sdf.Path(G1_PRIM_PATH))
            log.info(f"  collision filter: pallet/box_a/box_b "
                     f"<-> /World/g1 (no contact even if G1 has hidden "
                     f"colliders)")

    # ---- Phase B (2026-05-11): G1 IK chain + scripted-keyframe pickup ---
    # Build the right-arm IK chain (pelvis -> waist -> right shoulder ->
    # right elbow -> right wrist) using ikpy, and pre-compute joint-angle
    # keyframes for the REACH/LIFT/PLACE poses. The scenario loop later
    # interpolates between these keyframes and uses FK to teleport box_a
    # along the wrist trajectory during the carry window. G1 itself stays
    # fully kinematic (per v6 architecture) so the joint angles don't
    # propagate to link visuals; the BOX trajectory is the visible
    # interaction artefact.
    g1_ik_chain = None
    g1_ik_chain_left = None
    g1_keyframes: dict[str, "np.ndarray"] = {}
    g1_left_keyframes: dict[str, "np.ndarray"] = {}
    g1_world_xy_for_ik = g1_world_xy  # alias for clarity in the scenario block
    if args.g1 and args.scenario == "cabin_assembly" and g1_world_xy[0] is not None:
        try:
            sys.path.insert(0, str(REPO_ROOT / "tools"))
            import g1_ik
            import numpy as _np_g1
            g1_urdf_path = REPO_ROOT / "assets" / "urdf" / "g1_body29_hand14.urdf"
            if not g1_urdf_path.exists():
                log.warning(f"G1 IK: URDF not found at {g1_urdf_path}; "
                            f"skipping keyframe precompute.")
            else:
                g1_ik_chain = g1_ik.build_right_arm_chain(str(g1_urdf_path))
                log.info(f"  G1 IK chain built: {len(g1_ik_chain.links)} "
                         f"links (right arm + waist)")
                # Discover arm-link prims under /World/g1 by walking the
                # subtree for matching link names. Used by manual-FK
                # animation to set per-link USD xform each frame.
                # Also discover left-arm-link prims + head_link in the
                # same pass (Phase B+ 2026-05-12: head-tracking + left
                # arm mirror for visual "alive" cues).
                for p in Usd_prim_iter(g1_prim):
                    pname = p.GetName()
                    if pname in g1_ik.ARM_LINK_NAMES:
                        g1_arm_link_prims[pname] = p
                    elif pname in g1_ik.LEFT_ARM_LINK_NAMES:
                        g1_left_arm_link_prims[pname] = p
                    elif pname == "head_link":
                        g1_head_prim[0] = p
                log.info(f"  G1 arm-link prims discovered: "
                         f"{len(g1_arm_link_prims)}/{len(g1_ik.ARM_LINK_NAMES)} "
                         f"({list(g1_arm_link_prims.keys())})")
                log.info(f"  G1 left-arm-link prims: "
                         f"{len(g1_left_arm_link_prims)}/"
                         f"{len(g1_ik.LEFT_ARM_LINK_NAMES)}")
                log.info(f"  G1 head_link: "
                         f"{'found' if g1_head_prim[0] is not None else 'MISSING'}")

                # Critical fix (2026-05-13 v2): the 19:00 attempt at
                # disabling RigidBodyAPI on 14 animated link prims
                # triggered "cannot create a joint between static bodies"
                # PhysX errors because G1's finger-joints (under
                # /World/g1/joints/, e.g. R_ring_intermediate_joint) still
                # existed and now connected two static prims. The cascade
                # of PhysX failures caused GPU CUDA OOM during scene init
                # and broke ALL physics (bots, pallet, everything).
                #
                # Proper solution: REMOVE the joint prims first (they're
                # all unused since the articulation is disabled), THEN
                # safely disable RigidBodyAPI on the animated links. With
                # no joints to enforce, USD xform writes on the disabled-
                # RB link prims propagate cleanly to Hydra.
                joints_root = stage.GetPrimAtPath(f"{G1_PRIM_PATH}/joints")
                n_joints_removed = 0
                if joints_root.IsValid():
                    for jp in list(joints_root.GetChildren()):
                        try:
                            stage.RemovePrim(jp.GetPath())
                            n_joints_removed += 1
                        except Exception:
                            pass
                log.info(f"  G1 joints removed: {n_joints_removed} "
                         f"(prevents static-body joint errors; G1 is "
                         f"pure-visual now)")

                animated_names = (set(g1_ik.ARM_LINK_NAMES)
                                  | set(g1_ik.LEFT_ARM_LINK_NAMES)
                                  | {"head_link"})
                n_disabled = 0
                for p in Usd_prim_iter(g1_prim):
                    if p.GetName() in animated_names:
                        if p.HasAPI(UsdPhysics.RigidBodyAPI):
                            rb = UsdPhysics.RigidBodyAPI(p)
                            attr = rb.GetRigidBodyEnabledAttr()
                            if not attr:
                                attr = rb.CreateRigidBodyEnabledAttr()
                            attr.Set(False)
                            n_disabled += 1
                log.info(f"  G1 animated links disabled-RB: {n_disabled} "
                         f"prims (now pure visual; USD xform writes propagate)")
                # Build left-arm chain for FK-driven mirror animation.
                g1_ik_chain_left = g1_ik.build_left_arm_chain(str(g1_urdf_path))
                log.info(f"  G1 left-arm IK chain built: "
                         f"{len(g1_ik_chain_left.links)} links")

                # Box-A delivered position (computed): pallet at (-1.0, 0)
                # + BOX_A_OFFSET (-0.035, -0.302) = (-1.035, -0.302).
                # Box top-Z when settled on pallet ≈ 1.13.
                _box_top_world = _np_g1.array([-1.035, -0.302, 1.13])
                # Drop-zone close to G1 (right next to G1's left side).
                # With G1 at (-1.55, 0), drop at (G1_X, +0.46, ground+0.1).
                # Y = +0.46 (overshoot vs ideal +0.40) because the arm
                # length limit makes IK undershoot Y by ~0.06m. With
                # overshoot, achieved Y ≈ +0.40 → within spec 5cm
                # tolerance vs ideal-displayed drop position +0.4.
                # (Earlier attempts: +0.5 gave 11cm err, +0.4 gave 6cm err.)
                _drop_world = _np_g1.array([
                    g1_world_xy[0], g1_world_xy[1] + 0.46, CABIN_FLOOR_Z + 0.10
                ])
                # G1 pelvis world position (yaw=0 — no rotation, can subtract).
                _pelvis_world = _np_g1.array([
                    g1_world_xy[0],
                    g1_world_xy[1],
                    CABIN_FLOOR_Z + G1_FOOT_TO_PELVIS + G1_FOOT_EPSILON
                                  + args.g1_z_offset,
                ])

                def _w2p(xyz):
                    """World -> pelvis frame (translation only; G1 yaw=0)."""
                    return _np_g1.asarray(xyz) - _pelvis_world

                # Targets in pelvis frame for IK.
                _t_reach = _w2p(_box_top_world + _np_g1.array([0, 0, 0.10]))   # 10cm above box
                _t_grasp = _w2p(_box_top_world + _np_g1.array([0, 0, 0.02]))   # at box top
                _t_lift  = _w2p(_box_top_world + _np_g1.array([0, 0, 0.30]))   # 30cm above
                _t_place = _w2p(_drop_world  + _np_g1.array([0, 0, 0.10]))     # above drop zone
                _t_release = _w2p(_drop_world)                                  # at drop zone

                # WAIT pose: arm down (all joints 0). Solve for the others
                # so they're consistent in the chain.
                _q_wait = _np_g1.zeros(len(g1_ik_chain.links))
                # Solve IK for each pose, using previous as warm-start so
                # the solver tracks a continuous arm trajectory.
                _q_reach   = g1_ik.solve_ik(g1_ik_chain, _t_reach,   initial=_q_wait)
                _q_grasp   = g1_ik.solve_ik(g1_ik_chain, _t_grasp,   initial=_q_reach)
                _q_lift    = g1_ik.solve_ik(g1_ik_chain, _t_lift,    initial=_q_grasp)
                _q_place   = g1_ik.solve_ik(g1_ik_chain, _t_place,   initial=_q_lift)
                _q_release = g1_ik.solve_ik(g1_ik_chain, _t_release, initial=_q_place)

                g1_keyframes = {
                    "WAIT":    _q_wait,
                    "REACH":   _q_reach,
                    "GRASP":   _q_grasp,
                    "LIFT":    _q_lift,
                    "PLACE":   _q_place,
                    "RELEASE": _q_release,
                }
                # Verify each IK target by FK and log err.
                for name, q in g1_keyframes.items():
                    fk = g1_ik.forward_kinematics_all_links(g1_ik_chain, q)
                    tip = fk[list(fk.keys())[-1]][:3, 3]
                    log.info(f"  G1 keyframe [{name:8s}] tip-pos (pelvis) = "
                             f"{tip.round(3).tolist()}")

                # Left-arm keyframes (2026-05-12): hardcoded "supportive
                # gesture" joint angles, NOT IK-derived. The left arm
                # mirrors the right arm's energy without trying to reach
                # the same target — visually it looks like G1 is "actively
                # engaged in the task". Signs match URDF convention.
                # Joint order in chain (16 indices, identical to right):
                # [Base, waist_yaw, waist_roll, waist_pitch, shoulder_pitch,
                #  shoulder_roll, shoulder_yaw, elbow, wrist_roll,
                #  wrist_pitch, wrist_yaw, hand_palm, thumb_0/1/2, last]
                _n_left = len(g1_ik_chain_left.links)
                def _left_pose(shoulder_pitch=0.0, shoulder_roll=0.0,
                               shoulder_yaw=0.0, elbow=0.0,
                               wrist_pitch=0.0):
                    q = _np_g1.zeros(_n_left)
                    # waist 0..2 stay at 0 (don't fight right arm's waist)
                    q[4]  = shoulder_pitch
                    q[5]  = shoulder_roll
                    q[6]  = shoulder_yaw
                    q[7]  = elbow
                    q[9]  = wrist_pitch
                    return q

                g1_left_keyframes = {
                    "WAIT":    _left_pose(),
                    # REACH: slight forward+inward shoulder, elbow bent
                    "REACH":   _left_pose(shoulder_pitch=-0.3,
                                          shoulder_roll=+0.2,
                                          elbow=+0.6,
                                          wrist_pitch=-0.2),
                    # GRASP: similar to REACH (don't disturb right arm)
                    "GRASP":   _left_pose(shoulder_pitch=-0.35,
                                          shoulder_roll=+0.25,
                                          elbow=+0.7,
                                          wrist_pitch=-0.2),
                    # LIFT: extend slightly to "support" the lift
                    "LIFT":    _left_pose(shoulder_pitch=-0.5,
                                          shoulder_roll=+0.3,
                                          elbow=+0.9,
                                          wrist_pitch=-0.3),
                    # PLACE: forward+across body, helping the place motion
                    "PLACE":   _left_pose(shoulder_pitch=-0.6,
                                          shoulder_roll=+0.2,
                                          shoulder_yaw=-0.3,
                                          elbow=+1.0,
                                          wrist_pitch=-0.3),
                    # RELEASE: same as PLACE
                    "RELEASE": _left_pose(shoulder_pitch=-0.6,
                                          shoulder_roll=+0.2,
                                          shoulder_yaw=-0.3,
                                          elbow=+1.0,
                                          wrist_pitch=-0.3),
                }
        except Exception as e:
            log.warning(f"G1 IK setup failed: {e}. Falling back to v6 "
                        f"static box trajectory.")
            g1_ik_chain = None
            g1_keyframes = {}

    # ---- Optional: robot-mounted camera (D435i location) ----
    if args.cameras:
        # Camera attaches to robot 0's chassis_link. With multi-spawn, robot N's
        # links live under /World/tetrabot_N/<link>. (Per-robot cameras for the
        # multi-tetrabot demo are TODO; tracked in docs/MULTI_TETRABOT_DEMO.md.)
        if n_robots > 1:
            log.warning(
                f"--cameras with --num-tetrabots {n_robots}: only attaching to robot 0 for now. "
                "Per-robot cameras are pending (see docs/MULTI_TETRABOT_DEMO.md)."
            )
        cam_parent_path = f"{robot_paths[0]}/chassis_link"
        cam_path = f"{cam_parent_path}/d435i_camera"
        cam = UsdGeom.Camera.Define(stage, cam_path)
        cam.CreateFocalLengthAttr(18.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(11.78)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        xform = UsdGeom.Xformable(cam)
        xform.ClearXformOpOrder()
        t_op = xform.AddTranslateOp()
        r_op = xform.AddRotateXYZOp()
        # At the dome on top of the robot. Chassis (after 10x mesh scale) is
        # ~0.4 m tall; dome sits ~0.34 m above chassis_link origin.
        # Position above chassis where the dome sits; chassis (0.4 m tall)
        # extends to z~0.4, dome ~0.45 m above link origin.
        t_op.Set(Gf.Vec3d(0.04, 0.0, 0.45))
        # Forward-looking camera, Z-up. Derivation (USD applies XYZ extrinsic
        # = Rz·Ry·Rx, with Rx applied first to a vector):
        #   USD camera default looks down -Z_local, +Y_local up.
        #   Rx(+90) leaves -Z_local pointing forward; +Y_local rolls to +Z_world.
        #   Rz(-90) yaws so the new forward aligns with chassis +X.
        # Net composition R = Rz(-90)·Ry(0)·Rx(+90) maps:
        #   -Z_local → +X_chassis (forward) ✓
        #   +Y_local → +Z_chassis (up)      ✓
        # NOTE: the previous rpy=(0, 90, -90) was wrong — it pointed the
        # camera at +Y_chassis (left) and rolled +X_chassis to image-up,
        # so the rendered frame appeared 90° tilted. (User reported
        # "kamera schräg auf den kopf, müsste um 90° gedreht werden".)
        r_op.Set(Gf.Vec3f(90.0, 0.0, -90.0))
        log.info(f"Camera prim added at {cam_path}")
        log.info("To view: Window > Viewport > New Viewport;")
        log.info("  in the new viewport's camera dropdown choose 'd435i_camera'")

        # Also add an external 'tv camera' viewing the cabin from outside —
        # useful as a scenic reference shot for the demo.
        tv_path = "/World/scene_camera"
        tv = UsdGeom.Camera.Define(stage, tv_path)
        tv.CreateFocalLengthAttr(24.0)
        tv.CreateHorizontalApertureAttr(36.0)
        tv.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        tv_xform = UsdGeom.Xformable(tv)
        tv_xform.ClearXformOpOrder()
        tv_t = tv_xform.AddTranslateOp()
        tv_r = tv_xform.AddRotateXYZOp()
        # Position outside the cabin, looking inward at robot (which sits at origin)
        tv_t.Set(Gf.Vec3d(-2.5, -3.0, 1.5))
        tv_r.Set(Gf.Vec3f(70.0, 0.0, -45.0))
        log.info(f"External camera prim added at {tv_path}")

    # ---- ROS2 bridge: publish D435i camera as ROS2 topics ----
    if args.ros and args.cameras:
        log.info("Enabling Isaac Sim ROS2 bridge extension...")
        import omni.kit.app
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ros_ext = "isaacsim.ros2.bridge"
        ext_mgr.set_extension_enabled_immediate(ros_ext, True)
        # Allow the extension's subprocess-based health check + node
        # registration to finish before we build the graph.
        for _ in range(5):
            kit.update()

        # The extension self-disables on startup if ROS2 libs can't be
        # loaded (missing distro, missing DLLs). When that happens, node
        # types are never registered and graph creation fails with a
        # confusing "unrecognized type" error. Verify enablement first.
        if not ext_mgr.is_extension_enabled(ros_ext):
            log.error(
                "ROS2 bridge extension failed to enable — likely the ROS2 "
                "library check failed. ROS_DISTRO=%s. Either install ROS2 "
                "(Humble) on the system, or ensure the bundled libs at "
                "C:/isaac-sim/exts/isaacsim.ros2.bridge/humble/lib are on "
                "PATH (launch.bat sets this). Skipping ROS graph.",
                os.environ.get("ROS_DISTRO", "<unset>"),
            )
        else:
            log.info(f"Building ROS2 OmniGraph (camera={cam_path})")
            try:
                import omni.graph.core as og
                keys = og.Controller.Keys
                graph_path = "/Graph/ROS_TETRABot_Camera"
                topic_prefix = "/tetrabot/camera"
                frame_id = "tetrabot_d435i"

                og.Controller.edit(
                    {"graph_path": graph_path, "evaluator_name": "execution"},
                    {
                        keys.CREATE_NODES: [
                            ("OnTick", "omni.graph.action.OnPlaybackTick"),
                            ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                            ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                            ("CameraInfoPub", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                            ("RGBPub",   "isaacsim.ros2.bridge.ROS2CameraHelper"),
                            ("DepthPub", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                        ],
                        keys.SET_VALUES: [
                            ("RenderProduct.inputs:cameraPrim", cam_path),
                            ("RenderProduct.inputs:width",  640),
                            ("RenderProduct.inputs:height", 480),
                            ("CameraInfoPub.inputs:topicName", f"{topic_prefix}/camera_info"),
                            ("CameraInfoPub.inputs:frameId",   frame_id),
                            ("CameraInfoPub.inputs:resetSimulationTimeOnStop", True),
                            ("RGBPub.inputs:topicName", f"{topic_prefix}/rgb"),
                            ("RGBPub.inputs:type",      "rgb"),
                            ("RGBPub.inputs:frameId",   frame_id),
                            ("RGBPub.inputs:resetSimulationTimeOnStop", True),
                            ("DepthPub.inputs:topicName", f"{topic_prefix}/depth"),
                            ("DepthPub.inputs:type",      "depth"),
                            ("DepthPub.inputs:frameId",   frame_id),
                            ("DepthPub.inputs:resetSimulationTimeOnStop", True),
                        ],
                        keys.CONNECT: [
                            ("OnTick.outputs:tick",                      "RunOnce.inputs:execIn"),
                            ("RunOnce.outputs:step",                     "RenderProduct.inputs:execIn"),
                            ("RenderProduct.outputs:execOut",            "CameraInfoPub.inputs:execIn"),
                            ("RenderProduct.outputs:renderProductPath",  "CameraInfoPub.inputs:renderProductPath"),
                            ("Context.outputs:context",                  "CameraInfoPub.inputs:context"),
                            ("RenderProduct.outputs:execOut",            "RGBPub.inputs:execIn"),
                            ("RenderProduct.outputs:renderProductPath",  "RGBPub.inputs:renderProductPath"),
                            ("Context.outputs:context",                  "RGBPub.inputs:context"),
                            ("RenderProduct.outputs:execOut",            "DepthPub.inputs:execIn"),
                            ("RenderProduct.outputs:renderProductPath",  "DepthPub.inputs:renderProductPath"),
                            ("Context.outputs:context",                  "DepthPub.inputs:context"),
                        ],
                    },
                )
                log.info(f"  ROS2 graph at {graph_path} → topics: "
                         f"{topic_prefix}/{{rgb, depth, camera_info}} (frame_id={frame_id})")
                log.info("  Verify with: ros2 topic list  |  ros2 topic hz /tetrabot/camera/rgb")
            except Exception as e:
                log.error(f"ROS2 graph setup failed: {e}")
                log.error(traceback.format_exc())

    # ---- Synthetic Data Generation via Replicator ----
    if args.sdg and args.cameras:
        sdg_dir = LOG_DIR / f"sdg_{time.strftime('%Y%m%d_%H%M%S')}"
        sdg_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Setting up SDG capture (every frame) → {sdg_dir}")
        try:
            import omni.replicator.core as rep
            rp = rep.create.render_product(cam_path, (640, 480))
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                output_dir=str(sdg_dir),
                rgb=True,
                distance_to_camera=True,           # depth in meters
                instance_id_segmentation=True,     # per-prim instance IDs
                bounding_box_2d_tight=True,
                semantic_segmentation=False,       # needs scene-side semantics
                colorize_instance_id_segmentation=True,
            )
            writer.attach([rp])
            # BasicWriter captures one frame per timeline step that the
            # render product is rendered. Frame-rate-limiting via
            # rep.trigger.on_frame interferes with the main timeline; we
            # let it run every step and let the user choose --frames N
            # to bound the dataset size.
            log.info(f"  Replicator BasicWriter armed; output: {sdg_dir}")
        except Exception as e:
            log.error(f"SDG setup failed: {e}")
            log.error(traceback.format_exc())

    # ---- Planar dummies + lift: gentle position drive (all instances).
    # Earlier values (k=10000) were strong enough to rip the chassis through
    # wheel-ground friction contacts → robot tipped over under constant-direction
    # input (keyboard W held, or auto-demo near sin extrema). Lowered so chassis
    # accelerates smoothly while wheels can free-roll on their cylinders.
    #
    # NOTE: with multi-spawn, every joint name appears N times (once per
    # /World/tetrabot_<i>). Earlier code had a 'break' after the first match,
    # so only robot 0 got drives — silent multi-instance bug. Now we apply to
    # ALL matches.
    # Drive setup (2026-05-08 v3 URDF refactor):
    # Planar dummies removed from URDF; chassis is now the articulation
    # root and is driven directly via set_world_velocities() in the
    # scenario loop. Only lift_joint remains as a position-controlled
    # drive (wheels are passive — see next block).
    for jname, drive_axis, k, d in (
        ("lift_joint",    "linear",  1000.0,   50.0),
    ):
        matches = [p for p in stage.Traverse() if p.GetName() == jname]
        for p in matches:
            drive = UsdPhysics.DriveAPI.Get(p, drive_axis)
            if not drive:
                drive = UsdPhysics.DriveAPI.Apply(p, drive_axis)
            drive.GetStiffnessAttr().Set(k)
            drive.GetDampingAttr().Set(d)
        log.info(f"  {jname}: position drive (k={k:.0f}, d={d:.0f}) on {len(matches)} instance(s)")

    # ---- Wheel joints: passive drives (k=0, light damping) on all instances.
    for wname in ("wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint"):
        matches = [p for p in stage.Traverse() if p.GetName() == wname]
        if not matches:
            log.warning(f"  {wname}: no prim found — skipping")
            continue
        for joint_prim in matches:
            try:
                drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
                if not drive:
                    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
                drive.GetStiffnessAttr().Set(0.0)
                drive.GetDampingAttr().Set(5.0)
            except Exception as e:
                log.warning(f"  {wname} at {joint_prim.GetPath()}: drive setup failed: {e}")
        log.info(f"  {wname}: passive (k=0, d=5) on {len(matches)} instance(s)")

    # ---- Wheel-friction: holonomic mecanum approximation -------------
    # Diagnostic finding 2026-05-08: wheel_*_joint axis = (1,0,0). Sphere
    # collision (r=0.05) at hub. Y-motion = wheel rolls (rotates around X)
    # → low rolling resistance. X-motion = wheel must SLIDE along its own
    # rotation axis → full sliding friction. Result: base_x_joint drive
    # achieved only 4% of commanded target, base_y_joint achieved 100%.
    # No mecanum-roller geometry is modelled; cheapest fix is to make the
    # wheel-ground contact effectively frictionless so the chassis behaves
    # as a low-friction puck the planar drives can push in any direction.
    #
    # Initial attempt (2026-05-08 12:08): Bind material to wheel_*_link
    # parent prim → NO measurable effect on tracking. UsdShade material
    # binding does NOT propagate from a link prim to its CollisionAPI
    # children automatically.
    #
    # Root-cause-fix (2026-05-08 14:00): Bind material directly to the
    # collision-bearing descendant prims (those with CollisionAPI applied
    # by URDFImporter). Also bind to the ground plane so the friction-
    # combine resolves to the lower of the two (μ=0.02).
    def _make_low_friction_material(path: str, mu: float = 0.02):
        """Create a UsdPhysics material with low friction + min combine mode.
        PhysX combine-mode default is 'average'; using 'min' forces the
        contact friction to the lower of the two materials, so we don't
        depend on the other side (default ground/cabin) being equally low."""
        mat = UsdShade.Material.Define(stage, path)
        UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        phys_mat = UsdPhysics.MaterialAPI(mat.GetPrim())
        phys_mat.CreateStaticFrictionAttr(mu)
        phys_mat.CreateDynamicFrictionAttr(mu)
        phys_mat.CreateRestitutionAttr(0.0)
        # PhysxSchema extension lets us pin the combine mode.
        physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        physx_mat.CreateFrictionCombineModeAttr("min")
        physx_mat.CreateRestitutionCombineModeAttr("min")
        return mat

    wheel_link_names = ("wheel_fl_link", "wheel_fr_link",
                        "wheel_rl_link", "wheel_rr_link")
    wheel_collision_prims_bound = 0
    sample_paths_logged = 0
    for wlink_name in wheel_link_names:
        matches = [p for p in stage.Traverse() if p.GetName() == wlink_name]
        for link_prim in matches:
            mat_path = f"{link_prim.GetPath()}/_lowFrictionMat"
            mat_prim = _make_low_friction_material(mat_path, 0.02)
            for descendant in Usd_prim_iter(link_prim):
                if descendant.HasAPI(UsdPhysics.CollisionAPI):
                    binding_api = UsdShade.MaterialBindingAPI.Apply(descendant)
                    binding_api.Bind(mat_prim,
                                     bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                                     materialPurpose="physics")
                    wheel_collision_prims_bound += 1
                    if sample_paths_logged < 2:
                        log.info(f"    bound low-friction material on "
                                 f"{descendant.GetPath()}")
                        sample_paths_logged += 1
    log.info(f"  Wheel-friction: low-friction material bound on "
             f"{wheel_collision_prims_bound} collision prim(s) "
             f"(μs=μd=0.02, combineMode=min)")

    # Bind matching low-friction material to the ground plane. With
    # combineMode=min, only one side actually needs to be low — but we
    # set both for clarity (and so the wheel-cabin contact would also
    # be low if --cabin-collision is ever turned on).
    gp_prim = stage.GetPrimAtPath("/groundPlane")
    if gp_prim.IsValid():
        gp_mat_prim = _make_low_friction_material(
            "/groundPlane/_lowFrictionMat", 0.02)
        gp_bound = 0
        for descendant in Usd_prim_iter(gp_prim):
            if descendant.HasAPI(UsdPhysics.CollisionAPI):
                binding_api = UsdShade.MaterialBindingAPI.Apply(descendant)
                binding_api.Bind(gp_mat_prim,
                                 bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                                 materialPurpose="physics")
                gp_bound += 1
        log.info(f"  Ground-plane low-friction material bound on "
                 f"{gp_bound} collision prim(s)")

    # ---- Solver iterations: bump for tight slot-engagement contacts ----
    # Default PhysX articulation iterations (4 position / 1 velocity) are
    # too low for the multi-contact case where the docking_unit hooks the
    # pallet's rail inner walls during lift. The contact set is small but
    # the geometry is tight: a low-iteration solver oscillates and pushes
    # bodies apart. 16/4 is the recommended robotics setting and keeps
    # per-frame cost reasonable.
    for art_root in art_root_paths:
        art_prim = stage.GetPrimAtPath(art_root)
        if not art_prim.IsValid():
            log.warning(f"  Articulation API: prim invalid {art_root}")
            continue
        physx_art = PhysxSchema.PhysxArticulationAPI.Apply(art_prim)
        physx_art.CreateSolverPositionIterationCountAttr(32)
        physx_art.CreateSolverVelocityIterationCountAttr(8)
    log.info(f"  Articulation solver iters: pos=16, vel=4 on "
             f"{len(art_root_paths)} root(s)")

    # ---- Start simulation ----
    # ---- Timeline play (auto only in headless) ----
    # In GUI mode for --auto-demo and --scenario we wait for the user to
    # press Play in the toolbar. This makes the demo replayable: when the
    # scenario reaches T_DONE we Stop the timeline, the user can press Play
    # again to rerun. In headless there's no GUI to press, so auto-play.
    timeline_iface = omni.timeline.get_timeline_interface()
    autoplay_modes = (args.keyboard,)   # keyboard always auto-plays (interactive)
    headless_modes = (args.headless,)
    user_play_modes = (args.auto_demo, args.scenario != "none", args.drop_mode)
    should_autoplay = any(autoplay_modes) or (any(headless_modes) and any(user_play_modes))
    if should_autoplay:
        log.info("Starting timeline (Play)")
        timeline_iface.play()
    else:
        log.info("Timeline NOT auto-played — press Play in the GUI to start the demo "
                 "(stop and re-play to replay).")
    kit.update()

    # ---- Initialize articulation handle (batched if N>1) ----
    # For multi-instance we use a glob expression matching all spawned robots.
    # Articulation from isaacsim.core.prims supports prim_paths_expr globs and
    # gives a single handle covering all N — joint targets become (N, dof) arrays.
    if n_robots == 1:
        art_expr = prim_path
    else:
        last_seg = prim_path.rstrip("/").rsplit("/", 1)[-1]   # e.g. "root_joint"
        art_expr = f"/tetrabot_*/{last_seg}"  # top-level instances (see spawn loop)
    log.info(f"Articulation prim_paths_expr = {art_expr}")
    art = Articulation(prim_paths_expr=art_expr, name="tetrabots")

    # art.initialize() needs a live PhysX simulation view, which only exists
    # while the timeline is playing. Two failure modes are guarded against:
    # (1) Calling initialize() before Play raises
    #     'NoneType' object has no attribute 'create_articulation_view'
    #     and leaves dof_names empty, then set_joint_position_targets crashes
    #     with a (N, 0) vs (N, dof) shape mismatch.
    # (2) After Stop+Play (scenario replay), the previous sim view is torn
    #     down and the Articulation's internal _physics_view is stale or was
    #     never set. Calls to is_physics_handle_valid() then raise
    #     AttributeError because _physics_view doesn't exist as an attribute
    #     on the Articulation instance.
    # _is_art_handle_valid() guards (2); _ensure_art_initialized() guards
    # both, and is callable on every Play edge to refresh stale handles.
    art_state = {"initialized": False}

    def _is_art_handle_valid() -> bool:
        """Like art.is_physics_handle_valid() but tolerates the case where
        _physics_view was never set (initialize never succeeded)."""
        try:
            return art.is_physics_handle_valid()
        except AttributeError:
            return False

    def _ensure_art_initialized() -> bool:
        """Initialize art lazily. Re-runs if the previous handle is stale
        (e.g. after Stop+Play replay or after a FixedJoint add/remove on
        an articulation link tore down the sim view). Pump Kit a few
        frames first so PhysX has time to build the articulation
        simulation view."""
        if art_state["initialized"] and _is_art_handle_valid():
            return True
        # Either first call, or previous handle is stale — reset and retry.
        art_state["initialized"] = False
        # Isaac Sim's Articulation.initialize() internally calls
        # is_physics_handle_valid() which dereferences self._physics_view.
        # On sim-view teardown that attribute is DELETED (not set to None),
        # which makes a subsequent initialize() crash with AttributeError
        # before it gets a chance to re-bind. Re-create the attribute so
        # the internal check returns False cleanly and initialize() can
        # proceed with a fresh acquire.
        try:
            if not hasattr(art, "_physics_view"):
                art._physics_view = None
        except Exception:
            pass
        for _ in range(3):
            kit.update()
        try:
            art.initialize()
        except Exception as e:
            log.error(f"Articulation.initialize failed: {e}")
            log.error(traceback.format_exc())
            return False
        if not _is_art_handle_valid():
            log.warning("Articulation.initialize() did not produce a valid "
                        "physics handle yet. Will retry on next Play edge.")
            return False
        if not art.dof_names:
            log.warning("Articulation.initialize() returned no DOFs — Physics "
                        "may not be playing yet. Will retry on next Play edge.")
            return False
        art_state["initialized"] = True
        log.info(f"Articulation initialized: count={art.count}, "
                 f"dof_names={list(art.dof_names)}")
        return True

    if should_autoplay:
        _ensure_art_initialized()
    else:
        log.info("Deferring Articulation.initialize() until first Play edge.")

    # ---- Pickup helpers (shared by keyboard mode + scripted scenarios) ----
    # Leader-Follower architecture (Phase 2, 2026-05-08):
    # only the leader "carries" the pallet. Followers sync via velocity-
    # matching + lift-state mirroring (their waypoints share the same
    # delivery_dxy as the leader's, so they stay in formation visually).
    # This avoids the numerical instability of distributed rigid constraints
    # in multi-robot manipulation and matches established practice for
    # cooperative lifting in robotics literature.
    #
    # Carry mechanism: kinematic-set-pose, NOT FixedJoint. Earlier tests
    # showed FixedJoint between leader.docking and pallet caused PhysX to
    # accumulate impulse error each frame because the velocity-controlled
    # leader and the FixedJoint-constrained pallet conflicted (telemetry
    # 2026-05-08 16:30: pallet flew to ±10⁸ m). Manual pose-set bypasses
    # the constraint loop entirely. Boxes resting on the pallet may
    # decouple briefly during teleport but resettle within a frame.
    WORKPIECE_PATH = "/World/pallet"
    LEADER_BOT_INDEX = 0  # robot_0 carries the pallet
    # Mutable state for the kinematic-carry: True when carrying, plus the
    # constant relative offset captured at PICKUP time.
    carry_state = {
        "active": False,
        "pallet_offset_from_leader": None,  # np.ndarray(3,) world-frame offset
    }

    def _set_pallet_kinematic(enabled: bool):
        """Toggle pallet RigidBody kinematic-mode. Kinematic = PhysX honours
        xform translations, dynamics ignored. Used for clean carry/release."""
        pallet_prim = stage.GetPrimAtPath(WORKPIECE_PATH)
        if not pallet_prim.IsValid():
            return
        rb = UsdPhysics.RigidBodyAPI(pallet_prim)
        if not rb:
            return
        attr = rb.GetKinematicEnabledAttr()
        if not attr:
            attr = rb.CreateKinematicEnabledAttr()
        attr.Set(enabled)

    def _get_pallet_translate():
        """Read pallet's current xform translate (NOT bbox center).
        We need translate so the kinematic-carry per-frame `set` to
        leader_pose + offset evaluates to the SAME translate at PICKUP
        moment, avoiding a sudden teleport-jump that would launch the
        boxes resting on top."""
        pallet_prim = stage.GetPrimAtPath(WORKPIECE_PATH)
        if not pallet_prim.IsValid():
            return None
        xf = UsdGeom.Xformable(pallet_prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                v = op.Get()
                if v is not None:
                    return np.array([v[0], v[1], v[2]], dtype=np.float32)
        return None

    def _pickup_attach():
        wp_prim = stage.GetPrimAtPath(WORKPIECE_PATH)
        if not wp_prim.IsValid():
            log.warning(f"Pickup ignored: no {WORKPIECE_PATH} prim "
                        f"(start with --workpiece)")
            return
        if LEADER_BOT_INDEX >= len(robot_paths):
            log.warning(f"Leader bot index {LEADER_BOT_INDEX} >= "
                        f"n_robots {len(robot_paths)}; skipping pickup.")
            return
        try:
            leader_pos, _ = art.get_world_poses()
            leader_pos = np.asarray(leader_pos, dtype=np.float32)
            leader_xyz = leader_pos[LEADER_BOT_INDEX]
        except Exception as e:
            log.warning(f"Pickup: couldn't read leader pose: {e}")
            return
        # CRITICAL: capture offset relative to the pallet's xform TRANSLATE,
        # not bbox center. Bbox center is ~7cm above the translate (because
        # the pallet mesh is z-offset internally), so basing the offset on
        # bbox would cause a 7cm teleport jump at PICKUP — which launches
        # the boxes resting on the pallet top into the air at ~4 m/s
        # (user reported 2026-05-11: "pakete springen hoch in die luft").
        pallet_translate = _get_pallet_translate()
        if pallet_translate is None:
            log.warning("Pickup: couldn't read pallet xform translate.")
            return
        carry_state["pallet_offset_from_leader"] = pallet_translate - leader_xyz
        carry_state["active"] = True
        _set_pallet_kinematic(True)
        log.info(f"Pickup: leader bot_{LEADER_BOT_INDEX} kinematic-carrying "
                 f"pallet. Translate-based offset="
                 f"{carry_state['pallet_offset_from_leader']}. "
                 f"Pallet now kinematic. Followers track their own waypoints; "
                 f"no FixedJoint anywhere.")

    def _carry_update_pallet(leader_xyz_world):
        """Per-frame teleport of the pallet to follow the leader during carry.
        Computes target = leader_pos + offset and sets pallet xform translate."""
        if not carry_state["active"]:
            return
        offset = carry_state["pallet_offset_from_leader"]
        if offset is None:
            return
        target = leader_xyz_world + offset
        pallet_prim = stage.GetPrimAtPath(WORKPIECE_PATH)
        if not pallet_prim.IsValid():
            return
        xf = UsdGeom.Xformable(pallet_prim)
        translate_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xf.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(target[0]), float(target[1]),
                                   float(target[2])))

    def _pickup_release():
        removed = 0
        if carry_state["active"]:
            carry_state["active"] = False
            carry_state["pallet_offset_from_leader"] = None
            # 2026-05-13 polish-sprint fix: KEEP pallet kinematic after
            # release. Switching back to dynamic caused the pallet to
            # drift westward (~0.8m) when box_a became kinematic at G1
            # GRASP — the kinematic-set impulse on box-pallet contact
            # pushed the dynamic pallet. Keeping kinematic = immovable
            # pallet, looks correct (it just sits at delivery).
            # The pallet doesn't need gravity to "settle" — it was
            # kinematic-set to its target pose during carry, which is
            # already the final resting position.
            log.info("Pickup released: leader kinematic-carry stopped, "
                     "pallet remains kinematic (no drift during G1 pickup).")
            return
        # Legacy cleanup: remove any FixedJoints from older sessions if
        # a leftover stage is loaded.
        legacy_root = "/World/PickupJoints"
        for joint_name in ("joint_leader",) + tuple(f"joint_{i}" for i in range(len(robot_paths))):
            legacy_path = f"{legacy_root}/{joint_name}"
            legacy_prim = stage.GetPrimAtPath(legacy_path)
            if legacy_prim.IsValid():
                legacy_prim.SetActive(False)
                stage.RemovePrim(legacy_path)
                removed += 1
        if removed:
            log.info(f"Pickup: released {removed} fixed joint(s)")

    # ---- G1 box-pickup helpers (kinematic-only; no controller) -------
    # Box_a starts on the pallet; once the bots have retreated and the
    # pallet sits at delivery, transition box_a to kinematic and teleport
    # it along a linear path from its on-pallet rest pose to a "held in
    # front of G1 chest" target. Pure visual choreo — G1 itself is a
    # static USD reference, no joints animated.
    G1_BOX_PATH = "/World/box_a"
    g1_box_state = {
        "active": False,
        "start_xyz": None,   # captured at pickup-start
        "target_xyz": None,  # in front of G1's chest
    }

    def _set_box_kinematic(enabled: bool):
        box_prim = stage.GetPrimAtPath(G1_BOX_PATH)
        if not box_prim.IsValid():
            return
        rb = UsdPhysics.RigidBodyAPI(box_prim)
        if not rb:
            return
        attr = rb.GetKinematicEnabledAttr()
        if not attr:
            attr = rb.CreateKinematicEnabledAttr()
        attr.Set(enabled)
        # When transitioning kinematic -> dynamic at G1 RELEASE, the box
        # would otherwise stay suspended at its last kinematic-set pose
        # (telemetry 2026-05-11: box at Z=1.55 didn't fall after release).
        # Explicitly seed velocity to 0 and wake the body so PhysX picks
        # it up for gravity integration on the next physics step.
        if not enabled:
            for vattr_name, default in (("physics:velocity",        (0.0, 0.0, 0.0)),
                                        ("physics:angularVelocity", (0.0, 0.0, 0.0))):
                vattr = box_prim.GetAttribute(vattr_name)
                if not vattr:
                    vattr = box_prim.CreateAttribute(
                        vattr_name, Sdf.ValueTypeNames.Vector3f)
                vattr.Set(Gf.Vec3f(*default))
            # Disable startsAsleep flag so PhysX integrates immediately.
            sa_attr = box_prim.GetAttribute("physics:startsAsleep")
            if not sa_attr:
                sa_attr = box_prim.CreateAttribute(
                    "physics:startsAsleep", Sdf.ValueTypeNames.Bool)
            sa_attr.Set(False)

    def _box_set_translate(xyz):
        box_prim = stage.GetPrimAtPath(G1_BOX_PATH)
        if not box_prim.IsValid():
            return
        xf = UsdGeom.Xformable(box_prim)
        translate_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xf.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    def _g1_pickup_start():
        """Capture box_a's current world position and switch it to
        kinematic. The per-frame interpolator then teleports it toward
        the chest-target."""
        box_prim = stage.GetPrimAtPath(G1_BOX_PATH)
        if not box_prim.IsValid():
            log.warning(f"G1 pickup: no {G1_BOX_PATH} prim — skipping.")
            return
        # Read box current world pose via xform translate.
        bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
        bb = bbox_cache.ComputeWorldBound(box_prim).ComputeAlignedRange()
        mn, mx = bb.GetMin(), bb.GetMax()
        start_xyz = np.array([
            (mn[0] + mx[0]) / 2,
            (mn[1] + mx[1]) / 2,
            (mn[2] + mx[2]) / 2,
        ], dtype=np.float32)
        if g1_world_xy[0] is None:
            log.warning("G1 pickup: G1 not deployed; skipping.")
            return
        # Target = 0.4m in front of G1 (east of G1 since G1 faces east),
        # at chest height (~1.3m above the floor).
        target_xyz = np.array([
            g1_world_xy[0] + 0.40,
            g1_world_xy[1],
            CABIN_FLOOR_Z + 1.30,
        ], dtype=np.float32)
        g1_box_state["start_xyz"] = start_xyz
        g1_box_state["target_xyz"] = target_xyz
        g1_box_state["active"] = True
        _set_box_kinematic(True)
        log.info(f"G1 pickup START: box_a from {start_xyz.tolist()} -> "
                 f"target {target_xyz.tolist()} (kinematic).")

    def _g1_pickup_update(t_now: float, t_lift_start: float, t_lift_end: float):
        """Legacy linear interpolator — kept dormant for backwards compat
        with code paths that still call it. Phase B replaces this with
        the IK-driven _g1_step below."""
        if not g1_box_state["active"]:
            return
        s = g1_box_state["start_xyz"]
        e = g1_box_state["target_xyz"]
        if s is None or e is None:
            return
        if t_now <= t_lift_start:
            cur = s
        elif t_now >= t_lift_end:
            cur = e
        else:
            a = (t_now - t_lift_start) / (t_lift_end - t_lift_start)
            cur = s + (e - s) * a
        _box_set_translate(cur)

    # Phase B (2026-05-11): IK-driven pickup state machine. Per frame:
    # 1) Determine current segment from time (WAIT/REACH/GRASP/LIFT/PLACE/RELEASE/return-to-WAIT)
    # 2) Interpolate joint angles between the segment's source and target keyframes
    # 3) Run forward kinematics to find the wrist world pose
    # 4) During GRASP..RELEASE: teleport box_a to follow the wrist (kinematic-carry,
    #    same pattern as the pallet leader-follower carry — avoids FixedJoint explosions)
    # 5) After RELEASE: manually animate box free-fall (kinematic stays on, but
    #    we drive Z downward via _box_set_translate). Avoids the PhysX kinematic→
    #    dynamic transition bug that left the box suspended at hand height.
    g1_pelvis_world_cached = [None]  # set at scenario init from G1 spawn xform
    g1_state_label = ["WAIT"]         # exposed via telemetry
    g1_box_drop_state = {"start_xyz": None, "start_t": None}

    # Phase B+ (2026-05-12): Manual-FK arm-link visual animation.
    # G1 is fully kinematic (no articulation, no joint drives), so arm
    # joint angles don't propagate to link visuals automatically. We
    # compute each arm link's world transform via ikpy FK, then write
    # USD xform with !resetXformStack! so the prim's pose is absolute
    # (not parent-relative). This lets G1's right arm visibly move
    # through REACH→GRASP→LIFT→PLACE→RELEASE poses synchronously with
    # the box trajectory.
    # g1_arm_link_prims was forward-declared in the G1 spawn block and
    # populated there. Helpers below close over it for per-frame USD-xform
    # writes during the manual-FK animation.

    def _g1_set_link_world_transform(prim, T_world_np):
        """Write an absolute world transform onto a USD prim, bypassing
        parent xform inheritance via resetXformStack. Uses separate
        translate + rotateXYZ ops (more robust for kinematic / Hydra
        re-resolve than a single xformOp:transform 4x4)."""
        if not prim.IsValid():
            return
        xf = UsdGeom.Xformable(prim)
        translate_op = None
        rotate_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op = op
        if translate_op is None:
            translate_op = xf.AddTranslateOp()
        if rotate_op is None:
            rotate_op = xf.AddRotateXYZOp()
        xf.SetXformOpOrder([translate_op, rotate_op], resetXformStack=True)
        # Translation in standard form is in column 3.
        translate_op.Set(Gf.Vec3d(float(T_world_np[0, 3]),
                                   float(T_world_np[1, 3]),
                                   float(T_world_np[2, 3])))
        # 3x3 rotation -> Euler XYZ (degrees). Standard "intrinsic XYZ"
        # decomposition: R = Rx * Ry * Rz applied to vector.
        R = T_world_np[:3, :3]
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        if sy > 1e-6:
            rx = math.atan2(R[2, 1], R[2, 2])
            ry = math.atan2(-R[2, 0], sy)
            rz = math.atan2(R[1, 0], R[0, 0])
        else:
            rx = math.atan2(-R[1, 2], R[1, 1])
            ry = math.atan2(-R[2, 0], sy)
            rz = 0.0
        rotate_op.Set(Gf.Vec3f(float(math.degrees(rx)),
                                float(math.degrees(ry)),
                                float(math.degrees(rz))))

    def _g1_update_arm_link_visuals(fk_dict, pelvis_world_xyz):
        """Per-frame: compose pelvis-world transform with each chain
        link's pelvis-frame FK matrix, write to the corresponding USD
        link prim. G1 yaw=0 means pelvis_world has identity rotation,
        so composition is just translation-add for the position part."""
        if not g1_arm_link_prims:
            return
        for ikpy_name, urdf_link_name in g1_ik.IKPY_TO_URDF_LINK.items():
            T_pelvis = fk_dict.get(ikpy_name)
            if T_pelvis is None:
                continue
            prim = g1_arm_link_prims.get(urdf_link_name)
            if prim is None or not prim.IsValid():
                continue
            # Compose: world_T = pelvis_world_T (translate-only) @ T_pelvis.
            # Since pelvis rotation is identity, this is just shifting the
            # translation by pelvis_world_xyz.
            T_world = T_pelvis.copy()
            T_world[:3, 3] = T_world[:3, 3] + pelvis_world_xyz
            _g1_set_link_world_transform(prim, T_world)

    def _g1_update_left_arm_link_visuals(fk_dict, pelvis_world_xyz):
        """Same as the right-arm version but using the LEFT chain mapping."""
        if not g1_left_arm_link_prims:
            return
        for ikpy_name, urdf_link_name in g1_ik.LEFT_IKPY_TO_URDF_LINK.items():
            T_pelvis = fk_dict.get(ikpy_name)
            if T_pelvis is None:
                continue
            prim = g1_left_arm_link_prims.get(urdf_link_name)
            if prim is None or not prim.IsValid():
                continue
            T_world = T_pelvis.copy()
            T_world[:3, 3] = T_world[:3, 3] + pelvis_world_xyz
            _g1_set_link_world_transform(prim, T_world)

    def _g1_update_head_tracking(t_now, wrist_world_xyz, pelvis_world_xyz):
        """Look-at controller for G1's head_link.

        Target priority:
            * PICKUP_HOLD..RETREAT_END: pallet world position (G1 watches
              tetrabots arrive with pallet)
            * G1_REACH_END..G1_DONE: own hand world position (G1 watches
              the box pickup motion)
            * otherwise: pallet position by default
        Computes yaw+pitch from head world pose toward target. Writes
        head_link world xform with translate=head_estimated_world_pose,
        rotateXYZ=(0, pitch_deg, yaw_deg). resetXformStack so the head
        xform is absolute world.
        """
        prim = g1_head_prim[0]
        if prim is None or not prim.IsValid():
            return
        # Estimate head world position. The G1 URDF chain pelvis -> torso
        # -> head adds ~0.5m Z. With G1 pelvis at world (-1.55, 0, 1.579)
        # head sits roughly at (-1.55, 0, 2.05). Constant for stationary G1.
        head_world = pelvis_world_xyz + np.array([0.0, 0.0, 0.50])
        # Pick the target.
        target = None
        if (T_PICKUP_HOLD <= t_now < T_RETREAT_END):
            # Pallet position — query its xform translate (works for
            # both kinematic-carry and post-release dynamic).
            t = _get_pallet_translate()
            if t is not None:
                target = np.asarray(t, dtype=np.float64)
        if target is None and (T_G1_REACH_END <= t_now < T_G1_DONE):
            target = np.asarray(wrist_world_xyz, dtype=np.float64)
        if target is None:
            # Default fallback: where the pallet ends up.
            target = np.array([-1.0, 0.0, 0.86], dtype=np.float64)
        # Compute look-at angles (yaw around world Z, pitch around chassis Y).
        forward = target - head_world
        yaw = math.atan2(forward[1], forward[0])
        horiz = math.sqrt(forward[0]**2 + forward[1]**2)
        pitch = math.atan2(-forward[2], horiz) if horiz > 1e-6 else 0.0
        # G1 facing east means default yaw=0 -> head's "look forward" is +X.
        # We write absolute world xform on head_link: position at head_world,
        # rotateXYZ with pitch on Y, yaw on Z.
        xf = UsdGeom.Xformable(prim)
        translate_op = None
        rotate_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rotate_op = op
        if translate_op is None:
            translate_op = xf.AddTranslateOp()
        if rotate_op is None:
            rotate_op = xf.AddRotateXYZOp()
        xf.SetXformOpOrder([translate_op, rotate_op], resetXformStack=True)
        translate_op.Set(Gf.Vec3d(
            float(head_world[0]), float(head_world[1]), float(head_world[2])))
        # XYZ Euler: rotateX=0, rotateY=pitch, rotateZ=yaw (degrees).
        rotate_op.Set(Gf.Vec3f(
            0.0, float(math.degrees(pitch)), float(math.degrees(yaw))))

    def _g1_step(t_now: float):
        # No-ops if IK setup didn't succeed.
        if g1_ik_chain is None or not g1_keyframes:
            return
        # Lazily cache pelvis world position (constant for the whole demo
        # since G1 stays kinematic at spawn pose).
        if g1_pelvis_world_cached[0] is None:
            if g1_world_xy_for_ik[0] is None:
                return
            g1_pelvis_world_cached[0] = np.array([
                g1_world_xy_for_ik[0],
                g1_world_xy_for_ik[1],
                CABIN_FLOOR_Z + G1_FOOT_TO_PELVIS + G1_FOOT_EPSILON
                              + args.g1_z_offset,
            ], dtype=np.float64)

        # Define the 7 segments. (t_seg_end, src_keyframe, dst_keyframe).
        # Outside [T_G1_WAIT_END, T_G1_DONE] G1 holds WAIT.
        segs = [
            (T_G1_WAIT_END,    "WAIT",    "WAIT"),     # before WAIT_END: idle
            (T_G1_REACH_END,   "WAIT",    "REACH"),
            (T_G1_GRASP_END,   "REACH",   "GRASP"),
            (T_G1_LIFT_END,    "GRASP",   "LIFT"),
            (T_G1_PLACE_END,   "LIFT",    "PLACE"),
            (T_G1_RELEASE_END, "PLACE",   "RELEASE"),
            (T_G1_DONE,        "RELEASE", "WAIT"),     # return to idle
        ]
        if t_now <= T_G1_WAIT_END:
            q = g1_keyframes["WAIT"]
            g1_state_label[0] = "G1_WAIT"
        elif t_now >= T_G1_DONE:
            q = g1_keyframes["WAIT"]
            g1_state_label[0] = "G1_IDLE"
        else:
            t_prev = T_G1_WAIT_END
            q = g1_keyframes["WAIT"]
            for (t_end, src, dst) in segs[1:]:
                if t_prev <= t_now < t_end:
                    alpha = (t_now - t_prev) / max(1e-6, (t_end - t_prev))
                    q = g1_ik.interp_joint_angles(
                        g1_keyframes[src], g1_keyframes[dst], alpha)
                    g1_state_label[0] = f"G1_{src}_TO_{dst}"
                    break
                t_prev = t_end

        # FK to find wrist world pose + per-link poses for visual update.
        fk = g1_ik.forward_kinematics_all_links(g1_ik_chain, q)
        # Last entry = end-effector tip (palm centre, includes last_link_vector).
        last_key = list(fk.keys())[-1]
        wrist_pelvis = np.asarray(fk[last_key][:3, 3], dtype=np.float64)
        wrist_world = wrist_pelvis + g1_pelvis_world_cached[0]
        # Cache for telemetry (read by _scenario_snapshot).
        g1_state_label.append(("hand_world", wrist_world.tolist()))
        # Manual-FK arm visual update — write each arm link's USD xform
        # so G1 visibly moves through REACH/GRASP/LIFT/PLACE poses.
        # No-op if arm-link prims weren't discovered (fallback to v6
        # static-G1 visual + IK-driven box trajectory only).
        _g1_update_arm_link_visuals(fk, g1_pelvis_world_cached[0])

        # Left-arm supportive-gesture animation (2026-05-12 visual polish).
        # Mirrors the right-arm's temporal pattern with hardcoded joint
        # keyframes (no IK target — just a "looks alive" gesture). FK on
        # the left-arm chain produces world transforms for left arm links.
        if (g1_ik_chain_left is not None and g1_left_keyframes
                and g1_left_arm_link_prims):
            # Determine left-arm src/dst from same segment lookup as right
            if t_now <= T_G1_WAIT_END or t_now >= T_G1_DONE:
                q_left = g1_left_keyframes["WAIT"]
            else:
                t_prev = T_G1_WAIT_END
                q_left = g1_left_keyframes["WAIT"]
                for (t_end, src, dst) in segs[1:]:
                    if t_prev <= t_now < t_end:
                        alpha = (t_now - t_prev) / max(1e-6, (t_end - t_prev))
                        q_left = g1_ik.interp_joint_angles(
                            g1_left_keyframes[src], g1_left_keyframes[dst], alpha)
                        break
                    t_prev = t_end
            fk_left = g1_ik.forward_kinematics_all_links(
                g1_ik_chain_left, q_left)
            # Reuse the same update helper with left-arm mapping
            _g1_update_left_arm_link_visuals(fk_left, g1_pelvis_world_cached[0])

        # Head-tracking (2026-05-12 visual polish): G1 head looks at the
        # current "interest target":
        #   * During pallet transport (ENTER_CABIN phase): look at pallet
        #   * During G1 pickup (REACH..RELEASE): look at own hand
        #   * Otherwise: look at last-known box_a position
        # Computed look-at yaw+pitch written to head_link world xform.
        if g1_head_prim[0] is not None:
            _g1_update_head_tracking(t_now, wrist_world,
                                     g1_pelvis_world_cached[0])

        # Box carry: kinematic during GRASP..RELEASE window.
        in_carry = (T_G1_GRASP_END <= t_now < T_G1_RELEASE_END)
        if in_carry:
            if not g1_box_state["active"]:
                _set_box_kinematic(True)
                g1_box_state["active"] = True
                log.info(f"G1 box GRASP at t={t_now:.2f}s, "
                         f"hand_world={wrist_world.round(3).tolist()}")
            _box_set_translate(wrist_world)
        elif t_now >= T_G1_RELEASE_END:
            # Post-release: manually animate free-fall using kinematic-set
            # (instead of toggling kinematic→dynamic, which left the box
            # suspended at hand height because of a PhysX state-transition
            # bug). At RELEASE, capture the box's start pose; then per
            # frame compute z(t) = z_start - 0.5*g*t² until the box rests
            # on the ground (CABIN_FLOOR_Z + half-box-height ≈ floor+0.20).
            if g1_box_state["active"]:
                # First frame after carry ends: capture the start pose.
                g1_box_drop_state["start_xyz"] = wrist_world.copy()
                g1_box_drop_state["start_t"] = float(t_now)
                g1_box_state["active"] = False
                log.info(f"G1 box RELEASE at t={t_now:.2f}s, manual "
                         f"free-fall from {wrist_world.round(3).tolist()}")
            if g1_box_drop_state["start_xyz"] is not None:
                drop_dt = t_now - g1_box_drop_state["start_t"]
                z_start = float(g1_box_drop_state["start_xyz"][2])
                # Free-fall under gravity until box hits floor.
                z_floor = CABIN_FLOOR_Z + 0.20   # box bottom-half ≈ 20cm
                z_now = max(z_floor, z_start - 0.5 * 9.81 * drop_dt * drop_dt)
                _box_set_translate((
                    float(g1_box_drop_state["start_xyz"][0]),
                    float(g1_box_drop_state["start_xyz"][1]),
                    z_now,
                ))

    # ---- G1 arm-animation helpers (drives joint targetPosition) -------
    # The G1 USD ships with DriveAPI authored on each revolute joint
    # (because IsaacLab uses position-target control). We only need to
    # update the targetPosition attribute per frame to animate.
    # UsdPhysics.DriveAPI for "angular" joints uses DEGREES (USD
    # convention), so we convert from radians at the call site.
    def _g1_set_joint_target_rad(joint_name: str, target_rad: float):
        path = g1_joint_paths.get(joint_name)
        if not path:
            return
        joint_prim = stage.GetPrimAtPath(path)
        if not joint_prim.IsValid():
            return
        drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
        attr = drive.GetTargetPositionAttr()
        if not attr:
            attr = drive.CreateTargetPositionAttr()
        attr.Set(math.degrees(target_rad))

    # Right-arm "pick up box" pose (radians). G1 faces east and the box
    # ends up ~0.4m east of G1 at chest height, so the right arm is
    # extended forward and slightly inward, elbow bent, wrist tilted
    # down so the palm faces the box. Sign conventions follow the
    # standard Unitree G1 URDF (positive shoulder_pitch = arm down/back,
    # negative = forward/up — adjust if your USD differs).
    G1_ARM_REST = {
        "right_shoulder_pitch_joint": 0.0,
        "right_shoulder_roll_joint":  0.0,
        "right_shoulder_yaw_joint":   0.0,
        "right_elbow_joint":          0.0,
        "right_wrist_pitch_joint":    0.0,
    }
    G1_ARM_HOLD = {
        "right_shoulder_pitch_joint": -0.5,
        "right_shoulder_roll_joint":  -0.3,
        "right_shoulder_yaw_joint":    0.3,
        "right_elbow_joint":           1.4,
        "right_wrist_pitch_joint":    -0.3,
    }
    import math as _math_g1  # avoid shadow if math hasn't been imported yet here

    def _g1_arm_animate(t_now: float, t_start: float, t_end: float):
        """Linearly interpolate right-arm joint targets from REST to HOLD
        across [t_start, t_end]. Outside that window, hold the endpoint
        pose (REST before t_start, HOLD after t_end). No-op if G1 wasn't
        deployed."""
        if not g1_joint_paths:
            return
        if t_now <= t_start:
            alpha = 0.0
        elif t_now >= t_end:
            alpha = 1.0
        else:
            alpha = (t_now - t_start) / (t_end - t_start)
        for jn, rest in G1_ARM_REST.items():
            hold = G1_ARM_HOLD[jn]
            target = rest + (hold - rest) * alpha
            _g1_set_joint_target_rad(jn, target)

    # ---- Demo mode (optional) ----
    if args.auto_demo:
        log.info("Auto-demo: animating base_x / base_yaw / lift sinusoidally")
        import math

        import numpy as np

        try:
            from isaacsim.core.api.controllers.articulation_controller import (
                ArticulationController,
            )
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.controllers.articulation_controller import (  # type: ignore
                ArticulationController,
            )
            from omni.isaac.core.utils.types import ArticulationAction  # type: ignore

        # Controller / dof tables are filled on the first Play edge once
        # _ensure_art_initialized() has succeeded. See Bug 1 in
        # docs/HANDOFF_NEXT_CHAT.md — initializing before Play yields no DOFs.
        controller: ArticulationController | None = None
        dof_names: list[str] = []
        dof_idx: dict[str, int] = {}

        # Per-robot phase offsets so multi-spawn looks like a coordinated
        # ensemble rather than 4 robots doing the same sin in lockstep.
        phases = [r * (math.pi / 2.0) for r in range(n_robots)]
        t = 0.0
        dt = 1.0 / 60.0
        frames_done = 0
        prev_playing = False
        log.info("  Auto-demo: press Play in the GUI to start (replayable).")
        while kit.is_running():
            is_playing = timeline_iface.is_playing()
            if is_playing and not prev_playing:
                # Wipe stale state so a Stop+Play replay rebinds cleanly.
                controller = None
                dof_names = []
                dof_idx = {}
                if _ensure_art_initialized():
                    controller = ArticulationController()
                    controller.initialize(art)
                    dof_names = list(art.dof_names)
                    dof_idx = {name: i for i, name in enumerate(dof_names)}
                    log.info(f"Demo controller initialized for {len(dof_names)} DOFs")
                t = 0.0
                log.info("  Auto-demo started (Play edge), t reset to 0")
            prev_playing = is_playing
            if not is_playing:
                kit.update()
                frames_done += 1
                if args.frames and frames_done >= args.frames:
                    log.info(f"Reached frame limit ({args.frames}), stopping")
                    break
                continue
            if controller is None or not dof_names:
                # Articulation didn't finish initializing yet — pump and wait.
                kit.update()
                frames_done += 1
                continue
            if n_robots == 1:
                targets = np.zeros(len(dof_names), dtype=np.float32)
                if "base_x_joint" in dof_idx:
                    targets[dof_idx["base_x_joint"]] = 0.5 * math.sin(0.5 * t)
                if "base_yaw_joint" in dof_idx:
                    targets[dof_idx["base_yaw_joint"]] = 0.6 * math.sin(0.3 * t)
                if "lift_joint" in dof_idx:
                    targets[dof_idx["lift_joint"]] = 0.0
                controller.apply_action(ArticulationAction(joint_positions=targets))
            else:
                # Batched (N, dof) targets. ArticulationController.apply_action
                # expects 1D for single env; for multi-env we go directly via
                # Articulation.set_joint_position_targets which accepts (N, dof).
                targets = np.zeros((n_robots, len(dof_names)), dtype=np.float32)
                for r, phase in enumerate(phases):
                    if "base_x_joint" in dof_idx:
                        targets[r, dof_idx["base_x_joint"]] = 0.5 * math.sin(0.5 * t + phase)
                    if "base_yaw_joint" in dof_idx:
                        targets[r, dof_idx["base_yaw_joint"]] = 0.6 * math.sin(0.3 * t + phase)
                    # lift bleibt bei 0 (siehe single-instance Kommentar oben)
                art.set_joint_position_targets(targets)
            kit.update()
            t += dt
            frames_done += 1
            if args.frames and frames_done >= args.frames:
                log.info(f"Reached frame limit ({args.frames}), stopping")
                break
            if frames_done % 600 == 0:
                log.debug(f"  demo frame {frames_done} (sim t={t:.2f}s)")
    elif args.keyboard:
        # ---- Interactive keyboard control ----
        log.info("Keyboard control active.")
        log.info("  W / S      : forward / back  (base_x_joint)")
        log.info("  A / D      : left / right    (base_y_joint)")
        log.info("  Q / E      : rotate L / R    (base_yaw_joint)")
        log.info("  R / F      : lift up / down  (lift_joint)")
        log.info("  P          : pickup  workpiece  (FixedJoint chassis<->box)")
        log.info("  O          : release workpiece")
        log.info("  SPACE      : reset all joint targets to 0")
        log.info("  Click into the viewport once so it captures keyboard focus.")

        import math

        import carb.input
        import numpy as np
        import omni.appwindow

        try:
            from isaacsim.core.api.controllers.articulation_controller import (
                ArticulationController,
            )
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.controllers.articulation_controller import (  # type: ignore
                ArticulationController,
            )
            from omni.isaac.core.utils.types import ArticulationAction  # type: ignore

        controller = ArticulationController()
        controller.initialize(art)

        dof_names = list(art.dof_names) if art.dof_names else []
        dof_idx = {name: i for i, name in enumerate(dof_names)}

        appwindow = omni.appwindow.get_default_app_window()
        keyboard = appwindow.get_keyboard()
        input_iface = carb.input.acquire_input_interface()

        # Persistent press-state flags, updated by keyboard event subscriber.
        K = carb.input.KeyboardInput
        watched = {
            K.W: "w", K.S: "s", K.A: "a", K.D: "d",
            K.Q: "q", K.E: "e", K.R: "r", K.F: "f",
            K.SPACE: "space",
            K.P: "p", K.O: "o",   # one-shot pickup / release
        }
        held = {name: False for name in watched.values()}
        # One-shot edge events (consumed in main loop). Mutable dict so the
        # on_key closure can set values without nonlocal gymnastics.
        edge = {"p": False, "o": False}

        def on_key(event, *_):
            name = watched.get(event.input)
            if name is None:
                return True
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                held[name] = True
                if name in edge:
                    edge[name] = True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                held[name] = False
            return True

        sub = input_iface.subscribe_to_keyboard_events(keyboard, on_key)
        # Pickup helpers _pickup_attach / _pickup_release are defined at
        # function scope above (shared with --scenario).

        # Joint target state (accumulated)
        base_x = 0.0
        base_y = 0.0
        base_yaw = 0.0
        # lift starts retracted (=0). Wheels on floor in this state.
        # R/F still work but with fix_base=True, raising lift drives the
        # wheels down into the floor (chassis can't rise) — kept for
        # mechanism visibility only.
        lift_pos = 0.0
        wheel_angle = 0.0

        # Per-tick increments (60 Hz target). Kept gentle so the soft planar
        # drives can keep up without ripping the chassis through wheel friction.
        DX = 0.005    # ~0.30 m/s peak
        DY = 0.005
        DYAW = 0.01   # ~0.60 rad/s
        DLIFT = 0.002

        # Joint limits (clamp targets so PhysX doesn't reject)
        X_LIM = 5.0
        Y_LIM = 5.0
        YAW_LIM = math.pi * 1.9   # leave some margin to PhysX's 2pi cap
        LIFT_LIM_LO, LIFT_LIM_HI = 0.0, 0.10

        frames_done = 0
        while kit.is_running():
            # One-shot pickup / release edges. Consumed = reset to False so
            # the next press fires again.
            if edge["p"]:
                edge["p"] = False
                _pickup_attach()
            if edge["o"]:
                edge["o"] = False
                _pickup_release()

            if held["space"]:
                base_x = 0.0
                base_y = 0.0
                base_yaw = 0.0
                lift_pos = 0.05

            if held["w"]: base_x = min(X_LIM, base_x + DX)
            if held["s"]: base_x = max(-X_LIM, base_x - DX)
            if held["a"]: base_y = min(Y_LIM, base_y + DY)
            if held["d"]: base_y = max(-Y_LIM, base_y - DY)
            if held["q"]: base_yaw = min(YAW_LIM, base_yaw + DYAW)
            if held["e"]: base_yaw = max(-YAW_LIM, base_yaw - DYAW)
            if held["r"]: lift_pos = min(LIFT_LIM_HI, lift_pos + DLIFT)
            if held["f"]: lift_pos = max(LIFT_LIM_LO, lift_pos - DLIFT)

            # Wheels are passive (k=0); we don't drive their angle. They
            # rotate via ground friction as the chassis moves — physically
            # correct and avoids the chassis-vs-wheel torque fight.

            single = np.zeros(len(dof_names), dtype=np.float32)
            if "base_x_joint" in dof_idx:    single[dof_idx["base_x_joint"]] = base_x
            if "base_y_joint" in dof_idx:    single[dof_idx["base_y_joint"]] = base_y
            if "base_yaw_joint" in dof_idx:  single[dof_idx["base_yaw_joint"]] = base_yaw
            if "lift_joint" in dof_idx:      single[dof_idx["lift_joint"]] = lift_pos

            # Im Keyboard-Mode steuern alle Roboter identisch.
            if n_robots == 1:
                controller.apply_action(ArticulationAction(joint_positions=single))
            else:
                # Batched API direct (Controller.apply_action ist single-env)
                targets = np.tile(single, (n_robots, 1))
                art.set_joint_position_targets(targets)
            kit.update()
            frames_done += 1
            if args.frames and frames_done >= args.frames:
                log.info(f"Reached frame limit ({args.frames}), stopping")
                break

        try:
            input_iface.unsubscribe_to_keyboard_events(keyboard, sub)
        except Exception:
            pass
    elif args.scenario == "cabin_assembly":
        # ---- Scripted cabin assembly: 4 TETRABots cooperatively transport
        # the Euro-pallet from origin to delivery point ----
        log.info("Cabin assembly scenario active")
        import math
        import numpy as np

        try:
            from isaacsim.core.api.controllers.articulation_controller import (
                ArticulationController,
            )
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            from omni.isaac.core.controllers.articulation_controller import (  # type: ignore
                ArticulationController,
            )
            from omni.isaac.core.utils.types import ArticulationAction  # type: ignore

        # Camera auto-cycling (2026-05-13 polish-sprint): defined AFTER
        # T_* phase constants below — see ~line 2786+. Stub here so the
        # name exists in scope.
        _viewport_api = [None]
        _active_camera_path = [None]
        camera_schedule = []
        def _camera_for_time(t_now): return None
        def _maybe_switch_camera(t_now): pass

        # HUD overlay (2026-05-13 polish-sprint): live floating omni.ui
        # window showing phase + frame + snapshot counters during the
        # demo. GUI-only; skipped silently in headless. The window is
        # repositionable — for demo recording, drag it onto the viewport
        # corner where it sits cleanly in the OBS framing.
        hud_state = {"phase": None, "frame": None, "snap": None,
                     "g1": None, "carry": None}
        if not args.headless:
            try:
                import omni.ui as _ui
                _hud_window = _ui.Window(
                    "TETRABot demo — live status",
                    width=420, height=180)
                with _hud_window.frame:
                    with _ui.VStack(spacing=6):
                        hud_state["phase"] = _ui.Label(
                            "Phase: (waiting for Play)",
                            style={"font_size": 22, "color": 0xFFCCFFCC})
                        hud_state["frame"] = _ui.Label(
                            "Sim t: 0.00 s  |  Frame: 0",
                            style={"font_size": 16})
                        hud_state["snap"] = _ui.Label(
                            "Telemetry snapshots: 0",
                            style={"font_size": 16, "color": 0xFFFFD080})
                        hud_state["g1"] = _ui.Label(
                            "G1: idle",
                            style={"font_size": 16})
                        hud_state["carry"] = _ui.Label(
                            "Box carry: inactive",
                            style={"font_size": 16})
                log.info("  HUD overlay window 'TETRABot demo — live status' created")
            except Exception as _e:
                log.warning(f"  HUD overlay disabled ({_e})")
                hud_state = {k: None for k in hud_state}

        # RL-controller plumbing (Stufe 1, 2026-05-12). Stub policy by
        # default; identical visible behaviour to hand-coded P-controller.
        # Once weights exist (see tools/train_g1_locomotion.py), pass
        # --rl-weights <path> + --controller=rl to switch.
        rl_policy = None
        if args.controller == "rl":
            try:
                sys.path.insert(0, str(REPO_ROOT / "tools"))
                from rl_policy import TetraLocomotionPolicy
                rl_policy = TetraLocomotionPolicy(weights_path=args.rl_weights)
                rl_policy.load()
                log.info(f"RL controller engaged: {rl_policy.info()}")
            except Exception as e:
                log.warning(f"RL controller setup failed ({e}) — falling "
                            f"back to hand-coded P-controller.")
                rl_policy = None
        else:
            log.info("Controller: hand-coded P-controller "
                     "(use --controller=rl to engage RL plumbing)")

        # Controller / dof tables are filled on the first Play edge once
        # _ensure_art_initialized() has succeeded. See Bug 1 in
        # docs/HANDOFF_NEXT_CHAT.md — initializing before Play yields no DOFs.
        controller: ArticulationController | None = None
        dof_names: list[str] = []
        dof_idx: dict[str, int] = {}

        # Per-robot world targets at each phase boundary. Spawn was set up
        # earlier via _spawn_positions(n_robots) — same XY values.
        spawn_positions = np.array(positions, dtype=np.float32)
        n = len(spawn_positions)

        # ---- Waypoint-Follower choreography (refactored 2026-05-11 v2) --
        # East-entry layout. Bots+pallet spawn OUTSIDE the cabin (east), and
        # the demo transports the pallet WESTWARD into the cabin via the
        # open east fuselage end (the only opening big enough — passenger
        # doors at Y=±1.8 would have been too narrow for the 1.2m pallet).
        #
        # Phases:
        #   APPROACH_ROW : bots advance in a row westward (X+5.5 → X+4.8)
        #   SPLIT        : bots fan out around the pallet — OUTER bots wrap
        #                  around the pallet ends, INNER bots go directly
        #                  to the near corners (no path crossings)
        #   DOCK         : each bot reaches its assigned pallet corner
        #   LIFT_UP      : lift_joint ramps 0 → 0.10 m
        #   PICKUP       : kinematic-carry begins (leader bot_0 carries
        #                  pallet; followers track their own waypoints)
        #   ENTER_CABIN  : pallet+bots transit westward from X=+3.5 to X=-1.0
        #   RELEASE      : pallet handed off, becomes dynamic again
        #   LIFT_DOWN    : lift returns to 0
        #   RETREAT_CLEAR: bots sidestep OUTWARD in Y (away from pallet's
        #                  long axis) so cover_link disengages cleanly
        #   RETREAT      : bots traverse east back to the spawn row
        #
        # Pickup corners are pallet-relative (PALLET_XY-relative). Pallet
        # long-axis = world Y (1.2 m); the ±0.55 Y / ±0.69 X corners place
        # bots along the pallet long-sides, similar to the legacy layout.
        PALLET_XY = np.array([+3.5, 0.0], dtype=np.float32)
        # Bot ↔ corner assignment (outer wraps, inner direct — eliminates
        # the path crossings the user reported in the south-entry version):
        #   bot_0 (Y=-1.5, outermost south)  → SW corner (wraps via south)
        #   bot_1 (Y=-0.5, inner south)      → SE corner (direct)
        #   bot_2 (Y=+0.5, inner north)      → NE corner (direct)
        #   bot_3 (Y=+1.5, outermost north)  → NW corner (wraps via north)
        # Corner labels here use compass directions relative to pallet
        # centre at world (+3.5, 0). "W" corners are at the cabin-side edge
        # of the pallet (X = pallet_X - 0.69 = +2.81), "E" corners at the
        # row-side edge (X = +4.19).
        corner_offsets = np.array([
            [-0.69, -0.55],   # bot_0: SW (cabin-side, south)
            [+0.69, -0.55],   # bot_1: SE (row-side, south)
            [+0.69, +0.55],   # bot_2: NE (row-side, north)
            [-0.69, +0.55],   # bot_3: NW (cabin-side, north)
        ], dtype=np.float32)
        pickup_world = corner_offsets[:n] + PALLET_XY  # absolute world XY

        # Delivery: pallet target inside the cabin at X=-1.0, Y=0 — well
        # inside the cabin's west wall (-3.0) and centred Y-wise. Pallet's
        # 1.2m long-Y fits the 3.6m Y-corridor easily. Bot delivery = pickup
        # corner shifted by (DELIVERY - PALLET) = (-4.5, 0).
        DELIVERY_XY = np.array([-1.0, 0.0], dtype=np.float32)
        delivery_world = pickup_world + (DELIVERY_XY - PALLET_XY)

        # Split-staging: intermediate waypoint between row and dock corner.
        # Outer bots (0, 3) wrap around the pallet's south/north end going
        # FAR west of the pallet centre — this keeps the long wrap motion
        # in the chassis-forward direction (where mecanum is fast) and
        # leaves only a short Y-strafe to the corner (where mecanum is
        # slow). Earlier wrap waypoints at (+3.5, ±1.7) required ~1.3m of
        # diagonal half-strafe in 3s and the bots couldn't keep up — they
        # got stuck at (+3.16, ±1.23) far from the dock. Inner bots (1, 2)
        # have an almost-direct path to the row-side (east) corners.
        split_world = np.array([
            [+2.5, -1.5],   # bot_0: far-west south wrap
            [+5.0, -0.55],  # bot_1: just east of SE corner
            [+5.0, +0.55],  # bot_2: just east of NE corner
            [+2.5, +1.5],   # bot_3: far-west north wrap
        ], dtype=np.float32)[:n]

        # Row-advance: every bot keeps its Y but advances X from spawn
        # (+5.5) to +4.8. Pure forward motion in row formation.
        ROW_ADVANCE_X = +4.8
        row_advance_world = np.stack([
            np.full(n, ROW_ADVANCE_X, dtype=np.float32),
            spawn_positions[:, 1],
        ], axis=1)

        # Choreography timeline (sim seconds). Wrap phases extended by 2s
        # because the outer-bots' wrap-then-dock involves a slow Y-strafe.
        # Retreat split into CLEAR (Y-sidestep) → TRAVERSE (pure east at
        # clear-Y, past the pallet) → END (final Y-adjust to spawn-row),
        # so the inner bots don't re-enter the pallet's Y-zone while
        # passing it eastward and drag it along.
        # G1 pickup added 2026-05-11: after the bots retreat, the G1 next
        # to the delivered pallet picks up box_a via a scripted kinematic
        # lift (no controller; pure visual choreo).
        T_APPROACH_ROW_END  = 4.0    # row at X=+4.8, all bots in line
        T_SPLIT_END         = 9.0    # at split-staging waypoints
        T_DOCK_END          = 13.0   # at pallet corners
        T_LIFT_UP_END       = 15.0   # 2s ramp 0 → 0.10 m
        T_PICKUP_HOLD       = 15.5   # 0.5s settle, then kinematic-carry fires
        T_ENTER_CABIN_END   = 26.0   # pallet inside cabin at delivery centre
        T_RELEASE_HOLD      = 26.5   # carry released, pallet dynamic again
        T_LIFT_DOWN_END     = 28.5   # 2s ramp 0.10 → 0 m
        T_RETREAT_CLEAR     = 30.5   # bots sidestep Y-outward from pallet
        T_RETREAT_TRAVERSE  = 36.0   # pure east traverse at clear-Y
        T_RETREAT_END       = 38.5   # final align to spawn-row Y
        # Phase B (2026-05-11): 6-state G1 pickup machine.
        # Joint angles interpolate between IK-precomputed keyframes; box_a
        # follows the FK-computed wrist world pose during GRASP..RELEASE.
        T_G1_WAIT_END    = 39.5   # G1 at REST pose; bots have retreated
        T_G1_REACH_END   = 41.0   # arm: REST -> REACH (above box, 10cm up)
        T_G1_GRASP_END   = 41.5   # arm: REACH -> GRASP (at box top); carry on
        T_G1_LIFT_END    = 43.0   # arm: GRASP -> LIFT (30cm above box)
        T_G1_PLACE_END   = 45.5   # arm: LIFT -> PLACE (above drop zone)
        T_G1_RELEASE_END = 46.0   # arm: PLACE -> RELEASE; carry off, box drops
        T_G1_DONE        = 47.5   # arm: RELEASE -> REST (return to idle)
        # Legacy aliases — keep so other code that still references them
        # (snapshots, _g1_pickup_update arg list) compiles. Map to the
        # nearest equivalent boundary.
        T_G1_PICKUP_START    = T_G1_WAIT_END
        T_G1_PICKUP_LIFT_END = T_G1_LIFT_END
        T_G1_PICKUP_HOLD_END = T_G1_RELEASE_END
        T_DONE              = T_G1_DONE

        # Camera auto-cycling schedule (2026-05-13 polish-sprint).
        # Defined here AFTER T_* phase constants. Each tuple = (end_time,
        # camera_path). _camera_for_time scans linearly. The order maps
        # to dramatic beats: wide → close → POV → top → interior.
        camera_schedule[:] = [
            (T_APPROACH_ROW_END,   "/World/cameras/cam_overview"),      # 0-4s spawn + advance
            (T_SPLIT_END,          "/World/cameras/cam_top_down"),      # 4-9s show split clearly
            (T_DOCK_END,           "/World/cameras/cam_pallet_east"),   # 9-13s docking close
            (T_PICKUP_HOLD,        "/World/cameras/cam_overview"),      # 13-15.5s lift+pickup
            (T_ENTER_CABIN_END,    "/World/cameras/cam_pallet_east"),   # 15.5-26s transport
            (T_RETREAT_END,        "/World/cameras/cam_top_down"),      # 26-38.5s release+retreat
            (T_G1_REACH_END,       "/World/cameras/cam_inside_cabin"),  # 38.5-41s G1 wait+reach
            (T_G1_PLACE_END,       "/World/cameras/cam_g1_overshoulder"),  # 41-45.5s pickup+place
            (T_G1_DONE,            "/World/cameras/cam_inside_cabin"),  # 45.5-47.5s release+idle
        ]
        if not args.headless:
            try:
                from omni.kit.viewport.utility import get_active_viewport
                _viewport_api[0] = get_active_viewport()
                if _viewport_api[0] is not None:
                    log.info(f"  Camera auto-cycling: viewport API ready "
                             f"({len(camera_schedule)} phase-anchored switches)")
            except Exception as _e:
                log.warning(f"  Camera auto-cycling disabled "
                            f"(viewport API unavailable: {_e})")

        def _camera_for_time_impl(t_now):
            """Return the camera path for the phase that contains t_now."""
            for (t_end, path) in camera_schedule:
                if t_now < t_end:
                    return path
            return "/World/cameras/cam_inside_cabin"

        def _maybe_switch_camera_impl(t_now):
            """If the scheduled camera changed since last frame, switch
            the viewport's active camera. No-op in headless."""
            if _viewport_api[0] is None:
                return
            wanted = _camera_for_time_impl(t_now)
            if wanted == _active_camera_path[0]:
                return
            try:
                _viewport_api[0].set_active_camera(wanted)
                log.info(f"  camera switch @ t={t_now:.2f}s -> {wanted}")
                _active_camera_path[0] = wanted
            except Exception:
                pass

        # Bind to the names the run-loop uses (was stubbed earlier).
        _camera_for_time = _camera_for_time_impl
        if args.camera_cycle:
            log.info("  Camera auto-cycling ENABLED (--camera-cycle) — "
                     "viewport switches through 9 anchor cams across phases")
            _maybe_switch_camera = _maybe_switch_camera_impl
        else:
            _maybe_switch_camera = lambda t_now: None  # noqa: E731

        LIFT_MAX = 0.10

        def _lift_target(t):
            """Piecewise lift_joint position over the scenario timeline."""
            if t < T_DOCK_END:
                return 0.0
            if t < T_LIFT_UP_END:
                a = (t - T_DOCK_END) / (T_LIFT_UP_END - T_DOCK_END)
                return LIFT_MAX * a
            if t < T_RELEASE_HOLD:
                return LIFT_MAX
            if t < T_LIFT_DOWN_END:
                a = (t - T_RELEASE_HOLD) / (T_LIFT_DOWN_END - T_RELEASE_HOLD)
                return LIFT_MAX * (1.0 - a)
            return 0.0

        # Retreat-clear waypoint: after lift-down, bots first sidestep
        # OUTWARD in Y (away from pallet's long axis), then traverse east
        # past the pallet keeping that clear-Y, then align to spawn-Y.
        # Each bot gets a UNIQUE clear-Y so the four bots don't converge
        # to the same row during the eastward traverse and collide:
        #   outer bots (|spawn_y|=1.5) use their spawn-Y (already outside
        #     the pallet's ±0.6 Y extent)
        #   inner bots (|spawn_y|=0.5) use ±0.9 (just past the pallet edge)
        # Bot-bot minimum Y-separation during traverse = 0.6 m, well
        # above chassis edge-to-edge collision threshold.
        retreat_clear_world = np.zeros_like(delivery_world)
        for i in range(n):
            dx, dy = delivery_world[i]
            sy = spawn_positions[i, 1]
            if abs(sy) > 1.0:
                y_clear = float(sy)
            else:
                y_clear = -0.9 if sy < 0 else +0.9
            retreat_clear_world[i, 0] = dx     # keep X during sidestep
            retreat_clear_world[i, 1] = y_clear

        # Per-robot waypoints: list of (t, world_x, world_y, yaw).
        # Per user feedback 2026-05-11 (iter 3): the diagonal-pair rule
        # from iter 2 was wrong — the correct pattern is "face the pallet
        # along X". Bots WEST of the pallet centre (co_x < 0: SW & NW
        # corners) need yaw=π so their URDF -X (docking) side faces world
        # +X = toward pallet. Bots EAST of the pallet (co_x > 0: SE & NE)
        # need yaw=0 so docking faces world -X = toward pallet. With this
        # rule all four bots present their docking side to the pallet.
        waypoints_per_robot = []
        for i in range(n):
            sx, sy = spawn_positions[i]
            rx, ry = row_advance_world[i]
            spx, spy = split_world[i]
            px, py = pickup_world[i]
            dx, dy = delivery_world[i]
            cx, cy = retreat_clear_world[i]
            co_x, _ = corner_offsets[i]
            spawn_yaw = math.pi if co_x < 0 else 0.0
            waypoints_per_robot.append([
                (0.0,                 sx,  sy,  spawn_yaw),
                (T_APPROACH_ROW_END,  rx,  ry,  spawn_yaw),
                (T_SPLIT_END,         spx, spy, spawn_yaw),
                (T_DOCK_END,          px,  py,  spawn_yaw),
                (T_LIFT_UP_END,       px,  py,  spawn_yaw),
                (T_PICKUP_HOLD,       px,  py,  spawn_yaw),
                (T_ENTER_CABIN_END,   dx,  dy,  spawn_yaw),
                (T_RELEASE_HOLD,      dx,  dy,  spawn_yaw),
                (T_LIFT_DOWN_END,     dx,  dy,  spawn_yaw),
                (T_RETREAT_CLEAR,     cx,  cy,  spawn_yaw),
                (T_RETREAT_TRAVERSE,  sx,  cy,  spawn_yaw),
                (T_RETREAT_END,       sx,  sy,  spawn_yaw),
            ])

        # State events fire once when sim time crosses their threshold.
        # PICKUP fires AFTER the lift has settled so the carry-offset is
        # captured at the engaged pose. G1 pickup-start fires after the
        # bots have fully retreated so the box-on-pallet pose is settled.
        events = [
            (T_PICKUP_HOLD,     "PICKUP",     _pickup_attach),
            (T_RELEASE_HOLD,    "RELEASE",    _pickup_release),
            # Phase B: G1 pickup is now driven entirely by _g1_step (per-frame
            # state transitions + kinematic toggling), so no event needed.
        ]
        next_event = 0

        def _interp_waypoint(wps, t):
            """Smoothstep-eased interpolation of (x, y, yaw) along the
            waypoint list. smoothstep(a) = a²·(3-2a) makes motion
            accelerate gently from rest and decelerate before the next
            target — bots no longer jerk on phase boundaries."""
            if t <= wps[0][0]:
                return wps[0][1:]
            for j in range(len(wps) - 1):
                t0, x0, y0, yaw0 = wps[j]
                t1, x1, y1, yaw1 = wps[j + 1]
                if t0 <= t <= t1:
                    a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
                    a = a * a * (3.0 - 2.0 * a)   # smoothstep easing
                    return (
                        x0 + a * (x1 - x0),
                        y0 + a * (y1 - y0),
                        yaw0 + a * (yaw1 - yaw0),
                    )
            return wps[-1][1:]

        log.info(
            f"  scenario timeline: ROW@{T_APPROACH_ROW_END}s "
            f"->SPLIT@{T_SPLIT_END}s ->DOCK@{T_DOCK_END}s "
            f"->PICKUP@{T_PICKUP_HOLD}s ->ENTER_CABIN@{T_ENTER_CABIN_END}s "
            f"->RELEASE@{T_RELEASE_HOLD}s ->CLEAR@{T_RETREAT_CLEAR}s "
            f"->TRAVERSE@{T_RETREAT_TRAVERSE}s ->RETREAT@{T_RETREAT_END}s "
            f"->G1_LIFT@{T_G1_PICKUP_START}s ->G1_HOLD@{T_G1_PICKUP_LIFT_END}s, "
            f"total {T_DONE}s"
        )
        log.info("  Press Play in the GUI to start. Stop+Play to replay.")

        # ---- Diagnostic telemetry ----------------------------------------
        # Built per user request 2026-05-08: when something goes wrong
        # (pickup fails, robots fly apart, geometry is in the wrong place)
        # we need observable state to diagnose without GUI access. Snapshots
        # capture chassis/docking/pallet world positions, joint actual vs
        # target, and a docking↔pallet distance metric. Fired at
        # interesting timestamps around critical transitions.
        # Output goes to:
        #   - main launch.log (human-readable INFO lines)
        #   - logs/scenario_telem_<timestamp>.jsonl (structured, one
        #     snapshot per line, for offline analysis)
        import json
        telem_path = LOG_DIR / f"scenario_telem_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        log.info(f"  Telemetry JSONL: {telem_path}")
        snap_bbox_cache = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])

        def _bbox_mid(prim):
            snap_bbox_cache.Clear()
            r = snap_bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            mn, mx = r.GetMin(), r.GetMax()
            return (
                ((mn[0]+mx[0])/2, (mn[1]+mx[1])/2, (mn[2]+mx[2])/2),
                (mn[0], mn[1], mn[2]),
                (mx[0], mx[1], mx[2]),
            )

        def _scenario_snapshot(label: str, t_now: float, phase: str,
                               last_targets=None):
            """Dump physics state of bots/pallet/boxes + joint cur/target.
            Writes both human-readable lines to log and structured JSON to
            scenario_telem JSONL file for offline analysis."""
            lines = [f"=== SNAPSHOT [{label}] t={t_now:.2f}s phase={phase} ==="]
            record = {"label": label, "t": round(t_now, 3), "phase": phase,
                      "robots": [], "objects": {}}
            for i, robot_path in enumerate(robot_paths):
                # base_link is the dynamic chassis (URDF v3 refactor); the
                # legacy `chassis_link` is now just a fixed-joint tf-anchor
                # without bbox, so we query base_link for the actual pose.
                chassis = stage.GetPrimAtPath(f"{robot_path}/base_link")
                docking = stage.GetPrimAtPath(f"{robot_path}/docking_unit")
                ch_str = "?"
                dk_str = "?"
                ch_xyz = None
                dk_xyz = None
                if chassis.IsValid():
                    mid, mn, mx = _bbox_mid(chassis)
                    ch_str = f"({mid[0]:+.3f},{mid[1]:+.3f},{mid[2]:+.3f})"
                    ch_xyz = list(mid)
                if docking.IsValid():
                    mid, mn, mx = _bbox_mid(docking)
                    dk_str = f"({mid[0]:+.3f},{mid[1]:+.3f},{mid[2]:+.3f})"
                    dk_xyz = list(mid)
                lines.append(f"  robot_{i}  chassis@{ch_str}  docking@{dk_str}")
                record["robots"].append({"i": i, "chassis": ch_xyz,
                                         "docking": dk_xyz})

            for path_label, path in (
                ("pallet", "/World/pallet"),
                ("box_a",  "/World/box_a"),
                ("box_b",  "/World/box_b"),
            ):
                p = stage.GetPrimAtPath(path)
                if not p.IsValid():
                    continue
                mid, mn, mx = _bbox_mid(p)
                lines.append(f"  {path_label}@({mid[0]:+.3f},{mid[1]:+.3f},"
                             f"{mid[2]:+.3f})  Z[{mn[2]:+.3f}..{mx[2]:+.3f}]")
                record["objects"][path_label] = {"mid": list(mid),
                                                  "z_min": mn[2],
                                                  "z_max": mx[2]}

            # docking_0 ↔ pallet distance — key pickup-geometry metric
            docking0 = stage.GetPrimAtPath(f"{robot_paths[0]}/docking_unit")
            pallet = stage.GetPrimAtPath("/World/pallet")
            if docking0.IsValid() and pallet.IsValid():
                dmid, _, _ = _bbox_mid(docking0)
                pmid, _, _ = _bbox_mid(pallet)
                dx = dmid[0] - pmid[0]
                dy = dmid[1] - pmid[1]
                dz = dmid[2] - pmid[2]
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                lines.append(f"  d(docking_0 <-> pallet) = {dist:.3f}m  "
                             f"(dx={dx:+.3f} dy={dy:+.3f} dz={dz:+.3f})")

            # Joint actual positions vs last commanded targets. URDF v3
            # only exposes lift_joint + 4 wheel joints in the articulation;
            # holonomic XY+yaw is no longer joint-state-tracked here.
            if controller is not None and dof_names:
                try:
                    cur = art.get_joint_positions()
                    if cur is not None:
                        if hasattr(cur, "shape") and len(cur.shape) == 1:
                            cur = cur.reshape(1, -1)
                        for i in range(min(cur.shape[0], 2)):  # first 2 robots
                            parts = []
                            for jn in ("lift_joint",):
                                if jn in dof_idx:
                                    j_cur = float(cur[i, dof_idx[jn]])
                                    if last_targets is not None:
                                        j_tgt = float(last_targets[i, dof_idx[jn]])
                                        parts.append(
                                            f"{jn}=cur:{j_cur:+.3f}/tgt:{j_tgt:+.3f}")
                                    else:
                                        parts.append(f"{jn}={j_cur:+.3f}")
                            lines.append(f"  robot_{i} joints: {' '.join(parts)}")
                except Exception as e:
                    lines.append(f"  [joint state read failed: {e}]")
            # Chassis world pose (target-tracking accuracy: cur vs waypoint)
            # plus leader-follower sync metrics (key for Phase 2 verification).
            try:
                cur_positions_snap, _ = art.get_world_poses()
                cur_positions_snap = np.asarray(cur_positions_snap, dtype=np.float32)
                # Leader-anchored relative-position deltas
                if len(cur_positions_snap) > LEADER_BOT_INDEX:
                    leader_pos = cur_positions_snap[LEADER_BOT_INDEX]
                    leader_wp = _interp_waypoint(
                        waypoints_per_robot[LEADER_BOT_INDEX], t_now)
                    lines.append(
                        f"  LEADER bot_{LEADER_BOT_INDEX} cur@("
                        f"{leader_pos[0]:+.3f},{leader_pos[1]:+.3f},"
                        f"{leader_pos[2]:+.3f}) tgt@("
                        f"{leader_wp[0]:+.3f},{leader_wp[1]:+.3f},*) "
                        f"err=({leader_pos[0]-leader_wp[0]:+.3f},"
                        f"{leader_pos[1]-leader_wp[1]:+.3f})")
                    record["leader_idx"] = LEADER_BOT_INDEX
                    record["leader_err"] = [
                        float(leader_pos[0]-leader_wp[0]),
                        float(leader_pos[1]-leader_wp[1])]
                # Follower sync metric: |waypoint(follower) - waypoint(leader) - spawn_offset|
                # Should be 0 because waypoints share the same delivery_dxy.
                # Plus chassis-pos sync error to leader's actual pose.
                for i in range(len(cur_positions_snap)):
                    if i == LEADER_BOT_INDEX:
                        continue
                    cp = cur_positions_snap[i]
                    wx, wy, _ = _interp_waypoint(waypoints_per_robot[i], t_now)
                    err_x = cp[0] - wx
                    err_y = cp[1] - wy
                    # Sync vs leader: same shape of motion?
                    leader_pos = cur_positions_snap[LEADER_BOT_INDEX]
                    leader_wp = _interp_waypoint(
                        waypoints_per_robot[LEADER_BOT_INDEX], t_now)
                    expected_offset_x = wx - leader_wp[0]
                    expected_offset_y = wy - leader_wp[1]
                    actual_offset_x = cp[0] - leader_pos[0]
                    actual_offset_y = cp[1] - leader_pos[1]
                    sync_err_x = actual_offset_x - expected_offset_x
                    sync_err_y = actual_offset_y - expected_offset_y
                    lines.append(
                        f"  follower bot_{i} cur@({cp[0]:+.3f},"
                        f"{cp[1]:+.3f},{cp[2]:+.3f}) wp_err=("
                        f"{err_x:+.3f},{err_y:+.3f}) "
                        f"sync_to_leader=({sync_err_x:+.3f},"
                        f"{sync_err_y:+.3f})")
            except Exception as e:
                lines.append(f"  [chassis pose read failed: {e}]")

            # Phase B (2026-05-11): G1 state-machine telemetry.
            # g1_state_label[0] holds the current segment name (e.g.
            # "G1_REACH_TO_GRASP"); the hand-world pose is appended by
            # _g1_step. g1_pelvis_world_cached[0] is the pelvis world XYZ
            # for sanity-checking that G1 stays stationary.
            try:
                g1_state = g1_state_label[0]
                lines.append(f"  G1 state = {g1_state}")
                record["g1_state"] = g1_state
                # Pelvis world pos (sanity: must be constant for stationary G1)
                pelv = g1_pelvis_world_cached[0]
                if pelv is not None:
                    lines.append(f"  G1 pelvis world = ({pelv[0]:+.3f},"
                                 f"{pelv[1]:+.3f},{pelv[2]:+.3f}) "
                                 f"[invariant check]")
                    record["g1_pelvis_world"] = [float(x) for x in pelv]
                # Hand world (last entry from _g1_step is ('hand_world', [x,y,z]))
                hand_entry = None
                for entry in reversed(g1_state_label):
                    if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == "hand_world":
                        hand_entry = entry[1]
                        break
                if hand_entry is not None:
                    lines.append(f"  G1 hand world = ({hand_entry[0]:+.3f},"
                                 f"{hand_entry[1]:+.3f},{hand_entry[2]:+.3f})")
                    record["g1_hand_world"] = list(hand_entry)
                # G1 box-state (kinematic-carry active flag).
                record["g1_box_carry_active"] = bool(g1_box_state["active"])
            except Exception as e:
                lines.append(f"  [G1 telemetry skipped: {e}]")

            log.info("\n".join(lines))
            try:
                with open(telem_path, "a", encoding="utf-8") as fp:
                    fp.write(json.dumps(record) + "\n")
            except OSError as e:
                log.warning(f"  [telemetry write failed: {e}]")

        # Snapshot timestamps around critical transitions of the
        # waypoint-follower scenario. Sorted ascending; loop fires when t
        # crosses each. Dense around dock + pickup + release transitions.
        snap_times = sorted(set([
            0.10,                                 # SPAWN_SETTLED
            T_APPROACH_ROW_END - 0.01,            # row at advance line
            T_SPLIT_END - 0.01,                   # at split-staging waypoints
            T_DOCK_END - 0.01,                    # at pallet corners
            T_LIFT_UP_END - 0.01,                 # lift complete
            T_PICKUP_HOLD - 0.05,                 # just BEFORE kinematic-carry
            T_PICKUP_HOLD + 0.10,                 # just AFTER kinematic-carry
            T_PICKUP_HOLD + 0.50,                 # post-pickup settling
            (T_PICKUP_HOLD + T_ENTER_CABIN_END) / 2,  # mid-transport into cabin
            T_ENTER_CABIN_END - 0.01,             # arrived at delivery
            T_RELEASE_HOLD - 0.05,                # just before RELEASE
            T_RELEASE_HOLD + 0.10,                # just after RELEASE
            T_LIFT_DOWN_END - 0.01,               # lift back down
            T_RETREAT_CLEAR - 0.01,               # bots cleared sideways
            T_RETREAT_TRAVERSE - 0.01,            # bots past pallet eastward
            T_RETREAT_END - 0.01,                 # retreated to row
            T_G1_PICKUP_START + 0.10,             # just after G1 grabs box
            T_G1_PICKUP_LIFT_END - 0.01,          # box at chest height
            T_G1_PICKUP_HOLD_END - 0.01,          # G1 holding box
            # Phase B G1-state-machine snapshots:
            T_G1_REACH_END - 0.01,                # arm at REACH (above box)
            T_G1_GRASP_END + 0.05,                # right after kinematic-carry start
            T_G1_LIFT_END - 0.01,                 # box lifted 30cm
            T_G1_PLACE_END - 0.01,                # arm at PLACE (above drop)
            T_G1_RELEASE_END + 0.10,              # right after carry release
            T_G1_DONE - 0.01,                     # final box rest position (post-drop)
        ]))

        # Run state. Reset whenever the timeline transitions from
        # not-playing to playing (i.e. Play button pressed).
        def _reset_scenario():
            _pickup_release()  # in case a previous run left fixed-joints
            # Reset G1 pickup state too so a Stop+Play replay re-fires
            # the kinematic lift cleanly.
            g1_box_state["active"] = False
            g1_box_state["start_xyz"] = None
            g1_box_state["target_xyz"] = None
            _set_box_kinematic(False)
            return 0.0, 0, "ROW"

        def _phase_for_time(t_now: float, current: str) -> str:
            """Time-driven phase label for telemetry. Events PICKUP/RELEASE
            still override this with their event names — `current` carries
            that override forward until the next time-bucket transition."""
            if t_now < T_APPROACH_ROW_END:
                return "ROW"
            if t_now < T_SPLIT_END:
                return "SPLIT"
            if t_now < T_DOCK_END:
                return "DOCK"
            if t_now < T_LIFT_UP_END:
                return "LIFT_UP"
            if t_now < T_PICKUP_HOLD:
                return "PICKUP_SETTLE"
            if t_now < T_ENTER_CABIN_END:
                return "ENTER_CABIN"
            if t_now < T_RELEASE_HOLD:
                return "RELEASE_HOLD"
            if t_now < T_LIFT_DOWN_END:
                return "LIFT_DOWN"
            if t_now < T_RETREAT_CLEAR:
                return "RETREAT_CLEAR"
            if t_now < T_RETREAT_TRAVERSE:
                return "RETREAT_TRAVERSE"
            if t_now < T_RETREAT_END:
                return "RETREAT"
            if t_now < T_G1_PICKUP_START:
                return "G1_WAIT"
            if t_now < T_G1_PICKUP_LIFT_END:
                return "G1_LIFT"
            if t_now < T_G1_PICKUP_HOLD_END:
                return "G1_HOLD"
            return "DONE"

        t, next_event, current_phase = _reset_scenario()
        next_snap = 0
        last_commanded_targets = None
        dt = 1.0 / 60.0
        frames_done = 0
        prev_playing = False
        scenario_completed_this_run = False
        # Wheel-spin animation state. Wheels are k=0/d=5 passive joints with
        # μ=0.02 low-friction collision (intentional, so the planar dummy
        # joints can push the chassis in any direction without rolling
        # resistance). Side-effect: wheels don't rotate from ground contact.
        # Drive them visually instead by integrating angle = chassis_speed *
        # dt / wheel_radius and pushing the joint position each frame.
        WHEEL_RADIUS = 0.05  # 10 cm wheel diameter (per URDF)
        wheel_dof_idxs: list[int] = []  # populated at Play edge from dof_idx
        wheel_angles_state = np.zeros(n, dtype=np.float32)

        while kit.is_running():
            is_playing = timeline_iface.is_playing()

            # Detect Play edge: reset scenario state at the start of each Play.
            if is_playing and not prev_playing:
                # Wipe stale state so a Stop+Play replay rebinds cleanly.
                controller = None
                dof_names = []
                dof_idx = {}
                if _ensure_art_initialized():
                    controller = ArticulationController()
                    controller.initialize(art)
                    dof_names = list(art.dof_names)
                    dof_idx = {name: i for i, name in enumerate(dof_names)}
                    log.info(f"Scenario controller initialized for {len(dof_names)} DOFs")
                t, next_event, current_phase = _reset_scenario()
                next_snap = 0
                last_commanded_targets = None
                scenario_completed_this_run = False
                # Resolve wheel DOF indices for spin animation; reset the
                # accumulated angle so a Stop+Play replay starts at 0.
                wheel_dof_idxs = [
                    dof_idx[wn]
                    for wn in ("wheel_fl_joint", "wheel_fr_joint",
                               "wheel_rl_joint", "wheel_rr_joint")
                    if wn in dof_idx
                ]
                wheel_angles_state[:] = 0.0
                log.info(f"  wheel-spin animation: {len(wheel_dof_idxs)} "
                         f"wheel DOFs per bot")
                log.info("  scenario started (Play edge)")
            prev_playing = is_playing

            # If not playing, hold position — just pump Kit so the GUI stays
            # responsive (user can rotate viewport, click Play, etc.).
            if not is_playing:
                kit.update()
                frames_done += 1
                if args.frames and frames_done >= args.frames:
                    log.info(f"Reached frame limit ({args.frames}), stopping")
                    break
                continue

            if controller is None or not dof_names:
                # Articulation didn't finish initializing yet — pump and wait.
                kit.update()
                frames_done += 1
                continue
            if not _is_art_handle_valid():
                # Handle went stale mid-run (e.g. PhysX kernel failure tore
                # down the sim view). Drop targets this frame and clear the
                # init flag so the next Play edge re-binds cleanly.
                log.warning("Articulation handle invalidated mid-run; "
                            "skipping frame and awaiting Stop+Play.")
                art_state["initialized"] = False
                controller = None
                kit.update()
                frames_done += 1
                continue

            # Time-driven phase label updates each frame (so telemetry shows
            # ROW/SPLIT/DOCK/ENTER_CABIN/etc instead of staying on the last
            # event name). PICKUP/RELEASE events still override transiently
            # via current_phase = name below — the next frame's time bucket
            # then takes over.
            current_phase = _phase_for_time(t, current_phase)

            # Fire state-machine events
            while next_event < len(events) and t >= events[next_event][0]:
                _, name, fn = events[next_event]
                log.info(f"  scenario event @ t={t:.2f}s: {name}")
                fn()
                current_phase = name
                next_event += 1

            # Event functions like _pickup_release remove FixedJoints, which
            # can tear down the PhysX simulation view and invalidate the
            # articulation handle (_physics_view attribute disappears). If
            # that just happened, re-initialize art + controller in place so
            # the scenario doesn't freeze.
            if not _is_art_handle_valid():
                log.warning("Articulation handle invalidated by scenario "
                            "event; re-initializing in place.")
                if _ensure_art_initialized():
                    controller = ArticulationController()
                    controller.initialize(art)
                    dof_names = list(art.dof_names)
                    dof_idx = {name: i for i, name in enumerate(dof_names)}
                    log.info(f"Scenario controller re-bound: "
                             f"{len(dof_names)} DOFs")
                else:
                    controller = None
                    dof_names = []
                    dof_idx = {}
                    kit.update()
                    frames_done += 1
                    continue

            # ---- Chassis velocity control (URDF v3 + Phase-2 architecture) ----
            # Holonomic XY+yaw motion via velocity-based P-control on the
            # articulation root. Velocity-based (rather than pose-teleport)
            # keeps physics integration intact: PhysX sees a velocity-
            # commanded body, integrates it forward each step, and the
            # FixedJoint between leader's docking_unit and pallet (Phase-2
            # Leader-Follower) applies its constraint forces in the natural
            # physics flow. Earlier set_world_poses caused the pallet to
            # accumulate impulse error and fly into deep space because the
            # teleport conflicted with the constraint each frame.
            lift_pos = _lift_target(t)
            joint_targets = np.zeros((n, len(dof_names)), dtype=np.float32)
            if "lift_joint" in dof_idx:
                joint_targets[:, dof_idx["lift_joint"]] = lift_pos
            last_commanded_targets = joint_targets

            try:
                cur_positions, cur_orientations = art.get_world_poses()
                cur_positions = np.asarray(cur_positions, dtype=np.float32)
                # Articulation.get_velocities() returns shape (n, 6):
                # columns 0..2 = linear, 3..5 = angular.
                cur_velocities = np.asarray(art.get_velocities(), dtype=np.float32)
                cur_lin_vel = cur_velocities[:, 0:3].copy()
                cur_ang_vel = cur_velocities[:, 3:6].copy()
            except Exception as e:
                log.warning(f"velocity read failed: {e}; skipping update")
                cur_positions = None

            if cur_positions is not None:
                # P-controller on position → linear velocity. k_p=5.0/s
                # gives a ~0.2s time constant, ~1.8cm steady-state error
                # at 0.09 m/s waypoint velocity. Plenty for the demo.
                k_p = 5.0
                target_lin_vel = cur_lin_vel.copy()  # preserve Z (gravity)
                target_ang_vel = cur_ang_vel.copy()
                for i in range(n):
                    wx, wy, wyaw = _interp_waypoint(waypoints_per_robot[i], t)
                    cur_yaw = math.atan2(
                        2 * (cur_orientations[i][0] * cur_orientations[i][3]
                             + cur_orientations[i][1] * cur_orientations[i][2]),
                        1 - 2 * (cur_orientations[i][2] * cur_orientations[i][2]
                                 + cur_orientations[i][3] * cur_orientations[i][3]))
                    yaw_err = wyaw - cur_yaw
                    while yaw_err > math.pi: yaw_err -= 2 * math.pi
                    while yaw_err < -math.pi: yaw_err += 2 * math.pi
                    if args.controller == "rl" and rl_policy is not None:
                        # Stufe-1 RL plumbing exercise: build the 9-dim
                        # observation, ask the policy for an action,
                        # use it as target body velocity. Stub policy
                        # internally does P-control with the same k_p, so
                        # the visible behaviour is identical to the hand-
                        # coded path. Once a trained policy replaces the
                        # stub, this same plumbing handles it.
                        obs = np.array([
                            cur_positions[i, 0], cur_positions[i, 1], cur_yaw,
                            cur_lin_vel[i, 0], cur_lin_vel[i, 1], cur_ang_vel[i, 2],
                            wx - cur_positions[i, 0],
                            wy - cur_positions[i, 1],
                            yaw_err,
                        ], dtype=np.float32)
                        act = rl_policy.compute_action(obs)
                        target_lin_vel[i, 0] = float(act[0])
                        target_lin_vel[i, 1] = float(act[1])
                        target_ang_vel[i, 2] = float(act[2])
                    else:
                        target_lin_vel[i, 0] = (wx - cur_positions[i, 0]) * k_p
                        target_lin_vel[i, 1] = (wy - cur_positions[i, 1]) * k_p
                        target_ang_vel[i, 2] = yaw_err * k_p

            # Fire pre-target diagnostic snapshots BEFORE applying targets so
            # we see the true state from the previous frame's settled physics.
            while next_snap < len(snap_times) and t >= snap_times[next_snap]:
                _scenario_snapshot(f"t={t:.2f}", t, current_phase,
                                   last_targets=last_commanded_targets)
                next_snap += 1

            try:
                if cur_positions is not None:
                    art.set_velocities(np.concatenate(
                        [target_lin_vel, target_ang_vel], axis=-1))
                if "lift_joint" in dof_idx:
                    if n == 1:
                        controller.apply_action(
                            ArticulationAction(joint_positions=joint_targets[0]))
                    else:
                        art.set_joint_position_targets(joint_targets)
                # Wheel-spin animation. Wheels need explicit position-set
                # since their drive is k=0 (passive) and ground friction is
                # μ=0.02 (so no rolling from contact). Compute per-bot
                # chassis XY speed and integrate angle = speed * dt / r.
                # Wrap to (-π, π] each frame because PhysX rejects drive
                # targets outside [-2π, 2π] for revolute joints — and Isaac
                # Sim's set_joint_positions implicitly seeds the drive
                # target. The visual wrap is invisible at typical wheel-
                # spin rates (~6 rad/s → one revolution per second).
                if (cur_positions is not None and wheel_dof_idxs):
                    speeds = np.linalg.norm(target_lin_vel[:, :2], axis=1)
                    wheel_angles_state += (speeds * dt / WHEEL_RADIUS).astype(np.float32)
                    wheel_angles_state = np.mod(
                        wheel_angles_state + np.pi, 2 * np.pi
                    ).astype(np.float32) - np.pi
                    cur_jp = art.get_joint_positions()
                    if cur_jp is not None:
                        cur_jp = np.asarray(cur_jp, dtype=np.float32).copy()
                        for w_idx in wheel_dof_idxs:
                            cur_jp[:, w_idx] = wheel_angles_state
                        art.set_joint_positions(cur_jp)
                # Phase-2 leader carry: teleport pallet to follow leader's
                # current chassis position. Pallet is kinematic during
                # carry so this is the canonical way to move it.
                if cur_positions is not None and carry_state["active"]:
                    _carry_update_pallet(cur_positions[LEADER_BOT_INDEX])
                # Phase B: IK-driven G1 pickup state machine.
                # Computes joint angles per frame, runs FK, teleports
                # box_a to follow the wrist during GRASP..RELEASE.
                _g1_step(t)
                # G1 arm animation removed in v6 — G1 is fully kinematic
                # so joint drives don't propagate. The _g1_arm_animate
                # helper is kept dormant in case manual forward-kinematics
                # is added later.
            except AttributeError as e:
                if "_physics_view" in str(e):
                    log.warning(f"chassis velocity/target set stale: {e}; "
                                f"will re-init next iteration.")
                    art_state["initialized"] = False
                    controller = None
                else:
                    raise
            kit.update()
            t += dt
            frames_done += 1

            if args.frames and frames_done >= args.frames:
                log.info(f"Reached frame limit ({args.frames}), stopping")
                break

            # Scenario complete: stop timeline so user can Play again to
            # replay. Do NOT break — keep the app alive (replayable).
            if t >= T_DONE and not scenario_completed_this_run:
                log.info(f"Scenario complete at t={t:.2f}s. "
                         f"Click Play to replay; close window to exit.")
                timeline_iface.stop()
                scenario_completed_this_run = True
                # SDG/Data-Card stage-1 (2026-05-12): write a one-page
                # markdown summary of what data this run "produced".
                # Even without --sdg, every run produces telemetry +
                # frame-render-equivalent data. The card communicates
                # the synthetic-data-pipeline value to non-technical
                # demo audiences without needing any extra flag.
                try:
                    sdg_active = bool(args.sdg)
                    snapshot_count = next_snap
                    sim_seconds = float(t)
                    total_frames = int(frames_done)
                    sdg_dir_glob = sorted(LOG_DIR.glob("sdg_*"))
                    sdg_latest = (sdg_dir_glob[-1] if sdg_dir_glob and sdg_active
                                  else None)
                    card_path = LOG_DIR / f"data_card_{time.strftime('%Y%m%d_%H%M%S')}.md"
                    real_world_equivalent_min = sim_seconds / 60.0 * 15.0  # 15x factor
                    with open(card_path, "w", encoding="utf-8") as fp:
                        fp.write("# Synthetic Data Card — TETRABot cabin_assembly run\n\n")
                        fp.write(f"**Run timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        fp.write(f"**Sim duration**: {sim_seconds:.2f} s "
                                 f"({total_frames} frames)\n")
                        fp.write(f"**Telemetry snapshots**: {snapshot_count} "
                                 f"(JSONL: `{telem_path.name}`)\n")
                        fp.write(f"**Telemetry log**: `logs/{telem_path.name}`\n")
                        fp.write(f"**Stdout/stderr**: `logs/latest.log`\n\n")
                        fp.write("## Data products\n\n")
                        fp.write("| Channel | Per snapshot | Total |\n")
                        fp.write("|---|---|---|\n")
                        fp.write(f"| Robot poses (4 TETRABots) | 4 × pos+yaw | "
                                 f"{snapshot_count*4} sample sets |\n")
                        fp.write(f"| G1 hand world pose | 1 × XYZ | "
                                 f"{snapshot_count} sample sets |\n")
                        fp.write(f"| Pallet + 2 boxes pose | 3 × XYZ+bbox | "
                                 f"{snapshot_count*3} sample sets |\n")
                        fp.write(f"| Joint actual vs target | 5 DoFs × 4 bots | "
                                 f"{snapshot_count*20} measurements |\n")
                        if sdg_active:
                            fp.write(f"| RGB / Depth / Instance / BBox2D | "
                                     f"frame-rate (60 Hz target) | written to "
                                     f"`{sdg_latest.name if sdg_latest else 'logs/sdg_<timestamp>/'}` |\n")
                        else:
                            fp.write(f"| RGB / Depth / Instance / BBox2D | "
                                     f"DISABLED — re-run with `--sdg --cameras` to enable | n/a |\n")
                        fp.write(f"\n## Equivalent real-world acquisition cost\n\n")
                        fp.write(f"This sim produced **{snapshot_count} curated snapshots "
                                 f"+ {total_frames} render-frames in {sim_seconds:.0f} sim-seconds**. "
                                 f"Hand-collecting equivalent real-world traces (4 mobile robots + "
                                 f"humanoid arm motion + box manipulation) would take approximately "
                                 f"**{real_world_equivalent_min:.0f} minutes** of synchronised "
                                 f"multi-robot operation in a physical mockup cabin — assuming "
                                 f"perfect-take-rate (no resets needed).\n\n")
                        fp.write(f"## Suggested next-step uses\n\n")
                        fp.write(f"- Train RL/MARL pallet-carry policy (see "
                                 f"`tools/train_g1_locomotion.py` scaffold)\n")
                        fp.write(f"- Fine-tune GR00T-N1.5 grasp policy on the box-pickup phase\n")
                        fp.write(f"- Generate **N×** of this run with Replicator domain randomization\n")
                        fp.write(f"  (lighting/textures/camera-pose) — robust-policy training set\n")
                        fp.write(f"- Use telemetry JSONL as ground-truth for vision-based "
                                 f"object-tracking benchmarks\n")
                    log.info(f"Data card written: {card_path}")
                    log.info(f"  > Sim {sim_seconds:.1f}s, {snapshot_count} snapshots, "
                             f"{total_frames} frames "
                             f"({'SDG ON, '+sdg_latest.name if sdg_latest else 'SDG off'})")
                except Exception as e:
                    log.warning(f"Data card generation failed: {e}")
                if args.headless:
                    # In headless there's no UI to replay — exit cleanly.
                    break

            if frames_done % 300 == 0:
                log.debug(f"  scenario t={t:.2f}s phase={current_phase}")

            # Camera auto-cycling: check at ~10 Hz (every 6 frames) if
            # the scheduled camera changed. Cheap (set_active_camera is
            # only called on actual change). Skipped in headless.
            if frames_done % 6 == 0:
                _maybe_switch_camera(t)

            # HUD overlay live updates (cheap — only updates ui Label
            # strings). Refresh every 6 frames (~10 Hz at 60 fps) to
            # keep the rendering cost negligible.
            if (hud_state["phase"] is not None and frames_done % 6 == 0):
                try:
                    g1_st = g1_state_label[0] if isinstance(
                        g1_state_label[0], str) else "G1: idle"
                    carry_str = ("Box carry: ACTIVE (kinematic)" if
                                 g1_box_state["active"] else
                                 "Box carry: inactive")
                    hud_state["phase"].text = f"Phase: {current_phase}"
                    hud_state["frame"].text = (
                        f"Sim t: {t:.2f} s  |  Frame: {frames_done}")
                    hud_state["snap"].text = (
                        f"Telemetry snapshots: {next_snap}")
                    hud_state["g1"].text = f"G1 state: {g1_st}"
                    hud_state["carry"].text = carry_str
                except Exception:
                    pass
    else:
        if args.drop_mode:
            log.info("DROP MODE — press Play, watch objects fall. Settled positions "
                     "are logged every 60 frames so you can read them off.")
        else:
            log.info("Interactive mode — drive joints via Physics Inspector. Close window to exit.")

        # Track which prims to dump per-frame Z (and roll/pitch for robots)
        track_prims: list[tuple[str, str]] = []  # (label, prim_path)
        for i, rp in enumerate(robot_paths):
            # URDF v3 (2026-05-08): chassis_link is now a tf-only fixed
            # alias under base_link with no bbox. Track base_link instead.
            track_prims.append((f"tetrabot_{i}.chassis", f"{rp}/base_link"))
        if args.workpiece:
            track_prims.append(("pallet",  "/World/pallet"))
            track_prims.append(("box_a",   "/World/box_a"))
            track_prims.append(("box_b",   "/World/box_b"))

        bbox_cache_iv = UsdGeom.BBoxCache(0.0, [UsdGeom.Tokens.default_])
        frames_done = 0
        while kit.is_running():
            kit.update()
            frames_done += 1
            if args.frames and frames_done >= args.frames:
                log.info(f"Reached frame limit ({args.frames}), stopping")
                break
            if args.drop_mode and frames_done % 60 == 0 and timeline_iface.is_playing():
                lines = [f"Settled-position read @ frame {frames_done} (sim t≈{frames_done/60.0:.1f}s):"]
                for label, path in track_prims:
                    p = stage.GetPrimAtPath(path)
                    if not p.IsValid():
                        continue
                    bbox_cache_iv.Clear()  # invalidate so we get current world transform
                    r = bbox_cache_iv.ComputeWorldBound(p).ComputeAlignedRange()
                    lines.append(
                        f"  {label:24s}  "
                        f"X={r.GetMin()[0]:+.3f}..{r.GetMax()[0]:+.3f}  "
                        f"Y={r.GetMin()[1]:+.3f}..{r.GetMax()[1]:+.3f}  "
                        f"Z={r.GetMin()[2]:+.3f}..{r.GetMax()[2]:+.3f}"
                    )
                log.info("\n".join(lines))
            if not args.drop_mode and frames_done % 600 == 0:
                log.debug(f"  interactive frame {frames_done}")

    log.info("Stopping timeline")
    omni.timeline.get_timeline_interface().stop()
    return 0


# ---- Run with full traceback logging ----
exit_code = 1
try:
    exit_code = main()
except Exception as e:
    log.error(f"Unhandled exception: {e}")
    log.error(traceback.format_exc())
finally:
    log.info(f"Closing SimulationApp (exit_code={exit_code})")
    kit.close()
    sys.exit(exit_code)
