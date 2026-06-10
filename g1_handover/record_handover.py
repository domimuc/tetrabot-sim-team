"""Phase-gated G1 handover recording (cabin_assembly in Isaac Lab).

Flow (one continuous run; the video is the whole stream, the HDF5 is G1-only):
  PRE_DELIVERY  TETRABots drive away (render-only, NOT recorded)
   -> TRIGGER   pallet placed + bots retreated
   -> RECORD    the G1 manipulation is recorded; on N consecutive successes
                (`box_picked_and_turned`) the episode is exported to HDF5.

Two action sources in the RECORD phase, via --teleop_device:
  * scripted     (default, on-box proof): a kinematic box lift+turn fires the
                 success term — proves the pipeline without a headset. NOT a
                 replayable manipulation (actions are idle G1).
  * handtracking (Apple Vision Pro / OpenXR, run on a CloudXR machine): the human
                 teleoperates the G1; the recorded actions ARE the hand motion,
                 so the demo is replayable / trainable. See docs/AVP_CLOUDXR_TELEOP.md.
  * keyboard / spacemouse also work if the env exposes them.

Examples:
  # on-box proof (headless or +--livestream 2):
  ./isaaclab.sh -p g1_handover/record_handover.py --headless
  # real AVP teleop (on the CloudXR/Docker machine, AVP connected):
  ./isaaclab.sh -p g1_handover/record_handover.py --teleop_device handtracking --livestream 2
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase-gated G1 handover recording")
parser.add_argument("--task", type=str, default="Isaac-PickTurn-G1-CabinHandover-Abs-v0")
parser.add_argument("--dataset_file", type=str, default="/workspace/datasets/g1_handover.hdf5")
parser.add_argument("--pre_phase_seconds", type=float, default=4.0, help="Non-recorded TETRABot-retreat phase.")
parser.add_argument("--num_success_steps", type=int, default=10)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="scripted",
    help="scripted (on-box proof) | handtracking (Apple Vision Pro/OpenXR) | keyboard | spacemouse",
)
parser.add_argument("--lift_height", type=float, default=0.25, help="[scripted] box lift (m).")
parser.add_argument("--turn_deg", type=float, default=95.0, help="[scripted] box yaw turn (deg).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

_SCRIPTED = args_cli.teleop_device.lower() == "scripted"

# pinocchio before AppLauncher (Pink IK / G1 retargeter dep)
import pinocchio  # noqa: F401, E402

app_launcher_args = vars(args_cli)
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True  # OpenXR/CloudXR teleop needs XR mode
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import math  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import omni.log  # noqa: E402

sys.path.insert(0, "/workspace/tetrabot-sim")
import g1_handover  # noqa: F401, E402  (registers Isaac-PickTurn-G1-* tasks)

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.utils.math import quat_from_angle_axis, quat_mul  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # noqa: E402

if not _SCRIPTED:
    from isaaclab.devices.openxr import remove_camera_configs  # noqa: E402
    from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402

try:
    from isaacsim.core.prims import XFormPrim  # scripted TETRABot actor poses
except Exception:  # pragma: no cover
    XFormPrim = None


def main() -> None:
    out_dir = os.path.dirname(args_cli.dataset_file) or "."
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task

    # keep a handle to the success term; check it manually (not as a termination)
    success_term = getattr(env_cfg.terminations, "success", None)
    if success_term is not None:
        env_cfg.terminations.success = None
        if _SCRIPTED:
            # scripted proof: ignore the velocity gate (the dynamic box jitters
            # under gravity between kinematic pose writes). Real teleop keeps it.
            success_term.params["max_vel"] = 1.0e9
    else:
        omni.log.warn("No success term found — cannot mark demos successful.")
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    # XR teleop: external cameras conflict with XR rendering
    if getattr(args_cli, "xr", False):
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = out_dir
    env_cfg.recorders.dataset_filename = fname
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.sim.reset()
    env.reset()

    # G1 idle/hold action (valid IK target)
    if getattr(env_cfg, "idle_action", None) is not None:
        idle = env_cfg.idle_action.to(env.device).reshape(1, -1).repeat(env.num_envs, 1)
    else:
        idle = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)

    # teleop device (handtracking / keyboard / spacemouse) for the RECORD phase
    teleop = None
    if not _SCRIPTED:
        devices = getattr(env_cfg, "teleop_devices", None)
        if devices is None or args_cli.teleop_device not in devices.devices:
            omni.log.error(
                f"teleop_device '{args_cli.teleop_device}' not in env teleop_devices "
                f"({list(devices.devices) if devices else None})"
            )
            simulation_app.close()
            return
        teleop = create_teleop_device(args_cli.teleop_device, devices.devices, {})
        teleop.reset()
        print(f"[handover] teleop device '{args_cli.teleop_device}' ready")

    obj = env.scene["object"]
    rest_pos = obj.data.root_pos_w.clone()
    rest_quat = obj.data.root_quat_w.clone()
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=env.device).repeat(env.num_envs, 1)

    # ---- PRE_DELIVERY: TETRABots drive away; render only, NOT recorded ----
    bots = None
    if XFormPrim is not None:
        try:
            bots = XFormPrim("/World/Bot_.*")
            bot_start_pos, bot_start_quat = bots.get_world_poses()
            print(f"[handover] PRE_DELIVERY — {bot_start_pos.shape[0]} TETRABots retreating (not recorded)")
        except Exception as e:  # pragma: no cover
            omni.log.warn(f"No TETRABot actors to animate ({e}); timer pre-phase.")
            bots = None

    retreat_dist = 2.4
    t0 = time.time()
    while simulation_app.is_running():
        p = min(1.0, (time.time() - t0) / max(0.1, args_cli.pre_phase_seconds))
        if bots is not None:
            frac = 0.0 if p < 0.25 else (p - 0.25) / 0.75  # dwell, then retreat
            pos = bot_start_pos.clone()
            pos[:, 1] += retreat_dist * frac
            bots.set_world_poses(pos, bot_start_quat)
        env.sim.render()
        if p >= 1.0:
            break
    print("[handover] TRIGGER (pallet placed + bots retreated) -> recording G1 manipulation")

    # ---- RECORD phase ----
    success_count = 0
    ramp = 0.0
    exported = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            if _SCRIPTED:
                # kinematic box lift+turn drives the success term (plumbing proof)
                ramp = min(1.0, ramp + 1.0 / 90.0)
                tgt_pos = rest_pos.clone()
                tgt_pos[:, 2] += args_cli.lift_height * ramp
                angle = torch.full((env.num_envs,), math.radians(args_cli.turn_deg) * ramp, device=env.device)
                tgt_quat = quat_mul(quat_from_angle_axis(angle, z_axis), rest_quat)
                obj.write_root_pose_to_sim(torch.cat([tgt_pos, tgt_quat], dim=-1))
                obj.write_root_velocity_to_sim(torch.zeros((env.num_envs, 6), device=env.device))
                env.step(idle)
            else:
                # real teleop: the human moves the G1; its action IS recorded
                action = teleop.advance()
                if action is None:
                    env.sim.render()  # waiting for XR/teleop input
                    continue
                action = action.to(env.device)
                env.step(action.reshape(1, -1).repeat(env.num_envs, 1))

            if success_term is not None and bool(success_term.func(env, **success_term.params)[0]):
                success_count += 1
                if success_count >= args_cli.num_success_steps:
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])
                    exported += 1
                    print(f"[handover] SUCCESS — episode exported (#{exported})")
                    break
            else:
                success_count = 0

    print(f"[handover] done. exported={exported}  dataset={args_cli.dataset_file}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
