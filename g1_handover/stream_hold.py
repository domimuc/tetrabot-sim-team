"""Hold a G1 env at its idle pose while streaming over WebRTC.

For eyeballing the scene without an AVP, and as a scripted-agent base for
record/replay checks. The env expects a 38-dim retargeter action, so real teleop
needs the Apple Vision Pro handtracking device (keyboard's 7-dim Se3 won't do).

  cd /workspace/isaaclab && ./isaaclab.sh -p <repo>/g1_handover/stream_hold.py \
    --task Isaac-PickTurn-G1-CabinHandover-Abs-v0 --livestream 2
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 hold & stream harness")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G1-InspireFTP-Abs-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# pinocchio must be imported before AppLauncher so the IsaacLab build wins over
# the Isaac Sim one (Pink IK / G1 retargeter dependency).
import pinocchio  # noqa: F401, E402

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# Register our custom handover task (Isaac-PickTurn-G1-Box-Abs-v0) if available.
import sys as _sys  # noqa: E402

_sys.path.insert(0, "/workspace/tetrabot-sim")
try:
    import g1_handover  # noqa: F401
    print("[g1_stream_hold] g1_handover task registered")
except Exception as _e:  # pragma: no cover
    print(f"[g1_stream_hold] g1_handover not loaded: {_e}")


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.terminations.time_out = None  # don't auto-reset on timeout
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    env.reset()

    # Use the env's defined idle/hold action (a valid IK target) so the Pink IK
    # QP stays well-posed. Zero actions send a degenerate EEF target (origin),
    # which makes the OSQP solve non-convex and spams LDL_factor errors.
    action_dim = env.action_manager.total_action_dim
    if getattr(env_cfg, "idle_action", None) is not None:
        hold_action = env_cfg.idle_action.to(env.device).reshape(1, -1).repeat(env.num_envs, 1)
    else:
        hold_action = torch.zeros((env.num_envs, action_dim), device=env.device)
    print(f"[g1_stream_hold] env ready — action_dim={action_dim}, streaming. Ctrl-C / close to stop.")

    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(hold_action)
            count += 1
            if count % 200 == 0:
                print(f"[g1_stream_hold] alive, step {count}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
