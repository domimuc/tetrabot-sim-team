"""TETRABot single-bot locomotion RL training scaffold (Stufe 1).

This is the SCAFFOLD for the eventual training run that produces the
weights consumed by tools/rl_policy.py. Today it documents the planned
architecture without actually training (which requires IsaacLab + RSL-RL
+ a few hours of GPU time).

To actually train (future iteration):
    1. Install IsaacLab (separate from Isaac Sim base) per
       https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
    2. Implement TetrabotLocomotionEnvCfg below as an IsaacLab
       ManagerBasedRLEnvCfg
    3. Run: ./isaaclab.sh -p tools/train_g1_locomotion.py --num_envs 4096 --headless
    4. Export trained policy:
           torch.jit.save(policy, "models/tetrabot_locomotion_v1.pt")
    5. Use in launch.py:
           ./launch.bat --controller=rl --rl-weights models/tetrabot_locomotion_v1.pt

Planned reward components:
    +1.0 * dot(velocity, goal_direction)        # progress toward goal
    -0.1 * |action - prev_action|               # smoothness
    -0.5 * |angular_velocity| if no yaw goal    # don't spin without reason
    +5.0 once when goal_distance < 0.05         # success bonus
    -1.0 per collision with obstacle             # safety
    -0.001 per timestep                          # encourage speed

Planned observation:
    9-dim per bot:
        chassis position (x, y), yaw,
        chassis linear velocity (vx, vy), yaw rate,
        goal-relative position (dx, dy), goal yaw delta

Planned action:
    3-dim per bot: target body velocity (vx_target, vy_target, wyaw_target)

Network: small MLP [256, 128, 64] -> tanh activation, RSL-RL PPO config.
Train time estimate: 4096 envs × 2000 epochs × ~10 ms/step = ~10 GPU-min on
RTX 5060 Ti.
"""
from __future__ import annotations

import sys


def main() -> int:
    print("[TETRABot RL training scaffold]")
    print()
    print("This is a SCAFFOLD — actual training is not yet implemented.")
    print()
    print("To make this work, the following pieces need to be added:")
    print("  1. IsaacLab installation + ManagerBasedRLEnvCfg subclass")
    print("  2. TETRABot URDF -> IsaacLab ArticulationCfg")
    print("  3. Reward terms (see module docstring)")
    print("  4. PPO trainer integration via rsl_rl")
    print("  5. Checkpoint export to TorchScript .pt")
    print()
    print("Estimated effort: 1-2 weeks. Estimated train time: ~10 GPU-min.")
    print()
    print("Until then:")
    print("  * launch.py uses the hand-coded P-controller (default)")
    print("  * `--controller=rl` routes through tools/rl_policy.py STUB")
    print("    which mimics the trained-policy API but actually does")
    print("    P-control internally — same behaviour as default, but")
    print("    exercises the inference plumbing end-to-end.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
