"""TETRABot Stage 1 Watch Demo.

USAGE (Isaac Sim 5.1):
  1. File -> Open: c:/dev/tetrabot_sim/assets/source/tetrabot.usd
     (oder importiere die URDF frisch falls noch nicht geschehen)
  2. Add a Ground Plane: Create -> Physics -> Ground Plane (sonst fällt der Robot)
  3. Window -> Script Editor
  4. Click "+" tab, paste this entire file, click "Run"
  5. Press Play (Spacebar oder Toolbar)

The script drives base_x, base_yaw and lift sinusoidally. The robot oscillates
forward/back, turns left/right, and the lift goes up and down — even though
the lift movement is invisible (the mecanum drive geometry sits on chassis_link
in our MVP setup).

If you see a TypeError about the Articulation API, your Isaac Sim version may
expose it under a different module — see the import block below for variants.
"""

import math

import numpy as np


def _make_target_array(dof_names: list[str], t: float) -> np.ndarray:
    """Build a full-DOF position-target array, animating only what we want."""
    dof_idx = {name: i for i, name in enumerate(dof_names)}
    targets = np.zeros(len(dof_names), dtype=np.float32)

    if "base_x_joint" in dof_idx:
        targets[dof_idx["base_x_joint"]] = 0.5 * math.sin(0.5 * t)
    if "base_yaw_joint" in dof_idx:
        targets[dof_idx["base_yaw_joint"]] = 0.6 * math.sin(0.3 * t)
    if "lift_joint" in dof_idx:
        targets[dof_idx["lift_joint"]] = 0.05 + 0.05 * math.sin(0.6 * t)
    return targets


def _drive_demo():
    # Try the modern (5.1) import path first, fall back to legacy if needed
    try:
        from isaacsim.core.prims import Articulation
        from isaacsim.core.api.controllers.articulation_controller import (
            ArticulationController,
        )
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core.articulations import Articulation  # type: ignore
        from omni.isaac.core.controllers.articulation_controller import (  # type: ignore
            ArticulationController,
        )
        from omni.isaac.core.utils.types import ArticulationAction  # type: ignore

    import omni.physx as physx

    robot = Articulation(prim_paths_expr="/World/tetrabot", name="tetrabot")
    controller = ArticulationController()
    controller.initialize(robot)

    dof_names = list(robot.dof_names) if robot.dof_names else []
    print(f"[tetrabot demo] DOF names: {dof_names}")

    state = {"t": 0.0}

    def on_step(dt: float):
        state["t"] += dt
        targets = _make_target_array(dof_names, state["t"])
        controller.apply_action(ArticulationAction(joint_positions=targets))

    sub = physx.get_physx_interface().subscribe_physics_step_events(on_step)
    print("[tetrabot demo] subscribed — press Play if not already running.")
    return sub  # keep alive


_demo_sub = _drive_demo()
