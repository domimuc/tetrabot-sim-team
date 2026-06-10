# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Custom MDP terms for the G1 box pick-and-turn handover task."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat


def box_picked_and_turned(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    min_lift_height: float = 0.12,
    min_yaw_change_deg: float = 80.0,
    max_vel: float = 0.30,
) -> torch.Tensor:
    """Box lifted past ``min_lift_height``, yaw-rotated past ``min_yaw_change_deg``,
    and settled below ``max_vel``. All measured against the object's reset pose,
    so it's independent of env origin and spawn pose. Returns bool (num_envs,).
    """
    obj: RigidObject = env.scene[object_cfg.name]

    # --- lifted: current height minus default rest height (both env-local) ---
    cur_h = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    rest_h = obj.data.default_root_state[:, 2]  # already env-local
    lifted = (cur_h - rest_h) > min_lift_height

    # --- turned: |yaw - default_yaw| past threshold (wrapped to [-pi, pi]) ---
    cur_yaw = euler_xyz_from_quat(obj.data.root_quat_w)[2]
    rest_yaw = euler_xyz_from_quat(obj.data.default_root_state[:, 3:7])[2]
    dyaw = torch.atan2(torch.sin(cur_yaw - rest_yaw), torch.cos(cur_yaw - rest_yaw)).abs()
    turned = dyaw > torch.deg2rad(torch.tensor(min_yaw_change_deg, device=env.device))

    # --- settled: not mid-flight / being dropped ---
    settled = torch.norm(obj.data.root_vel_w, dim=-1) < max_vel

    return lifted & turned & settled
