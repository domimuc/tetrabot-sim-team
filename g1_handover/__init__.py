# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""G1 box pick-and-turn handover for Isaac Lab 2.3.

Importing this package registers two gym tasks (run via stream_hold.py /
record_handover.py here, or Isaac Lab's teleop_se3_agent.py / replay_demos.py):
  Isaac-PickTurn-G1-Box-Abs-v0           G1 + pallet + parcel (fast pipeline tests)
  Isaac-PickTurn-G1-CabinHandover-Abs-v0 full A30X cabin + pallet + TETRABots + G1
"""

import gymnasium as gym

from . import cabin_handover_env_cfg, pick_turn_env_cfg

# G1 + pallet + parcel — fast, for pipeline tests
gym.register(
    id="Isaac-PickTurn-G1-Box-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": pick_turn_env_cfg.G1BoxPickTurnEnvCfg},
    disable_env_checker=True,
)

# Full handover scene: A30X cabin + pallet + parcel + TETRABot actors + G1
gym.register(
    id="Isaac-PickTurn-G1-CabinHandover-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": cabin_handover_env_cfg.G1CabinHandoverEnvCfg},
    disable_env_checker=True,
)
