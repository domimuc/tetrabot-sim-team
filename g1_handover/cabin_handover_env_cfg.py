# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""Full handover scene: the box pick-and-turn env inside the real A30X cabin
with four TETRABot delivery actors.

The cabin (CATIA shell, mm -> scaled 0.001) and the TETRABots are static visual
props; the grasp target stays a primitive parcel (the tetrabot box USDs carry no
RigidBodyAPI). Layout is relative to the G1, which the teleop env fixes at the
origin facing +Y. The constants below were tuned by eye against the stream.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg import (
    ObjectTableSceneCfg,
)

from .pick_turn_env_cfg import G1BoxPickTurnEnvCfg

# --- tunable layout (env-local; G1 sits at origin facing +Y) --------------
_ASSET_DIR = "/workspace/tetrabot-sim/assets/environment"
_BOT_USD = "/workspace/tetrabot-sim/urdf/tetrabot/tetrabot.usd"
_FLOOR_Z = 0.79                      # tetrabot GROUND_PLANE_Z / cabin floor
_CABIN_POS = (0.0, -2.40, 0.36)  # cabin floor lands ~z=0.21 (G1 feet); tune by eye
_CABIN_ROT = (0.70711, 0.0, 0.0, 0.70711)  # +90deg about z: cabin long-axis -> +Y
_BOX_B_POS = (0.28, 0.66, _FLOOR_Z + 0.16)  # 2nd (real) package on the pallet

# 4 TETRABots, initial "just-delivered" row on the +Y side of the pallet.
# Visual/kinematic actors: record_handover.py animates their poses (deliver ->
# set down -> drive away) via XFormPrim. Positions are a first-pass guess.
_BOT_Z = 0.42
_BOT_POSITIONS = [
    (-0.60, 2.10, _BOT_Z),
    (-0.20, 2.10, _BOT_Z),
    (0.20, 2.10, _BOT_Z),
    (0.60, 2.10, _BOT_Z),
]
_BOT_ROT = (1.0, 0.0, 0.0, 0.0)

# TETRABot proxy: visible cuboid (~base footprint). The URDF-imported tetrabot.usd
# has a broken internal visual reference (visuals point at the empty
# tetrabot_physics.usd) so it loads invisible; a proxy is reliable + visible for
# the delivery choreography. Real mesh = flatten/re-export the URDF USD ("wenn nötig").
_BOT_SPAWN = sim_utils.CuboidCfg(
    size=(0.70, 0.70, 0.42),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.22, 0.35), metallic=0.4),
)


@configclass
class CabinHandoverSceneCfg(ObjectTableSceneCfg):
    """Stock G1 scene + the real A30X cabin shell (global prim, like ground/light).

    The cabin USD has a valid defaultPrim and references cleanly. The tetrabot
    box meshes (pallet_box_*.usd) are bare, physics-less and tripped Isaac Lab's
    AddReference validity check, so a second decorative package is added as a
    primitive in the env __post_init__ instead of referencing box_b.
    """

    # cabin_wrapped.usd = thin Xform wrapper around A30X_AllCATPart.usd; Isaac
    # Lab's AddReference command rejects the raw CATIA export but accepts the
    # wrapper (clean single defaultPrim). Created by tools/make_cabin_wrapper.py.
    cabin = AssetBaseCfg(
        prim_path="/World/Cabin",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_CABIN_POS, rot=_CABIN_ROT),
        # A30X CATIA export is in MILLIMETRES -> scale 0.001 (matches launch.py
        # CABIN_SCALE); without this the cabin renders ~1000x too big / off-screen.
        spawn=sim_utils.UsdFileCfg(usd_path=f"{_ASSET_DIR}/cabin_wrapped.usd", scale=(0.001, 0.001, 0.001)),
    )

    # 4 TETRABots (visual/kinematic actors; record_handover.py animates them).
    bot_0 = AssetBaseCfg(
        prim_path="/World/Bot_0",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_BOT_POSITIONS[0], rot=_BOT_ROT),
        spawn=_BOT_SPAWN,
    )
    bot_1 = AssetBaseCfg(
        prim_path="/World/Bot_1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_BOT_POSITIONS[1], rot=_BOT_ROT),
        spawn=_BOT_SPAWN,
    )
    bot_2 = AssetBaseCfg(
        prim_path="/World/Bot_2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_BOT_POSITIONS[2], rot=_BOT_ROT),
        spawn=_BOT_SPAWN,
    )
    bot_3 = AssetBaseCfg(
        prim_path="/World/Bot_3",
        init_state=AssetBaseCfg.InitialStateCfg(pos=_BOT_POSITIONS[3], rot=_BOT_ROT),
        spawn=_BOT_SPAWN,
    )

    # G1 head/ego camera. The G1 29dof has no dedicated head link, so it is
    # mounted on `torso_link` at ~head height, looking forward. Renders only with
    # --enable_cameras. Offset (pos/rot) is a first pass — tune the POV by eye.
    head_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/head_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.10, 0.0, 0.45), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


@configclass
class G1CabinHandoverEnvCfg(G1BoxPickTurnEnvCfg):
    """G1 picks a package off the delivered pallet inside the A30X cabin."""

    scene: CabinHandoverSceneCfg = CabinHandoverSceneCfg(
        num_envs=1, env_spacing=2.5, replicate_physics=True
    )

    # pallet sits on the cabin floor (~z=0.21), not floating
    pallet_z: float = 0.21

    def __post_init__(self):
        # Parent sets the Euro-pallet (packing_table), the primitive grasp parcel
        # (object) and the "lifted + turned" success term.
        super().__post_init__()
        # The head camera forces --enable_cameras, which breaks the normal
        # view/record runs. Keep it OFF by default; enable the ego POV with
        #   G1_HEAD_CAM=1 ... --enable_cameras
        if not os.environ.get("G1_HEAD_CAM"):
            self.scene.head_cam = None
