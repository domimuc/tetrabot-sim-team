# Copyright (c) 2026. SPDX-License-Identifier: BSD-3-Clause
"""G1 box pick-and-turn handover.

Built on Isaac Lab's stock G1 Inspire-hand pick-place env: swaps the table
object for a graspable box (a Euro-pallet isn't humanoid-liftable) and the
place-success for a lift+turn success (`mdp_terms.box_picked_and_turned`). The
Inspire hand, Pink upper-body IK and the handtracking/manusvive (Apple Vision
Pro) teleop devices are inherited unchanged.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_unitree_g1_inspire_hand_env_cfg import (
    PickPlaceG1InspireFTPEnvCfg,
)

from . import mdp_terms


@configclass
class G1BoxPickTurnEnvCfg(PickPlaceG1InspireFTPEnvCfg):
    """G1 picks up a handheld box and turns with it (Apple Vision Pro teleop)."""

    # tetrabot-sim assets (Euro-pallet + delivered package boxes)
    asset_dir: str = os.environ.get(
        "TETRABOT_ASSET_DIR", "/workspace/tetrabot-sim/assets/environment"
    )
    # Height of the pallet's underside. Default 0.0 = on the ground (box env).
    # The cabin env overrides this to sit on the cabin floor (~0.21).
    pallet_z: float = 0.0

    def __post_init__(self):
        # Run the parent setup first (URDF conversion, Pink IK paths, teleop devices).
        super().__post_init__()

        # --- replace the stock packing table with the delivered Euro-pallet ---
        # Static visual prop: the euro_pallet.usd already carries convex-
        # decomposition collision, so it stands and supports the package without
        # a rigid body (the TETRABots "set it down" — it must not move/fall).
        self.scene.packing_table = AssetBaseCfg(
            prim_path="/World/envs/env_.*/Pallet",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.70, self.pallet_z], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{self.asset_dir}/euro_pallet.usd",
                # ensure the (static) pallet is a collider so the package rests on
                # it instead of falling through
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            ),
        )

        # --- pickable package resting on the pallet top ---
        # The tetrabot box USDs (pallet_box_*.usd) are bare meshes with no
        # RigidBodyAPI, so we use a primitive parcel as the reliable grasp target
        # (cardboard-coloured, ~26 cm, 0.5 kg). Swap to a rigged box USD later.
        box_top = self.pallet_z + 0.144  # pallet support surface
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.0, 0.55, box_top + 0.13], rot=[1.0, 0.0, 0.0, 0.0]
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.26, 0.26, 0.24),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=1.0
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.45, 0.27)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.2, dynamic_friction=1.1
                ),
            ),
        )

        # --- success: box lifted off the pallet AND turned past ~80 deg ---
        self.terminations.success = DoneTerm(
            func=mdp_terms.box_picked_and_turned,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "min_lift_height": 0.12,
                "min_yaw_change_deg": 80.0,
                "max_vel": 0.30,
            },
        )

        # Leave the viewport camera unlocked (freely navigable: RMB+WASD, F).

        # XR anchor: the value Isaac Lab's fixed-base / loco-manip G1 envs use.
        self.xr.anchor_pos = (0.0, 0.0, -0.45)
        self.xr.anchor_rot = (1.0, 0.0, 0.0, 0.0)
