"""G1 right-arm inverse kinematics + forward kinematics helpers.

Phase B (2026-05-11) — scripted-keyframes implementation. Pinocchio/Pink
couldn't be installed on Windows (no pre-built wheel, source build needs a
VS toolchain). ikpy is a pure-Python alternative that works the same way:
takes a URDF + joint angles, computes IK and FK.

Public API:
    build_right_arm_chain(urdf_path) -> Chain
    solve_ik(chain, target_pelvis, initial=None) -> np.ndarray
    forward_kinematics_all_links(chain, joint_angles) -> dict[str, np.ndarray]
        Returns each link's 4x4 transform in the pelvis (chain base) frame.

The chain walks pelvis -> waist (3 joints) -> right shoulder (3 joints) ->
right elbow -> right wrist (3 joints), stopping at right_wrist_yaw_link
(no hand fingers — they're decorative in our kinematic-carry demo).
"""
from __future__ import annotations

import numpy as np
from ikpy.chain import Chain


# Chain definition: alternating link/joint names from pelvis to wrist.
# Matches Unitree's g1_body29_hand14.urdf joint+link names.
RIGHT_ARM_CHAIN_PATH = [
    "pelvis",
    "waist_yaw_joint",            "waist_yaw_link",
    "waist_roll_joint",           "waist_roll_link",
    "waist_pitch_joint",          "torso_link",
    "right_shoulder_pitch_joint", "right_shoulder_pitch_link",
    "right_shoulder_roll_joint",  "right_shoulder_roll_link",
    "right_shoulder_yaw_joint",   "right_shoulder_yaw_link",
    "right_elbow_joint",          "right_elbow_link",
    "right_wrist_roll_joint",     "right_wrist_roll_link",
    "right_wrist_pitch_joint",    "right_wrist_pitch_link",
    "right_wrist_yaw_joint",      "right_wrist_yaw_link",
]

# Arm link names that the FK update writes USD xforms to. Convention:
# ikpy names each Link by the JOINT that produces it, so 'right_shoulder_
# pitch_joint' in the FK output corresponds to the URDF LINK named
# 'right_shoulder_pitch_link'. We store both names so the caller can map
# between FK output and USD prim paths.
ARM_LINK_NAMES = [
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]

# Map: ikpy chain.link.name (joint name) -> URDF link name
# Used to look up FK output by chain index, then write USD prim at link name.
IKPY_TO_URDF_LINK = {
    "right_shoulder_pitch_joint": "right_shoulder_pitch_link",
    "right_shoulder_roll_joint":  "right_shoulder_roll_link",
    "right_shoulder_yaw_joint":   "right_shoulder_yaw_link",
    "right_elbow_joint":          "right_elbow_link",
    "right_wrist_roll_joint":     "right_wrist_roll_link",
    "right_wrist_pitch_joint":    "right_wrist_pitch_link",
    "right_wrist_yaw_joint":      "right_wrist_yaw_link",
}

# Index of the end-effector in the chain (right_wrist_yaw_joint), used to
# pull the hand pose for kinematic-carry of the grasped box.
HAND_LINK_NAME_IKPY = "right_wrist_yaw_joint"

# --- Left arm chain (2026-05-12, animation-mirror to right arm) ---
LEFT_ARM_CHAIN_PATH = [
    "pelvis",
    "waist_yaw_joint",           "waist_yaw_link",
    "waist_roll_joint",          "waist_roll_link",
    "waist_pitch_joint",         "torso_link",
    "left_shoulder_pitch_joint", "left_shoulder_pitch_link",
    "left_shoulder_roll_joint",  "left_shoulder_roll_link",
    "left_shoulder_yaw_joint",   "left_shoulder_yaw_link",
    "left_elbow_joint",          "left_elbow_link",
    "left_wrist_roll_joint",     "left_wrist_roll_link",
    "left_wrist_pitch_joint",    "left_wrist_pitch_link",
    "left_wrist_yaw_joint",      "left_wrist_yaw_link",
]

LEFT_ARM_LINK_NAMES = [
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
]

LEFT_IKPY_TO_URDF_LINK = {
    "left_shoulder_pitch_joint": "left_shoulder_pitch_link",
    "left_shoulder_roll_joint":  "left_shoulder_roll_link",
    "left_shoulder_yaw_joint":   "left_shoulder_yaw_link",
    "left_elbow_joint":          "left_elbow_link",
    "left_wrist_roll_joint":     "left_wrist_roll_link",
    "left_wrist_pitch_joint":    "left_wrist_pitch_link",
    "left_wrist_yaw_joint":      "left_wrist_yaw_link",
}


def build_left_arm_chain(urdf_path: str) -> Chain:
    """Build the left-arm IK chain (analogous to right-arm). Same
    last_link_vector offset for palm centre."""
    return Chain.from_urdf_file(
        urdf_path,
        base_elements=LEFT_ARM_CHAIN_PATH,
        name="g1_left_arm",
        last_link_vector=[0.0, +0.05, 0.0],  # mirror sign vs right arm
    )


def build_right_arm_chain(urdf_path: str) -> Chain:
    """Build the right-arm IK chain from the Unitree G1 URDF.

    `last_link_vector=[0, -0.05, 0]` adds a virtual 5cm tip offset from
    the wrist_yaw frame to the palm centre. Without this argument ikpy's
    IK solver doesn't converge for our chain (it auto-extends into the
    hand-finger sub-chain but reports zero-solution); with it, sub-mm
    convergence is reliable. Value 0.05 = approximate palm half-length
    based on the URDF visual offsets.
    """
    chain = Chain.from_urdf_file(
        urdf_path,
        base_elements=RIGHT_ARM_CHAIN_PATH,
        name="g1_right_arm",
        last_link_vector=[0.0, -0.05, 0.0],
    )
    return chain


def solve_ik(
    chain: Chain,
    target_pelvis_xyz: np.ndarray,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Solve IK for the right-hand end-effector at the given pelvis-frame
    target position. Returns the full joint vector (one entry per chain link,
    including the fixed base link as 0)."""
    target = np.asarray(target_pelvis_xyz, dtype=np.float64)
    if initial is None:
        return chain.inverse_kinematics(target)
    return chain.inverse_kinematics(target, initial_position=initial)


def forward_kinematics_all_links(
    chain: Chain,
    joint_angles: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute per-link 4x4 transforms in the pelvis (chain base) frame.

    Uses ikpy.chain.Chain.forward_kinematics(full_kinematics=True) which
    returns a list of 4x4 transforms (one per link, accumulated from base
    to tip). Maps to link names for lookup.
    """
    fk_list = chain.forward_kinematics(joint_angles, full_kinematics=True)
    transforms: dict[str, np.ndarray] = {}
    for link, T in zip(chain.links, fk_list):
        if link.name:
            transforms[link.name] = np.asarray(T, dtype=np.float64).copy()
    return transforms


def interp_joint_angles(
    a: np.ndarray, b: np.ndarray, alpha: float
) -> np.ndarray:
    """Smoothed interpolation between two joint configurations via
    smoothstep curve: t' = t² * (3 - 2t). This makes joint motion
    accelerate gently from rest and decelerate before the target,
    instead of the harsh linear motion that made the arm "fly" at
    constant speed. Applies to both right + left arm chains."""
    alpha = max(0.0, min(1.0, alpha))
    eased = alpha * alpha * (3.0 - 2.0 * alpha)
    return a + (b - a) * eased
