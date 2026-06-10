"""TETRABot RL-policy interface — Stufe-1 scaffold (2026-05-12).

This module is the integration seam between launch.py and a learned
locomotion policy. Today the policy is a thin "P-controller-with-policy-API
shape" stub so the integration plumbing can be exercised end-to-end without
needing trained weights.

Roadmap:
    Stufe 1 (this file): scaffold + stub policy. Demonstrates the API.
    Stufe 2: Single-bot policy trained in IsaacLab via tools/train_*.py
             (target reward: goal-distance + smoothness penalty + obstacle-
             avoidance bonus). Export as TorchScript .pt and load below.
    Stufe 3: Multi-Agent (MARL) cooperative pallet-carry. Same loader API,
             policy outputs per-bot velocity sets given joint state +
             pallet-relative pose.

Usage from launch.py:
    --controller=hand   uses the hand-coded P-controller (current default)
    --controller=rl     routes target velocities through TetraLocomotionPolicy

Public API:
    class TetraLocomotionPolicy:
        load(weights_path: str | None) -> None
        compute_action(observation: np.ndarray) -> np.ndarray
        # observation = chassis pose+velocity+goal in some agreed frame
        # action = (vx, vy, wz) target body velocity, same shape the
        # P-controller would emit
"""
from __future__ import annotations

from pathlib import Path
import logging
import numpy as np

log = logging.getLogger("tetrabot.rl")


class TetraLocomotionPolicy:
    """Stufe-1 stub: policy that mimics a P-controller's output shape but
    with a small learned-style noise floor, plus a hook for loading real
    .pt / .onnx weights once available.

    Observation vector (defined here for future trainer compatibility):
        [pos_x, pos_y, yaw, vel_x, vel_y, yaw_rate, goal_dx, goal_dy, goal_dyaw]
    Action vector:
        [v_target_x, v_target_y, v_target_yaw]  — body-velocity targets
    """

    OBS_DIM = 9
    ACT_DIM = 3

    def __init__(self, weights_path: str | None = None,
                 k_p: float = 5.0, exploration_noise: float = 0.0):
        self.weights_path = weights_path
        self.k_p = k_p
        self.exploration_noise = exploration_noise
        self._loaded = False
        self._is_stub = True
        self._inference_count = 0

    def load(self) -> bool:
        """Load weights if a path was given. Returns True if loaded a real
        policy, False if running stub (no weights or load failed)."""
        if not self.weights_path:
            log.info("TetraLocomotionPolicy: no weights path -> stub mode "
                     f"(P-controller-shaped, k_p={self.k_p})")
            self._loaded = True
            self._is_stub = True
            return False
        path = Path(self.weights_path)
        if not path.exists():
            log.warning(f"TetraLocomotionPolicy: weights {path} not found "
                        "-> falling back to stub")
            self._loaded = True
            self._is_stub = True
            return False
        # Future: torch.jit.load(str(path)) or onnxruntime.InferenceSession(...)
        log.warning(f"TetraLocomotionPolicy: real-weights load not implemented "
                    f"yet (would load {path}); using stub.")
        self._loaded = True
        self._is_stub = True
        return False

    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """Per-bot action computation.

        observation shape: (OBS_DIM,) for one bot, OR (n_bots, OBS_DIM)
        action shape: (ACT_DIM,) or (n_bots, ACT_DIM) matching input.

        Stub implementation: P-controller form
            v = k_p * goal_delta + noise
        Future: torch policy forward pass.
        """
        self._inference_count += 1
        single = (observation.ndim == 1)
        obs = observation[None, :] if single else observation
        # Indices in obs: pos(0,1,2) vel(3,4,5) goal(6,7,8)
        goal_dx   = obs[:, 6]
        goal_dy   = obs[:, 7]
        goal_dyaw = obs[:, 8]
        target_vx   = self.k_p * goal_dx
        target_vy   = self.k_p * goal_dy
        target_wyaw = self.k_p * goal_dyaw
        action = np.stack([target_vx, target_vy, target_wyaw], axis=1)
        if self.exploration_noise > 0:
            action = action + np.random.randn(*action.shape) * self.exploration_noise
        return action[0] if single else action

    def info(self) -> dict:
        return {
            "is_stub": self._is_stub,
            "weights_path": self.weights_path,
            "loaded": self._loaded,
            "inference_count": self._inference_count,
            "obs_dim": self.OBS_DIM,
            "act_dim": self.ACT_DIM,
            "k_p": self.k_p,
        }
