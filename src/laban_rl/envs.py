"""
Gym/Gymnasium environment for the Laban RL trajectory styler.

This is a one-step environment.

At reset:
    - choose gesture and target
    - build reference trajectory
    - compute reference features
    - return observation

At step:
    - apply 6D style action
    - compute variant features
    - compute reward
    - terminate immediately

Observation:
    reference normalised features  : 5
    target profile                 : 5
    feature mask                   : 5
    gesture one-hot                : number of configured gestures

Total observation size:
    4 * number of Laban features + 3 reference statistics + gesture count
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

import robust_laban_normalisation_balanced_3gestures as laban

from .config import FEATURE_KEYS, GESTURE_TYPES, RewardWeights, JointLimits
from .targets import TARGET_PROFILES
from .trajectories import make_reference_trajectory
from .style_actions import apply_style_action
from .features import (
    compute_raw_and_norm_features,
    feature_dict_to_array,
    target_dict_to_array,
    get_feature_mask,
)
from .rewards import compute_reward

OBSERVATION_SIZE = 4 * len(FEATURE_KEYS) + 3 + len(GESTURE_TYPES)


class LabanTrajectoryStylerEnv(gym.Env if gym is not None else object):
    """
    One-step RL environment for trajectory styling.

    The policy outputs one 6D action:
        [amplitude, timing, curve, pause, smoothing, envelope_blend]

    The episode ends after one step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        gestures: List[str],
        targets: List[str],
        arm: laban.ArmConfig,
        filter_config: laban.FilterConfig,
        ranges: Dict[str, Tuple[float, float]],
        reward_weights: RewardWeights | None = None,
        joint_limits: JointLimits | None = None,
    ):
        if gym is None or spaces is None:
            raise ImportError(
                "gymnasium is required for LabanTrajectoryStylerEnv. "
                "Install with: pip install gymnasium"
            )

        super().__init__()

        self.gestures = gestures
        self.targets = targets
        self.arm = arm
        self.filter_config = filter_config
        self.ranges = ranges

        self.reward_weights = reward_weights or RewardWeights()
        self.joint_limits = joint_limits or JointLimits()

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.gesture_type: str | None = None
        self.target_name: str | None = None
        self.target_profile: Dict[str, float] | None = None

        self.q_ref: np.ndarray | None = None
        self.ref_raw: Dict[str, float] | None = None
        self.ref_norm: Dict[str, float] | None = None

    def _make_gesture_one_hot(self, gesture_type: str) -> np.ndarray:
        one_hot = np.zeros(len(GESTURE_TYPES), dtype=np.float32)
        idx = GESTURE_TYPES.index(gesture_type)
        one_hot[idx] = 1.0
        return one_hot

    def _compute_reference_stats(self) -> np.ndarray:
        assert self.q_ref is not None

        _, _, wrist = laban.forward_kinematics_2link(
            self.q_ref,
            l1=self.arm.l1,
            l2=self.arm.l2,
        )

        displacement = np.linalg.norm(wrist[-1] - wrist[0])
        path_length = float(np.sum(np.linalg.norm(np.diff(wrist, axis=0), axis=1)))
        straightness = float(displacement / (path_length + 1e-8))

        return np.array([displacement, path_length, straightness], dtype=np.float32)

    def _make_observation(self) -> np.ndarray:
        assert self.ref_norm is not None
        assert self.target_profile is not None
        assert self.gesture_type is not None

        ref_arr = feature_dict_to_array(self.ref_norm).astype(np.float32)
        target_arr = target_dict_to_array(self.target_profile).astype(np.float32)
        delta_arr = (target_arr - ref_arr).astype(np.float32)
        mask_arr = get_feature_mask(self.gesture_type, self.ref_norm).astype(np.float32)
        gesture_arr = self._make_gesture_one_hot(self.gesture_type).astype(np.float32)
        stats_arr = self._compute_reference_stats().astype(np.float32)

        obs = np.concatenate(
            [
                ref_arr,
                target_arr,
                delta_arr,
                mask_arr,
                stats_arr,
                gesture_arr,
            ],
            axis=0,
        )

        return obs.astype(np.float32)

    def reset(
        self,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)

        options = options or {}

        if "gesture_type" in options:
            self.gesture_type = options["gesture_type"]
        else:
            self.gesture_type = str(self.np_random.choice(self.gestures))

        if "target_name" in options:
            self.target_name = options["target_name"]
        else:
            self.target_name = str(self.np_random.choice(self.targets))

        self.target_profile = TARGET_PROFILES[self.target_name]

        self.q_ref = make_reference_trajectory(
            gesture_type=self.gesture_type,
            arm=self.arm,
        )

        self.ref_raw, self.ref_norm = compute_raw_and_norm_features(
            q=self.q_ref,
            arm=self.arm,
            filter_config=self.filter_config,
            ranges=self.ranges,
        )

        obs = self._make_observation()

        info = {
            "gesture_type": self.gesture_type,
            "target_name": self.target_name,
            "target_profile": self.target_profile,
            "ref_raw": self.ref_raw,
            "ref_norm": self.ref_norm,
        }

        return obs, info

    def step(self, action: np.ndarray):
        assert self.q_ref is not None
        assert self.gesture_type is not None
        assert self.target_name is not None
        assert self.target_profile is not None

        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        q_var, style_params = apply_style_action(
            q_ref=self.q_ref,
            action=action,
            gesture_type=self.gesture_type,
            target_name=self.target_name,
            preserve_endpoints=False,
        )

        var_raw, var_norm = compute_raw_and_norm_features(
            q=q_var,
            arm=self.arm,
            filter_config=self.filter_config,
            ranges=self.ranges,
        )

        reward, reward_info = compute_reward(
            q_ref=self.q_ref,
            q_var=q_var,
            features_norm=var_norm,
            target_profile=self.target_profile,
            gesture_type=self.gesture_type,
            reward_weights=self.reward_weights,
            joint_limits=self.joint_limits,
            arm=self.arm,
            action=action,
            target_name=self.target_name,
        )

        obs = self._make_observation()

        terminated = True
        truncated = False

        info = {
            "gesture_type": self.gesture_type,
            "target_name": self.target_name,
            "target_profile": self.target_profile,
            "q_ref": self.q_ref,
            "q_var": q_var,
            "ref_raw": self.ref_raw,
            "ref_norm": self.ref_norm,
            "var_raw": var_raw,
            "var_norm": var_norm,
            "style_params": style_params,
            "reward_info": reward_info,
        }

        return obs, float(reward), terminated, truncated, info