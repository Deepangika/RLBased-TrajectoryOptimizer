"""
Reward calculation for the Laban RL trajectory styler.

This file is where reward tuning should happen.

The current reward includes:
    1. Laban style matching
    2. trajectory preservation
    3. soft endpoint preservation
    4. joint-space smoothness
    5. joint-limit feasibility
    6. action regularisation
    7. action saturation penalty
    8. target-dependent action preference penalty

The style error currently uses:
    - feature-specific weights
    - reduced weight for space_indirectness
    - stronger overshoot penalty for Weight and Time
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import robust_laban_normalisation_balanced_3gestures as laban

from .config import (
    FEATURE_KEYS,
    FEATURE_REWARD_WEIGHTS,
    RewardWeights,
    JointLimits,
)
from .features import (
    feature_dict_to_array,
    target_dict_to_array,
    get_feature_mask,
)


def get_feature_reward_weights() -> np.ndarray:
    """Return feature reward weights in FEATURE_KEYS order."""
    return np.array(
        [FEATURE_REWARD_WEIGHTS[key] for key in FEATURE_KEYS],
        dtype=np.float32,
    )


def compute_joint_limit_penalty(
    q: np.ndarray,
    limits: JointLimits,
) -> float:
    """Penalty for violating simple shoulder/elbow joint limits."""
    q = np.asarray(q, dtype=np.float64)

    shoulder = q[:, 0]
    elbow = q[:, 1]

    violations = [
        np.maximum(limits.shoulder_min - shoulder, 0.0),
        np.maximum(shoulder - limits.shoulder_max, 0.0),
        np.maximum(limits.elbow_min - elbow, 0.0),
        np.maximum(elbow - limits.elbow_max, 0.0),
    ]

    total = sum(np.mean(v ** 2) for v in violations)
    return float(total)


def compute_joint_jerk_penalty(
    q: np.ndarray,
    dt: float,
) -> float:
    """Small smoothness penalty based on joint-space jerk."""
    q = np.asarray(q, dtype=np.float64)

    v = np.gradient(q, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)

    return float(np.mean(np.linalg.norm(j, axis=1)))


def compute_endpoint_error(
    q_ref: np.ndarray,
    q_var: np.ndarray,
) -> float:
    """
    Soft endpoint error.

    We preserve the start pose exactly in style_actions.py.
    The end pose is allowed to vary, but large drift is discouraged here.

    This is currently joint-space endpoint error.
    Later, this could be supplemented with wrist-space endpoint error.
    """
    q_ref = np.asarray(q_ref, dtype=np.float64)
    q_var = np.asarray(q_var, dtype=np.float64)

    start_error = np.linalg.norm(q_var[0] - q_ref[0])
    end_error = np.linalg.norm(q_var[-1] - q_ref[-1])

    return float(start_error**2 + end_error**2)


def compute_style_error(
    current: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute weighted asymmetric Laban style error.

    Plain absolute error treats undershoot and overshoot equally:
        target=0.75, current=0.50 -> error=0.25
        target=0.75, current=1.00 -> error=0.25

    In our PPO experiments, Weight and Time often overshot to 1.0.
    So overshooting Weight and Time is penalised more strongly.

    Space is weighted lower through FEATURE_REWARD_WEIGHTS because the
    current styler cannot easily increase space_indirectness for reach/point.
    """
    feature_weights = get_feature_reward_weights()
    weighted_mask = mask * feature_weights

    under_error = np.maximum(target - current, 0.0)
    over_error = np.maximum(current - target, 0.0)

    under_weights = np.array(
        [1.0, 1.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )

    # Penalise overshooting Weight and Time more strongly.
    over_weights = np.array(
        [2.0, 2.0, 1.0, 1.0, 1.0],
        dtype=np.float32,
    )

    asymmetric_error = (
        under_weights * under_error
        + over_weights * over_error
    )

    style_error = float(
        np.sum(asymmetric_error * weighted_mask)
        / (np.sum(weighted_mask) + 1e-8)
    )

    return style_error


def compute_action_penalties(
    action: np.ndarray | None,
) -> Tuple[float, float]:
    """
    Compute action magnitude and saturation penalties.

    action_penalty:
        penalises overall action magnitude using mean(action^2)

    saturation_penalty:
        penalises values close to the action boundaries.
        Nothing is penalised until abs(action) > 0.85.
    """
    if action is None:
        return 0.0, 0.0

    action = np.asarray(action, dtype=np.float64)

    action_penalty = float(np.mean(action ** 2))

    saturation = np.maximum(np.abs(action) - 0.85, 0.0)
    saturation_penalty = float(np.mean(saturation ** 2))

    return action_penalty, saturation_penalty


def compute_target_action_preference_penalty(
    action: np.ndarray | None,
    target_name: str | None,
) -> float:
    """
    Soft penalty for action directions that conflict with target style intuition.

    For confident motion, we usually expect amplitude to increase or be neutral,
    not strongly shrink.

    This is not a hard constraint. It simply makes semantically odd shortcuts
    less attractive to the optimiser.
    """
    if action is None or target_name is None:
        return 0.0

    action = np.asarray(action, dtype=np.float64)
    target_name = target_name.strip().lower()

    amp = float(action[0])

    penalty = 0.0

    if target_name == "confident":
        # Penalise negative amplitude for confident motion.
        penalty += max(-amp, 0.0) ** 2

    return float(penalty)


def compute_reward(
    q_ref: np.ndarray,
    q_var: np.ndarray,
    features_norm: Dict[str, float],
    target_profile: Dict[str, float],
    gesture_type: str,
    reward_weights: RewardWeights,
    joint_limits: JointLimits,
    arm: laban.ArmConfig,
    action: np.ndarray | None = None,
    target_name: str | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute total reward and reward diagnostics."""
    current = feature_dict_to_array(features_norm)
    target = target_dict_to_array(target_profile)
    mask = get_feature_mask(gesture_type, features_norm)

    style_error = compute_style_error(
        current=current,
        target=target,
        mask=mask,
    )
    style_reward = -style_error

    preservation_error = float(np.mean((q_var - q_ref) ** 2))
    preservation_reward = -preservation_error

    endpoint_error = compute_endpoint_error(q_ref, q_var)
    endpoint_reward = -endpoint_error

    jerk_penalty = compute_joint_jerk_penalty(q_var, arm.dt)
    smoothness_reward = -0.001 * jerk_penalty

    limit_penalty = compute_joint_limit_penalty(q_var, joint_limits)
    limit_reward = -limit_penalty

    action_penalty, saturation_penalty = compute_action_penalties(action)
    action_reward = -action_penalty
    saturation_reward = -saturation_penalty

    target_action_pref_penalty = compute_target_action_preference_penalty(
        action=action,
        target_name=target_name,
    )
    target_action_pref_reward = -target_action_pref_penalty

    total = (
        reward_weights.style * style_reward
        + reward_weights.preserve * preservation_reward
        + reward_weights.endpoint * endpoint_reward
        + reward_weights.smoothness * smoothness_reward
        + reward_weights.joint_limits * limit_reward
        + reward_weights.action * action_reward
        + reward_weights.saturation * saturation_reward
        + reward_weights.target_action_pref * target_action_pref_reward
    )

    diagnostics = {
        "total_reward": float(total),
        "style_reward": float(style_reward),
        "style_error": float(style_error),
        "preservation_reward": float(preservation_reward),
        "preservation_error": float(preservation_error),
        "endpoint_reward": float(endpoint_reward),
        "endpoint_error": float(endpoint_error),
        "smoothness_reward": float(smoothness_reward),
        "joint_limit_reward": float(limit_reward),
        "valid_feature_count": float(np.sum(mask)),
        "action_reward": float(action_reward),
        "action_penalty": float(action_penalty),
        "saturation_reward": float(saturation_reward),
        "saturation_penalty": float(saturation_penalty),
        "target_action_pref_reward": float(target_action_pref_reward),
        "target_action_pref_penalty": float(target_action_pref_penalty),
        "target_delta_weight": float(np.abs(target[0] - current[0])),
        "target_delta_time": float(np.abs(target[1] - current[1])),
        "target_delta_flow_boundness": float(np.abs(target[2] - current[2])),
        "target_delta_space_indirectness": float(np.abs(target[3] - current[3])),
        "target_delta_shape_arcness": float(np.abs(target[4] - current[4])),
    }

    return float(total), diagnostics