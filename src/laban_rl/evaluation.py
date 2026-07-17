"""
Evaluation helpers.

This file contains:
    - random search baseline
    - trained policy evaluation

It should not contain reward definitions, environment definitions, or plotting.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import robust_laban_normalisation_balanced_3gestures as laban

from .config import RewardWeights, JointLimits
from .targets import TARGET_PROFILES
from .trajectories import make_reference_trajectory
from .style_actions import apply_style_action
from .features import compute_raw_and_norm_features
from .rewards import compute_reward
from .envs import LabanTrajectoryStylerEnv


def run_random_search(
    gesture: str,
    target_name: str,
    n_trials: int,
    arm: laban.ArmConfig,
    filter_config: laban.FilterConfig,
    ranges: Dict[str, Tuple[float, float]],
    seed: int = 7,
) -> dict:
    """
    Run random search over the 6D action space for one gesture/target pair.

    This is a strong per-case baseline because the problem is one-step and
    low-dimensional.
    """
    rng = np.random.default_rng(seed)

    target_profile = TARGET_PROFILES[target_name]

    q_ref = make_reference_trajectory(
        gesture_type=gesture,
        arm=arm,
    )

    ref_raw, ref_norm = compute_raw_and_norm_features(
        q=q_ref,
        arm=arm,
        filter_config=filter_config,
        ranges=ranges,
    )

    best_result = None
    best_reward = -np.inf

    for trial in range(n_trials):
        action = rng.uniform(-1.0, 1.0, size=6)

        q_var, style_params = apply_style_action(
            q_ref=q_ref,
            action=action,
            gesture_type=gesture,
            target_name=target_name,
            preserve_endpoints=False,
        )

        var_raw, var_norm = compute_raw_and_norm_features(
            q=q_var,
            arm=arm,
            filter_config=filter_config,
            ranges=ranges,
        )

        reward, reward_info = compute_reward(
            q_ref=q_ref,
            q_var=q_var,
            features_norm=var_norm,
            target_profile=target_profile,
            gesture_type=gesture,
            reward_weights=RewardWeights(),
            joint_limits=JointLimits(),
            arm=arm,
            action=action,
            target_name=target_name,
        )

        if reward > best_reward:
            best_reward = reward

            best_result = {
                "trial": trial,
                "gesture_type": gesture,
                "target_name": target_name,
                "target_profile": target_profile,
                "reward": float(reward),
                "action": np.asarray(action, dtype=np.float64),
                "q_ref": q_ref,
                "q_var": q_var,
                "ref_raw": ref_raw,
                "ref_norm": ref_norm,
                "var_raw": var_raw,
                "var_norm": var_norm,
                "style_params": style_params,
                "reward_info": reward_info,
            }

    assert best_result is not None
    return best_result


def evaluate_policy(
    model,
    gesture: str,
    target_name: str,
    arm: laban.ArmConfig,
    filter_config: laban.FilterConfig,
    ranges: Dict[str, Tuple[float, float]],
) -> dict:
    """
    Evaluate a trained policy on one fixed gesture/target pair.

    The environment is reset with fixed options so the policy is evaluated
    on the exact case requested.
    """
    env = LabanTrajectoryStylerEnv(
        gestures=[gesture],
        targets=[target_name],
        arm=arm,
        filter_config=filter_config,
        ranges=ranges,
    )

    obs, reset_info = env.reset(
        options={
            "gesture_type": gesture,
            "target_name": target_name,
        }
    )

    action, _ = model.predict(obs, deterministic=True)

    _, reward, terminated, truncated, info = env.step(action)

    reward_info = info.get("reward_info", {})

    result = {
        "gesture_type": gesture,
        "target_name": target_name,
        "target_profile": TARGET_PROFILES[target_name],
        "reward": float(reward),
        "action": np.asarray(action, dtype=np.float64),
        "q_ref": info["q_ref"],
        "q_var": info["q_var"],
        "ref_raw": info["ref_raw"],
        "ref_norm": info["ref_norm"],
        "var_raw": info["var_raw"],
        "var_norm": info["var_norm"],
        "style_params": info.get("style_params", {}),
        "reward_info": reward_info,
        "terminated": terminated,
        "truncated": truncated,
        "reset_info": reset_info,
    }

    return result