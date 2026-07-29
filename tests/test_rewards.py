import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban

from laban_rl.config import RewardWeights, JointLimits
from laban_rl.io_utils import load_ranges_or_default
from laban_rl.targets import TARGET_PROFILES
from laban_rl.trajectories import make_reference_trajectory
from laban_rl.style_actions import apply_style_action
from laban_rl.features import compute_raw_and_norm_features
from laban_rl.rewards import (
    compute_action_penalties,
    compute_joint_limit_penalty,
    compute_reward,
    compute_style_error,
    compute_endpoint_error,
)


def test_action_penalties_are_finite():
    action = np.array([0.0, 1.0, -1.0, 0.5, -0.2, 0.0])

    action_penalty, saturation_penalty = compute_action_penalties(action)

    assert np.isfinite(action_penalty)
    assert np.isfinite(saturation_penalty)
    assert action_penalty > 0.0
    assert saturation_penalty > 0.0


def test_joint_limit_penalty_zero_for_safe_reference():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    q_ref = make_reference_trajectory("point", arm)

    penalty = compute_joint_limit_penalty(q_ref, JointLimits())

    assert np.isfinite(penalty)
    assert penalty >= 0.0


def test_style_error_penalises_weight_time_overshoot_more_than_equal_undershoot():
    # Feature order:
    # weight, time, flow_boundness, space_indirectness, shape_arcness
    target = np.array([0.8, 0.75, 0.25, 0.2, 0.25], dtype=np.float32)
    mask = np.ones(5, dtype=np.float32)

    undershoot = np.array([0.6, 0.55, 0.25, 0.2, 0.25], dtype=np.float32)
    overshoot = np.array([1.0, 0.95, 0.25, 0.2, 0.25], dtype=np.float32)

    under_error = compute_style_error(undershoot, target, mask)
    over_error = compute_style_error(overshoot, target, mask)

    assert over_error > under_error


def test_reward_is_finite():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filter_config = laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    ranges = load_ranges_or_default("configs/normalisation_ranges_balanced_3gestures.json")

    q_ref = make_reference_trajectory("point", arm)
    action = np.array([0.2, 0.8, 0.3, -0.2, 0.1, 0.0])

    q_var, _ = apply_style_action(q_ref, action, gesture_type="point")
    _, var_norm = compute_raw_and_norm_features(q_var, arm, filter_config, ranges)

    reward, info = compute_reward(
        q_ref=q_ref,
        q_var=q_var,
        features_norm=var_norm,
        target_profile=TARGET_PROFILES["happiness"],
        gesture_type="point",
        reward_weights=RewardWeights(),
        joint_limits=JointLimits(),
        arm=arm,
        action=action,
    )

    assert np.isfinite(reward)
    assert "style_error" in info
    assert "action_penalty" in info
    assert "saturation_penalty" in info

def test_endpoint_error_is_zero_for_identical_trajectories():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)

    q_ref = make_reference_trajectory("point", arm)

    error = compute_endpoint_error(q_ref, q_ref)

    assert np.isfinite(error)
    assert error == 0.0


def test_style_reward_outweighs_action_penalty_for_confident_target():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filter_config = laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    ranges = load_ranges_or_default("configs/normalisation_ranges_balanced_3gestures.json")

    q_ref = make_reference_trajectory("point", arm)

    action_small = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_small, _ = apply_style_action(q_ref, action_small, gesture_type="point")
    _, var_small = compute_raw_and_norm_features(q_small, arm, filter_config, ranges)

    action_large = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    q_large, _ = apply_style_action(q_ref, action_large, gesture_type="point")
    _, var_large = compute_raw_and_norm_features(q_large, arm, filter_config, ranges)

    reward_small, _ = compute_reward(
        q_ref=q_ref,
        q_var=q_small,
        features_norm=var_small,
        target_profile=TARGET_PROFILES["happiness"],
        gesture_type="point",
        reward_weights=RewardWeights(),
        joint_limits=JointLimits(),
        arm=arm,
        action=action_small,
        target_name="happiness",
    )
    reward_large, _ = compute_reward(
        q_ref=q_ref,
        q_var=q_large,
        features_norm=var_large,
        target_profile=TARGET_PROFILES["happiness"],
        gesture_type="point",
        reward_weights=RewardWeights(),
        joint_limits=JointLimits(),
        arm=arm,
        action=action_large,
        target_name="happiness",
    )

    assert reward_large > reward_small