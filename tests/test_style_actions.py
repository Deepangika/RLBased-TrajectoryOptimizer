import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban

from laban_rl.trajectories import make_reference_trajectory
from laban_rl.style_actions import apply_style_action


def test_style_action_preserves_start_pose_by_default():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)

    q_ref = make_reference_trajectory("point", arm)
    action = np.array([0.8, 0.9, 0.5, -0.2, 0.3, 0.0])

    q_var, _ = apply_style_action(
        q_ref=q_ref,
        action=action,
        gesture_type="point",
        target_name="confident",
        preserve_endpoints=False,
    )

    assert q_var.shape == q_ref.shape
    assert np.allclose(q_var[0], q_ref[0])

    # Endpoint is allowed to vary now, but should not explode.
    assert np.linalg.norm(q_var[-1] - q_ref[-1]) < 1.0


def test_style_action_can_preserve_both_endpoints_when_requested():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)

    q_ref = make_reference_trajectory("point", arm)
    action = np.array([0.8, 0.9, 0.5, -0.2, 0.3, 0.0])

    q_var, _ = apply_style_action(
        q_ref=q_ref,
        action=action,
        gesture_type="point",
        target_name="confident",
        preserve_endpoints=True,
    )

    assert q_var.shape == q_ref.shape
    assert np.allclose(q_var[0], q_ref[0])
    assert np.allclose(q_var[-1], q_ref[-1])


def test_confident_uses_sustained_amplitude_envelope():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)

    q_ref = make_reference_trajectory("point", arm)
    action = np.array([0.8, 0.5, 0.2, 0.0, 0.0, 1.0])

    _, style_params = apply_style_action(
        q_ref=q_ref,
        action=action,
        gesture_type="point",
        target_name="confident",
        preserve_endpoints=False,
    )

    assert style_params["amplitude_envelope"] == "sustained_smoothstep"


def test_hesitant_uses_middle_bump_amplitude_envelope():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)

    q_ref = make_reference_trajectory("point", arm)
    action = np.array([0.8, 0.5, 0.2, 0.0, 0.0, -1.0])

    _, style_params = apply_style_action(
        q_ref=q_ref,
        action=action,
        gesture_type="point",
        target_name="hesitant",
        preserve_endpoints=False,
    )

    assert style_params["amplitude_envelope"] == "middle_bump"


def test_style_action_output_is_finite():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)

    q_ref = make_reference_trajectory("reach", arm)
    action = np.zeros(6)

    q_var, style_params = apply_style_action(
        q_ref=q_ref,
        action=action,
        gesture_type="reach",
        target_name="calm",
        preserve_endpoints=False,
    )

    assert q_var.shape == q_ref.shape
    assert np.all(np.isfinite(q_var))
    assert isinstance(style_params, dict)