"""Reference trajectory generation."""

from __future__ import annotations
import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban

def make_reference_trajectory(gesture_type: str, arm: laban.ArmConfig) -> np.ndarray:
    """Create a reference trajectory for the requested gesture type."""
    if gesture_type == "wave":
        return laban.generate_wave_trajectory(
            n_points=arm.n_points,
            shoulder_amp_scale=1.0,
            elbow_amp_scale=1.0,
            speed_scale=1.0,
            phase_offset=0.0,
            rng=np.random.default_rng(1),
        )
    if gesture_type == "reach":
        return laban.generate_reach_trajectory(
            n_points=arm.n_points,
            reach_scale=1.0,
            curve_scale=0.1,
            lift_scale=0.2,
            pause_fraction=0.1,
            rng=np.random.default_rng(2),
        )
    if gesture_type == "point":
        return laban.generate_point_trajectory(
            n_points=arm.n_points,
            sharpness=2.0,
            extension_scale=1.3,
            pause_fraction=0.35,
            rng=np.random.default_rng(3),
        )
    raise ValueError(f"Unknown gesture_type: {gesture_type}")
