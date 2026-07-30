"""Reference trajectory generation."""

from __future__ import annotations
import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban


def _smoothstep(u: np.ndarray) -> np.ndarray:
    return 3.0 * u**2 - 2.0 * u**3


def _make_circle_trajectory(n_points: int) -> np.ndarray:
    """Trace a smooth closed loop with coordinated shoulder and elbow motion."""
    phase = 2.0 * np.pi * np.linspace(0.0, 1.0, n_points)
    shoulder = 0.50 + 0.22 * np.sin(phase)
    elbow = 0.95 + 0.30 * np.cos(phase)
    return np.column_stack([shoulder, elbow]).astype(np.float64)


def _make_beckon_trajectory(n_points: int) -> np.ndarray:
    """Extend the arm, perform two inward curls, then return to rest."""
    u = np.linspace(0.0, 1.0, n_points)
    q_rest = np.array([0.25, 1.25])
    q_extended = np.array([0.55, 0.62])

    extend = _smoothstep(np.clip(u / 0.25, 0.0, 1.0))
    retract = _smoothstep(np.clip((u - 0.82) / 0.18, 0.0, 1.0))
    base = (
        (1.0 - extend[:, None]) * q_rest
        + extend[:, None] * q_extended
    )

    active = np.clip((u - 0.25) / 0.57, 0.0, 1.0)
    window = np.sin(np.pi * active) ** 2
    curls = np.sin(4.0 * np.pi * active) ** 2
    base[:, 0] -= 0.05 * window * curls
    base[:, 1] += 0.42 * window * curls

    return (
        (1.0 - retract[:, None]) * base
        + retract[:, None] * q_rest
    ).astype(np.float64)


def _make_celebratory_pump_trajectory(n_points: int) -> np.ndarray:
    """Raise the arm and perform two emphatic upward pumping motions."""
    u = np.linspace(0.0, 1.0, n_points)
    q_rest = np.array([0.20, 1.20])
    q_raised = np.array([1.05, 0.82])

    rise = _smoothstep(np.clip(u / 0.28, 0.0, 1.0))
    settle = _smoothstep(np.clip((u - 0.88) / 0.12, 0.0, 1.0))
    q = (1.0 - rise[:, None]) * q_rest + rise[:, None] * q_raised

    active = np.clip((u - 0.28) / 0.60, 0.0, 1.0)
    window = np.sin(np.pi * active) ** 2
    pumps = np.sin(4.0 * np.pi * active)
    q[:, 0] += 0.16 * window * pumps
    q[:, 1] -= 0.28 * window * pumps

    q = (1.0 - settle[:, None]) * q + settle[:, None] * q_raised
    return q.astype(np.float64)


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
    if gesture_type == "circle":
        return _make_circle_trajectory(arm.n_points)
    if gesture_type == "beckon":
        return _make_beckon_trajectory(arm.n_points)
    if gesture_type == "celebratory_pump":
        return _make_celebratory_pump_trajectory(arm.n_points)
    raise ValueError(f"Unknown gesture_type: {gesture_type}")
