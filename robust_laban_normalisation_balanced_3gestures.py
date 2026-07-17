"""
robust_laban_normalisation.py

Self-contained example for:
1. Generating three clear 2-joint gesture trajectories: wave, reach, and point.
2. Computing Laban-inspired features:
   - Weight Effort
   - Time Effort
   - Flow Boundness
   - Space Indirectness
   - Shape Arcness
3. Running a robust random parameter sweep for normalisation.
4. Using percentile-based feature ranges instead of fragile min/max ranges.
5. Normalising feature values for use in an RL reward.

Assumptions:
- 2-link planar arm.
- q[:, 0] = shoulder angle in radians.
- q[:, 1] = elbow angle in radians.
- Link lengths are in metres.
- Time is in seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np

try:
    from scipy.signal import butter, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================
# 1. CONFIGURATION
# ============================================================

@dataclass
class ArmConfig:
    """Physical and sampling configuration."""
    n_points: int = 160
    duration: float = 2.0
    l1: float = 0.30   # upper arm length, metres
    l2: float = 0.25   # forearm length, metres

    @property
    def dt(self) -> float:
        return self.duration / (self.n_points - 1)


@dataclass
class FilterConfig:
    """Optional low-pass filtering configuration."""
    enabled: bool = True
    cutoff_hz: float = 5.0
    order: int = 4


@dataclass
class SweepConfig:
    """Robust normalisation sweep configuration."""
    n_samples: int = 3000
    low_percentile: float = 5.0
    high_percentile: float = 95.0
    seed: int = 42


# ============================================================
# 2. BASIC MATH HELPERS
# ============================================================

def smoothstep(u: np.ndarray) -> np.ndarray:
    """
    Smooth interpolation from 0 to 1 with zero velocity at both ends.

    This is useful for reach-like gestures because it avoids abrupt starts/stops.
    """
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def derivative(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Numerical derivative using central differences.

    Units:
    - position -> velocity: m/s
    - velocity -> acceleration: m/s^2
    - acceleration -> jerk: m/s^3
    """
    x = np.asarray(x, dtype=np.float64)
    return np.gradient(x, dt, axis=0)


def butterworth_lowpass_filter(
    q: np.ndarray,
    dt: float,
    cutoff_hz: float = 5.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth low-pass filter to a joint trajectory.

    q has shape (T, D), where T is time and D is number of joints.

    This removes high-frequency jitter before forward kinematics and derivatives.
    If scipy is unavailable or the trajectory is too short, the original q is returned.
    """
    q = np.asarray(q, dtype=np.float64)

    if not SCIPY_AVAILABLE:
        return q

    fs = 1.0 / dt
    nyquist = 0.5 * fs

    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        return q

    normal_cutoff = cutoff_hz / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # filtfilt needs enough samples for padding.
    padlen = 3 * max(len(a), len(b))
    if q.shape[0] <= padlen:
        return q

    return filtfilt(b, a, q, axis=0)


# ============================================================
# 3. FORWARD KINEMATICS
# ============================================================

def forward_kinematics_2link(
    q: np.ndarray,
    l1: float = 0.30,
    l2: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert 2-joint planar arm trajectory into shoulder, elbow, and wrist positions.

    Input:
        q[:, 0] = shoulder angle, radians
        q[:, 1] = elbow angle, radians

    Output units:
        shoulder, elbow, wrist positions are in metres.
    """
    q = np.asarray(q, dtype=np.float64)

    theta1 = q[:, 0]
    theta2 = q[:, 1]

    shoulder = np.zeros((len(q), 2), dtype=np.float64)

    elbow = np.stack([
        l1 * np.cos(theta1),
        l1 * np.sin(theta1)
    ], axis=1)

    wrist = np.stack([
        l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
        l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    ], axis=1)

    return shoulder, elbow, wrist


# ============================================================
# 4. GESTURE GENERATORS
# ============================================================

def generate_wave_trajectory(
    n_points: int,
    shoulder_amp_scale: float = 1.0,
    elbow_amp_scale: float = 1.0,
    speed_scale: float = 1.0,
    phase_offset: float = 0.0,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Cyclic waving-like gesture.

    This is close to your original reference trajectory. The local-window
    Space descriptor is defined even when the start and end positions coincide.
    """
    if rng is None:
        rng = np.random.default_rng()

    u = np.linspace(0.0, 1.0, n_points)
    phase = 2.0 * np.pi * speed_scale * u

    shoulder_center = 0.35
    elbow_center = 1.10

    shoulder_amp = 0.08 * shoulder_amp_scale
    elbow_amp = 0.35 * elbow_amp_scale

    theta1 = shoulder_center + shoulder_amp * np.sin(phase)
    theta2 = elbow_center + elbow_amp * np.sin(2.0 * phase + phase_offset)

    q = np.stack([theta1, theta2], axis=1)

    if noise_scale > 0.0:
        q += rng.normal(0.0, noise_scale, size=q.shape)

    return q.astype(np.float64)


def generate_reach_trajectory(
    n_points: int,
    reach_scale: float = 1.0,
    curve_scale: float = 0.0,
    lift_scale: float = 0.0,
    pause_fraction: float = 0.0,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Non-cyclic reaching gesture.

    This creates a smooth movement from a start joint configuration to an end joint
    configuration. Because the start and end are different, Space Indirectness is
    usually meaningful.
    """
    if rng is None:
        rng = np.random.default_rng()

    u = np.linspace(0.0, 1.0, n_points)

    # Optional hold at the end.
    active_fraction = max(1e-3, 1.0 - pause_fraction)
    u_active = np.clip(u / active_fraction, 0.0, 1.0)
    s = smoothstep(u_active)

    q_start = np.array([0.20, 1.25])
    q_end = np.array([
        0.45 + 0.18 * reach_scale,
        0.85 - 0.20 * reach_scale
    ])

    q = (1.0 - s[:, None]) * q_start + s[:, None] * q_end

    # Mid-gesture expressive bump.
    bump = np.sin(np.pi * s)
    q[:, 0] += lift_scale * 0.18 * bump
    q[:, 1] += curve_scale * 0.25 * bump

    if noise_scale > 0.0:
        q += rng.normal(0.0, noise_scale, size=q.shape)

    return q.astype(np.float64)


def generate_arc_reach_trajectory(
    n_points: int,
    arc_scale: float = 1.0,
    speed_shape: float = 1.0,
    pause_fraction: float = 0.0,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Reach-like gesture with stronger arc-like joint coordination.

    arc_scale controls the mid-trajectory curvature.
    speed_shape changes the time profile:
        < 1: slower start / faster end
        > 1: faster start / slower end
    """
    if rng is None:
        rng = np.random.default_rng()

    u = np.linspace(0.0, 1.0, n_points)

    active_fraction = max(1e-3, 1.0 - pause_fraction)
    u_active = np.clip(u / active_fraction, 0.0, 1.0)

    # Nonlinear progression to vary acceleration profile.
    shaped = np.power(u_active, speed_shape)
    s = smoothstep(shaped)

    q_start = np.array([0.15, 1.35])
    q_end = np.array([0.75, 0.75])

    q = (1.0 - s[:, None]) * q_start + s[:, None] * q_end

    # Stronger coordinated arc.
    bump = np.sin(np.pi * s)
    q[:, 0] += 0.20 * arc_scale * bump
    q[:, 1] -= 0.15 * arc_scale * bump

    if noise_scale > 0.0:
        q += rng.normal(0.0, noise_scale, size=q.shape)

    return q.astype(np.float64)


def generate_point_trajectory(
    n_points: int,
    sharpness: float = 1.0,
    extension_scale: float = 1.0,
    pause_fraction: float = 0.25,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Pointing-like gesture.

    Moves quickly toward an extended posture, then holds.
    sharpness controls how sudden the motion is.
    pause_fraction controls the hold at the end.
    """
    if rng is None:
        rng = np.random.default_rng()

    u = np.linspace(0.0, 1.0, n_points)

    active_fraction = max(1e-3, 1.0 - pause_fraction)
    u_active = np.clip(u / active_fraction, 0.0, 1.0)

    # Higher sharpness makes the movement more sudden.
    # Use a power curve then smoothstep for basic smoothness.
    shaped = np.power(u_active, 1.0 / max(0.2, sharpness))
    s = smoothstep(shaped)

    q_start = np.array([0.30, 1.20])
    q_end = np.array([
        0.35 + 0.12 * extension_scale,
        0.70 - 0.18 * extension_scale
    ])

    q = (1.0 - s[:, None]) * q_start + s[:, None] * q_end

    if noise_scale > 0.0:
        q += rng.normal(0.0, noise_scale, size=q.shape)

    return q.astype(np.float64)


def generate_small_gesture_trajectory(
    n_points: int,
    amp_scale: float = 1.0,
    speed_scale: float = 1.0,
    phase_offset: float = 0.0,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Small subtle gesture.

    Useful for creating low Weight / low Time examples.
    """
    if rng is None:
        rng = np.random.default_rng()

    u = np.linspace(0.0, 1.0, n_points)
    phase = 2.0 * np.pi * speed_scale * u

    shoulder_center = 0.32
    elbow_center = 1.05

    theta1 = shoulder_center + 0.035 * amp_scale * np.sin(phase)
    theta2 = elbow_center + 0.080 * amp_scale * np.sin(phase + phase_offset)

    q = np.stack([theta1, theta2], axis=1)

    if noise_scale > 0.0:
        q += rng.normal(0.0, noise_scale, size=q.shape)

    return q.astype(np.float64)


def generate_random_gesture(
    gesture_type: str,
    n_points: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate one random gesture and return both q and the parameters used.

    The random ranges should roughly match the region your RL action space is
    allowed to explore.
    """
    if gesture_type == "wave":
        params = {
            "shoulder_amp_scale": rng.uniform(0.3, 1.8),
            "elbow_amp_scale": rng.uniform(0.3, 1.8),
            "speed_scale": rng.uniform(0.5, 2.0),
            "phase_offset": rng.uniform(-np.pi, np.pi),
            "noise_scale": rng.uniform(0.0, 0.005),
        }
        q = generate_wave_trajectory(n_points=n_points, rng=rng, **params)

    elif gesture_type == "reach":
        params = {
            "reach_scale": rng.uniform(0.4, 1.6),
            "curve_scale": rng.uniform(-1.0, 1.0),
            "lift_scale": rng.uniform(-0.5, 1.2),
            "pause_fraction": rng.uniform(0.0, 0.35),
            "noise_scale": rng.uniform(0.0, 0.004),
        }
        q = generate_reach_trajectory(n_points=n_points, rng=rng, **params)

    elif gesture_type == "arc_reach":
        params = {
            "arc_scale": rng.uniform(0.2, 1.8),
            "speed_shape": rng.uniform(0.5, 2.0),
            "pause_fraction": rng.uniform(0.0, 0.30),
            "noise_scale": rng.uniform(0.0, 0.004),
        }
        q = generate_arc_reach_trajectory(n_points=n_points, rng=rng, **params)

    elif gesture_type == "point":
        params = {
            "sharpness": rng.uniform(0.5, 2.5),
            "extension_scale": rng.uniform(0.5, 1.8),
            "pause_fraction": rng.uniform(0.10, 0.50),
            "noise_scale": rng.uniform(0.0, 0.004),
        }
        q = generate_point_trajectory(n_points=n_points, rng=rng, **params)

    elif gesture_type == "small_gesture":
        params = {
            "amp_scale": rng.uniform(0.3, 1.5),
            "speed_scale": rng.uniform(0.5, 1.5),
            "phase_offset": rng.uniform(-np.pi, np.pi),
            "noise_scale": rng.uniform(0.0, 0.003),
        }
        q = generate_small_gesture_trajectory(n_points=n_points, rng=rng, **params)

    else:
        raise ValueError(f"Unknown gesture_type: {gesture_type}")

    return q, params


# ============================================================
# 5. LABAN FEATURE COMPUTATION
# ============================================================

def compute_weight_effort(
    elbow_pos: np.ndarray,
    wrist_pos: np.ndarray,
    dt: float
) -> float:
    """
    Weight Effort: Light <-> Strong.

    Computed as peak kinetic-energy-like value:
        max_t(||v_elbow||^2 + ||v_wrist||^2)

    Unit:
        m^2/s^2

    Interpretation:
        low = Light
        high = Strong
    """
    v_elbow = derivative(elbow_pos, dt)
    v_wrist = derivative(wrist_pos, dt)

    elbow_speed_sq = np.sum(v_elbow ** 2, axis=1)
    wrist_speed_sq = np.sum(v_wrist ** 2, axis=1)

    energy_like = elbow_speed_sq + wrist_speed_sq

    return float(np.max(energy_like))


def compute_time_effort(
    elbow_pos: np.ndarray,
    wrist_pos: np.ndarray,
    dt: float
) -> float:
    """
    Time Effort: Sustained <-> Sudden.

    Computed as peak combined acceleration:
        max_t(||a_elbow|| + ||a_wrist||)

    Unit:
        m/s^2

    Interpretation:
        low = Sustained
        high = Sudden
    """
    v_elbow = derivative(elbow_pos, dt)
    v_wrist = derivative(wrist_pos, dt)

    a_elbow = derivative(v_elbow, dt)
    a_wrist = derivative(v_wrist, dt)

    total_acc = np.linalg.norm(a_elbow, axis=1) + np.linalg.norm(a_wrist, axis=1)

    return float(np.max(total_acc))


def compute_flow_boundness(
    elbow_pos: np.ndarray,
    wrist_pos: np.ndarray,
    dt: float
) -> float:
    """
    Flow Boundness: Free <-> Bound.

    Computed as mean combined jerk:
        mean_t(||j_elbow|| + ||j_wrist||)

    Unit:
        m/s^3

    Interpretation:
        low = Free / smooth / released
        high = Bound / jerky / constrained
    """
    v_elbow = derivative(elbow_pos, dt)
    v_wrist = derivative(wrist_pos, dt)

    a_elbow = derivative(v_elbow, dt)
    a_wrist = derivative(v_wrist, dt)

    j_elbow = derivative(a_elbow, dt)
    j_wrist = derivative(a_wrist, dt)

    total_jerk = np.linalg.norm(j_elbow, axis=1) + np.linalg.norm(j_wrist, axis=1)

    return float(np.mean(total_jerk))


def compute_space_indirectness(
    wrist_pos: np.ndarray,
    window_size: Optional[int] = None,
    window_fraction: float = 0.075,
    min_window_path_length: float = 1e-6,
    min_window_path_fraction: float = 0.01,
) -> float:
    """
    Bounded local Space Effort proxy: Direct <-> Indirect.

    For each overlapping local window, compute path efficiency as the
    straight-line distance between its endpoints divided by the distance
    travelled inside the window. Space indirectness is one minus the mean
    efficiency:

        mean_i(1 - endpoint_displacement_i / local_path_length_i)

    Interpretation:
        0 = locally straight/direct
        1 = maximally indirect (a window travels but returns to its start)

    Unlike the former whole-trajectory path-length/displacement ratio, this
    descriptor remains bounded and finite when a cyclic gesture finishes near
    its starting point. Near-stationary windows are excluded rather than being
    labelled maximally indirect. A completely stationary path returns 0.0.

    ``window_fraction`` makes the descriptor stable to temporal resampling.
    With the default 160 samples and 0.075 fraction, the window spans 12 sample
    intervals. ``window_size`` can override this for sensitivity analysis.
    """
    wrist_pos = np.asarray(wrist_pos, dtype=np.float64)
    if wrist_pos.ndim != 2 or wrist_pos.shape[0] < 2:
        raise ValueError("wrist_pos must have shape (n_samples, n_dimensions) with n_samples >= 2")
    if not np.all(np.isfinite(wrist_pos)):
        return float("nan")
    if min_window_path_length < 0.0:
        raise ValueError("min_window_path_length must be non-negative")
    if min_window_path_fraction < 0.0:
        raise ValueError("min_window_path_fraction must be non-negative")

    n_intervals = wrist_pos.shape[0] - 1
    if window_size is None:
        if not 0.0 < window_fraction <= 1.0:
            raise ValueError("window_fraction must be in (0, 1]")
        window_size = int(round(window_fraction * n_intervals))
    window_size = int(np.clip(window_size, 1, n_intervals))

    segment_lengths = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)
    cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    starts = np.arange(0, wrist_pos.shape[0] - window_size)
    ends = starts + window_size
    local_lengths = cumulative_length[ends] - cumulative_length[starts]
    local_displacements = np.linalg.norm(wrist_pos[ends] - wrist_pos[starts], axis=1)

    # A relative threshold prevents tiny filter ringing inside an otherwise
    # stationary hold from being interpreted as a highly indirect path. The
    # scale is the path length an average window would contain.
    expected_average_window_length = float(cumulative_length[-1]) * window_size / n_intervals
    effective_minimum = max(
        float(min_window_path_length),
        float(min_window_path_fraction) * expected_average_window_length,
    )
    valid = local_lengths > effective_minimum
    if not np.any(valid):
        return 0.0

    efficiencies = np.clip(
        local_displacements[valid] / local_lengths[valid],
        0.0,
        1.0,
    )
    return float(np.clip(np.mean(1.0 - efficiencies), 0.0, 1.0))


def compute_shape_arcness(
    wrist_pos: np.ndarray,
    dt: float,
    speed_threshold: float = 1e-4
) -> float:
    """
    Shape Directional proxy: Spoke-like <-> Arc-like.

    Computed using path-length-weighted curvature.

    Unit:
        1/m

    Interpretation:
        low = Spoke-like / straighter
        high = Arc-like / more curved
    """
    wrist_pos = np.asarray(wrist_pos, dtype=np.float64)

    x = wrist_pos[:, 0]
    y = wrist_pos[:, 1]

    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)

    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)

    speed = np.sqrt(dx ** 2 + dy ** 2)

    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx ** 2 + dy ** 2) ** 1.5

    valid = speed > speed_threshold

    if np.sum(valid) == 0:
        return 0.0

    curvature = numerator[valid] / (denominator[valid] + 1e-8)
    weights = speed[valid]

    shape_arcness = np.sum(curvature * weights) / (np.sum(weights) + 1e-8)

    return float(shape_arcness)


def compute_laban_features(
    q: np.ndarray,
    arm: ArmConfig,
    filter_config: Optional[FilterConfig] = None
) -> Dict[str, float]:
    """
    Full Laban-inspired feature extractor.

    Pipeline:
        joint trajectory q
        -> optional low-pass filter
        -> forward kinematics
        -> derivatives/statistics
        -> raw Laban feature values
    """
    q = np.asarray(q, dtype=np.float64)

    if filter_config is not None and filter_config.enabled:
        q = butterworth_lowpass_filter(
            q=q,
            dt=arm.dt,
            cutoff_hz=filter_config.cutoff_hz,
            order=filter_config.order
        )

    _, elbow_pos, wrist_pos = forward_kinematics_2link(
        q=q,
        l1=arm.l1,
        l2=arm.l2
    )

    return {
        "weight": compute_weight_effort(elbow_pos, wrist_pos, dt=arm.dt),
        "time": compute_time_effort(elbow_pos, wrist_pos, dt=arm.dt),
        "flow_boundness": compute_flow_boundness(elbow_pos, wrist_pos, dt=arm.dt),
        "space_indirectness": compute_space_indirectness(wrist_pos),
        "shape_arcness": compute_shape_arcness(wrist_pos, dt=arm.dt),
    }


# ============================================================
# 6. NORMALISATION
# ============================================================

FEATURE_KEYS = [
    "weight",
    "time",
    "flow_boundness",
    "space_indirectness",
    "shape_arcness",
]


def normalise_value(
    value: float,
    min_value: float,
    max_value: float,
    clip: bool = True
) -> float:
    """
    Min-max normalise one value to [0, 1].

    Important:
    - Raw values have physical units.
    - Normalised values are unitless relative scores.

    If clip=True:
        values outside the normalisation range are clipped to 0 or 1.
    """
    if np.isnan(value):
        return float("nan")

    if max_value - min_value < 1e-8:
        return 0.0

    norm_value = (value - min_value) / (max_value - min_value)

    if clip:
        norm_value = np.clip(norm_value, 0.0, 1.0)

    return float(norm_value)


def normalise_laban_features(
    features: Dict[str, float],
    normalisation_ranges: Dict[str, Tuple[float, float]],
    clip: bool = True
) -> Dict[str, float]:
    """Normalise a full feature dictionary."""
    norm_features = {}

    for key, value in features.items():
        if key not in normalisation_ranges:
            norm_features[key] = value
            continue

        min_value, max_value = normalisation_ranges[key]

        norm_features[key] = normalise_value(
            value=value,
            min_value=min_value,
            max_value=max_value,
            clip=clip
        )

    return norm_features


def compute_robust_normalisation_ranges(
    sweep_results: List[Dict[str, float]],
    low_percentile: float = 5.0,
    high_percentile: float = 95.0
) -> Dict[str, Tuple[float, float]]:
    """
    Compute robust normalisation ranges using percentiles.

    This is more robust than using absolute min/max because a single weird
    outlier will not define the full scale.
    """
    ranges = {}

    for key in FEATURE_KEYS:
        values = np.array(
            [result[key] for result in sweep_results if key in result],
            dtype=np.float64
        )

        valid_values = values[~np.isnan(values)]

        if len(valid_values) == 0:
            continue

        low = float(np.percentile(valid_values, low_percentile))
        high = float(np.percentile(valid_values, high_percentile))

        if high - low < 1e-8:
            continue

        ranges[key] = (low, high)

    return ranges


def compute_robust_normalisation_ranges_by_gesture(
    sweep_results: List[Dict[str, float]],
    gesture_types: Optional[List[str]] = None,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Compute robust normalisation ranges separately for each gesture type.

    The returned mapping is gesture -> feature -> (min, max).
    """
    if gesture_types is None:
        gesture_types = ["wave", "reach", "point"]

    gesture_ranges: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for gesture_type in gesture_types:
        gesture_results = [
            result for result in sweep_results
            if result.get("gesture_type") == gesture_type
        ]
        if not gesture_results:
            continue

        gesture_ranges[gesture_type] = compute_robust_normalisation_ranges(
            sweep_results=gesture_results,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
        )

    return gesture_ranges


# ============================================================
# 7. ROBUST RANDOM PARAMETER SWEEP
# ============================================================

def run_robust_parameter_sweep(
    arm: ArmConfig,
    sweep: SweepConfig,
    filter_config: Optional[FilterConfig] = None,
    gesture_types: Optional[List[str]] = None
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, Tuple[float, float]]]]:
    """
    Run random sweep across several gesture families.

    This is intended for robust RL reward normalisation.
    """
    rng = np.random.default_rng(sweep.seed)

    if gesture_types is None:
        # For the current prototype, keep the sweep focused on the
        # three clearest and most useful gesture families.
        gesture_types = [
            "wave",
            "reach",
            "point",
        ]

    sweep_results = []

    # ------------------------------------------------------------
    # Balanced sampling
    # ------------------------------------------------------------
    # Instead of choosing gesture types randomly, allocate an equal
    # number of samples to each gesture family.
    #
    # Example:
    #     n_samples = 3000
    #     gesture_types = ["wave", "reach", "point"]
    #
    # Then:
    #     wave  -> 1000 samples
    #     reach -> 1000 samples
    #     point -> 1000 samples
    #
    # This prevents one gesture type from contributing more strongly
    # to the percentile-based normalisation ranges just because it
    # appeared more often in the random draw.
    # ------------------------------------------------------------

    samples_per_gesture = sweep.n_samples // len(gesture_types)
    remainder = sweep.n_samples % len(gesture_types)

    sample_id = 0

    for gesture_index, gesture_type in enumerate(gesture_types):

        # If n_samples is not perfectly divisible by the number of
        # gesture types, distribute the leftover samples across the
        # first few gesture types.
        n_for_this_gesture = samples_per_gesture
        if gesture_index < remainder:
            n_for_this_gesture += 1

        for _ in range(n_for_this_gesture):

            q, params = generate_random_gesture(
                gesture_type=gesture_type,
                n_points=arm.n_points,
                rng=rng
            )

            features = compute_laban_features(
                q=q,
                arm=arm,
                filter_config=filter_config
            )

            result = {
                "sample_id": sample_id,
                "gesture_type": gesture_type,
                **params,
                **features,
            }

            sweep_results.append(result)
            sample_id += 1

    # Shuffle the completed sweep so results are not ordered by gesture type.
    # This does not affect the percentile ranges, but makes saved/debug data
    # look more like a mixed dataset.
    rng.shuffle(sweep_results)

    balanced_ranges = compute_robust_normalisation_ranges(
        sweep_results=sweep_results,
        low_percentile=sweep.low_percentile,
        high_percentile=sweep.high_percentile
    )

    gesture_ranges = compute_robust_normalisation_ranges_by_gesture(
        sweep_results=sweep_results,
        gesture_types=gesture_types,
        low_percentile=sweep.low_percentile,
        high_percentile=sweep.high_percentile,
    )

    # Keep the overall balanced range for backwards compatibility, while also
    # exposing per-gesture ranges for gesture-specific normalisation.
    normalisation_ranges = {"balanced": balanced_ranges, **gesture_ranges}

    return sweep_results, normalisation_ranges


# ============================================================
# 8. DIAGNOSTICS
# ============================================================

def summarise_sweep_results(
    sweep_results: List[Dict[str, float]],
    normalisation_ranges: Dict[str, Tuple[float, float]] | Dict[str, Dict[str, Tuple[float, float]]]
) -> None:
    """
    Print raw feature distributions and robust normalisation ranges.
    """
    print("\n" + "=" * 70)
    print("ROBUST PARAMETER SWEEP SUMMARY")
    print("=" * 70)

    print(f"Number of samples: {len(sweep_results)}")

    gesture_counts = {}
    for result in sweep_results:
        gesture = result["gesture_type"]
        gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1

    print("\nGesture counts:")
    for gesture, count in sorted(gesture_counts.items()):
        print(f"  {gesture:15s}: {count}")

    print("\nRaw feature distribution:")
    for key in FEATURE_KEYS:
        values = np.array([result[key] for result in sweep_results], dtype=np.float64)
        valid = values[~np.isnan(values)]
        nan_count = int(np.sum(np.isnan(values)))

        if len(valid) == 0:
            print(f"  {key:22s}: all nan")
            continue

        print(
            f"  {key:22s}: "
            f"min={np.min(valid):10.5f}, "
            f"p05={np.percentile(valid, 5):10.5f}, "
            f"median={np.median(valid):10.5f}, "
            f"p95={np.percentile(valid, 95):10.5f}, "
            f"max={np.max(valid):10.5f}, "
            f"nan={nan_count}"
        )

    print("\nNormalisation ranges (balanced):")
    if normalisation_ranges and isinstance(next(iter(normalisation_ranges.values())), dict):
        balanced_ranges = normalisation_ranges.get("balanced", {})
    else:
        balanced_ranges = normalisation_ranges

    for key, value_range in balanced_ranges.items():
        low, high = value_range
        print(f"  {key:22s}: low={low:.6f}, high={high:.6f}")

    if normalisation_ranges and isinstance(next(iter(normalisation_ranges.values())), dict):
        print("\nNormalisation ranges by gesture:")
        for gesture, ranges in normalisation_ranges.items():
            if gesture == "balanced":
                continue
            print(f"  Gesture: {gesture}")
            for key, value_range in ranges.items():
                low, high = value_range
                print(f"    {key:18s}: low={low:.6f}, high={high:.6f}")


def compute_clipping_diagnostics(
    sweep_results: List[Dict[str, float]],
    normalisation_ranges: Dict[str, Tuple[float, float]] | Dict[str, Dict[str, Tuple[float, float]]]
) -> Dict[str, Dict[str, float]]:
    """
    Estimate how often features would be clipped to 0 or 1.

    This helps diagnose whether your normalisation ranges are too narrow.
    """
    diagnostics = {}

    if normalisation_ranges and isinstance(next(iter(normalisation_ranges.values())), dict):
        normalisation_ranges = normalisation_ranges.get("balanced", {})

    for key in FEATURE_KEYS:
        if key not in normalisation_ranges:
            continue

        low, high = normalisation_ranges[key]
        values = np.array([result[key] for result in sweep_results], dtype=np.float64)
        valid = values[~np.isnan(values)]

        if len(valid) == 0:
            continue

        below = np.mean(valid < low)
        above = np.mean(valid > high)
        inside = np.mean((valid >= low) & (valid <= high))

        diagnostics[key] = {
            "below_range_fraction": float(below),
            "inside_range_fraction": float(inside),
            "above_range_fraction": float(above),
        }

    return diagnostics


def print_clipping_diagnostics(
    diagnostics: Dict[str, Dict[str, float]]
) -> None:
    """Print clipping diagnostics."""
    print("\n" + "=" * 70)
    print("CLIPPING DIAGNOSTICS")
    print("=" * 70)

    for key, stats in diagnostics.items():
        print(
            f"{key:22s}: "
            f"below={100 * stats['below_range_fraction']:5.1f}% | "
            f"inside={100 * stats['inside_range_fraction']:5.1f}% | "
            f"above={100 * stats['above_range_fraction']:5.1f}%"
        )


def print_features(title: str, features: Dict[str, float]) -> None:
    """Pretty-print a feature dictionary."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for key, value in features.items():
        print(f"{key:22s}: {value}")


# ============================================================
# 9. REWARD EXAMPLE
# ============================================================

def compute_laban_target_reward(
    current_features_norm: Dict[str, float],
    target_features_norm: Dict[str, float],
    feature_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Example reward based on normalised Laban feature error.

    Returns:
        0 for perfect match.
        Negative values for mismatch.

    You can use this as a starting point in the RL environment.
    """
    if feature_weights is None:
        feature_weights = {
            "weight": 1.0,
            "time": 1.0,
            "flow_boundness": 1.0,
            "space_indirectness": 1.0,
            "shape_arcness": 1.0,
        }

    total_error = 0.0
    total_weight = 0.0

    for key, w in feature_weights.items():
        if key not in current_features_norm or key not in target_features_norm:
            continue

        current_value = current_features_norm[key]
        target_value = target_features_norm[key]

        if np.isnan(current_value) or np.isnan(target_value):
            continue

        error = abs(current_value - target_value)

        total_error += w * error
        total_weight += w

    if total_weight <= 0.0:
        return 0.0

    return float(-total_error / total_weight)


# ============================================================
# 10. SAVE / LOAD NORMALISATION RANGES
# ============================================================

def save_normalisation_ranges(
    normalisation_ranges: Dict[str, Tuple[float, float]] | Dict[str, Dict[str, Tuple[float, float]]],
    path: str | Path
) -> None:
    """Save normalisation ranges to JSON.

    Supports either flat feature ranges or nested gesture-specific ranges.
    """
    path = Path(path)

    sample = next(iter(normalisation_ranges.values()), None)

    if sample is None:
        serialisable = {}
    elif isinstance(sample, tuple):
        serialisable = {
            key: {"min": value[0], "max": value[1]}
            for key, value in normalisation_ranges.items()
        }
    else:
        serialisable = {
            gesture: {
                key: {"min": value[0], "max": value[1]}
                for key, value in ranges.items()
            }
            for gesture, ranges in normalisation_ranges.items()
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)


def load_normalisation_ranges(path: str | Path) -> Dict[str, Tuple[float, float]] | Dict[str, Dict[str, Tuple[float, float]]]:
    """Load normalisation ranges from JSON.

    The file may contain either:
      1) flat feature->(min,max) ranges, or
      2) gesture->feature->(min,max) mappings.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return {}

    sample = next(iter(data.values()))
    if isinstance(sample, dict) and "min" in sample and "max" in sample:
        return {
            key: (float(value["min"]), float(value["max"]))
            for key, value in data.items()
        }

    return {
        gesture: {
            key: (float(value["min"]), float(value["max"]))
            for key, value in ranges.items()
        }
        for gesture, ranges in data.items()
    }


# ============================================================
# 11. MAIN SCRIPT
# ============================================================

def main() -> None:
    arm = ArmConfig(
        n_points=160,
        duration=2.0,
        l1=0.30,
        l2=0.25
    )

    filter_config = FilterConfig(
        enabled=True,
        cutoff_hz=5.0,
        order=4
    )

    sweep_config = SweepConfig(
        n_samples=3000,
        low_percentile=5.0,
        high_percentile=95.0,
        seed=42
    )

    print("Scipy available for Butterworth filtering:", SCIPY_AVAILABLE)
    print("Arm config:", asdict(arm))
    print("Filter config:", asdict(filter_config))
    print("Sweep config:", asdict(sweep_config))

    # ------------------------------------------------------------
    # Step 1: Run robust random sweep for normalisation ranges
    # ------------------------------------------------------------

    sweep_results, normalisation_ranges = run_robust_parameter_sweep(
        arm=arm,
        sweep=sweep_config,
        filter_config=filter_config
    )

    summarise_sweep_results(
        sweep_results=sweep_results,
        normalisation_ranges=normalisation_ranges
    )

    diagnostics = compute_clipping_diagnostics(
        sweep_results=sweep_results,
        normalisation_ranges=normalisation_ranges
    )

    print_clipping_diagnostics(diagnostics)

    save_normalisation_ranges(
        normalisation_ranges=normalisation_ranges,
        path="normalisation_ranges_balanced_3gestures_by_gesture.json"
    )

    print("\nSaved gesture-specific normalisation ranges to: normalisation_ranges_balanced_3gestures_by_gesture.json")

    # ------------------------------------------------------------
    # Step 2: Test several example gestures
    # ------------------------------------------------------------

    examples = {
        "wave_reference": generate_wave_trajectory(
            n_points=arm.n_points,
            shoulder_amp_scale=1.0,
            elbow_amp_scale=1.0,
            speed_scale=1.0,
            phase_offset=0.0,
            rng=np.random.default_rng(1)
        ),
        "larger_faster_wave": generate_wave_trajectory(
            n_points=arm.n_points,
            shoulder_amp_scale=1.4,
            elbow_amp_scale=1.5,
            speed_scale=1.5,
            phase_offset=0.5,
            rng=np.random.default_rng(2)
        ),
        "smooth_reach": generate_reach_trajectory(
            n_points=arm.n_points,
            reach_scale=1.0,
            curve_scale=0.1,
            lift_scale=0.2,
            pause_fraction=0.1,
            rng=np.random.default_rng(3)
        ),
        "sharp_point": generate_point_trajectory(
            n_points=arm.n_points,
            sharpness=2.0,
            extension_scale=1.3,
            pause_fraction=0.35,
            rng=np.random.default_rng(4)
        ),
    }

    for name, q in examples.items():
        raw_features = compute_laban_features(
            q=q,
            arm=arm,
            filter_config=filter_config
        )

        norm_features = normalise_laban_features(
            features=raw_features,
            normalisation_ranges=normalisation_ranges,
            clip=True
        )

        print_features(f"{name}: raw features", raw_features)
        print_features(f"{name}: normalised features", norm_features)

    # ------------------------------------------------------------
    # Step 3: Example RL target reward
    # ------------------------------------------------------------

    # Example: a confident target profile.
    # These values are already normalised expressive targets in [0, 1].
    target_confident = {
        "weight": 0.80,           # strong
        "time": 0.70,             # fairly sudden
        "flow_boundness": 0.30,   # not too bound
        "space_indirectness": 0.35,
        "shape_arcness": 0.45,
    }

    q_test = examples["smooth_reach"]
    test_raw = compute_laban_features(q_test, arm=arm, filter_config=filter_config)
    test_norm = normalise_laban_features(test_raw, normalisation_ranges)

    reward = compute_laban_target_reward(
        current_features_norm=test_norm,
        target_features_norm=target_confident,
        feature_weights={
            "weight": 1.0,
            "time": 1.0,
            "flow_boundness": 1.0,
            "space_indirectness": 0.5,
            "shape_arcness": 1.0,
        }
    )

    print("\n" + "=" * 70)
    print("EXAMPLE RL REWARD")
    print("=" * 70)
    print("Target confident profile:", target_confident)
    print("Current normalised features:", test_norm)
    print("Reward:", reward)


if __name__ == "__main__":
    main()
