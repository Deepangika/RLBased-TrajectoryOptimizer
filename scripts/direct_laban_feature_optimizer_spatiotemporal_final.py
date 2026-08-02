"""
direct_laban_feature_optimizer_spatiotemporal_final.py

Spatiotemporal direct Laban-feature optimiser.

This version tests whether target Laban values can be matched using both:

    1. spatial joint-basis residuals
    2. monotonic temporal retiming

The key idea is:

    q_spatial = q_ref + spatial_residual
    q_variant = q_spatial(warped_time)

Spatial changes mainly affect Shape/Space.
Temporal retiming mainly affects Weight/Time/Flow.

Run from project root:

    python scripts/direct_laban_feature_optimizer_spatiotemporal_final.py --gesture point --target confident --out outputs/direct_laban_point_confident_spatiotemporal

Stronger run:

    python scripts/direct_laban_feature_optimizer_spatiotemporal_final.py --gesture point --target confident --n-spatial-basis 6 --n-timing-basis 4 --max-delta 0.35 --time-scale 1.8 --maxiter 100 --out outputs/direct_laban_point_confident_spatiotemporal_stronger

More gesture-preserving run:

    python scripts/direct_laban_feature_optimizer_spatiotemporal_final.py --gesture point --target confident --nearest-path-weight 8.0 --path-length-weight 4.0 --detour-weight 2.0 --max-dev-weight 4.0 --time-roughness-weight 0.2 --out outputs/direct_laban_point_confident_spatiotemporal_strict

Key v2 changes:

    - adds symmetric path-length preservation to stop waves collapsing
      to a much shorter path.
    - removes the hard endpoint assumption by default. The start is still fixed,
      but the final pose can move naturally using a smooth endpoint offset.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

HERE = Path(__file__).resolve()
candidate_roots = [HERE.parent, HERE.parent.parent]
PROJECT_ROOT = None
for candidate in candidate_roots:
    if (candidate / "src" / "laban_rl").exists():
        PROJECT_ROOT = candidate
        break

if PROJECT_ROOT is None:
    raise RuntimeError(
        "Could not find project root. Put this script either in the project root "
        "or in the scripts/ folder of laban_rl_pipeline_refactor."
    )

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import robust_laban_normalisation_balanced_3gestures as laban

from laban_rl.config import FEATURE_KEYS, GESTURE_TYPES, JointLimits
from laban_rl.targets import TARGET_PROFILES
from laban_rl.llm_profiles import resolve_target_profile, save_laban_profile_json
from laban_rl.io_utils import load_ranges_or_default
from laban_rl.trajectories import make_reference_trajectory
from laban_rl.features import compute_raw_and_norm_features, target_dict_to_array
from laban_rl.rewards import compute_joint_limit_penalty, compute_joint_jerk_penalty
from laban_rl.visualisation import save_outputs


def print_section(title: str) -> None:
    print("\n" + "=" * 102)
    print(title)
    print("=" * 102)


def forward_kinematics(q: np.ndarray, arm: laban.ArmConfig) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    shoulder = q[:, 0]
    elbow = q[:, 1]
    x = arm.l1 * np.cos(shoulder) + arm.l2 * np.cos(shoulder + elbow)
    y = arm.l1 * np.sin(shoulder) + arm.l2 * np.sin(shoulder + elbow)
    return np.column_stack([x, y])


def path_length(path: np.ndarray) -> float:
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def make_sine_basis(n_points: int, n_basis: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, n_points)
    return np.asarray([np.sin(np.pi * k * u) for k in range(1, n_basis + 1)], dtype=np.float64)


def smoothstep(u: np.ndarray) -> np.ndarray:
    """0 at start, 1 at end, with zero slope at both ends."""
    return u * u * (3.0 - 2.0 * u)


def make_spatial_trajectory(
    spatial_coeffs_flat: np.ndarray,
    q_ref: np.ndarray,
    spatial_basis: np.ndarray,
    max_delta: float,
    max_end_delta: float,
    endpoint_mode: str,
    max_total_delta: float | None = None,
) -> np.ndarray:
    """
    Apply spatial residuals.

    The old version used only sine bases, which forced the final joint
    configuration to equal the reference endpoint. That can create an
    unnatural late loop when the optimiser tries to match expressive features.

    This version keeps the start fixed, but allows a smooth learned endpoint
    offset when endpoint_mode is 'soft' or 'free'.
    """
    q_ref = np.asarray(q_ref, dtype=np.float64)
    n_basis = spatial_basis.shape[0]

    expected_without_endpoint = 2 * n_basis
    expected_with_endpoint = 2 * n_basis + 2
    coeffs_flat = np.asarray(spatial_coeffs_flat, dtype=np.float64)

    if endpoint_mode == "hard":
        spatial_core = coeffs_flat[:expected_without_endpoint]
        end_offset_coeffs = np.zeros(2, dtype=np.float64)
    else:
        if len(coeffs_flat) < expected_with_endpoint:
            raise ValueError(
                f"Expected at least {expected_with_endpoint} spatial coefficients for endpoint_mode={endpoint_mode!r}, "
                f"got {len(coeffs_flat)}."
            )
        spatial_core = coeffs_flat[:expected_without_endpoint]
        end_offset_coeffs = coeffs_flat[expected_without_endpoint:expected_with_endpoint]

    coeffs = spatial_core.reshape(2, n_basis) * max_delta

    delta = np.zeros_like(q_ref)
    for joint_idx in range(2):
        delta[:, joint_idx] = coeffs[joint_idx] @ spatial_basis

    # Smooth endpoint offset: zero at t=0, reaches learned offset at t=T.
    if endpoint_mode in {"soft", "free"}:
        u = np.linspace(0.0, 1.0, q_ref.shape[0])
        ramp = smoothstep(u)
        end_offset = end_offset_coeffs * max_end_delta
        delta += ramp[:, None] * end_offset[None, :]

    # Bound the combined shoulder-elbow residual, rather than only scaling
    # each basis coefficient independently. Without this guard, several bases
    # can reinforce one another and create a much larger-than-advertised
    # perturbation. Radial rescaling preserves the residual direction.
    if max_total_delta is not None and max_total_delta > 0.0:
        residual_norm = np.linalg.norm(delta, axis=1, keepdims=True)
        scale = np.minimum(1.0, float(max_total_delta) / (residual_norm + 1e-12))
        delta = delta * scale

    q_spatial = q_ref + delta

    # Start should remain fixed to avoid an initial discontinuity.
    q_spatial[0] = q_ref[0]

    # Only hard endpoint mode forces the final pose to match the reference.
    if endpoint_mode == "hard":
        q_spatial[-1] = q_ref[-1]

    return q_spatial


def make_time_warp(timing_coeffs: np.ndarray, n_points: int, time_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create monotonic warp from a positive speed profile."""
    timing_coeffs = np.asarray(timing_coeffs, dtype=np.float64)
    u = np.linspace(0.0, 1.0, n_points)
    log_speed = np.zeros_like(u)
    for k, coeff in enumerate(timing_coeffs, start=1):
        log_speed += float(coeff) * np.sin(np.pi * k * u)
    log_speed = np.clip(log_speed * float(time_scale), -5.0, 5.0)
    speed = np.exp(log_speed)
    warped_u = np.cumsum(speed)
    warped_u = warped_u - warped_u[0]
    warped_u = warped_u / (warped_u[-1] + 1e-8)
    warped_u[0] = 0.0
    warped_u[-1] = 1.0
    return warped_u, speed


def apply_time_warp(q_spatial: np.ndarray, warped_u: np.ndarray) -> np.ndarray:
    q_spatial = np.asarray(q_spatial, dtype=np.float64)
    n_points = q_spatial.shape[0]
    original_u = np.linspace(0.0, 1.0, n_points)
    q_warped = np.zeros_like(q_spatial)
    for joint_idx in range(q_spatial.shape[1]):
        q_warped[:, joint_idx] = np.interp(warped_u, original_u, q_spatial[:, joint_idx])
    q_warped[0] = q_spatial[0]
    q_warped[-1] = q_spatial[-1]
    return q_warped


def spatial_variable_count(n_spatial_basis: int, endpoint_mode: str) -> int:
    base = 2 * n_spatial_basis
    if endpoint_mode == "hard":
        return base
    return base + 2  # two smooth endpoint-offset coefficients, one per joint


def split_variables(x: np.ndarray, n_spatial_basis: int, n_timing_basis: int, endpoint_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    spatial_n = spatial_variable_count(n_spatial_basis, endpoint_mode)
    spatial_coeffs = np.asarray(x[:spatial_n], dtype=np.float64)
    timing_coeffs = np.asarray(x[spatial_n:spatial_n + n_timing_basis], dtype=np.float64)
    return spatial_coeffs, timing_coeffs


def build_variant(
    x: np.ndarray,
    q_ref: np.ndarray,
    spatial_basis: np.ndarray,
    n_spatial_basis: int,
    n_timing_basis: int,
    max_delta: float,
    max_end_delta: float,
    endpoint_mode: str,
    time_scale: float,
    max_total_delta: float | None = None,
):
    spatial_coeffs, timing_coeffs = split_variables(x, n_spatial_basis, n_timing_basis, endpoint_mode)
    q_spatial = make_spatial_trajectory(
        spatial_coeffs,
        q_ref,
        spatial_basis,
        max_delta=max_delta,
        max_end_delta=max_end_delta,
        endpoint_mode=endpoint_mode,
        max_total_delta=max_total_delta,
    )
    warped_u, speed = make_time_warp(timing_coeffs, q_ref.shape[0], time_scale)
    q_var = apply_time_warp(q_spatial, warped_u)
    return q_var, q_spatial, warped_u, speed


def weighted_feature_rmse(var_norm: Dict[str, float], target_profile: Dict[str, float], feature_weights: Dict[str, float]):
    weighted_sq_errors = []
    weight_sum = 0.0
    unweighted_sq_errors = []
    diagnostics = {}
    for key in FEATURE_KEYS:
        value = float(var_norm.get(key, np.nan))
        target = float(target_profile[key])
        if np.isnan(value):
            continue
        err = value - target
        w = float(feature_weights.get(key, 1.0))
        unweighted_sq_errors.append(err ** 2)
        weighted_sq_errors.append(w * err ** 2)
        weight_sum += w
        diagnostics[f"{key}_value"] = value
        diagnostics[f"{key}_target"] = target
        diagnostics[f"{key}_error"] = err
        diagnostics[f"{key}_abs_error"] = abs(err)
    rmse = float(np.sqrt(np.mean(unweighted_sq_errors))) if unweighted_sq_errors else float("inf")
    weighted_rmse = float(np.sqrt(np.sum(weighted_sq_errors) / (weight_sum + 1e-8))) if weighted_sq_errors else float("inf")
    diagnostics["feature_rmse"] = rmse
    diagnostics["weighted_feature_rmse"] = weighted_rmse
    return weighted_rmse, diagnostics


def make_feature_weights(
    space_weight: float,
    shape_weight: float,
    flow_weight: float = 1.0,
) -> Dict[str, float]:
    return {
        "weight": 1.0,
        "time": 1.0,
        "flow_boundness": float(flow_weight),
        "space_indirectness": float(space_weight),
        "shape_arcness": float(shape_weight),
    }


def nearest_path_mse(path_var: np.ndarray, path_ref: np.ndarray) -> Tuple[float, float]:
    diffs = path_var[:, None, :] - path_ref[None, :, :]
    sq_dists = np.sum(diffs ** 2, axis=2)
    min_sq = np.min(sq_dists, axis=1)
    return float(np.mean(min_sq)), float(np.sqrt(np.max(min_sq)))


def compute_preservation_terms(q_ref: np.ndarray, q_var: np.ndarray, q_spatial: np.ndarray, warped_u: np.ndarray, speed: np.ndarray, arm: laban.ArmConfig, detour_tolerance: float, max_dev_tolerance: float) -> Dict[str, float]:
    p_ref = forward_kinematics(q_ref, arm)
    p_var = forward_kinematics(q_var, arm)
    p_spatial = forward_kinematics(q_spatial, arm)

    nearest_mse, nearest_max_dist = nearest_path_mse(p_var, p_ref)
    pointwise_cart_mse = float(np.mean(np.sum((p_var - p_ref) ** 2, axis=1)))
    endpoint_error = float(np.linalg.norm(p_var[0] - p_ref[0]) + np.linalg.norm(p_var[-1] - p_ref[-1]))

    ref_vec = p_ref[-1] - p_ref[0]
    var_vec = p_var[-1] - p_var[0]
    ref_norm = float(np.linalg.norm(ref_vec))
    var_norm = float(np.linalg.norm(var_vec))
    if ref_norm < 1e-8 or var_norm < 1e-8:
        direction_error = 0.0
    else:
        cos_sim = float(np.dot(ref_vec, var_vec) / (ref_norm * var_norm + 1e-8))
        direction_error = float(1.0 - np.clip(cos_sim, -1.0, 1.0))

    ref_len = path_length(p_ref)
    var_len = path_length(p_var)
    spatial_len = path_length(p_spatial)
    length_ratio = float(var_len / (ref_len + 1e-8))
    spatial_length_ratio = float(spatial_len / (ref_len + 1e-8))
    # Old behaviour only penalised paths that became too long.
    # For wave, the optimiser collapsed the wave to ~60% path length.
    # This symmetric term penalises both too-short and too-long paths.
    path_length_error = float((length_ratio - 1.0) ** 2)
    spatial_path_length_error = float((spatial_length_ratio - 1.0) ** 2)

    # Keep the one-sided detour term as an additional guard against large loops.
    detour_excess = max(0.0, length_ratio - (1.0 + detour_tolerance))
    detour_error = float(detour_excess ** 2)
    max_dev_excess = max(0.0, nearest_max_dist - max_dev_tolerance)
    max_dev_error = float(max_dev_excess ** 2)

    du = np.diff(warped_u)
    nominal_du = 1.0 / (len(warped_u) - 1)
    time_warp_deviation = float(np.mean((du - nominal_du) ** 2))
    speed_norm = speed / (np.mean(speed) + 1e-8)
    time_roughness = float(np.mean(np.diff(speed_norm, n=2) ** 2))
    log_speed_range = float(np.max(np.log(speed + 1e-8)) - np.min(np.log(speed + 1e-8)))

    return {
        "nearest_path_mse": nearest_mse,
        "nearest_path_max_dist": nearest_max_dist,
        "pointwise_cart_mse": pointwise_cart_mse,
        "endpoint_error": endpoint_error,
        "direction_error": direction_error,
        "ref_path_length": ref_len,
        "var_path_length": var_len,
        "spatial_path_length": spatial_len,
        "path_length_ratio": length_ratio,
        "spatial_path_length_ratio": spatial_length_ratio,
        "path_length_error": path_length_error,
        "spatial_path_length_error": spatial_path_length_error,
        "detour_error": detour_error,
        "max_dev_error": max_dev_error,
        "time_warp_deviation": time_warp_deviation,
        "time_roughness": time_roughness,
        "log_speed_range": log_speed_range,
    }


def feature_values_are_finite(features: Dict[str, float]) -> bool:
    """Return True only when every required feature exists and is finite."""
    return all(np.isfinite(float(features.get(key, np.nan))) for key in FEATURE_KEYS)


def compute_out_of_range_penalty(features_unclipped: Dict[str, float]) -> float:
    """Smoothly penalise normalised values outside the calibrated [0, 1] range."""
    values = np.asarray([float(features_unclipped[key]) for key in FEATURE_KEYS], dtype=np.float64)
    below = np.maximum(0.0, -values)
    above = np.maximum(0.0, values - 1.0)
    return float(np.mean(below ** 2 + above ** 2))


def compose_loss_components(
    *,
    feature_loss: float,
    target_tracking_errors: Dict[str, float],
    out_of_range_error: float,
    joint_preservation_error: float,
    smoothness_error: float,
    joint_limit_error: float,
    spatial_coeff_error: float,
    timing_coeff_error: float,
    terms: Dict[str, float],
    args,
) -> Dict[str, float]:
    """Single source of truth for both optimisation and final loss reporting."""
    components = {
        "feature_loss": float(feature_loss),
        "target_weight": (
            float(args.weight_target_weight)
            * float(target_tracking_errors["weight"])
        ),
        "target_time": (
            float(args.time_target_weight)
            * float(target_tracking_errors["time"])
        ),
        "target_flow_boundness": (
            float(args.flow_boundness_target_weight)
            * float(target_tracking_errors["flow_boundness"])
        ),
        "target_space_indirectness": (
            float(args.space_indirectness_target_weight)
            * float(target_tracking_errors["space_indirectness"])
        ),
        "target_shape_arcness": (
            float(args.shape_arcness_target_weight)
            * float(target_tracking_errors["shape_arcness"])
        ),
        "out_of_range": float(args.out_of_range_weight) * float(out_of_range_error),
        "joint_preservation": float(args.preserve_weight) * float(joint_preservation_error),
        "nearest_path": float(args.nearest_path_weight) * float(terms["nearest_path_mse"]),
        "endpoint": float(args.endpoint_weight) * float(terms["endpoint_error"]),
        "direction": float(args.direction_weight) * float(terms["direction_error"]),
        "path_length": float(args.path_length_weight) * float(terms["path_length_error"]),
        "detour": float(args.detour_weight) * float(terms["detour_error"]),
        "max_deviation": float(args.max_dev_weight) * float(terms["max_dev_error"]),
        "smoothness": float(args.smooth_weight) * float(smoothness_error),
        "joint_limits": float(args.joint_limit_weight) * float(joint_limit_error),
        "spatial_coefficients": float(args.coeff_weight) * float(spatial_coeff_error),
        "timing_coefficients": float(args.time_coeff_weight) * float(timing_coeff_error),
        "time_warp": float(args.time_warp_weight) * float(terms["time_warp_deviation"]),
        "time_roughness": float(args.time_roughness_weight) * float(terms["time_roughness"]),
    }
    components["total_loss"] = float(sum(components.values()))
    return components


def _validate_external_target_profile(profile: Dict[str, float]) -> Dict[str, float]:
    """Validate and canonicalise a profile supplied by an outer learning loop."""
    missing = [key for key in FEATURE_KEYS if key not in profile]
    extra = [key for key in profile if key not in FEATURE_KEYS]
    if missing:
        raise ValueError(f"External target profile is missing keys: {missing}")
    if extra:
        raise ValueError(f"External target profile has unknown keys: {extra}")

    validated: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        value = float(profile[key])
        if not np.isfinite(value):
            raise ValueError(f"External target value for {key!r} is not finite: {value}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"External target value for {key!r} must be in [0, 1], got {value}."
            )
        validated[key] = value
    return validated


def optimise(args, external_target_profile: Dict[str, float] | None = None):
    try:
        from scipy.optimize import differential_evolution, minimize
    except ImportError as exc:
        raise ImportError("This script requires scipy. Install with: pip install scipy") from exc

    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filter_config = laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    ranges_path = args.ranges
    if not Path(ranges_path).exists():
        candidate = PROJECT_ROOT / ranges_path
        if candidate.exists():
            ranges_path = str(candidate)
    ranges = load_ranges_or_default(ranges_path, gesture=args.gesture)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    q_ref = make_reference_trajectory(gesture_type=args.gesture, arm=arm)
    ref_raw, ref_norm = compute_raw_and_norm_features(q_ref, arm, filter_config, ranges)
    ref_norm_unclipped = laban.normalise_laban_features(
        features=ref_raw,
        normalisation_ranges=ranges,
        clip=False,
    )

    if external_target_profile is not None:
        target_profile = _validate_external_target_profile(external_target_profile)
        target_metadata = {
            "source": "external",
            "description": "Target profile supplied programmatically by an outer learner or experiment.",
        }
    else:
        profile_json = args.profile_json
        if args.profile_source == "llm" and profile_json is None:
            safe_target = str(args.target).strip().lower().replace(" ", "_")
            profile_json = out_path / f"llm_profile_{args.gesture}_{safe_target}.json"

        target_profile, target_metadata = resolve_target_profile(
            target_name=args.target,
            gesture_type=args.gesture,
            profile_source=args.profile_source,
            profile_json=profile_json,
            context=args.llm_context,
            reference_features=ref_norm,
            llm_model=args.llm_model,
            llm_temperature=args.llm_temperature,
            refresh_llm_profile=args.refresh_llm_profile,
        )

    target_arr = target_dict_to_array(target_profile)

    if external_target_profile is not None or args.profile_source != "predefined":
        save_laban_profile_json(
            out_path / "resolved_target_profile.json",
            {
                "target_name": args.target,
                "gesture_type": args.gesture,
                "laban_profile": target_profile,
                "metadata": target_metadata,
            },
        )
    spatial_basis = make_sine_basis(arm.n_points, args.n_spatial_basis)
    joint_limits = JointLimits()
    feature_weights = make_feature_weights(
        args.space_weight,
        args.shape_weight,
        args.flow_weight,
    )

    history = []
    generation_history = []
    eval_count = 0
    best_loss_seen = float("inf")

    def objective(x: np.ndarray) -> float:
        nonlocal eval_count, best_loss_seen
        eval_count += 1
        spatial_coeffs, timing_coeffs = split_variables(x, args.n_spatial_basis, args.n_timing_basis, args.endpoint_mode)
        q_var, q_spatial, warped_u, speed = build_variant(
            x, q_ref, spatial_basis, args.n_spatial_basis, args.n_timing_basis,
            args.max_delta, args.max_end_delta, args.endpoint_mode, args.time_scale,
            args.max_total_delta,
        )
        var_raw, var_norm = compute_raw_and_norm_features(q_var, arm, filter_config, ranges)
        var_norm_unclipped = laban.normalise_laban_features(
            features=var_raw,
            normalisation_ranges=ranges,
            clip=False,
        )

        # Every target dimension is required. Invalid candidates receive a
        # large finite loss so the population can rank and discard them without
        # poisoning SciPy's optimisation state with NaN.
        if not feature_values_are_finite(var_norm_unclipped):
            invalid_loss = float(args.invalid_feature_loss + np.mean(np.asarray(x) ** 2))
            if eval_count % 25 == 0:
                history.append({
                    "eval": eval_count,
                    "total_loss": invalid_loss,
                    "valid_candidate": False,
                })
            best_loss_seen = min(best_loss_seen, invalid_loss)
            return invalid_loss

        # Optimise the linear, unclipped values so extreme candidates remain
        # distinguishable. Clipped values are retained only for compatibility
        # and presentation.
        feature_loss, feature_diag = weighted_feature_rmse(
            var_norm_unclipped,
            target_profile,
            feature_weights,
        )
        target_tracking_errors = {
            key: (
                float(var_norm_unclipped[key]) - float(target_profile[key])
            ) ** 2
            for key in FEATURE_KEYS
        }
        out_of_range_error = compute_out_of_range_penalty(var_norm_unclipped)
        
        joint_preservation_error = float(np.mean((q_spatial - q_ref) ** 2))
        terms = compute_preservation_terms(q_ref, q_var, q_spatial, warped_u, speed, arm, args.detour_tolerance, args.max_dev_tolerance)
        jerk_penalty = compute_joint_jerk_penalty(q_var, arm.dt)
        smoothness_error = 0.001 * jerk_penalty
        joint_limit_error = compute_joint_limit_penalty(q_var, joint_limits)
        spatial_coeff_error = float(np.mean(spatial_coeffs ** 2))
        timing_coeff_error = float(np.mean(timing_coeffs ** 2)) if len(timing_coeffs) else 0.0
        loss_components = compose_loss_components(
            feature_loss=feature_loss,
            target_tracking_errors=target_tracking_errors,
            out_of_range_error=out_of_range_error,
            joint_preservation_error=joint_preservation_error,
            smoothness_error=smoothness_error,
            joint_limit_error=joint_limit_error,
            spatial_coeff_error=spatial_coeff_error,
            timing_coeff_error=timing_coeff_error,
            terms=terms,
            args=args,
        )
        total_loss = loss_components["total_loss"]
        if eval_count % 25 == 0:
            row = {
                "eval": eval_count,
                "total_loss": float(total_loss),
                "feature_loss": float(feature_loss),
                "out_of_range_error": float(out_of_range_error),
                "valid_candidate": True,
                "joint_preservation_error": joint_preservation_error,
                "smoothness_error": smoothness_error,
                "joint_limit_error": joint_limit_error,
                "spatial_coeff_error": spatial_coeff_error,
                "timing_coeff_error": timing_coeff_error,
                **terms,
                **{f"weighted_{key}": value for key, value in loss_components.items()},
                **feature_diag,
            }
            history.append(row)
            print(
                f"eval={eval_count:05d} | loss={total_loss:>8.4f} | feat={feature_loss:>7.4f} | "
                f"near={terms['nearest_path_mse']:>8.6f} | ratio={terms['path_length_ratio']:>5.2f} | plen={terms['path_length_error']:>6.3f} | "
                f"maxd={terms['nearest_path_max_dist']:>6.3f} | warp={terms['log_speed_range']:>5.2f} | "
                f"W={var_norm_unclipped.get('weight', np.nan):>5.3f}, T={var_norm_unclipped.get('time', np.nan):>5.3f}, "
                f"F={var_norm_unclipped.get('flow_boundness', np.nan):>5.3f}, S={var_norm_unclipped.get('space_indirectness', np.nan):>5.3f}, "
                f"Sh={var_norm_unclipped.get('shape_arcness', np.nan):>5.3f}"
            )
        best_loss_seen = min(best_loss_seen, float(total_loss))
        return float(total_loss)

    def de_callback(xk: np.ndarray, convergence: float) -> bool:
        generation_loss = objective(np.asarray(xk, dtype=np.float64))
        generation_history.append({
            "generation": len(generation_history) + 1,
            "generation_best_loss": float(generation_loss),
            "cumulative_best_loss": float(best_loss_seen),
            "convergence": float(convergence),
            "eval_count": int(eval_count),
        })
        return False

    n_vars = spatial_variable_count(args.n_spatial_basis, args.endpoint_mode) + args.n_timing_basis
    bounds = [(-1.0, 1.0)] * n_vars

    print_section("SPATIOTEMPORAL DIRECT LABAN FEATURE OPTIMISER")
    print(f"Project root:          {PROJECT_ROOT}")
    print(f"Gesture:               {args.gesture}")
    print(f"Target:                {args.target}")
    print(f"n_spatial_basis:       {args.n_spatial_basis}")
    print(f"n_timing_basis:        {args.n_timing_basis}")
    print(f"max_delta:             {args.max_delta} rad")
    print(f"max_total_delta:       {args.max_total_delta} rad")
    print(f"max_end_delta:         {args.max_end_delta} rad")
    print(f"endpoint_mode:         {args.endpoint_mode}")
    print(f"time_scale:            {args.time_scale}")
    print(f"maxiter:               {args.maxiter}")
    print(f"popsize:               {args.popsize}")
    print(f"Feature weights:       {feature_weights}")
    print(f"path_length_weight:    {args.path_length_weight}")
    print(f"endpoint_weight:       {args.endpoint_weight}")

    print_section("REFERENCE FEATURES")
    for key in FEATURE_KEYS:
        print(f"  {key:22s}: {ref_norm.get(key, np.nan):.4f}")
    print_section("TARGET PROFILE")
    for key in FEATURE_KEYS:
        print(f"  {key:22s}: {target_profile[key]:.4f}")

    print_section("OPTIMISING")
    result_de = differential_evolution(
        objective, bounds=bounds, maxiter=args.maxiter, popsize=args.popsize,
        mutation=args.de_mutation, recombination=args.de_recombination,
        seed=args.seed, polish=False, updating="immediate", workers=1, tol=1e-4,
        callback=de_callback,
    )

    if args.local_method == "none":
        print_section("LOCAL POLISHING DISABLED")
        result_local = None
        x_best = result_de.x
    else:
        print_section(f"POLISHING WITH {args.local_method}")
        if args.local_method == "Powell":
            local_options = {"maxiter": args.local_maxiter, "ftol": 1e-9, "xtol": 1e-6}
        else:
            local_options = {"maxiter": args.local_maxiter, "ftol": 1e-9}
        result_local = minimize(
            objective,
            x0=result_de.x,
            method=args.local_method,
            bounds=bounds,
            options=local_options,
        )
        x_best = result_local.x if result_local.fun <= result_de.fun else result_de.x

    spatial_coeffs, timing_coeffs = split_variables(x_best, args.n_spatial_basis, args.n_timing_basis, args.endpoint_mode)
    q_var, q_spatial, warped_u, speed = build_variant(
        x_best, q_ref, spatial_basis, args.n_spatial_basis, args.n_timing_basis,
        args.max_delta, args.max_end_delta, args.endpoint_mode, args.time_scale,
        args.max_total_delta,
    )
    var_raw, var_norm = compute_raw_and_norm_features(q_var, arm, filter_config, ranges)
    var_norm_unclipped = laban.normalise_laban_features(
        features=var_raw,
        normalisation_ranges=ranges,
        clip=False,
    )
    if not feature_values_are_finite(var_norm_unclipped):
        raise RuntimeError(
            "Optimisation returned a final trajectory with non-finite required features: "
            f"{var_norm_unclipped}"
        )
    feature_loss, feature_diag = weighted_feature_rmse(var_norm_unclipped, target_profile, feature_weights)
    target_tracking_errors = {
        key: (float(var_norm_unclipped[key]) - float(target_profile[key])) ** 2
        for key in FEATURE_KEYS
    }
    out_of_range_error = compute_out_of_range_penalty(var_norm_unclipped)
    joint_preservation_error = float(np.mean((q_spatial - q_ref) ** 2))
    terms = compute_preservation_terms(q_ref, q_var, q_spatial, warped_u, speed, arm, args.detour_tolerance, args.max_dev_tolerance)
    smoothness_error = 0.001 * compute_joint_jerk_penalty(q_var, arm.dt)
    joint_limit_error = compute_joint_limit_penalty(q_var, joint_limits)
    spatial_coeff_error = float(np.mean(spatial_coeffs ** 2))
    timing_coeff_error = float(np.mean(timing_coeffs ** 2)) if len(timing_coeffs) else 0.0
    loss_components = compose_loss_components(
        feature_loss=feature_loss,
        target_tracking_errors=target_tracking_errors,
        out_of_range_error=out_of_range_error,
        joint_preservation_error=joint_preservation_error,
        smoothness_error=smoothness_error,
        joint_limit_error=joint_limit_error,
        spatial_coeff_error=spatial_coeff_error,
        timing_coeff_error=timing_coeff_error,
        terms=terms,
        args=args,
    )
    total_loss = loss_components["total_loss"]

    reward_info = {
        "total_loss": float(total_loss),
        "feature_loss": float(feature_loss),
        "out_of_range_error": float(out_of_range_error),
        "loss_components": loss_components,
        "joint_preservation_error": joint_preservation_error,
        "smoothness_error": smoothness_error,
        "joint_limit_error": joint_limit_error,
        "spatial_coeff_error": spatial_coeff_error,
        "timing_coeff_error": timing_coeff_error,
        "n_spatial_basis": args.n_spatial_basis,
        "n_timing_basis": args.n_timing_basis,
        "max_delta": args.max_delta,
        "max_total_delta": args.max_total_delta,
        "max_end_delta": args.max_end_delta,
        "endpoint_mode": args.endpoint_mode,
        "time_scale": args.time_scale,
        "preserve_weight": args.preserve_weight,
        "nearest_path_weight": args.nearest_path_weight,
        "endpoint_weight": args.endpoint_weight,
        "direction_weight": args.direction_weight,
        "path_length_weight": args.path_length_weight,
        "detour_weight": args.detour_weight,
        "max_dev_weight": args.max_dev_weight,
        "smooth_weight": args.smooth_weight,
        "joint_limit_weight": args.joint_limit_weight,
        "coeff_weight": args.coeff_weight,
        "time_coeff_weight": args.time_coeff_weight,
        "time_warp_weight": args.time_warp_weight,
        "time_roughness_weight": args.time_roughness_weight,
        "flow_weight": args.flow_weight,
        "space_weight": args.space_weight,
        "shape_weight": args.shape_weight,
        "detour_tolerance": args.detour_tolerance,
        "max_dev_tolerance": args.max_dev_tolerance,
        **terms,
        **feature_diag,
    }

    result = {
        "gesture_type": args.gesture,
        "target_name": f"{args.target}_direct_laban_spatiotemporal",
        "target_profile": target_profile,
        "target_profile_metadata": target_metadata,
        "reward": -float(total_loss),
        "action": x_best,
        "style_params": {
            "method": "spatiotemporal_direct_laban_feature_optimisation",
            "target_profile_source": "external" if external_target_profile is not None else args.profile_source,
            "target_profile_metadata": target_metadata,
            "n_spatial_basis": args.n_spatial_basis,
            "n_timing_basis": args.n_timing_basis,
            "max_delta_rad": args.max_delta,
            "max_total_delta_rad": args.max_total_delta,
            "max_end_delta_rad": args.max_end_delta,
            "endpoint_mode": args.endpoint_mode,
            "time_scale": args.time_scale,
            "local_method": args.local_method,
            "n_coefficients": len(x_best),
            "spatial_coefficients": spatial_coeffs.tolist(),
            "timing_coefficients": timing_coeffs.tolist(),
        },
        "q_ref": q_ref,
        "q_var": q_var,
        "ref_raw": ref_raw,
        "ref_norm": ref_norm,
        "ref_norm_unclipped": ref_norm_unclipped,
        "var_raw": var_raw,
        "var_norm": var_norm,
        "var_norm_unclipped": var_norm_unclipped,
        "reward_info": reward_info,
    }

    save_outputs(result, out_path, arm)
    np.savez(
        out_path / "spatiotemporal_direct_laban_coefficients.npz",
        x=x_best, spatial_coeffs=spatial_coeffs, timing_coeffs=timing_coeffs,
        q_ref=q_ref, q_spatial=q_spatial, q_var=q_var, basis=spatial_basis,
        warped_u=warped_u, speed=speed, target=np.asarray(target_arr)
    )
    if history:
        # Some rows may contain a slightly different set of feature-diagnostic keys
        # when a feature is NaN or unavailable during an intermediate evaluation.
        # Build the CSV header from the union of all row keys instead of only the
        # first row, otherwise csv.DictWriter can crash at the end of optimisation.
        fieldnames = sorted({key for row in history for key in row.keys()})
        with open(out_path / "optimisation_history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(history)
    if generation_history:
        with open(out_path / "generation_convergence.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(generation_history[0].keys()))
            writer.writeheader()
            writer.writerows(generation_history)

    print_section("SPATIOTEMPORAL DIRECT OPTIMISER RESULT")
    print(f"Total loss:              {total_loss:.6f}")
    print(f"Feature loss:            {feature_loss:.6f}")
    print(f"Joint preservation err:  {joint_preservation_error:.6f}")
    print(f"Nearest-path MSE:        {terms['nearest_path_mse']:.6f}")
    print(f"Nearest-path max dist:   {terms['nearest_path_max_dist']:.6f}")
    print(f"Pointwise cart MSE:      {terms['pointwise_cart_mse']:.6f}")
    print(f"Endpoint error:          {terms['endpoint_error']:.6f}")
    print(f"Direction error:         {terms['direction_error']:.6f}")
    print(f"Path length ratio:       {terms['path_length_ratio']:.6f}")
    print(f"Spatial length ratio:    {terms['spatial_path_length_ratio']:.6f}")
    print(f"Path length error:       {terms['path_length_error']:.6f}")
    print(f"Spatial length error:    {terms['spatial_path_length_error']:.6f}")
    print(f"Detour error:            {terms['detour_error']:.6f}")
    print(f"Max dev error:           {terms['max_dev_error']:.6f}")
    print(f"Time warp deviation:     {terms['time_warp_deviation']:.8f}")
    print(f"Time roughness:          {terms['time_roughness']:.8f}")
    print(f"Log speed range:         {terms['log_speed_range']:.6f}")
    print(f"Smoothness err:          {smoothness_error:.6f}")
    print(f"Joint limit err:         {joint_limit_error:.6f}")
    print(f"Spatial coeff err:       {spatial_coeff_error:.6f}")
    print(f"Timing coeff err:        {timing_coeff_error:.6f}")
    print("\nTiming coefficients:")
    print(timing_coeffs)
    print("\nTarget vs variant normalised features:")
    for key in FEATURE_KEYS:
        print(f"  {key:22s}: target={target_profile[key]:.4f} | variant={var_norm.get(key, np.nan):.4f} | ref={ref_norm.get(key, np.nan):.4f}")
    print_section("SAVED OUTPUTS")
    print(f"Saved to: {out_path.resolve()}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", default="point", choices=GESTURE_TYPES)
    parser.add_argument("--target", default="confident", help="Affective target name. With --profile-source predefined, this must exist in TARGET_PROFILES. With --profile-source llm, this can be a new state such as curious or encouraging.")
    parser.add_argument(
        "--ranges",
        default="configs/normalisation_ranges_balanced_3gestures_by_gesture.json",
        help="Path to normalisation ranges JSON. Can contain gesture-specific ranges."
    )
    parser.add_argument("--out", default="outputs/direct_laban_point_confident_spatiotemporal")
    parser.add_argument("--profile-source", choices=["predefined", "llm", "json"], default="predefined", help="Where to get the target Laban profile from.")
    parser.add_argument("--profile-json", default=None, help="Path to save/load a profile JSON. For LLM mode, an existing file is reused unless --refresh-llm-profile is set.")
    parser.add_argument("--llm-context", default="", help="Interaction context sent to the LLM when generating a target profile.")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI model used for LLM profile generation.")
    parser.add_argument("--llm-temperature", type=float, default=0.2, help="Temperature for LLM profile generation. Keep low for reproducibility.")
    parser.add_argument("--refresh-llm-profile", action="store_true", help="Regenerate the LLM profile even if --profile-json already exists.")
    parser.add_argument("--n-spatial-basis", type=int, default=6)
    parser.add_argument("--n-timing-basis", type=int, default=4)
    parser.add_argument("--max-delta", type=float, default=0.35)
    parser.add_argument("--max-total-delta", type=float, default=0.50, help="Strict maximum norm of the combined shoulder-elbow residual at each frame, in radians.")
    parser.add_argument("--max-end-delta", type=float, default=0.22, help="Maximum smooth learned endpoint offset in radians when endpoint mode is soft/free.")
    parser.add_argument("--endpoint-mode", choices=["hard", "soft", "free"], default="soft", help="hard forces final joint pose to match reference; soft/free allow a natural endpoint. soft still uses endpoint_weight as a penalty; free usually pairs with endpoint_weight=0.")
    parser.add_argument("--time-scale", type=float, default=1.6)
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--local-maxiter", type=int, default=100, help="Maximum L-BFGS-B polishing iterations. Lower values are useful for API smoke tests.")
    parser.add_argument("--de-mutation", type=float, default=0.5, help="Fixed DE differential weight F. Tuned application default: 0.5.")
    parser.add_argument("--de-recombination", type=float, default=0.65, help="DE binomial crossover probability CR. Tuned application default: 0.65.")
    parser.add_argument("--local-method", choices=["L-BFGS-B", "Powell", "none"], default="L-BFGS-B", help="Bounded local polishing method. Powell remains available for ablation but is substantially more expensive.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--space-weight", type=float, default=1.0)
    parser.add_argument("--shape-weight", type=float, default=1.0)
    parser.add_argument(
        "--flow-weight",
        type=float,
        default=1.0,
        help="Relative Flow feature weight in the inner optimizer.",
    )
    parser.add_argument("--preserve-weight", type=float, default=0.05)
    parser.add_argument("--nearest-path-weight", type=float, default=1.5)
    parser.add_argument("--endpoint-weight", type=float, default=0.2)
    parser.add_argument("--direction-weight", type=float, default=0.5)
    parser.add_argument("--path-length-weight", type=float, default=1.0, help="Symmetric penalty for path length ratio moving away from 1.0; prevents wave collapse and excessive loops.")
    parser.add_argument("--detour-weight", type=float, default=0.5)
    parser.add_argument("--max-dev-weight", type=float, default=1.0)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--joint-limit-weight", type=float, default=5.0)
    parser.add_argument("--coeff-weight", type=float, default=0.01)
    parser.add_argument("--time-coeff-weight", type=float, default=0.01)
    parser.add_argument("--time-warp-weight", type=float, default=0.01)
    parser.add_argument("--time-roughness-weight", type=float, default=0.03)
    parser.add_argument("--out-of-range-weight", type=float, default=0.25, help="Smooth squared penalty for unclipped normalised features outside [0, 1].")
    parser.add_argument("--invalid-feature-loss", type=float, default=1e6, help="Large finite loss returned when any required feature is NaN or infinite.")
    parser.add_argument("--weight-target-weight", type=float, default=0.05, help="Penalty weight for weight deviation from target.")
    parser.add_argument("--flow-boundness-target-weight", type=float, default=0.05, help="Penalty weight for flow_boundness deviation from target.")
    parser.add_argument("--space-indirectness-target-weight", type=float, default=0.03, help="Penalty weight for space_indirectness deviation from target.")
    parser.add_argument("--shape-arcness-target-weight", type=float, default=0.03, help="Penalty weight for shape_arcness deviation from target.")
    parser.add_argument("--time-target-weight", type=float, default=0.05, help="Penalty weight for time deviation from target profile; prevents overshooting to maximum time.")
    parser.add_argument("--detour-tolerance", type=float, default=0.40)
    parser.add_argument("--max-dev-tolerance", type=float, default=0.08)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    optimise(args)


if __name__ == "__main__":
    main()
