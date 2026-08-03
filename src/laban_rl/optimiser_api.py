"""Reusable Python API for the working spatiotemporal Laban optimiser.

This module is the boundary between the future outer contextual-bandit loop and
our existing trajectory optimiser. The outer learner chooses a normalised Laban
target profile; this API passes it into the unchanged optimisation machinery and
returns the complete optimisation result.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import FEATURE_KEYS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPTIMISER_SCRIPT = PROJECT_ROOT / "scripts" / "direct_laban_feature_optimizer_spatiotemporal_final.py"


@dataclass(frozen=True)
class LabanOptimisationResult:
    """Typed view of the values needed by the future outer learner."""

    gesture: str
    target_state: str
    requested_profile: dict[str, float]
    achieved_profile: dict[str, float]
    achieved_profile_clipped: dict[str, float]
    inner_reward: float
    inner_loss: float
    action_coefficients: np.ndarray
    q_ref: np.ndarray
    q_var: np.ndarray
    output_dir: Path
    raw_result: dict[str, Any]

    @property
    def realisation_rmse(self) -> float:
        requested = np.asarray([self.requested_profile[k] for k in FEATURE_KEYS], dtype=float)
        achieved = np.asarray([self.achieved_profile[k] for k in FEATURE_KEYS], dtype=float)
        return float(np.sqrt(np.mean((requested - achieved) ** 2)))


def _load_optimiser_module():
    if not OPTIMISER_SCRIPT.exists():
        raise FileNotFoundError(f"Could not find optimiser script: {OPTIMISER_SCRIPT}")

    spec = spec_from_file_location("laban_spatiotemporal_optimiser", OPTIMISER_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import optimiser script: {OPTIMISER_SCRIPT}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_profile(profile: Mapping[str, float]) -> dict[str, float]:
    missing = [key for key in FEATURE_KEYS if key not in profile]
    extra = [key for key in profile if key not in FEATURE_KEYS]
    if missing:
        raise ValueError(f"Target profile is missing keys: {missing}")
    if extra:
        raise ValueError(f"Target profile has unknown keys: {extra}")

    validated: dict[str, float] = {}
    for key in FEATURE_KEYS:
        value = float(profile[key])
        if not np.isfinite(value):
            raise ValueError(f"Target value for {key!r} is not finite: {value}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Target value for {key!r} must be in [0, 1], got {value}")
        validated[key] = value
    return validated


def optimise_laban_target(
    *,
    gesture: str,
    target_state: str,
    target_profile: Mapping[str, float],
    out_dir: str | Path,
    optimiser_overrides: Mapping[str, Any] | None = None,
) -> LabanOptimisationResult:
    """Optimise one reference gesture against an externally supplied profile.

    Parameters
    ----------
    gesture:
        Reference gesture identity from ``GESTURE_TYPES``.
    target_state:
        Human-readable intended state. It is metadata for the outer learner; it
        does not determine the target profile inside the optimiser.
    target_profile:
        Five normalised Laban targets with keys from ``FEATURE_KEYS``.
    out_dir:
        Directory where the existing optimiser writes its normal outputs.
    optimiser_overrides:
        Optional command-line-style optimiser settings, e.g.
        ``{"maxiter": 20, "popsize": 4, "seed": 17}``.
    """
    module = _load_optimiser_module()
    profile = _validate_profile(target_profile)

    parser = module.build_parser()
    args = parser.parse_args([])
    args.gesture = gesture
    args.target = target_state
    args.out = str(Path(out_dir))

    for name, value in dict(optimiser_overrides or {}).items():
        if not hasattr(args, name):
            raise ValueError(f"Unknown optimiser override: {name!r}")
        setattr(args, name, value)

    raw = module.optimise(args, external_target_profile=profile)

    # The outer loop must see the true residual. Clipped values can hide an
    # infeasible target by making every overshoot look exactly like 0 or 1.
    achieved = {
        key: float(raw["var_norm_unclipped"][key])
        for key in FEATURE_KEYS
    }
    achieved_clipped = {
        key: float(raw["var_norm"][key])
        for key in FEATURE_KEYS
    }
    return LabanOptimisationResult(
        gesture=gesture,
        target_state=target_state,
        requested_profile={key: float(profile[key]) for key in FEATURE_KEYS},
        achieved_profile=achieved,
        achieved_profile_clipped=achieved_clipped,
        inner_reward=float(raw["reward"]),
        inner_loss=float(raw["reward_info"]["total_loss"]),
        action_coefficients=np.asarray(raw["action"], dtype=float),
        q_ref=np.asarray(raw["q_ref"], dtype=float),
        q_var=np.asarray(raw["q_var"], dtype=float),
        output_dir=Path(out_dir),
        raw_result=raw,
    )


def build_reference_motion(
    *,
    gesture: str,
    target_state: str,
    out_dir: str | Path,
) -> LabanOptimisationResult:
    """Build an unstyled reference result for matched perceptual evaluation."""
    module = _load_optimiser_module()
    args = module.build_parser().parse_args([])
    args.gesture = gesture
    args.target = target_state
    args.out = str(Path(out_dir))

    arm = module.laban.ArmConfig(
        n_points=160,
        duration=2.0,
        l1=0.30,
        l2=0.25,
    )
    filter_config = module.laban.FilterConfig(
        enabled=True,
        cutoff_hz=5.0,
        order=4,
    )
    ranges_path = Path(args.ranges)
    if not ranges_path.exists():
        ranges_path = module.PROJECT_ROOT / ranges_path
    ranges = module.load_ranges_or_default(ranges_path, gesture=gesture)
    q_ref = module.make_reference_trajectory(gesture_type=gesture, arm=arm)
    raw_features, clipped_profile = module.compute_raw_and_norm_features(
        q_ref,
        arm,
        filter_config,
        ranges,
    )
    achieved_profile = module.laban.normalise_laban_features(
        features=raw_features,
        normalisation_ranges=ranges,
        clip=False,
    )
    requested_profile = {
        key: float(clipped_profile[key]) for key in FEATURE_KEYS
    }
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return LabanOptimisationResult(
        gesture=gesture,
        target_state=target_state,
        requested_profile=requested_profile,
        achieved_profile={
            key: float(achieved_profile[key]) for key in FEATURE_KEYS
        },
        achieved_profile_clipped=requested_profile,
        inner_reward=0.0,
        inner_loss=0.0,
        action_coefficients=np.zeros(0, dtype=float),
        q_ref=np.asarray(q_ref, dtype=float),
        q_var=np.asarray(q_ref, dtype=float),
        output_dir=output_dir,
        raw_result={
            "reward_info": {
                "total_loss": 0.0,
                "path_length_ratio": 1.0,
                "joint_limit_error": 0.0,
            },
            "motion_source": "reference",
        },
    )
