"""Project difficult wave targets onto a seed-validated feasible profile region."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.config import FEATURE_KEYS
from laban_rl.targets import TARGET_PROFILES
from scripts.evaluation.run_focused_inner_screen import (
    load_optimiser,
    write_json_atomic,
)
from scripts.train_cem_contextual_bandit import _build_optimiser_overrides

STATES = ("anger", "disgust", "sadness")
VALIDATION_SEEDS = (7, 17, 27)
TOLERANCE = 0.10
PROJECTION_VERSION = 1
PRODUCTION_CONFIG = ROOT / "configs" / "wave_feasible_target_projections.json"


def profile_distance(
    profile: Mapping[str, float],
    original: Mapping[str, float],
) -> float:
    errors = np.asarray(
        [float(profile[key]) - float(original[key]) for key in FEATURE_KEYS],
        dtype=float,
    )
    return float(np.sqrt(np.mean(errors**2)))


def wave_reference_profile(module: Any) -> dict[str, float]:
    arm = module.laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filter_config = module.laban.FilterConfig(
        enabled=True, cutoff_hz=5.0, order=4
    )
    ranges = module.load_ranges_or_default(
        ROOT / "configs" / "normalisation_ranges_balanced_3gestures_by_gesture.json",
        gesture="wave",
    )
    q_ref = module.make_reference_trajectory(gesture_type="wave", arm=arm)
    raw, _ = module.compute_raw_and_norm_features(
        q_ref, arm, filter_config, ranges
    )
    return {
        key: float(value)
        for key, value in module.laban.normalise_laban_features(
            features=raw,
            normalisation_ranges=ranges,
            clip=False,
        ).items()
    }


def generate_candidates(
    original: Mapping[str, float],
    reference: Mapping[str, float],
    *,
    count: int,
    seed: int,
) -> list[dict[str, float]]:
    if count < 32:
        raise ValueError("candidate count must be at least 32.")
    original_array = np.asarray([original[key] for key in FEATURE_KEYS], dtype=float)
    reference_array = np.asarray(
        [reference[key] for key in FEATURE_KEYS], dtype=float
    )
    line_count = min(32, count)
    candidates = [
        (1.0 - alpha) * original_array + alpha * reference_array
        for alpha in np.linspace(0.0, 1.0, line_count)
    ]

    remaining = count - line_count
    if remaining:
        dimension = len(FEATURE_KEYS) + 1
        exponent = int(np.ceil(np.log2(remaining)))
        points = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(
            exponent
        )[:remaining]
        for point in points:
            alpha = 0.05 + 0.90 * point[0]
            centre = (
                (1.0 - alpha) * original_array + alpha * reference_array
            )
            radius = 0.025 + 0.075 * alpha
            perturbation = radius * (2.0 * point[1:] - 1.0)
            candidates.append(np.clip(centre + perturbation, 0.0, 1.0))

    result = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        rounded = tuple(float(round(value, 10)) for value in candidate)
        if rounded in seen:
            continue
        seen.add(rounded)
        result.append(dict(zip(FEATURE_KEYS, rounded)))
    if len(result) != count:
        raise RuntimeError(
            f"Candidate generator produced {len(result)} unique profiles, expected {count}."
        )
    return result


def projection_overrides(state: str, seed: int) -> dict[str, Any]:
    args = SimpleNamespace(
        maxiter=75,
        popsize=8,
        local_maxiter=100,
        de_mutation=0.5,
        de_recombination=0.65,
        seed=seed,
        target_state=state,
        wave_flow_target_weight=1.5,
    )
    overrides = _build_optimiser_overrides(args, "wave")
    overrides.update(
        {
            "maxiter": 75,
            "popsize": 8,
            "n_timing_basis": 6,
            "flow_boundness_target_weight": 1.5,
            "seed": seed,
        }
    )
    return overrides


def candidate_id(
    state: str,
    profile: Mapping[str, float],
    seed: int,
) -> str:
    payload = json.dumps(
        dict(profile), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"v{PROJECTION_VERSION}-{state}-{digest}-seed_{seed}"


def evaluate_profile(
    module: Any,
    *,
    state: str,
    profile: Mapping[str, float],
    original: Mapping[str, float],
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    args = module.build_parser().parse_args([])
    args.gesture = "wave"
    args.target = state
    args.out = str(out_dir / "optimiser_outputs")
    args.skip_output_files = True
    overrides = projection_overrides(state, seed)
    for name, value in overrides.items():
        setattr(args, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        result = module.optimise(args, external_target_profile=dict(profile))

    achieved = {
        key: float(result["var_norm_unclipped"][key])
        for key in FEATURE_KEYS
    }
    errors = {
        key: abs(achieved[key] - float(profile[key])) for key in FEATURE_KEYS
    }
    reward_info = result["reward_info"]
    path_ratio = float(reward_info["path_length_ratio"])
    joint_limit_error = float(reward_info["joint_limit_error"])
    finite = bool(np.all(np.isfinite(list(achieved.values()))))
    physical = bool(
        0.70 <= path_ratio <= 1.30 and joint_limit_error <= 1e-8
    )
    return {
        "projection_version": PROJECTION_VERSION,
        "state": state,
        "seed": seed,
        "original_affect_target": dict(original),
        "candidate_profile": dict(profile),
        "achieved_profile": achieved,
        "equal_weight_distance_to_original": profile_distance(
            profile, original
        ),
        "per_feature_abs_error": errors,
        "rmse": float(np.sqrt(np.mean(np.square(list(errors.values()))))),
        "max_abs_feature_error": float(max(errors.values())),
        "max_error_feature": max(errors, key=errors.get),
        "valid_features": finite,
        "path_preserved": 0.70 <= path_ratio <= 1.30,
        "joint_limits_satisfied": joint_limit_error <= 1e-8,
        "physically_acceptable": physical,
        "strictly_feasible": bool(
            finite and physical and max(errors.values()) <= TOLERANCE
        ),
        "path_length_ratio": path_ratio,
        "nearest_path_mse": float(reward_info["nearest_path_mse"]),
        "nearest_path_max_dist": float(
            reward_info["nearest_path_max_dist"]
        ),
        "endpoint_error": float(reward_info["endpoint_error"]),
        "direction_error": float(reward_info["direction_error"]),
        "smoothness_error": float(reward_info["smoothness_error"]),
        "joint_limit_error": joint_limit_error,
        "optimizer_overrides": overrides,
    }


def load_or_evaluate(
    module: Any,
    *,
    state: str,
    profile: Mapping[str, float],
    original: Mapping[str, float],
    seed: int,
    cases_dir: Path,
) -> dict[str, Any]:
    identifier = candidate_id(state, profile, seed)
    result_path = cases_dir / identifier / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    result = evaluate_profile(
        module,
        state=state,
        profile=profile,
        original=original,
        seed=seed,
        out_dir=result_path.parent,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_path, result)
    return result


def select_projection(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    robust = [
        item
        for item in candidates
        if item["strictly_feasible_seeds"] == len(VALIDATION_SEEDS)
    ]
    if not robust:
        return None
    return min(
        robust,
        key=lambda item: (
            item["equal_weight_distance_to_original"],
            item["worst_max_abs_feature_error"],
            item["mean_nearest_path_mse"],
        ),
    )


def validate_publication_scope(states: list[str], config_out: str | Path) -> None:
    if Path(config_out).resolve() == PRODUCTION_CONFIG.resolve() and set(states) != set(STATES):
        raise ValueError(
            "The production projection config requires all three states. "
            "Use an alternate --config-out for a partial diagnostic run."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", choices=STATES, default=list(STATES))
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--finalists", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--out", default="outputs/wave_feasible_projection"
    )
    parser.add_argument(
        "--config-out",
        default="configs/wave_feasible_target_projections.json",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    validate_publication_scope(cli.states, cli.config_out)
    if cli.finalists < 1:
        raise SystemExit("--finalists must be at least 1.")
    module = load_optimiser()
    reference = wave_reference_profile(module)
    out_dir = Path(cli.out)
    cases_dir = out_dir / "cases"
    summaries: dict[str, Any] = {}

    for state_index, state in enumerate(cli.states):
        original = TARGET_PROFILES[state]
        profiles = generate_candidates(
            original,
            reference,
            count=cli.candidates,
            seed=cli.seed + state_index,
        )
        screened = []
        for index, profile in enumerate(profiles, start=1):
            print(f"SCREEN {state} {index}/{len(profiles)}", flush=True)
            screened.append(
                load_or_evaluate(
                    module,
                    state=state,
                    profile=profile,
                    original=original,
                    seed=VALIDATION_SEEDS[0],
                    cases_dir=cases_dir,
                )
            )
        feasible = sorted(
            (row for row in screened if row["strictly_feasible"]),
            key=lambda row: (
                row["equal_weight_distance_to_original"],
                row["max_abs_feature_error"],
            ),
        )
        finalists = feasible[: cli.finalists]
        validated = []
        for finalist_index, screened_result in enumerate(finalists, start=1):
            profile = screened_result["candidate_profile"]
            seed_results = [screened_result]
            for seed in VALIDATION_SEEDS[1:]:
                print(
                    f"VALIDATE {state} finalist {finalist_index}/{len(finalists)} "
                    f"seed={seed}",
                    flush=True,
                )
                seed_results.append(
                    load_or_evaluate(
                        module,
                        state=state,
                        profile=profile,
                        original=original,
                        seed=seed,
                        cases_dir=cases_dir,
                    )
                )
            validated.append(
                {
                    "candidate_profile": profile,
                    "equal_weight_distance_to_original": (
                        screened_result["equal_weight_distance_to_original"]
                    ),
                    "strictly_feasible_seeds": sum(
                        row["strictly_feasible"] for row in seed_results
                    ),
                    "worst_max_abs_feature_error": max(
                        row["max_abs_feature_error"] for row in seed_results
                    ),
                    "mean_nearest_path_mse": float(
                        np.mean(
                            [row["nearest_path_mse"] for row in seed_results]
                        )
                    ),
                    "minimum_path_length_ratio": min(
                        row["path_length_ratio"] for row in seed_results
                    ),
                    "maximum_path_length_ratio": max(
                        row["path_length_ratio"] for row in seed_results
                    ),
                    "seed_results": seed_results,
                }
            )
        selected = select_projection(validated)
        summaries[state] = {
            "original_affect_target": dict(original),
            "wave_reference_profile": reference,
            "candidate_count": len(profiles),
            "screen_feasible_count": len(feasible),
            "finalists_validated": len(validated),
            "validation_seeds": list(VALIDATION_SEEDS),
            "distance": {
                "name": "equal_weight_normalized_laban_rmse",
                "feature_weights": {key: 1.0 for key in FEATURE_KEYS},
            },
            "selected": selected,
            "validated_finalists": validated,
        }
        write_json_atomic(out_dir / "projection_summary.json", summaries)

    unresolved = [state for state in cli.states if summaries[state]["selected"] is None]
    if unresolved:
        raise SystemExit(
            "No three-seed feasible projection found for: " + ", ".join(unresolved)
        )
    config = {
        "format_version": 1,
        "gesture": "wave",
        "selection_method": "equal_weight_nearest_three_seed_feasible_profile",
        "screening_candidate_count": cli.candidates,
        "validation_seeds": list(VALIDATION_SEEDS),
        "states": {
            state: {
                "original_affect_target": summaries[state][
                    "original_affect_target"
                ],
                "projected_feasible_target": summaries[state]["selected"][
                    "candidate_profile"
                ],
                "equal_weight_distance_to_original": summaries[state][
                    "selected"
                ]["equal_weight_distance_to_original"],
                "validation": {
                    key: summaries[state]["selected"][key]
                    for key in (
                        "strictly_feasible_seeds",
                        "worst_max_abs_feature_error",
                        "mean_nearest_path_mse",
                        "minimum_path_length_ratio",
                        "maximum_path_length_ratio",
                    )
                },
            }
            for state in cli.states
        },
    }
    config_path = Path(cli.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(config_path, config)
    print(config_path, flush=True)


if __name__ == "__main__":
    main()
