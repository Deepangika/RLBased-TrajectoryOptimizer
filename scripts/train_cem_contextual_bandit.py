"""Cross-Entropy Method (CEM) for contextual-bandit training.

Replaces REINFORCE with elite-guided search:
  - Sample N profiles per round
  - Evaluate each profile K times (averaging for noise reduction)
  - Keep top-K elites
  - Refit Beta distributions to elite percentiles
  - Automatic exploration decay (Beta width shrinks over time)

This connects:
    CEM elite tracking -> optimiser -> variant-only video -> Gemini -> reward -> elite selection
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for candidate in [PROJECT_ROOT, SRC_DIR]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    MockNoisyPerceptualEvaluator,
    PerceptualBanditEnvironment,
)
from laban_rl.affect import VAD_KEYS, VAD_TARGETS
from laban_rl.config import EMOTION_STATES, FEATURE_KEYS, GESTURE_TYPES
from laban_rl.perceptual_bandit.cem import CEMOptimizer
from laban_rl.perceptual_bandit.compatibility import (
    checkpoint_metadata,
    migrate_metadata_only_checkpoint,
    validate_checkpoint,
)
from laban_rl.perceptual_bandit.evaluation_cache import PerceptualObservationCache
from laban_rl.perceptual_bandit.baseline import guard_baseline_output_path
from laban_rl.perceptual_bandit.selection import (
    select_feasible_incumbent,
    strict_realisability,
)
from laban_rl.targets import TARGET_PROFILES

WAVE_PROJECTION_CONFIG = (
    PROJECT_ROOT / "configs" / "wave_feasible_target_projections.json"
)


def _build_optimiser_overrides(args, gesture: str) -> dict:
    """Build optimizer overrides with gesture-specific preservation weights.
    
    Wave gestures require much stricter spatial constraints and higher preservation 
    weights to maintain their characteristic smooth, curved motion without collapsing
    into erratic, chaotic paths.
    """
    overrides = {
        "maxiter": args.maxiter,
        "popsize": args.popsize,
        "local_maxiter": args.local_maxiter,
        "de_mutation": args.de_mutation,
        "de_recombination": args.de_recombination,
        "seed": args.seed,
    }
    
    target_state = getattr(args, "target_state", None)

    # Wave gesture needs VERY strong preservation to maintain flowing structure
    if gesture == "wave":
        overrides.update({
            # Spatial constraints: much tighter for wave
            "max_delta": 0.15,                  # Reduced from 0.35: limit spatial deviation
            "max_end_delta": 0.1,              # Reduced from 0.22: subtle endpoint offset
            "n_spatial_basis": 4,              # Reduced from 6: fewer degrees of freedom
            
            # Preservation weights: very strong
            "nearest_path_weight": 12.0,       # Very strong: preserve wrist trajectory
            "path_length_weight": 6.0,         # Very strong: prevent wave collapse/loops
            "preserve_weight": 0.5,            # Strong: preserve joint structure
            "detour_weight": 4.0,              # Strong: prevent excessive loops
            "max_dev_weight": 6.0,             # Strong: bound spatial deviations
            "time_roughness_weight": 0.1,      # Smooth temporal warping
            "smooth_weight": 0.05,             # Smooth joint trajectories
            
            # Target feature tracking: gentle nudge all dimensions toward sampled target
            "weight_target_weight": 0.1,                # Gentle: nudge weight toward target
            "flow_boundness_target_weight": args.wave_flow_target_weight,
            "shape_arcness_target_weight": 0.05,       # Very gentle: shape_arcness
            "time_target_weight": 0.1,                 # Gentle: prevent time overshooting
        })
        if target_state in {"anger", "disgust", "fear", "sadness"}:
            overrides.update({
                "maxiter": max(int(args.maxiter), 75),
                "popsize": max(int(args.popsize), 8),
                "n_timing_basis": 6,
                "flow_boundness_target_weight": max(
                    float(args.wave_flow_target_weight), 1.5
                ),
            })

    if gesture == "beckon" and target_state in {"fear", "surprise"}:
        overrides.update({
            "maxiter": max(int(args.maxiter), 75),
            "popsize": max(int(args.popsize), 8),
            "n_timing_basis": 5 if target_state == "fear" else 6,
            "time_target_weight": 0.10,
            "flow_boundness_target_weight": 0.10,
        })
        if target_state == "fear":
            overrides["shape_arcness_target_weight"] = 0.06

    if gesture == "celebratory_pump":
        overrides.update({
            # Pump expressivity is predominantly temporal. Extra timing
            # resolution improves Flow without permitting larger spatial
            # departures from the raised-arm pumping path.
            "n_timing_basis": 5,
            "time_scale": 2.0,
            "flow_weight": 3.0,
        })
    
    return overrides


def load_projected_initial_profile(
    gesture: str,
    target_state: str | None,
    *,
    path: Path = WAVE_PROJECTION_CONFIG,
) -> tuple[dict[str, float] | None, dict | None]:
    if gesture != "wave" or target_state not in {"anger", "disgust", "sadness"}:
        return None, None
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or payload.get("gesture") != "wave":
        raise RuntimeError(f"Wave projection config is incompatible: {path}")
    state_payload = payload.get("states", {}).get(target_state)
    if not isinstance(state_payload, dict):
        raise RuntimeError(
            f"Wave projection config has no state {target_state!r}: {path}"
        )
    original = state_payload.get("original_affect_target")
    if original != TARGET_PROFILES[target_state]:
        raise RuntimeError(
            f"Wave projection original target for {target_state!r} does not "
            "match the current affect-derived target."
        )
    projected = state_payload.get("projected_feasible_target")
    if not isinstance(projected, dict) or set(projected) != set(FEATURE_KEYS):
        raise RuntimeError(
            f"Wave projected target for {target_state!r} has invalid features."
        )
    profile = {key: float(projected[key]) for key in FEATURE_KEYS}
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in profile.values()):
        raise RuntimeError(
            f"Wave projected target for {target_state!r} is outside [0, 1]."
        )
    validation = state_payload.get("validation", {})
    if (
        validation.get("strictly_feasible_seeds") != 3
        or float(validation.get("worst_max_abs_feature_error", np.inf)) > 0.10
        or float(validation.get("minimum_path_length_ratio", 0.0)) < 0.70
        or float(validation.get("maximum_path_length_ratio", np.inf)) > 1.30
    ):
        raise RuntimeError(
            f"Wave projected target for {target_state!r} lacks strict "
            "three-seed feasibility evidence."
        )
    return profile, {
        "source": "projected_feasible_wave_target",
        "config_path": str(path),
        "original_affect_target": dict(original),
        "projected_feasible_target": dict(profile),
        "equal_weight_distance_to_original": float(
            state_payload["equal_weight_distance_to_original"]
        ),
        "validation": dict(validation),
    }


# All 12 gesture-state profiles for informed initialization.
INFORMED_PROFILES = {
    # Wave gesture: flowing, curved base movement
    "wave::confident": {
        "weight": 0.65,
        "time": 0.70,
        "flow_boundness": 0.35,
        "space_indirectness": 0.45,
        "shape_arcness": 0.65,
    },
    "wave::calm": {
        "weight": 0.30,
        "time": 0.25,
        "flow_boundness": 0.25,
        "space_indirectness": 0.50,
        "shape_arcness": 0.35,
    },
    "wave::hesitant": {
        "weight": 0.25,
        "time": 0.35,
        "flow_boundness": 0.40,
        "space_indirectness": 0.60,
        "shape_arcness": 0.30,
    },
    "wave::friendly": {
        "weight": 0.42,
        "time": 0.40,
        "flow_boundness": 0.20,
        "space_indirectness": 0.30,
        "shape_arcness": 0.60,
    },
    "wave::happy": {
        "weight": 0.50,
        "time": 0.55,
        "flow_boundness": 0.20,
        "space_indirectness": 0.30,
        "shape_arcness": 0.70,
    },
    "wave::sad": {
        "weight": 0.12,
        "time": 0.15,
        "flow_boundness": 0.35,
        "space_indirectness": 0.25,
        "shape_arcness": 0.25,
    },
    "wave::confused": {
        "weight": 0.20,
        "time": 0.30,
        "flow_boundness": 0.15,
        "space_indirectness": 0.75,
        "shape_arcness": 0.50,
    },
    "wave::angry": {
        "weight": 0.85,
        "time": 0.80,
        "flow_boundness": 0.85,
        "space_indirectness": 0.20,
        "shape_arcness": 0.30,
    },
    "wave::fearful": {
        "weight": 0.35,
        "time": 0.80,
        "flow_boundness": 0.75,
        "space_indirectness": 0.65,
        "shape_arcness": 0.25,
    },
    # Reach gesture: extending outward base movement
    "reach::confident": {
        "weight": 0.55,
        "time": 0.70,
        "flow_boundness": 0.30,
        "space_indirectness": 0.15,
        "shape_arcness": 0.45,
    },
    "reach::calm": {
        "weight": 0.30,
        "time": 0.30,
        "flow_boundness": 0.25,
        "space_indirectness": 0.20,
        "shape_arcness": 0.30,
    },
    "reach::hesitant": {
        "weight": 0.25,
        "time": 0.35,
        "flow_boundness": 0.40,
        "space_indirectness": 0.35,
        "shape_arcness": 0.35,
    },
    "reach::friendly": {
        "weight": 0.45,
        "time": 0.50,
        "flow_boundness": 0.35,
        "space_indirectness": 0.30,
        "shape_arcness": 0.55,
    },
    "reach::happy": {
        "weight": 0.50,
        "time": 0.55,
        "flow_boundness": 0.20,
        "space_indirectness": 0.30,
        "shape_arcness": 0.70,
    },
    "reach::sad": {
        "weight": 0.12,
        "time": 0.15,
        "flow_boundness": 0.35,
        "space_indirectness": 0.25,
        "shape_arcness": 0.25,
    },
    "reach::confused": {
        "weight": 0.20,
        "time": 0.35,
        "flow_boundness": 0.15,
        "space_indirectness": 0.70,
        "shape_arcness": 0.45,
    },
    "reach::angry": {
        "weight": 0.85,
        "time": 0.85,
        "flow_boundness": 0.85,
        "space_indirectness": 0.10,
        "shape_arcness": 0.20,
    },
    "reach::fearful": {
        "weight": 0.35,
        "time": 0.80,
        "flow_boundness": 0.75,
        "space_indirectness": 0.65,
        "shape_arcness": 0.25,
    },
    # Point gesture: direct, linear base movement
    "point::confident": {
        "weight": 0.40,
        "time": 0.70,
        "flow_boundness": 0.25,
        "space_indirectness": 0.05,
        "shape_arcness": 0.45,
    },
    "point::calm": {
        "weight": 0.25,
        "time": 0.30,
        "flow_boundness": 0.20,
        "space_indirectness": 0.03,
        "shape_arcness": 0.25,
    },
    "point::hesitant": {
        "weight": 0.20,
        "time": 0.35,
        "flow_boundness": 0.35,
        "space_indirectness": 0.15,
        "shape_arcness": 0.30,
    },
    "point::friendly": {
        "weight": 0.35,
        "time": 0.50,
        "flow_boundness": 0.30,
        "space_indirectness": 0.10,
        "shape_arcness": 0.40,
    },
    "point::happy": {
        "weight": 0.50,
        "time": 0.55,
        "flow_boundness": 0.20,
        "space_indirectness": 0.30,
        "shape_arcness": 0.70,
    },
    "point::sad": {
        "weight": 0.12,
        "time": 0.15,
        "flow_boundness": 0.35,
        "space_indirectness": 0.25,
        "shape_arcness": 0.25,
    },
    "point::confused": {
        "weight": 0.15,
        "time": 0.30,
        "flow_boundness": 0.10,
        "space_indirectness": 0.55,
        "shape_arcness": 0.40,
    },
    "point::angry": {
        "weight": 0.80,
        "time": 0.85,
        "flow_boundness": 0.85,
        "space_indirectness": 0.05,
        "shape_arcness": 0.15,
    },
    "point::fearful": {
        "weight": 0.35,
        "time": 0.80,
        "flow_boundness": 0.75,
        "space_indirectness": 0.65,
        "shape_arcness": 0.25,
    },
}

# Map the previous internal labels to Ekman's six classes.
_EKMAN_ALIAS_MAP = {
    "anger": "angry",
    "fear": "fearful",
    "happiness": "happy",
    "sadness": "sad",
}

for _gesture in ("wave", "reach", "point"):
    for _ekman_state, _legacy_state in _EKMAN_ALIAS_MAP.items():
        INFORMED_PROFILES[f"{_gesture}::{_ekman_state}"] = dict(
            INFORMED_PROFILES[f"{_gesture}::{_legacy_state}"]
        )

# Add dedicated priors for Ekman-specific classes that did not exist before.
INFORMED_PROFILES.update(
    {
        "wave::disgust": {
            "weight": 0.45,
            "time": 0.40,
            "flow_boundness": 0.70,
            "space_indirectness": 0.18,
            "shape_arcness": 0.20,
        },
        "reach::disgust": {
            "weight": 0.50,
            "time": 0.45,
            "flow_boundness": 0.75,
            "space_indirectness": 0.12,
            "shape_arcness": 0.18,
        },
        "point::disgust": {
            "weight": 0.55,
            "time": 0.50,
            "flow_boundness": 0.80,
            "space_indirectness": 0.08,
            "shape_arcness": 0.12,
        },
        "wave::surprise": {
            "weight": 0.60,
            "time": 0.90,
            "flow_boundness": 0.25,
            "space_indirectness": 0.55,
            "shape_arcness": 0.70,
        },
        "reach::surprise": {
            "weight": 0.65,
            "time": 0.92,
            "flow_boundness": 0.30,
            "space_indirectness": 0.45,
            "shape_arcness": 0.55,
        },
        "point::surprise": {
            "weight": 0.70,
            "time": 0.95,
            "flow_boundness": 0.30,
            "space_indirectness": 0.25,
            "shape_arcness": 0.40,
        },
    }
)

# New gestures inherit a semantically similar prior until enough evaluations
# are available to establish gesture-specific initialization profiles.
for _gesture, _source in {
    "circle": "wave",
    "beckon": "reach",
    "celebratory_pump": "point",
}.items():
    for _state in EMOTION_STATES:
        INFORMED_PROFILES[f"{_gesture}::{_state}"] = dict(
            INFORMED_PROFILES[f"{_source}::{_state}"]
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gesture",
        choices=GESTURE_TYPES,
        required=True,
    )
    parser.add_argument(
        "--target-state",
        choices=list(EMOTION_STATES),
        help="Named affect target resolved through its configured VAD anchor.",
    )
    parser.add_argument("--target-valence", type=float)
    parser.add_argument("--target-arousal", type=float)
    parser.add_argument("--target-dominance", type=float)
    parser.add_argument("--rounds", type=int, default=15)

    parser.add_argument(
        "--cem-samples-per-round",
        type=int,
        default=10,
        help="Number of profiles to sample per round.",
    )
    parser.add_argument(
        "--cem-elite-fraction",
        type=float,
        default=0.4,
        help="Fraction of samples to keep as elites (e.g., 0.5 = top 50%%).",
    )
    parser.add_argument(
        "--cem-initial-width",
        type=float,
        default=0.15,
        help="Initial exploration width (std dev of Beta distributions).",
    )
    parser.add_argument(
        "--exploration-decay-rate",
        type=float,
        default=0.98,
        help="Decay the learned elite standard deviation after each update.",
    )
    parser.add_argument("--cem-smoothing", type=float, default=0.7)
    parser.add_argument("--cem-min-std", type=float, default=0.03)
    parser.add_argument("--cem-min-elites", type=int, default=3)

    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--validation-repeats", type=int, default=10)
    parser.add_argument("--validation-top-k", type=int, default=3)
    parser.add_argument("--evaluator-max-attempts", type=int, default=3)
    parser.add_argument("--evaluator-retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--evaluator", choices=["gemini", "mock"], default="gemini")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--mock-noise-std", type=float, default=0.08)
    parser.add_argument("--mock-distance-scale", type=float, default=8.0)

    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--local-maxiter", type=int, default=100)
    parser.add_argument("--de-mutation", type=float, default=0.5)
    parser.add_argument("--de-recombination", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument(
        "--realisation-penalty-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument("--max-feature-error-threshold", type=float, default=0.10)
    parser.add_argument("--max-feature-error-penalty-weight", type=float, default=0.50)
    parser.add_argument(
        "--reject-excessive-feature-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject candidates with any feature error above tolerance before VLM evaluation.",
    )
    parser.add_argument("--wave-flow-target-weight", type=float, default=0.35)
    parser.add_argument(
        "--stability-penalty-weight",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--perceptual-reward-mode",
        choices=["vad", "categorical"],
        default="vad",
        help="Primary perceptual objective. VAD is recommended; categorical reproduces legacy runs.",
    )
    parser.add_argument("--valence-weight", type=float, default=0.20)
    parser.add_argument("--arousal-weight", type=float, default=0.40)
    parser.add_argument("--dominance-weight", type=float, default=0.40)
    parser.add_argument(
        "--reward-margin-mode",
        choices=["raw", "clipped"],
        default="raw",
        help="'raw': use actual margin; 'clipped': max(0, margin).",
    )

    parser.add_argument(
        "--allow-default-profile",
        action="store_true",
        help="Allow missing informed profile to default to [0.5, 0.5, 0.5, 0.5, 0.5]. Default: error.",
    )

    parser.add_argument("--out", required=True)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folder if it is non-empty. Default: error.",
    )
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly resume a compatible checkpoint in the output folder.",
    )
    parser.add_argument(
        "--overwrite-baseline",
        action="store_true",
        help=(
            "Developer-only: permit writing inside the protected fixed-profile "
            "baseline directory."
        ),
    )
    args = parser.parse_args(argv)
    explicit_values = (
        args.target_valence,
        args.target_arousal,
        args.target_dominance,
    )
    has_any_explicit = any(value is not None for value in explicit_values)
    has_all_explicit = all(value is not None for value in explicit_values)
    if args.target_state is not None and has_any_explicit:
        parser.error(
            "--target-state cannot be combined with explicit "
            "--target-valence/--target-arousal/--target-dominance."
        )
    if args.target_state is None and not has_all_explicit:
        parser.error(
            "specify either --target-state or all three explicit VAD options: "
            "--target-valence, --target-arousal, and --target-dominance."
        )
    if has_any_explicit and args.perceptual_reward_mode == "categorical":
        parser.error(
            "explicit VAD targets require --perceptual-reward-mode vad; "
            "categorical mode requires --target-state."
        )
    try:
        context_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def context_from_args(args: argparse.Namespace) -> Context:
    if args.target_state is not None:
        return Context(gesture=args.gesture, target_state=args.target_state)
    return Context(
        gesture=args.gesture,
        target_vad={
            "valence": args.target_valence,
            "arousal": args.target_arousal,
            "dominance": args.target_dominance,
        },
    )


def nearest_vad_anchor(context: Context) -> str:
    """Choose an informed-profile prior without changing the reward target."""
    if context.target_state is not None:
        return context.target_state
    assert context.target_vad is not None
    return min(
        VAD_TARGETS,
        key=lambda state: sum(
            (context.target_vad[axis] - VAD_TARGETS[state][axis]) ** 2
            for axis in VAD_KEYS
        ),
    )


def target_history_metadata(context: Context) -> dict:
    assert context.target_vad is not None
    return {
        "target_mode": context.target_mode,
        "target_state": context.target_state,
        "target_valence": context.target_vad["valence"],
        "target_arousal": context.target_vad["arousal"],
        "target_dominance": context.target_vad["dominance"],
    }


def build_resume_config(args: argparse.Namespace) -> dict:
    """Capture settings that must remain stable across a resumed experiment."""
    projected_profile, projection_metadata = load_projected_initial_profile(
        args.gesture, args.target_state
    )
    return {
        "cem": {
            "samples_per_round": args.cem_samples_per_round,
            "elite_fraction": args.cem_elite_fraction,
            "initial_width": args.cem_initial_width,
            "exploration_decay_rate": args.exploration_decay_rate,
            "smoothing": args.cem_smoothing,
            "min_std": args.cem_min_std,
            "min_elites": args.cem_min_elites,
        },
        "evaluation": {
            "evaluator": args.evaluator,
            "model": args.model if args.evaluator == "gemini" else None,
            "temperature": args.temperature if args.evaluator == "gemini" else None,
            "mock_noise_std": (
                args.mock_noise_std if args.evaluator == "mock" else None
            ),
            "mock_distance_scale": (
                args.mock_distance_scale if args.evaluator == "mock" else None
            ),
            "training_repeats": args.repeats,
            "validation_repeats": args.validation_repeats,
            "validation_top_k": args.validation_top_k,
        },
        "inner_optimizer": {
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "local_maxiter": args.local_maxiter,
            "de_mutation": args.de_mutation,
            "de_recombination": args.de_recombination,
            "wave_flow_target_weight": args.wave_flow_target_weight,
            "seed": args.seed,
            "effective_overrides": _build_optimiser_overrides(
                args, args.gesture
            ),
        },
        "reward": {
            "perceptual_reward_mode": args.perceptual_reward_mode,
            "valence_weight": args.valence_weight,
            "arousal_weight": args.arousal_weight,
            "dominance_weight": args.dominance_weight,
            "reward_margin_mode": args.reward_margin_mode,
            "realisation_penalty_weight": args.realisation_penalty_weight,
            "max_feature_error_threshold": args.max_feature_error_threshold,
            "max_feature_error_penalty_weight": (
                args.max_feature_error_penalty_weight
            ),
            "reject_excessive_feature_error": args.reject_excessive_feature_error,
            "stability_penalty_weight": args.stability_penalty_weight,
        },
        "initialization": {
            "projected_profile": projected_profile,
            "projection_metadata": projection_metadata,
        },
    }


def validate_resume_config(saved_config: dict | None, current_config: dict) -> None:
    if saved_config is None:
        raise RuntimeError(
            "Checkpoint predates complete experiment metadata and cannot be "
            "resumed safely. Use --overwrite to start a new experiment."
        )
    if saved_config != current_config:
        raise RuntimeError(
            f"Checkpoint experiment config {saved_config!r} does not match "
            f"the current config {current_config!r}. "
            "Use matching arguments or --overwrite."
        )


def validate_checkpoint_context(
    saved_context: dict | None,
    current_context: Context,
) -> None:
    if saved_context is None:
        raise RuntimeError(
            "Checkpoint has no saved target context and cannot be resumed safely. "
            "Use --overwrite to start a new experiment."
        )
    try:
        restored_context = Context.from_dict(saved_context)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Checkpoint target context is invalid: {saved_context!r}. "
            "Use --overwrite to start a new experiment."
        ) from error
    if restored_context.to_dict() != current_context.to_dict():
        raise RuntimeError(
            f"Checkpoint context {restored_context.to_dict()!r} does not match "
            f"the current context {current_context.to_dict()!r}. "
            "Use --overwrite to start a new experiment."
        )


def safe_output_folder(
    out_dir: Path,
    overwrite: bool,
    resume: bool = False,
    *,
    overwrite_baseline: bool = False,
) -> bool:
    """
    Ensure output folder is safe. Returns True if resuming from checkpoint.
    
    --overwrite always takes priority: if set, the folder is wiped and a fresh
    run begins, even when a checkpoint is present.  Without --overwrite, if a
    --resume is required to continue a checkpoint; otherwise an error is raised.
    """
    guard_baseline_output_path(
        out_dir,
        overwrite_baseline=overwrite_baseline,
    )
    checkpoint_path = out_dir / "latest_checkpoint.pt"
    has_checkpoint = checkpoint_path.exists()
    
    if out_dir.exists() and list(out_dir.iterdir()):
        if overwrite:
            # Explicit restart requested; remove everything and start fresh.
            import shutil
            shutil.rmtree(out_dir)
        elif resume:
            if has_checkpoint:
                print(f"  Resuming from checkpoint: {checkpoint_path}")
                return True
            recovery_metadata = (
                out_dir / "target_context.json",
                out_dir / "resume_config.json",
            )
            if not all(path.exists() for path in recovery_metadata):
                raise RuntimeError(
                    f"Output folder {out_dir} is not a recoverable run. Use "
                    "--overwrite for a fresh run or choose a different path."
                )
            print(
                "  Recovering a pre-checkpoint run from deterministic state "
                "and cached perceptual repeats."
            )
            return False
        else:
            raise RuntimeError(
                f"Output folder {out_dir} is non-empty. Use --resume for a "
                "compatible checkpoint, --overwrite for a fresh run, or choose "
                "a different output path."
            )
    
    out_dir.mkdir(parents=True, exist_ok=True)
    return False


def update_cem_from_feasible_candidates(
    cem: CEMOptimizer,
    feasible_candidates: list[tuple[dict[str, float], float]],
    *,
    elite_fraction: float,
    exploration_decay_rate: float,
    feasible_improved: bool,
) -> bool:
    """Update CEM only from a sufficiently large strictly feasible batch."""
    if len(feasible_candidates) < cem.min_elites:
        return False
    cem.update_elites(feasible_candidates, elite_fraction=elite_fraction)
    if feasible_improved:
        cem.decay_exploration(exploration_decay_rate)
    return True


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load training state from checkpoint."""
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    checkpoint, migrated = migrate_metadata_only_checkpoint(checkpoint)
    validate_checkpoint(
        checkpoint,
        expected_kind="cem",
        context=checkpoint.get("context"),
    )
    if migrated:
        print(
            "  Loaded eligible legacy CEM checkpoint with metadata migrated "
            "in memory; use the migration CLI to persist a converted copy."
        )
    print(f"  Loaded checkpoint from round {checkpoint['round']}")
    return checkpoint


def save_checkpoint_atomic(checkpoint: dict, checkpoint_path: Path) -> None:
    """Persist a checkpoint without exposing a partially written latest file."""
    temporary_path = checkpoint_path.with_suffix(
        f"{checkpoint_path.suffix}.tmp"
    )
    with temporary_path.open("wb") as handle:
        pickle.dump(checkpoint, handle)
    temporary_path.replace(checkpoint_path)


def load_history_csv(csv_path: Path) -> list[dict]:
    """Load existing training history from CSV."""
    if not csv_path.exists():
        return []
    
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Convert numeric fields back to float
    numeric_fields = {
        "round", "best_round_reward", "num_samples_evaluated", "mean_sample_reward",
        "max_sample_reward", "min_sample_reward", "mean_target_probability",
        "mean_realisation_rmse", "mean_max_abs_feature_error",
        "feature_realisation_acceptance_rate", "mean_target_classification_rate",
        "mean_winner_agreement_rate", "mean_probability_entropy",
        "mean_exploration_std", "min_exploration_std", "max_exploration_std",
        "log_search_volume", "elite_reward_std", "mean_margin", "max_margin",
        "max_target_probability", "physical_acceptance_rate", "num_elites",
        "invalid_or_infeasible_samples",
        "mean_vad_reward", "mean_vad_error", "mean_vad_reward_std",
        "mean_observed_valence", "mean_observed_arousal",
        "mean_observed_dominance", "target_valence", "target_arousal",
        "target_dominance",
    }
    for row in rows:
        for field in numeric_fields:
            if field in row and row[field]:
                try:
                    row[field] = float(row[field])
                except (ValueError, TypeError):
                    pass
    
    return rows


def save_history_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(dict.fromkeys(
                key for row in rows for key in row
            )),
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plots(rows: list[dict], out_dir: Path) -> None:
    if not rows:
        return

    rounds = [row["round"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, [row["best_round_reward"] for row in rows], marker="o", label="Best found", linewidth=2)
    plt.plot(rounds, [row["mean_sample_reward"] for row in rows], marker=".", label="Mean sample", alpha=0.7)
    plt.xlabel("Round")
    plt.ylabel("Reward")
    plt.title("CEM: Best elite vs mean sample per round")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_curve.png", dpi=160)
    plt.close()

    if any("mean_vad_reward" in row for row in rows):
        plt.figure(figsize=(8, 5))
        plt.plot(
            rounds,
            [row.get("mean_vad_reward", float("nan")) for row in rows],
            marker="o",
            label="Mean VAD reward",
        )
        plt.plot(
            rounds,
            [row.get("mean_vad_error", float("nan")) for row in rows],
            marker=".",
            label="Mean weighted VAD error",
        )
        plt.xlabel("Round")
        plt.ylabel("Score")
        plt.title("VAD perceptual objective")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "vad_reward_curve.png", dpi=160)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        rounds,
        [row["mean_margin"] for row in rows],
        marker="o",
        label="Mean raw margin",
    )
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Round")
    plt.ylabel("Target minus strongest competitor")
    plt.title("Perceptual classification margin")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "margin_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        rounds,
        [row["mean_exploration_std"] for row in rows],
        marker="o",
        label="Mean sampling std",
    )
    plt.plot(
        rounds,
        [row["max_exploration_std"] for row in rows],
        marker=".",
        label="Maximum sampling std",
    )
    plt.xlabel("Round")
    plt.ylabel("CEM distribution standard deviation")
    plt.title("CEM exploration and distribution contraction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cem_diversity_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        rounds,
        [row["mean_target_probability"] for row in rows],
        marker="o",
    )
    plt.xlabel("Round")
    plt.ylabel("Mean target probability")
    plt.title("Gemini target-state probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        out_dir / "target_probability_curve.png",
        dpi=160,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        rounds,
        [row["mean_realisation_rmse"] for row in rows],
        marker="o",
    )
    plt.xlabel("Round")
    plt.ylabel("Mean realisation RMSE")
    plt.title("Requested-to-achieved profile error")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        out_dir / "realisation_rmse_curve.png",
        dpi=160,
    )
    plt.close()

    for feature in FEATURE_KEYS:
        plt.figure(figsize=(8, 5))
        plt.plot(
            rounds,
            [row[f"mean_{feature}"] for row in rows],
            marker="o",
            label="Distribution mean",
        )
        plt.plot(
            rounds,
            [row[f"best_elite_{feature}"] for row in rows],
            marker="*",
            label="Best elite",
            markersize=10,
        )
        plt.ylim(0.0, 1.0)
        plt.xlabel("Round")
        plt.ylabel("Normalised value")
        plt.title(f"CEM distribution: {feature}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / f"policy_{feature}_curve.png",
            dpi=160,
        )
        plt.close()


def main() -> None:
    args = parse_args()
    environment_context = context_from_args(args)

    # Note: CEMOptimizer and MockNoisyPerceptualEvaluator use
    # np.random.default_rng(seed) internally, which is seeded through their
    # constructors.  The legacy np.random.seed() has no effect on default_rng
    # instances, so it is not called here.

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    # Check if we're resuming from checkpoint
    is_resuming = safe_output_folder(
        out_dir,
        args.overwrite,
        args.resume,
        overwrite_baseline=args.overwrite_baseline,
    )

    # Load checkpoint if resuming
    checkpoint_data = None
    if is_resuming:
        checkpoint_path = out_dir / "latest_checkpoint.pt"
        checkpoint_data = load_checkpoint(checkpoint_path)
        # Verify the checkpoint belongs to the same experiment context so a
        # checkpoint from one run cannot silently continue as a different one.
        validate_checkpoint_context(
            checkpoint_data.get("context"),
            environment_context,
        )
        validate_resume_config(
            checkpoint_data.get("resume_config"),
            build_resume_config(args),
        )
        print(f"  Resuming from round {checkpoint_data['round']} of {args.rounds}")

    target_context_metadata = environment_context.to_dict()
    target_context_path = out_dir / "target_context.json"
    resume_config = build_resume_config(args)
    resume_config_path = out_dir / "resume_config.json"
    if args.resume and not is_resuming and target_context_path.exists():
        saved_context = json.loads(target_context_path.read_text(encoding="utf-8"))
        validate_checkpoint_context(saved_context, environment_context)
        if not resume_config_path.exists():
            raise RuntimeError(
                "Pre-checkpoint run has no complete resume configuration and "
                "cannot be recovered safely. Use --overwrite to start again."
            )
        saved_resume_config = json.loads(
            resume_config_path.read_text(encoding="utf-8")
        )
        validate_resume_config(saved_resume_config, resume_config)
    target_context_path.write_text(
        json.dumps(target_context_metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    resume_config_path.write_text(
        json.dumps(resume_config, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # Get informed profile or raise error if missing and not allowed.
    context_key = environment_context.key
    initialization_anchor_state = nearest_vad_anchor(environment_context)
    initialization_context_key = (
        f"{args.gesture}::{initialization_anchor_state}"
    )
    initial_profile = INFORMED_PROFILES.get(initialization_context_key)
    initialization_profile_metadata = {
        "source": "informed_named_anchor",
        "anchor_state": initialization_anchor_state,
        "profile": dict(initial_profile) if initial_profile is not None else None,
    }
    projected_profile, projection_metadata = load_projected_initial_profile(
        args.gesture, environment_context.target_state
    )
    if projected_profile is not None:
        initial_profile = projected_profile
        initialization_profile_metadata = dict(projection_metadata or {})

    if initial_profile is None:
        if not args.allow_default_profile:
            raise ValueError(
                f"No informed profile for context {initialization_context_key!r}. "
                "Either add the profile to INFORMED_PROFILES, or use --allow-default-profile."
            )
        initial_profile = None

    (out_dir / "initialization_profile.json").write_text(
        json.dumps(
            initialization_profile_metadata,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    cem = CEMOptimizer(
        initial_profile=initial_profile,
        initial_width=args.cem_initial_width,
        seed=args.seed,
        smoothing=args.cem_smoothing,
        min_std=args.cem_min_std,
        min_elites=args.cem_min_elites,
    )

    # If resuming, restore CEM state from checkpoint
    if checkpoint_data:
        if "cem_state" not in checkpoint_data:
            raise RuntimeError(
                "This checkpoint was produced by the legacy CEM implementation and "
                "cannot be resumed safely. Start a new output directory."
            )
        cem.load_state_dict(checkpoint_data["cem_state"])

    if args.evaluator == "gemini":
        from laban_rl.perceptual_bandit.gemini_evaluator import GeminiProVideoEvaluator
        evaluator = GeminiProVideoEvaluator(model=args.model, temperature=args.temperature)
    else:
        evaluator = MockNoisyPerceptualEvaluator(
            noise_std=args.mock_noise_std,
            distance_scale=args.mock_distance_scale,
            seed=args.seed,
        )
        if (
            args.target_state is not None
            and args.target_state not in evaluator.state_labels
        ):
            raise ValueError(
                f"Mock evaluator does not support target state "
                f"{args.target_state!r}. Available labels: "
                f"{list(evaluator.state_labels)}"
            )

    training_observation_cache = PerceptualObservationCache(
        out_dir / "cache" / "training"
    )
    validation_observation_cache = PerceptualObservationCache(
        out_dir / "cache" / "validation"
    )
    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=args.repeats,
            perceptual_reward_mode=args.perceptual_reward_mode,
            valence_weight=args.valence_weight,
            arousal_weight=args.arousal_weight,
            dominance_weight=args.dominance_weight,
            reward_margin_mode=args.reward_margin_mode,
            realisation_penalty_weight=(
                args.realisation_penalty_weight
            ),
            max_feature_error_threshold=args.max_feature_error_threshold,
            max_feature_error_penalty_weight=args.max_feature_error_penalty_weight,
            reject_excessive_feature_error=args.reject_excessive_feature_error,
            stability_penalty_weight=(
                args.stability_penalty_weight
            ),
            evaluator_failure_mode="raise",
        ),
        optimiser_overrides=_build_optimiser_overrides(
            args=args,
            gesture=args.gesture,
        ),
        observation_cache=training_observation_cache,
    )
    validation_environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=args.validation_repeats,
            perceptual_reward_mode=args.perceptual_reward_mode,
            valence_weight=args.valence_weight,
            arousal_weight=args.arousal_weight,
            dominance_weight=args.dominance_weight,
            reward_margin_mode=args.reward_margin_mode,
            realisation_penalty_weight=args.realisation_penalty_weight,
            max_feature_error_threshold=args.max_feature_error_threshold,
            max_feature_error_penalty_weight=args.max_feature_error_penalty_weight,
            reject_excessive_feature_error=args.reject_excessive_feature_error,
            stability_penalty_weight=args.stability_penalty_weight,
            evaluator_failure_mode="raise",
        ),
        optimiser_overrides=_build_optimiser_overrides(args=args, gesture=args.gesture),
        observation_cache=validation_observation_cache,
    )

    # Load existing history if resuming
    if is_resuming:
        history = load_history_csv(out_dir / "training_history.csv")
        checkpoint_round = int(checkpoint_data["round"])
        history = [
            row
            for row in history
            if int(float(row["round"])) <= checkpoint_round
        ]
        target_columns = target_history_metadata(environment_context)
        for row in history:
            for key, value in target_columns.items():
                row.setdefault(key, value)
        best_reward = checkpoint_data["best_reward"]
        best_profile = checkpoint_data["best_profile"]
        best_observed_reward = checkpoint_data.get("best_observed_reward")
        best_observed_profile = checkpoint_data.get("best_observed_profile")
        # best_round_index is stored directly in the checkpoint (see save below)
        best_round_index = checkpoint_data.get("best_round_index")
        start_round = checkpoint_round + 1
        print(f"  Loaded history from {len(history)} previous rounds")
        print(f"  Best reward so far: {best_reward:.6f} (from round {best_round_index})")
    else:
        history: list[dict] = []
        best_reward = float("-inf")
        best_profile = None
        best_observed_reward = None
        best_observed_profile = None
        best_round_index = None
        start_round = 1

    validation_results: dict[str, dict] = {}
    validation_path = out_dir / "independent_validation.json"
    if is_resuming and validation_path.exists():
        validation_results = json.loads(validation_path.read_text(encoding="utf-8"))

    def evaluate_with_retry(env, *, context, profile, output_dir):
        """Run inner optimiser once; retry only the evaluation on API failures.

        The expensive inner optimiser (DE/CEM + video rendering) is called
        exactly once.  Only the perceptual evaluation phase (Gemini API call)
        is retried, so a transient 503 does not repeat the full optimisation.
        """
        # Phase 1 – run the inner optimiser once; not retried.
        opt_result = env.run_inner_step(
            context=context,
            action_profile=profile,
            out_dir=output_dir,
        )

        # Phase 2 – retry only the evaluation on transient API failures.
        last_error = None
        for attempt in range(1, args.evaluator_max_attempts + 1):
            try:
                return env.step_from_result(
                    context=context,
                    optimisation_result=opt_result,
                )
            except RuntimeError as error:
                last_error = error
                if attempt >= args.evaluator_max_attempts:
                    break
                delay = args.evaluator_retry_base_seconds * (2 ** (attempt - 1))
                print(
                    f"  Evaluator attempt {attempt} failed; retrying in "
                    f"{delay:.1f}s. The inner optimiser result is reused."
                )
                time.sleep(delay)
        raise RuntimeError(
            f"Evaluator failed after {args.evaluator_max_attempts} attempts. "
            "Training stopped without updating CEM."
        ) from last_error


    def evaluate_validation(name: str, profile: dict[str, float]):
        result = evaluate_with_retry(
            validation_environment,
            context=environment_context,
            profile=profile,
            output_dir=out_dir / "validation" / name / "optimiser_outputs",
        )
        validation_results[name] = result.to_dict()
        validation_path.write_text(
            json.dumps(validation_results, indent=2, allow_nan=False), encoding="utf-8"
        )
        return result

    try:
        # Evaluate the informed initial profile if not already recorded.
        # This also covers a resume after a crash that happened before the
        # initial evaluation completed.
        if "initial_profile" not in validation_results:
            print("\nEvaluating informed initial profile independently...")
            evaluate_validation("initial_profile", dict(initial_profile or cem.get_mean_profile()))

        for round_index in range(start_round, args.rounds + 1):
            print("\n" + "#" * 100)
            print(
                f"CEM ROUND {round_index}/{args.rounds} "
                f"| {context_key}"
            )
            print("#" * 100)

            # Sample batch of profiles
            sampled_profiles = cem.sample_batch(args.cem_samples_per_round)

            # Evaluate each sampled profile
            candidates = []
            valid_candidates = []
            feasible_improved_this_round = False
            round_rewards = []
            round_rmses = []
            round_target_probs = []
            round_margins = []
            round_max_feature_errors = []
            round_feature_acceptance = []
            round_physical_acceptance = []
            round_classification_rates = []
            round_winner_agreements = []
            round_entropies = []
            round_vad_rewards = []
            round_vad_errors = []
            round_vad_reward_stds = []
            round_observed_vad = {key: [] for key in ("valence", "arousal", "dominance")}

            for sample_idx, profile in enumerate(sampled_profiles):
                print(f"\n  Sample {sample_idx + 1}/{args.cem_samples_per_round}: {profile}")

                round_dir = out_dir / "rounds" / f"round_{round_index:03d}_sample_{sample_idx:02d}"

                result = evaluate_with_retry(
                    environment,
                    context=environment_context,
                    profile=profile,
                    output_dir=round_dir / "optimiser_outputs",
                )

                # Defensive check: ensure result has valid reward
                if result.outer_reward is None:
                    raise RuntimeError(
                        f"Environment step returned None for outer_reward at "
                        f"round {round_index}, sample {sample_idx}. "
                        f"This indicates an evaluator failure. Check the evaluator logs."
                    )

                candidates.append((profile, result.outer_reward))
                is_strictly_feasible, _ = strict_realisability(
                    result.to_dict(),
                    tolerance=args.max_feature_error_threshold,
                    required_repeats=args.repeats,
                )
                if is_strictly_feasible:
                    valid_candidates.append((profile, result.outer_reward))

                round_rewards.append(result.outer_reward)
                if result.realisation_rmse is not None:
                    round_rmses.append(result.realisation_rmse)
                if result.mean_target_probability is not None:
                    round_target_probs.append(result.mean_target_probability)
                if result.mean_margin is not None:
                    round_margins.append(result.mean_margin)
                if result.max_abs_feature_error is not None:
                    round_max_feature_errors.append(result.max_abs_feature_error)
                round_feature_acceptance.append(float(result.feature_realisation_acceptable))
                round_physical_acceptance.append(float(result.physically_acceptable))
                if result.target_classification_rate is not None:
                    round_classification_rates.append(result.target_classification_rate)
                if result.winner_agreement_rate is not None:
                    round_winner_agreements.append(result.winner_agreement_rate)
                if result.mean_probability_entropy is not None:
                    round_entropies.append(result.mean_probability_entropy)
                if result.mean_vad_reward is not None:
                    round_vad_rewards.append(result.mean_vad_reward)
                if result.mean_vad_error is not None:
                    round_vad_errors.append(result.mean_vad_error)
                if result.vad_reward_std is not None:
                    round_vad_reward_stds.append(result.vad_reward_std)
                if result.mean_observed_vad is not None:
                    for axis in round_observed_vad:
                        round_observed_vad[axis].append(result.mean_observed_vad[axis])

                print(
                    f"    Reward: {result.outer_reward:.6f}, "
                    f"VAD reward: {result.mean_vad_reward if result.mean_vad_reward is not None else 'N/A'}, "
                    f"Target prob: {result.mean_target_probability if result.mean_target_probability is not None else 'N/A'}, "
                    f"RMSE: {result.realisation_rmse if result.realisation_rmse is not None else 'N/A'}, "
                    f"valid: {result.valid_realisation}"
                )

                if (
                    best_observed_reward is None
                    or result.outer_reward > best_observed_reward
                ):
                    best_observed_reward = float(result.outer_reward)
                    best_observed_profile = dict(profile)
                if is_strictly_feasible and result.outer_reward > best_reward:
                    best_reward = float(result.outer_reward)
                    best_profile = dict(profile)
                    best_round_index = round_index
                    feasible_improved_this_round = True

                # Save per-sample summary (use allow_nan=False for valid JSON)
                (round_dir / "sample_summary.json").write_text(
                    json.dumps(
                        {
                            "sampled_profile": profile,
                            "environment_result": result.to_dict(),
                        },
                        indent=2,
                        allow_nan=False,
                    ),
                    encoding="utf-8",
                )

            cem_updated = update_cem_from_feasible_candidates(
                cem,
                valid_candidates,
                elite_fraction=args.cem_elite_fraction,
                exploration_decay_rate=args.exploration_decay_rate,
                feasible_improved=feasible_improved_this_round,
            )
            if not cem_updated:
                print(
                    "  Skipping CEM update: only "
                    f"{len(valid_candidates)} strictly feasible candidates; "
                    f"{cem.min_elites} required."
                )
            cem_diagnostics = cem.diagnostics()

            # Record round statistics
            best_elite = cem.get_best_elite()
            current_mean = cem.get_mean_profile()
            current_std = cem.get_std_profile()

            row = {
                "round": round_index,
                **target_history_metadata(environment_context),
                # best_round_reward is the best reward seen in THIS round only
                # (not a cumulative maximum).  The cumulative best is stored
                # separately in best_profile.json.
                "best_round_reward": float(np.max(round_rewards)),
                "num_samples_evaluated": len(sampled_profiles),
                "mean_sample_reward": float(np.mean(round_rewards)),
                "max_sample_reward": float(np.max(round_rewards)),
                "min_sample_reward": float(np.min(round_rewards)),
                "mean_target_probability": float(np.mean(round_target_probs)) if round_target_probs else float("nan"),
                "max_target_probability": float(np.max(round_target_probs)) if round_target_probs else float("nan"),
                "mean_margin": float(np.mean(round_margins)) if round_margins else float("nan"),
                "max_margin": float(np.max(round_margins)) if round_margins else float("nan"),
                "mean_realisation_rmse": float(np.mean(round_rmses)) if round_rmses else float("nan"),
                "mean_max_abs_feature_error": float(np.mean(round_max_feature_errors)) if round_max_feature_errors else float("nan"),
                "feature_realisation_acceptance_rate": float(np.mean(round_feature_acceptance)) if round_feature_acceptance else float("nan"),
                "physical_acceptance_rate": float(np.mean(round_physical_acceptance)) if round_physical_acceptance else float("nan"),
                "mean_target_classification_rate": float(np.mean(round_classification_rates)) if round_classification_rates else float("nan"),
                "mean_winner_agreement_rate": float(np.mean(round_winner_agreements)) if round_winner_agreements else float("nan"),
                "mean_probability_entropy": float(np.mean(round_entropies)) if round_entropies else float("nan"),
                "mean_vad_reward": float(np.mean(round_vad_rewards)) if round_vad_rewards else float("nan"),
                "mean_vad_error": float(np.mean(round_vad_errors)) if round_vad_errors else float("nan"),
                "mean_vad_reward_std": float(np.mean(round_vad_reward_stds)) if round_vad_reward_stds else float("nan"),
                "mean_observed_valence": float(np.mean(round_observed_vad["valence"])) if round_observed_vad["valence"] else float("nan"),
                "mean_observed_arousal": float(np.mean(round_observed_vad["arousal"])) if round_observed_vad["arousal"] else float("nan"),
                "mean_observed_dominance": float(np.mean(round_observed_vad["dominance"])) if round_observed_vad["dominance"] else float("nan"),
                "mean_exploration_std": float(np.mean(list(current_std.values()))),
                "min_exploration_std": float(cem_diagnostics["min_profile_std"]),
                "max_exploration_std": float(cem_diagnostics["max_profile_std"]),
                "log_search_volume": float(cem_diagnostics["log_search_volume"]),
                "elite_reward_std": float(cem_diagnostics["elite_reward_std"]),
                "invalid_or_infeasible_samples": int(sum(reward <= -1.0 for reward in round_rewards)),
                "num_elites": len(cem.elites),
            }

            if best_elite:
                row["best_elite_reward"] = best_elite.reward
                for feature in FEATURE_KEYS:
                    row[f"best_elite_{feature}"] = best_elite.profile[feature]
            else:
                row["best_elite_reward"] = float("nan")
                for feature in FEATURE_KEYS:
                    row[f"best_elite_{feature}"] = float("nan")

            for feature in FEATURE_KEYS:
                row[f"mean_{feature}"] = current_mean[feature]
                row[f"std_{feature}"] = current_std[feature]

            history.append(row)

            save_history_csv(history, out_dir / "training_history.csv")
            save_plots(history, out_dir)

            # Save checkpoint
            save_checkpoint_atomic(
                {
                    "metadata": checkpoint_metadata(
                        "cem", context=environment_context.to_dict()
                    ),
                    "round": round_index,
                    "cem_state": cem.state_dict(),
                    "context": environment_context.to_dict(),
                    "resume_config": build_resume_config(args),
                    "perceptual_reward_config": {
                        "mode": args.perceptual_reward_mode,
                        "valence_weight": args.valence_weight,
                        "arousal_weight": args.arousal_weight,
                        "dominance_weight": args.dominance_weight,
                    },
                    "best_reward": best_reward,
                    "best_profile": best_profile,
                    "best_round_index": best_round_index,
                    "best_observed_reward": best_observed_reward,
                    "best_observed_profile": best_observed_profile,
                },
                out_dir / "latest_checkpoint.pt",
            )

            (out_dir / "best_profile.json").write_text(
                json.dumps(
                    {
                        "best_reward": (
                            best_reward if np.isfinite(best_reward) else None
                        ),
                        "best_profile": best_profile,
                        "best_observed_reward": best_observed_reward,
                        "best_observed_profile": best_observed_profile,
                    },
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            print("\nRound summary:")
            print(f"  Mean sample reward:   {np.mean(round_rewards):.6f}")
            print(f"  Best sample reward:   {np.max(round_rewards):.6f}")
            print(f"  Best found so far:    {best_reward:.6f} (round {best_round_index})")
            print(f"  Num elites:           {len(cem.elites)}")
            print(f"  Distribution mean:    {current_mean}")

        final_mean = cem.get_mean_profile()
        print("\nIndependently evaluating final distribution mean...")
        evaluate_validation("final_distribution_mean", final_mean)

        # Re-rank several high-training-reward candidates using fresh repeats.
        # Selecting only the single training winner is vulnerable to evaluator
        # noise (the winner's curse).
        archived_candidates = []
        for sample_path in (out_dir / "rounds").glob("round_*_sample_*/sample_summary.json"):
            payload = json.loads(sample_path.read_text(encoding="utf-8"))
            archived_candidates.append(
                (
                    float(payload["environment_result"]["outer_reward"]),
                    dict(payload["sampled_profile"]),
                    str(sample_path.parent.name),
                )
            )
        archived_candidates.sort(key=lambda item: item[0], reverse=True)
        shortlist = archived_candidates[: max(1, args.validation_top_k)]
        shortlist_results = []
        for rank, (training_reward, profile, source) in enumerate(shortlist, start=1):
            name = f"shortlist_rank_{rank:02d}"
            print(f"\nIndependently evaluating shortlist candidate {rank}/{len(shortlist)}...")
            result = evaluate_validation(name, profile)
            shortlist_results.append(
                {
                    "rank": rank,
                    "source": source,
                    "training_reward": training_reward,
                    "profile": profile,
                    "validation_reward": float(result.outer_reward),
                    "validation_result_key": name,
                }
            )

        selection = select_feasible_incumbent(
            validation_results,
            shortlist_results,
            tolerance=args.max_feature_error_threshold,
            required_repeats=args.validation_repeats,
        )
        selected = selection["selected"]
        validation_results["best_sampled_profile"] = dict(
            validation_results[selected["validation_result_key"]]
        )
        validation_results["selection"] = selection
        validation_path.write_text(
            json.dumps(validation_results, indent=2, allow_nan=False), encoding="utf-8"
        )
        (out_dir / "selected_validated_profile.json").write_text(
            json.dumps(selected, indent=2, allow_nan=False), encoding="utf-8"
        )

    finally:
        close = getattr(evaluator, "close", None)
        if callable(close):
            close()

    # Compute final summaries
    final_mean = cem.get_mean_profile()
    final_std = cem.get_std_profile()

    num_rounds_completed = len(history)
    mean_reward_all = (
        float(np.mean([row["mean_sample_reward"] for row in history]))
        if history
        else 0.0
    )
    mean_reward_last5 = (
        float(
            np.mean(
                [row["mean_sample_reward"] for row in history[-5:]]
            )
        )
        if len(history) >= 5
        else mean_reward_all
    )
    target_probabilities_all = [
        float(row["mean_target_probability"])
        for row in history
        if np.isfinite(float(row["mean_target_probability"]))
    ]
    target_probabilities_last5 = [
        float(row["mean_target_probability"])
        for row in history[-5:]
        if np.isfinite(float(row["mean_target_probability"]))
    ]
    mean_target_prob_all = (
        float(np.mean(target_probabilities_all)) if target_probabilities_all else None
    )
    mean_target_prob_last5 = (
        float(np.mean(target_probabilities_last5))
        if target_probabilities_last5 else mean_target_prob_all
    )
    successful_training_vlm_evaluations = 0
    for sample_path in (out_dir / "rounds").glob("round_*_sample_*/sample_summary.json"):
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        successful_training_vlm_evaluations += len(
            payload["environment_result"].get("affective_evaluations") or []
        )
    successful_validation_vlm_evaluations = sum(
        len(result.get("affective_evaluations") or [])
        for name, result in validation_results.items()
        if name not in {"best_sampled_profile", "selection"}
    )

    results_summary = {
        "gesture": args.gesture,
        "target_mode": environment_context.target_mode,
        "target_state": environment_context.target_state,
        "target_vad": dict(environment_context.target_vad),
        "target": environment_context.to_dict(),
        "initialization_anchor_state": initialization_anchor_state,
        "initialization_profile": initialization_profile_metadata,
        "num_rounds_completed": num_rounds_completed,
        "samples_per_round": args.cem_samples_per_round,
        "repeats_per_round": args.repeats,
        "planned_training_vlm_evaluations": num_rounds_completed * args.cem_samples_per_round * args.repeats,
        "successful_training_vlm_evaluations": successful_training_vlm_evaluations,
        "successful_validation_vlm_evaluations": successful_validation_vlm_evaluations,
        "successful_vlm_evaluations": (
            successful_training_vlm_evaluations
            + successful_validation_vlm_evaluations
        ),
        "planned_training_evaluator_calls": num_rounds_completed * args.cem_samples_per_round * args.repeats,
        "successful_training_evaluator_calls": successful_training_vlm_evaluations,
        "successful_validation_evaluator_calls": successful_validation_vlm_evaluations,
        "successful_evaluator_calls": (
            successful_training_vlm_evaluations
            + successful_validation_vlm_evaluations
        ),
        "best_round_index": best_round_index,
        "best_sampled_profile": best_profile,
        "best_reward": best_reward if np.isfinite(best_reward) else None,
        "best_observed_profile": best_observed_profile,
        "best_observed_reward": best_observed_reward,
        "final_distribution_mean": final_mean,
        "final_distribution_std": final_std,
        "mean_reward_all_rounds": mean_reward_all,
        "mean_reward_last_5_rounds": mean_reward_last5,
        "mean_target_probability_all_rounds": mean_target_prob_all,
        "mean_target_probability_last_5_rounds": mean_target_prob_last5,
        "initial_profile_validation": validation_results.get("initial_profile"),
        "final_distribution_mean_validation": validation_results.get("final_distribution_mean"),
        "best_sampled_profile_validation": validation_results.get("best_sampled_profile"),
        "independent_selection": validation_results.get("selection"),
        "num_elites_at_end": len(cem.elites),
        "reward_margin_mode": args.reward_margin_mode,
        "perceptual_reward_mode": args.perceptual_reward_mode,
        "vad_weights": {
            "valence": args.valence_weight,
            "arousal": args.arousal_weight,
            "dominance": args.dominance_weight,
        },
        "cem_elite_fraction": args.cem_elite_fraction,
        "cem_initial_width": args.cem_initial_width,
        "cem_smoothing": args.cem_smoothing,
        "cem_min_std": args.cem_min_std,
        "cem_min_elites": args.cem_min_elites,
        "exploration_decay_rate": args.exploration_decay_rate,
        "training_repeats": args.repeats,
        "validation_repeats": args.validation_repeats,
        "model": args.model if args.evaluator == "gemini" else None,
        "evaluator": args.evaluator,
        "mock_noise_std": args.mock_noise_std if args.evaluator == "mock" else None,
        "mock_distance_scale": args.mock_distance_scale if args.evaluator == "mock" else None,
        "temperature": args.temperature,
        "realisation_penalty_weight": args.realisation_penalty_weight,
        "max_feature_error_threshold": args.max_feature_error_threshold,
        "max_feature_error_penalty_weight": args.max_feature_error_penalty_weight,
        "reject_excessive_feature_error": args.reject_excessive_feature_error,
        "wave_flow_target_weight": args.wave_flow_target_weight,
        "validation_top_k": args.validation_top_k,
        "stability_penalty_weight": args.stability_penalty_weight,
        "inner_maxiter": args.maxiter,
        "inner_popsize": args.popsize,
        "inner_local_maxiter": args.local_maxiter,
        "inner_de_mutation": args.de_mutation,
        "inner_de_recombination": args.de_recombination,
        "inner_optimizer_overrides": _build_optimiser_overrides(
            args, args.gesture
        ),
        "evaluator_max_attempts": args.evaluator_max_attempts,
        "evaluator_retry_base_seconds": args.evaluator_retry_base_seconds,
        "seed": args.seed,
    }

    (out_dir / "results_summary.json").write_text(
        json.dumps(results_summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("CEM TRAINING COMPLETE")
    print("=" * 100)
    if best_profile is None:
        print("Best strictly feasible sampled profile: none")
    else:
        print(f"Best reward:           {best_reward:.6f}")
        print(f"Best profile (round {best_round_index}):")
        for k, v in best_profile.items():
            print(f"  {k}: {v:.6f}")
    print(f"\nFinal distribution mean:")
    for k, v in final_mean.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nFinal distribution std (exploration):")
    for k, v in final_std.items():
        print(f"  {k}: {v:.6f}")
    print(f"\nOutputs:               {out_dir}")
    print(f"Results summary:       {out_dir / 'results_summary.json'}")
    print(f"Elites preserved:      {len(cem.elites)}")


if __name__ == "__main__":
    main()