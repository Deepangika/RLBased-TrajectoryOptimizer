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
from laban_rl.config import EMOTION_STATES, FEATURE_KEYS
from laban_rl.perceptual_bandit.cem import CEMOptimizer
from laban_rl.perceptual_bandit.selection import select_feasible_incumbent


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
    
    return overrides


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gesture",
        choices=["wave", "reach", "point"],
        required=True,
    )
    parser.add_argument(
        "--target-state",
        choices=list(EMOTION_STATES),
        required=True,
    )
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folder if it is non-empty. Default: error.",
    )
    return parser.parse_args()


def safe_output_folder(out_dir: Path, overwrite: bool, allow_resume: bool = False) -> bool:
    """
    Ensure output folder is safe. Returns True if resuming from checkpoint.
    
    If allow_resume and checkpoint exists, preserve it and return True.
    Otherwise, delete and recreate (if overwrite) or raise error.
    """
    checkpoint_path = out_dir / "latest_checkpoint.pt"
    has_checkpoint = checkpoint_path.exists()
    
    if out_dir.exists() and list(out_dir.iterdir()):
        if has_checkpoint and allow_resume:
            # Preserve checkpoint, but clean up round outputs that may be incomplete
            print(f"  Resuming from checkpoint: {checkpoint_path}")
            return True
        
        if not overwrite:
            raise RuntimeError(
                f"Output folder {out_dir} is non-empty and --overwrite not set. "
                "Either use --overwrite, choose a different output path, or delete the existing folder."
            )
        import shutil
        shutil.rmtree(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    return False


def load_checkpoint(checkpoint_path: Path) -> dict:
    """Load training state from checkpoint."""
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    print(f"  Loaded checkpoint from round {checkpoint['round']}")
    return checkpoint


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
            fieldnames=list(rows[0].keys()),
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

    np.random.seed(args.seed)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    # Check if we're resuming from checkpoint
    is_resuming = safe_output_folder(out_dir, args.overwrite, allow_resume=True)

    environment_context = Context(
        gesture=args.gesture,
        target_state=args.target_state,
    )

    # Load checkpoint if resuming
    checkpoint_data = None
    if is_resuming:
        checkpoint_path = out_dir / "latest_checkpoint.pt"
        checkpoint_data = load_checkpoint(checkpoint_path)
        print(f"  Resuming from round {checkpoint_data['round']} of {args.rounds}")

    # Get informed profile or raise error if missing and not allowed.
    context_key = f"{args.gesture}::{args.target_state}"
    initial_profile = INFORMED_PROFILES.get(context_key)

    if initial_profile is None:
        if not args.allow_default_profile:
            raise ValueError(
                f"No informed profile for context {context_key!r}. "
                "Either add the profile to INFORMED_PROFILES, or use --allow-default-profile."
            )
        initial_profile = None

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
        if args.target_state not in evaluator.state_labels:
            raise ValueError(
                f"Mock evaluator does not support target state "
                f"{args.target_state!r}. Available labels: "
                f"{list(evaluator.state_labels)}"
            )

    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=args.repeats,
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
    )
    validation_environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=args.validation_repeats,
            reward_margin_mode=args.reward_margin_mode,
            realisation_penalty_weight=args.realisation_penalty_weight,
            max_feature_error_threshold=args.max_feature_error_threshold,
            max_feature_error_penalty_weight=args.max_feature_error_penalty_weight,
            reject_excessive_feature_error=args.reject_excessive_feature_error,
            stability_penalty_weight=args.stability_penalty_weight,
            evaluator_failure_mode="raise",
        ),
        optimiser_overrides=_build_optimiser_overrides(args=args, gesture=args.gesture),
    )

    # Load existing history if resuming
    if is_resuming:
        history = load_history_csv(out_dir / "training_history.csv")
        best_reward = checkpoint_data["best_reward"]
        best_profile = checkpoint_data["best_profile"]
        best_round_index = None
        # Find the round where best_reward was achieved
        for row in history:
            if abs(row.get("best_round_reward", float("-inf")) - best_reward) < 1e-6:
                best_round_index = int(row["round"]) if row.get("round") is not None else None
                break
        start_round = checkpoint_data["round"] + 1
        print(f"  Loaded history from {len(history)} previous rounds")
        print(f"  Best reward so far: {best_reward:.6f} (from round {best_round_index})")
    else:
        history: list[dict] = []
        best_reward = float("-inf")
        best_profile = None
        best_round_index = None
        start_round = 1

    validation_results: dict[str, dict] = {}
    validation_path = out_dir / "independent_validation.json"
    if is_resuming and validation_path.exists():
        validation_results = json.loads(validation_path.read_text(encoding="utf-8"))

    def evaluate_with_retry(env, *, context, profile, output_dir):
        last_error = None
        for attempt in range(1, args.evaluator_max_attempts + 1):
            try:
                return env.step(
                    context=context,
                    action_profile=profile,
                    out_dir=output_dir,
                )
            except RuntimeError as error:
                last_error = error
                if attempt >= args.evaluator_max_attempts:
                    break
                delay = args.evaluator_retry_base_seconds * (2 ** (attempt - 1))
                print(
                    f"  Evaluator attempt {attempt} failed; retrying in "
                    f"{delay:.1f}s. The failed call is not assigned a reward."
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
        if not is_resuming:
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

                print(
                    f"    Reward: {result.outer_reward:.6f}, "
                    f"Target prob: {result.mean_target_probability if result.mean_target_probability is not None else 'N/A'}, "
                    f"RMSE: {result.realisation_rmse if result.realisation_rmse is not None else 'N/A'}, "
                    f"valid: {result.valid_realisation}"
                )

                if result.outer_reward > best_reward:
                    best_reward = float(result.outer_reward)
                    best_profile = dict(profile)
                    best_round_index = round_index

                # Save per-sample summary
                (round_dir / "sample_summary.json").write_text(
                    json.dumps(
                        {
                            "sampled_profile": profile,
                            "environment_result": result.to_dict(),
                        },
                        indent=2,
                        allow_nan=True,
                    ),
                    encoding="utf-8",
                )

            # Current-round elites update the distribution. Best-so-far is
            # tracked separately and cannot lock the sampling distribution.
            cem.update_elites(candidates, elite_fraction=args.cem_elite_fraction)
            cem.decay_exploration(args.exploration_decay_rate)
            cem_diagnostics = cem.diagnostics()

            # Record round statistics
            best_elite = cem.get_best_elite()
            current_mean = cem.get_mean_profile()
            current_std = cem.get_std_profile()

            row = {
                "round": round_index,
                "best_round_reward": float(best_reward),
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
            with (out_dir / "latest_checkpoint.pt").open("wb") as handle:
                pickle.dump({
                    "round": round_index,
                    "cem_state": cem.state_dict(),
                    "context": {
                        "gesture": args.gesture,
                        "target_state": args.target_state,
                    },
                    "best_reward": best_reward,
                    "best_profile": best_profile,
                }, handle)

            (out_dir / "best_profile.json").write_text(
                json.dumps(
                    {
                        "best_reward": best_reward,
                        "best_profile": best_profile,
                    },
                    indent=2,
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
        if best_profile is None:
            raise RuntimeError("Training completed without a sampled profile.")
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
    successful_vlm_evaluations = 0
    for sample_path in (out_dir / "rounds").glob("round_*_sample_*/sample_summary.json"):
        payload = json.loads(sample_path.read_text(encoding="utf-8"))
        successful_vlm_evaluations += len(
            payload["environment_result"].get("perceptual_evaluations") or []
        )
    successful_vlm_evaluations += sum(
        len(result.get("perceptual_evaluations") or [])
        for name, result in validation_results.items()
        if name not in {"best_sampled_profile", "selection"}
    )

    results_summary = {
        "gesture": args.gesture,
        "target_state": args.target_state,
        "num_rounds_completed": num_rounds_completed,
        "samples_per_round": args.cem_samples_per_round,
        "repeats_per_round": args.repeats,
        "planned_training_vlm_evaluations": num_rounds_completed * args.cem_samples_per_round * args.repeats,
        "successful_vlm_evaluations": successful_vlm_evaluations,
        "planned_training_evaluator_calls": num_rounds_completed * args.cem_samples_per_round * args.repeats,
        "successful_evaluator_calls": successful_vlm_evaluations,
        "best_round_index": best_round_index,
        "best_sampled_profile": best_profile,
        "best_reward": best_reward,
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