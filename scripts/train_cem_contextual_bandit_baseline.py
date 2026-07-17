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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import beta as scipy_beta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for candidate in [PROJECT_ROOT, SRC_DIR]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    PerceptualBanditEnvironment,
)
from laban_rl.perceptual_bandit.gemini_evaluator import (
    GeminiProVideoEvaluator,
)
from laban_rl.perceptual_bandit.policy import (
    BanditContext,
    FEATURE_KEYS,
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
            "flow_boundness_target_weight": 0.1,       # Gentle: nudge flow_boundness toward target
            "space_indirectness_target_weight": 0.05,  # Very gentle: space_indirectness
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
}


@dataclass
class Elite:
    """A single elite profile with its reward."""

    profile: dict[str, float]
    reward: float
    round_index: int


class CEMOptimizer:
    """Cross-Entropy Method optimizer maintaining Beta distributions and elites."""

    def __init__(
        self,
        initial_profile: Optional[dict[str, float]] = None,
        initial_width: float = 0.15,
    ):
        """
        Args:
            initial_profile: Starting means for Beta distributions (default 0.5 for all)
            initial_width: Initial std dev for Beta parameterization
        """
        self.feature_keys = FEATURE_KEYS
        
        # Initialize Beta distribution parameters (mean, std) for each feature
        self.beta_params = {}
        
        if initial_profile is None:
            initial_profile = {k: 0.5 for k in self.feature_keys}

        for feature in self.feature_keys:
            mean = initial_profile.get(feature, 0.5)
            # Clamp to avoid boundary issues
            mean = np.clip(mean, 0.01, 0.99)
            self.beta_params[feature] = {
                "mean": mean,
                "std": initial_width,
            }

        self.elites: list[Elite] = []
        self.round_counter = 0

    def sample_batch(self, batch_size: int) -> list[dict[str, float]]:
        """Sample N profiles from current Beta distributions."""
        batch = []
        
        for _ in range(batch_size):
            profile = {}
            for feature in self.feature_keys:
                params = self.beta_params[feature]
                # Convert mean/std to Beta distribution parameters (a, b)
                mean = params["mean"]
                std = params["std"]
                
                # Numerical stability: avoid extreme std
                std = np.clip(std, 0.01, 0.50)
                
                # Convert to Beta(a, b) parameters
                # Using method of moments: mean = a/(a+b), var = ab/((a+b)^2(a+b+1))
                # Solve for a, b given mean and variance
                variance = std ** 2
                
                if variance < 1e-6:
                    # Degenerate case: sample from point mass
                    value = np.clip(mean, 0.0, 1.0)
                else:
                    # Compute Beta parameters
                    denom = variance * (mean * (1 - mean) - variance)
                    if denom > 0:
                        a = mean * (mean * (1 - mean) / variance - 1)
                        b = (1 - mean) * (mean * (1 - mean) / variance - 1)
                        a = np.clip(a, 0.1, 1000)
                        b = np.clip(b, 0.1, 1000)
                    else:
                        # Fallback to Normal truncated to [0, 1]
                        a, b = None, None
                    
                    if a is not None and b is not None:
                        value = np.random.beta(a, b)
                    else:
                        # Fallback: Normal distribution truncated to [0, 1]
                        value = np.random.normal(mean, std)
                        value = np.clip(value, 0.0, 1.0)
                
                profile[feature] = float(np.clip(value, 0.0, 1.0))
            
            batch.append(profile)
        
        return batch

    def update_elites(self, candidates: list[tuple[dict, float]], elite_fraction: float = 0.5):
        """Update elite set and refit Beta distributions.
        
        Args:
            candidates: List of (profile, reward) tuples
            elite_fraction: Keep top-K% (e.g., 0.5 = keep top 50%)
        """
        # Combine new candidates with existing elites
        all_candidates = [
            (elite.profile, elite.reward)
            for elite in self.elites
        ] + list(candidates)

        # Sort by reward descending and keep top-K%
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        num_elites = max(2, int(len(candidates) * elite_fraction))
        elite_candidates = all_candidates[:num_elites]

        # Update elite list
        self.elites = [
            Elite(profile=p, reward=r, round_index=self.round_counter)
            for p, r in elite_candidates
        ]

        # Refit Beta distributions to elite profiles
        self._refit_distributions(elite_candidates)

        self.round_counter += 1

    def _refit_distributions(self, elite_profiles: list[tuple[dict, float]]):
        """Refit Beta distributions to elite profiles using method of moments."""
        profiles_only = [p for p, _ in elite_profiles]
        
        for feature in self.feature_keys:
            values = np.array([p[feature] for p in profiles_only])
            
            # Compute empirical mean and std from elites
            mean = np.mean(values)
            std = np.std(values)
            
            # Update Beta parameters
            self.beta_params[feature]["mean"] = float(np.clip(mean, 0.01, 0.99))
            self.beta_params[feature]["std"] = float(np.clip(std, 0.01, 0.40))

    def get_mean_profile(self) -> dict[str, float]:
        """Get current distribution mean (exploitation focus)."""
        return {
            feature: params["mean"]
            for feature, params in self.beta_params.items()
        }

    def get_std_profile(self) -> dict[str, float]:
        """Get current distribution std (exploration measure)."""
        return {
            feature: params["std"]
            for feature, params in self.beta_params.items()
        }

    def get_best_elite(self) -> Optional[Elite]:
        """Return best elite profile found so far."""
        if not self.elites:
            return None
        return max(self.elites, key=lambda e: e.reward)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gesture",
        choices=["wave", "reach", "point"],
        required=True,
    )
    parser.add_argument(
        "--target-state",
        choices=["confident", "calm", "hesitant", "friendly", "confused", "angry"],
        required=True,
    )
    parser.add_argument("--rounds", type=int, default=15)

    parser.add_argument(
        "--cem-samples-per-round",
        type=int,
        default=5,
        help="Number of profiles to sample per round.",
    )
    parser.add_argument(
        "--cem-elite-fraction",
        type=float,
        default=0.5,
        help="Fraction of samples to keep as elites (e.g., 0.5 = top 50%).",
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
        default=0.90,
        help="Multiply width by this factor each round (exploration decay).",
    )

    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--temperature", type=float, default=0.7)

    parser.add_argument("--maxiter", type=int, default=90)
    parser.add_argument("--popsize", type=int, default=8)
    parser.add_argument("--local-maxiter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument(
        "--realisation-penalty-weight",
        type=float,
        default=0.25,
    )
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
        "best_round_reward", "num_samples_evaluated", "mean_sample_reward",
        "max_sample_reward", "min_sample_reward", "mean_target_probability",
        "mean_realisation_rmse", "exploration_width", "num_elites",
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
    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    # Check if we're resuming from checkpoint
    is_resuming = safe_output_folder(out_dir, args.overwrite, allow_resume=True)

    bandit_context = BanditContext(
        args.gesture,
        args.target_state,
    )
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
    context_key = bandit_context.key
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
    )

    # If resuming, restore CEM state from checkpoint
    if checkpoint_data:
        cem.beta_params = checkpoint_data["cem_beta_params"]
        cem.elites = [
            Elite(profile=e["profile"], reward=e["reward"], round_index=e["round"])
            for e in checkpoint_data["cem_elites"]
        ]

    evaluator = GeminiProVideoEvaluator(
        model=args.model,
        temperature=args.temperature,
    )

    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=args.repeats,
            reward_margin_mode=args.reward_margin_mode,
            realisation_penalty_weight=(
                args.realisation_penalty_weight
            ),
            stability_penalty_weight=(
                args.stability_penalty_weight
            ),
        ),
        optimiser_overrides=_build_optimiser_overrides(
            args=args,
            gesture=bandit_context.gesture,
        ),
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
                best_round_index = row.get("round", None)
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

    try:
        for round_index in range(start_round, args.rounds + 1):
            print("\n" + "#" * 100)
            print(
                f"CEM ROUND {round_index}/{args.rounds} "
                f"| {bandit_context.key}"
            )
            print("#" * 100)

            # Apply exploration decay
            current_width = args.cem_initial_width * (
                args.exploration_decay_rate ** (round_index - 1)
            )
            for feature in FEATURE_KEYS:
                cem.beta_params[feature]["std"] = current_width

            # Sample batch of profiles
            sampled_profiles = cem.sample_batch(args.cem_samples_per_round)

            # Evaluate each sampled profile
            candidates = []
            round_rewards = []
            round_rmses = []
            round_target_probs = []

            for sample_idx, profile in enumerate(sampled_profiles):
                print(f"\n  Sample {sample_idx + 1}/{args.cem_samples_per_round}: {profile}")

                round_dir = out_dir / "rounds" / f"round_{round_index:03d}_sample_{sample_idx:02d}"

                result = environment.step(
                    context=environment_context,
                    action_profile=profile,
                    out_dir=round_dir / "optimiser_outputs",
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
                round_rmses.append(result.realisation_rmse)
                round_target_probs.append(result.mean_target_probability)

                print(f"    Reward: {result.outer_reward:.6f}, "
                      f"Target prob: {result.mean_target_probability:.6f}, "
                      f"RMSE: {result.realisation_rmse:.6f}")

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

            # Update elites (keeps historical bests + new samples)
            cem.update_elites(candidates, elite_fraction=args.cem_elite_fraction)

            # Record round statistics
            best_elite = cem.get_best_elite()
            current_mean = cem.get_mean_profile()

            row = {
                "round": round_index,
                "best_round_reward": float(best_reward),
                "num_samples_evaluated": len(sampled_profiles),
                "mean_sample_reward": float(np.mean(round_rewards)),
                "max_sample_reward": float(np.max(round_rewards)),
                "min_sample_reward": float(np.min(round_rewards)),
                "mean_target_probability": float(np.mean(round_target_probs)),
                "mean_realisation_rmse": float(np.mean(round_rmses)),
                "exploration_width": current_width,
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

            history.append(row)

            save_history_csv(history, out_dir / "training_history.csv")
            save_plots(history, out_dir)

            # Save checkpoint
            torch.save(
                {
                    "round": round_index,
                    "cem_beta_params": cem.beta_params,
                    "cem_elites": [
                        {"profile": e.profile, "reward": e.reward, "round": e.round_index}
                        for e in cem.elites
                    ],
                    "context": {
                        "gesture": args.gesture,
                        "target_state": args.target_state,
                    },
                    "best_reward": best_reward,
                    "best_profile": best_profile,
                },
                out_dir / "latest_checkpoint.pt",
            )

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

    finally:
        evaluator.close()

    # Compute final summaries
    final_mean = cem.get_mean_profile()
    final_std = cem.get_std_profile()

    num_rounds_completed = len(history)
    mean_reward_all = (
        float(np.mean([row["best_round_reward"] for row in history]))
        if history
        else 0.0
    )
    mean_reward_last5 = (
        float(
            np.mean(
                [row["best_round_reward"] for row in history[-5:]]
            )
        )
        if len(history) >= 5
        else mean_reward_all
    )
    mean_target_prob_all = (
        float(
            np.mean(
                [row["mean_target_probability"] for row in history]
            )
        )
        if history
        else 0.0
    )
    mean_target_prob_last5 = (
        float(
            np.mean(
                [row["mean_target_probability"] for row in history[-5:]]
            )
        )
        if len(history) >= 5
        else mean_target_prob_all
    )

    results_summary = {
        "gesture": args.gesture,
        "target_state": args.target_state,
        "num_rounds_completed": num_rounds_completed,
        "samples_per_round": args.cem_samples_per_round,
        "repeats_per_round": args.repeats,
        "total_gemini_evals": num_rounds_completed * args.cem_samples_per_round * args.repeats,
        "best_round_index": best_round_index,
        "best_sampled_profile": best_profile,
        "best_reward": best_reward,
        "final_distribution_mean": final_mean,
        "final_distribution_std": final_std,
        "mean_reward_all_rounds": mean_reward_all,
        "mean_reward_last_5_rounds": mean_reward_last5,
        "mean_target_probability_all_rounds": mean_target_prob_all,
        "mean_target_probability_last_5_rounds": mean_target_prob_last5,
        "num_elites_at_end": len(cem.elites),
        "reward_margin_mode": args.reward_margin_mode,
        "cem_elite_fraction": args.cem_elite_fraction,
        "cem_initial_width": args.cem_initial_width,
        "exploration_decay_rate": args.exploration_decay_rate,
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
