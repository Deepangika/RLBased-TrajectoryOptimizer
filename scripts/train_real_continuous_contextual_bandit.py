"""Train one real continuous contextual-bandit context.

Drop into:
    scripts/train_real_continuous_contextual_bandit.py

This connects:
    policy -> real optimiser -> variant-only video -> Gemini -> reward -> update

The first version intentionally trains ONE context at a time.
Every round is checkpointed because each environment interaction is expensive.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

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
    ContextRewardBaseline,
    ContinuousContextualBanditPolicy,
    FEATURE_KEYS,
    reinforce_update,
)


# All 12 gesture-state profiles for informed initialization.
INFORMED_PROFILES = {
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
        "weight": 0.50,
        "time": 0.55,
        "flow_boundness": 0.45,
        "space_indirectness": 0.55,
        "shape_arcness": 0.70,
    },
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
}


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
    parser.add_argument("--rounds", type=int, default=10)

    parser.add_argument("--policy-lr", type=float, default=0.01)
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.0,
        help="Entropy bonus coefficient. Default 0.0 (off). Set 0.01-0.05 to keep exploration.",
    )
    parser.add_argument(
        "--entropy-decay-steps",
        type=int,
        default=0,
        help="Decay entropy linearly to zero over N steps. 0 = no decay (constant entropy-weight).",
    )
    parser.add_argument(
        "--initial-action-std",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--baseline-momentum",
        type=float,
        default=0.90,
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
        "--rmse-normal-threshold",
        type=float,
        default=0.05,
        help="RMSE below this is normal; policy updates proceed without warning.",
    )
    parser.add_argument(
        "--rmse-skip-threshold",
        type=float,
        default=0.10,
        help="RMSE at or above this causes policy update to be skipped.",
    )
    parser.add_argument(
        "--skip-high-rmse-updates",
        action="store_true",
        help="Skip policy updates when realisation RMSE is too high.",
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


def safe_output_folder(out_dir: Path, overwrite: bool) -> None:
    """Ensure output folder is safe (empty or new)."""
    if out_dir.exists() and list(out_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output folder {out_dir} is non-empty and --overwrite not set. "
                "Either use --overwrite, choose a different output path, or delete the existing folder."
            )
        # Remove the existing folder to ensure a clean slate.
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


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
    plt.plot(rounds, [row["outer_reward"] for row in rows], marker="o")
    plt.xlabel("Round")
    plt.ylabel("Outer reward")
    plt.title("Real contextual-bandit reward")
    plt.grid(True)
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
        [row["realisation_rmse"] for row in rows],
        marker="o",
    )
    plt.xlabel("Round")
    plt.ylabel("Realisation RMSE")
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
            [row[f"sampled_{feature}"] for row in rows],
            marker="o",
            label="Sampled action",
        )
        plt.plot(
            rounds,
            [row[f"mean_{feature}"] for row in rows],
            marker=".",
            label="Policy mean",
        )
        plt.ylim(0.0, 1.0)
        plt.xlabel("Round")
        plt.ylabel("Normalised value")
        plt.title(f"Policy trajectory: {feature}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            out_dir / f"policy_{feature}_curve.png",
            dpi=160,
        )
        plt.close()


def compute_entropy_weight(round_index: int, args) -> float:
    """Compute entropy weight for this round (with optional decay)."""
    if args.entropy_decay_steps <= 0:
        return args.entropy_weight
    decay_progress = min(1.0, round_index / args.entropy_decay_steps)
    return args.entropy_weight * (1.0 - decay_progress)


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    # Safe output folder creation.
    safe_output_folder(out_dir, args.overwrite)

    bandit_context = BanditContext(
        args.gesture,
        args.target_state,
    )
    environment_context = Context(
        gesture=args.gesture,
        target_state=args.target_state,
    )

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

    informed = (
        {context_key: initial_profile}
        if initial_profile is not None
        else None
    )

    policy = ContinuousContextualBanditPolicy(
        informed_initial_profiles=informed,
        initial_action_std=args.initial_action_std,
    )

    policy_optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.policy_lr,
    )

    baseline = ContextRewardBaseline(
        momentum=args.baseline_momentum,
    )

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
        optimiser_overrides={
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "local_maxiter": args.local_maxiter,
            "seed": args.seed,
        },
    )

    history: list[dict] = []
    best_reward = float("-inf")
    best_profile = None

    try:
        for round_index in range(1, args.rounds + 1):
            print("\n" + "#" * 100)
            print(
                f"REAL BANDIT ROUND {round_index}/{args.rounds} "
                f"| {bandit_context.key}"
            )
            print("#" * 100)

            sample = policy.sample_action(bandit_context)

            round_dir = out_dir / "rounds" / f"round_{round_index:03d}"

            result = environment.step(
                context=environment_context,
                action_profile=sample.action_profile,
                out_dir=round_dir / "optimiser_outputs",
            )

            # Compute entropy weight for this round (may be decayed).
            effective_entropy_weight = compute_entropy_weight(round_index, args)

            # Apply RMSE gating.
            update_stats = reinforce_update(
                policy=policy,
                optimizer=policy_optimizer,
                sample=sample,
                reward=result.outer_reward,
                baseline=baseline,
                entropy_weight=effective_entropy_weight,
                realisation_rmse=result.realisation_rmse,
                rmse_normal_threshold=args.rmse_normal_threshold,
                rmse_skip_threshold=args.rmse_skip_threshold,
                skip_high_rmse_updates=args.skip_high_rmse_updates,
            )

            current_mean = policy.mean_profile(
                bandit_context
            )

            row = {
                "round": round_index,
                "outer_reward": float(result.outer_reward),
                "outer_reward_clipped": float(result.outer_reward_clipped),
                "mean_target_probability": float(
                    result.mean_target_probability
                ),
                "mean_margin": float(result.mean_margin),
                "mean_margin_clipped": float(result.mean_margin_clipped),
                "mean_perceptual_reward": float(
                    result.mean_perceptual_reward
                ),
                "mean_perceptual_reward_clipped": float(
                    result.mean_perceptual_reward_clipped
                ),
                "perceptual_reward_std": float(
                    result.perceptual_reward_std
                ),
                "realisation_rmse": float(
                    result.realisation_rmse
                ),
                "rmse_skipped": bool(update_stats.get("rmse_skipped", False)),
                "advantage": float(update_stats["advantage"]),
                "baseline_before": float(
                    update_stats["baseline_before"]
                ),
                "baseline_after": float(
                    update_stats["baseline_after"]
                ),
                "entropy": float(update_stats.get("entropy", 0.0)),
                "effective_entropy_weight": float(
                    update_stats.get("effective_entropy_weight", 0.0)
                ),
                "policy_loss": float(update_stats.get("policy_loss", 0.0)),
                "policy_updated": bool(
                    update_stats["policy_updated"]
                ),
            }

            for feature in FEATURE_KEYS:
                row[f"sampled_{feature}"] = float(
                    sample.action_profile[feature]
                )
                row[f"mean_{feature}"] = float(
                    current_mean[feature]
                )

            history.append(row)

            if result.outer_reward > best_reward:
                best_reward = float(result.outer_reward)
                best_profile = dict(sample.action_profile)

            save_history_csv(
                history,
                out_dir / "training_history.csv",
            )
            save_plots(history, out_dir)

            torch.save(
                {
                    "round": round_index,
                    "policy_state_dict": policy.state_dict(),
                    "optimizer_state_dict": (
                        policy_optimizer.state_dict()
                    ),
                    "baseline_state_dict": baseline.state_dict(),
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

            (round_dir / "round_summary.json").write_text(
                json.dumps(
                    {
                        "sampled_profile": sample.action_profile,
                        "policy_mean_after_update": current_mean,
                        "environment_result": result.to_dict(),
                        "update_stats": update_stats,
                    },
                    indent=2,
                    allow_nan=True,
                ),
                encoding="utf-8",
            )

            print("\nRound summary:")
            print(f"  Sampled profile: {sample.action_profile}")
            print(f"  Outer reward:    {result.outer_reward:.6f}")
            print(
                f"  Target prob:     "
                f"{result.mean_target_probability:.6f}"
            )
            print(
                f"  Realisation RMSE:"
                f" {result.realisation_rmse:.6f}"
            )
            print(
                f"  Advantage:       "
                f"{float(update_stats['advantage']):+.6f}"
            )
            print(
                f"  Policy updated:  "
                f"{update_stats['policy_updated']}"
            )
            print(f"  Policy mean:     {current_mean}")

    finally:
        evaluator.close()

    # Compute final summaries.
    final_mean = policy.mean_profile(bandit_context)
    final_std = policy.approximate_action_std_profile(
        bandit_context
    )

    num_rounds_completed = len(history)
    num_skipped_rmse = sum(
        1 for row in history if row.get("rmse_skipped", False)
    )
    mean_reward_all = (
        float(np.mean([row["outer_reward"] for row in history]))
        if history
        else 0.0
    )
    mean_reward_last5 = (
        float(
            np.mean(
                [row["outer_reward"] for row in history[-5:]]
            )
        )
        if len(history) >= 5
        else mean_reward_all
    )
    mean_target_prob_all = (
        float(
            np.mean(
                [
                    row["mean_target_probability"]
                    for row in history
                    if row["mean_target_probability"] is not None
                ]
            )
        )
        if history
        else 0.0
    )
    mean_target_prob_last5 = (
        float(
            np.mean(
                [
                    row["mean_target_probability"]
                    for row in history[-5:]
                    if row["mean_target_probability"] is not None
                ]
            )
        )
        if len(history) >= 5
        else mean_target_prob_all
    )

    results_summary = {
        "gesture": args.gesture,
        "target_state": args.target_state,
        "num_rounds_completed": num_rounds_completed,
        "repeats_per_round": args.repeats,
        "best_round_index": (
            history.index(next(r for r in history if r["outer_reward"] == best_reward))
            + 1
            if best_reward != float("-inf")
            else None
        ),
        "best_sampled_profile": best_profile,
        "best_reward": best_reward,
        "final_policy_mean": final_mean,
        "final_policy_std": final_std,
        "mean_reward_all_rounds": mean_reward_all,
        "mean_reward_last_5_rounds": mean_reward_last5,
        "mean_target_probability_all_rounds": mean_target_prob_all,
        "mean_target_probability_last_5_rounds": mean_target_prob_last5,
        "num_skipped_rmse_updates": num_skipped_rmse,
        "num_invalid_realisations": 0,  # Would track failed optimiser runs.
        "reward_margin_mode": args.reward_margin_mode,
        "entropy_weight_initial": args.entropy_weight,
        "entropy_decay_steps": args.entropy_decay_steps,
        "rmse_normal_threshold": args.rmse_normal_threshold,
        "rmse_skip_threshold": args.rmse_skip_threshold,
        "skip_high_rmse_updates": args.skip_high_rmse_updates,
    }

    (out_dir / "results_summary.json").write_text(
        json.dumps(results_summary, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("TRAINING COMPLETE")
    print("=" * 100)
    print(f"Best reward:  {best_reward:.6f}")
    print(f"Best profile: {best_profile}")
    print(f"Outputs:      {out_dir}")
    print(f"Results summary saved to: {out_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
