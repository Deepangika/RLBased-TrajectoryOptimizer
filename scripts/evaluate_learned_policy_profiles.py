"""Compare the initial, best-sampled, and final learned Laban profiles.

Drop into:
    scripts/evaluate_learned_policy_profiles.py

This script does NOT train or update the policy.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    ContinuousContextualBanditPolicy,
    FEATURE_KEYS,
)

INFORMED_PROFILES = {
    "point::confident": {
        "weight": 0.40,
        "time": 0.70,
        "flow_boundness": 0.25,
        "space_indirectness": 0.05,
        "shape_arcness": 0.45,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
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
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_best_profile(training_dir: Path) -> dict[str, float]:
    payload = json.loads(
        (training_dir / "best_profile.json").read_text(encoding="utf-8")
    )
    profile = payload["best_profile"]
    return {key: float(profile[key]) for key in FEATURE_KEYS}


def reconstruct_final_mean(
    checkpoint: dict,
    gesture: str,
    target_state: str,
    initial_profile: dict[str, float],
) -> dict[str, float]:
    context = BanditContext(gesture, target_state)

    policy = ContinuousContextualBanditPolicy(
        informed_initial_profiles={
            context.key: initial_profile,
        },
    )
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()

    return policy.mean_profile(context)


def flatten_result(label: str, profile: dict[str, float], result) -> dict:
    row = {
        "candidate": label,
        "outer_reward": float(result.outer_reward),
        "mean_target_probability": float(result.mean_target_probability),
        "mean_margin": float(result.mean_margin),
        "mean_perceptual_reward": float(result.mean_perceptual_reward),
        "perceptual_reward_std": float(result.perceptual_reward_std),
        "realisation_rmse": float(result.realisation_rmse),
    }

    for key in FEATURE_KEYS:
        row[f"requested_{key}"] = float(profile[key])

    return row


def save_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(rows: list[dict], out_dir: Path) -> None:
    labels = [row["candidate"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, [row["outer_reward"] for row in rows])
    plt.ylabel("Mean outer reward")
    plt.title("Repeated VLM evaluation: outer reward")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "comparison_outer_reward.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        labels,
        [row["mean_target_probability"] for row in rows],
    )
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean target-state probability")
    plt.title("Repeated VLM evaluation: target probability")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(
        out_dir / "comparison_target_probability.png",
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(
        labels,
        [row["perceptual_reward_std"] for row in rows],
    )
    plt.ylabel("Perceptual reward standard deviation")
    plt.title("Repeated VLM evaluation: reward variability")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(
        out_dir / "comparison_reward_variability.png",
        dpi=180,
    )
    plt.close()


def main() -> None:
    args = parse_args()

    training_dir = resolve_path(args.training_output)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        training_dir / "latest_checkpoint.pt",
        map_location="cpu",
    )

    gesture = str(checkpoint["context"]["gesture"])
    target_state = str(checkpoint["context"]["target_state"])
    context_key = f"{gesture}::{target_state}"

    if context_key not in INFORMED_PROFILES:
        raise KeyError(
            f"No initial informed profile configured for {context_key}."
        )

    initial_profile = dict(INFORMED_PROFILES[context_key])
    best_profile = load_best_profile(training_dir)
    final_mean_profile = reconstruct_final_mean(
        checkpoint,
        gesture,
        target_state,
        initial_profile,
    )

    candidates = {
        "initial_informed": initial_profile,
        "best_sampled": best_profile,
        "final_policy_mean": final_mean_profile,
    }

    print("=" * 100)
    print("POST-TRAINING PROFILE COMPARISON")
    print("=" * 100)
    print(f"Gesture:      {gesture}")
    print(f"Target state: {target_state}")
    print(f"Repeats:      {args.repeats}")

    evaluator = GeminiProVideoEvaluator(
        model=args.model,
        temperature=args.temperature,
    )

    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=args.repeats,
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

    env_context = Context(
        gesture=gesture,
        target_state=target_state,
    )

    rows = []
    detailed = {}

    try:
        for label, profile in candidates.items():
            print("\n" + "#" * 100)
            print(f"EVALUATING: {label}")
            print("#" * 100)

            result = environment.step(
                context=env_context,
                action_profile=profile,
                out_dir=(
                    out_dir
                    / label
                    / "optimiser_outputs"
                ),
            )

            rows.append(flatten_result(label, profile, result))
            detailed[label] = {
                "requested_profile": profile,
                "result": result.to_dict(),
            }

            print(f"Mean target probability: {result.mean_target_probability:.6f}")
            print(f"Mean margin:             {result.mean_margin:.6f}")
            print(f"Mean perceptual reward:  {result.mean_perceptual_reward:.6f}")
            print(f"Reward std:              {result.perceptual_reward_std:.6f}")
            print(f"Realisation RMSE:        {result.realisation_rmse:.6f}")
            print(f"Outer reward:            {result.outer_reward:.6f}")

    finally:
        evaluator.close()

    save_csv(rows, out_dir / "profile_comparison.csv")

    (out_dir / "profile_comparison.json").write_text(
        json.dumps(detailed, indent=2, allow_nan=True),
        encoding="utf-8",
    )

    save_plots(rows, out_dir)

    ranked = sorted(
        rows,
        key=lambda row: row["outer_reward"],
        reverse=True,
    )

    print("\n" + "=" * 100)
    print("FINAL RANKING")
    print("=" * 100)

    for rank, row in enumerate(ranked, start=1):
        print(
            f"{rank}. {row['candidate']:20s} | "
            f"outer reward={row['outer_reward']:+.6f} | "
            f"target prob={row['mean_target_probability']:.4f} | "
            f"reward std={row['perceptual_reward_std']:.4f}"
        )

    print(f"\nSaved comparison to: {out_dir}")


if __name__ == "__main__":
    main()
