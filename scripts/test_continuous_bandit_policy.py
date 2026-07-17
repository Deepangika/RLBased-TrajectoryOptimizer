"""Smoke test for the Beta contextual-bandit policy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from laban_rl.perceptual_bandit.policy import (
    BanditContext,
    ContextRewardBaseline,
    ContinuousContextualBanditPolicy,
    FEATURE_KEYS,
    reinforce_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def profile_array(profile: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [profile[key] for key in FEATURE_KEYS],
        dtype=float,
    )


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    context = BanditContext("point", "confident")

    informed_profile = {
        "weight": 0.40,
        "time": 0.70,
        "flow_boundness": 0.25,
        "space_indirectness": 0.05,
        "shape_arcness": 0.45,
    }

    hidden_optimum = np.asarray(
        [0.58, 0.82, 0.18, 0.03, 0.32],
        dtype=float,
    )

    policy = ContinuousContextualBanditPolicy(
        informed_initial_profiles={
            context.key: informed_profile,
        },
        initial_action_std=0.08,
    )

    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.lr,
    )
    baseline = ContextRewardBaseline(
        momentum=0.50,
    )

    initial_mean = profile_array(
        policy.mean_profile(context)
    )
    initial_std = profile_array(
        policy.approximate_action_std_profile(context)
    )

    initial_rmse = float(
        np.sqrt(np.mean((initial_mean - hidden_optimum) ** 2))
    )

    rewards = []

    print("=" * 88)
    print("BETA CONTEXTUAL-BANDIT POLICY SMOKE TEST")
    print("=" * 88)
    print(f"Context:        {context.key}")
    print(f"Initial mean:   {np.round(initial_mean, 4)}")
    print(f"Initial std:    {np.round(initial_std, 4)}")
    print(f"Hidden target:  {np.round(hidden_optimum, 4)}")
    print(f"Initial RMSE:   {initial_rmse:.6f}")

    for round_index in range(1, args.rounds + 1):
        sample = policy.sample_action(context)
        action = profile_array(sample.action_profile)

        mse = float(np.mean((action - hidden_optimum) ** 2))
        reward = 1.0 - 20.0 * mse
        rewards.append(reward)

        stats = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            sample=sample,
            reward=reward,
            baseline=baseline,
        )

        if (
            round_index == 1
            or round_index % 50 == 0
            or round_index == args.rounds
        ):
            current_mean = profile_array(
                policy.mean_profile(context)
            )
            current_std = profile_array(
                policy.approximate_action_std_profile(context)
            )
            current_rmse = float(
                np.sqrt(
                    np.mean(
                        (current_mean - hidden_optimum) ** 2
                    )
                )
            )

            print(
                f"Round {round_index:4d} | "
                f"reward={reward:+.6f} | "
                f"advantage={float(stats['advantage']):+.6f} | "
                f"updated={stats['policy_updated']} | "
                f"mean RMSE={current_rmse:.6f}"
            )

    final_mean = profile_array(
        policy.mean_profile(context)
    )
    final_std = profile_array(
        policy.approximate_action_std_profile(context)
    )
    final_rmse = float(
        np.sqrt(np.mean((final_mean - hidden_optimum) ** 2))
    )

    print("\n" + "=" * 88)
    print("FINAL RESULT")
    print("=" * 88)
    print(f"Final mean:     {np.round(final_mean, 4)}")
    print(f"Final std:      {np.round(final_std, 4)}")
    print(f"Hidden target:  {np.round(hidden_optimum, 4)}")
    print(f"Final RMSE:     {final_rmse:.6f}")
    print(f"Improvement:    {initial_rmse - final_rmse:.6f}")
    print(
        f"Mean reward, last 50 rounds: "
        f"{np.mean(rewards[-50:]):.6f}"
    )

    if final_rmse >= initial_rmse:
        raise RuntimeError(
            "FAIL: policy mean did not move closer to the optimum."
        )

    if final_rmse > 0.07:
        raise RuntimeError(
            "FAIL: learning worked but remained too weak."
        )

    # Important regression check for the original bug:
    # low initial Space mean should not have huge exploration.
    if initial_std[3] > 0.12:
        raise RuntimeError(
            "FAIL: Space exploration is still too broad."
        )

    print("\nPASS: Beta policy learning and bounded exploration work.")
    print("=" * 88)


if __name__ == "__main__":
    main()
