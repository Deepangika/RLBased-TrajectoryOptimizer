"""Analyze and visualize contextual-bandit training results."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_results(output_dir: Path):
    """Load results summary and training history."""
    summary_path = output_dir / "results_summary.json"
    history_path = output_dir / "training_history.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"No results_summary.json in {output_dir}")
    if not history_path.exists():
        raise FileNotFoundError(f"No training_history.csv in {output_dir}")

    with summary_path.open() as f:
        summary = json.load(f)

    history = pd.read_csv(history_path)

    return summary, history


def print_analysis(summary: dict, history: pd.DataFrame, output_dir: Path) -> None:
    """Print comprehensive analysis of results."""
    print("\n" + "=" * 100)
    print("BANDIT TRAINING ANALYSIS")
    print("=" * 100)

    print(f"\nGesture:           {summary['gesture']}")
    if summary.get("target_state") is not None:
        print(f"Target state:      {summary['target_state']}")
    else:
        print(f"Target VAD:        {summary['target_vad']}")
    print(f"Rounds completed:  {summary['num_rounds_completed']}")
    print(f"Repeats per round: {summary['repeats_per_round']}")

    print("\n" + "-" * 100)
    print("BEST PROFILE FOUND")
    print("-" * 100)

    best_round = summary["best_round_index"]
    best_reward = summary["best_reward"]
    best_profile = summary["best_sampled_profile"]

    print(f"\nRound:   {best_round}")
    print(f"Reward:  {best_reward:.6f}")
    print(f"Profile:")
    for key, val in best_profile.items():
        print(f"  {key:25s}: {val:.6f}")

    print("\n" + "-" * 100)
    print("FINAL POLICY STATE")
    print("-" * 100)

    final_mean = summary["final_policy_mean"]
    final_std = summary["final_policy_std"]

    print(f"\nFinal Policy Mean:")
    for key, val in final_mean.items():
        print(f"  {key:25s}: {val:.6f}")

    print(f"\nFinal Policy Std (approx exploration):")
    for key, val in final_std.items():
        print(f"  {key:25s}: {val:.6f}")

    print("\n" + "-" * 100)
    print("LEARNING TRAJECTORY")
    print("-" * 100)

    mean_all = summary["mean_reward_all_rounds"]
    mean_last5 = summary["mean_reward_last_5_rounds"]
    mean_target_prob_all = summary["mean_target_probability_all_rounds"]
    mean_target_prob_last5 = summary["mean_target_probability_last_5_rounds"]

    print(f"\nMean reward (all rounds):       {mean_all:.6f}")
    print(f"Mean reward (last 5 rounds):   {mean_last5:.6f}")
    print(f"Mean target probability (all): {mean_target_prob_all:.6f}")
    print(f"Mean target probability (last 5): {mean_target_prob_last5:.6f}")

    print("\n" + "-" * 100)
    print("SAFETY AND DIAGNOSTICS")
    print("-" * 100)

    num_skipped = summary["num_skipped_rmse_updates"]
    num_invalid = summary["num_invalid_realisations"]

    print(f"\nRMSE-gated updates skipped:    {num_skipped}")
    print(f"Invalid realisations:          {num_invalid}")
    print(f"Reward margin mode:            {summary['reward_margin_mode']}")
    print(f"Entropy weight (initial):      {summary.get('entropy_weight_initial', 'N/A')}")
    print(f"Entropy decay steps:           {summary.get('entropy_decay_steps', 'N/A')}")
    print(f"RMSE normal threshold:         {summary.get('rmse_normal_threshold', 'N/A')}")
    print(f"RMSE skip threshold:           {summary.get('rmse_skip_threshold', 'N/A')}")
    print(f"Skip high-RMSE updates:        {summary.get('skip_high_rmse_updates', 'N/A')}")

    print("\n" + "-" * 100)
    print("DETAILED PER-ROUND METRICS")
    print("-" * 100)

    # Find rounds with special events
    best_row_idx = history["outer_reward"].idxmax()
    worst_row_idx = history["outer_reward"].idxmin()
    skipped_rows = history[history["rmse_skipped"] == True]

    print(f"\nBest round (by outer_reward):")
    row = history.iloc[best_row_idx]
    print(f"  Round {int(row['round'])}: "
          f"reward={row['outer_reward']:.6f}, "
          f"target_prob={row['mean_target_probability']:.6f}, "
          f"rmse={row['realisation_rmse']:.6f}, "
          f"updated={row['policy_updated']}")

    print(f"\nWorst round (by outer_reward):")
    row = history.iloc[worst_row_idx]
    print(f"  Round {int(row['round'])}: "
          f"reward={row['outer_reward']:.6f}, "
          f"target_prob={row['mean_target_probability']:.6f}, "
          f"rmse={row['realisation_rmse']:.6f}, "
          f"updated={row['policy_updated']}")

    if len(skipped_rows) > 0:
        print(f"\nRounds with RMSE-skipped updates ({len(skipped_rows)} total):")
        for _, row in skipped_rows.iterrows():
            print(f"  Round {int(row['round'])}: "
                  f"reward={row['outer_reward']:.6f}, "
                  f"rmse={row['realisation_rmse']:.6f}")
    else:
        print(f"\nNo RMSE-skipped updates.")

    # Reward vs clipped reward comparison
    raw_clipped_diff = (
        (history["outer_reward"] - history["outer_reward_clipped"]).abs().mean()
    )
    print(f"\nMean absolute difference between raw and clipped rewards: {raw_clipped_diff:.6f}")

    print("\n" + "=" * 100)


def create_visualizations(summary: dict, history: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive diagnostic plots."""
    best_round = summary["best_round_index"]
    best_reward = summary["best_reward"]

    fig = plt.figure(figsize=(16, 12))

    # 1. Reward trajectory with best round highlighted
    ax1 = plt.subplot(3, 3, 1)
    rounds = history["round"].values
    ax1.plot(rounds, history["outer_reward"].values, "o-", label="Outer reward (raw)", linewidth=2)
    ax1.axhline(y=best_reward, color="r", linestyle="--", alpha=0.7, label="Best reward")
    ax1.axvline(x=best_round, color="g", linestyle="--", alpha=0.7, label=f"Best round ({best_round})")
    ax1.scatter([best_round], [best_reward], color="red", s=100, marker="*", zorder=5)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Reward")
    ax1.set_title("Outer Reward Trajectory")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Raw vs clipped reward
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(rounds, history["outer_reward"].values, "o-", label="Raw", linewidth=2, alpha=0.7)
    ax2.plot(
        rounds,
        history["outer_reward_clipped"].values,
        "s-",
        label="Clipped",
        linewidth=2,
        alpha=0.7,
    )
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Reward")
    ax2.set_title("Raw vs Clipped Rewards")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Target probability trajectory
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(rounds, history["mean_target_probability"].values, "o-", linewidth=2, color="purple")
    ax3.axhline(y=0.25, color="r", linestyle="--", alpha=0.5, label="Chance (4 classes)")
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Target Probability")
    ax3.set_title("Gemini Target-State Probability")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. RMSE trajectory with skip markers
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(rounds, history["realisation_rmse"].values, "o-", linewidth=2, color="orange")
    ax4.axhline(
        y=summary["rmse_normal_threshold"],
        color="g",
        linestyle="--",
        alpha=0.7,
        label=f"Normal threshold ({summary.get('rmse_normal_threshold', 'N/A')})",
    )
    ax4.axhline(
        y=summary["rmse_skip_threshold"],
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Skip threshold ({summary.get('rmse_skip_threshold', 'N/A')})",
    )
    skipped = history[history["rmse_skipped"] == True]
    if len(skipped) > 0:
        ax4.scatter(
            skipped["round"].values,
            skipped["realisation_rmse"].values,
            color="red",
            s=100,
            marker="X",
            zorder=5,
            label="Skipped",
        )
    ax4.set_xlabel("Round")
    ax4.set_ylabel("RMSE")
    ax4.set_title("Realisation RMSE (Requested vs Achieved)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 5. Policy update status
    ax5 = plt.subplot(3, 3, 5)
    updated = history["policy_updated"].astype(int).values
    colors = ["red" if not u else "green" for u in history["policy_updated"].values]
    ax5.scatter(rounds, updated, c=colors, s=100, alpha=0.6)
    ax5.set_xlabel("Round")
    ax5.set_ylabel("Updated")
    ax5.set_title("Policy Update Status")
    ax5.set_ylim(-0.1, 1.1)
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(["Skipped", "Updated"])
    ax5.grid(True, alpha=0.3)

    # 6. Margin (raw vs clipped)
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(rounds, history["mean_margin"].values, "o-", label="Raw", linewidth=2, alpha=0.7)
    ax6.plot(rounds, history["mean_margin_clipped"].values, "s-", label="Clipped", linewidth=2, alpha=0.7)
    ax6.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax6.set_xlabel("Round")
    ax6.set_ylabel("Margin")
    ax6.set_title("Classification Margin (Target vs Best Competitor)")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    # 7. Advantage
    ax7 = plt.subplot(3, 3, 7)
    ax7.plot(rounds, history["advantage"].values, "o-", linewidth=2, color="brown")
    ax7.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax7.set_xlabel("Round")
    ax7.set_ylabel("Advantage")
    ax7.set_title("Policy Advantage (Reward - Baseline)")
    ax7.grid(True, alpha=0.3)

    # 8. Policy loss
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(rounds, history["policy_loss"].values, "o-", linewidth=2, color="darkblue")
    ax8.set_xlabel("Round")
    ax8.set_ylabel("Loss")
    ax8.set_title("Policy Loss")
    ax8.grid(True, alpha=0.3)

    # 9. Perceptual reward std (stability)
    ax9 = plt.subplot(3, 3, 9)
    ax9.plot(rounds, history["perceptual_reward_std"].values, "o-", linewidth=2, color="purple")
    ax9.set_xlabel("Round")
    ax9.set_ylabel("Std Dev")
    ax9.set_title("Perceptual Reward Std (VLM Instability)")
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_summary.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_dir / 'diagnostic_summary.png'}")

    # Additional: Feature evolution
    fig, axes = plt.subplots(5, 1, figsize=(12, 10))
    features = ["weight", "time", "flow_boundness", "space_indirectness", "shape_arcness"]

    for idx, feature in enumerate(features):
        ax = axes[idx]
        ax.plot(rounds, history[f"sampled_{feature}"].values, "o-", label="Sampled", linewidth=2, alpha=0.7)
        ax.plot(rounds, history[f"mean_{feature}"].values, "s-", label="Policy mean", linewidth=2, alpha=0.7)
        ax.axhline(y=summary["best_sampled_profile"][feature], color="r", linestyle="--", alpha=0.5, label="Best")
        ax.axhline(
            y=summary["final_policy_mean"][feature],
            color="g",
            linestyle="--",
            alpha=0.5,
            label="Final mean",
        )
        ax.set_ylabel(feature)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best")

    axes[-1].set_xlabel("Round")
    plt.suptitle("Feature Evolution: Sampled vs Policy Mean vs Best vs Final", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_evolution.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_dir / 'feature_evolution.png'}")

    # Best vs Final profile comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    features = list(summary["best_sampled_profile"].keys())
    best_vals = [summary["best_sampled_profile"][f] for f in features]
    final_vals = [summary["final_policy_mean"][f] for f in features]
    final_std_vals = [summary["final_policy_std"][f] for f in features]

    x = np.arange(len(features))
    width = 0.35

    ax.bar(x - width / 2, best_vals, width, label="Best profile (round 25)", alpha=0.8)
    ax.bar(
        x + width / 2,
        final_vals,
        width,
        label="Final policy mean",
        alpha=0.8,
        yerr=final_std_vals,
        capsize=5,
    )

    ax.set_ylabel("Normalized value")
    ax.set_title("Best Sampled Profile vs Final Policy Mean (with exploration bounds)")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_dir / "best_vs_final_profile.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_dir / 'best_vs_final_profile.png'}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_bandit_results.py <output_dir>")
        print()
        print("Example:")
        print("  python scripts/analyze_bandit_results.py outputs/experiment_clean_v1")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        sys.exit(1)

    print(f"Loading results from: {output_dir}")

    summary, history = load_results(output_dir)
    print_analysis(summary, history, output_dir)
    create_visualizations(summary, history, output_dir)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
