#!/usr/bin/env python3
"""
Comprehensive visualization of CEM training results.
Creates plots for reward progression, profile comparison, and convergence metrics.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# Target profiles for reference
TARGET_PROFILES = {
    "confident": {
        "weight": 0.8,
        "time": 0.8,
        "flow_boundness": 0.8,
        "space_indirectness": 0.2,
        "shape_arcness": 0.3,
    },
    "calm": {
        "weight": 0.2,
        "time": 0.2,
        "flow_boundness": 0.2,
        "space_indirectness": 0.5,
        "shape_arcness": 0.4,
    },
    "hesitant": {
        "weight": 0.25,
        "time": 0.3,
        "flow_boundness": 0.4,
        "space_indirectness": 0.6,
        "shape_arcness": 0.35,
    },
    "friendly": {
        "weight": 0.5,
        "time": 0.5,
        "flow_boundness": 0.45,
        "space_indirectness": 0.55,
        "shape_arcness": 0.65,
    },
    "confused": {
        "weight": 0.2,
        "time": 0.3,
        "flow_boundness": 0.15,
        "space_indirectness": 0.75,
        "shape_arcness": 0.5,
    },
    "angry": {
        "weight": 0.85,
        "time": 0.85,
        "flow_boundness": 0.85,
        "space_indirectness": 0.2,
        "shape_arcness": 0.25,
    },
}

DIMENSIONS = ["weight", "time", "flow_boundness", "space_indirectness", "shape_arcness"]


def target_label(summary: Dict) -> str:
    if summary.get("target_state") is not None:
        return str(summary["target_state"])
    vad = summary["target_vad"]
    return (
        f"VAD ({vad['valence']:.2f}, "
        f"{vad['arousal']:.2f}, {vad['dominance']:.2f})"
    )


def reference_profile(summary: Dict) -> Dict:
    initial_validation = summary.get("initial_profile_validation") or {}
    profile = initial_validation.get("requested_profile")
    if profile is not None:
        return profile
    state = summary.get("target_state")
    if state in TARGET_PROFILES:
        return TARGET_PROFILES[state]
    return summary["final_distribution_mean"]


def format_optional_probability(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f} ({value * 100:.1f}%)"


def load_results(experiment_dir: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load training history CSV and results summary JSON."""
    history_csv = experiment_dir / "training_history.csv"
    summary_json = experiment_dir / "results_summary.json"
    
    history = pd.read_csv(history_csv)
    with open(summary_json) as f:
        summary = json.load(f)
    
    return history, summary


def plot_reward_progression(history: pd.DataFrame, summary: Dict, output_path: Path):
    """Plot reward progression over rounds."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{summary['gesture'].title()} + {target_label(summary)} | CEM Training Progress",
        fontsize=16, fontweight="bold"
    )
    
    # Reward over rounds
    ax = axes[0, 0]
    ax.plot(history["round"], history["best_round_reward"], "o-", linewidth=2, markersize=8, label="Best Reward", color="#2E86AB")
    ax.axhline(summary["best_reward"], color="#A23B72", linestyle="--", label=f"Peak: {summary['best_reward']:.4f}", linewidth=2)
    ax.fill_between(history["round"], history["min_sample_reward"], history["max_sample_reward"], alpha=0.2, color="#2E86AB", label="Sample Range")
    ax.set_xlabel("Round", fontsize=11, fontweight="bold")
    ax.set_ylabel("Reward", fontsize=11, fontweight="bold")
    ax.set_title("Reward Progression", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Target probability
    ax = axes[0, 1]
    if summary.get("target_state") is None:
        metric_column = "mean_vad_reward"
        metric_label = "Mean VAD reward"
        metric_title = "Continuous VAD target reward"
    else:
        metric_column = "mean_target_probability"
        metric_label = "Mean target probability"
        metric_title = "Target state probability"
    ax.plot(history["round"], history[metric_column], "s-", linewidth=2, markersize=7, label=metric_label, color="#F18F01")
    ax.fill_between(history["round"], 0, history[metric_column], alpha=0.2, color="#F18F01")
    ax.set_xlabel("Round", fontsize=11, fontweight="bold")
    ax.set_ylabel(metric_label, fontsize=11, fontweight="bold")
    ax.set_title(metric_title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Exploration width decay
    ax = axes[1, 0]
    exploration_column = (
        "mean_exploration_std"
        if "mean_exploration_std" in history
        else "exploration_width"
    )
    ax.plot(history["round"], history[exploration_column], "^-", linewidth=2, markersize=7, label="Exploration Width", color="#C73E1D")
    ax.set_xlabel("Round", fontsize=11, fontweight="bold")
    ax.set_ylabel("Width", fontsize=11, fontweight="bold")
    ax.set_title("Exploration Width Decay", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=10)
    
    # RMSE convergence
    ax = axes[1, 1]
    ax.plot(history["round"], history["mean_realisation_rmse"], "d-", linewidth=2, markersize=7, label="Mean RMSE", color="#06A77D")
    ax.fill_between(history["round"], 0, history["mean_realisation_rmse"], alpha=0.2, color="#06A77D")
    ax.set_xlabel("Round", fontsize=11, fontweight="bold")
    ax.set_ylabel("RMSE", fontsize=11, fontweight="bold")
    ax.set_title("Profile Realization RMSE", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_profile_comparison(summary: Dict, output_path: Path):
    """Create radar/spider plot comparing best profile vs target."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"{summary['gesture'].title()} + {target_label(summary)} | Profile Comparison",
        fontsize=14, fontweight="bold"
    )
    
    target = reference_profile(summary)
    best = summary["best_sampled_profile"]
    final_mean = summary["final_distribution_mean"]
    
    # Radar chart
    ax = axes[0]
    angles = np.linspace(0, 2 * np.pi, len(DIMENSIONS), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    target_vals = [target[d] for d in DIMENSIONS] + [target[DIMENSIONS[0]]]
    best_vals = [best[d] for d in DIMENSIONS] + [best[DIMENSIONS[0]]]
    final_vals = [final_mean[d] for d in DIMENSIONS] + [final_mean[DIMENSIONS[0]]]
    
    ax = plt.subplot(121, projection="polar")
    ax.plot(angles, target_vals, "o-", linewidth=2, label="Target Ideal", color="#2E86AB", markersize=8)
    ax.fill(angles, target_vals, alpha=0.15, color="#2E86AB")
    ax.plot(angles, best_vals, "s-", linewidth=2.5, label="Best Learned", color="#A23B72", markersize=8)
    ax.fill(angles, best_vals, alpha=0.15, color="#A23B72")
    ax.plot(angles, final_vals, "^-", linewidth=2, label="Final Mean", color="#F18F01", markersize=7, linestyle="--")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIMENSIONS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Profile Dimensions (Polar)", fontsize=12, fontweight="bold", pad=20)
    
    # Bar chart comparison
    ax = axes[1]
    x = np.arange(len(DIMENSIONS))
    width = 0.25
    
    target_vals = [target[d] for d in DIMENSIONS]
    best_vals = [best[d] for d in DIMENSIONS]
    final_vals = [final_mean[d] for d in DIMENSIONS]
    
    ax.bar(x - width, target_vals, width, label="Target Ideal", color="#2E86AB", alpha=0.8)
    ax.bar(x, best_vals, width, label="Best Learned", color="#A23B72", alpha=0.8)
    ax.bar(x + width, final_vals, width, label="Final Mean", color="#F18F01", alpha=0.8)
    
    ax.set_xlabel("Dimension", fontsize=11, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11, fontweight="bold")
    ax.set_title("Dimension Comparison (Bars)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(DIMENSIONS, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_distribution_convergence(history: pd.DataFrame, summary: Dict, output_path: Path):
    """Plot how distribution means and stds converge over time."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"{summary['gesture'].title()} + {target_label(summary)} | Distribution Convergence",
        fontsize=14, fontweight="bold"
    )
    target = reference_profile(summary)
    
    for idx, dim in enumerate(DIMENSIONS):
        ax = axes[idx // 3, idx % 3]
        
        mean_col = f"mean_{dim}"
        std_col = f"mean_{dim}"  # We use mean values; would need std values from rounds
        
        # Plot mean trajectory
        ax.plot(history["round"], history[mean_col], "o-", linewidth=2.5, markersize=7, color="#2E86AB", label="Mean")
        
        # Add target ideal as reference
        target_val = target[dim]
        ax.axhline(target_val, color="#A23B72", linestyle="--", linewidth=2, label=f"Target: {target_val:.2f}")
        
        # Add best value
        best_val = summary["best_sampled_profile"][dim]
        ax.axhline(best_val, color="#F18F01", linestyle=":", linewidth=2, label=f"Best: {best_val:.2f}")
        
        ax.set_xlabel("Round", fontsize=10, fontweight="bold")
        ax.set_ylabel("Value", fontsize=10, fontweight="bold")
        ax.set_title(dim.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_summary_table(summary: Dict, output_path: Path):
    """Create a summary statistics table as an image."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")
    
    # Prepare data
    data = [
        ["Metric", "Value"],
        ["Gesture", summary["gesture"].upper()],
        ["Target", target_label(summary)],
        ["", ""],
        ["Training Stats", ""],
        ["Total Rounds", str(summary["num_rounds_completed"])],
        ["Best Round", f"Round {summary['best_round_index']}"],
        ["Best Reward", f"{summary['best_reward']:.6f}"],
        ["Mean Reward (All Rounds)", f"{summary['mean_reward_all_rounds']:.6f}"],
        ["Mean Reward (Last 5)", f"{summary['mean_reward_last_5_rounds']:.6f}"],
        ["", ""],
        ["Perceptual Metrics", ""],
        ["Mean Target Prob (All)", format_optional_probability(summary["mean_target_probability_all_rounds"])],
        ["Mean Target Prob (Last 5)", format_optional_probability(summary["mean_target_probability_last_5_rounds"])],
        ["", ""],
        ["CEM Config", ""],
        ["Elite Fraction", f"{summary['cem_elite_fraction']}"],
        ["Initial Width", f"{summary['cem_initial_width']}"],
        ["Exploration Decay Rate", f"{summary['exploration_decay_rate']}"],
        ["Reward Margin Mode", summary["reward_margin_mode"]],
        ["", ""],
        ["Best Profile", ""],
    ]
    
    # Add dimensions
    target = reference_profile(summary)
    for dim in DIMENSIONS:
        val = summary["best_sampled_profile"][dim]
        target_val = target[dim]
        delta = val - target_val
        sign = "+" if delta >= 0 else ""
        data.append([f"  {dim.replace('_', ' ').title()}", f"{val:.4f} (delta {sign}{delta:.4f})"])
    
    # Create table
    table = ax.table(cellText=data, cellLoc="left", loc="center", colWidths=[0.4, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    # Style section headers
    section_rows = [4, 11, 15, 20]
    for row in section_rows:
        if row < len(data):
            table[(row, 0)].set_facecolor("#E8E8E8")
            table[(row, 1)].set_facecolor("#E8E8E8")
            table[(row, 0)].set_text_props(weight="bold")
            table[(row, 1)].set_text_props(weight="bold")
    
    # Alternate row colors
    for i in range(1, len(data)):
        if i not in section_rows and data[i][0] != "":
            color = "#F5F5F5" if i % 2 == 0 else "white"
            table[(i, 0)].set_facecolor(color)
            table[(i, 1)].set_facecolor(color)
    
    plt.title(
        f"{summary['gesture'].title()} + {target_label(summary)} | Results Summary",
        fontsize=14, fontweight="bold", pad=20
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    experiment_dir = Path("outputs/experiment_cem_wave_friendly")
    
    if not experiment_dir.exists():
        print(f"Experiment directory not found: {experiment_dir}")
        return
    
    # Load data
    print(f"Loading results from {experiment_dir}...")
    history, summary = load_results(experiment_dir)
    
    # Create visualizations
    output_dir = experiment_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    print("\nGenerating visualizations...")
    plot_reward_progression(history, summary, output_dir / "01_reward_progression.png")
    plot_profile_comparison(summary, output_dir / "02_profile_comparison.png")
    plot_distribution_convergence(history, summary, output_dir / "03_distribution_convergence.png")
    plot_summary_table(summary, output_dir / "04_summary_table.png")
    
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
