#!/usr/bin/env python3
"""
Aggregate and analyze CEM results from all gesture-state combinations.

Generates:
  1. Summary statistics table
  2. Best profiles per gesture/state
  3. Comparative visualizations
  4. Efficiency metrics
"""
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

def load_all_results() -> Dict[str, dict]:
    """Load results from all CEM training runs."""
    
    results = {}
    combinations = [
        (g, s) for g in ["point", "wave", "reach"] 
        for s in ["confident", "calm", "hesitant", "friendly", "confused", "angry"]
    ]
    
    for gesture, state in combinations:
        key = f"{gesture}_{state}"
        output_dir = OUTPUTS_DIR / f"experiment_cem_{key}"
        summary_file = output_dir / "results_summary.json"
        
        if summary_file.exists():
            try:
                with open(summary_file) as f:
                    results[key] = json.load(f)
                results[key]["_path"] = str(output_dir)
            except Exception as e:
                print(f"⚠️ Failed to load {key}: {e}")
                results[key] = None
        else:
            results[key] = None
    
    return results

def print_summary_table(results: Dict[str, dict]):
    """Print comprehensive summary table."""
    
    print("\n" + "="*110)
    print("CEM TRAINING RESULTS: ALL GESTURE-STATE COMBINATIONS")
    print("="*110)
    
    # Group by gesture
    gestures = ["point", "wave", "reach"]
    states = ["friendly", "confused", "angry"]
    
    for gesture in gestures:
        print(f"\n{'GESTURE: ' + gesture.upper():<50}")
        print("-" * 110)
        print(f"{'State':<15} {'Best Reward':<15} {'Best Round':<12} {'Final Mean':<15} {'Target Prob':<15} {'Status':<10}")
        print("-" * 110)
        
        for state in states:
            key = f"{gesture}_{state}"
            
            if results[key] is None:
                print(f"{state:<15} {'[Not Complete]':<15} {'-':<12} {'-':<15} {'-':<15} {'❌':<10}")
                continue
            
            r = results[key]
            best_reward = r.get("best_reward", 0)
            best_round = r.get("best_round_index", -1) + 1
            final_mean_reward = r.get("mean_reward_all_rounds", 0)
            target_prob = r.get("mean_target_probability_all_rounds", 0)
            
            status = "✅" if best_reward > 0.3 else "⚠️"
            
            print(f"{state:<15} {best_reward:<15.4f} {best_round:<12} {final_mean_reward:<15.4f} {target_prob:<15.4f} {status:<10}")

def get_best_profiles_per_gesture(results: Dict[str, dict]) -> Dict[str, dict]:
    """Find best performing state for each gesture."""
    
    gestures = ["point", "wave", "reach"]
    states = ["friendly", "confused", "angry"]
    
    best_per_gesture = {}
    
    for gesture in gestures:
        best_state = None
        best_reward = -float('inf')
        
        for state in states:
            key = f"{gesture}_{state}"
            if results[key] and results[key].get("best_reward", 0) > best_reward:
                best_state = state
                best_reward = results[key].get("best_reward", 0)
        
        if best_state:
            key = f"{gesture}_{best_state}"
            best_per_gesture[gesture] = {
                "state": best_state,
                "reward": results[key].get("best_reward", 0),
                "profile": results[key].get("best_sampled_profile", {}),
                "results": results[key]
            }
    
    return best_per_gesture

def create_summary_plot(results: Dict[str, dict]):
    """Create comprehensive visualization of all results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("CEM Results: All Gesture-State Combinations", fontsize=16, fontweight='bold')
    
    gestures = ["point", "wave", "reach"]
    states = ["friendly", "confused", "angry"]
    
    # 1. Best rewards heatmap
    ax = axes[0, 0]
    rewards = np.zeros((len(gestures), len(states)))
    
    for i, gesture in enumerate(gestures):
        for j, state in enumerate(states):
            key = f"{gesture}_{state}"
            if results[key]:
                rewards[i, j] = results[key].get("best_reward", 0)
    
    im = ax.imshow(rewards, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.7)
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(gestures)))
    ax.set_xticklabels(states)
    ax.set_yticklabels(gestures)
    ax.set_xlabel("State", fontweight='bold')
    ax.set_ylabel("Gesture", fontweight='bold')
    ax.set_title("Best Reward Heatmap", fontweight='bold')
    
    # Add values
    for i in range(len(gestures)):
        for j in range(len(states)):
            text = ax.text(j, i, f'{rewards[i, j]:.3f}', ha="center", va="center", 
                         color="white" if rewards[i, j] > 0.35 else "black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Best Reward")
    
    # 2. Target probability heatmap
    ax = axes[0, 1]
    target_probs = np.zeros((len(gestures), len(states)))
    
    for i, gesture in enumerate(gestures):
        for j, state in enumerate(states):
            key = f"{gesture}_{state}"
            if results[key]:
                target_probs[i, j] = results[key].get("mean_target_probability_all_rounds", 0)
    
    im = ax.imshow(target_probs, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(gestures)))
    ax.set_xticklabels(states)
    ax.set_yticklabels(gestures)
    ax.set_xlabel("State", fontweight='bold')
    ax.set_ylabel("Gesture", fontweight='bold')
    ax.set_title("Mean Target Probability Heatmap", fontweight='bold')
    
    for i in range(len(gestures)):
        for j in range(len(states)):
            text = ax.text(j, i, f'{target_probs[i, j]:.2f}', ha="center", va="center", 
                         color="white" if target_probs[i, j] > 0.5 else "black", fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Target Probability")
    
    # 3. Bar chart by gesture
    ax = axes[1, 0]
    gesture_rewards = []
    gesture_labels = []
    
    for gesture in gestures:
        rewards_list = []
        for state in states:
            key = f"{gesture}_{state}"
            if results[key]:
                rewards_list.append(results[key].get("best_reward", 0))
        if rewards_list:
            gesture_rewards.append(np.mean(rewards_list))
            gesture_labels.append(gesture)
    
    bars = ax.bar(gesture_labels, gesture_rewards, color=['#FF6B6B', '#4ECDC4', '#95E1D3'], alpha=0.7, edgecolor='black')
    ax.set_ylabel("Mean Best Reward", fontweight='bold')
    ax.set_title("Average Performance by Gesture", fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, gesture_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Bar chart by state
    ax = axes[1, 1]
    state_rewards = []
    state_labels = []
    
    for state in states:
        rewards_list = []
        for gesture in gestures:
            key = f"{gesture}_{state}"
            if results[key]:
                rewards_list.append(results[key].get("best_reward", 0))
        if rewards_list:
            state_rewards.append(np.mean(rewards_list))
            state_labels.append(state)
    
    bars = ax.bar(state_labels, state_rewards, color=['#FFE66D', '#A8E6CF', '#FF8B94', '#C7CEEA'], alpha=0.7, edgecolor='black')
    ax.set_ylabel("Mean Best Reward", fontweight='bold')
    ax.set_title("Average Performance by State", fontweight='bold')
    ax.set_ylim(0, 0.6)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, state_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "cem_all_combinations_summary.png", dpi=300, bbox_inches='tight')
    print("\n✅ Saved: cem_all_combinations_summary.png")

def create_best_profiles_chart(best_per_gesture: Dict[str, dict]):
    """Create visualization of best profiles per gesture."""
    
    if not best_per_gesture:
        print("⚠️ No results to visualize")
        return
    
    fig, axes = plt.subplots(1, len(best_per_gesture), figsize=(15, 5))
    if len(best_per_gesture) == 1:
        axes = [axes]
    
    fig.suptitle("Best Profiles Found per Gesture (Best State)", fontsize=14, fontweight='bold')
    
    features = ["weight", "time", "flow_boundness", "space_indirectness", "shape_arcness"]
    
    for idx, (gesture, data) in enumerate(best_per_gesture.items()):
        ax = axes[idx]
        
        profile = data["profile"]
        values = [profile.get(f, 0) for f in features]
        
        bars = ax.bar(range(len(features)), values, color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Feature Value [0-1]", fontweight='bold')
        ax.set_title(f"{gesture.upper()}\n({data['state']}, reward: {data['reward']:.4f})", fontweight='bold')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([f.replace('_', '\n') for f in features], fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "cem_best_profiles_per_gesture.png", dpi=300, bbox_inches='tight')
    print("✅ Saved: cem_best_profiles_per_gesture.png")

def main():
    """Generate comprehensive results analysis."""
    
    print("\n" + "="*80)
    print("AGGREGATING CEM RESULTS FROM ALL COMBINATIONS")
    print("="*80)
    
    # Load all results
    results = load_all_results()
    completed = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    print(f"\nResults found: {completed}/{total}")
    
    if completed == 0:
        print("⚠️ No completed runs found yet. Training may still be in progress.")
        return 1
    
    # Print summary
    print_summary_table(results)
    
    # Get best per gesture
    best_per_gesture = get_best_profiles_per_gesture(results)
    
    print("\n" + "="*80)
    print("BEST PERFORMING STATE PER GESTURE")
    print("="*80)
    
    for gesture, data in best_per_gesture.items():
        print(f"\n{gesture.upper()}:")
        print(f"  Best State: {data['state']}")
        print(f"  Best Reward: {data['reward']:.4f}")
        print(f"  Profile:")
        for feature, value in data['profile'].items():
            print(f"    {feature}: {value:.4f}")
    
    # Calculate statistics
    print("\n" + "="*80)
    print("AGGREGATE STATISTICS")
    print("="*80)
    
    all_rewards = [r.get("best_reward", 0) for r in results.values() if r is not None]
    all_probs = [r.get("mean_target_probability_all_rounds", 0) for r in results.values() if r is not None]
    
    if all_rewards:
        print(f"\nBest Rewards:")
        print(f"  Mean: {np.mean(all_rewards):.4f}")
        print(f"  Median: {np.median(all_rewards):.4f}")
        print(f"  Min: {np.min(all_rewards):.4f}")
        print(f"  Max: {np.max(all_rewards):.4f}")
        print(f"  Std: {np.std(all_rewards):.4f}")
    
    if all_probs:
        print(f"\nTarget Probabilities:")
        print(f"  Mean: {np.mean(all_probs):.4f}")
        print(f"  Median: {np.median(all_probs):.4f}")
        print(f"  Min: {np.min(all_probs):.4f}")
        print(f"  Max: {np.max(all_probs):.4f}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    create_summary_plot(results)
    create_best_profiles_chart(best_per_gesture)
    
    # Save comprehensive CSV
    csv_file = OUTPUTS_DIR / "cem_all_results_summary.csv"
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "gesture", "state", "best_reward", "best_round", "final_mean_reward",
            "target_probability", "num_rounds", "num_skipped_updates"
        ])
        
        for gesture in ["point", "wave", "reach"]:
            for state in ["friendly", "confused", "angry"]:
                key = f"{gesture}_{state}"
                if results[key]:
                    r = results[key]
                    writer.writerow([
                        gesture,
                        state,
                        r.get("best_reward", ""),
                        r.get("best_round_index", "") + 1 if r.get("best_round_index") is not None else "",
                        r.get("mean_reward_all_rounds", ""),
                        r.get("mean_target_probability_all_rounds", ""),
                        r.get("num_rounds_completed", ""),
                        r.get("num_skipped_rmse_updates", "")
                    ])
    
    print(f"\n✅ Saved: {csv_file}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
