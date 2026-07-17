"""
Compare CEM training results across multiple emotional states for a gesture
"""
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

def load_results(output_dir):
    """Load results summary from directory"""
    with open(output_dir / 'results_summary.json', 'r') as f:
        return json.load(f)

def load_training_history(output_dir):
    """Load training history CSV"""
    history = []
    with open(output_dir / 'training_history.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                if key != 'round':
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            history.append(row)
    return history

def compare_emotional_states(gesture, states_dir):
    """Compare training results across emotional states"""
    
    # Load results for each state
    results = {}
    histories = {}
    
    for state in ['friendly', 'confused', 'angry']:
        result_dir = states_dir / f'experiment_cem_{gesture}_{state}'
        if result_dir.exists():
            try:
                results[state] = load_results(result_dir)
                histories[state] = load_training_history(result_dir)
            except FileNotFoundError:
                print(f"⚠ Results not found for {state}")
    
    if not results:
        print("❌ No results found")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPARISON: {gesture.upper()} EMOTIONAL STATES")
    print(f"{'='*80}\n")
    
    # Print summary table
    print(f"{'State':<15} {'Best Reward':<15} {'Best Round':<12} {'Final Mean':<15} {'Target Prob':<12}")
    print("-" * 70)
    
    for state in ['friendly', 'confused', 'angry']:
        if state in results:
            r = results[state]
            best_reward = r['best_reward']
            best_round = r['best_round_index']
            final_mean_weight = r['final_distribution_mean']['weight']
            target_prob = r['mean_target_probability_all_rounds']
            print(f"{state:<15} {best_reward:>14.6f} {best_round:>11} "
                  f"{final_mean_weight:>14.4f} {target_prob:>11.4f}")
    
    print()
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{gesture.upper()} Gesture - Emotional States Comparison", 
                 fontsize=16, fontweight='bold')
    
    states_list = sorted([s for s in results.keys()])
    colors = {'friendly': '#2ecc71', 'confused': '#f39c12', 'angry': '#e74c3c'}
    
    # 1. Best reward by state
    ax = axes[0, 0]
    rewards = [results[s]['best_reward'] for s in states_list]
    bars = ax.bar(states_list, rewards, color=[colors[s] for s in states_list], 
                   alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Best Reward', fontsize=12, fontweight='bold')
    ax.set_title('Best Reward Achieved', fontweight='bold', fontsize=13)
    ax.set_ylim([0, max(rewards) * 1.15])
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.4f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Target probability by state
    ax = axes[0, 1]
    target_probs = [results[s]['mean_target_probability_all_rounds'] for s in states_list]
    bars = ax.bar(states_list, target_probs, color=[colors[s] for s in states_list],
                   alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Mean Target Probability', fontsize=12, fontweight='bold')
    ax.set_title('Model Confidence (Gemini)', fontweight='bold', fontsize=13)
    ax.set_ylim([0, max(target_probs) * 1.15])
    for bar, prob in zip(bars, target_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Profile dimension comparison (radar-like)
    ax = axes[1, 0]
    dimensions = ['weight', 'time', 'flow_boundness', 'space_indirectness', 'shape_arcness']
    x_pos = np.arange(len(dimensions))
    width = 0.25
    
    for i, state in enumerate(states_list):
        profile = results[state]['best_sampled_profile']
        values = [profile[d] for d in dimensions]
        ax.bar(x_pos + i * width, values, width, label=state, 
               color=colors[state], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Profile Dimension', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Value', fontsize=12, fontweight='bold')
    ax.set_title('Best Profiles - Dimension Comparison', fontweight='bold', fontsize=13)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([d.replace('_', '\n') for d in dimensions], fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Convergence curves
    ax = axes[1, 1]
    for state in states_list:
        if state in histories:
            history = histories[state]
            rounds = [h['round'] for h in history]
            best_rewards = [h['best_round_reward'] for h in history]
            ax.plot(rounds, best_rewards, 'o-', linewidth=2.5, markersize=5,
                   label=state, color=colors[state])
    
    ax.set_xlabel('Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Reward', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Over Rounds', fontweight='bold', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = states_dir / f'comparison_{gesture}_states.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path.name}\n")

if __name__ == '__main__':
    # Output directory
    outputs_dir = Path(__file__).parent / 'outputs'
    
    # Compare wave gesture across states
    compare_emotional_states('wave', outputs_dir)
    
    print("✅ Comparison complete!")
