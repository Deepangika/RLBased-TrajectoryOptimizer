"""
Visualize CEM training results for Laban gesture profiles
"""
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Configure styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def load_results(output_dir):
    """Load results from CEM training output directory"""
    output_path = Path(output_dir)
    
    # Load JSON summary
    with open(output_path / 'results_summary.json', 'r') as f:
        summary = json.load(f)
    
    # Load training history CSV
    history = []
    with open(output_path / 'training_history.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key != 'round':
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        pass
            history.append(row)
    
    return summary, history

def plot_convergence_analysis(summary, history, output_dir):
    """Plot convergence metrics over rounds"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{summary['gesture'].upper()} - {summary['target_state'].title()} State - Convergence Analysis", 
                 fontsize=16, fontweight='bold')
    
    rounds = [h['round'] for h in history]
    
    # 1. Best reward over time (should be constant after discovery)
    ax = axes[0, 0]
    best_rewards = [h['best_round_reward'] for h in history]
    ax.plot(rounds, best_rewards, 'o-', linewidth=2, markersize=6, color='#2ecc71')
    ax.axhline(y=summary['best_reward'], color='red', linestyle='--', alpha=0.5, label='Final Best')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Best Reward', fontsize=11)
    ax.set_title('Best Reward Trajectory (discovered in round 1)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Exploration width decay
    ax = axes[0, 1]
    exploration_widths = [h['exploration_width'] for h in history]
    ax.plot(rounds, exploration_widths, 'o-', linewidth=2, markersize=6, color='#3498db')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Exploration Width', fontsize=11)
    ax.set_title('CEM Exploration Decay', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Mean vs Max reward per round
    ax = axes[1, 0]
    mean_rewards = [h['mean_sample_reward'] for h in history]
    max_rewards = [h['max_sample_reward'] for h in history]
    ax.fill_between(rounds, mean_rewards, max_rewards, alpha=0.3, color='#9b59b6', label='Sample Range')
    ax.plot(rounds, mean_rewards, 'o-', linewidth=2, markersize=5, color='#8e44ad', label='Mean')
    ax.plot(rounds, max_rewards, 's-', linewidth=2, markersize=5, color='#e74c3c', label='Max')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('Per-Round Sample Reward Statistics', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 4. Realization RMSE (tracking target profile match)
    ax = axes[1, 1]
    rmse = [h['mean_realisation_rmse'] for h in history]
    ax.plot(rounds, rmse, 'o-', linewidth=2, markersize=6, color='#e67e22')
    ax.fill_between(rounds, rmse, alpha=0.2, color='#e67e22')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Mean RMSE', fontsize=11)
    ax.set_title('Profile Realization Error', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'convergence_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: convergence_analysis.png")

def plot_dimension_evolution(summary, history, output_dir):
    """Plot how each profile dimension evolved"""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle(f"{summary['gesture'].upper()} - {summary['target_state'].title()} - Dimension Evolution", 
                 fontsize=14, fontweight='bold')
    
    rounds = [h['round'] for h in history]
    dimensions = ['weight', 'time', 'flow_boundness', 'space_indirectness', 'shape_arcness']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    best_profile = summary['best_sampled_profile']
    final_mean = summary['final_distribution_mean']
    final_std = summary['final_distribution_std']
    
    for idx, (ax, dim, color) in enumerate(zip(axes, dimensions, colors)):
        # Extract mean values for this dimension
        dim_key = f'mean_{dim}'
        means = [h[dim_key] for h in history]
        
        # Plot evolution
        ax.plot(rounds, means, 'o-', linewidth=2.5, markersize=6, color=color, label='Mean', zorder=3)
        
        # Mark best profile value
        best_val = best_profile[dim]
        ax.axhline(y=best_val, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Best Found')
        
        # Mark final mean ± std
        final_val = final_mean[dim]
        final_std_val = final_std[dim]
        ax.axhline(y=final_val, color='red', linestyle=':', linewidth=1.5, alpha=0.7, label='Final Mean')
        ax.fill_between(rounds, final_val - final_std_val, final_val + final_std_val, 
                        alpha=0.15, color='red')
        
        ax.set_xlabel('Round', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_title(dim.replace('_', ' ').title(), fontweight='bold', fontsize=11)
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.2)
        if idx == 0:
            ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'dimension_evolution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: dimension_evolution.png")

def plot_profile_comparison(summary, output_dir):
    """Radar/spider chart comparing best vs final mean profile"""
    dimensions = ['weight', 'time', 'flow_boundness', 'space_indirectness', 'shape_arcness']
    
    best_vals = [summary['best_sampled_profile'][d] for d in dimensions]
    final_vals = [summary['final_distribution_mean'][d] for d in dimensions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
    fig.suptitle(f"{summary['gesture'].upper()} - {summary['target_state'].title()} - Profile Comparison", 
                 fontsize=14, fontweight='bold')
    
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    
    best_vals_plot = best_vals + best_vals[:1]
    final_vals_plot = final_vals + final_vals[:1]
    
    # Plot 1: Best profile
    ax1.plot(angles, best_vals_plot, 'o-', linewidth=2.5, markersize=8, color='#2ecc71', label='Best Found')
    ax1.fill(angles, best_vals_plot, alpha=0.25, color='#2ecc71')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels([d.replace('_', '\n').title() for d in dimensions], fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_title('Best Profile (Round 1)', fontweight='bold', pad=20)
    ax1.grid(True)
    
    # Plot 2: Final mean distribution
    ax2.plot(angles, final_vals_plot, 'o-', linewidth=2.5, markersize=8, color='#e74c3c', label='Final Mean')
    ax2.fill(angles, final_vals_plot, alpha=0.25, color='#e74c3c')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([d.replace('_', '\n').title() for d in dimensions], fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title('Final Distribution Mean', fontweight='bold', pad=20)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'profile_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: profile_comparison.png")

def plot_exploration_analysis(summary, history, output_dir):
    """Analyze exploration vs exploitation"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{summary['gesture'].upper()} - {summary['target_state'].title()} - Exploration Analysis", 
                 fontsize=16, fontweight='bold')
    
    rounds = [h['round'] for h in history]
    
    # 1. Target probability over rounds
    ax = axes[0, 0]
    target_prob = [h['mean_target_probability'] for h in history]
    ax.bar(rounds, target_prob, color='#3498db', alpha=0.7, edgecolor='#2c3e50', linewidth=1.5)
    ax.axhline(y=summary['mean_target_probability_all_rounds'], color='red', linestyle='--', 
               linewidth=2, alpha=0.7, label=f"Avg: {summary['mean_target_probability_all_rounds']:.3f}")
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Target Probability', fontsize=11)
    ax.set_title('Gemini Model\'s Target State Confidence', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Number of elites over time
    ax = axes[0, 1]
    num_elites = [h['num_elites'] for h in history]
    ax.plot(rounds, num_elites, 'o-', linewidth=2.5, markersize=7, color='#9b59b6')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Number of Elites', fontsize=11)
    ax.set_title('Elite Population Size', fontweight='bold')
    ax.set_ylim([0, max(num_elites) + 1])
    ax.grid(True, alpha=0.3)
    
    # 3. Elite reward quality
    ax = axes[1, 0]
    elite_rewards = [h['best_elite_reward'] for h in history]
    ax.plot(rounds, elite_rewards, 'o-', linewidth=2.5, markersize=7, color='#e74c3c')
    ax.fill_between(rounds, elite_rewards, alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Best Elite Reward', fontsize=11)
    ax.set_title('Elite Population Best Reward', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. Exploration width vs target probability
    ax = axes[1, 1]
    exploration_widths = [h['exploration_width'] for h in history]
    ax2 = ax.twinx()
    
    line1 = ax.plot(rounds, exploration_widths, 'o-', linewidth=2.5, markersize=6, 
                    color='#3498db', label='Exploration Width')
    line2 = ax2.plot(rounds, target_prob, 's-', linewidth=2.5, markersize=6, 
                     color='#2ecc71', label='Target Prob')
    
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Exploration Width', fontsize=11, color='#3498db')
    ax2.set_ylabel('Target Probability', fontsize=11, color='#2ecc71')
    ax.set_title('Exploration Decay vs Model Confidence', fontweight='bold')
    ax.tick_params(axis='y', labelcolor='#3498db')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    ax.grid(True, alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'exploration_analysis.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: exploration_analysis.png")

def create_summary_report(summary, history, output_dir):
    """Generate a text summary report"""
    report = f"""
{'='*80}
CEM TRAINING RESULTS REPORT
{'='*80}

CONFIGURATION
  Gesture: {summary['gesture']}
  Target State: {summary['target_state']}
  Rounds: {summary['num_rounds_completed']}
  Samples per Round: {summary['samples_per_round']}
  Repeats per Sample: {summary['repeats_per_round']}
  Total Gemini Evaluations: {summary['total_gemini_evals']}
  CEM Elite Fraction: {summary['cem_elite_fraction']}
  Initial Width: {summary['cem_initial_width']}
  Exploration Decay Rate: {summary['exploration_decay_rate']}

PERFORMANCE SUMMARY
  Best Reward: {summary['best_reward']:.6f}
  Best Round: {summary['best_round_index']}
  Mean Reward (All Rounds): {summary['mean_reward_all_rounds']:.6f}
  Mean Reward (Last 5 Rounds): {summary['mean_reward_last_5_rounds']:.6f}
  Target Probability (Mean): {summary['mean_target_probability_all_rounds']:.6f}
  Target Probability (Last 5): {summary['mean_target_probability_last_5_rounds']:.6f}
  Elites at End: {summary['num_elites_at_end']}

BEST PROFILE FOUND (ROUND {summary['best_round_index']})
  Weight: {summary['best_sampled_profile']['weight']:.6f}
  Time: {summary['best_sampled_profile']['time']:.6f}
  Flow Boundness: {summary['best_sampled_profile']['flow_boundness']:.6f}
  Space Indirectness: {summary['best_sampled_profile']['space_indirectness']:.6f}
  Shape Arcness: {summary['best_sampled_profile']['shape_arcness']:.6f}

FINAL DISTRIBUTION (AFTER 15 ROUNDS)
  Mean:
    Weight: {summary['final_distribution_mean']['weight']:.6f}
    Time: {summary['final_distribution_mean']['time']:.6f}
    Flow Boundness: {summary['final_distribution_mean']['flow_boundness']:.6f}
    Space Indirectness: {summary['final_distribution_mean']['space_indirectness']:.6f}
    Shape Arcness: {summary['final_distribution_mean']['shape_arcness']:.6f}
  
  Exploration Std (Width):
    Weight: {summary['final_distribution_std']['weight']:.6f}
    Time: {summary['final_distribution_std']['time']:.6f}
    Flow Boundness: {summary['final_distribution_std']['flow_boundness']:.6f}
    Space Indirectness: {summary['final_distribution_std']['space_indirectness']:.6f}
    Shape Arcness: {summary['final_distribution_std']['shape_arcness']:.6f}

KEY INSIGHTS
  1. Best profile discovered IMMEDIATELY in Round 1
  2. No improvement found across remaining 14 rounds
  3. Exploration properly decayed from {summary['cem_initial_width']} to ~{history[-1]['exploration_width']:.4f}
  4. Model confidence (target probability) relatively stable: {summary['mean_target_probability_all_rounds']:.3f}
  5. 2 elite profiles maintained throughout training
  6. Algorithm successfully explored but elite remained optimal

COMPARISON POINTS FOR OTHER STATES
  - Reward baseline: {summary['best_reward']:.6f}
  - Profile dimensions: See radar chart
  - Exploration behavior: See convergence analysis

{'='*80}
"""
    
    report_path = Path(output_dir) / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Saved: analysis_report.txt")
    print(report)

if __name__ == '__main__':
    # Output directory
    output_dir = Path(__file__).parent / 'outputs' / 'experiment_cem_wave_friendly'
    
    # Load data
    summary, history = load_results(output_dir)
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations for {summary['gesture']} - {summary['target_state']}...\n")
    
    plot_convergence_analysis(summary, history, output_dir)
    plot_dimension_evolution(summary, history, output_dir)
    plot_profile_comparison(summary, output_dir)
    plot_exploration_analysis(summary, history, output_dir)
    create_summary_report(summary, history, output_dir)
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print(f"   - convergence_analysis.png")
    print(f"   - dimension_evolution.png")
    print(f"   - profile_comparison.png")
    print(f"   - exploration_analysis.png")
    print(f"   - analysis_report.txt\n")
