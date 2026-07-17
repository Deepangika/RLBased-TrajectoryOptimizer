#!/usr/bin/env python3
"""
Generate best profile GIF with Laban dimension annotations.
Creates a high-quality visualization showing:
  - Animated arm trajectory (reference vs best learned)
  - Dimension values overlaid with convergence info
  - Target comparison
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in [PROJECT_ROOT, SRC_DIR]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import robust_laban_normalisation_balanced_3gestures as laban
from laban_rl.optimiser_api import optimise_laban_target


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
DIMENSION_LABELS = {
    "weight": "Weight",
    "time": "Time",
    "flow_boundness": "Flow Boundness",
    "space_indirectness": "Space Indirectness",
    "shape_arcness": "Shape Arcness",
}


def compute_equal_axes(points_list, padding: float = 0.02):
    """Compute equal axis limits for 2D plots."""
    all_points = np.vstack(points_list)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    max_range = max(x_max - x_min, y_max - y_min, 1e-6)
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    half = 0.5 * max_range + padding
    return (x_mid - half, x_mid + half), (y_mid - half, y_mid + half)


def generate_best_profile_gif(
    experiment_dir: Path,
    output_path: Path,
    gesture: str = "wave",
):
    """Generate animated GIF of best learned profile with dimension annotations."""
    
    # Load results
    summary_file = experiment_dir / "results_summary.json"
    with open(summary_file) as f:
        summary = json.load(f)
    
    best_profile = summary["best_sampled_profile"]
    target_state = summary["target_state"]
    best_reward = summary["best_reward"]
    
    target_profile = TARGET_PROFILES[target_state]
    
    print(f"🎨 Generating GIF for {gesture} + {target_state}")
    print(f"   Best reward: {best_reward:.4f}")
    print(f"   Profile: {best_profile}")
    
    # Generate trajectory with best profile
    print(f"   Optimizing trajectory...")
    arm = laban.ArmConfig()
    
    # Create temp directory for optimization
    temp_dir = experiment_dir / "profile_optimization"
    temp_dir.mkdir(exist_ok=True)
    
    result = optimise_laban_target(
        gesture=gesture,
        target_state=target_state,
        target_profile=best_profile,
        out_dir=temp_dir,
    )
    
    q_ref = result.q_ref
    q_var = result.q_var
    
    # Subsample frames
    step = 4
    frame_indices = list(range(0, len(q_ref), step))
    if frame_indices[-1] != len(q_ref) - 1:
        frame_indices.append(len(q_ref) - 1)
    frame_indices = np.array(frame_indices, dtype=int)
    
    q_ref_sub = q_ref[frame_indices]
    q_var_sub = q_var[frame_indices]
    
    # Compute kinematics
    shoulder_ref, elbow_ref, wrist_ref = laban.forward_kinematics_2link(
        q_ref_sub, l1=arm.l1, l2=arm.l2
    )
    shoulder_var, elbow_var, wrist_var = laban.forward_kinematics_2link(
        q_var_sub, l1=arm.l1, l2=arm.l2
    )
    
    t_full = np.linspace(0.0, arm.duration, len(q_ref))
    t = t_full[frame_indices]
    T = len(q_ref_sub)
    
    xlim, ylim = compute_equal_axes(
        [shoulder_ref, elbow_ref, wrist_ref, shoulder_var, elbow_var, wrist_var],
        padding=0.08,
    )
    
    # Create figure with larger size for annotations
    fig = plt.figure(figsize=(14, 8))
    
    # Left: Arm animation
    ax_arm = plt.subplot(1, 2, 1)
    ax_arm.set_aspect("equal")
    ax_arm.set_xlim(*xlim)
    ax_arm.set_ylim(*ylim)
    ax_arm.set_xlabel("x position (m)", fontsize=10, fontweight="bold")
    ax_arm.set_ylabel("y position (m)", fontsize=10, fontweight="bold")
    ax_arm.set_title("Arm Animation", fontsize=12, fontweight="bold")
    ax_arm.grid(True, alpha=0.2)
    
    ref_line, = ax_arm.plot([], [], marker="o", markersize=6, label="Reference", color="#2E86AB", linewidth=2)
    var_line, = ax_arm.plot([], [], marker="o", markersize=6, label="Learned", color="#A23B72", linewidth=2)
    ref_trace, = ax_arm.plot([], [], alpha=0.3, color="#2E86AB")
    var_trace, = ax_arm.plot([], [], alpha=0.3, color="#A23B72")
    time_text = ax_arm.text(0.02, 0.95, "", transform=ax_arm.transAxes, va="top", fontsize=10, fontweight="bold")
    ax_arm.legend(fontsize=10, loc="upper right")
    
    # Right: Dimension display
    ax_dims = plt.subplot(1, 2, 2)
    ax_dims.axis("off")
    
    def init():
        ref_line.set_data([], [])
        var_line.set_data([], [])
        ref_trace.set_data([], [])
        var_trace.set_data([], [])
        time_text.set_text("")
        return ref_line, var_line, ref_trace, var_trace, time_text
    
    def update(frame):
        # Update arm
        ref_xs = [shoulder_ref[frame, 0], elbow_ref[frame, 0], wrist_ref[frame, 0]]
        ref_ys = [shoulder_ref[frame, 1], elbow_ref[frame, 1], wrist_ref[frame, 1]]
        var_xs = [shoulder_var[frame, 0], elbow_var[frame, 0], wrist_var[frame, 0]]
        var_ys = [shoulder_var[frame, 1], elbow_var[frame, 1], wrist_var[frame, 1]]
        
        ref_line.set_data(ref_xs, ref_ys)
        var_line.set_data(var_xs, var_ys)
        ref_trace.set_data(wrist_ref[:frame + 1, 0], wrist_ref[:frame + 1, 1])
        var_trace.set_data(wrist_var[:frame + 1, 0], wrist_var[:frame + 1, 1])
        time_text.set_text(f"t = {t[frame]:.2f} s")
        
        # Clear and redraw dimensions panel
        ax_dims.clear()
        ax_dims.axis("off")
        
        # Title
        title_y = 0.95
        ax_dims.text(
            0.5, title_y, f"{gesture.upper()} + {target_state.upper()}",
            ha="center", fontsize=14, fontweight="bold",
            transform=ax_dims.transAxes
        )
        
        # Reward info
        ax_dims.text(
            0.05, title_y - 0.08, f"Best Reward: {best_reward:.4f}",
            fontsize=11, fontweight="bold", color="#06A77D",
            transform=ax_dims.transAxes
        )
        
        # Dimension comparison table
        y_pos = title_y - 0.15
        
        # Header
        ax_dims.text(0.05, y_pos, "Dimension", fontsize=10, fontweight="bold", transform=ax_dims.transAxes)
        ax_dims.text(0.35, y_pos, "Learned", fontsize=10, fontweight="bold", transform=ax_dims.transAxes)
        ax_dims.text(0.55, y_pos, "Target", fontsize=10, fontweight="bold", transform=ax_dims.transAxes)
        ax_dims.text(0.72, y_pos, "Δ", fontsize=10, fontweight="bold", transform=ax_dims.transAxes)
        
        y_pos -= 0.06
        
        # Draw lines for each dimension
        for dim in DIMENSIONS:
            learned_val = best_profile[dim]
            target_val = target_profile[dim]
            delta = learned_val - target_val
            
            # Color based on how close to target
            if abs(delta) < 0.05:
                color = "#06A77D"  # Green - very close
            elif abs(delta) < 0.15:
                color = "#F18F01"  # Orange - moderate
            else:
                color = "#C73E1D"  # Red - far
            
            # Dimension name
            ax_dims.text(
                0.05, y_pos, DIMENSION_LABELS[dim],
                fontsize=9, transform=ax_dims.transAxes, color=color, fontweight="bold"
            )
            
            # Learned value with bar
            ax_dims.text(
                0.35, y_pos, f"{learned_val:.3f}",
                fontsize=9, transform=ax_dims.transAxes, color=color, fontweight="bold"
            )
            
            # Target value
            ax_dims.text(
                0.55, y_pos, f"{target_val:.3f}",
                fontsize=9, transform=ax_dims.transAxes, color="#2E86AB"
            )
            
            # Delta with sign
            sign = "+" if delta >= 0 else ""
            ax_dims.text(
                0.72, y_pos, f"{sign}{delta:.3f}",
                fontsize=9, transform=ax_dims.transAxes, color=color, fontweight="bold"
            )
            
            y_pos -= 0.06
        
        # RMSE
        y_pos -= 0.02
        rmse = result.realisation_rmse
        ax_dims.text(
            0.05, y_pos, "Profile RMSE",
            fontsize=10, fontweight="bold", transform=ax_dims.transAxes
        )
        ax_dims.text(
            0.35, y_pos, f"{rmse:.4f}",
            fontsize=10, fontweight="bold", transform=ax_dims.transAxes,
            color="#06A77D" if rmse < 0.1 else "#F18F01"
        )
        
        # Optimization info at bottom
        y_pos = 0.05
        ax_dims.text(
            0.05, y_pos,
            f"Inner Reward: {result.inner_reward:.4f}",
            fontsize=9, transform=ax_dims.transAxes, style="italic"
        )
        
        return ref_line, var_line, ref_trace, var_trace, time_text
    
    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=60,
        blit=False,
    )
    
    print(f"   Saving GIF...")
    anim.save(output_path, writer=PillowWriter(fps=12))
    plt.close(fig)
    
    print(f"✅ Saved: {output_path}")


def main():
    experiment_dir = Path("outputs/experiment_cem_wave_friendly")
    
    if not experiment_dir.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return
    
    output_dir = experiment_dir / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "best_profile_animation.gif"
    
    generate_best_profile_gif(
        experiment_dir=experiment_dir,
        output_path=output_path,
        gesture="wave",
    )


if __name__ == "__main__":
    main()
