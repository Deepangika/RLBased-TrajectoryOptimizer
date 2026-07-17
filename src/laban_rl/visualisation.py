"""
Visualisation and output-saving helpers.

This file handles:
    - best_variant.npz
    - best_summary.txt
    - feature comparison plot
    - joint trajectory plot
    - wrist path plot
    - arm comparison GIF

Keeping this separate prevents plotting code from cluttering the
reward, environment, and training logic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import robust_laban_normalisation_balanced_3gestures as laban

from .config import FEATURE_KEYS
from .features import feature_dict_to_array, target_dict_to_array


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


def save_feature_comparison_plot(result: dict, out_dir: Path) -> Path:
    """Save bar plot comparing reference, variant, and target features."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_arr = target_dict_to_array(result["target_profile"])
    ref_arr = feature_dict_to_array(result["ref_norm"])
    var_arr = feature_dict_to_array(result["var_norm"])

    x = np.arange(len(FEATURE_KEYS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, ref_arr, width, label="Reference")
    ax.bar(x, var_arr, width, label="Variant")
    ax.bar(x + width, target_arr, width, label="Target")

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_KEYS, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Normalised feature value")
    ax.set_title(
        f"Feature comparison: {result['gesture_type']} → {result['target_name']}"
    )
    ax.grid(True, axis="y")
    ax.legend()

    fig.tight_layout()

    path = out_dir / "feature_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def save_joint_plot(result: dict, out_dir: Path, arm: laban.ArmConfig) -> Path:
    """Save joint trajectory comparison plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_ref = result["q_ref"]
    q_var = result["q_var"]

    t = np.linspace(0.0, arm.duration, arm.n_points)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, q_ref[:, 0], label="Reference shoulder")
    ax.plot(t, q_var[:, 0], label="Variant shoulder")
    ax.plot(t, q_ref[:, 1], label="Reference elbow")
    ax.plot(t, q_var[:, 1], label="Variant elbow")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint angle (rad)")
    ax.set_title("Joint trajectory comparison")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    path = out_dir / "joint_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def save_wrist_path_plot(result: dict, out_dir: Path, arm: laban.ArmConfig) -> Path:
    """Save Cartesian wrist path comparison plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_ref = result["q_ref"]
    q_var = result["q_var"]

    _, _, wrist_ref = laban.forward_kinematics_2link(q_ref, l1=arm.l1, l2=arm.l2)
    _, _, wrist_var = laban.forward_kinematics_2link(q_var, l1=arm.l1, l2=arm.l2)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(wrist_ref[:, 0], wrist_ref[:, 1], label="Reference")
    ax.plot(wrist_var[:, 0], wrist_var[:, 1], label="Variant")
    ax.scatter([wrist_ref[0, 0]], [wrist_ref[0, 1]], label="Start")
    ax.scatter([wrist_ref[-1, 0]], [wrist_ref[-1, 1]], marker="x", label="Reference end")
    ax.scatter([wrist_var[-1, 0]], [wrist_var[-1, 1]], marker="x", label="Variant end")

    xlim, ylim = compute_equal_axes([wrist_ref, wrist_var], padding=0.02)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect("equal")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Wrist path comparison")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    path = out_dir / "wrist_path_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)

    return path


def save_arm_gif(
    q_ref: np.ndarray,
    q_var: np.ndarray,
    path: str | Path,
    arm: laban.ArmConfig,
) -> Path:
    """Save reference-vs-variant arm GIF."""
    path = Path(path)

    step = 4
    frame_indices = list(range(0, len(q_ref), step))

    if frame_indices[-1] != len(q_ref) - 1:
        frame_indices.append(len(q_ref) - 1)

    frame_indices = np.array(frame_indices, dtype=int)

    q_ref_sub = q_ref[frame_indices]
    q_var_sub = q_var[frame_indices]

    shoulder_ref, elbow_ref, wrist_ref = laban.forward_kinematics_2link(
        q_ref_sub,
        l1=arm.l1,
        l2=arm.l2,
    )

    shoulder_var, elbow_var, wrist_var = laban.forward_kinematics_2link(
        q_var_sub,
        l1=arm.l1,
        l2=arm.l2,
    )

    t_full = np.linspace(0.0, arm.duration, len(q_ref))
    t = t_full[frame_indices]
    T = len(q_ref_sub)

    xlim, ylim = compute_equal_axes(
        [shoulder_ref, elbow_ref, wrist_ref, shoulder_var, elbow_var, wrist_var],
        padding=0.08,
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Reference vs styled 2D arm")
    ax.grid(True)

    ref_line, = ax.plot([], [], marker="o", label="Reference")
    var_line, = ax.plot([], [], marker="o", label="Variant")
    ref_trace, = ax.plot([], [], alpha=0.5)
    var_trace, = ax.plot([], [], alpha=0.5)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    ax.legend()

    def init():
        ref_line.set_data([], [])
        var_line.set_data([], [])
        ref_trace.set_data([], [])
        var_trace.set_data([], [])
        time_text.set_text("")
        return ref_line, var_line, ref_trace, var_trace, time_text

    def update(frame):
        ref_xs = [
            shoulder_ref[frame, 0],
            elbow_ref[frame, 0],
            wrist_ref[frame, 0],
        ]
        ref_ys = [
            shoulder_ref[frame, 1],
            elbow_ref[frame, 1],
            wrist_ref[frame, 1],
        ]

        var_xs = [
            shoulder_var[frame, 0],
            elbow_var[frame, 0],
            wrist_var[frame, 0],
        ]
        var_ys = [
            shoulder_var[frame, 1],
            elbow_var[frame, 1],
            wrist_var[frame, 1],
        ]

        ref_line.set_data(ref_xs, ref_ys)
        var_line.set_data(var_xs, var_ys)
        ref_trace.set_data(wrist_ref[: frame + 1, 0], wrist_ref[: frame + 1, 1])
        var_trace.set_data(wrist_var[: frame + 1, 0], wrist_var[: frame + 1, 1])
        time_text.set_text(f"t = {t[frame]:.2f} s")

        return ref_line, var_line, ref_trace, var_trace, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=60,
        blit=True,
    )

    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)

    return path


def save_summary(result: dict, out_dir: Path) -> Path:
    """Save text summary of a result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "best_summary.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Gesture: {result['gesture_type']}\n")
        f.write(f"Target: {result['target_name']}\n")
        f.write(f"Reward: {result['reward']:.6f}\n\n")

        f.write("Action:\n")
        f.write(str(np.asarray(result["action"])) + "\n\n")

        f.write("Style parameters:\n")
        for key, value in result.get("style_params", {}).items():
            f.write(f"  {key:28s}: {value}\n")

        f.write("\nTarget profile:\n")
        for key, value in result["target_profile"].items():
            f.write(f"  {key:22s}: {value:.4f}\n")

        f.write("\nReference normalised features:\n")
        for key in FEATURE_KEYS:
            f.write(f"  {key:22s}: {result['ref_norm'].get(key, np.nan)}\n")

        f.write("\nVariant normalised features:\n")
        for key in FEATURE_KEYS:
            f.write(f"  {key:22s}: {result['var_norm'].get(key, np.nan)}\n")

        f.write("\nReward diagnostics:\n")
        for key, value in result.get("reward_info", {}).items():
            f.write(f"  {key:28s}: {value}\n")

    return path


def save_variant_npz(result: dict, out_dir: Path) -> Path:
    """Save reference and variant trajectories as an npz file."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "best_variant.npz"

    np.savez(
        path,
        q_ref=result["q_ref"],
        q_var=result["q_var"],
        action=np.asarray(result["action"]),
        gesture_type=result["gesture_type"],
        target_name=result["target_name"],
    )

    return path


def save_outputs(result: dict, out_dir: str | Path, arm: laban.ArmConfig) -> list[Path]:
    """Save all standard outputs for a result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []

    paths.append(save_variant_npz(result, out_dir))
    paths.append(save_summary(result, out_dir))
    paths.append(save_feature_comparison_plot(result, out_dir))
    paths.append(save_joint_plot(result, out_dir, arm))
    paths.append(save_wrist_path_plot(result, out_dir, arm))
    paths.append(save_arm_gif(result["q_ref"], result["q_var"], out_dir / "arm_comparison.gif", arm))

    return paths
