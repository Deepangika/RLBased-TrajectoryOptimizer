
"""Render a variant-only MP4 with the same visual style as arm_comparison.gif.

Drop into:
    src/laban_rl/perceptual_bandit/variant_video.py

This renderer deliberately mirrors the project's existing `save_arm_gif`:
- same 6x6 Matplotlib figure,
- same x/y axes and grid,
- same joint markers,
- same wrist trace,
- same time annotation,
- same orange colour used for the Variant in the comparison GIF,
- same camera limits as the comparison animation.

The only visual element removed is the blue Reference arm and its trace.
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

import robust_laban_normalisation_balanced_3gestures as laban

from laban_rl.visualisation import compute_equal_axes


def _comparison_style_frame_indices(n_points: int) -> np.ndarray:
    """Match the frame subsampling used by save_arm_gif."""
    step = 4
    frame_indices = list(range(0, n_points, step))

    if frame_indices[-1] != n_points - 1:
        frame_indices.append(n_points - 1)

    return np.asarray(frame_indices, dtype=int)


def render_variant_only_mp4(
    q_ref: np.ndarray,
    q_var: np.ndarray,
    output_path: str | Path,
    *,
    duration_seconds: float = 2.0,
    fps: float | None = None,
    l1: float = 0.30,
    l2: float = 0.25,
    overwrite: bool = False,
) -> Path:
    """Render only the Variant while preserving the old animation's look.

    `q_ref` is used only to reproduce the exact same camera limits as the
    original reference-vs-variant animation. It is never drawn.
    """
    q_ref = np.asarray(q_ref, dtype=float)
    q_var = np.asarray(q_var, dtype=float)

    if q_ref.ndim != 2 or q_ref.shape[1] != 2:
        raise ValueError(
            f"q_ref must have shape (T, 2), got {q_ref.shape}."
        )
    if q_var.ndim != 2 or q_var.shape[1] != 2:
        raise ValueError(
            f"q_var must have shape (T, 2), got {q_var.shape}."
        )
    if len(q_ref) != len(q_var):
        raise ValueError(
            "q_ref and q_var must contain the same number of points."
        )
    if not np.all(np.isfinite(q_ref)) or not np.all(np.isfinite(q_var)):
        raise ValueError("q_ref/q_var contain NaN or infinite values.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        return output_path

    frame_indices = _comparison_style_frame_indices(len(q_var))

    q_ref_sub = q_ref[frame_indices]
    q_var_sub = q_var[frame_indices]

    shoulder_ref, elbow_ref, wrist_ref = laban.forward_kinematics_2link(
        q_ref_sub,
        l1=l1,
        l2=l2,
    )
    shoulder_var, elbow_var, wrist_var = laban.forward_kinematics_2link(
        q_var_sub,
        l1=l1,
        l2=l2,
    )

    # EXACTLY match the comparison GIF's camera framing.
    xlim, ylim = compute_equal_axes(
        [
            shoulder_ref,
            elbow_ref,
            wrist_ref,
            shoulder_var,
            elbow_var,
            wrist_var,
        ],
        padding=0.08,
    )

    t_full = np.linspace(0.0, duration_seconds, len(q_var))
    t = t_full[frame_indices]

    if fps is None:
        # Keep the motion duration faithful to the trajectory duration.
        fps = len(frame_indices) / duration_seconds

    fps = float(fps)

    fig, ax = plt.subplots(figsize=(6, 6))
    canvas = FigureCanvasAgg(fig)

    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_title("Styled 2D arm")
    ax.grid(True)

    # Match the Variant colour from the original comparison GIF.
    variant_colour = "C1"

    var_line, = ax.plot(
        [],
        [],
        marker="o",
        color=variant_colour,
        label="Variant",
    )
    var_trace, = ax.plot(
        [],
        [],
        alpha=0.5,
        color=variant_colour,
    )
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        va="top",
    )

    ax.legend()

    frames: list[np.ndarray] = []

    for frame in range(len(frame_indices)):
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

        var_line.set_data(var_xs, var_ys)
        var_trace.set_data(
            wrist_var[: frame + 1, 0],
            wrist_var[: frame + 1, 1],
        )
        time_text.set_text(f"t = {t[frame]:.2f} s")

        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(np.array(rgba[:, :, :3], copy=True))

    plt.close(fig)

    imageio.mimwrite(
        output_path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    print(
        "Variant-only MP4 rendered in original animation style: "
        f"duration={duration_seconds:.3f}s | "
        f"fps={fps:.3f} | frames={len(frames)}"
    )

    return output_path
