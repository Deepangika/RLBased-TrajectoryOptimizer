
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


def compose_standardized_sequence(
    trajectory: np.ndarray,
    *,
    gesture_duration_seconds: float,
    lead_in_seconds: float,
    repetitions: int,
    inter_repeat_transition_seconds: float,
    final_hold_seconds: float,
) -> np.ndarray:
    """Add initial/final holds and repeat a trajectory at its natural speed."""
    values = np.asarray(trajectory, dtype=float)
    if gesture_duration_seconds <= 0.0:
        raise ValueError("gesture_duration_seconds must be positive.")
    if (
        lead_in_seconds < 0.0
        or inter_repeat_transition_seconds < 0.0
        or final_hold_seconds < 0.0
    ):
        raise ValueError("Sequence segment durations cannot be negative.")
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1.")
    samples_per_second = len(values) / gesture_duration_seconds
    lead_count = int(round(lead_in_seconds * samples_per_second))
    hold_count = int(round(final_hold_seconds * samples_per_second))
    segments = []
    if lead_count:
        segments.append(np.repeat(values[:1], lead_count, axis=0))
    transition_count = int(
        round(inter_repeat_transition_seconds * samples_per_second)
    )
    for repetition in range(repetitions):
        if repetition and transition_count:
            progress = np.linspace(
                0.0,
                1.0,
                transition_count + 2,
                dtype=float,
            )[1:-1]
            smooth = progress * progress * (3.0 - 2.0 * progress)
            transition = (
                values[-1][None, :] * (1.0 - smooth[:, None])
                + values[0][None, :] * smooth[:, None]
            )
            segments.append(transition)
        segments.append(values.copy())
    if hold_count:
        segments.append(np.repeat(values[-1:], hold_count, axis=0))
    return np.concatenate(segments, axis=0)


def shared_camera_limits(
    trajectories: list[np.ndarray],
    *,
    duration_seconds: float,
    lead_in_seconds: float,
    repetitions: int,
    inter_repeat_transition_seconds: float,
    final_hold_seconds: float,
    l1: float = 0.30,
    l2: float = 0.25,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute identical camera limits for every clip in a blinded pair."""
    positions = []
    for trajectory in trajectories:
        sequence = compose_standardized_sequence(
            trajectory,
            gesture_duration_seconds=duration_seconds,
            lead_in_seconds=lead_in_seconds,
            repetitions=repetitions,
            inter_repeat_transition_seconds=(
                inter_repeat_transition_seconds
            ),
            final_hold_seconds=final_hold_seconds,
        )
        shoulder, elbow, wrist = laban.forward_kinematics_2link(
            sequence,
            l1=l1,
            l2=l2,
        )
        positions.extend((shoulder, elbow, wrist))
    return compute_equal_axes(positions, padding=0.08)


def render_variant_only_mp4(
    q_ref: np.ndarray,
    q_var: np.ndarray,
    output_path: str | Path,
    *,
    duration_seconds: float = 2.0,
    lead_in_seconds: float = 0.0,
    repetitions: int = 1,
    inter_repeat_transition_seconds: float = 0.0,
    final_hold_seconds: float = 0.0,
    fps: float | None = None,
    l1: float = 0.30,
    l2: float = 0.25,
    camera_limits: tuple[
        tuple[float, float],
        tuple[float, float],
    ] | None = None,
    presentation_style: str = "plot",
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
    if presentation_style not in {"plot", "arm_only"}:
        raise ValueError(
            "presentation_style must be 'plot' or 'arm_only'."
        )

    q_ref = compose_standardized_sequence(
        q_ref,
        gesture_duration_seconds=duration_seconds,
        lead_in_seconds=lead_in_seconds,
        repetitions=repetitions,
        inter_repeat_transition_seconds=inter_repeat_transition_seconds,
        final_hold_seconds=final_hold_seconds,
    )
    q_var = compose_standardized_sequence(
        q_var,
        gesture_duration_seconds=duration_seconds,
        lead_in_seconds=lead_in_seconds,
        repetitions=repetitions,
        inter_repeat_transition_seconds=inter_repeat_transition_seconds,
        final_hold_seconds=final_hold_seconds,
    )
    total_duration_seconds = (
        lead_in_seconds
        + repetitions * duration_seconds
        + (repetitions - 1) * inter_repeat_transition_seconds
        + final_hold_seconds
    )

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
    if camera_limits is None:
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
    else:
        xlim, ylim = camera_limits

    t_full = np.linspace(0.0, total_duration_seconds, len(q_var))
    t = t_full[frame_indices]

    if fps is None:
        # Keep the motion duration faithful to the trajectory duration.
        fps = len(frame_indices) / total_duration_seconds

    fps = float(fps)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    canvas = FigureCanvasAgg(fig)

    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if presentation_style == "plot":
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.set_title("Styled 2D arm")
        ax.grid(True)
        variant_colour = "C1"
    else:
        ax.set_axis_off()
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        variant_colour = "#202020"

    var_line, = ax.plot(
        [],
        [],
        marker="o",
        color=variant_colour,
        label="Variant",
        linewidth=3.0 if presentation_style == "arm_only" else 1.5,
        markersize=8.0 if presentation_style == "arm_only" else 6.0,
    )
    var_trace = None
    time_text = None
    if presentation_style == "plot":
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
        if var_trace is not None:
            var_trace.set_data(
                wrist_var[: frame + 1, 0],
                wrist_var[: frame + 1, 1],
            )
        if time_text is not None:
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
        f"Variant-only MP4 rendered with {presentation_style} presentation: "
        f"duration={total_duration_seconds:.3f}s | "
        f"fps={fps:.3f} | frames={len(frames)}"
    )

    return output_path
