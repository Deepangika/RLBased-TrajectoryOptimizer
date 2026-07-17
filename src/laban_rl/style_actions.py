"""
Style action implementation.

This file defines the bridge:

    action [amplitude, timing, curve, pause, smoothing]
        -> q_variant

The Laban values are not directly changed here. They are measured later
from the generated q_variant.

Design update:
    Earlier versions preserved both the start and end joint pose exactly.
    That is useful for strict task preservation, but too restrictive for
    expressive variants.

    Now:
        - the start pose is preserved
        - the endpoint is allowed to vary by default
        - large endpoint drift is discouraged softly in the reward
        - amplitude envelope is chosen by the policy through envelope_blend
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def resample_trajectory(q: np.ndarray, new_progress: np.ndarray) -> np.ndarray:
    """Resample q at progress values in [0, 1]."""
    q = np.asarray(q, dtype=np.float64)
    new_progress = np.clip(np.asarray(new_progress, dtype=np.float64), 0.0, 1.0)

    old_progress = np.linspace(0.0, 1.0, len(q))
    q_resampled = np.zeros_like(q)

    for joint_idx in range(q.shape[1]):
        q_resampled[:, joint_idx] = np.interp(
            new_progress,
            old_progress,
            q[:, joint_idx],
        )

    return q_resampled


def endpoint_preserving_time_warp(u: np.ndarray, speed_action: float) -> np.ndarray:
    """
    Nonlinear progress curve that preserves start and end in time.

    speed_action:
        -1 -> slower/sustained
         0 -> unchanged
        +1 -> faster/earlier

    The scale is 0.45 rather than 0.9 because earlier PPO experiments showed
    that timing was too powerful and pushed Weight/Time to 1.0 too easily.
    """
    speed_action = float(np.clip(speed_action, -1.0, 1.0))

    gamma = np.exp(-0.45 * speed_action)

    progress = u ** gamma
    progress[0] = 0.0
    progress[-1] = 1.0

    return np.clip(progress, 0.0, 1.0)


def moving_average_smooth(q: np.ndarray, strength: float) -> np.ndarray:
    """Simple moving-average smoothing while preserving endpoints of the filtered signal."""
    q = np.asarray(q, dtype=np.float64)
    strength = float(np.clip(strength, 0.0, 1.0))

    if strength <= 1e-6:
        return q.copy()

    max_window = 13
    window = int(3 + strength * (max_window - 3))

    if window % 2 == 0:
        window += 1

    pad = window // 2
    kernel = np.ones(window) / window
    q_smooth = np.zeros_like(q)

    for joint_idx in range(q.shape[1]):
        padded = np.pad(q[:, joint_idx], pad_width=pad, mode="edge")
        q_smooth[:, joint_idx] = np.convolve(padded, kernel, mode="valid")

    # This preserves the endpoints of whatever trajectory is passed into smoothing.
    q_smooth[0] = q[0]
    q_smooth[-1] = q[-1]

    return q_smooth


def apply_pause_hold_endpoint_preserving(q: np.ndarray, pause_fraction: float) -> np.ndarray:
    """
    Compress active motion earlier and hold the final pose.

    This preserves the endpoint of the input q, but that endpoint may already
    be an expressive endpoint that differs from q_ref[-1].
    """
    q = np.asarray(q, dtype=np.float64)
    pause_fraction = float(np.clip(pause_fraction, 0.0, 0.45))

    if pause_fraction <= 1e-6:
        return q.copy()

    T = len(q)
    u = np.linspace(0.0, 1.0, T)

    active_fraction = 1.0 - pause_fraction
    progress = np.clip(u / active_fraction, 0.0, 1.0)

    q_paused = resample_trajectory(q, progress)

    hold_start_idx = int(np.ceil(active_fraction * (T - 1)))
    q_paused[hold_start_idx:] = q[-1]

    return q_paused


def smoothstep(u: np.ndarray) -> np.ndarray:
    """
    Smooth ramp from 0 to 1.

    This is useful for confident/decisive styles because the expressive
    amplitude grows through the motion and remains at the end.
    """
    return 3.0 * u**2 - 2.0 * u**3


def early_rise(u: np.ndarray, sharpness: float = 5.0) -> np.ndarray:
    """
    Quick rise then hold.

    Useful for sudden/strong styles such as anger or very decisive actions.
    """
    env = 1.0 - np.exp(-sharpness * u)
    env = env / max(env[-1], 1e-8)
    return env


def middle_bump(u: np.ndarray) -> np.ndarray:
    """
    0 -> 1 -> 0 envelope.

    Useful for hesitant/confused motions because it creates mid-motion swelling
    without changing the final pose much.
    """
    return np.sin(np.pi * u)


def blend_envelopes(
    u: np.ndarray,
    envelope_blend: float,
) -> Tuple[np.ndarray, str]:
    """
    Blend between middle-bump and sustained ramp envelopes.

    envelope_blend:
        -1 -> mostly middle bump
         0 -> equal blend
        +1 -> mostly sustained ramp
    """
    envelope_blend = float(np.clip(envelope_blend, -1.0, 1.0))

    bump = middle_bump(u)
    sustained = smoothstep(u)

    blend_factor = (envelope_blend + 1.0) / 2.0
    envelope = (1.0 - blend_factor) * bump + blend_factor * sustained
    envelope_name = (
        "middle_bump"
        if envelope_blend < -0.33
        else "sustained_smoothstep"
        if envelope_blend > 0.33
        else "mixed_bump_ramp"
    )

    return envelope, envelope_name


def apply_style_action(
    q_ref: np.ndarray,
    action: np.ndarray,
    gesture_type: str = "reach",
    target_name: str | None = None,
    preserve_endpoints: bool = False,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convert a 6D action into a styled trajectory.

    action = [
        amplitude emphasis,
        timing / speed profile,
        curve / arc emphasis,
        pause / hold,
        smoothing,
        envelope blend
    ]

    Each action component is expected in [-1, 1].

    preserve_endpoints:
        False:
            preserve only the start pose.
            allow the final pose to vary expressively.

        True:
            preserve both start and end exactly.
            useful for strict task-preserving tests.
    """
    q_ref = np.asarray(q_ref, dtype=np.float64)
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

    if action.shape[0] != 6:
        raise ValueError("Expected action shape (6,).")

    T = len(q_ref)
    u = np.linspace(0.0, 1.0, T)

    amp_env, amp_env_name = blend_envelopes(u, envelope_blend=action[5])

    # Keep curve as a mid-motion bump for now.
    # This bends the pathway without necessarily dragging the endpoint too much.
    curve_env = middle_bump(u)

    amp_delta = 0.45 * action[0]

    # Reduced from 0.10 to 0.07 to avoid overpowered curve action.
    curve_strength = 0.07 * action[2]

    pause_fraction = 0.35 * ((action[3] + 1.0) / 2.0)
    smooth_strength = 0.65 * ((action[4] + 1.0) / 2.0)

    # Wave is cyclic; final hold is often semantically odd.
    if gesture_type == "wave":
        pause_fraction *= 0.25

    # 1. Time warp.
    progress = endpoint_preserving_time_warp(u, speed_action=action[1])
    q = resample_trajectory(q_ref, progress)

    # 2. Amplitude emphasis.
    # For confident, amp_env is sustained, so endpoint can change.
    # For hesitant/calm/friendly, amp_env is middle_bump, so the effect is mostly mid-motion.
    q_start = q_ref[0]
    movement_from_start = q - q_start
    q = q + amp_delta * amp_env[:, None] * movement_from_start

    # 3. Mid-trajectory curve / arc.
    curve_scale_for_gesture = 0.5 if gesture_type == "wave" else 1.0
    q[:, 0] += curve_scale_for_gesture * curve_strength * curve_env
    q[:, 1] -= curve_scale_for_gesture * 0.75 * curve_strength * curve_env

    # 4. Pause / hold.
    q = apply_pause_hold_endpoint_preserving(q, pause_fraction=pause_fraction)

    # 5. Smoothing.
    q = moving_average_smooth(q, strength=smooth_strength)

    # Preserve start exactly.
    q[0] = q_ref[0]

    # Optionally preserve the final endpoint exactly.
    # Default is False for expressive variants.
    if preserve_endpoints:
        q[-1] = q_ref[-1]

    style_params = {
        "amp_delta": float(amp_delta),
        "speed_action": float(action[1]),
        "curve_strength": float(curve_strength),
        "pause_fraction": float(pause_fraction),
        "smooth_strength": float(smooth_strength),
        "envelope_blend": float(action[5]),
        "amplitude_envelope": amp_env_name,
        "preserve_endpoints": bool(preserve_endpoints),
    }

    return q, style_params