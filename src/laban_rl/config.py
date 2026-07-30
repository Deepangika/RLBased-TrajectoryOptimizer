"""
Shared configuration for the Laban RL trajectory styler.

This file should contain constants and lightweight dataclasses only.
It should not contain training code, plotting code, or environment logic.
"""

from __future__ import annotations

from dataclasses import dataclass


FEATURE_KEYS = [
    "weight",
    "time",
    "flow_boundness",
    "space_indirectness",
    "shape_arcness",
]

GESTURE_TYPES = [
    "wave",
    "reach",
    "point",
    "circle",
    "beckon",
    "celebratory_pump",
]

CYCLIC_GESTURES = {"wave", "circle"}

# Ekman's six basic emotions.
EMOTION_STATES = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
]


# Feature-specific reward weights.
#
# Space is weighted lower because the current 2D styler has limited ability
# to increase space_indirectness for reach/point without changing the gesture
# structure too much.
#
# Shape is weighted up slightly because PPO was previously ignoring
# shape_arcness.
FEATURE_REWARD_WEIGHTS = {
    "weight": 1.0,
    "time": 1.0,
    "flow_boundness": 1.0,
    "space_indirectness": 0.2,
    "shape_arcness": 1.5,
}


@dataclass
class RewardWeights:
    """Weights used to combine reward terms."""

    style: float = 1.0
    preserve: float = 0.05
    smoothness: float = 0.02
    joint_limits: float = 0.10
    action: float = 0.03
    saturation: float = 0.05

    # Soft endpoint penalty.
    # We no longer force q_var[-1] == q_ref[-1].
    # Expressive variants may end slightly differently.
    endpoint: float = 0.02

    # Semantic action preference.
    # For example, confident motion should not strongly shrink amplitude.
    target_action_pref: float = 0.04


@dataclass
class JointLimits:
    """Simple joint limits for the 2D shoulder/elbow prototype."""

    shoulder_min: float = -0.50
    shoulder_max: float = 1.40
    elbow_min: float = 0.00
    elbow_max: float = 1.70