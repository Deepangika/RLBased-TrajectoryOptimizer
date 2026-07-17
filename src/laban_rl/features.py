"""
Feature helpers for the Laban RL trajectory styler.

This file handles:
    - computing raw and normalised Laban features
    - converting feature dictionaries to arrays
    - converting target dictionaries to arrays
    - building reward masks for unreliable/missing features

It does not compute rewards. Reward logic belongs in rewards.py.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import robust_laban_normalisation_balanced_3gestures as laban

from .config import FEATURE_KEYS


def compute_raw_and_norm_features(
    q: np.ndarray,
    arm: laban.ArmConfig,
    filter_config: laban.FilterConfig,
    ranges: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute raw and normalised Laban features for a trajectory."""
    raw = laban.compute_laban_features(
        q=q,
        arm=arm,
        filter_config=filter_config,
    )

    norm = laban.normalise_laban_features(
        features=raw,
        normalisation_ranges=ranges,
        clip=True,
    )

    return raw, norm


def feature_dict_to_array(features: Dict[str, float]) -> np.ndarray:
    """
    Convert a feature dictionary to an ordered feature array.

    Any missing or NaN feature is replaced with 0.0.
    Masking is handled separately by get_feature_mask().
    """
    arr = []

    for key in FEATURE_KEYS:
        value = features.get(key, np.nan)

        if value is None or np.isnan(value):
            value = 0.0

        arr.append(float(value))

    return np.array(arr, dtype=np.float32)


def target_dict_to_array(target: Dict[str, float]) -> np.ndarray:
    """Convert a target Laban profile dictionary to an ordered array."""
    return np.array(
        [float(target[key]) for key in FEATURE_KEYS],
        dtype=np.float32,
    )


def get_feature_mask(
    gesture_type: str,
    features_norm: Dict[str, float],
) -> np.ndarray:
    """
    Create a mask for reward calculation.

    A value of 1 means the feature is included in the style reward.
    A value of 0 means the feature is ignored.

    Space is ignored for wave because wave is cyclic and can make
    space_indirectness unstable or undefined.
    """
    mask = np.ones(len(FEATURE_KEYS), dtype=np.float32)

    if gesture_type == "wave":
        mask[FEATURE_KEYS.index("space_indirectness")] = 0.0

    for idx, key in enumerate(FEATURE_KEYS):
        value = features_norm.get(key, np.nan)

        if value is None or np.isnan(value):
            mask[idx] = 0.0

    return mask
