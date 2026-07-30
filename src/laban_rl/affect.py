"""Continuous affect targets used by perceptual evaluation."""

from __future__ import annotations

from typing import Mapping, TypedDict

import numpy as np

VAD_KEYS = ("valence", "arousal", "dominance")


class VADVector(TypedDict):
    """Serializable normalized valence-arousal-dominance coordinates."""

    valence: float
    arousal: float
    dominance: float


# Normalized [0, 1] anchors derived from common circumplex/PAD placements.
# They are initialization targets, not empirical annotations for this robot.
VAD_TARGETS: dict[str, VADVector] = {
    "anger": {"valence": 0.15, "arousal": 0.85, "dominance": 0.80},
    "disgust": {"valence": 0.10, "arousal": 0.45, "dominance": 0.60},
    "fear": {"valence": 0.10, "arousal": 0.85, "dominance": 0.20},
    "happiness": {"valence": 0.90, "arousal": 0.75, "dominance": 0.65},
    "sadness": {"valence": 0.10, "arousal": 0.20, "dominance": 0.20},
    "surprise": {"valence": 0.60, "arousal": 0.90, "dominance": 0.50},
}


def validate_vad(values: Mapping[str, float], *, name: str = "VAD") -> VADVector:
    """Validate and normalize a VAD mapping into canonical key order."""
    missing = [key for key in VAD_KEYS if key not in values]
    extra = [key for key in values if key not in VAD_KEYS]
    if missing or extra:
        raise ValueError(f"{name} keys must be {VAD_KEYS}; missing={missing}, extra={extra}.")

    try:
        validated = VADVector(
            valence=float(values["valence"]),
            arousal=float(values["arousal"]),
            dominance=float(values["dominance"]),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} values must be finite numbers.") from error
    vector = np.asarray(list(validated.values()), dtype=float)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} values must be finite.")
    if np.any(vector < 0.0) or np.any(vector > 1.0):
        raise ValueError(f"{name} values must lie in [0, 1].")
    return validated


def target_vad(target_state: str) -> VADVector:
    """Return the configured VAD anchor for an affective state."""
    if target_state not in VAD_TARGETS:
        raise ValueError(
            f"No VAD target configured for {target_state!r}. "
            f"Available targets: {sorted(VAD_TARGETS)}"
        )
    return dict(VAD_TARGETS[target_state])
