"""Input/output helpers."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import robust_laban_normalisation_balanced_3gestures as laban

def load_ranges_or_default(path: str | Path, gesture: str | None = None) -> Dict[str, Tuple[float, float]]:
    """Load normalisation ranges from JSON, or fallback ranges if missing.

    If the JSON contains gesture-specific ranges, the current gesture is used
    to select the appropriate sub-dictionary.
    """
    path = Path(path)
    if path.exists():
        ranges = laban.load_normalisation_ranges(path)
    else:
        candidate = Path("configs") / Path(path).name
        if candidate.exists():
            ranges = laban.load_normalisation_ranges(candidate)
        else:
            print(f"Warning: {path} not found. Using fallback ranges.")
            ranges = {
                "weight": (0.015250, 1.769108),
                "time": (0.574731, 13.351330),
                "flow_boundness": (1.950910, 104.285395),
                "space_indirectness": (1.026376, 50.978867),
                "shape_arcness": (11.573163, 105.162206),
            }
    if gesture is None:
        if not ranges:
            return ranges

        sample = next(iter(ranges.values()))
        if isinstance(sample, tuple):
            return ranges

        if "balanced" in ranges:
            return ranges["balanced"]

        raise ValueError(
            "Loaded gesture-specific normalisation ranges, but no gesture was supplied. "
            "Pass gesture to load_ranges_or_default(..., gesture=gesture)."
        )

    sample = next(iter(ranges.values()))
    if isinstance(sample, tuple):
        return ranges

    if gesture not in ranges:
        if "balanced" in ranges:
            print(
                f"Warning: no normalisation ranges calibrated for {gesture!r}; "
                "using balanced ranges."
            )
            return ranges["balanced"]
        raise ValueError(
            f"Gesture-specific normalisation ranges do not contain values for gesture {gesture!r}. "
            f"Available gestures: {sorted(ranges)}"
        )

    return ranges[gesture]
