"""Calibrate robust normalisation ranges for the three planar-arm gestures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import robust_laban_normalisation_balanced_3gestures as laban


NEW_GESTURES = ("circle", "beckon", "celebratory_pump")
DEFAULT_CONFIG_PATH = ROOT / "configs" / "normalisation_ranges_balanced_3gestures_by_gesture.json"


def calibrate_new_gesture_ranges(
    *,
    n_samples: int = 9000,
    seed: int = 20260731,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Return deterministic robust ranges using balanced semantic variants."""
    if n_samples < len(NEW_GESTURES):
        raise ValueError(
            f"n_samples must be at least {len(NEW_GESTURES)} for balanced sampling"
        )
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError(
            "percentiles must satisfy 0 <= low_percentile < high_percentile <= 100"
        )

    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filter_config = laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    sweep = laban.SweepConfig(
        n_samples=n_samples,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
        seed=seed,
    )
    _, ranges = laban.run_robust_parameter_sweep(
        arm=arm,
        sweep=sweep,
        filter_config=filter_config,
        gesture_types=list(NEW_GESTURES),
    )
    return {gesture: ranges[gesture] for gesture in NEW_GESTURES}


def update_config(
    config_path: str | Path,
    calibrated_ranges: dict[str, dict[str, tuple[float, float]]],
) -> None:
    """Update only the new-gesture entries in the existing range config."""
    path = Path(config_path)
    source_path = path if path.exists() else DEFAULT_CONFIG_PATH
    with source_path.open("r", encoding="utf-8") as handle:
        existing = json.load(handle)

    for gesture in NEW_GESTURES:
        existing[gesture] = {
            feature: {"min": low, "max": high}
            for feature, (low, high) in calibrated_ranges[gesture].items()
        }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(existing, handle, indent=2)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--low-percentile", type=float, default=5.0)
    parser.add_argument("--high-percentile", type=float, default=95.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ranges = calibrate_new_gesture_ranges(
        n_samples=args.samples,
        seed=args.seed,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
    )
    update_config(args.output, ranges)
    print(f"Updated {args.output}")
    for gesture, feature_ranges in ranges.items():
        print(gesture)
        for feature, (low, high) in feature_ranges.items():
            print(f"  {feature:22s}: {low:.9f} .. {high:.9f}")


if __name__ == "__main__":
    main()
