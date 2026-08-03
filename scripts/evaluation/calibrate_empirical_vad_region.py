"""Estimate a conservative empirical VAD envelope from live clip ratings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.affect import VAD_KEYS, VAD_TARGETS


VAD_WEIGHTS = np.asarray([0.2, 0.4, 0.4], dtype=float)


def project_to_empirical_hull(
    target: Mapping[str, float],
    observed_points: Sequence[Mapping[str, float]],
) -> tuple[dict[str, float], float]:
    points = np.asarray(
        [[row[key] for key in VAD_KEYS] for row in observed_points],
        dtype=float,
    )
    target_array = np.asarray([target[key] for key in VAD_KEYS], dtype=float)
    if len(points) < 1 or not np.all(np.isfinite(points)):
        raise ValueError("At least one finite observed VAD point is required.")

    def objective(coefficients: np.ndarray) -> float:
        projected = coefficients @ points
        return float(np.sum(VAD_WEIGHTS * (projected - target_array) ** 2))

    initial = np.full(len(points), 1.0 / len(points), dtype=float)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(points),
        constraints={
            "type": "eq",
            "fun": lambda coefficients: float(np.sum(coefficients) - 1.0),
        },
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"VAD hull projection failed: {result.message}")
    projected = result.x @ points
    return (
        {
            key: float(projected[index])
            for index, key in enumerate(VAD_KEYS)
        },
        float(np.sqrt(objective(result.x))),
    )


def build_report(payload: Mapping) -> dict:
    unique_records = {}
    for record in payload.get("records", []):
        reliability = record.get("reliability")
        cache_key = record.get("observation_cache_key")
        if reliability is None or not cache_key:
            continue
        unique_records.setdefault(cache_key, record)
    if not unique_records:
        raise ValueError("No independently rated clip records were found.")

    clips = []
    for cache_key, record in unique_records.items():
        mean_vad = {
            key: float(record["reliability"]["mean_observed_vad"][key])
            for key in VAD_KEYS
        }
        clips.append(
            {
                "observation_cache_key": cache_key,
                "candidate_id": record["candidate_id"],
                "gesture": record["context"]["gesture"],
                "candidate_name": record["candidate_name"],
                "achieved_laban_profile": dict(record["achieved_profile"]),
                "mean_observed_vad": mean_vad,
                "per_axis_vad_std": dict(
                    record["reliability"]["per_axis_vad_std"]
                ),
            }
        )
    observed = [clip["mean_observed_vad"] for clip in clips]
    envelope = {
        key: {
            "minimum_clip_mean": min(row[key] for row in observed),
            "maximum_clip_mean": max(row[key] for row in observed),
        }
        for key in VAD_KEYS
    }
    targets = {}
    for state, target in VAD_TARGETS.items():
        projected, hull_distance = project_to_empirical_hull(target, observed)
        nearest = min(
            clips,
            key=lambda clip: float(
                np.sqrt(
                    np.sum(
                        VAD_WEIGHTS
                        * (
                            np.asarray(
                                [
                                    clip["mean_observed_vad"][key]
                                    for key in VAD_KEYS
                                ]
                            )
                            - np.asarray([target[key] for key in VAD_KEYS])
                        )
                        ** 2
                    )
                )
            ),
        )
        targets[state] = {
            "canonical_target_vad": dict(target),
            "empirical_hull_projection": projected,
            "weighted_distance_to_empirical_hull": hull_distance,
            "nearest_observed_clip": {
                "candidate_id": nearest["candidate_id"],
                "mean_observed_vad": nearest["mean_observed_vad"],
            },
            "outside_clip_mean_axis_range": [
                key
                for key in VAD_KEYS
                if target[key] < envelope[key]["minimum_clip_mean"]
                or target[key] > envelope[key]["maximum_clip_mean"]
            ],
        }
    return {
        "format_version": 1,
        "source_experiment_format_version": payload["run_identity"][
            "configuration"
        ]["format_version"],
        "unique_clip_count": len(clips),
        "vad_distance_weights": {
            key: float(VAD_WEIGHTS[index])
            for index, key in enumerate(VAD_KEYS)
        },
        "observed_clip_mean_envelope": envelope,
        "named_target_diagnostics": targets,
        "clips": clips,
        "limitations": [
            "The convex hull is based on sparse Gemini clip means, not human labels.",
            "Convex combinations of observed clips are an envelope, not proof that an intermediate motion is realizable.",
            "Gesture identity and renderer duration may shift perceived VAD.",
            "Do not replace canonical targets without broader repeated and human evaluation.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    report = build_report(payload)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
