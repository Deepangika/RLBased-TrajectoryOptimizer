"""Freeze recalibrated VAD targets for the 2x2 ablation from archived evidence.

Procedure (prespecified before any ablation run):
- Evidence source: the archived fixed-profile baseline paired evaluation
  (``paired_results.json``), which pre-dates all outer learning and therefore
  cannot leak information from any selected learned candidate.
- For each gesture, collect every per-repeat Gemini VAD observation across all
  72 baseline records (reference and styled clips, all six emotions).
- Compute the per-axis 10th and 90th percentiles of those observations. This
  is the conservative reachable envelope for that gesture under the frozen
  evaluator protocol.
- The recalibrated target is the canonical named target clipped, per axis, to
  that envelope. No axis is moved further than necessary and axes already
  inside the envelope are unchanged.

The output is evaluator-derived and provisional: it reflects what the frozen
Gemini protocol reports as reachable, not human perceptual ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from laban_rl.affect import VAD_KEYS, target_vad

PROCEDURE_VERSION = "recalibrated-targets-p10-p90-clip-v1"
LOWER_PERCENTILE = 10.0
UPPER_PERCENTILE = 90.0


def gesture_observations(payload: dict, gesture: str) -> np.ndarray:
    rows: list[list[float]] = []
    for record in payload["records"]:
        if record["context"]["gesture"] != gesture:
            continue
        for observation in record.get("observations") or []:
            ratings = observation.get("affect_ratings") or {}
            if all(key in ratings for key in VAD_KEYS):
                rows.append([float(ratings[key]) for key in VAD_KEYS])
    if not rows:
        raise ValueError(f"No archived VAD observations found for {gesture!r}.")
    return np.asarray(rows, dtype=float)


def freeze_context(
    payload: dict,
    *,
    gesture: str,
    target_state: str,
) -> dict:
    observations = gesture_observations(payload, gesture)
    lower = np.percentile(observations, LOWER_PERCENTILE, axis=0)
    upper = np.percentile(observations, UPPER_PERCENTILE, axis=0)
    canonical = target_vad(target_state)
    canonical_array = np.asarray(
        [canonical[key] for key in VAD_KEYS], dtype=float
    )
    recalibrated = np.clip(canonical_array, lower, upper)
    return {
        "gesture": gesture,
        "target_state": target_state,
        "canonical_target_vad": {
            key: float(canonical[key]) for key in VAD_KEYS
        },
        "recalibrated_target_vad": {
            key: round(float(recalibrated[index]), 3)
            for index, key in enumerate(VAD_KEYS)
        },
        "reachable_envelope": {
            key: {
                "p10": float(lower[index]),
                "p90": float(upper[index]),
            }
            for index, key in enumerate(VAD_KEYS)
        },
        "observation_count": int(observations.shape[0]),
        "axes_moved": [
            key
            for index, key in enumerate(VAD_KEYS)
            if not np.isclose(canonical_array[index], recalibrated[index])
        ],
    }


def build_artifact(payload: dict, contexts: list[tuple[str, str]]) -> dict:
    run_identity = payload.get("run_identity", {})
    return {
        "format_version": 1,
        "procedure_version": PROCEDURE_VERSION,
        "status": "frozen",
        "evidence": {
            "source_artifact": (
                "outputs/experiments/baseline_fixed_profiles/full_gemini_live/"
                "paired_results.json"
            ),
            "source_run_identity": run_identity,
            "record_count": len(payload.get("records", [])),
            "predates_outer_learning": True,
            "excludes_learned_candidates": True,
        },
        "procedure": {
            "description": (
                "Per gesture, pool every per-repeat Gemini VAD observation "
                "from the archived baseline paired evaluation, take the "
                "per-axis p10-p90 envelope, and clip the canonical named "
                "target to that envelope."
            ),
            "lower_percentile": LOWER_PERCENTILE,
            "upper_percentile": UPPER_PERCENTILE,
            "aggregation": "per-repeat observations pooled per gesture",
        },
        "label": (
            "evaluator-derived provisional targets; reflects the frozen "
            "Gemini protocol, not human perceptual ground truth"
        ),
        "contexts": [
            freeze_context(payload, gesture=gesture, target_state=state)
            for gesture, state in contexts
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default=(
            "outputs/experiments/baseline_fixed_profiles/full_gemini_live/"
            "paired_results.json"
        ),
    )
    parser.add_argument(
        "--out",
        default="configs/ablations/recalibrated_targets_v1.json",
    )
    args = parser.parse_args()
    payload = json.loads(
        (ROOT / args.results).read_text(encoding="utf-8")
        if not Path(args.results).is_absolute()
        else Path(args.results).read_text(encoding="utf-8")
    )
    artifact = build_artifact(
        payload,
        contexts=[("beckon", "fear"), ("wave", "sadness")],
    )
    destination = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
