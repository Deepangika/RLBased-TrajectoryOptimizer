"""Offline smoothness grid for the ablation's tightened-smoothness factor.

Runs the real inner optimiser (zero Gemini calls) over a grid of
``smooth_weight`` values for both ablation contexts, using saved profiles from
the completed live runs (Condition A) plus the baseline initialisation
profiles. Reports RMS/peak jerk, feature error, feasibility, and collapse
diagnostics so ONE smoothness setting can be frozen for Conditions C and D.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import optimise_laban_target
from laban_rl.perceptual_bandit.baseline import (
    baseline_condition,
    baseline_optimizer_overrides,
)

GRID = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)
FEASIBILITY_TOLERANCE = 0.10
SEED = 7

CONTEXTS = (
    ("beckon", "fear", "outputs/experiments/outer_learning_beckon_fear_live_flash_v1"),
    ("wave", "sadness", "outputs/experiments/outer_learning_wave_sadness_live_flash_v1"),
)


def jerk_metrics(q: np.ndarray, duration: float = 2.0) -> dict[str, float]:
    dt = duration / max(q.shape[0] - 1, 1)
    jerk = np.gradient(
        np.gradient(np.gradient(q, dt, axis=0), dt, axis=0), dt, axis=0
    )
    norms = np.linalg.norm(jerk, axis=1)
    return {
        "rms_jerk": float(np.sqrt(np.mean(norms**2))),
        "peak_jerk": float(np.max(norms)),
        "mean_jerk": float(np.mean(norms)),
    }


def profile_sources(gesture: str, state: str, run_dir: str) -> list[tuple[str, dict[str, float]]]:
    sources: list[tuple[str, dict[str, float]]] = []
    baseline = baseline_condition(gesture, state)
    initial = (
        baseline.projected_feasible_initialisation
        or baseline.requested_optimization
    )
    sources.append(("baseline_initialisation", dict(initial)))
    selected_path = (
        ROOT / run_dir / "live_stage_a" / gesture / state
        / "selected_validated_profile.json"
    )
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        sources.append(("live_selected_learned", dict(selected["profile"])))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/experiments/ablation_smoothness_grid_v1",
    )
    parser.add_argument("--grid", type=float, nargs="*", default=list(GRID))
    args = parser.parse_args()
    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for gesture, state, run_dir in CONTEXTS:
        overrides_base = baseline_optimizer_overrides(gesture, state)
        for source_name, profile in profile_sources(gesture, state, run_dir):
            for weight in args.grid:
                overrides = dict(overrides_base)
                overrides["smooth_weight"] = float(weight)
                overrides["seed"] = SEED
                label = f"{gesture}_{state}_{source_name}_sw{weight:g}"
                work_dir = out_root / "runs" / label
                print(f"[grid] {label}", flush=True)
                result = optimise_laban_target(
                    gesture=gesture,
                    target_state=state,
                    target_profile=profile,
                    out_dir=work_dir,
                    optimiser_overrides=overrides,
                )
                requested = np.asarray(
                    [profile[key] for key in FEATURE_KEYS], dtype=float
                )
                achieved = np.asarray(
                    [result.achieved_profile[key] for key in FEATURE_KEYS],
                    dtype=float,
                )
                max_abs_error = float(np.max(np.abs(requested - achieved)))
                variant_metrics = jerk_metrics(result.q_var)
                reference_metrics = jerk_metrics(result.q_ref)
                deviation = float(
                    np.mean(np.linalg.norm(result.q_var - result.q_ref, axis=1))
                )
                rows.append(
                    {
                        "gesture": gesture,
                        "target_state": state,
                        "profile_source": source_name,
                        "smooth_weight": float(weight),
                        "seed": SEED,
                        "max_abs_feature_error": max_abs_error,
                        "realisation_rmse": result.realisation_rmse,
                        "feasible": max_abs_error <= FEASIBILITY_TOLERANCE,
                        "variant_rms_jerk": variant_metrics["rms_jerk"],
                        "variant_peak_jerk": variant_metrics["peak_jerk"],
                        "reference_rms_jerk": reference_metrics["rms_jerk"],
                        "jerk_ratio_vs_reference": (
                            variant_metrics["rms_jerk"]
                            / max(reference_metrics["rms_jerk"], 1e-9)
                        ),
                        "mean_deviation_from_reference_rad": deviation,
                        "collapsed_to_reference": deviation < 1e-3,
                        "inner_reward": result.inner_reward,
                        "inner_loss": result.inner_loss,
                    }
                )
                summary_path = out_root / "smoothness_grid_results.json"
                temporary = summary_path.with_suffix(".json.tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "format_version": 1,
                            "feasibility_tolerance": FEASIBILITY_TOLERANCE,
                            "grid": [float(w) for w in args.grid],
                            "seed": SEED,
                            "rows": rows,
                        },
                        indent=2,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                temporary.replace(summary_path)
    print(f"[grid] complete: {len(rows)} rows -> {out_root}")


if __name__ == "__main__":
    main()
