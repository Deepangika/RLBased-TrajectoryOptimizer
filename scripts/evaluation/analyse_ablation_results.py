"""Descriptive factorial analysis of the 2x2 ablation. Zero API calls.

Reads only saved run outputs. Computes per-condition metrics for each context
and the standard 2x2 contrasts:
- target-recalibration main effect: ((B - A) + (D - C)) / 2
- tightened-smoothness main effect: ((C - A) + (D - B)) / 2
- interaction: (D - C) - (B - A)

Rewards from recalibrated conditions are computed against a different target
than Conditions A/C, so cross-condition reward levels are not directly
comparable; contrasts are reported on within-run improvements (learned minus
fixed baseline under the same run's own target), paired-preference fractions,
feasibility proportions, and motion smoothness, which are comparable.
All outputs are descriptive; no perceptual claims are made.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "configs" / "ablations" / "ablation_manifest_v1.json"


def jerk_metrics(q: np.ndarray, duration: float = 2.0) -> dict[str, float]:
    dt = duration / max(q.shape[0] - 1, 1)
    jerk = np.gradient(
        np.gradient(np.gradient(q, dt, axis=0), dt, axis=0), dt, axis=0
    )
    norms = np.linalg.norm(jerk, axis=1)
    return {
        "rms_jerk": float(np.sqrt(np.mean(norms**2))),
        "peak_jerk": float(np.max(norms)),
    }


def paired_fractions(record: dict | None) -> dict[str, float | None]:
    if not record or record.get("status") not in ("complete",):
        return {"learned_fraction": None, "opponent_fraction": None, "neither_fraction": None}
    counts = record["records"][0]["choice_counts"]
    styled = int(counts.get("styled", 0))
    reference = int(counts.get("reference", 0))
    neither = int(counts.get("neither", 0))
    total = styled + reference + neither
    if total == 0:
        return {"learned_fraction": None, "opponent_fraction": None, "neither_fraction": None}
    return {
        "learned_fraction": styled / total,
        "opponent_fraction": reference / total,
        "neither_fraction": neither / total,
    }


def context_metrics(run_dir: Path, gesture: str, state: str) -> dict | None:
    context_dir = run_dir / "live_stage_a" / gesture / state
    comparison_path = context_dir / "profile_comparison.json"
    if not comparison_path.exists():
        return None
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    stopping = json.loads(
        (context_dir / "stopping_reason.json").read_text(encoding="utf-8")
    )

    feasible = attempted = 0
    sample_history = context_dir / "sample_history.csv"
    if sample_history.exists():
        with sample_history.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                attempted += 1
                if str(row.get("feasible", "")).lower() == "true":
                    feasible += 1

    motion: dict[str, float] = {}
    selection_dir = context_dir / "final_selection"
    candidates = sorted(context_dir.rglob("best_variant.npz"))
    selected_npz = None
    if (context_dir / "selected_validated_profile.json").exists():
        selected = json.loads(
            (context_dir / "selected_validated_profile.json").read_text(
                encoding="utf-8"
            )
        )
        # locate the validation candidate whose profile matches the selection
        for validation_npz in sorted(
            (context_dir / "validation").rglob("best_variant.npz")
        ):
            profile_file = validation_npz.parent / "resolved_target_profile.json"
            if profile_file.exists():
                profile = json.loads(profile_file.read_text(encoding="utf-8"))
                if profile.get("laban_profile") == selected.get("profile"):
                    selected_npz = validation_npz
                    break
    if selected_npz is not None:
        data = np.load(selected_npz)
        motion = jerk_metrics(np.asarray(data["q_var"], dtype=float))
        motion["reference_rms_jerk"] = jerk_metrics(
            np.asarray(data["q_ref"], dtype=float)
        )["rms_jerk"]

    calls = 0
    for budget in run_dir.rglob("gemini_call_budget.json"):
        calls += int(json.loads(budget.read_text(encoding="utf-8")).get("total_calls", 0))

    paired = comparison.get("paired_preference_results", {})
    corrected = comparison.get("corrected_assessment") or stopping.get(
        "corrected_assessment", {}
    )
    return {
        "gesture": gesture,
        "target_state": state,
        "selection_status": comparison.get("selection_status"),
        "effective_target_vad": comparison.get(
            "effective_target_vad", comparison.get("canonical_target_vad")
        ),
        "canonical_target_vad": comparison.get("canonical_target_vad"),
        "target_vad_recalibrated": comparison.get("target_vad_recalibrated", False),
        "reward_improvement_vs_fixed_baseline": comparison.get(
            "vad_improvement_over_fixed_baseline"
        ),
        "paired_vs_baseline": paired_fractions(
            paired.get("learned_vs_baseline")
        ),
        "paired_vs_reference": paired_fractions(
            paired.get("learned_vs_reference")
        ),
        "corrected_outcome": corrected.get("perceptual_outcome"),
        "candidates_attempted": attempted,
        "candidates_feasible": feasible,
        "feasible_proportion": (feasible / attempted) if attempted else None,
        "rounds_completed": stopping.get("rounds_completed"),
        "stopping_reason": stopping.get("stopping_reason"),
        "selected_motion": motion or None,
        "gemini_calls_used": calls,
    }


def _metric(value: dict | None, path: list[str]) -> float | None:
    node: object = value
    for key in path:
        if not isinstance(node, dict) or node.get(key) is None:
            return None
        node = node[key]
    return float(node)  # type: ignore[arg-type]


def contrasts(by_condition: dict[str, dict | None], path: list[str]) -> dict:
    values = {c: _metric(by_condition.get(c), path) for c in "ABCD"}
    if any(values[c] is None for c in "ABCD"):
        return {"values": values, "status": "incomplete"}
    a, b, c, d = (values[k] for k in "ABCD")
    return {
        "values": values,
        "target_recalibration_effect": ((b - a) + (d - c)) / 2.0,
        "tightened_smoothness_effect": ((c - a) + (d - b)) / 2.0,
        "interaction": (d - c) - (b - a),
        "status": "complete",
    }


CONTRAST_METRICS = {
    "reward_improvement_vs_fixed_baseline": ["reward_improvement_vs_fixed_baseline"],
    "paired_vs_baseline_learned_fraction": ["paired_vs_baseline", "learned_fraction"],
    "paired_vs_reference_learned_fraction": ["paired_vs_reference", "learned_fraction"],
    "feasible_proportion": ["feasible_proportion"],
    "selected_rms_jerk": ["selected_motion", "rms_jerk"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/experiments/ablation_analysis_v1",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Out-dir suffix, e.g. _mock for the mock validation pass.",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    run_dirs: dict[tuple[str, str, str], Path] = {}

    def parse_condition(condition_id: str) -> tuple[str, str, str]:
        parts = condition_id.split("_")
        return parts[0], parts[1], parts[2]

    for entry in manifest["condition_a_reference_runs"]:
        gesture, state, letter = parse_condition(entry["condition_id"])
        run_dirs[(gesture, state, letter)] = ROOT / entry["out_dir"]
    for entry in manifest["live_conditions"]:
        gesture, state, letter = parse_condition(entry["condition_id"])
        run_dirs[(gesture, state, letter)] = ROOT / (entry["out_dir"] + args.suffix)

    report: dict = {
        "format_version": 1,
        "descriptive_only": True,
        "note": (
            "Recalibrated conditions use a different reward target; only "
            "within-run improvements and preference/feasibility/smoothness "
            "metrics are compared across conditions."
        ),
        "contexts": {},
    }
    for gesture, state in (("beckon", "fear"), ("wave", "sadness")):
        by_condition: dict[str, dict | None] = {}
        for letter in "ABCD":
            run_dir = run_dirs.get((gesture, state, letter))
            by_condition[letter] = (
                context_metrics(run_dir, gesture, state)
                if run_dir is not None and run_dir.exists()
                else None
            )
        report["contexts"][f"{gesture}_{state}"] = {
            "conditions": by_condition,
            "contrasts": {
                name: contrasts(by_condition, path)
                for name, path in CONTRAST_METRICS.items()
            },
        }

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"ablation_analysis{args.suffix}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
