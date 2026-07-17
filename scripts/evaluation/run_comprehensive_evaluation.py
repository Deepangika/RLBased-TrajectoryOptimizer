"""Run and analyse a comprehensive revised-descriptor evaluation.

The suite deliberately separates:
  1. inner-loop realisability (no VLM calls),
  2. outer contextual-bandit search,
  3. frozen-profile holdout evaluation, and
  4. aggregation into paper-ready CSV/JSON tables and figures.

Every experiment has a deterministic directory. Existing completed stages are
skipped, while interrupted CEM runs resume from their checkpoints.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ["weight", "time", "flow_boundness", "space_indirectness", "shape_arcness"]
FEATURE_LABELS = ["Weight", "Time", "Flow", "Space", "Shape"]
GESTURES = ["wave", "reach", "point"]
STATES = ["confident", "calm", "hesitant", "friendly", "confused", "angry"]
EXPECTED_SPACE_RANGES = {
    "wave": (0.06269632, 0.11528429),
    "reach": (0.01198525, 0.05514839),
    "point": (0.02318369, 0.08070654),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["inner", "outer", "holdout", "report", "all"], required=True)
    p.add_argument("--out", default="outputs/comprehensive_revised_space")
    p.add_argument("--gestures", nargs="+", choices=GESTURES, default=GESTURES)
    p.add_argument("--states", nargs="+", choices=STATES, default=STATES)
    p.add_argument("--seeds", nargs="+", type=int, default=[7, 41, 97])
    p.add_argument("--inner-seeds", nargs="+", type=int, default=[7, 17, 27, 37, 47])
    p.add_argument("--evaluator", choices=["gemini", "mock"], default="gemini")
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--samples-per-round", type=int, default=6)
    p.add_argument("--training-repeats", type=int, default=3)
    p.add_argument("--validation-top-k", type=int, default=3)
    p.add_argument("--validation-repeats", type=int, default=10)
    p.add_argument("--holdout-repeats", type=int, default=20)
    p.add_argument("--maxiter", type=int, default=90)
    p.add_argument("--popsize", type=int, default=8)
    p.add_argument("--local-maxiter", type=int, default=300)
    p.add_argument("--inner-matrix-maxiter", type=int, default=45)
    p.add_argument("--inner-matrix-popsize", type=int, default=5)
    p.add_argument("--inner-matrix-local-maxiter", type=int, default=100)
    p.add_argument("--tolerance", type=float, default=0.10)
    p.add_argument("--confirm-live-cost", action="store_true")
    p.add_argument("--overwrite-holdout", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def experiment_dir(base: Path, gesture: str, state: str, seed: int) -> Path:
    return base / "outer" / gesture / state / f"seed_{seed}"


def run(command: list[str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def estimate_calls(args: argparse.Namespace) -> dict[str, int]:
    contexts = len(args.gestures) * len(args.states) * len(args.seeds)
    # The trainer evaluates the informed initial profile with validation repeats,
    # trains every sampled profile, and re-evaluates top-K profiles.
    per_outer = (
        2 * args.validation_repeats  # informed profile and final CEM mean
        + args.rounds * args.samples_per_round * args.training_repeats
        + args.validation_top_k * args.validation_repeats
    )
    holdout = args.holdout_repeats
    return {
        "context_seed_runs": contexts,
        "estimated_outer_calls": contexts * per_outer,
        "estimated_holdout_calls": contexts * holdout,
        "estimated_total_live_calls": contexts * (per_outer + holdout),
    }


def require_live_confirmation(args: argparse.Namespace, phases: Iterable[str]) -> None:
    live = args.evaluator == "gemini" and bool({"outer", "holdout"} & set(phases))
    estimate = estimate_calls(args)
    print(json.dumps(estimate, indent=2), flush=True)
    if live and not args.confirm_live_cost and not args.dry_run:
        raise SystemExit(
            "Live VLM work requires --confirm-live-cost after reviewing the estimate above. "
            "Run first with --dry-run or --evaluator mock."
        )


def preflight_revised_space_ranges() -> None:
    """Refuse to mix the local Space equation with legacy global ranges."""
    path = ROOT / "configs" / "normalisation_ranges_balanced_3gestures_by_gesture.json"
    if not path.exists():
        raise SystemExit(
            f"Missing revised normalisation file: {path}. Extract the complete "
            "evaluation patch before running experiments."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for gesture, expected in EXPECTED_SPACE_RANGES.items():
        found = payload.get(gesture, {}).get("space_indirectness", {})
        pair = (finite(found.get("min")), finite(found.get("max")))
        if not np.allclose(pair, expected, rtol=0.0, atol=1e-8):
            failures.append(f"{gesture}: expected {expected}, found {pair}")
    if failures:
        raise SystemExit(
            "The Space descriptor/range preflight failed. The local-window "
            "descriptor is being paired with legacy normalisation ranges:\n  "
            + "\n  ".join(failures)
        )
    print(f"Space descriptor/range preflight passed: {path}", flush=True)


def run_inner(args: argparse.Namespace, base: Path) -> None:
    out = base / "inner_matrix"
    out.mkdir(parents=True, exist_ok=True)
    for gesture in args.gestures:
        expected = out / f"matrix_{gesture}.csv"
        if expected.exists():
            print(f"SKIP completed inner matrix: {expected}")
            continue
        command = [
            sys.executable, "scripts/debug/run_full_realisability_matrix.py",
            "--gesture", gesture,
            "--states", *args.states,
            "--seeds", *map(str, args.inner_seeds),
            "--maxiter", str(args.inner_matrix_maxiter),
            "--popsize", str(args.inner_matrix_popsize),
            "--local-maxiter", str(args.inner_matrix_local_maxiter),
            "--tolerance", str(args.tolerance),
            "--out", str(out),
        ]
        run(command, dry_run=args.dry_run)


def run_outer(args: argparse.Namespace, base: Path) -> None:
    for gesture in args.gestures:
        for state in args.states:
            for seed in args.seeds:
                out = experiment_dir(base, gesture, state, seed)
                summary = out / "results_summary.json"
                if summary.exists():
                    print(f"SKIP completed outer run: {summary}")
                    continue
                command = [
                    sys.executable, "scripts/train_cem_contextual_bandit.py",
                    "--gesture", gesture, "--target-state", state,
                    "--seed", str(seed), "--out", str(out),
                    "--evaluator", args.evaluator, "--model", args.model,
                    "--temperature", str(args.temperature),
                    "--rounds", str(args.rounds),
                    "--cem-samples-per-round", str(args.samples_per_round),
                    "--repeats", str(args.training_repeats),
                    "--validation-top-k", str(args.validation_top_k),
                    "--validation-repeats", str(args.validation_repeats),
                    "--maxiter", str(args.maxiter), "--popsize", str(args.popsize),
                    "--local-maxiter", str(args.local_maxiter),
                    "--max-feature-error-threshold", str(args.tolerance),
                ]
                # A non-empty folder with a checkpoint resumes automatically.
                run(command, dry_run=args.dry_run)


def run_holdout(args: argparse.Namespace, base: Path) -> None:
    if args.evaluator != "gemini":
        raise SystemExit("Frozen holdout uses Gemini; rerun this phase with --evaluator gemini.")
    for gesture in args.gestures:
        for state in args.states:
            for seed in args.seeds:
                exp = experiment_dir(base, gesture, state, seed)
                summary = exp / "results_summary.json"
                result = exp / "fixed_profile_holdout.json"
                if not summary.exists():
                    print(f"MISSING outer result, cannot hold out: {summary}")
                    continue
                if result.exists() and not args.overwrite_holdout:
                    print(f"SKIP existing frozen holdout: {result}")
                    continue
                command = [
                    sys.executable, "scripts/debug/evaluate_fixed_profile_holdout.py",
                    "--experiment-dir", str(exp),
                    "--repeats", str(args.holdout_repeats),
                    "--model", args.model, "--temperature", str(args.temperature),
                ]
                if args.overwrite_holdout:
                    command.append("--overwrite")
                run(command, dry_run=args.dry_run)


def finite(value) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    a = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    if not len(a):
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def collect_holdouts(base: Path, args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    for gesture in args.gestures:
        for state in args.states:
            for seed in args.seeds:
                path = experiment_dir(base, gesture, state, seed) / "fixed_profile_holdout.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                result = payload["holdout_result"]
                errors = result.get("per_feature_abs_error") or {}
                row = {
                    "gesture": gesture, "state": state, "seed": seed,
                    "completed_repeats": int(payload.get("completed_repeats", 0)),
                    "mean_target_probability": finite(result.get("mean_target_probability")),
                    "classification_rate": finite(result.get("target_classification_rate")),
                    "mean_margin": finite(result.get("mean_margin")),
                    "perceptual_sd": finite(result.get("perceptual_reward_std")),
                    "outer_reward": finite(result.get("outer_reward")),
                    "realisation_rmse": finite(result.get("realisation_rmse")),
                    "max_feature_error": finite(result.get("max_abs_feature_error")),
                    "valid_realisation": bool(result.get("valid_realisation")),
                    "physically_acceptable": bool(result.get("physically_acceptable")),
                    "feature_acceptable": bool(result.get("feature_realisation_acceptable")),
                }
                row["perceptual_success"] = bool(
                    row["classification_rate"] >= 0.70 and row["mean_margin"] > 0
                )
                row["full_success"] = bool(
                    row["perceptual_success"] and row["physically_acceptable"]
                    and row["realisation_rmse"] <= args.tolerance
                    and row["max_feature_error"] <= args.tolerance
                )
                for feature in FEATURES:
                    row[f"error_{feature}"] = finite(errors.get(feature))
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def aggregate_contexts(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["gesture"], row["state"])].append(row)
    output = []
    metrics = [
        "mean_target_probability", "classification_rate", "mean_margin",
        "perceptual_sd", "outer_reward", "realisation_rmse", "max_feature_error",
        *[f"error_{f}" for f in FEATURES],
    ]
    for gesture in GESTURES:
        for state in STATES:
            group = grouped.get((gesture, state), [])
            if not group:
                continue
            item = {"gesture": gesture, "state": state, "n_seeds": len(group)}
            for metric in metrics:
                item[f"mean_{metric}"], item[f"sd_{metric}"] = mean_sd(
                    finite(x.get(metric)) for x in group
                )
            for metric in ["valid_realisation", "physically_acceptable", "feature_acceptable", "perceptual_success", "full_success"]:
                item[f"{metric}_rate"] = float(np.mean([bool(x[metric]) for x in group]))
            output.append(item)
    return output


def matrix(aggregate: list[dict], field: str, gestures: list[str], states: list[str]) -> np.ndarray:
    lookup = {(x["gesture"], x["state"]): finite(x.get(field)) for x in aggregate}
    return np.asarray([[lookup.get((g, s), np.nan) for s in states] for g in gestures], float)


def heatmap(data: np.ndarray, gestures: list[str], states: list[str], title: str, label: str, path: Path, *, vmin=None, vmax=None, percent=False) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
    im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(states)), [s.title() for s in states], rotation=25, ha="right")
    ax.set_yticks(range(len(gestures)), [g.title() for g in gestures])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                text = f"{100*data[i,j]:.1f}%" if percent else f"{data[i,j]:.3f}"
                ax.text(j, i, text, ha="center", va="center", color="white" if data[i,j] < np.nanmean(data) else "black", fontsize=8)
    ax.set_title(title, loc="left", fontweight="bold"); fig.colorbar(im, ax=ax, label=label)
    fig.savefig(path, dpi=220); plt.close(fig)


def analyse_inner(base: Path, report: Path, args: argparse.Namespace) -> dict:
    rows = []
    for gesture in args.gestures:
        path = base / "inner_matrix" / f"matrix_{gesture}.csv"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
    if not rows:
        return {"inner_runs": 0}
    summary = {
        "inner_runs": len(rows),
        "finite_feature_rate": float(np.mean([r["valid_features"].lower() == "true" for r in rows])),
        "feature_realisation_rate": float(np.mean([r["realised_all_features"].lower() == "true" for r in rows])),
        "full_physical_realisation_rate": float(np.mean([r["fully_acceptable"].lower() == "true" for r in rows])),
        "mean_unclipped_rmse": float(np.mean([finite(r["unclipped_rmse"]) for r in rows])),
        "mean_max_feature_error": float(np.mean([finite(r["unclipped_max_abs_error"]) for r in rows])),
    }
    write_csv(report / "inner_run_results.csv", rows)
    return summary


def run_report(args: argparse.Namespace, base: Path) -> None:
    report = base / "report"
    report.mkdir(parents=True, exist_ok=True)
    rows = collect_holdouts(base, args)
    aggregate = aggregate_contexts(rows)
    write_csv(report / "holdout_seed_results.csv", rows)
    write_csv(report / "holdout_context_summary.csv", aggregate)
    if aggregate:
        heatmap(matrix(aggregate, "mean_classification_rate", args.gestures, args.states), args.gestures, args.states,
                "Frozen-holdout target classification rate", "Rate", report / "classification_rate_heatmap.png", vmin=0, vmax=1, percent=True)
        heatmap(matrix(aggregate, "mean_mean_target_probability", args.gestures, args.states), args.gestures, args.states,
                "Frozen-holdout mean target probability", "Probability", report / "target_probability_heatmap.png", vmin=0, vmax=1)
        heatmap(matrix(aggregate, "mean_realisation_rmse", args.gestures, args.states), args.gestures, args.states,
                "Requested-to-achieved Laban RMSE", "RMSE", report / "realisation_rmse_heatmap.png", vmin=0)
        heatmap(matrix(aggregate, "full_success_rate", args.gestures, args.states), args.gestures, args.states,
                "Full-success rate across seeds", "Rate", report / "full_success_heatmap.png", vmin=0, vmax=1, percent=True)
        feature_means = []
        labels = []
        for feature, label in zip(FEATURES, FEATURE_LABELS):
            values = [finite(r[f"error_{feature}"]) for r in rows]
            feature_means.append(np.nanmean(values)); labels.append(label)
        fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
        ax.bar(labels, feature_means, color="#2878b5"); ax.axhline(args.tolerance, ls="--", color="#c43c39", label=f"tolerance={args.tolerance:.2f}")
        ax.set_ylabel("Mean absolute normalised error"); ax.set_title("Per-feature realisation error across all holdouts", loc="left", fontweight="bold")
        ax.legend(); ax.grid(axis="y", alpha=.2); fig.savefig(report / "per_feature_error.png", dpi=220); plt.close(fig)

    expected = len(args.gestures) * len(args.states) * len(args.seeds)
    inner = analyse_inner(base, report, args)
    overall = {
        "descriptor_version": "bounded_local_space_window_fraction_0.075",
        "expected_outer_context_seed_runs": expected,
        "completed_holdout_runs": len(rows),
        "coverage_rate": len(rows) / expected if expected else 0.0,
        "holdout_judgements": int(sum(r["completed_repeats"] for r in rows)),
        "seed_level_perceptual_success_rate": float(np.mean([r["perceptual_success"] for r in rows])) if rows else None,
        "seed_level_full_success_rate": float(np.mean([r["full_success"] for r in rows])) if rows else None,
        "mean_classification_rate": mean_sd(r["classification_rate"] for r in rows)[0] if rows else None,
        "mean_target_probability": mean_sd(r["mean_target_probability"] for r in rows)[0] if rows else None,
        "mean_realisation_rmse": mean_sd(r["realisation_rmse"] for r in rows)[0] if rows else None,
        "mean_max_feature_error": mean_sd(r["max_feature_error"] for r in rows)[0] if rows else None,
        **inner,
    }
    (report / "overall_metrics.json").write_text(json.dumps(overall, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(overall, indent=2))
    print(f"Report written to {report}")


def main() -> None:
    args = parse_args()
    base = Path(args.out)
    if not base.is_absolute():
        base = ROOT / base
    base.mkdir(parents=True, exist_ok=True)
    phases = [args.phase] if args.phase != "all" else ["inner", "outer", "holdout", "report"]
    if {"inner", "outer", "holdout"} & set(phases):
        preflight_revised_space_ranges()
    require_live_confirmation(args, phases)
    metadata = {**vars(args), "out": str(base), "call_estimate": estimate_calls(args)}
    (base / "evaluation_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for phase in phases:
        if phase == "inner": run_inner(args, base)
        elif phase == "outer": run_outer(args, base)
        elif phase == "holdout": run_holdout(args, base)
        elif phase == "report": run_report(args, base)


if __name__ == "__main__":
    main()
