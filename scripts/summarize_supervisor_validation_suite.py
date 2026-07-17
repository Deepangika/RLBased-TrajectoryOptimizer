#!/usr/bin/env python3
"""Aggregate a supervisor validation suite into tables, plots, and a report.

Only frozen-profile holdout evaluations are used for the headline success
metrics. Training and shortlist-validation values are retained as diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/laban_rl_matplotlib")
import matplotlib.pyplot as plt
import numpy as np


FEATURES = [
    "weight",
    "time",
    "flow_boundness",
    "space_indirectness",
    "shape_arcness",
]
GESTURES = ["wave", "reach", "point"]
STATES = ["confident", "calm", "hesitant", "friendly", "confused", "angry"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", required=True)
    parser.add_argument("--classification-threshold", type=float, default=0.70)
    parser.add_argument("--rmse-threshold", type=float, default=0.10)
    parser.add_argument("--max-error-threshold", type=float, default=0.10)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def context_dirs(suite_dir: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in suite_dir.glob("*/results_summary.json")
        if (path.parent / "fixed_profile_holdout.json").exists()
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return centre - radius, centre + radius


def collect_rows(suite_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment_dir in context_dirs(suite_dir):
        summary = load_json(experiment_dir / "results_summary.json")
        holdout_payload = load_json(experiment_dir / "fixed_profile_holdout.json")
        holdout = holdout_payload["holdout_result"]
        initial = summary["initial_profile_validation"]
        selected = summary.get("independent_selection", {}).get("selected", {})

        classification = float(holdout["target_classification_rate"])
        margin = float(holdout["mean_margin"])
        rmse = float(holdout["realisation_rmse"])
        max_error = float(holdout["max_abs_feature_error"])
        perceptual_success = classification >= args.classification_threshold and margin > 0.0
        feature_success = rmse <= args.rmse_threshold and max_error <= args.max_error_threshold
        physical_success = bool(holdout["physically_acceptable"])

        row: dict[str, Any] = {
            "gesture": summary["gesture"],
            "target_state": summary["target_state"],
            "seed": int(summary["seed"]),
            "training_rounds": int(summary["num_rounds_completed"]),
            "selected_training_rank": selected.get("rank"),
            "completed_holdout_repeats": int(holdout_payload["completed_repeats"]),
            "initial_target_probability": float(initial["mean_target_probability"]),
            "holdout_target_probability": float(holdout["mean_target_probability"]),
            "probability_improvement": float(holdout["mean_target_probability"]) - float(initial["mean_target_probability"]),
            "holdout_classification_rate": classification,
            "holdout_margin": margin,
            "holdout_outer_reward": float(holdout["outer_reward"]),
            "holdout_reward_std": float(holdout["perceptual_reward_std"]),
            "realisation_rmse": rmse,
            "max_feature_error": max_error,
            "max_error_feature": holdout["max_error_feature"],
            "physical_success": physical_success,
            "perceptual_success": perceptual_success,
            "feature_success": feature_success,
            "full_success": physical_success and perceptual_success and feature_success,
            "requested_profile": holdout["requested_profile"],
            "achieved_profile": holdout["achieved_profile"],
            "per_feature_abs_error": holdout["per_feature_abs_error"],
            "experiment_dir": str(experiment_dir.resolve()),
        }
        rows.append(row)
    return rows


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    scalar_fields = [key for key, value in rows[0].items() if not isinstance(value, dict)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in scalar_fields})


def make_plots(rows: list[dict[str, Any]], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['gesture']}\n{r['target_state']}" for r in rows]
    x = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    width = 0.34
    ax.bar(x - width / 2, [r["initial_target_probability"] for r in rows], width, label="Initial profile", color="#9aa6b2")
    ax.bar(x + width / 2, [r["holdout_target_probability"] for r in rows], width, label="Frozen-profile holdout", color="#2878b5")
    ax.axhline(0.5, color="#333333", linestyle="--", linewidth=1, label="0.50 reference")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean target-state probability")
    ax.set_title("Target-state probability before and after optimisation")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "target_probability_improvement.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colours = ["#2a9d8f" if r["perceptual_success"] else "#d1495b" for r in rows]
    bars = ax.bar(x, [r["holdout_classification_rate"] for r in rows], color=colours)
    ax.axhline(0.70, color="#222222", linestyle="--", linewidth=1.2, label="Success threshold (0.70)")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Target classification rate")
    ax.set_title("Independent frozen-profile classification performance")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for bar, row in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{row['holdout_classification_rate']:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(figures_dir / "holdout_classification_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    rmse_bars = axes[0].bar(x, [r["realisation_rmse"] for r in rows], color="#5b8e7d")
    axes[0].axhline(0.10, color="#222222", linestyle="--", linewidth=1.2)
    axes[0].set_title("Feature-vector RMSE")
    axes[0].set_ylabel("Normalised error")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    max_bars = axes[1].bar(x, [r["max_feature_error"] for r in rows], color="#e09f3e")
    axes[1].axhline(0.10, color="#222222", linestyle="--", linewidth=1.2, label="Acceptance threshold")
    axes[1].set_title("Worst individual feature error")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    for bars, axis, key in ((rmse_bars, axes[0], "realisation_rmse"), (max_bars, axes[1], "max_feature_error")):
        for bar, row in zip(bars, rows):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"{row[key]:.3f}", ha="center", fontsize=8)
    fig.suptitle("Inner-loop trajectory realisation accuracy")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_realisability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    matrix = np.full((len(GESTURES), len(STATES)), np.nan)
    for row in rows:
        matrix[GESTURES.index(row["gesture"]), STATES.index(row["target_state"])] = row["holdout_classification_rate"]
    masked = np.ma.masked_invalid(matrix)
    fig, ax = plt.subplots(figsize=(10, 4.3))
    image = ax.imshow(masked, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(STATES)), STATES, rotation=25, ha="right")
    ax.set_yticks(range(len(GESTURES)), GESTURES)
    ax.set_xlabel("Target affective state")
    ax.set_ylabel("Gesture")
    ax.set_title("Gesture–state holdout classification matrix")
    for i in range(len(GESTURES)):
        for j in range(len(STATES)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center", fontweight="bold")
            else:
                ax.text(j, i, "not run", ha="center", va="center", color="#777777", fontsize=8)
    fig.colorbar(image, ax=ax, label="Classification rate")
    fig.tight_layout()
    fig.savefig(figures_dir / "gesture_state_matrix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(rows: list[dict[str, Any]], aggregate: dict[str, Any], path: Path) -> None:
    lines = [
        "# Supervisor validation suite",
        "",
        "## Evaluation design",
        "",
        "Each context was optimised with CEM, independently reranked, frozen, and then evaluated on fresh holdout judgements. Headline metrics below use only the frozen-profile holdout; they do not reuse training rewards.",
        "",
        "Success requires: classification rate $\\geq 0.70$, positive mean margin, feature RMSE $\\leq 0.10$, maximum feature error $\\leq 0.10$, and physical acceptance.",
        "",
        "## Aggregate results",
        "",
        f"- Completed contexts: **{aggregate['completed_contexts']}**",
        f"- Completed holdout judgements: **{aggregate['completed_holdout_judgements']}**",
        f"- Physical-validity rate: **{aggregate['physical_success_rate']:.1%}**",
        f"- Feature-realisability success rate: **{aggregate['feature_success_rate']:.1%}**",
        f"- Perceptual success rate: **{aggregate['perceptual_success_rate']:.1%}**",
        f"- Full-system success rate: **{aggregate['full_success_rate']:.1%}**",
        f"- Mean holdout target probability: **{aggregate['mean_holdout_target_probability']:.3f}**",
        f"- Mean feature RMSE: **{aggregate['mean_realisation_rmse']:.3f}**",
        "",
        "Because this suite contains a small number of gesture–state contexts, Wilson intervals are descriptive uncertainty bounds rather than population-level claims.",
        "",
        "## Per-context results",
        "",
        "| Gesture | State | Initial $p$ | Holdout $p$ | Class. rate | Margin | RMSE | Max error | Full success |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['gesture']} | {row['target_state']} | {row['initial_target_probability']:.3f} | "
            f"{row['holdout_target_probability']:.3f} | {row['holdout_classification_rate']:.1%} | "
            f"{row['holdout_margin']:+.3f} | {row['realisation_rmse']:.3f} | "
            f"{row['max_feature_error']:.3f} ({row['max_error_feature']}) | "
            f"{'PASS' if row['full_success'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Figures",
        "",
        "![Target probability improvement](figures/target_probability_improvement.png)",
        "",
        "![Holdout classification rate](figures/holdout_classification_rate.png)",
        "",
        "![Feature realisability](figures/feature_realisability.png)",
        "",
        "![Gesture-state matrix](figures/gesture_state_matrix.png)",
        "",
        "## Interpretation rule",
        "",
        "A low trajectory error with poor perceptual performance indicates a representational or evaluator limitation, rather than failure of the inner trajectory optimiser. Conversely, strong perception with excessive feature error should not be presented as faithful realisation of the requested Laban profile.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    suite_dir = Path(args.suite_dir).resolve()
    rows = collect_rows(suite_dir, args)
    if not rows:
        raise RuntimeError(f"No completed fixed-profile holdouts found below {suite_dir}")

    output_dir = suite_dir / "supervisor_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, output_dir / "context_metrics.csv")

    total = len(rows)
    counts = {
        "physical": sum(r["physical_success"] for r in rows),
        "feature": sum(r["feature_success"] for r in rows),
        "perceptual": sum(r["perceptual_success"] for r in rows),
        "full": sum(r["full_success"] for r in rows),
    }
    aggregate = {
        "completed_contexts": total,
        "completed_holdout_judgements": sum(r["completed_holdout_repeats"] for r in rows),
        "thresholds": {
            "classification_rate": args.classification_threshold,
            "positive_margin_required": True,
            "realisation_rmse": args.rmse_threshold,
            "max_feature_error": args.max_error_threshold,
        },
        "physical_success_rate": counts["physical"] / total,
        "feature_success_rate": counts["feature"] / total,
        "perceptual_success_rate": counts["perceptual"] / total,
        "full_success_rate": counts["full"] / total,
        "perceptual_success_wilson_95": wilson_interval(counts["perceptual"], total),
        "full_success_wilson_95": wilson_interval(counts["full"], total),
        "mean_holdout_target_probability": float(np.mean([r["holdout_target_probability"] for r in rows])),
        "mean_classification_rate": float(np.mean([r["holdout_classification_rate"] for r in rows])),
        "mean_margin": float(np.mean([r["holdout_margin"] for r in rows])),
        "mean_realisation_rmse": float(np.mean([r["realisation_rmse"] for r in rows])),
        "mean_max_feature_error": float(np.mean([r["max_feature_error"] for r in rows])),
    }
    payload = {"aggregate": aggregate, "contexts": rows}
    (output_dir / "supervisor_results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    make_plots(rows, output_dir / "figures")
    build_report(rows, aggregate, output_dir / "SUPERVISOR_REPORT.md")
    print(f"Completed contexts: {total}")
    print(f"Perceptual success rate: {aggregate['perceptual_success_rate']:.1%}")
    print(f"Full-system success rate: {aggregate['full_success_rate']:.1%}")
    print(f"Report: {output_dir / 'SUPERVISOR_REPORT.md'}")


if __name__ == "__main__":
    main()
