"""Compare benchmark-tuned DE parameters on a frozen Laban target matrix.

This runner evaluates three configurations:

1. current_scipy_default:
       mutation dithering F ~ U(0.5, 1.0), CR = 0.7
2. tuned_A:
       fixed F = 0.4, CR = 0.35
3. tuned_B:
       fixed F = 0.5, CR = 0.65

Every configuration uses the same gestures, states, seeds, DE generations,
SciPy population multiplier, local-polishing budget, and realisability rule.
States are discovered dynamically from ``TARGET_PROFILES``. No VLM calls are
made.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for candidate in (ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.targets import TARGET_PROFILES  # noqa: E402

GESTURES = ("wave", "reach", "point")
STATES = tuple(TARGET_PROFILES)
SEEDS = (7, 17, 27, 37, 47)
FEATURES = (
    "weight",
    "time",
    "flow_boundness",
    "space_indirectness",
    "shape_arcness",
)
CONFIGURATIONS = (
    {
        "name": "current_scipy_default",
        "label": "Current: dithered F=[0.5,1.0], CR=0.7",
        "mutation": (0.5, 1.0),
        "recombination": 0.7,
    },
    {
        "name": "tuned_A",
        "label": "Tuned A: F=0.4, CR=0.35",
        "mutation": (0.4,),
        "recombination": 0.35,
    },
    {
        "name": "tuned_B",
        "label": "Tuned B: F=0.5, CR=0.65",
        "mutation": (0.5,),
        "recombination": 0.65,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="outputs/de_parameter_transfer",
    )
    parser.add_argument(
        "--gestures",
        nargs="+",
        choices=GESTURES,
        default=list(GESTURES),
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=STATES,
        default=list(STATES),
        help=(
            "Affective states to evaluate. Valid values are loaded dynamically "
            "from laban_rl.targets.TARGET_PROFILES."
        ),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(SEEDS),
    )
    parser.add_argument("--maxiter", type=int, default=45)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--local-maxiter", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun matrices even when their CSV already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running optimisation.",
    )
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    print("\n$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def run_matrices(args: argparse.Namespace, output_root: Path) -> None:
    matrix_script = ROOT / "scripts" / "debug" / "run_full_realisability_matrix.py"
    if not matrix_script.exists():
        raise FileNotFoundError(matrix_script)

    for configuration in CONFIGURATIONS:
        configuration_dir = output_root / str(configuration["name"])
        configuration_dir.mkdir(parents=True, exist_ok=True)
        for gesture in args.gestures:
            expected = configuration_dir / f"matrix_{gesture}.csv"
            if expected.exists() and not args.overwrite:
                print(f"SKIP existing matrix: {expected}", flush=True)
                continue

            command = [
                sys.executable,
                str(matrix_script),
                "--gesture",
                gesture,
                "--states",
                *args.states,
                "--seeds",
                *map(str, args.seeds),
                "--maxiter",
                str(args.maxiter),
                "--popsize",
                str(args.popsize),
                "--local-maxiter",
                str(args.local_maxiter),
                "--tolerance",
                str(args.tolerance),
                "--de-mutation",
                *map(str, configuration["mutation"]),
                "--de-recombination",
                str(configuration["recombination"]),
                "--out",
                str(configuration_dir),
            ]
            run_command(command, args.dry_run)


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def finite(value: str | float | int | None) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_rows(
    args: argparse.Namespace,
    output_root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        for gesture in args.gestures:
            path = (
                output_root
                / str(configuration["name"])
                / f"matrix_{gesture}.csv"
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing completed matrix: {path}. "
                    "Run without --dry-run or inspect the failed command."
                )
            with path.open(newline="", encoding="utf-8") as stream:
                for source in csv.DictReader(stream):
                    row: dict[str, object] = dict(source)
                    row["configuration"] = configuration["name"]
                    row["configuration_label"] = configuration["label"]
                    for key in (
                        "valid_features",
                        "realised_all_features",
                        "path_preserved_0.70_1.30",
                        "joint_limits_satisfied",
                        "fully_acceptable",
                    ):
                        row[key] = parse_bool(str(source[key]))
                    rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    clean = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    return float(np.mean(clean)) if len(clean) else float("nan")


def sample_sd(values: list[float]) -> float:
    clean = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    return float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["configuration"])].append(row)

    summary: list[dict[str, object]] = []
    for configuration in CONFIGURATIONS:
        name = str(configuration["name"])
        selected = groups[name]
        acceptable = [bool(row["fully_acceptable"]) for row in selected]
        realised = [bool(row["realised_all_features"]) for row in selected]
        rmse = [finite(row["unclipped_rmse"]) for row in selected]
        maximum_error = [
            finite(row["unclipped_max_abs_error"]) for row in selected
        ]
        result: dict[str, object] = {
            "configuration": name,
            "configuration_label": configuration["label"],
            "de_mutation": list(configuration["mutation"]),
            "de_recombination": configuration["recombination"],
            "n_runs": len(selected),
            "realisability_success_count": int(sum(realised)),
            "realisability_success_rate_percent": 100.0 * mean(
                [float(x) for x in realised]
            ),
            "fully_acceptable_count": int(sum(acceptable)),
            "fully_acceptable_rate_percent": 100.0 * mean(
                [float(x) for x in acceptable]
            ),
            "mean_unclipped_rmse": mean(rmse),
            "sd_unclipped_rmse": sample_sd(rmse),
            "mean_max_abs_feature_error": mean(maximum_error),
            "mean_total_objective_calls": mean(
                [
                    finite(row["total_objective_calls_including_callbacks"])
                    for row in selected
                ]
            ),
            "mean_de_runtime_seconds": mean(
                [finite(row["de_runtime_seconds"]) for row in selected]
            ),
            "mean_local_runtime_seconds": mean(
                [finite(row["local_runtime_seconds"]) for row in selected]
            ),
            "mean_total_runtime_seconds": mean(
                [
                    finite(row["total_optimisation_runtime_seconds"])
                    for row in selected
                ]
            ),
        }
        for feature in FEATURES:
            result[f"mean_error_{feature}"] = mean(
                [
                    finite(row[f"unclipped_error_{feature}"])
                    for row in selected
                ]
            )
        summary.append(result)
    return summary


def context_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["configuration"]),
            str(row["gesture"]),
            str(row["state"]),
        )
        groups[key].append(row)

    output: list[dict[str, object]] = []
    for (configuration, gesture, state), selected in sorted(groups.items()):
        output.append(
            {
                "configuration": configuration,
                "gesture": gesture,
                "state": state,
                "n_runs": len(selected),
                "fully_acceptable_count": int(
                    sum(bool(row["fully_acceptable"]) for row in selected)
                ),
                "fully_acceptable_rate_percent": 100.0
                * mean(
                    [
                        float(bool(row["fully_acceptable"]))
                        for row in selected
                    ]
                ),
                "mean_unclipped_rmse": mean(
                    [finite(row["unclipped_rmse"]) for row in selected]
                ),
                "mean_max_abs_feature_error": mean(
                    [
                        finite(row["unclipped_max_abs_error"])
                        for row in selected
                    ]
                ),
            }
        )
    return output


def paired_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_case: dict[
        tuple[str, str, int],
        dict[str, dict[str, object]],
    ] = defaultdict(dict)
    for row in rows:
        key = (
            str(row["gesture"]),
            str(row["state"]),
            int(row["seed"]),
        )
        by_case[key][str(row["configuration"])] = row

    output: list[dict[str, object]] = []
    baseline_name = "current_scipy_default"
    for candidate in ("tuned_A", "tuned_B"):
        paired = [
            variants
            for variants in by_case.values()
            if baseline_name in variants and candidate in variants
        ]
        baseline_rmse = np.asarray(
            [finite(item[baseline_name]["unclipped_rmse"]) for item in paired],
            dtype=float,
        )
        candidate_rmse = np.asarray(
            [finite(item[candidate]["unclipped_rmse"]) for item in paired],
            dtype=float,
        )
        output.append(
            {
                "candidate": candidate,
                "baseline": baseline_name,
                "paired_cases": len(paired),
                "candidate_lower_rmse_count": int(
                    np.sum(candidate_rmse < baseline_rmse)
                ),
                "candidate_lower_rmse_rate_percent": float(
                    100.0 * np.mean(candidate_rmse < baseline_rmse)
                ),
                "mean_paired_rmse_difference_candidate_minus_baseline": float(
                    np.mean(candidate_rmse - baseline_rmse)
                ),
                "candidate_acceptability_wins": int(
                    sum(
                        bool(item[candidate]["fully_acceptable"])
                        and not bool(item[baseline_name]["fully_acceptable"])
                        for item in paired
                    )
                ),
                "candidate_acceptability_losses": int(
                    sum(
                        bool(item[baseline_name]["fully_acceptable"])
                        and not bool(item[candidate]["fully_acceptable"])
                        for item in paired
                    )
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    summary: list[dict[str, object]],
) -> None:
    lines = [
        "| Configuration | Strict realisability | Fully acceptable | "
        "Mean RMSE | Mean max error | Mean evaluations | Mean runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['configuration_label']} | "
            f"{row['realisability_success_count']}/{row['n_runs']} "
            f"({float(row['realisability_success_rate_percent']):.1f}%) | "
            f"{row['fully_acceptable_count']}/{row['n_runs']} "
            f"({float(row['fully_acceptable_rate_percent']):.1f}%) | "
            f"{float(row['mean_unclipped_rmse']):.5f} | "
            f"{float(row['mean_max_abs_feature_error']):.5f} | "
            f"{float(row['mean_total_objective_calls']):.1f} | "
            f"{float(row['mean_total_runtime_seconds']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_overall(
    summary: list[dict[str, object]],
    output_path: Path,
) -> None:
    labels = ["Current", "Tuned A", "Tuned B"]
    x = np.arange(len(summary))
    success = [
        float(row["fully_acceptable_rate_percent"]) for row in summary
    ]
    rmse = [float(row["mean_unclipped_rmse"]) for row in summary]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    bars = left.bar(x, success, color=("#4c78a8", "#54a24b", "#eeca3b"))
    left.set_xticks(x, labels)
    left.set_ylim(0, 105)
    left.set_ylabel("Fully acceptable runs (%)")
    left.set_title("Strict inner-loop success")
    left.bar_label(bars, fmt="%.1f%%", padding=3)
    left.grid(True, axis="y", alpha=0.25)

    bars = right.bar(x, rmse, color=("#4c78a8", "#54a24b", "#eeca3b"))
    right.set_xticks(x, labels)
    right.set_ylabel("Mean unclipped feature RMSE")
    right.set_title("Laban-profile realisation error")
    right.bar_label(bars, fmt="%.4f", padding=3)
    right.grid(True, axis="y", alpha=0.25)

    fig.suptitle("DE parameter transfer to the Laban inner optimiser")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    missing_features = {
        state: sorted(set(FEATURES) - set(TARGET_PROFILES[state]))
        for state in args.states
        if set(FEATURES) - set(TARGET_PROFILES[state])
    }
    if missing_features:
        raise ValueError(
            "Every target profile must define all five Laban features: "
            f"{missing_features}"
        )
    output_root = (ROOT / args.out).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "configurations": CONFIGURATIONS,
                "gestures": args.gestures,
                "states": args.states,
                "seeds": args.seeds,
                "expected_runs_per_configuration": (
                    len(args.gestures)
                    * len(args.states)
                    * len(args.seeds)
                ),
                "maxiter": args.maxiter,
                "popsize": args.popsize,
                "local_maxiter": args.local_maxiter,
                "tolerance": args.tolerance,
            },
            indent=2,
        ),
        flush=True,
    )

    run_matrices(args, output_root)
    if args.dry_run:
        print("\nDry run complete; no aggregation was attempted.")
        return 0

    rows = load_rows(args, output_root)
    overall = summarize(rows)
    contexts = context_summary(rows)
    paired = paired_summary(rows)

    write_csv(output_root / "all_transfer_runs.csv", rows)
    write_csv(output_root / "configuration_summary.csv", overall)
    write_csv(output_root / "context_summary.csv", contexts)
    write_csv(output_root / "paired_comparison.csv", paired)
    write_markdown(output_root / "configuration_summary.md", overall)
    plot_overall(overall, output_root / "configuration_comparison.png")

    payload = {
        "configurations": CONFIGURATIONS,
        "experiment": {
            "gestures": args.gestures,
            "states": args.states,
            "seeds": args.seeds,
            "maxiter": args.maxiter,
            "popsize": args.popsize,
            "local_maxiter": args.local_maxiter,
            "tolerance": args.tolerance,
        },
        "overall_summary": overall,
        "paired_comparison": paired,
    }
    (output_root / "transfer_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\n" + (output_root / "configuration_summary.md").read_text())
    print(f"Outputs saved to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
