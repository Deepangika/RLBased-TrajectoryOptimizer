"""Focused DE parameter study and DE/L-BFGS-B ablation.

Place beside:

    de_evaluation_suite.py
    de_inner_optimizer_adapter.py

Full run:

    python scripts/evaluation/de_focused_ablation.py \
        --output-dir outputs/de_focused_ablation

Quick integration test:

    python scripts/evaluation/de_focused_ablation.py \
        --quick \
        --output-dir outputs/de_focused_ablation_quick

The experiment contains:

1. A focused Rastrigin grid:
       F  = [0.3, 0.4, 0.5]
       CR = [0.35, 0.50, 0.65]
   with 10 independent trials per combination.

2. A three-way ablation on Sphere, Rosenbrock, and Rastrigin:
       - DE only
       - L-BFGS-B only
       - DE followed by L-BFGS-B

For each seed, all methods are tied to the same initial population. L-BFGS-B
starts from the best member of that population; DE uses the complete
population; the hybrid applies L-BFGS-B to DE's final best solution.

The methods do not receive identical evaluation budgets. This is deliberate:
the CSV and summary report objective-function evaluations so performance and
computational cost can be interpreted together.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from de_evaluation_suite import (
    BENCHMARKS,
    DEConfig,
    Benchmark,
    BoundaryMonitor,
)
from de_inner_optimizer_adapter import run_custom_de


FOCUSED_F = (0.3, 0.4, 0.5)
FOCUSED_CR = (0.35, 0.50, 0.65)
METHOD_ORDER = ("DE only", "L-BFGS-B only", "DE + L-BFGS-B")
METHOD_COLORS = ("#4c78a8", "#f58518", "#54a24b")


@dataclass(frozen=True)
class SensitivityRecord:
    crossover_rate: float
    differential_weight: float
    trial: int
    seed: int
    final_fitness: float
    runtime_seconds: float


@dataclass(frozen=True)
class AblationRecord:
    function: str
    method: str
    trial: int
    seed: int
    initial_best_fitness: float
    final_fitness: float
    success: bool
    objective_evaluations: int
    runtime_seconds: float


class CountingObjective:
    """Count objective calls without changing the underlying function."""

    def __init__(self, objective: Any) -> None:
        self.objective = objective
        self.calls = 0

    def __call__(self, vector: np.ndarray) -> float:
        self.calls += 1
        value = float(self.objective(vector))
        if not math.isfinite(value):
            raise ValueError("Objective returned NaN or infinite fitness")
        return value


def shared_initial_population(config: DEConfig, seed: int) -> np.ndarray:
    """Reproduce the population used by ``run_custom_de`` for a seed."""
    rng = np.random.default_rng(seed)
    return rng.uniform(
        config.lower_bound,
        config.upper_bound,
        size=(config.population_size, config.dimensions),
    )


def initial_best(
    benchmark: Benchmark,
    config: DEConfig,
    seed: int,
) -> tuple[np.ndarray, float]:
    population = shared_initial_population(config, seed)
    fitness = np.asarray(
        [benchmark.objective(vector) for vector in population],
        dtype=float,
    )
    index = int(np.argmin(fitness))
    return population[index].copy(), float(fitness[index])


def run_local_search(
    benchmark: Benchmark,
    config: DEConfig,
    start: np.ndarray,
    local_maxiter: int,
) -> tuple[float, int, float]:
    """Run bounded L-BFGS-B and return fitness, evaluations, and runtime."""
    counted = CountingObjective(benchmark.objective)
    bounds = [
        (config.lower_bound, config.upper_bound)
        for _ in range(config.dimensions)
    ]
    started = time.perf_counter()
    result = minimize(
        counted,
        np.asarray(start, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": local_maxiter,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 30,
        },
    )
    elapsed = time.perf_counter() - started
    return float(result.fun), counted.calls, elapsed


def run_de_counted(
    benchmark: Benchmark,
    config: DEConfig,
    seed: int,
) -> tuple[Any, int, float]:
    """Run the project's SciPy-DE adapter while counting evaluations."""
    counted = CountingObjective(benchmark.objective)
    started = time.perf_counter()
    result = run_custom_de(
        counted,
        config,
        seed,
        generation_callback=None,
        boundary_callback=BoundaryMonitor(warn=False),
    )
    elapsed = time.perf_counter() - started
    return result, counted.calls, elapsed


def run_focused_sensitivity(
    config: DEConfig,
    trials: int,
    seed_base: int,
) -> list[SensitivityRecord]:
    """Evaluate the refined F/CR grid on Rastrigin."""
    rastrigin = next(item for item in BENCHMARKS if item.name == "Rastrigin")
    records: list[SensitivityRecord] = []

    print("\nFOCUSED RASTRIGIN PARAMETER STUDY")
    for cr_index, cr in enumerate(FOCUSED_CR):
        for f_index, differential_weight in enumerate(FOCUSED_F):
            values: list[float] = []
            for trial in range(trials):
                seed = (
                    seed_base
                    + cr_index * 100_000
                    + f_index * 10_000
                    + trial
                )
                trial_config = replace(
                    config,
                    crossover_rate=cr,
                    differential_weight=differential_weight,
                )
                started = time.perf_counter()
                result = run_custom_de(
                    rastrigin.objective,
                    trial_config,
                    seed,
                    generation_callback=None,
                    boundary_callback=None,
                )
                elapsed = time.perf_counter() - started
                values.append(float(result.best_fitness))
                records.append(
                    SensitivityRecord(
                        crossover_rate=cr,
                        differential_weight=differential_weight,
                        trial=trial + 1,
                        seed=seed,
                        final_fitness=float(result.best_fitness),
                        runtime_seconds=elapsed,
                    )
                )
            print(
                f"  CR={cr:.2f}, F={differential_weight:.2f}: "
                f"mean={np.mean(values):.6e}, SD={np.std(values, ddof=1):.3e}"
            )
    return records


def run_ablation(
    config: DEConfig,
    trials: int,
    local_maxiter: int,
    success_threshold: float,
    seed_base: int,
) -> list[AblationRecord]:
    """Run DE, local-only, and hybrid variants from shared seeded starts."""
    records: list[AblationRecord] = []

    print("\nDE / L-BFGS-B ABLATION")
    for function_index, benchmark in enumerate(BENCHMARKS):
        print(f"\n{benchmark.name}")
        for trial in range(trials):
            seed = seed_base + function_index * 100_000 + trial
            start, starting_fitness = initial_best(benchmark, config, seed)

            de_result, de_calls, de_time = run_de_counted(
                benchmark,
                config,
                seed,
            )
            local_fitness, local_calls, local_time = run_local_search(
                benchmark,
                config,
                start,
                local_maxiter,
            )
            hybrid_fitness, hybrid_local_calls, hybrid_local_time = (
                run_local_search(
                    benchmark,
                    config,
                    de_result.best_vector,
                    local_maxiter,
                )
            )

            rows = (
                (
                    "DE only",
                    float(de_result.best_fitness),
                    de_calls,
                    de_time,
                ),
                (
                    "L-BFGS-B only",
                    local_fitness,
                    local_calls,
                    local_time,
                ),
                (
                    "DE + L-BFGS-B",
                    hybrid_fitness,
                    de_calls + hybrid_local_calls,
                    de_time + hybrid_local_time,
                ),
            )
            for method, fitness, calls, runtime in rows:
                records.append(
                    AblationRecord(
                        function=benchmark.name,
                        method=method,
                        trial=trial + 1,
                        seed=seed,
                        initial_best_fitness=starting_fitness,
                        final_fitness=fitness,
                        success=fitness < success_threshold,
                        objective_evaluations=calls,
                        runtime_seconds=runtime,
                    )
                )

            print(
                f"  Trial {trial + 1:>2}/{trials}: "
                f"DE={de_result.best_fitness:.3e}, "
                f"local={local_fitness:.3e}, "
                f"hybrid={hybrid_fitness:.3e}",
                flush=True,
            )
    return records


def summarize_sensitivity(
    records: list[SensitivityRecord],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for cr in FOCUSED_CR:
        for differential_weight in FOCUSED_F:
            values = np.asarray(
                [
                    row.final_fitness
                    for row in records
                    if row.crossover_rate == cr
                    and row.differential_weight == differential_weight
                ],
                dtype=float,
            )
            rows.append(
                {
                    "crossover_rate": cr,
                    "differential_weight": differential_weight,
                    "mean_final_fitness": float(np.mean(values)),
                    "median_final_fitness": float(np.median(values)),
                    "standard_deviation": float(np.std(values, ddof=1)),
                    "minimum_final_fitness": float(np.min(values)),
                }
            )
    return rows


def summarize_ablation(
    records: list[AblationRecord],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for benchmark in BENCHMARKS:
        for method in METHOD_ORDER:
            selected = [
                row
                for row in records
                if row.function == benchmark.name and row.method == method
            ]
            fitness = np.asarray(
                [row.final_fitness for row in selected],
                dtype=float,
            )
            evaluations = np.asarray(
                [row.objective_evaluations for row in selected],
                dtype=float,
            )
            runtimes = np.asarray(
                [row.runtime_seconds for row in selected],
                dtype=float,
            )
            rows.append(
                {
                    "function": benchmark.name,
                    "method": method,
                    "mean_final_fitness": float(np.mean(fitness)),
                    "median_final_fitness": float(np.median(fitness)),
                    "standard_deviation": float(np.std(fitness, ddof=1)),
                    "success_rate_percent": float(
                        100.0
                        * np.mean([row.success for row in selected])
                    ),
                    "mean_objective_evaluations": float(
                        np.mean(evaluations)
                    ),
                    "mean_runtime_seconds": float(np.mean(runtimes)),
                }
            )
    return rows


def save_csv(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def plot_focused_heatmap(
    summary: list[dict[str, float]],
    output_path: Path,
) -> None:
    matrix = np.empty((len(FOCUSED_CR), len(FOCUSED_F)), dtype=float)
    for row in summary:
        i = FOCUSED_CR.index(float(row["crossover_rate"]))
        j = FOCUSED_F.index(float(row["differential_weight"]))
        matrix[i, j] = float(row["mean_final_fitness"])

    fig, axis = plt.subplots(figsize=(9.4, 7.0))
    image = axis.imshow(matrix, cmap="viridis_r", aspect="auto")
    axis.set_xticks(range(len(FOCUSED_F)), [f"{x:.2f}" for x in FOCUSED_F])
    axis.set_yticks(
        range(len(FOCUSED_CR)),
        [f"{x:.2f}" for x in FOCUSED_CR],
    )
    axis.set_xlabel("Differential weight F")
    axis.set_ylabel("Crossover rate CR")
    axis.set_title("Focused Rastrigin parameter sensitivity")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("Mean final fitness (lower is better)")

    midpoint = float(np.mean([np.min(matrix), np.max(matrix)]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > midpoint else "black"
            axis.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(
    records: list[AblationRecord],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.5))
    for axis, benchmark in zip(axes, BENCHMARKS, strict=True):
        values = [
            np.maximum(
                [
                    row.final_fitness
                    for row in records
                    if row.function == benchmark.name
                    and row.method == method
                ],
                1e-16,
            )
            for method in METHOD_ORDER
        ]
        boxes = axis.boxplot(
            values,
            tick_labels=("DE", "L-BFGS-B", "Hybrid"),
            patch_artist=True,
            widths=0.6,
        )
        for patch, color in zip(
            boxes["boxes"],
            METHOD_COLORS,
            strict=True,
        ):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axis.set_yscale("log")
        axis.set_title(benchmark.name)
        axis.set_ylabel("Final fitness (log scale)")
        axis.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Inner-optimiser component ablation", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def markdown_ablation_table(
    summary: list[dict[str, float | str]],
) -> str:
    lines = [
        "| Function | Method | Mean fitness | Median fitness | SD | "
        "Success | Mean evaluations | Mean runtime (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['function']} | {row['method']} | "
            f"{float(row['mean_final_fitness']):.6e} | "
            f"{float(row['median_final_fitness']):.6e} | "
            f"{float(row['standard_deviation']):.6e} | "
            f"{float(row['success_rate_percent']):.1f}% | "
            f"{float(row['mean_objective_evaluations']):.1f} | "
            f"{float(row['mean_runtime_seconds']):.3f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Focused DE tuning and DE/L-BFGS-B ablation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/de_focused_ablation"),
    )
    parser.add_argument("--sensitivity-trials", type=int, default=10)
    parser.add_argument("--ablation-trials", type=int, default=30)
    parser.add_argument("--dimensions", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--max-generations", type=int, default=200)
    parser.add_argument("--local-maxiter", type=int, default=100)
    parser.add_argument("--success-threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=730_000)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use two trials and 20 generations for an integration test",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_trials = 2 if args.quick else args.sensitivity_trials
    ablation_trials = 2 if args.quick else args.ablation_trials
    generations = 20 if args.quick else args.max_generations
    local_maxiter = 25 if args.quick else args.local_maxiter

    config = DEConfig(
        dimensions=args.dimensions,
        lower_bound=-5.12,
        upper_bound=5.12,
        max_generations=generations,
        population_size=args.population_size,
        differential_weight=0.8,
        crossover_rate=0.9,
    )
    config.validate()

    print("Focused DE Evaluation")
    print(json.dumps(asdict(config), indent=2))

    sensitivity_records = run_focused_sensitivity(
        config,
        sensitivity_trials,
        args.seed,
    )
    ablation_records = run_ablation(
        config,
        ablation_trials,
        local_maxiter,
        args.success_threshold,
        args.seed + 1_000_000,
    )

    sensitivity_summary = summarize_sensitivity(sensitivity_records)
    ablation_summary = summarize_ablation(ablation_records)

    save_csv(
        output_dir / "focused_rastrigin_trials.csv",
        sensitivity_records,
    )
    save_csv(output_dir / "optimizer_ablation_trials.csv", ablation_records)
    plot_focused_heatmap(
        sensitivity_summary,
        output_dir / "focused_rastrigin_heatmap.png",
    )
    plot_ablation(
        ablation_records,
        output_dir / "optimizer_ablation_boxplots.png",
    )

    markdown = markdown_ablation_table(ablation_summary)
    (output_dir / "optimizer_ablation_summary.md").write_text(
        markdown,
        encoding="utf-8",
    )
    payload = {
        "config": asdict(config),
        "local_maxiter": local_maxiter,
        "sensitivity_trials_per_pair": sensitivity_trials,
        "ablation_trials_per_method_and_function": ablation_trials,
        "success_threshold": args.success_threshold,
        "focused_sensitivity": sensitivity_summary,
        "ablation": ablation_summary,
        "interpretation_note": (
            "Methods share seeded initial populations but not equal objective-"
            "evaluation budgets. Compare fitness together with evaluations."
        ),
    }
    (output_dir / "focused_ablation_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\nABLATION SUMMARY\n")
    print(markdown)
    print(f"Outputs saved to: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
