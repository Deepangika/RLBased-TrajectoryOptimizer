#!/usr/bin/env python3
"""Comprehensive, reproducible evaluation suite for Differential Evolution.

The file is self-contained and runs with a reference DE/rand/1/bin
implementation. To evaluate a custom DE implementation, either:

1. replace ``reference_de`` while preserving its interface; or
2. pass ``--optimizer your_module:your_function``.

The custom callable must accept:

    objective, config, seed, generation_callback, boundary_callback

and return a ``DEResult`` instance.

Default experiment:
    * Sphere, Rosenbrock, Rastrigin
    * 30 independent trials per function
    * D=30, bounds=[-5.12, 5.12], generations=200
    * NP=50, F=0.8, CR=0.9
    * 3x3 Rastrigin F/CR sensitivity grid, 5 trials per cell
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray
Objective = Callable[[Array], float]
GenerationCallback = Callable[[int, float, float, Array], None]
BoundaryCallback = Callable[[str, int, Array, float, float], None]


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def sphere(x: Array) -> float:
    """Sphere benchmark; global optimum f(0, ..., 0) = 0."""
    x = np.asarray(x, dtype=float)
    return float(np.dot(x, x))


def rosenbrock(x: Array) -> float:
    """Rosenbrock benchmark; global optimum f(1, ..., 1) = 0."""
    x = np.asarray(x, dtype=float)
    return float(
        np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)
    )


def rastrigin(x: Array) -> float:
    """Rastrigin benchmark; global optimum f(0, ..., 0) = 0."""
    x = np.asarray(x, dtype=float)
    return float(10.0 * x.size + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))


@dataclass(frozen=True)
class Benchmark:
    name: str
    objective: Objective
    known_optimum: float = 0.0


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark("Sphere", sphere),
    Benchmark("Rosenbrock", rosenbrock),
    Benchmark("Rastrigin", rastrigin),
)


# ---------------------------------------------------------------------------
# DE configuration and result contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DEConfig:
    dimensions: int = 30
    lower_bound: float = -5.12
    upper_bound: float = 5.12
    max_generations: int = 200
    population_size: int = 50
    differential_weight: float = 0.8
    crossover_rate: float = 0.9

    def validate(self) -> None:
        if self.dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        if not self.lower_bound < self.upper_bound:
            raise ValueError("lower_bound must be smaller than upper_bound")
        if self.max_generations < 1:
            raise ValueError("max_generations must be at least 1")
        if self.population_size < 4:
            raise ValueError("DE/rand/1 requires population_size >= 4")
        if not 0.0 < self.differential_weight <= 2.0:
            raise ValueError("differential_weight F must be in (0, 2]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate CR must be in [0, 1]")


@dataclass
class DEResult:
    best_vector: Array
    best_fitness: float
    final_population: Array
    final_fitness: Array
    best_history: Array
    diversity_history: Array
    boundary_violations: int = 0

    def validate(self, config: DEConfig) -> None:
        expected_history = config.max_generations + 1
        if self.best_vector.shape != (config.dimensions,):
            raise ValueError(f"best_vector must have shape {(config.dimensions,)}")
        if self.final_population.shape != (
            config.population_size,
            config.dimensions,
        ):
            raise ValueError(
                "final_population has an invalid shape: "
                f"{self.final_population.shape}"
            )
        if self.final_fitness.shape != (config.population_size,):
            raise ValueError("final_fitness has an invalid shape")
        if self.best_history.shape != (expected_history,):
            raise ValueError(
                f"best_history must contain generation 0 through "
                f"{config.max_generations}"
            )
        if self.diversity_history.shape != (expected_history,):
            raise ValueError("diversity_history has an invalid shape")
        arrays = (
            self.best_vector,
            self.final_population,
            self.final_fitness,
            self.best_history,
            self.diversity_history,
        )
        if not all(np.all(np.isfinite(item)) for item in arrays):
            raise ValueError("DE result contains NaN or infinite values")
        if not np.isfinite(self.best_fitness):
            raise ValueError("best_fitness must be finite")


class DECallable(Protocol):
    def __call__(
        self,
        objective: Objective,
        config: DEConfig,
        seed: int,
        generation_callback: GenerationCallback | None = None,
        boundary_callback: BoundaryCallback | None = None,
    ) -> DEResult: ...


def population_diversity(population: Array) -> float:
    """Mean per-coordinate population standard deviation."""
    return float(np.mean(np.std(population, axis=0, ddof=0)))


class BoundaryMonitor:
    """Counts boundary excursions and emits a single explicit warning."""

    def __init__(self, warn: bool = True) -> None:
        self.count = 0
        self.max_overshoot = 0.0
        self._warn = warn
        self._warning_emitted = False

    def __call__(
        self,
        stage: str,
        generation: int,
        vector: Array,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        below = np.maximum(lower_bound - vector, 0.0)
        above = np.maximum(vector - upper_bound, 0.0)
        violations = (below > 0.0) | (above > 0.0)
        count = int(np.count_nonzero(violations))
        if count == 0:
            return
        self.count += count
        self.max_overshoot = max(
            self.max_overshoot,
            float(np.max(np.maximum(below, above))),
        )
        if self._warn and not self._warning_emitted:
            print(
                "WARNING: Boundary excursion detected before repair "
                f"during {stage} at generation {generation}. "
                "Coordinates will be clipped to the configured bounds.",
                file=sys.stderr,
            )
            self._warning_emitted = True


# ---------------------------------------------------------------------------
# Reference DE implementation -- swap this function for the custom DE core
# ---------------------------------------------------------------------------


def reference_de(
    objective: Objective,
    config: DEConfig,
    seed: int,
    generation_callback: GenerationCallback | None = None,
    boundary_callback: BoundaryCallback | None = None,
) -> DEResult:
    """Run canonical DE/rand/1/bin with deterministic clipped boundary repair."""
    config.validate()
    rng = np.random.default_rng(seed)
    lower, upper = config.lower_bound, config.upper_bound
    npop, ndim = config.population_size, config.dimensions

    population = rng.uniform(lower, upper, size=(npop, ndim))
    fitness = np.fromiter(
        (objective(vector) for vector in population),
        dtype=float,
        count=npop,
    )
    if not np.all(np.isfinite(fitness)):
        raise ValueError("Objective returned NaN or infinite fitness")

    best_history = np.empty(config.max_generations + 1, dtype=float)
    diversity_history = np.empty(config.max_generations + 1, dtype=float)
    best_history[0] = float(np.min(fitness))
    diversity_history[0] = population_diversity(population)

    local_boundary_count = 0

    for generation in range(1, config.max_generations + 1):
        next_population = population.copy()
        next_fitness = fitness.copy()

        for target_index in range(npop):
            candidates = np.delete(np.arange(npop), target_index)
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
            mutant = (
                population[r1]
                + config.differential_weight
                * (population[r2] - population[r3])
            )

            out_of_bounds = (mutant < lower) | (mutant > upper)
            violation_count = int(np.count_nonzero(out_of_bounds))
            local_boundary_count += violation_count
            if violation_count and boundary_callback is not None:
                boundary_callback(
                    "mutation",
                    generation,
                    mutant.copy(),
                    lower,
                    upper,
                )

            # Deterministic repair. Swap this for reflection/resampling if that
            # is what the custom implementation uses.
            mutant = np.clip(mutant, lower, upper)

            crossover_mask = rng.random(ndim) < config.crossover_rate
            crossover_mask[rng.integers(0, ndim)] = True  # j_rand guarantee
            trial = np.where(crossover_mask, mutant, population[target_index])

            if np.any((trial < lower) | (trial > upper)):
                if boundary_callback is not None:
                    boundary_callback(
                        "crossover",
                        generation,
                        trial.copy(),
                        lower,
                        upper,
                    )
                raise RuntimeError("Boundary repair failed: trial is out of bounds")

            trial_fitness = objective(trial)
            if not math.isfinite(trial_fitness):
                raise ValueError("Objective returned NaN or infinite fitness")

            if trial_fitness <= fitness[target_index]:
                next_population[target_index] = trial
                next_fitness[target_index] = trial_fitness

        population = next_population
        fitness = next_fitness
        best_history[generation] = float(np.min(fitness))
        diversity_history[generation] = population_diversity(population)

        if generation_callback is not None:
            generation_callback(
                generation,
                best_history[generation],
                diversity_history[generation],
                population.copy(),
            )

    best_index = int(np.argmin(fitness))
    result = DEResult(
        best_vector=population[best_index].copy(),
        best_fitness=float(fitness[best_index]),
        final_population=population.copy(),
        final_fitness=fitness.copy(),
        best_history=best_history,
        diversity_history=diversity_history,
        boundary_violations=local_boundary_count,
    )
    result.validate(config)
    return result


def load_optimizer(specification: str | None) -> DECallable:
    """Load ``module:function`` or return the bundled reference DE."""
    if not specification:
        return reference_de
    if ":" not in specification:
        raise ValueError("--optimizer must use module:function syntax")
    module_name, function_name = specification.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    optimizer = getattr(module, function_name)
    if not callable(optimizer):
        raise TypeError(f"{specification!r} is not callable")
    return optimizer


# ---------------------------------------------------------------------------
# Experiment execution and statistical summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialRecord:
    function: str
    trial: int
    seed: int
    final_fitness: float
    success: bool
    runtime_seconds: float
    boundary_violations: int


def make_seed(base_seed: int, function_index: int, trial: int) -> int:
    """Generate stable, non-overlapping seeds without relying on hash()."""
    return int(base_seed + function_index * 100_000 + trial)


def run_experiments(
    optimizer: DECallable,
    config: DEConfig,
    trials: int,
    success_threshold: float,
    base_seed: int,
    boundary_monitor: BoundaryMonitor,
) -> tuple[list[TrialRecord], dict[str, Array]]:
    """Run all benchmark trials and return raw records and final values."""
    records: list[TrialRecord] = []
    values_by_function: dict[str, list[float]] = {
        benchmark.name: [] for benchmark in BENCHMARKS
    }

    for function_index, benchmark in enumerate(BENCHMARKS):
        print(f"\nRunning {benchmark.name}: {trials} independent trials")
        for trial in range(trials):
            seed = make_seed(base_seed, function_index, trial)
            started = time.perf_counter()
            result = optimizer(
                benchmark.objective,
                config,
                seed,
                generation_callback=None,
                boundary_callback=boundary_monitor,
            )
            result.validate(config)
            assert_population_in_bounds(result.final_population, config)
            elapsed = time.perf_counter() - started
            value = float(result.best_fitness)
            values_by_function[benchmark.name].append(value)
            records.append(
                TrialRecord(
                    function=benchmark.name,
                    trial=trial + 1,
                    seed=seed,
                    final_fitness=value,
                    success=value < success_threshold,
                    runtime_seconds=elapsed,
                    boundary_violations=int(result.boundary_violations),
                )
            )
            print(
                f"  Trial {trial + 1:>2}/{trials}: "
                f"fitness={value:.6e}, time={elapsed:.2f}s",
                flush=True,
            )

    return records, {
        name: np.asarray(values, dtype=float)
        for name, values in values_by_function.items()
    }


def calculate_statistics(
    values_by_function: dict[str, Array],
    success_threshold: float,
) -> list[dict[str, float | str]]:
    """Calculate the requested descriptive statistics."""
    statistics: list[dict[str, float | str]] = []
    optima = {benchmark.name: benchmark.known_optimum for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        values = values_by_function[benchmark.name]
        statistics.append(
            {
                "function": benchmark.name,
                "known_optimum": optima[benchmark.name],
                "mean_final_fitness": float(np.mean(values)),
                "median_final_fitness": float(np.median(values)),
                "standard_deviation": float(np.std(values, ddof=1))
                if values.size > 1
                else 0.0,
                "success_rate_percent": float(
                    100.0 * np.mean(values < success_threshold)
                ),
            }
        )
    return statistics


def markdown_statistics_table(statistics: Sequence[dict[str, float | str]]) -> str:
    """Return a clean text-based Markdown table."""
    headers = (
        "Function",
        "Known Optimum",
        "Mean Final Fitness",
        "Median Final Fitness",
        "Standard Deviation",
        "Success Rate",
    )
    rows = [headers, tuple("---" for _ in headers)]
    for item in statistics:
        rows.append(
            (
                str(item["function"]),
                f'{float(item["known_optimum"]):.1f}',
                f'{float(item["mean_final_fitness"]):.6e}',
                f'{float(item["median_final_fitness"]):.6e}',
                f'{float(item["standard_deviation"]):.6e}',
                f'{float(item["success_rate_percent"]):.1f}%',
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]

    def format_row(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    return "\n".join(format_row(row) for row in rows)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def use_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
        }
    )


def plot_convergence(result: DEResult, output_path: Path) -> None:
    """Save dual-axis best-fitness and diversity convergence plot."""
    generations = np.arange(result.best_history.size)
    safe_fitness = np.maximum(result.best_history, np.finfo(float).tiny)

    figure, left_axis = plt.subplots(figsize=(8.2, 4.8))
    right_axis = left_axis.twinx()
    fitness_line = left_axis.plot(
        generations,
        safe_fitness,
        color="#1261A0",
        linewidth=2.0,
        label="Best fitness",
    )
    diversity_line = right_axis.plot(
        generations,
        result.diversity_history,
        color="#D1495B",
        linewidth=1.8,
        label="Population diversity",
    )
    left_axis.set_yscale("log")
    left_axis.set_xlabel("Generation")
    left_axis.set_ylabel("Best fitness (log scale)", color="#1261A0")
    right_axis.set_ylabel("Mean coordinate standard deviation", color="#D1495B")
    left_axis.tick_params(axis="y", colors="#1261A0")
    right_axis.tick_params(axis="y", colors="#D1495B")
    left_axis.set_title("Rastrigin convergence and population diversity")
    lines = fitness_line + diversity_line
    left_axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def plot_fitness_boxplots(
    values_by_function: dict[str, Array],
    output_path: Path,
) -> None:
    """Save three side-by-side final-fitness box plots."""
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 4.4))
    colors = ("#4C78A8", "#F58518", "#54A24B")
    for axis, benchmark, color in zip(axes, BENCHMARKS, colors):
        plot = axis.boxplot(
            values_by_function[benchmark.name],
            patch_artist=True,
            widths=0.55,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"facecolor": color, "alpha": 0.75},
        )
        _ = plot
        axis.set_title(benchmark.name)
        axis.set_xticks([1], ["Final fitness"])
        axis.set_ylabel("Fitness")
    figure.suptitle("Distribution of final fitness across independent trials")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def generate_heatmap(
    optimizer: DECallable,
    base_config: DEConfig,
    cr_values: Sequence[float],
    f_values: Sequence[float],
    trials_per_pair: int,
    base_seed: int,
    boundary_monitor: BoundaryMonitor,
    output_path: Path,
) -> tuple[Array, list[dict[str, float | int]]]:
    """Run Rastrigin sensitivity analysis and save an annotated heatmap."""
    matrix = np.empty((len(cr_values), len(f_values)), dtype=float)
    raw_rows: list[dict[str, float | int]] = []

    print("\nRunning Rastrigin F/CR sensitivity analysis")
    for cr_index, cr in enumerate(cr_values):
        for f_index, differential_weight in enumerate(f_values):
            values: list[float] = []
            for trial in range(trials_per_pair):
                seed = int(
                    base_seed
                    + 900_000
                    + cr_index * 10_000
                    + f_index * 1_000
                    + trial
                )
                config = DEConfig(
                    **{
                        **asdict(base_config),
                        "differential_weight": differential_weight,
                        "crossover_rate": cr,
                    }
                )
                result = optimizer(
                    rastrigin,
                    config,
                    seed,
                    generation_callback=None,
                    boundary_callback=boundary_monitor,
                )
                result.validate(config)
                assert_population_in_bounds(result.final_population, config)
                values.append(float(result.best_fitness))
                raw_rows.append(
                    {
                        "crossover_rate": cr,
                        "differential_weight": differential_weight,
                        "trial": trial + 1,
                        "seed": seed,
                        "final_fitness": float(result.best_fitness),
                    }
                )
            matrix[cr_index, f_index] = float(np.mean(values))
            print(
                f"  CR={cr:.1f}, F={differential_weight:.1f}: "
                f"mean={matrix[cr_index, f_index]:.6e}"
            )

    figure, axis = plt.subplots(figsize=(6.8, 5.1))
    image = axis.imshow(matrix, cmap="viridis_r", aspect="auto")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Mean final fitness (lower is better)")
    axis.set_xticks(np.arange(len(f_values)), [f"{value:.1f}" for value in f_values])
    axis.set_yticks(np.arange(len(cr_values)), [f"{value:.1f}" for value in cr_values])
    axis.set_xlabel("Differential weight F")
    axis.set_ylabel("Crossover rate CR")
    axis.set_title("Rastrigin parameter sensitivity")

    threshold = float(np.mean([matrix.min(), matrix.max()]))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2e}",
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=9,
            )

    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return matrix, raw_rows


# ---------------------------------------------------------------------------
# Integrity checks and persistence
# ---------------------------------------------------------------------------


def assert_population_in_bounds(population: Array, config: DEConfig) -> None:
    tolerance = 1e-12
    if np.any(population < config.lower_bound - tolerance) or np.any(
        population > config.upper_bound + tolerance
    ):
        raise AssertionError("FAIL: final population exceeds configured bounds")


def reproducibility_check(
    optimizer: DECallable,
    config: DEConfig,
    seed: int = 42,
) -> bool:
    """Run two identical Rastrigin trials and compare to ten decimal places."""
    first = optimizer(rastrigin, config, seed)
    second = optimizer(rastrigin, config, seed)
    first.validate(config)
    second.validate(config)
    fitness_match = round(first.best_fitness, 10) == round(second.best_fitness, 10)
    history_match = np.array_equal(first.best_history, second.best_history)
    vector_match = np.array_equal(first.best_vector, second.best_vector)
    passed = fitness_match and history_match and vector_match
    print(
        "\nREPRODUCIBILITY CHECK: "
        + ("PASS" if passed else "FAIL")
        + f" | seed={seed} | "
        + f"fitness_1={first.best_fitness:.10f} | "
        + f"fitness_2={second.best_fitness:.10f}"
    )
    return passed


def save_trial_records(records: Sequence[TrialRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def save_sensitivity_records(
    records: Sequence[dict[str, float | int]],
    path: Path,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation suite for Differential Evolution."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("de_evaluation_outputs"))
    parser.add_argument("--optimizer", help="Custom optimiser as module:function")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--dimensions", type=int, default=30)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=50)
    parser.add_argument("--f", type=float, default=0.8)
    parser.add_argument("--cr", type=float, default=0.9)
    parser.add_argument("--lower-bound", type=float, default=-5.12)
    parser.add_argument("--upper-bound", type=float, default=5.12)
    parser.add_argument("--success-threshold", type=float, default=1e-5)
    parser.add_argument("--seed-base", type=int, default=12_345)
    parser.add_argument("--representative-seed", type=int, default=42)
    parser.add_argument("--sensitivity-trials", type=int, default=5)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: 2 trials, 20 generations, and 1 sensitivity trial.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.quick:
        args.trials = 2
        args.generations = 20
        args.sensitivity_trials = 1
    if args.trials < 1 or args.sensitivity_trials < 1:
        raise SystemExit("Trial counts must be at least 1")

    use_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = load_optimizer(args.optimizer)
    config = DEConfig(
        dimensions=args.dimensions,
        lower_bound=args.lower_bound,
        upper_bound=args.upper_bound,
        max_generations=args.generations,
        population_size=args.population_size,
        differential_weight=args.f,
        crossover_rate=args.cr,
    )
    config.validate()
    boundary_monitor = BoundaryMonitor(warn=True)

    print("Differential Evolution Evaluation Suite")
    print(json.dumps(asdict(config), indent=2))

    if not reproducibility_check(optimizer, config, seed=args.representative_seed):
        print("ERROR: reproducibility check failed.", file=sys.stderr)
        return 2

    records, values_by_function = run_experiments(
        optimizer=optimizer,
        config=config,
        trials=args.trials,
        success_threshold=args.success_threshold,
        base_seed=args.seed_base,
        boundary_monitor=boundary_monitor,
    )
    statistics = calculate_statistics(values_by_function, args.success_threshold)
    table = markdown_statistics_table(statistics)
    print("\nSTATISTICAL PERFORMANCE\n")
    print(table)
    (args.output_dir / "statistical_performance.md").write_text(
        table + "\n",
        encoding="utf-8",
    )
    save_trial_records(records, args.output_dir / "benchmark_trials.csv")

    representative = optimizer(
        rastrigin,
        config,
        args.representative_seed,
        generation_callback=None,
        boundary_callback=boundary_monitor,
    )
    representative.validate(config)
    plot_convergence(
        representative,
        args.output_dir / "rastrigin_convergence_diversity.png",
    )
    plot_fitness_boxplots(
        values_by_function,
        args.output_dir / "final_fitness_boxplots.png",
    )

    _, sensitivity_records = generate_heatmap(
        optimizer=optimizer,
        base_config=config,
        cr_values=(0.2, 0.5, 0.9),
        f_values=(0.4, 0.7, 1.0),
        trials_per_pair=args.sensitivity_trials,
        base_seed=args.seed_base,
        boundary_monitor=boundary_monitor,
        output_path=args.output_dir / "rastrigin_parameter_heatmap.png",
    )
    save_sensitivity_records(
        sensitivity_records,
        args.output_dir / "rastrigin_sensitivity_trials.csv",
    )

    metadata = {
        "created_unix_time": time.time(),
        "optimizer": args.optimizer or "reference_de",
        "config": asdict(config),
        "benchmark_trials_per_function": args.trials,
        "sensitivity_trials_per_pair": args.sensitivity_trials,
        "success_threshold": args.success_threshold,
        "statistics": statistics,
        "boundary_monitor": {
            "coordinate_violations_before_repair": boundary_monitor.count,
            "maximum_overshoot": boundary_monitor.max_overshoot,
        },
        "representative_rastrigin": {
            "seed": args.representative_seed,
            "best_fitness": representative.best_fitness,
        },
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if boundary_monitor.count:
        print(
            "\nBOUNDARY CHECK: WARNING | "
            f"{boundary_monitor.count} coordinates exceeded bounds before repair; "
            f"maximum overshoot={boundary_monitor.max_overshoot:.6e}. "
            "All accepted trial vectors and final populations remained in bounds."
        )
    else:
        print("\nBOUNDARY CHECK: PASS | No boundary excursions detected.")

    print(f"\nOutputs saved to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
