"""Adapter between ``de_evaluation_suite.py`` and the project's SciPy DE stage.

Place this file beside ``de_evaluation_suite.py``:

    scripts/
    └── evaluation/
        ├── de_evaluation_suite.py
        └── de_inner_optimizer_adapter.py

Run from the repository root:

    python scripts/evaluation/de_evaluation_suite.py \
        --optimizer de_inner_optimizer_adapter:run_custom_de \
        --output-dir outputs/de_benchmark_custom

Why polishing is disabled
-------------------------
The Laban inner optimiser uses Differential Evolution for global search and
L-BFGS-B as a subsequent local-polishing stage. This adapter intentionally
tests only the DE stage, ensuring the benchmark measures DE rather than the
combined hybrid optimiser.

Boundary-monitoring limitation
------------------------------
SciPy repairs out-of-bounds trial coordinates internally before exposing each
generation to its public callback. Consequently, this adapter can verify every
exposed population but cannot count private, pre-repair mutant excursions.
Such counting would require modifying SciPy's DE core. A detected out-of-bounds
exposed population is still reported immediately through ``boundary_callback``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import differential_evolution

# These imports resolve because this adapter is stored beside the evaluation
# script. They provide the result contract expected by the harness.
from de_evaluation_suite import (  # type: ignore
    DEConfig,
    DEResult,
    population_diversity,
)


Objective = Callable[[np.ndarray], float]
GenerationCallback = Callable[
    [int, float, float, np.ndarray],
    None,
]
BoundaryCallback = Callable[
    [str, int, np.ndarray, float, float],
    None,
]


def _evaluate_population(
    objective: Objective,
    population: np.ndarray,
) -> np.ndarray:
    """Evaluate one population and reject non-finite objective values."""
    fitness = np.asarray(
        [float(objective(vector)) for vector in population],
        dtype=float,
    )
    if not np.all(np.isfinite(fitness)):
        raise ValueError("Objective returned NaN or infinite fitness")
    return fitness


def _report_exposed_boundary_violation(
    population: np.ndarray,
    generation: int,
    config: DEConfig,
    boundary_callback: BoundaryCallback | None,
) -> int:
    """Validate a population exposed by SciPy and report any violation."""
    mask = (
        (population < config.lower_bound)
        | (population > config.upper_bound)
    )
    count = int(np.count_nonzero(mask))
    if count and boundary_callback is not None:
        boundary_callback(
            "scipy_exposed_population",
            generation,
            population.copy(),
            config.lower_bound,
            config.upper_bound,
        )
    return count


def run_custom_de(
    objective: Objective,
    config: DEConfig,
    seed: int,
    generation_callback: GenerationCallback | None = None,
    boundary_callback: BoundaryCallback | None = None,
) -> DEResult:
    """Run the same SciPy DE engine used by the inner trajectory optimiser.

    The adapter maps the evaluation suite's conventional DE parameters as:

    * ``NP`` -> an explicit initial population of exactly ``NP`` vectors;
    * ``F`` -> SciPy's ``mutation`` argument;
    * ``CR`` -> SciPy's ``recombination`` argument;
    * generations -> SciPy's ``maxiter`` argument.

    ``best1bin`` and immediate updating match SciPy's conventional defaults.
    If the project's inner optimiser explicitly uses another strategy or
    updating mode, change ``strategy`` or ``updating`` below accordingly.
    """
    config.validate()

    rng = np.random.default_rng(seed)
    bounds = [
        (config.lower_bound, config.upper_bound)
        for _ in range(config.dimensions)
    ]

    # Providing an explicit initial population is important. SciPy normally
    # interprets ``popsize`` as a dimension multiplier, whereas the benchmark
    # specification defines NP as the total number of individuals.
    initial_population = rng.uniform(
        config.lower_bound,
        config.upper_bound,
        size=(config.population_size, config.dimensions),
    )
    initial_fitness = _evaluate_population(objective, initial_population)

    best_history: list[float] = [float(np.min(initial_fitness))]
    diversity_history: list[float] = [
        population_diversity(initial_population)
    ]
    boundary_violations = _report_exposed_boundary_violation(
        initial_population,
        generation=0,
        config=config,
        boundary_callback=boundary_callback,
    )

    def scipy_callback(intermediate_result: Any) -> bool:
        """Capture SciPy's population after each completed generation."""
        nonlocal boundary_violations

        generation = len(best_history)
        population = np.asarray(
            intermediate_result.population,
            dtype=float,
        ).copy()
        population_fitness = np.asarray(
            intermediate_result.population_energies,
            dtype=float,
        )

        boundary_violations += _report_exposed_boundary_violation(
            population,
            generation=generation,
            config=config,
            boundary_callback=boundary_callback,
        )

        best_fitness = float(np.min(population_fitness))
        diversity = population_diversity(population)
        best_history.append(best_fitness)
        diversity_history.append(diversity)

        if generation_callback is not None:
            generation_callback(
                generation,
                best_fitness,
                diversity,
                population.copy(),
            )

        return False

    scipy_result = differential_evolution(
        func=objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=config.max_generations,
        mutation=config.differential_weight,
        recombination=config.crossover_rate,
        init=initial_population,
        seed=rng,
        callback=scipy_callback,
        polish=False,
        updating="immediate",
        workers=1,
        tol=0.0,
        atol=0.0,
        disp=False,
    )

    final_population = np.asarray(
        scipy_result.population,
        dtype=float,
    ).copy()
    final_fitness = np.asarray(
        scipy_result.population_energies,
        dtype=float,
    ).copy()

    boundary_violations += _report_exposed_boundary_violation(
        final_population,
        generation=int(scipy_result.nit),
        config=config,
        boundary_callback=boundary_callback,
    )

    # The harness expects generation 0 through max_generations. SciPy should
    # complete every generation because tol=atol=0, but padding makes the
    # adapter robust if the population becomes exactly identical.
    expected_length = config.max_generations + 1
    if len(best_history) < expected_length:
        missing = expected_length - len(best_history)
        best_history.extend([best_history[-1]] * missing)
        diversity_history.extend([diversity_history[-1]] * missing)
    elif len(best_history) > expected_length:
        best_history = best_history[:expected_length]
        diversity_history = diversity_history[:expected_length]

    best_index = int(np.argmin(final_fitness))
    result = DEResult(
        best_vector=final_population[best_index].copy(),
        best_fitness=float(final_fitness[best_index]),
        final_population=final_population,
        final_fitness=final_fitness,
        best_history=np.asarray(best_history, dtype=float),
        diversity_history=np.asarray(diversity_history, dtype=float),
        boundary_violations=boundary_violations,
    )
    result.validate(config)
    return result

