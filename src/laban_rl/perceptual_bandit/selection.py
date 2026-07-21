"""Feasibility-first selection for independently validated CEM candidates."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


def strict_realisability(
    result: Mapping[str, Any],
    *,
    tolerance: float,
) -> tuple[bool, list[str]]:
    """Return whether a validation result passes every deployment constraint."""
    reasons: list[str] = []
    if not bool(result.get("valid_realisation", False)):
        reasons.append("invalid_realisation")
    if not bool(result.get("physically_acceptable", False)):
        reasons.append("physical_acceptance_failed")
    try:
        rmse = float(result["realisation_rmse"])
    except (KeyError, TypeError, ValueError):
        rmse = float("inf")
    try:
        max_error = float(result["max_abs_feature_error"])
    except (KeyError, TypeError, ValueError):
        max_error = float("inf")
    if rmse > tolerance:
        reasons.append("rmse_above_tolerance")
    if max_error > tolerance:
        reasons.append("max_feature_error_above_tolerance")
    return not reasons, reasons


def select_feasible_incumbent(
    validation_results: Mapping[str, Any],
    shortlist: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Select the best feasible result from initial, final-mean, and top-K pool.

    If no candidate is strictly feasible, the highest-reward candidate is retained
    only as a diagnostic fallback and ``strict_feasibility_satisfied`` is false.
    """
    pool: list[dict[str, Any]] = []

    def add_candidate(
        *, result_key: str, source: str, source_type: str,
        profile: Mapping[str, Any] | None = None,
        rank: int | None = None,
        training_reward: float | None = None,
    ) -> None:
        result = validation_results.get(result_key)
        if not isinstance(result, Mapping):
            return
        eligible, reasons = strict_realisability(result, tolerance=tolerance)
        candidate = {
            "source": source,
            "source_type": source_type,
            "profile": dict(profile or result.get("requested_profile") or {}),
            "validation_reward": float(result["outer_reward"]),
            "validation_result_key": result_key,
            "strictly_realisable": eligible,
            "ineligibility_reasons": reasons,
        }
        if rank is not None:
            candidate["rank"] = int(rank)
        if training_reward is not None:
            candidate["training_reward"] = float(training_reward)
        pool.append(candidate)

    add_candidate(
        result_key="initial_profile",
        source="informed_initial_profile",
        source_type="incumbent_initial",
    )
    add_candidate(
        result_key="final_distribution_mean",
        source="final_distribution_mean",
        source_type="cem_distribution_mean",
    )
    for item in shortlist:
        add_candidate(
            result_key=str(item["validation_result_key"]),
            source=str(item["source"]),
            source_type="cem_shortlist",
            profile=item.get("profile"),
            rank=int(item["rank"]),
            training_reward=item.get("training_reward"),
        )

    if not pool:
        raise RuntimeError("No independently validated candidates were available.")
    eligible = [candidate for candidate in pool if candidate["strictly_realisable"]]
    selection_pool = eligible or pool
    selected = max(selection_pool, key=lambda item: item["validation_reward"])
    return {
        "method": "feasibility_first_incumbent_safe_reranking",
        "tolerance": float(tolerance),
        "strict_feasibility_satisfied": bool(eligible),
        "selected": deepcopy(selected),
        "candidate_pool": deepcopy(pool),
        "shortlist": [deepcopy(dict(item)) for item in shortlist],
    }
