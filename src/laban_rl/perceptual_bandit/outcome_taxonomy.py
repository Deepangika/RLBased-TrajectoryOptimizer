"""Corrected, versioned outcome taxonomy for perceptual experiments.

Version 2 separates three concepts that the legacy (version 1) labels in
``classify_outcome`` conflated:

1. **Execution success** — the run completed correctly, respected
   invariants, and produced valid outputs.
2. **Selection success** — at least one independently validated feasible
   candidate was selected.
3. **Perceptual improvement** — the selected motion improved according to a
   predefined perceptual criterion.

``all_successful`` in stage summaries is an execution/selection statement and
must never be quoted as evidence of perceptual improvement.

Conservative paired-preference rule (descriptive, for small repeat counts;
fractions generalise the user-specified five-repeat rule):

- ``strong_preference_improvement``: learned wins >= 80% of repeats (4/5);
- ``no_preference_improvement``: learned wins <= 20% of repeats (0/5, 1/5);
- ``weak_or_inconclusive_preference``: anything in between, including ties
  and cases where "neither" responses prevent a clear majority;
- ``not_evaluated``: paired results unavailable.

Raw counts must always be recorded alongside the label.
"""

from __future__ import annotations

from typing import Any, Mapping

TAXONOMY_VERSION = 2

PAIRED_STRONG_FRACTION = 0.8
PAIRED_NONE_FRACTION = 0.2


def classify_paired_preference(
    learned_wins: int | None,
    opponent_wins: int | None,
    neither_count: int | None,
    repeats: int | None,
) -> str:
    """Classify a paired-preference result with the conservative v2 rule."""
    if repeats is None or learned_wins is None:
        return "not_evaluated"
    repeats = int(repeats)
    if repeats <= 0:
        return "not_evaluated"
    learned = int(learned_wins)
    opponent = int(opponent_wins or 0)
    neither = int(neither_count or 0)
    if learned + opponent + neither != repeats:
        raise ValueError(
            f"Paired counts do not sum to repeats: "
            f"{learned}+{opponent}+{neither} != {repeats}"
        )
    fraction = learned / repeats
    if fraction >= PAIRED_STRONG_FRACTION:
        return "strong_preference_improvement"
    if fraction <= PAIRED_NONE_FRACTION:
        return "no_preference_improvement"
    return "weak_or_inconclusive_preference"


def _paired_counts(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not record or not record.get("records"):
        return {
            "available": False,
            "learned_wins": None,
            "opponent_wins": None,
            "neither": None,
            "repeats": None,
        }
    counts = dict(record["records"][0]["choice_counts"])
    learned = int(counts.get("styled", 0))
    opponent = int(counts.get("reference", 0))
    neither = int(counts.get("neither", 0))
    return {
        "available": True,
        "learned_wins": learned,
        "opponent_wins": opponent,
        "neither": neither,
        "repeats": learned + opponent + neither,
    }


def assess_perceptual_outcome(
    *,
    execution_success: bool,
    selection_success: bool,
    selected_validation_reward: float | None,
    baseline_validation_reward: float | None,
    paired_vs_baseline: Mapping[str, Any] | None,
    paired_vs_reference: Mapping[str, Any] | None,
    comparison_status: str | None = None,
) -> dict[str, Any]:
    """Produce the corrected v2 assessment fields.

    ``paired_vs_baseline`` / ``paired_vs_reference`` are the saved paired
    result payloads (with ``records[0].choice_counts``) or ``None``.
    """
    comparable = comparison_status in (None, "comparable")
    absolute_delta = (
        float(selected_validation_reward) - float(baseline_validation_reward)
        if comparable
        and selected_validation_reward is not None
        and baseline_validation_reward is not None
        else None
    )
    absolute_improvement = (
        bool(absolute_delta > 0.0) if absolute_delta is not None else None
    )

    base_counts = _paired_counts(paired_vs_baseline)
    ref_counts = _paired_counts(paired_vs_reference)
    paired_baseline_label = classify_paired_preference(
        base_counts["learned_wins"],
        base_counts["opponent_wins"],
        base_counts["neither"],
        base_counts["repeats"],
    )
    paired_reference_label = classify_paired_preference(
        ref_counts["learned_wins"],
        ref_counts["opponent_wins"],
        ref_counts["neither"],
        ref_counts["repeats"],
    )

    absolute_known = absolute_improvement is not None
    paired_known = paired_baseline_label != "not_evaluated"
    strong_paired = paired_baseline_label == "strong_preference_improvement"

    if not selection_success:
        perceptual_improvement: bool | None = False
        perceptual_outcome = "no_valid_selection"
    elif not absolute_known and not paired_known:
        perceptual_improvement = None
        perceptual_outcome = "not_evaluated"
    elif bool(absolute_improvement) and strong_paired:
        perceptual_improvement = True
        perceptual_outcome = "absolute_and_paired_improvement"
    elif strong_paired:
        perceptual_improvement = True
        perceptual_outcome = "paired_only_improvement"
    elif bool(absolute_improvement):
        perceptual_improvement = True
        perceptual_outcome = "absolute_only_improvement"
    else:
        perceptual_improvement = False
        perceptual_outcome = "no_perceptual_improvement"

    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "execution_success": bool(execution_success),
        "selection_success": bool(selection_success),
        "absolute_reward_delta_vs_baseline": absolute_delta,
        "absolute_reward_improvement": absolute_improvement,
        "paired_improvement_vs_baseline": paired_baseline_label,
        "paired_vs_baseline_counts": base_counts,
        "paired_improvement_vs_reference": paired_reference_label,
        "paired_vs_reference_counts": ref_counts,
        "perceptual_improvement": perceptual_improvement,
        "perceptual_outcome": perceptual_outcome,
        "note": (
            "execution_success and selection_success describe pipeline "
            "behaviour only; perceptual_improvement is the only field that "
            "may be quoted as evidence of (machine-evaluator) perceptual "
            "gain. Raw paired counts are authoritative over labels."
        ),
    }
