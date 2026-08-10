"""Reference-relative perceptual evaluation metrics and diagnostics.

This module quantifies how an optimised candidate changes perceived affect
relative to the unmodified reference gesture, as a complementary diagnostic
to the existing absolute candidate-to-target evaluation.

Scientific interpretation (must accompany generated analyses):

    Reference subtraction does not change the mathematical candidate-to-target
    distance when the same reference estimate is used on both sides. Its purpose
    is diagnostic: it reveals the direction and magnitude of the perceptual change
    caused by modifying the original gesture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

VAD_AXES = ("valence", "arousal", "dominance")

REFERENCE_RELATIVE_INTERPRETATION = (
    "Reference subtraction does not change the mathematical candidate-to-target "
    "distance when the same reference estimate is used on both sides. Its purpose "
    "is diagnostic: it reveals the direction and magnitude of the perceptual change "
    "caused by modifying the original gesture."
)


@dataclass(frozen=True)
class ReferenceRelativeThresholds:
    """Centralised thresholds for the reference-relative diagnostic outcome."""

    near_target_tolerance: float = 0.10
    minimum_valid_evaluations: int = 2
    meaningful_shift_sd_multiplier: float = 1.0
    minimum_directional_alignment: float = 0.0
    minimum_progress: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "near_target_tolerance": float(self.near_target_tolerance),
            "minimum_valid_evaluations": int(self.minimum_valid_evaluations),
            "meaningful_shift_sd_multiplier": float(
                self.meaningful_shift_sd_multiplier
            ),
            "minimum_directional_alignment": float(
                self.minimum_directional_alignment
            ),
            "minimum_progress": float(self.minimum_progress),
        }


def _vad_vector(vad: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(vad[axis]) for axis in VAD_AXES], dtype=float)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def summarise_vad_observations(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Mean/sd VAD and validity accounting for repeated evaluator observations.

    An observation is valid when its ``affect_ratings`` contain a finite value
    for every VAD axis.
    """
    valid_rows: list[np.ndarray] = []
    invalid = 0
    for observation in observations:
        ratings = observation.get("affect_ratings") or {}
        try:
            vector = _vad_vector(ratings)
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if not np.all(np.isfinite(vector)):
            invalid += 1
            continue
        valid_rows.append(vector)
    if valid_rows:
        stacked = np.vstack(valid_rows)
        mean = {
            axis: float(stacked[:, index].mean())
            for index, axis in enumerate(VAD_AXES)
        }
        sd = {
            axis: float(stacked[:, index].std(ddof=1)) if len(valid_rows) > 1 else 0.0
            for index, axis in enumerate(VAD_AXES)
        }
    else:
        mean = None
        sd = None
    return {
        "vad_mean": mean,
        "vad_sd": sd,
        "valid_count": len(valid_rows),
        "invalid_count": invalid,
    }


def compute_reference_relative_metrics(
    *,
    reference_vad: Mapping[str, float],
    candidate_vad: Mapping[str, float],
    target_vad: Mapping[str, float],
    reference_sd: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """All reference-relative metrics with safe null handling (no NaN/inf)."""
    v_ref = _vad_vector(reference_vad)
    v_cand = _vad_vector(candidate_vad)
    v_target = _vad_vector(target_vad)

    d_candidate_target = float(np.linalg.norm(v_cand - v_target))
    d_reference_target = float(np.linalg.norm(v_ref - v_target))
    delta_observed = v_cand - v_ref
    delta_required = v_target - v_ref
    shift_magnitude = float(np.linalg.norm(delta_observed))
    required_magnitude = float(np.linalg.norm(delta_required))

    alignment: float | None
    alignment_null_reason = None
    if shift_magnitude == 0.0 and required_magnitude == 0.0:
        alignment = None
        alignment_null_reason = "both_shift_vectors_zero_length"
    elif shift_magnitude == 0.0:
        alignment = None
        alignment_null_reason = "observed_shift_zero_length"
    elif required_magnitude == 0.0:
        alignment = None
        alignment_null_reason = "required_shift_zero_length_reference_on_target"
    else:
        alignment = _finite_or_none(
            float(np.dot(delta_observed, delta_required))
            / (shift_magnitude * required_magnitude)
        )

    progress: float | None
    progress_null_reason = None
    if d_reference_target == 0.0:
        progress = None
        progress_null_reason = "reference_target_distance_zero"
    else:
        progress = _finite_or_none(
            (d_reference_target - d_candidate_target) / d_reference_target
        )

    dimension_error_reduction = {}
    for index, axis in enumerate(VAD_AXES):
        error_reference = abs(float(v_ref[index]) - float(v_target[index]))
        error_candidate = abs(float(v_cand[index]) - float(v_target[index]))
        dimension_error_reduction[axis] = {
            "absolute_error_reference": error_reference,
            "absolute_error_candidate": error_candidate,
            "error_reduction": error_reference - error_candidate,
        }

    standardised_shift: dict[str, Any] = {}
    for index, axis in enumerate(VAD_AXES):
        sd_value = (
            float(reference_sd[axis])
            if reference_sd is not None and axis in reference_sd
            and reference_sd[axis] is not None
            else None
        )
        if sd_value is None:
            standardised_shift[axis] = {
                "value": None,
                "null_reason": "reference_sd_unavailable",
            }
        elif not math.isfinite(sd_value) or sd_value <= 0.0:
            standardised_shift[axis] = {
                "value": None,
                "null_reason": "reference_sd_zero_or_invalid",
            }
        else:
            standardised_shift[axis] = {
                "value": _finite_or_none(float(delta_observed[index]) / sd_value),
                "null_reason": None,
            }

    return {
        "reference_vad": {axis: float(v_ref[i]) for i, axis in enumerate(VAD_AXES)},
        "candidate_vad": {axis: float(v_cand[i]) for i, axis in enumerate(VAD_AXES)},
        "target_vad": {axis: float(v_target[i]) for i, axis in enumerate(VAD_AXES)},
        "candidate_target_distance": d_candidate_target,
        "reference_target_distance": d_reference_target,
        "observed_shift": {
            axis: float(delta_observed[i]) for i, axis in enumerate(VAD_AXES)
        },
        "required_shift": {
            axis: float(delta_required[i]) for i, axis in enumerate(VAD_AXES)
        },
        "shift_magnitude": shift_magnitude,
        "directional_alignment": alignment,
        "directional_alignment_null_reason": alignment_null_reason,
        "normalised_progress": progress,
        "normalised_progress_null_reason": progress_null_reason,
        "dimension_error_reduction": dimension_error_reduction,
        "standardised_shift": standardised_shift,
        "interpretation": REFERENCE_RELATIVE_INTERPRETATION,
    }


def bootstrap_shift_interval(
    *,
    reference_observations: Sequence[Mapping[str, Any]],
    candidate_observations: Sequence[Mapping[str, Any]],
    seed: int,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Percentile bootstrap interval for the candidate-minus-reference VAD shift.

    Uses only numpy; returns nulls with a reason when either side has fewer
    than two valid observations.
    """
    def _rows(observations: Sequence[Mapping[str, Any]]) -> np.ndarray | None:
        rows = []
        for observation in observations:
            ratings = observation.get("affect_ratings") or {}
            try:
                vector = _vad_vector(ratings)
            except (KeyError, TypeError, ValueError):
                continue
            if np.all(np.isfinite(vector)):
                rows.append(vector)
        return np.vstack(rows) if len(rows) >= 2 else None

    reference_rows = _rows(reference_observations)
    candidate_rows = _rows(candidate_observations)
    if reference_rows is None or candidate_rows is None:
        return {
            "status": "not_computed",
            "null_reason": "fewer_than_two_valid_observations_on_one_side",
            "intervals": None,
        }
    rng = np.random.default_rng(seed)
    samples = np.empty((n_bootstrap, len(VAD_AXES)), dtype=float)
    for index in range(n_bootstrap):
        ref_pick = reference_rows[
            rng.integers(0, len(reference_rows), size=len(reference_rows))
        ].mean(axis=0)
        cand_pick = candidate_rows[
            rng.integers(0, len(candidate_rows), size=len(candidate_rows))
        ].mean(axis=0)
        samples[index] = cand_pick - ref_pick
    lower = np.percentile(samples, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(samples, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return {
        "status": "computed",
        "null_reason": None,
        "n_bootstrap": int(n_bootstrap),
        "alpha": float(alpha),
        "seed": int(seed),
        "intervals": {
            axis: {"lower": float(lower[i]), "upper": float(upper[i])}
            for i, axis in enumerate(VAD_AXES)
        },
    }


def aggregate_paired_diagnostic(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate blinded paired reference-vs-candidate diagnostic judgments.

    In these observations the candidate is stored as ``styled`` and the
    unmodified motion as ``reference`` (project-wide naming).
    """
    counts = {"styled": 0, "reference": 0, "neither": 0}
    invalid = 0
    same_gesture_votes = 0
    same_gesture_answered = 0
    dimension_counts: dict[str, dict[str, int]] = {
        axis: {"styled": 0, "reference": 0, "similar": 0} for axis in VAD_AXES
    }
    for observation in observations:
        choice = observation.get("choice")
        if choice not in counts:
            invalid += 1
            continue
        counts[choice] += 1
        same = observation.get("same_gesture")
        if isinstance(same, bool):
            same_gesture_answered += 1
            if same:
                same_gesture_votes += 1
        for axis in VAD_AXES:
            higher = observation.get(f"{axis}_higher")
            if higher in dimension_counts[axis]:
                dimension_counts[axis][higher] += 1
    valid = sum(counts.values())
    decisive = counts["styled"] + counts["reference"]
    return {
        "candidate_preferred": counts["styled"],
        "reference_preferred": counts["reference"],
        "neither": counts["neither"],
        "valid_responses": valid,
        "invalid_responses": invalid,
        "candidate_win_rate_excluding_neither": (
            counts["styled"] / decisive if decisive else None
        ),
        "candidate_win_rate_including_neither": (
            counts["styled"] / valid if valid else None
        ),
        "gesture_preservation_agreement": (
            same_gesture_votes / same_gesture_answered
            if same_gesture_answered
            else None
        ),
        "dimension_level_preferences": dimension_counts,
    }


def classify_reference_relative_outcome(
    *,
    metrics: Mapping[str, Any] | None,
    thresholds: ReferenceRelativeThresholds,
    reference_valid_count: int,
    candidate_valid_count: int,
    reference_sd: Mapping[str, float] | None = None,
    paired_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Diagnostic classification, separate from the top-level outcome taxonomy."""
    minimum = int(thresholds.minimum_valid_evaluations)
    if (
        metrics is None
        or reference_valid_count < minimum
        or candidate_valid_count < minimum
    ):
        return {
            "label": "insufficient_valid_evaluations",
            "reason": (
                f"reference_valid={reference_valid_count}, "
                f"candidate_valid={candidate_valid_count}, "
                f"minimum_required={minimum}"
            ),
            "thresholds": thresholds.to_dict(),
        }

    d_reference_target = float(metrics["reference_target_distance"])
    if d_reference_target <= float(thresholds.near_target_tolerance):
        return {
            "label": "reference_already_near_target",
            "reason": (
                f"reference_target_distance={d_reference_target:.4f} <= "
                f"tolerance={thresholds.near_target_tolerance}"
            ),
            "thresholds": thresholds.to_dict(),
        }

    progress = metrics.get("normalised_progress")
    alignment = metrics.get("directional_alignment")
    if (progress is not None and progress < float(thresholds.minimum_progress)) or (
        alignment is not None
        and alignment < float(thresholds.minimum_directional_alignment)
    ):
        return {
            "label": "moved_away_from_target",
            "reason": f"progress={progress}, directional_alignment={alignment}",
            "thresholds": thresholds.to_dict(),
        }

    # Meaningful-shift test: shift magnitude relative to reference variability.
    sd_values = [
        float(reference_sd[axis])
        for axis in VAD_AXES
        if reference_sd is not None
        and reference_sd.get(axis) is not None
        and math.isfinite(float(reference_sd[axis]))
    ]
    noise_scale = float(np.linalg.norm(sd_values)) if sd_values else None
    shift_magnitude = float(metrics["shift_magnitude"])
    meaningful_shift = (
        shift_magnitude
        > float(thresholds.meaningful_shift_sd_multiplier) * noise_scale
        if noise_scale is not None and noise_scale > 0.0
        else shift_magnitude > 0.0
    )
    paired_decisive = bool(
        paired_summary is not None
        and paired_summary.get("candidate_win_rate_excluding_neither") is not None
        and float(paired_summary["candidate_win_rate_excluding_neither"]) > 0.5
    )
    if not meaningful_shift and not paired_decisive:
        return {
            "label": "no_meaningful_perceptual_shift",
            "reason": (
                f"shift_magnitude={shift_magnitude:.4f} within reference "
                f"variability (noise_scale={noise_scale}) and no reliable "
                "paired preference"
            ),
            "thresholds": thresholds.to_dict(),
        }

    d_candidate_target = float(metrics["candidate_target_distance"])
    if d_candidate_target <= float(thresholds.near_target_tolerance) or (
        progress is not None
        and progress > float(thresholds.minimum_progress)
        and (alignment is None or alignment > 0.0)
        and d_candidate_target < d_reference_target
    ):
        return {
            "label": "reached_or_improved_toward_target",
            "reason": (
                f"candidate_target_distance={d_candidate_target:.4f} < "
                f"reference_target_distance={d_reference_target:.4f}, "
                f"alignment={alignment}"
            ),
            "thresholds": thresholds.to_dict(),
        }

    return {
        "label": "directional_improvement_but_target_not_reached",
        "reason": (
            f"alignment={alignment}, paired_decisive={paired_decisive}, "
            f"candidate_target_distance={d_candidate_target:.4f} remains above "
            f"tolerance={thresholds.near_target_tolerance}"
        ),
        "thresholds": thresholds.to_dict(),
    }


REFERENCE_RELATIVE_CSV_FIELDS = [
    "gesture",
    "target_state",
    "reference_valence",
    "reference_arousal",
    "reference_dominance",
    "candidate_valence",
    "candidate_arousal",
    "candidate_dominance",
    "target_valence",
    "target_arousal",
    "target_dominance",
    "observed_shift_valence",
    "observed_shift_arousal",
    "observed_shift_dominance",
    "candidate_target_distance",
    "reference_target_distance",
    "directional_alignment",
    "normalised_progress",
    "paired_candidate_preferred",
    "paired_reference_preferred",
    "paired_neither",
    "reference_relative_outcome",
]


def reference_relative_csv_row(
    *,
    gesture: str,
    target_state: str,
    metrics: Mapping[str, Any] | None,
    paired_summary: Mapping[str, Any] | None,
    outcome_label: str,
) -> dict[str, Any]:
    def _axis(section: str, axis: str) -> Any:
        if metrics is None:
            return None
        return metrics[section][axis]

    return {
        "gesture": gesture,
        "target_state": target_state,
        "reference_valence": _axis("reference_vad", "valence"),
        "reference_arousal": _axis("reference_vad", "arousal"),
        "reference_dominance": _axis("reference_vad", "dominance"),
        "candidate_valence": _axis("candidate_vad", "valence"),
        "candidate_arousal": _axis("candidate_vad", "arousal"),
        "candidate_dominance": _axis("candidate_vad", "dominance"),
        "target_valence": _axis("target_vad", "valence"),
        "target_arousal": _axis("target_vad", "arousal"),
        "target_dominance": _axis("target_vad", "dominance"),
        "observed_shift_valence": _axis("observed_shift", "valence"),
        "observed_shift_arousal": _axis("observed_shift", "arousal"),
        "observed_shift_dominance": _axis("observed_shift", "dominance"),
        "candidate_target_distance": (
            None if metrics is None else metrics["candidate_target_distance"]
        ),
        "reference_target_distance": (
            None if metrics is None else metrics["reference_target_distance"]
        ),
        "directional_alignment": (
            None if metrics is None else metrics["directional_alignment"]
        ),
        "normalised_progress": (
            None if metrics is None else metrics["normalised_progress"]
        ),
        "paired_candidate_preferred": (
            None if paired_summary is None else paired_summary["candidate_preferred"]
        ),
        "paired_reference_preferred": (
            None if paired_summary is None else paired_summary["reference_preferred"]
        ),
        "paired_neither": (
            None if paired_summary is None else paired_summary["neither"]
        ),
        "reference_relative_outcome": outcome_label,
    }
