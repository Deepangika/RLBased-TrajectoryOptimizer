"""Pure scoring and reliability summaries for perceptual observations."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from laban_rl.affect import VAD_KEYS, validate_vad


def _field(observation: Any, name: str, default: Any = None) -> Any:
    if isinstance(observation, Mapping):
        return observation.get(name, default)
    return getattr(observation, name, default)


def observation_to_dict(observation: Any) -> dict[str, Any]:
    """Return the complete structured evaluator observation as JSON-safe data."""
    return {
        "affect_ratings": {
            key: float(value)
            for key, value in dict(_field(observation, "affect_ratings", {})).items()
        },
        "probabilities": {
            key: float(value)
            for key, value in dict(_field(observation, "probabilities", {})).items()
        },
        "confidence": (
            None
            if _field(observation, "confidence") is None
            else float(_field(observation, "confidence"))
        ),
        "perceived_state": _field(observation, "perceived_state"),
        "category_status": (
            _field(observation, "category_status")
            or (
                "complete"
                if _field(observation, "probabilities", {})
                else "missing"
            )
        ),
        "category_intensities": {
            key: float(value)
            for key, value in dict(
                _field(observation, "category_intensities", {})
            ).items()
        },
        "reasoning_summary": _field(observation, "reasoning_summary"),
    }


def _validate_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    values = np.asarray(list(probabilities.values()), dtype=float)
    if not probabilities:
        return {}
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Evaluator probabilities must be finite and non-negative.")
    if not np.isclose(float(np.sum(values)), 1.0, atol=1e-5):
        raise ValueError("Evaluator probabilities must sum to 1.0.")
    return {str(label): float(value) for label, value in probabilities.items()}


def score_perceptual_observations(
    observations: Sequence[Any],
    *,
    target_vad: Mapping[str, float],
    target_state: str | None,
    reward_config: Any,
) -> dict[str, Any]:
    """Score one repeated observation set without evaluator or environment state."""
    if not observations:
        raise ValueError("At least one perceptual observation is required.")
    affect_target = validate_vad(target_vad, name="Target VAD")
    target_vector = np.asarray([affect_target[key] for key in VAD_KEYS], dtype=float)
    affect_weights = np.asarray(
        [
            reward_config.valence_weight,
            reward_config.arousal_weight,
            reward_config.dominance_weight,
        ],
        dtype=float,
    )

    affective_records: list[dict[str, float]] = []
    probability_records: list[dict[str, float]] = []
    vad_rewards: list[float] = []
    categorical_rewards: list[float] = []
    categorical_rewards_clipped: list[float] = []
    target_probabilities: list[float] = []
    margins: list[float] = []
    margins_clipped: list[float] = []
    winning_labels: list[str] = []
    probability_entropies: list[float] = []
    confidences: list[float] = []
    category_statuses: list[str] = []
    category_intensity_records: list[dict[str, float]] = []

    for observation in observations:
        affect = validate_vad(
            dict(_field(observation, "affect_ratings", {})),
            name="Evaluator VAD",
        )
        affective_records.append(affect)
        affect_vector = np.asarray([affect[key] for key in VAD_KEYS], dtype=float)
        vad_rewards.append(
            1.0 - float(np.sum(affect_weights * np.abs(affect_vector - target_vector)))
        )

        confidence = _field(observation, "confidence")
        if confidence is not None:
            confidence = float(confidence)
            if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("Evaluator confidence must be finite and in [0, 1].")
            confidences.append(confidence)

        probabilities = _validate_probabilities(
            dict(_field(observation, "probabilities", {}))
        )
        category_status = str(
            _field(observation, "category_status")
            or ("complete" if probabilities else "missing")
        )
        if category_status not in ("complete", "ambiguous", "missing"):
            raise ValueError(f"Unknown category_status {category_status!r}.")
        category_statuses.append(category_status)
        intensities = {
            str(label): float(value)
            for label, value in dict(
                _field(observation, "category_intensities", {})
            ).items()
        }
        if intensities:
            values = np.asarray(list(intensities.values()), dtype=float)
            if (
                not np.all(np.isfinite(values))
                or np.any(values < 0.0)
                or np.any(values > 1.0)
            ):
                raise ValueError(
                    "Category intensities must be finite and in [0, 1]."
                )
            category_intensity_records.append(intensities)
        if not probabilities:
            continue
        probability_records.append(probabilities)
        if category_status != "complete":
            continue
        winning_labels.append(max(probabilities, key=probabilities.get))
        probability_values = np.asarray(list(probabilities.values()), dtype=float)
        probability_entropies.append(
            float(-np.sum(probability_values * np.log(probability_values + 1e-12)))
        )
        if target_state is None:
            continue
        if target_state not in probabilities:
            raise ValueError(
                f"Evaluator output does not contain target state {target_state!r}."
            )
        target_probability = float(probabilities[target_state])
        competitors = [
            value for label, value in probabilities.items() if label != target_state
        ]
        if not competitors:
            raise ValueError("Categorical reward requires at least two labels.")
        margin = target_probability - max(competitors)
        margin_clipped = max(0.0, margin)
        raw_reward = (
            reward_config.target_probability_weight * target_probability
            + reward_config.margin_weight * margin
        )
        clipped_reward = (
            reward_config.target_probability_weight * target_probability
            + reward_config.margin_weight * margin_clipped
        )
        target_probabilities.append(target_probability)
        margins.append(float(margin))
        margins_clipped.append(float(margin_clipped))
        categorical_rewards.append(float(raw_reward))
        categorical_rewards_clipped.append(float(clipped_reward))

    if reward_config.perceptual_reward_mode == "categorical":
        if target_state is None:
            raise ValueError("Categorical reward mode requires a named target_state.")
        if len(categorical_rewards) != len(observations):
            raise ValueError("Categorical reward mode requires evaluator probabilities.")

    use_clipped = (
        reward_config.reward_margin_mode == "clipped"
        or reward_config.clip_negative_margin
    )
    if reward_config.perceptual_reward_mode == "vad":
        effective_rewards = vad_rewards
        clipped_effective_rewards = vad_rewards
    else:
        effective_rewards = (
            categorical_rewards_clipped if use_clipped else categorical_rewards
        )
        clipped_effective_rewards = categorical_rewards_clipped

    affect_matrix = np.asarray(
        [[record[key] for key in VAD_KEYS] for record in affective_records],
        dtype=float,
    )
    mean_affect = np.mean(affect_matrix, axis=0)
    axis_std = np.std(affect_matrix, axis=0)
    per_axis_error = {
        key: float(abs(mean_affect[index] - target_vector[index]))
        for index, key in enumerate(VAD_KEYS)
    }
    label_values = sorted(
        {label for record in probability_records for label in record}
    )
    probability_std = {
        label: float(np.std([record.get(label, 0.0) for record in probability_records]))
        for label in label_values
    }
    winner_counts = Counter(winning_labels)
    repeat_count = len(observations)
    categorical_repeat_count = len(categorical_rewards)
    distribution_repeat_count = len(probability_records)
    complete_category_count = category_statuses.count("complete")
    ambiguous_category_count = category_statuses.count("ambiguous")
    missing_category_count = category_statuses.count("missing")
    categorical_complete = bool(
        target_state is not None and categorical_repeat_count == repeat_count
    )

    return {
        "affective_evaluations": affective_records,
        "perceptual_evaluations": probability_records,
        "category_intensity_evaluations": category_intensity_records,
        "mean_observed_vad": {
            key: float(mean_affect[index]) for index, key in enumerate(VAD_KEYS)
        },
        "per_axis_vad_std": {
            key: float(axis_std[index]) for index, key in enumerate(VAD_KEYS)
        },
        "per_axis_vad_error": per_axis_error,
        "mean_vad_error": float(
            np.sum(
                affect_weights
                * np.asarray([per_axis_error[key] for key in VAD_KEYS], dtype=float)
            )
        ),
        "mean_vad_reward": float(np.mean(vad_rewards)),
        "vad_reward_std": float(np.std(vad_rewards)),
        "categorical_reward": (
            float(
                np.mean(
                    categorical_rewards_clipped
                    if use_clipped
                    else categorical_rewards
                )
            )
            if categorical_complete
            else None
        ),
        "categorical_reward_clipped": (
            float(np.mean(categorical_rewards_clipped))
            if categorical_complete
            else None
        ),
        "categorical_reward_std": (
            float(
                np.std(
                    categorical_rewards_clipped
                    if use_clipped
                    else categorical_rewards
                )
            )
            if categorical_complete
            else None
        ),
        "categorical_repeat_count": categorical_repeat_count,
        "categorical_coverage": float(categorical_repeat_count / repeat_count),
        "categorical_distribution_coverage": float(
            distribution_repeat_count / repeat_count
        ),
        "categorical_unambiguous_coverage": float(
            complete_category_count / repeat_count
        ),
        "ambiguous_category_count": ambiguous_category_count,
        "missing_category_count": missing_category_count,
        "categorical_complete": categorical_complete,
        "mean_target_probability": (
            float(np.mean(target_probabilities)) if target_probabilities else None
        ),
        "mean_margin": float(np.mean(margins)) if margins else None,
        "mean_margin_clipped": (
            float(np.mean(margins_clipped)) if margins_clipped else None
        ),
        "mean_perceptual_reward": float(np.mean(effective_rewards)),
        "mean_perceptual_reward_clipped": float(
            np.mean(clipped_effective_rewards)
        ),
        "perceptual_reward_std": float(np.std(effective_rewards)),
        "target_classification_rate": (
            float(np.mean([label == target_state for label in winning_labels]))
            if winning_labels and target_state is not None
            else None
        ),
        "winner_agreement_rate": (
            float(max(winner_counts.values()) / len(winning_labels))
            if winning_labels
            else None
        ),
        "mean_probability_entropy": (
            float(np.mean(probability_entropies))
            if probability_entropies
            else None
        ),
        "probability_std": probability_std,
        "mean_confidence": (
            float(np.mean(confidences)) if confidences else None
        ),
        "confidence_std": (
            float(np.std(confidences)) if confidences else None
        ),
        "repeat_reliability": {
            "repeat_count": repeat_count,
            "icc": None,
            "icc_status": (
                "insufficient_repeats"
                if repeat_count < 2
                else "single_clip_dispersion_only"
            ),
            "within_clip_vad_std": {
                key: float(axis_std[index])
                for index, key in enumerate(VAD_KEYS)
            },
        },
    }


def apply_outer_penalties(
    perceptual_reward: float,
    *,
    perceptual_reward_std: float,
    realisation_rmse: float,
    max_abs_feature_error: float,
    reward_config: Any,
) -> float:
    """Apply the environment's non-perceptual penalties to any reward view."""
    return float(
        perceptual_reward
        - reward_config.realisation_penalty_weight * realisation_rmse
        - reward_config.stability_penalty_weight * perceptual_reward_std
        - reward_config.max_feature_error_penalty_weight
        * max(
            0.0,
            max_abs_feature_error - reward_config.max_feature_error_threshold,
        )
    )


def test_retest_reliability(
    repeated_observation_sets: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    """Compute ICC(2,1) per VAD axis when clips and repeats identify it."""
    repeat_counts = [len(records) for records in repeated_observation_sets]
    if len(repeated_observation_sets) < 2:
        return {
            "status": "insufficient_clips",
            "clip_count": len(repeated_observation_sets),
            "repeat_counts": repeat_counts,
            "icc": {key: None for key in VAD_KEYS},
        }
    if not repeat_counts or min(repeat_counts) < 2:
        return {
            "status": "insufficient_repeats",
            "clip_count": len(repeated_observation_sets),
            "repeat_counts": repeat_counts,
            "icc": {key: None for key in VAD_KEYS},
        }

    repeats = min(repeat_counts)
    matrices = {
        key: np.asarray(
            [
                [
                    validate_vad(
                        dict(_field(observation, "affect_ratings", {})),
                        name="Evaluator VAD",
                    )[key]
                    for observation in records[:repeats]
                ]
                for records in repeated_observation_sets
            ],
            dtype=float,
        )
        for key in VAD_KEYS
    }
    icc: dict[str, float | None] = {}
    axis_status: dict[str, str] = {}
    for key, matrix in matrices.items():
        n, k = matrix.shape
        grand = float(np.mean(matrix))
        row_means = np.mean(matrix, axis=1)
        col_means = np.mean(matrix, axis=0)
        ms_rows = float(k * np.sum((row_means - grand) ** 2) / (n - 1))
        ms_cols = float(n * np.sum((col_means - grand) ** 2) / (k - 1))
        residual = (
            matrix - row_means[:, None] - col_means[None, :] + grand
        )
        ms_error = float(np.sum(residual ** 2) / ((n - 1) * (k - 1)))
        denominator = (
            ms_rows
            + (k - 1) * ms_error
            + (k * (ms_cols - ms_error) / n)
        )
        if not np.isfinite(denominator) or abs(denominator) < 1e-12:
            icc[key] = None
            axis_status[key] = "degenerate_zero_variance"
        else:
            value = (ms_rows - ms_error) / denominator
            icc[key] = float(value) if np.isfinite(value) else None
            axis_status[key] = (
                "ok" if np.isfinite(value) else "non_finite_result"
            )
    return {
        "status": "ok" if all(value is not None for value in icc.values()) else "partial",
        "method": "ICC(2,1)",
        "clip_count": len(repeated_observation_sets),
        "repeat_counts": repeat_counts,
        "repeats_used": repeats,
        "icc": icc,
        "axis_status": axis_status,
    }
