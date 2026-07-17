"""One complete environment step for the perceptual Laban bandit.

This module implements the outer-loop transition:

    context
        -> continuous target Laban profile
        -> real trajectory optimiser
        -> validity check
        -> repeated perceptual evaluation
        -> outer reward

The policy is deliberately NOT implemented here yet. This environment accepts
a manually supplied continuous action so the full step can be tested safely
before a learning algorithm is attached.

Drop this file into:
    src/laban_rl/perceptual_bandit/environment.py
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from laban_rl.config import FEATURE_KEYS, GESTURE_TYPES
from laban_rl.optimiser_api import (
    LabanOptimisationResult,
    optimise_laban_target,
)


@dataclass(frozen=True)
class Context:
    """Context observed by the future contextual-bandit policy."""

    gesture: str
    target_state: str

    def validate(self) -> None:
        if self.gesture not in GESTURE_TYPES:
            raise ValueError(
                f"Unknown gesture {self.gesture!r}. "
                f"Expected one of {GESTURE_TYPES}."
            )
        if not self.target_state.strip():
            raise ValueError("target_state must be a non-empty string.")


@dataclass(frozen=True)
class PerceptualEvaluation:
    """One evaluator observation for one generated gesture."""

    probabilities: dict[str, float]

    def validate(self, target_state: str) -> None:
        if target_state not in self.probabilities:
            raise ValueError(
                f"Evaluator output does not contain target state "
                f"{target_state!r}. Available labels: "
                f"{list(self.probabilities)}"
            )

        values = np.asarray(
            list(self.probabilities.values()),
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Evaluator returned non-finite probabilities."
            )
        if np.any(values < 0.0):
            raise ValueError(
                "Evaluator probabilities must be non-negative."
            )

        total = float(np.sum(values))
        if not np.isclose(total, 1.0, atol=1e-5):
            raise ValueError(
                f"Evaluator probabilities must sum to 1.0; got {total:.8f}."
            )


class PerceptualEvaluator(Protocol):
    """Interface that the future real VLM adapter must implement."""

    def evaluate(
        self,
        context: Context,
        optimisation_result: LabanOptimisationResult,
    ) -> PerceptualEvaluation:
        ...


@dataclass(frozen=True)
class EnvironmentRewardConfig:
    """Weights for the outer perceptual reward."""

    repeat_evaluations: int = 3

    # Perceptual score:
    # alpha * P(target) + beta * classification margin.
    target_probability_weight: float = 0.5
    margin_weight: float = 0.5

    # Penalise requested profiles that the inner optimiser does not realise.
    realisation_penalty_weight: float = 0.25

    # Penalise the excess above an interpretable per-feature tolerance. RMSE
    # alone can hide one badly missed feature among four accurately realised
    # features.
    max_feature_error_threshold: float = 0.10
    max_feature_error_penalty_weight: float = 0.0
    reject_excessive_feature_error: bool = False

    # Leave at zero initially. Increase only after real VLM repeatability is
    # measured and there is evidence that instability should be penalised.
    stability_penalty_weight: float = 0.0

    # Reward margin mode: 'raw' or 'clipped'.
    # 'raw': use actual margin (can be negative if target loses)
    # 'clipped': use max(0, margin) to soften negative signals
    reward_margin_mode: str = "raw"

    # Deprecated flag; kept for backwards compatibility. Use reward_margin_mode instead.
    # When True, behaves like reward_margin_mode='clipped'.
    clip_negative_margin: bool = False

    # Used when any achieved Laban feature is NaN/inf.
    invalid_realisation_reward: float = -1.0

    # A finite feature vector is necessary but not sufficient. Do not send a
    # structurally invalid or joint-unsafe trajectory to the perceptual model.
    minimum_path_length_ratio: float = 0.70
    maximum_path_length_ratio: float = 1.30
    joint_limit_tolerance: float = 1e-6

    def validate(self) -> None:
        if self.repeat_evaluations < 1:
            raise ValueError(
                "repeat_evaluations must be at least 1."
            )

        if self.reward_margin_mode not in ("raw", "clipped"):
            raise ValueError(
                f"reward_margin_mode must be 'raw' or 'clipped', got {self.reward_margin_mode!r}."
            )

        for name in (
            "target_probability_weight",
            "margin_weight",
            "realisation_penalty_weight",
            "max_feature_error_penalty_weight",
            "stability_penalty_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative."
                )

        if not np.isfinite(self.max_feature_error_threshold) or self.max_feature_error_threshold < 0.0:
            raise ValueError("max_feature_error_threshold must be finite and non-negative.")

        if not np.isfinite(self.invalid_realisation_reward):
            raise ValueError(
                "invalid_realisation_reward must be finite."
            )
        if not 0.0 < self.minimum_path_length_ratio <= self.maximum_path_length_ratio:
            raise ValueError("Invalid path-length-ratio acceptance interval.")
        if self.joint_limit_tolerance < 0.0:
            raise ValueError("joint_limit_tolerance must be non-negative.")


@dataclass
class EnvironmentStepResult:
    """Complete record of one outer environment step."""

    context: Context

    requested_profile: dict[str, float]
    achieved_profile: dict[str, float]

    valid_realisation: bool
    failure_reason: str | None
    invalid_features: list[str]
    path_preserved: bool
    joint_limits_satisfied: bool
    physically_acceptable: bool

    realisation_rmse: float | None
    per_feature_abs_error: dict[str, float]
    max_abs_feature_error: float | None
    max_error_feature: str | None
    feature_realisation_acceptable: bool
    inner_loss: float
    inner_reward: float

    perceptual_evaluations: list[dict[str, float]]
    mean_target_probability: float | None
    mean_margin: float | None
    mean_margin_clipped: float | None
    mean_perceptual_reward: float | None
    mean_perceptual_reward_clipped: float | None
    perceptual_reward_std: float | None
    target_classification_rate: float | None
    winner_agreement_rate: float | None
    mean_probability_entropy: float | None

    outer_reward: float
    outer_reward_clipped: float

    optimiser_output_dir: str
    action_coefficients: list[float]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


class MockNoisyPerceptualEvaluator:
    """Temporary noisy evaluator used only to test the outer loop.

    It assigns each state a hidden ideal Laban profile. The evaluator compares
    the ACHIEVED profile, not the requested profile, with these hidden ideals.
    Gaussian logit noise makes repeated evaluation of the same generated
    gesture produce slightly different probabilities.

    Replace this class later with a real VLM adapter. The environment itself
    does not need to change.
    """

    DEFAULT_IDEALS: dict[str, tuple[float, ...]] = {
        "confident": (0.82, 0.78, 0.60, 0.22, 0.78),
        "calm": (0.35, 0.28, 0.28, 0.45, 0.45),
        "hesitant": (0.25, 0.32, 0.35, 0.70, 0.28),
        "friendly": (0.55, 0.58, 0.55, 0.48, 0.72),
        "confused": (0.20, 0.30, 0.15, 0.80, 0.50),
        "angry": (0.90, 0.85, 0.85, 0.10, 0.15),
    }

    def __init__(
        self,
        *,
        state_labels: Sequence[str] = (
            "confident",
            "calm",
            "hesitant",
            "friendly",
            "confused",
            "angry",
        ),
        ideal_profiles: Mapping[str, Sequence[float]] | None = None,
        noise_std: float = 0.08,
        distance_scale: float = 8.0,
        seed: int = 7,
    ) -> None:
        self.state_labels = tuple(state_labels)
        self.noise_std = float(noise_std)
        self.distance_scale = float(distance_scale)
        self.rng = np.random.default_rng(seed)

        source = dict(
            ideal_profiles
            if ideal_profiles is not None
            else self.DEFAULT_IDEALS
        )

        self.ideal_profiles: dict[str, np.ndarray] = {}
        for label in self.state_labels:
            if label not in source:
                raise ValueError(
                    f"No mock ideal profile provided for state {label!r}."
                )
            vector = np.asarray(source[label], dtype=float)
            if vector.shape != (len(FEATURE_KEYS),):
                raise ValueError(
                    f"Ideal profile for {label!r} must have "
                    f"{len(FEATURE_KEYS)} values."
                )
            if not np.all(np.isfinite(vector)):
                raise ValueError(
                    f"Ideal profile for {label!r} is non-finite."
                )
            self.ideal_profiles[label] = vector

    def evaluate(
        self,
        context: Context,
        optimisation_result: LabanOptimisationResult,
    ) -> PerceptualEvaluation:
        achieved = np.asarray(
            [
                optimisation_result.achieved_profile[key]
                for key in FEATURE_KEYS
            ],
            dtype=float,
        )

        if not np.all(np.isfinite(achieved)):
            raise ValueError(
                "Mock evaluator received an invalid achieved profile. "
                "The environment should reject invalid realisations before "
                "calling the evaluator."
            )

        logits: list[float] = []
        for label in self.state_labels:
            ideal = self.ideal_profiles[label]
            distance_sq = float(
                np.mean((achieved - ideal) ** 2)
            )
            logit = -self.distance_scale * distance_sq
            logit += float(
                self.rng.normal(0.0, self.noise_std)
            )
            logits.append(logit)

        logits_array = np.asarray(logits, dtype=float)
        logits_array -= np.max(logits_array)
        exp_logits = np.exp(logits_array)
        probabilities = exp_logits / np.sum(exp_logits)

        return PerceptualEvaluation(
            probabilities={
                label: float(probability)
                for label, probability in zip(
                    self.state_labels,
                    probabilities,
                )
            }
        )


class PerceptualBanditEnvironment:
    """Runs one complete context-action-reward round."""

    def __init__(
        self,
        *,
        evaluator: PerceptualEvaluator,
        reward_config: EnvironmentRewardConfig | None = None,
        optimiser_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.reward_config = (
            reward_config
            if reward_config is not None
            else EnvironmentRewardConfig()
        )
        self.reward_config.validate()
        self.optimiser_overrides = dict(
            optimiser_overrides or {}
        )

    def step(
        self,
        *,
        context: Context,
        action_profile: Mapping[str, float],
        out_dir: str | Path,
    ) -> EnvironmentStepResult:
        """Run one complete outer environment step.

        Parameters
        ----------
        context:
            Gesture identity and intended state.
        action_profile:
            Continuous five-dimensional Laban target selected by the future
            contextual-bandit policy.
        out_dir:
            Folder for the existing inner optimiser's normal outputs.
        """
        context.validate()

        optimisation_result = optimise_laban_target(
            gesture=context.gesture,
            target_state=context.target_state,
            target_profile=action_profile,
            out_dir=out_dir,
            optimiser_overrides=self.optimiser_overrides,
        )

        invalid_features = [
            key
            for key in FEATURE_KEYS
            if not np.isfinite(
                optimisation_result.achieved_profile[key]
            )
        ]

        reward_info = optimisation_result.raw_result.get("reward_info", {})
        path_length_ratio = float(reward_info.get("path_length_ratio", np.nan))
        joint_limit_error = float(reward_info.get("joint_limit_error", np.inf))
        path_preserved = bool(
            np.isfinite(path_length_ratio)
            and self.reward_config.minimum_path_length_ratio
            <= path_length_ratio
            <= self.reward_config.maximum_path_length_ratio
        )
        joint_limits_satisfied = bool(
            np.isfinite(joint_limit_error)
            and joint_limit_error <= self.reward_config.joint_limit_tolerance
        )
        physically_acceptable = bool(path_preserved and joint_limits_satisfied)

        if invalid_features:
            reason = (
                "non_finite_achieved_features:"
                + ",".join(invalid_features)
            )
            return EnvironmentStepResult(
                context=context,
                requested_profile=dict(
                    optimisation_result.requested_profile
                ),
                achieved_profile=dict(
                    optimisation_result.achieved_profile
                ),
                valid_realisation=False,
                failure_reason=reason,
                invalid_features=invalid_features,
                path_preserved=path_preserved,
                joint_limits_satisfied=joint_limits_satisfied,
                physically_acceptable=False,
                realisation_rmse=None,
                per_feature_abs_error={},
                max_abs_feature_error=None,
                max_error_feature=None,
                feature_realisation_acceptable=False,
                inner_loss=float(
                    optimisation_result.inner_loss
                ),
                inner_reward=float(
                    optimisation_result.inner_reward
                ),
                perceptual_evaluations=[],
                mean_target_probability=None,
                mean_margin=None,
                mean_margin_clipped=None,
                mean_perceptual_reward=None,
                mean_perceptual_reward_clipped=None,
                perceptual_reward_std=None,
                target_classification_rate=None,
                winner_agreement_rate=None,
                mean_probability_entropy=None,
                outer_reward=float(
                    self.reward_config.invalid_realisation_reward
                ),
                outer_reward_clipped=float(
                    self.reward_config.invalid_realisation_reward
                ),
                optimiser_output_dir=str(
                    optimisation_result.output_dir
                ),
                action_coefficients=(
                    optimisation_result
                    .action_coefficients
                    .astype(float)
                    .tolist()
                ),
            )

        if not physically_acceptable:
            failed = []
            if not path_preserved:
                failed.append("path_preservation")
            if not joint_limits_satisfied:
                failed.append("joint_limits")
            return EnvironmentStepResult(
                context=context,
                requested_profile=dict(optimisation_result.requested_profile),
                achieved_profile=dict(optimisation_result.achieved_profile),
                valid_realisation=False,
                failure_reason="physical_acceptance_failed:" + ",".join(failed),
                invalid_features=[],
                path_preserved=path_preserved,
                joint_limits_satisfied=joint_limits_satisfied,
                physically_acceptable=False,
                realisation_rmse=float(optimisation_result.realisation_rmse),
                per_feature_abs_error={},
                max_abs_feature_error=None,
                max_error_feature=None,
                feature_realisation_acceptable=False,
                inner_loss=float(optimisation_result.inner_loss),
                inner_reward=float(optimisation_result.inner_reward),
                perceptual_evaluations=[],
                mean_target_probability=None,
                mean_margin=None,
                mean_margin_clipped=None,
                mean_perceptual_reward=None,
                mean_perceptual_reward_clipped=None,
                perceptual_reward_std=None,
                target_classification_rate=None,
                winner_agreement_rate=None,
                mean_probability_entropy=None,
                outer_reward=float(self.reward_config.invalid_realisation_reward),
                outer_reward_clipped=float(self.reward_config.invalid_realisation_reward),
                optimiser_output_dir=str(optimisation_result.output_dir),
                action_coefficients=optimisation_result.action_coefficients.astype(float).tolist(),
            )

        requested_vector = np.asarray(
            [
                optimisation_result.requested_profile[key]
                for key in FEATURE_KEYS
            ],
            dtype=float,
        )
        achieved_vector = np.asarray(
            [
                optimisation_result.achieved_profile[key]
                for key in FEATURE_KEYS
            ],
            dtype=float,
        )

        realisation_rmse = float(
            np.sqrt(
                np.mean(
                    (requested_vector - achieved_vector) ** 2
                )
            )
        )
        per_feature_abs_error = {
            key: float(abs(requested_vector[index] - achieved_vector[index]))
            for index, key in enumerate(FEATURE_KEYS)
        }
        max_error_feature = max(per_feature_abs_error, key=per_feature_abs_error.get)
        max_abs_feature_error = per_feature_abs_error[max_error_feature]
        feature_realisation_acceptable = bool(
            max_abs_feature_error <= self.reward_config.max_feature_error_threshold
        )

        if (
            self.reward_config.reject_excessive_feature_error
            and not feature_realisation_acceptable
        ):
            return EnvironmentStepResult(
                context=context,
                requested_profile=dict(optimisation_result.requested_profile),
                achieved_profile=dict(optimisation_result.achieved_profile),
                valid_realisation=False,
                failure_reason=(
                    "feature_realisation_failed:"
                    f"{max_error_feature}={max_abs_feature_error:.6f}>"
                    f"{self.reward_config.max_feature_error_threshold:.6f}"
                ),
                invalid_features=[],
                path_preserved=path_preserved,
                joint_limits_satisfied=joint_limits_satisfied,
                physically_acceptable=physically_acceptable,
                realisation_rmse=realisation_rmse,
                per_feature_abs_error=per_feature_abs_error,
                max_abs_feature_error=max_abs_feature_error,
                max_error_feature=max_error_feature,
                feature_realisation_acceptable=False,
                inner_loss=float(optimisation_result.inner_loss),
                inner_reward=float(optimisation_result.inner_reward),
                perceptual_evaluations=[],
                mean_target_probability=None,
                mean_margin=None,
                mean_margin_clipped=None,
                mean_perceptual_reward=None,
                mean_perceptual_reward_clipped=None,
                perceptual_reward_std=None,
                target_classification_rate=None,
                winner_agreement_rate=None,
                mean_probability_entropy=None,
                outer_reward=float(self.reward_config.invalid_realisation_reward),
                outer_reward_clipped=float(self.reward_config.invalid_realisation_reward),
                optimiser_output_dir=str(optimisation_result.output_dir),
                action_coefficients=optimisation_result.action_coefficients.astype(float).tolist(),
            )

        probability_records: list[dict[str, float]] = []
        target_probabilities: list[float] = []
        margins: list[float] = []
        margins_clipped: list[float] = []
        perceptual_rewards: list[float] = []
        perceptual_rewards_clipped: list[float] = []
        winning_labels: list[str] = []
        probability_entropies: list[float] = []

        try:
            for _ in range(
                self.reward_config.repeat_evaluations
            ):
                evaluation = self.evaluator.evaluate(
                    context,
                    optimisation_result,
                )
                evaluation.validate(context.target_state)

                probabilities = dict(evaluation.probabilities)
                probability_records.append(probabilities)
                winning_labels.append(max(probabilities, key=probabilities.get))
                probability_values = np.asarray(list(probabilities.values()), dtype=float)
                probability_entropies.append(float(-np.sum(
                    probability_values * np.log(probability_values + 1e-12)
                )))

                target_probability = float(
                    probabilities[context.target_state]
                )
                best_competitor = max(
                    probability
                    for label, probability in probabilities.items()
                    if label != context.target_state
                )
                margin = (
                    target_probability - best_competitor
                )
                margin_clipped = max(0.0, margin)

                # Use raw or clipped margin for perceptual reward.
                use_clipped = (
                    self.reward_config.reward_margin_mode == "clipped"
                    or self.reward_config.clip_negative_margin
                )
                effective_margin = (
                    margin_clipped if use_clipped else margin
                )

                perceptual_reward = (
                    self.reward_config.target_probability_weight
                    * target_probability
                    + self.reward_config.margin_weight
                    * effective_margin
                )

                perceptual_reward_clipped = (
                    self.reward_config.target_probability_weight
                    * target_probability
                    + self.reward_config.margin_weight
                    * margin_clipped
                )

                target_probabilities.append(target_probability)
                margins.append(margin)
                margins_clipped.append(margin_clipped)
                perceptual_rewards.append(
                    float(perceptual_reward)
                )
                perceptual_rewards_clipped.append(
                    float(perceptual_reward_clipped)
                )
        except Exception as evaluator_error:
            print(f"⚠ Evaluator failed: {evaluator_error}")
            print(f"  Using fallback: penalizing with invalid_realisation_reward")
            # Return early with penalty reward
            return EnvironmentStepResult(
                context=context,
                requested_profile=dict(
                    optimisation_result.requested_profile
                ),
                achieved_profile=dict(
                    optimisation_result.achieved_profile
                ),
                valid_realisation=True,  # Realisation was valid, evaluator failed
                failure_reason=f"evaluator_error: {str(evaluator_error)}",
                invalid_features=[],
                path_preserved=path_preserved,
                joint_limits_satisfied=joint_limits_satisfied,
                physically_acceptable=physically_acceptable,
                realisation_rmse=realisation_rmse,
                per_feature_abs_error=per_feature_abs_error,
                max_abs_feature_error=max_abs_feature_error,
                max_error_feature=max_error_feature,
                feature_realisation_acceptable=feature_realisation_acceptable,
                inner_loss=float(
                    optimisation_result.inner_loss
                ),
                inner_reward=float(
                    optimisation_result.inner_reward
                ),
                perceptual_evaluations=[],
                mean_target_probability=None,
                mean_margin=None,
                mean_margin_clipped=None,
                mean_perceptual_reward=None,
                mean_perceptual_reward_clipped=None,
                perceptual_reward_std=None,
                target_classification_rate=None,
                winner_agreement_rate=None,
                mean_probability_entropy=None,
                outer_reward=float(
                    self.reward_config.invalid_realisation_reward
                ),
                outer_reward_clipped=float(
                    self.reward_config.invalid_realisation_reward
                ),
                optimiser_output_dir=str(
                    optimisation_result.output_dir
                ),
                action_coefficients=(
                    optimisation_result
                    .action_coefficients
                    .astype(float)
                    .tolist()
                ),
            )

        mean_target_probability = float(
            np.mean(target_probabilities)
        )
        mean_margin = float(np.mean(margins))
        mean_margin_clipped = float(np.mean(margins_clipped))
        mean_perceptual_reward = float(
            np.mean(perceptual_rewards)
        )
        mean_perceptual_reward_clipped = float(
            np.mean(perceptual_rewards_clipped)
        )
        perceptual_reward_std = float(
            np.std(perceptual_rewards)
        )
        target_classification_rate = float(np.mean([
            label == context.target_state for label in winning_labels
        ]))
        winner_counts = {
            label: winning_labels.count(label) for label in set(winning_labels)
        }
        winner_agreement_rate = float(max(winner_counts.values()) / len(winning_labels))
        mean_probability_entropy = float(np.mean(probability_entropies))

        # Always compute both raw and clipped for logging.
        outer_reward = (
            mean_perceptual_reward
            - self.reward_config.realisation_penalty_weight
            * realisation_rmse
            - self.reward_config.stability_penalty_weight
            * perceptual_reward_std
            - self.reward_config.max_feature_error_penalty_weight
            * max(
                0.0,
                max_abs_feature_error - self.reward_config.max_feature_error_threshold,
            )
        )

        outer_reward_clipped = (
            mean_perceptual_reward_clipped
            - self.reward_config.realisation_penalty_weight
            * realisation_rmse
            - self.reward_config.stability_penalty_weight
            * perceptual_reward_std
            - self.reward_config.max_feature_error_penalty_weight
            * max(
                0.0,
                max_abs_feature_error - self.reward_config.max_feature_error_threshold,
            )
        )

        return EnvironmentStepResult(
            context=context,
            requested_profile=dict(
                optimisation_result.requested_profile
            ),
            achieved_profile=dict(
                optimisation_result.achieved_profile
            ),
            valid_realisation=True,
            failure_reason=None,
            invalid_features=[],
            path_preserved=path_preserved,
            joint_limits_satisfied=joint_limits_satisfied,
            physically_acceptable=physically_acceptable,
            realisation_rmse=realisation_rmse,
            per_feature_abs_error=per_feature_abs_error,
            max_abs_feature_error=max_abs_feature_error,
            max_error_feature=max_error_feature,
            feature_realisation_acceptable=feature_realisation_acceptable,
            inner_loss=float(
                optimisation_result.inner_loss
            ),
            inner_reward=float(
                optimisation_result.inner_reward
            ),
            perceptual_evaluations=probability_records,
            mean_target_probability=mean_target_probability,
            mean_margin=mean_margin,
            mean_margin_clipped=mean_margin_clipped,
            mean_perceptual_reward=mean_perceptual_reward,
            mean_perceptual_reward_clipped=mean_perceptual_reward_clipped,
            perceptual_reward_std=perceptual_reward_std,
            target_classification_rate=target_classification_rate,
            winner_agreement_rate=winner_agreement_rate,
            mean_probability_entropy=mean_probability_entropy,
            outer_reward=float(outer_reward),
            outer_reward_clipped=float(outer_reward_clipped),
            optimiser_output_dir=str(
                optimisation_result.output_dir
            ),
            action_coefficients=(
                optimisation_result
                .action_coefficients
                .astype(float)
                .tolist()
            ),
        )
