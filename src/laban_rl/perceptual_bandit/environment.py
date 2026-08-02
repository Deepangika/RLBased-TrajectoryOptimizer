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

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from laban_rl.config import EMOTION_STATES, FEATURE_KEYS, GESTURE_TYPES
from laban_rl.affect import (
    VAD_KEYS,
    VAD_TARGETS,
    VADVector,
    target_vad,
    validate_vad,
)
from laban_rl.optimiser_api import (
    LabanOptimisationResult,
    optimise_laban_target,
)
from laban_rl.perceptual_bandit.scoring import (
    apply_outer_penalties,
    score_perceptual_observations,
)


@dataclass(frozen=True)
class Context:
    """Gesture plus exactly one named or explicit continuous affect target."""

    gesture: str
    target_state: str | None = None
    target_vad: Mapping[str, float] | None = None
    target_mode: Literal["named", "vad"] = field(init=False)

    def __post_init__(self) -> None:
        supplied_state = self.target_state
        supplied_vad = self.target_vad
        if (supplied_state is None) == (supplied_vad is None):
            raise ValueError(
                "Specify exactly one of target_state or target_vad."
            )

        if supplied_state is not None:
            if not isinstance(supplied_state, str) or not supplied_state.strip():
                raise ValueError("target_state must be a non-empty string.")
            resolved_vad = target_vad(supplied_state)
            mode: Literal["named", "vad"] = "named"
        else:
            resolved_vad = validate_vad(
                supplied_vad or {},
                name="Target VAD",
            )
            mode = "vad"

        object.__setattr__(self, "target_vad", resolved_vad)
        object.__setattr__(self, "target_mode", mode)
        self.validate()

    def validate(self) -> None:
        if self.gesture not in GESTURE_TYPES:
            raise ValueError(
                f"Unknown gesture {self.gesture!r}. "
                f"Expected one of {GESTURE_TYPES}."
            )
        resolved_vad = validate_vad(self.target_vad or {}, name="Target VAD")
        if self.target_mode == "named":
            if self.target_state is None:
                raise ValueError("Named target context requires target_state.")
            if resolved_vad != target_vad(self.target_state):
                raise ValueError(
                    "Named target VAD does not match its configured anchor."
                )
        elif self.target_state is not None:
            raise ValueError("Direct VAD target context cannot have target_state.")

    @property
    def target_label(self) -> str:
        """Human-readable metadata label for the inner optimizer."""
        if self.target_state is not None:
            return self.target_state
        assert self.target_vad is not None
        return (
            f"vad_v{self.target_vad['valence']:.3f}"
            f"_a{self.target_vad['arousal']:.3f}"
            f"_d{self.target_vad['dominance']:.3f}"
        )

    @property
    def key(self) -> str:
        return f"{self.gesture}::{self.target_label}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gesture": self.gesture,
            "target_mode": self.target_mode,
            "target_state": self.target_state,
            "target_vad": dict(self.target_vad or {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Context:
        """Load current metadata or a legacy named-state context."""
        gesture = str(payload["gesture"])
        mode = payload.get("target_mode")
        saved_state = payload.get("target_state")
        saved_vad = payload.get("target_vad")

        if mode is None:
            mode = "named" if saved_state is not None else "vad"

        if mode == "named":
            if saved_state is None:
                raise ValueError("Named context metadata is missing target_state.")
            context = cls(gesture=gesture, target_state=str(saved_state))
            if saved_vad is not None and validate_vad(
                saved_vad,
                name="Saved target VAD",
            ) != context.target_vad:
                raise ValueError(
                    "Saved named target VAD does not match its configured anchor."
                )
            return context
        if mode == "vad":
            if saved_state is not None:
                raise ValueError(
                    "Direct VAD context metadata must not include target_state."
                )
            if saved_vad is None:
                raise ValueError("Direct VAD context metadata is missing target_vad.")
            return cls(gesture=gesture, target_vad=saved_vad)
        raise ValueError(f"Unknown target_mode {mode!r}.")


@dataclass(frozen=True)
class PerceptualEvaluation:
    """One evaluator observation for one generated gesture."""

    affect_ratings: dict[str, float]
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    perceived_state: str | None = None
    category_status: Literal["complete", "ambiguous", "missing"] | None = None
    category_intensities: dict[str, float] = field(default_factory=dict)
    reasoning_summary: str | None = None

    def validate(self, target_state: str | None = None) -> None:
        validate_vad(self.affect_ratings, name="Evaluator VAD")
        resolved_status = self.category_status or (
            "complete" if self.probabilities else "missing"
        )
        if resolved_status not in ("complete", "ambiguous", "missing"):
            raise ValueError(
                "category_status must be 'complete', 'ambiguous', or 'missing'."
            )
        intensity_values = np.asarray(
            list(self.category_intensities.values()), dtype=float
        )
        if self.category_intensities and (
            not np.all(np.isfinite(intensity_values))
            or np.any(intensity_values < 0.0)
            or np.any(intensity_values > 1.0)
        ):
            raise ValueError("Category intensities must be finite and in [0, 1].")

        if not self.probabilities:
            if resolved_status == "complete":
                raise ValueError(
                    "Complete categorical output requires normalized probabilities."
                )
            if self.confidence is not None and (
                not np.isfinite(self.confidence)
                or not 0.0 <= self.confidence <= 1.0
            ):
                raise ValueError("Evaluator confidence must be finite and in [0, 1].")
            return
        if resolved_status == "missing":
            raise ValueError(
                "Missing categorical output must not contain probabilities."
            )
        if target_state is not None and target_state not in self.probabilities:
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
        if self.confidence is not None and (
            not np.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("Evaluator confidence must be finite and in [0, 1].")


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

    # VAD is the primary perceptual objective. Categorical mode remains
    # available so legacy experiments can be reproduced.
    perceptual_reward_mode: str = "vad"
    valence_weight: float = 0.20
    arousal_weight: float = 0.40
    dominance_weight: float = 0.40

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

    # Infrastructure/API failures are not evidence that an action is poor.
    # Raise by default so the caller can retry or checkpoint cleanly instead
    # of contaminating the CEM ranking with an artificial -1 reward.
    evaluator_failure_mode: str = "raise"

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

        if self.perceptual_reward_mode not in ("vad", "categorical"):
            raise ValueError(
                "perceptual_reward_mode must be 'vad' or 'categorical'."
            )
        vad_weights = np.asarray(
            [self.valence_weight, self.arousal_weight, self.dominance_weight],
            dtype=float,
        )
        if not np.all(np.isfinite(vad_weights)) or np.any(vad_weights < 0.0):
            raise ValueError("VAD reward weights must be finite and non-negative.")
        if not np.isclose(float(np.sum(vad_weights)), 1.0, atol=1e-8):
            raise ValueError("VAD reward weights must sum to 1.0.")

        if self.reward_margin_mode not in ("raw", "clipped"):
            raise ValueError(
                f"reward_margin_mode must be 'raw' or 'clipped', got {self.reward_margin_mode!r}."
            )
        if self.evaluator_failure_mode not in ("raise", "penalise"):
            raise ValueError(
                "evaluator_failure_mode must be 'raise' or 'penalise'."
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

    target_vad: dict[str, float] | None = None
    affective_evaluations: list[dict[str, float]] = field(default_factory=list)
    mean_observed_vad: dict[str, float] | None = None
    per_axis_vad_error: dict[str, float] = field(default_factory=dict)
    mean_vad_error: float | None = None
    mean_vad_reward: float | None = None
    vad_reward_std: float | None = None
    categorical_reward: float | None = None
    per_axis_vad_std: dict[str, float] = field(default_factory=dict)
    probability_std: dict[str, float] = field(default_factory=dict)
    mean_confidence: float | None = None
    confidence_std: float | None = None
    repeat_reliability: dict[str, Any] = field(default_factory=dict)
    category_intensity_evaluations: list[dict[str, float]] = field(
        default_factory=list
    )
    categorical_distribution_coverage: float = 0.0
    categorical_unambiguous_coverage: float = 0.0
    ambiguous_category_count: int = 0
    missing_category_count: int = 0
    categorical_complete: bool = False

    def __post_init__(self) -> None:
        if self.target_vad is None:
            self.target_vad = dict(self.context.target_vad or {})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["context"] = self.context.to_dict()
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
        # Earlier communicative-state set
        "confident": (0.82, 0.78, 0.60, 0.22, 0.78),
        "calm": (0.35, 0.28, 0.28, 0.45, 0.45),
        "hesitant": (0.25, 0.32, 0.35, 0.70, 0.28),
        "friendly": (0.55, 0.58, 0.55, 0.48, 0.72),
        "confused": (0.20, 0.30, 0.15, 0.80, 0.50),
        "angry": (0.90, 0.85, 0.85, 0.10, 0.15),
        # Ekman emotion set. Values are synthetic mock targets used only for
        # pipeline testing; they are not empirical affect annotations.
        "anger": (0.80, 0.80, 0.75, 0.25, 0.30),
        "disgust": (0.55, 0.45, 0.75, 0.20, 0.20),
        "fear": (0.35, 0.75, 0.80, 0.65, 0.35),
        "happiness": (0.60, 0.65, 0.25, 0.45, 0.70),
        "sadness": (0.20, 0.20, 0.65, 0.35, 0.25),
        "surprise": (0.65, 0.90, 0.30, 0.50, 0.55),
    }

    def __init__(
        self,
        *,
        state_labels: Sequence[str] | None = None,
        ideal_profiles: Mapping[str, Sequence[float]] | None = None,
        noise_std: float = 0.08,
        distance_scale: float = 8.0,
        seed: int = 7,
    ) -> None:
        self.state_labels = tuple(
            EMOTION_STATES if state_labels is None else state_labels
        )
        if not self.state_labels:
            raise ValueError("Mock evaluator requires at least one state label.")
        self.noise_std = float(noise_std)
        self.distance_scale = float(distance_scale)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

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

    def cache_identity(self) -> dict[str, Any]:
        return {
            "provider": "mock",
            "model": "synthetic-laban-distance",
            "prompt_version": "mock-vad-mixture-v1",
            "schema_version": "perceptual-evaluation-v2",
            "settings": {
                "state_labels": list(self.state_labels),
                "ideal_profiles": {
                    label: self.ideal_profiles[label].astype(float).tolist()
                    for label in self.state_labels
                },
                "noise_std": self.noise_std,
                "distance_scale": self.distance_scale,
                "seed": self.seed,
            },
        }

    def _evaluate_with_rng(
        self,
        context: Context,
        optimisation_result: LabanOptimisationResult,
        rng: np.random.Generator,
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
                rng.normal(0.0, self.noise_std)
            )
            logits.append(logit)

        logits_array = np.asarray(logits, dtype=float)
        logits_array -= np.max(logits_array)
        exp_logits = np.exp(logits_array)
        probabilities = exp_logits / np.sum(exp_logits)
        affect_ratings = {
            axis: float(
                sum(
                    probability * VAD_TARGETS[label][axis]
                    for label, probability in zip(self.state_labels, probabilities)
                )
            )
            for axis in VAD_KEYS
        }

        return PerceptualEvaluation(
            affect_ratings=affect_ratings,
            probabilities={
                label: float(probability)
                for label, probability in zip(
                    self.state_labels,
                    probabilities,
                )
            },
            confidence=float(np.max(probabilities)),
            perceived_state=self.state_labels[int(np.argmax(probabilities))],
        )

    def evaluate(
        self,
        context: Context,
        optimisation_result: LabanOptimisationResult,
    ) -> PerceptualEvaluation:
        return self._evaluate_with_rng(context, optimisation_result, self.rng)

    def evaluate_repeat(
        self,
        context: Context,
        optimisation_result: LabanOptimisationResult,
        *,
        repeat_index: int,
        clip_hash: str,
    ) -> PerceptualEvaluation:
        material = f"{self.seed}:{clip_hash}:{repeat_index}".encode("ascii")
        repeat_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        return self._evaluate_with_rng(
            context,
            optimisation_result,
            np.random.default_rng(repeat_seed),
        )


class PerceptualBanditEnvironment:
    """Runs one complete context-action-reward round."""

    def __init__(
        self,
        *,
        evaluator: PerceptualEvaluator,
        reward_config: EnvironmentRewardConfig | None = None,
        optimiser_overrides: Mapping[str, Any] | None = None,
        observation_cache: Any | None = None,
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
        self.observation_cache = observation_cache

    def run_inner_step(
        self,
        *,
        context: Context,
        action_profile: Mapping[str, float],
        out_dir: str | Path,
    ) -> LabanOptimisationResult:
        """Run only the inner trajectory optimiser without perceptual evaluation.

        Returns the raw ``LabanOptimisationResult`` so that the caller can
        subsequently pass it to :meth:`step_from_result`, allowing evaluator
        retries without repeating the expensive inner optimisation.
        """
        context.validate()
        return optimise_laban_target(
            gesture=context.gesture,
            target_state=context.target_label,
            target_profile=action_profile,
            out_dir=out_dir,
            optimiser_overrides=self.optimiser_overrides,
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
            target_state=context.target_label,
            target_profile=action_profile,
            out_dir=out_dir,
            optimiser_overrides=self.optimiser_overrides,
        )

        return self.step_from_result(
            context=context,
            optimisation_result=optimisation_result,
        )

    def step_from_result(
        self,
        *,
        context: Context,
        optimisation_result: LabanOptimisationResult,
    ) -> EnvironmentStepResult:
        """Run only the evaluation phase on an already-computed optimisation result.

        Use this method when the inner optimiser has already been called and you
        want to retry only the (cheaper) perceptual evaluation, for example after
        a transient API failure.
        """
        context.validate()
        if optimisation_result.gesture != context.gesture:
            raise ValueError(
                "Optimisation result gesture does not match evaluation context: "
                f"{optimisation_result.gesture!r} != {context.gesture!r}."
            )
        if (
            context.target_state is not None
            and optimisation_result.target_state != context.target_state
        ):
            raise ValueError(
                "Optimisation result target state does not match evaluation context: "
                f"{optimisation_result.target_state!r} != {context.target_state!r}."
            )
        if (
            self.reward_config.perceptual_reward_mode == "categorical"
            and context.target_state is None
        ):
            raise ValueError(
                "Categorical reward mode requires a named target_state; "
                "direct target_vad contexts use VAD reward mode."
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

        observations: list[PerceptualEvaluation] = []
        affect_target: VADVector = validate_vad(
            context.target_vad or {},
            name="Target VAD",
        )

        try:
            if self.observation_cache is not None:
                observations = self.observation_cache.collect(
                    context=context,
                    result=optimisation_result,
                    evaluator=self.evaluator,
                    repeats=self.reward_config.repeat_evaluations,
                )
            else:
                for _ in range(self.reward_config.repeat_evaluations):
                    evaluation = self.evaluator.evaluate(
                        context,
                        optimisation_result,
                    )
                    evaluation.validate(context.target_state)
                    observations.append(evaluation)
        except Exception as evaluator_error:
            if self.reward_config.evaluator_failure_mode == "raise":
                raise RuntimeError(
                    "Perceptual evaluator did not complete every requested "
                    "repetition; candidate reward is unavailable."
                ) from evaluator_error
            else:
                print(f"⚠ Evaluator repeat set was incomplete: {evaluator_error}")
                print(f"  Using fallback: penalizing with invalid_realisation_reward")
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

        try:
            paired = score_perceptual_observations(
                observations,
                target_vad=affect_target,
                target_state=context.target_state,
                reward_config=self.reward_config,
            )
        except Exception as evaluator_error:
            raise RuntimeError(
                "Perceptual evaluator failed on every repetition; "
                "candidate reward is unavailable."
            ) from evaluator_error

        outer_reward = apply_outer_penalties(
            paired["mean_perceptual_reward"],
            perceptual_reward_std=paired["perceptual_reward_std"],
            realisation_rmse=realisation_rmse,
            max_abs_feature_error=max_abs_feature_error,
            reward_config=self.reward_config,
        )
        outer_reward_clipped = apply_outer_penalties(
            paired["mean_perceptual_reward_clipped"],
            perceptual_reward_std=paired["perceptual_reward_std"],
            realisation_rmse=realisation_rmse,
            max_abs_feature_error=max_abs_feature_error,
            reward_config=self.reward_config,
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
            perceptual_evaluations=paired["perceptual_evaluations"],
            mean_target_probability=paired["mean_target_probability"],
            mean_margin=paired["mean_margin"],
            mean_margin_clipped=paired["mean_margin_clipped"],
            mean_perceptual_reward=paired["mean_perceptual_reward"],
            mean_perceptual_reward_clipped=paired[
                "mean_perceptual_reward_clipped"
            ],
            perceptual_reward_std=paired["perceptual_reward_std"],
            target_classification_rate=paired["target_classification_rate"],
            winner_agreement_rate=paired["winner_agreement_rate"],
            mean_probability_entropy=paired["mean_probability_entropy"],
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
            target_vad=affect_target,
            affective_evaluations=paired["affective_evaluations"],
            mean_observed_vad=paired["mean_observed_vad"],
            per_axis_vad_error=paired["per_axis_vad_error"],
            mean_vad_error=paired["mean_vad_error"],
            mean_vad_reward=paired["mean_vad_reward"],
            vad_reward_std=paired["vad_reward_std"],
            categorical_reward=paired["categorical_reward"],
            per_axis_vad_std=paired["per_axis_vad_std"],
            probability_std=paired["probability_std"],
            mean_confidence=paired["mean_confidence"],
            confidence_std=paired["confidence_std"],
            repeat_reliability=paired["repeat_reliability"],
            category_intensity_evaluations=paired[
                "category_intensity_evaluations"
            ],
            categorical_distribution_coverage=paired[
                "categorical_distribution_coverage"
            ],
            categorical_unambiguous_coverage=paired[
                "categorical_unambiguous_coverage"
            ],
            ambiguous_category_count=paired["ambiguous_category_count"],
            missing_category_count=paired["missing_category_count"],
            categorical_complete=paired["categorical_complete"],
        )