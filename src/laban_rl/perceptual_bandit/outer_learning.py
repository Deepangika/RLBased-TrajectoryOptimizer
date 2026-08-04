"""Gesture-conditioned correlated outer-learning utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from laban_rl.affect import VAD_KEYS, validate_vad
from laban_rl.config import FEATURE_KEYS

ACTION_DIMENSION = len(FEATURE_KEYS)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(clipped) - np.log1p(-clipped)


def profile_to_vector(profile: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(profile[key]) for key in FEATURE_KEYS], dtype=float)


def vector_to_profile(vector: Sequence[float]) -> dict[str, float]:
    values = np.asarray(vector, dtype=float)
    return {
        key: float(values[index]) for index, key in enumerate(FEATURE_KEYS)
    }


@dataclass(frozen=True)
class OuterLearningContext:
    gesture: str
    target_state: str

    @property
    def key(self) -> str:
        return f"{self.gesture}::{self.target_state}"


@dataclass(frozen=True)
class LatentSample:
    sample_id: str
    round_index: int
    action: dict[str, float]
    latent: tuple[float, ...]
    reward: float
    feasible: bool = True
    evaluation_consumed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributionUpdateDiagnostics:
    round_index: int
    elite_count: int
    covariance_source: str
    correction: str | None
    log_determinant: float
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    eigenvalues: list[float]
    correlation_matrix: list[list[float]]


def weighted_vad_reward(
    observed_vad: Mapping[str, float],
    target_vad: Mapping[str, float],
    *,
    valence_weight: float = 0.20,
    arousal_weight: float = 0.40,
    dominance_weight: float = 0.40,
) -> float:
    observed = validate_vad(observed_vad, name="Observed VAD")
    target = validate_vad(target_vad, name="Target VAD")
    distance = (
        valence_weight * abs(observed["valence"] - target["valence"])
        + arousal_weight * abs(observed["arousal"] - target["arousal"])
        + dominance_weight * abs(observed["dominance"] - target["dominance"])
    )
    return 1.0 - float(distance)


def penalized_vad_reward(
    observed_vad: Mapping[str, float],
    target_vad: Mapping[str, float],
    *,
    realisation_error: float,
    realisation_penalty_weight: float = 0.25,
    valence_weight: float = 0.20,
    arousal_weight: float = 0.40,
    dominance_weight: float = 0.40,
) -> float:
    return weighted_vad_reward(
        observed_vad,
        target_vad,
        valence_weight=valence_weight,
        arousal_weight=arousal_weight,
        dominance_weight=dominance_weight,
    ) - realisation_penalty_weight * float(realisation_error)


def robust_rank_key(sample: LatentSample) -> tuple[float, float, float, float, int]:
    """Lexicographic feasibility-first ranking key for formally feasible samples.

    Sort ascending. Ordering priority:
    1. robustly feasible candidates before marginally feasible ones;
    2. higher reward;
    3. lower maximum absolute feature error;
    4. lower realisation RMSE;
    5. deterministic sample-index tie-breaker.

    Samples without robustness metadata (e.g. synthetic diagnostics) are
    treated as robust so ranking degrades to the original reward ordering.
    The key never redefines formal feasibility; infeasible candidates must
    be excluded before ranking.
    """
    metadata = sample.metadata or {}
    robust = metadata.get("robustly_feasible")
    robust_flag = 1 if (robust is None or bool(robust)) else 0
    raw_error = metadata.get("max_abs_feature_error")
    max_error = float(raw_error) if raw_error is not None else 0.0
    raw_rmse = metadata.get("realisation_rmse")
    rmse = float(raw_rmse) if raw_rmse is not None else 0.0
    tie_break = int(metadata.get("sample_index", 0))
    return (-float(robust_flag), -float(sample.reward), max_error, rmse, tie_break)


def classify_outcome(
    *,
    valid_realisation: bool,
    validated_vad_improvement_over_baseline: float,
    learned_preference_wins_over_baseline: int,
    paired_repeats: int,
    in_target_quadrant: bool,
    stable_reward: bool,
    converged: bool,
) -> str:
    if not valid_realisation:
        return "feasible_only"
    if (
        validated_vad_improvement_over_baseline >= 0.05
        and in_target_quadrant
        and learned_preference_wins_over_baseline >= max(4, paired_repeats - 1)
    ):
        return "full_perceptual_success"
    if converged and validated_vad_improvement_over_baseline < 0.05:
        return "converged_unsuccessfully"
    if validated_vad_improvement_over_baseline >= 0.05 and not in_target_quadrant:
        return "vad_improvement_without_quadrant_match"
    if in_target_quadrant and not stable_reward:
        return "quadrant_match_without_stable_reward"
    return "relative_preference_improvement"


def stopping_reason(
    *,
    round_index: int,
    minimum_rounds: int,
    maximum_rounds: int,
    plateau_rounds: int,
    plateau_patience: int,
    current_action_std: float,
    maximum_action_std_for_convergence: float,
    feasible_candidates: int,
    evaluator_failed: bool = False,
) -> str | None:
    if evaluator_failed:
        return "evaluator_failure"
    if feasible_candidates == 0:
        return "no_feasible_candidates"
    if round_index < minimum_rounds:
        return None
    if plateau_rounds >= plateau_patience:
        return "reward_plateau"
    if current_action_std <= maximum_action_std_for_convergence:
        return "distribution_converged"
    if round_index >= maximum_rounds:
        return "maximum_budget_reached"
    return None


def require_reward_consumption(
    samples: Sequence[LatentSample],
) -> None:
    missing = [sample.sample_id for sample in samples if not sample.evaluation_consumed]
    if missing:
        raise RuntimeError(
            "Evaluator results were generated but not consumed by the CEM update: "
            + ", ".join(sorted(missing))
        )


class LatentGaussianCEMDistribution:
    def __init__(
        self,
        *,
        initial_action_mean: Mapping[str, float],
        initial_covariance: np.ndarray,
        covariance_shrinkage: float = 0.50,
        covariance_smoothing: float = 0.30,
        mean_smoothing: float = 0.40,
        minimum_eigenvalue: float = 1e-3,
        maximum_eigenvalue: float = 1.0,
        covariance_history_rounds: int = 3,
        round_weight_decay: float = 0.5,
        fallback_diagonal_variance: float = 0.10,
        min_full_covariance_elites: int = 3,
        seed: int = 7,
    ) -> None:
        if not 0.0 <= covariance_shrinkage <= 1.0:
            raise ValueError("covariance_shrinkage must lie in [0, 1].")
        if not 0.0 < covariance_smoothing <= 1.0:
            raise ValueError("covariance_smoothing must lie in (0, 1].")
        if not 0.0 < mean_smoothing <= 1.0:
            raise ValueError("mean_smoothing must lie in (0, 1].")
        if covariance_history_rounds < 1:
            raise ValueError("covariance_history_rounds must be at least 1.")
        if min_full_covariance_elites < 2:
            raise ValueError("min_full_covariance_elites must be at least 2.")
        initial_vector = profile_to_vector(initial_action_mean)
        if np.any(initial_vector <= 0.0) or np.any(initial_vector >= 1.0):
            raise ValueError("Initial action mean must lie strictly inside (0, 1).")
        self.mu_z = logit(initial_vector)
        self.covariance_shrinkage = float(covariance_shrinkage)
        self.covariance_smoothing = float(covariance_smoothing)
        self.mean_smoothing = float(mean_smoothing)
        self.minimum_eigenvalue = float(minimum_eigenvalue)
        self.maximum_eigenvalue = float(maximum_eigenvalue)
        self.covariance_history_rounds = int(covariance_history_rounds)
        self.round_weight_decay = float(round_weight_decay)
        self.fallback_diagonal_variance = float(fallback_diagonal_variance)
        self.min_full_covariance_elites = int(min_full_covariance_elites)
        covariance = np.asarray(initial_covariance, dtype=float)
        if covariance.shape != (ACTION_DIMENSION, ACTION_DIMENSION):
            raise ValueError(
                f"initial_covariance must have shape {(ACTION_DIMENSION, ACTION_DIMENSION)}."
            )
        self.covariance = self._repair_covariance(covariance)[0]
        self.rng = np.random.default_rng(seed)
        self.elite_history: list[tuple[int, list[LatentSample]]] = []
        self.last_update = DistributionUpdateDiagnostics(
            round_index=0,
            elite_count=0,
            covariance_source="initial",
            correction=None,
            log_determinant=float(np.linalg.slogdet(self.covariance)[1]),
            min_eigenvalue=float(np.min(np.linalg.eigvalsh(self.covariance))),
            max_eigenvalue=float(np.max(np.linalg.eigvalsh(self.covariance))),
            condition_number=float(np.linalg.cond(self.covariance)),
            eigenvalues=[float(value) for value in np.linalg.eigvalsh(self.covariance)],
            correlation_matrix=self._correlation_matrix(self.covariance).tolist(),
        )

    @staticmethod
    def _correlation_matrix(covariance: np.ndarray) -> np.ndarray:
        diagonal = np.sqrt(np.clip(np.diag(covariance), 1e-12, None))
        scale = np.outer(diagonal, diagonal)
        correlation = covariance / scale
        np.fill_diagonal(correlation, 1.0)
        return correlation

    def _repair_covariance(
        self,
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, str | None]:
        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        correction = None
        bounded = np.clip(
            eigenvalues,
            self.minimum_eigenvalue,
            self.maximum_eigenvalue,
        )
        if not np.allclose(eigenvalues, bounded):
            correction = "eigenvalue_clamped"
        repaired = eigenvectors @ np.diag(bounded) @ eigenvectors.T
        repaired = 0.5 * (repaired + repaired.T)
        return repaired, correction

    def state_dict(self) -> dict[str, Any]:
        return {
            "mu_z": self.mu_z.tolist(),
            "covariance": self.covariance.tolist(),
            "covariance_shrinkage": self.covariance_shrinkage,
            "covariance_smoothing": self.covariance_smoothing,
            "mean_smoothing": self.mean_smoothing,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "maximum_eigenvalue": self.maximum_eigenvalue,
            "covariance_history_rounds": self.covariance_history_rounds,
            "round_weight_decay": self.round_weight_decay,
            "fallback_diagonal_variance": self.fallback_diagonal_variance,
            "min_full_covariance_elites": self.min_full_covariance_elites,
            "rng_state": self.rng.bit_generator.state,
            "elite_history": [
                {
                    "round_index": round_index,
                    "samples": [asdict(sample) for sample in samples],
                }
                for round_index, samples in self.elite_history
            ],
            "last_update": asdict(self.last_update),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.mu_z = np.asarray(state["mu_z"], dtype=float)
        self.covariance = np.asarray(state["covariance"], dtype=float)
        self.rng.bit_generator.state = dict(state["rng_state"])
        self.elite_history = [
            (
                int(item["round_index"]),
                [LatentSample(**sample) for sample in item["samples"]],
            )
            for item in state.get("elite_history", [])
        ]
        self.last_update = DistributionUpdateDiagnostics(**state["last_update"])

    def mean_action(self) -> dict[str, float]:
        return vector_to_profile(sigmoid(self.mu_z))

    def action_standard_deviation(self) -> float:
        samples = sigmoid(
            self.rng.multivariate_normal(self.mu_z, self.covariance, size=256)
        )
        return float(np.mean(np.std(samples, axis=0, ddof=1)))

    def sample_batch(
        self,
        batch_size: int,
        *,
        round_index: int,
        prefix: str = "sample",
    ) -> list[LatentSample]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        latents = self.rng.multivariate_normal(
            self.mu_z,
            self.covariance,
            size=batch_size,
        )
        actions = sigmoid(latents)
        return [
            LatentSample(
                sample_id=f"{prefix}_{round_index:03d}_{index:03d}",
                round_index=round_index,
                action=vector_to_profile(actions[index]),
                latent=tuple(float(value) for value in latents[index]),
                reward=float("nan"),
                feasible=True,
                evaluation_consumed=False,
            )
            for index in range(batch_size)
        ]

    def _weighted_recent_elites(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        recent = self.elite_history[-self.covariance_history_rounds :]
        latents: list[np.ndarray] = []
        weights: list[float] = []
        for age, (_, samples) in enumerate(reversed(recent)):
            weight = self.round_weight_decay**age
            for sample in samples:
                latents.append(np.asarray(sample.latent, dtype=float))
                weights.append(weight)
        return np.asarray(latents, dtype=float), np.asarray(weights, dtype=float)

    @staticmethod
    def _weighted_covariance(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value.")
        normalized = weights / total
        mean = np.sum(values * normalized[:, None], axis=0)
        centered = values - mean
        return (centered * normalized[:, None]).T @ centered

    def update(
        self,
        samples: Sequence[LatentSample],
        *,
        round_index: int,
        elite_count: int,
    ) -> list[LatentSample]:
        require_reward_consumption(samples)
        feasible = [sample for sample in samples if sample.feasible]
        if not feasible:
            self.last_update = DistributionUpdateDiagnostics(
                round_index=round_index,
                elite_count=0,
                covariance_source="none",
                correction=None,
                log_determinant=float(np.linalg.slogdet(self.covariance)[1]),
                min_eigenvalue=float(np.min(np.linalg.eigvalsh(self.covariance))),
                max_eigenvalue=float(np.max(np.linalg.eigvalsh(self.covariance))),
                condition_number=float(np.linalg.cond(self.covariance)),
                eigenvalues=[float(value) for value in np.linalg.eigvalsh(self.covariance)],
                correlation_matrix=self._correlation_matrix(self.covariance).tolist(),
            )
            return []
        ranked = sorted(feasible, key=robust_rank_key)
        elites = ranked[: max(1, min(elite_count, len(ranked)))]
        self.elite_history.append((round_index, elites))
        if len(self.elite_history) > self.covariance_history_rounds:
            self.elite_history = self.elite_history[-self.covariance_history_rounds :]
        elite_latents = np.asarray([sample.latent for sample in elites], dtype=float)
        elite_mean = np.mean(elite_latents, axis=0)
        self.mu_z = (
            (1.0 - self.mean_smoothing) * self.mu_z
            + self.mean_smoothing * elite_mean
        )

        recent_latents, weights = self._weighted_recent_elites()
        covariance_source = "full"
        if len(recent_latents) < self.min_full_covariance_elites:
            covariance_source = "diagonal_fallback"
            variances = np.var(elite_latents, axis=0, ddof=1) if len(elites) > 1 else np.full(ACTION_DIMENSION, self.fallback_diagonal_variance, dtype=float)
            covariance_estimate = np.diag(np.maximum(variances, self.minimum_eigenvalue))
        else:
            covariance_estimate = self._weighted_covariance(recent_latents, weights)
        diagonal = np.diag(np.diag(covariance_estimate))
        shrunk = (
            (1.0 - self.covariance_shrinkage) * covariance_estimate
            + self.covariance_shrinkage * diagonal
            + self.minimum_eigenvalue * np.eye(ACTION_DIMENSION)
        )
        smoothed = (
            (1.0 - self.covariance_smoothing) * self.covariance
            + self.covariance_smoothing * shrunk
        )
        self.covariance, correction = self._repair_covariance(smoothed)
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        self.last_update = DistributionUpdateDiagnostics(
            round_index=round_index,
            elite_count=len(elites),
            covariance_source=covariance_source,
            correction=correction,
            log_determinant=float(np.linalg.slogdet(self.covariance)[1]),
            min_eigenvalue=float(np.min(eigenvalues)),
            max_eigenvalue=float(np.max(eigenvalues)),
            condition_number=float(np.linalg.cond(self.covariance)),
            eigenvalues=[float(value) for value in eigenvalues],
            correlation_matrix=self._correlation_matrix(self.covariance).tolist(),
        )
        return elites


class ContextualOuterLearner:
    def __init__(
        self,
        distributions: Mapping[str, LatentGaussianCEMDistribution],
    ) -> None:
        self.distributions = dict(distributions)

    def distribution(self, context: OuterLearningContext) -> LatentGaussianCEMDistribution:
        try:
            return self.distributions[context.key]
        except KeyError as exc:
            raise KeyError(f"Unknown context {context.key!r}.") from exc
