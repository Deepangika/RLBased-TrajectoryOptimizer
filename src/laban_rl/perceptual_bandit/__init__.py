"""Outer perceptual contextual-bandit environment components."""

from .environment import (
    Context,
    EnvironmentRewardConfig,
    EnvironmentStepResult,
    MockNoisyPerceptualEvaluator,
    PerceptualBanditEnvironment,
    PerceptualEvaluation,
)
from laban_rl.affect import VAD_KEYS, VAD_TARGETS
from .evaluation_cache import (
    CacheCompatibilityError,
    EvaluatorCacheIdentity,
    PerceptualObservationCache,
)
from .experiment import (
    ExperimentCase,
    cases_from_matrix,
    paired_comparison_summary,
    run_paired_experiment,
)
from .scoring import score_perceptual_observations, test_retest_reliability

__all__ = [
    "Context",
    "EnvironmentRewardConfig",
    "EnvironmentStepResult",
    "MockNoisyPerceptualEvaluator",
    "PerceptualBanditEnvironment",
    "PerceptualEvaluation",
    "VAD_KEYS",
    "VAD_TARGETS",
    "CacheCompatibilityError",
    "EvaluatorCacheIdentity",
    "PerceptualObservationCache",
    "ExperimentCase",
    "cases_from_matrix",
    "paired_comparison_summary",
    "run_paired_experiment",
    "score_perceptual_observations",
    "test_retest_reliability",
]
