from pathlib import Path

import numpy as np
import pytest

from laban_rl.affect import VAD_TARGETS, target_vad, validate_vad
from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    PerceptualBanditEnvironment,
    PerceptualEvaluation,
)


class FixedEvaluator:
    def __init__(self, affect_ratings, probabilities):
        self.affect_ratings = affect_ratings
        self.probabilities = probabilities

    def evaluate(self, context, optimisation_result):
        return PerceptualEvaluation(
            affect_ratings=dict(self.affect_ratings),
            probabilities=dict(self.probabilities),
        )


def make_result(tmp_path: Path) -> LabanOptimisationResult:
    profile = {key: 0.5 for key in FEATURE_KEYS}
    q = np.zeros((20, 2), dtype=float)
    return LabanOptimisationResult(
        gesture="point",
        target_state="happiness",
        requested_profile=profile,
        achieved_profile=profile,
        achieved_profile_clipped=profile,
        inner_reward=0.0,
        inner_loss=0.0,
        action_coefficients=np.zeros(4),
        q_ref=q,
        q_var=q,
        output_dir=tmp_path,
        raw_result={
            "reward_info": {
                "path_length_ratio": 1.0,
                "joint_limit_error": 0.0,
            }
        },
    )


def test_vad_targets_are_complete_and_bounded():
    for state, values in VAD_TARGETS.items():
        assert validate_vad(values, name=state) == values
        assert target_vad(state) == values


def test_invalid_vad_is_rejected():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_vad({"valence": 1.2, "arousal": 0.5, "dominance": 0.5})


def test_vad_is_default_primary_reward(tmp_path):
    observed = {"valence": 0.70, "arousal": 0.55, "dominance": 0.45}
    probabilities = {
        "anger": 0.10,
        "disgust": 0.05,
        "fear": 0.05,
        "happiness": 0.10,
        "sadness": 0.70,
        "surprise": 0.00,
    }
    environment = PerceptualBanditEnvironment(
        evaluator=FixedEvaluator(observed, probabilities),
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=1,
            realisation_penalty_weight=0.0,
        ),
    )

    result = environment.step_from_result(
        context=Context("point", "happiness"),
        optimisation_result=make_result(tmp_path),
    )

    assert result.target_vad == VAD_TARGETS["happiness"]
    assert result.mean_observed_vad == observed
    assert result.mean_vad_error == pytest.approx(0.20)
    assert result.mean_vad_reward == pytest.approx(0.80)
    assert result.outer_reward == pytest.approx(0.80)
    assert result.categorical_reward == pytest.approx(-0.25)


def test_legacy_categorical_reward_remains_available(tmp_path):
    evaluator = FixedEvaluator(
        {"valence": 0.70, "arousal": 0.55, "dominance": 0.45},
        {
            "anger": 0.10,
            "disgust": 0.05,
            "fear": 0.05,
            "happiness": 0.10,
            "sadness": 0.70,
            "surprise": 0.00,
        },
    )
    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=1,
            perceptual_reward_mode="categorical",
            realisation_penalty_weight=0.0,
        ),
    )

    result = environment.step_from_result(
        context=Context("point", "happiness"),
        optimisation_result=make_result(tmp_path),
    )

    assert result.outer_reward == pytest.approx(-0.25)
    assert result.mean_vad_reward == pytest.approx(0.80)


def test_vad_only_evaluation_is_valid(tmp_path):
    environment = PerceptualBanditEnvironment(
        evaluator=FixedEvaluator(
            {"valence": 0.90, "arousal": 0.75, "dominance": 0.65},
            {},
        ),
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=1,
            realisation_penalty_weight=0.0,
        ),
    )

    result = environment.step_from_result(
        context=Context("point", "happiness"),
        optimisation_result=make_result(tmp_path),
    )

    assert result.outer_reward == pytest.approx(1.0)
    assert result.affective_evaluations
    assert not result.perceptual_evaluations


def test_categorical_mode_requires_probabilities(tmp_path):
    environment = PerceptualBanditEnvironment(
        evaluator=FixedEvaluator(
            {"valence": 0.90, "arousal": 0.75, "dominance": 0.65},
            {},
        ),
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=1,
            perceptual_reward_mode="categorical",
        ),
    )

    with pytest.raises(RuntimeError, match="failed on every repetition"):
        environment.step_from_result(
            context=Context("point", "happiness"),
            optimisation_result=make_result(tmp_path),
        )


def test_vad_weights_must_sum_to_one():
    config = EnvironmentRewardConfig(
        valence_weight=0.2,
        arousal_weight=0.2,
        dominance_weight=0.2,
    )
    with pytest.raises(ValueError, match="sum to 1.0"):
        config.validate()
