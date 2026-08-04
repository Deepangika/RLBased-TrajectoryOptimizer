import json
from pathlib import Path

import numpy as np
import pytest

from laban_rl.affect import VAD_TARGETS
from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    MockNoisyPerceptualEvaluator,
    PerceptualBanditEnvironment,
    PerceptualEvaluation,
)
from scripts.train_cem_contextual_bandit import (
    build_resume_config,
    context_from_args,
    nearest_vad_anchor,
    parse_args,
    validate_checkpoint_context,
    validate_resume_config,
)


DIRECT_VAD = {
    "valence": 0.37,
    "arousal": 0.62,
    "dominance": 0.48,
}


class FixedEvaluator:
    def evaluate(self, context, optimisation_result):
        return PerceptualEvaluation(
            affect_ratings=dict(DIRECT_VAD),
            probabilities={
                "anger": 0.10,
                "disgust": 0.10,
                "fear": 0.10,
                "happiness": 0.30,
                "sadness": 0.15,
                "surprise": 0.25,
            },
        )


def make_result(tmp_path: Path) -> LabanOptimisationResult:
    profile = {key: 0.5 for key in FEATURE_KEYS}
    q = np.zeros((20, 2), dtype=float)
    return LabanOptimisationResult(
        gesture="point",
        target_state="direct-vad-test",
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


def cli_args(*target_args: str):
    return [
        "--gesture",
        "point",
        *target_args,
        "--out",
        "outputs/test",
    ]


def test_named_target_resolves_anchor_and_preserves_legacy_constructor():
    context = Context("point", "happiness")

    assert context.target_mode == "named"
    assert context.target_state == "happiness"
    assert context.target_vad == VAD_TARGETS["happiness"]
    assert context.key == "point::happiness"


def test_explicit_target_is_validated_and_serializable():
    context = Context("point", target_vad=DIRECT_VAD)
    payload = context.to_dict()

    assert context.target_mode == "vad"
    assert context.target_state is None
    assert context.target_vad == DIRECT_VAD
    assert Context.from_dict(json.loads(json.dumps(payload))) == context


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "exactly one"),
        (
            {"target_state": "", "target_vad": DIRECT_VAD},
            "non-empty string",
        ),
        (
            {
                "target_vad": {
                    "valence": 1.1,
                    "arousal": 0.5,
                    "dominance": 0.5,
                }
            },
            r"\[0, 1\]",
        ),
        (
            {"target_vad": {"valence": 0.5, "arousal": 0.5}},
            "keys must be",
        ),
        (
            {
                "target_vad": {
                    "valence": None,
                    "arousal": 0.5,
                    "dominance": 0.5,
                }
            },
            "finite numbers",
        ),
    ],
)
def test_context_rejects_ambiguous_or_invalid_targets(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Context("point", **kwargs)


def test_direct_vad_scores_without_categorical_target_lookup(tmp_path):
    context = Context("point", target_vad=DIRECT_VAD)
    environment = PerceptualBanditEnvironment(
        evaluator=FixedEvaluator(),
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=1,
            realisation_penalty_weight=0.0,
        ),
    )

    result = environment.step_from_result(
        context=context,
        optimisation_result=make_result(tmp_path),
    )

    assert result.outer_reward == pytest.approx(1.0)
    assert result.target_vad == DIRECT_VAD
    assert result.mean_target_probability is None
    assert result.target_classification_rate is None
    assert result.winner_agreement_rate == pytest.approx(1.0)
    assert result.perceptual_evaluations
    assert result.to_dict()["context"] == context.to_dict()


def test_mock_evaluator_supports_direct_vad_context(tmp_path):
    context = Context("point", target_vad=DIRECT_VAD)
    environment = PerceptualBanditEnvironment(
        evaluator=MockNoisyPerceptualEvaluator(noise_std=0.0),
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=1,
            realisation_penalty_weight=0.0,
        ),
    )

    result = environment.step_from_result(
        context=context,
        optimisation_result=make_result(tmp_path),
    )

    assert result.mean_vad_reward is not None
    assert result.target_vad == DIRECT_VAD
    assert result.perceptual_evaluations


def test_categorical_mode_remains_named_only(tmp_path):
    environment = PerceptualBanditEnvironment(
        evaluator=FixedEvaluator(),
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=1,
            perceptual_reward_mode="categorical",
        ),
    )

    with pytest.raises(ValueError, match="requires a named target_state"):
        environment.step_from_result(
            context=Context("point", target_vad=DIRECT_VAD),
            optimisation_result=make_result(tmp_path),
        )


def test_cli_parses_named_and_explicit_modes():
    named = parse_args(cli_args("--target-state", "happiness"))
    direct = parse_args(
        cli_args(
            "--target-valence",
            "0.37",
            "--target-arousal",
            "0.62",
            "--target-dominance",
            "0.48",
            "--evaluator",
            "mock",
        )
    )

    assert context_from_args(named).target_vad == VAD_TARGETS["happiness"]
    assert context_from_args(direct).target_vad == DIRECT_VAD


@pytest.mark.parametrize(
    "target_args",
    [
        (),
        ("--target-valence", "0.2"),
        (
            "--target-state",
            "happiness",
            "--target-valence",
            "0.2",
            "--target-arousal",
            "0.3",
            "--target-dominance",
            "0.4",
        ),
        (
            "--target-valence",
            "0.2",
            "--target-arousal",
            "0.3",
            "--target-dominance",
            "0.4",
            "--perceptual-reward-mode",
            "categorical",
        ),
        (
            "--target-valence",
            "-0.1",
            "--target-arousal",
            "0.3",
            "--target-dominance",
            "0.4",
        ),
    ],
)
def test_cli_rejects_incomplete_ambiguous_or_invalid_modes(target_args):
    with pytest.raises(SystemExit):
        parse_args(cli_args(*target_args))


def test_checkpoint_context_accepts_legacy_named_metadata():
    current = Context("point", "happiness")

    validate_checkpoint_context(
        {"gesture": "point", "target_state": "happiness"},
        current,
    )


def test_checkpoint_context_rejects_different_direct_target():
    current = Context("point", target_vad=DIRECT_VAD)

    with pytest.raises(RuntimeError, match="does not match"):
        validate_checkpoint_context(
            {
                "gesture": "point",
                "target_mode": "vad",
                "target_state": None,
                "target_vad": {
                    "valence": 0.38,
                    "arousal": 0.62,
                    "dominance": 0.48,
                },
            },
            current,
        )


def test_resume_config_rejects_evaluator_or_reward_changes():
    mock_args = parse_args(
        cli_args(
            "--target-valence",
            "0.37",
            "--target-arousal",
            "0.62",
            "--target-dominance",
            "0.48",
            "--evaluator",
            "mock",
        )
    )
    gemini_args = parse_args(
        cli_args(
            "--target-valence",
            "0.37",
            "--target-arousal",
            "0.62",
            "--target-dominance",
            "0.48",
        )
    )
    saved = build_resume_config(mock_args)

    validate_resume_config(saved, build_resume_config(mock_args))
    with pytest.raises(RuntimeError, match="does not match"):
        validate_resume_config(saved, build_resume_config(gemini_args))
    with pytest.raises(RuntimeError, match="predates complete"):
        validate_resume_config(None, saved)


def test_direct_vad_uses_nearest_named_anchor_only_for_initialization():
    context = Context(
        "point",
        target_vad={"valence": 0.89, "arousal": 0.74, "dominance": 0.66},
    )

    assert nearest_vad_anchor(context) == "happiness"
    assert context.target_state is None
