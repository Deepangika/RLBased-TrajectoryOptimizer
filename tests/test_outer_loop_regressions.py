from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    PerceptualBanditEnvironment,
    PerceptualEvaluation,
)
from laban_rl.perceptual_bandit.evaluation_cache import PerceptualObservationCache
from laban_rl.perceptual_bandit.selection import (
    select_feasible_incumbent,
    strict_realisability,
)
from scripts.train_cem_contextual_bandit import (
    safe_output_folder,
    save_checkpoint_atomic,
    update_cem_from_feasible_candidates,
)


def make_result(path: Path) -> LabanOptimisationResult:
    profile = {key: 0.5 for key in FEATURE_KEYS}
    return LabanOptimisationResult(
        gesture="point",
        target_state="happiness",
        requested_profile=profile,
        achieved_profile=profile,
        achieved_profile_clipped=profile,
        inner_reward=0.0,
        inner_loss=0.0,
        action_coefficients=np.zeros(4),
        q_ref=np.zeros((20, 2)),
        q_var=np.zeros((20, 2)),
        output_dir=path,
        raw_result={
            "reward_info": {
                "path_length_ratio": 1.0,
                "endpoint_error": 0.0,
                "direction_error": 0.0,
                "joint_limit_error": 0.0,
            }
        },
    )


class InterruptOnceEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def cache_identity(self):
        return {
            "provider": "test",
            "model": "fixed",
            "prompt_version": "v1",
            "schema_version": "v1",
            "settings": {},
        }

    def evaluate(self, context, optimisation_result):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("transient interruption")
        return fixed_evaluation()


class CountingEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    def cache_identity(self):
        return {
            "provider": "test",
            "model": "counting",
            "prompt_version": "v1",
            "schema_version": "v1",
            "settings": {},
        }

    def evaluate(self, context, optimisation_result):
        self.calls += 1
        return fixed_evaluation()


def fixed_evaluation() -> PerceptualEvaluation:
    return PerceptualEvaluation(
        affect_ratings={
            "valence": 0.8,
            "arousal": 0.7,
            "dominance": 0.6,
        },
        probabilities={
            "anger": 0.05,
            "disgust": 0.05,
            "fear": 0.05,
            "happiness": 0.70,
            "sadness": 0.05,
            "surprise": 0.10,
        },
    )


class FakeCEM:
    min_elites = 2

    def __init__(self) -> None:
        self.updated = False
        self.decayed = False

    def update_elites(self, candidates, *, elite_fraction):
        self.updated = True
        self.candidates = list(candidates)

    def decay_exploration(self, rate):
        self.decayed = True


def test_resume_requires_explicit_flag(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "latest_checkpoint.pt").write_bytes(b"checkpoint")

    with pytest.raises(RuntimeError, match="--resume"):
        safe_output_folder(output, overwrite=False, resume=False)
    assert safe_output_folder(output, overwrite=False, resume=True)


def test_explicit_resume_preserves_precheckpoint_cache(tmp_path):
    output = tmp_path / "run"
    cache_entry = output / "perceptual_observation_cache" / "training" / "entry.json"
    cache_entry.parent.mkdir(parents=True)
    cache_entry.write_text("cached", encoding="utf-8")
    (output / "target_context.json").write_text("{}", encoding="utf-8")
    (output / "resume_config.json").write_text("{}", encoding="utf-8")

    assert not safe_output_folder(output, overwrite=False, resume=True)
    assert cache_entry.read_text(encoding="utf-8") == "cached"


def test_resume_rejects_non_run_directory(tmp_path):
    output = tmp_path / "unrelated"
    output.mkdir()
    (output / "notes.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not a recoverable run"):
        safe_output_folder(output, overwrite=False, resume=True)


def test_checkpoint_save_atomically_replaces_latest_file(tmp_path):
    checkpoint_path = tmp_path / "latest_checkpoint.pt"
    checkpoint_path.write_bytes(b"old")

    save_checkpoint_atomic({"round": 2}, checkpoint_path)

    import pickle

    with checkpoint_path.open("rb") as handle:
        assert pickle.load(handle) == {"round": 2}
    assert not checkpoint_path.with_suffix(".pt.tmp").exists()


def test_insufficient_feasible_elites_skip_update_and_decay():
    cem = FakeCEM()
    updated = update_cem_from_feasible_candidates(
        cem,
        [({key: 0.5 for key in FEATURE_KEYS}, 0.8)],
        elite_fraction=0.25,
        exploration_decay_rate=0.9,
        feasible_improved=True,
    )
    assert not updated
    assert not cem.updated
    assert not cem.decayed


def test_exploration_decays_only_after_feasible_improvement():
    candidates = [
        ({key: value for key in FEATURE_KEYS}, reward)
        for value, reward in ((0.4, 0.7), (0.6, 0.8))
    ]
    no_improvement = FakeCEM()
    assert update_cem_from_feasible_candidates(
        no_improvement,
        candidates,
        elite_fraction=0.25,
        exploration_decay_rate=0.9,
        feasible_improved=False,
    )
    assert no_improvement.updated
    assert not no_improvement.decayed

    improvement = FakeCEM()
    assert update_cem_from_feasible_candidates(
        improvement,
        candidates,
        elite_fraction=0.25,
        exploration_decay_rate=0.9,
        feasible_improved=True,
    )
    assert improvement.updated
    assert improvement.decayed


def test_strict_feasibility_requires_complete_repeats():
    result = {
        "valid_realisation": True,
        "physically_acceptable": True,
        "feature_realisation_acceptable": True,
        "realisation_rmse": 0.02,
        "max_abs_feature_error": 0.04,
        "affective_evaluations": [{}, {}],
    }
    feasible, reasons = strict_realisability(
        result, tolerance=0.10, required_repeats=3
    )
    assert not feasible
    assert "incomplete_perceptual_repeats:2/3" in reasons


def test_final_selection_enforces_validation_repeat_count():
    result = {
        "valid_realisation": True,
        "physically_acceptable": True,
        "feature_realisation_acceptable": True,
        "realisation_rmse": 0.02,
        "max_abs_feature_error": 0.04,
        "outer_reward": 0.8,
        "requested_profile": {key: 0.5 for key in FEATURE_KEYS},
        "affective_evaluations": [{}, {}],
    }
    with pytest.raises(RuntimeError, match="No independently validated candidate"):
        select_feasible_incumbent(
            {"initial_profile": result},
            [],
            tolerance=0.10,
            required_repeats=3,
        )


def test_cached_retry_completes_only_missing_repeats(tmp_path):
    evaluator = InterruptOnceEvaluator()
    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=3,
            realisation_penalty_weight=0.0,
            evaluator_failure_mode="raise",
        ),
        observation_cache=PerceptualObservationCache(tmp_path / "cache"),
    )
    context = Context("point", "happiness")
    result = make_result(tmp_path / "clip")

    with pytest.raises(RuntimeError, match="every requested repetition"):
        environment.step_from_result(
            context=context,
            optimisation_result=result,
        )

    completed = environment.step_from_result(
        context=context,
        optimisation_result=result,
    )
    assert len(completed.affective_evaluations) == 3
    assert evaluator.calls == 4


def test_incomplete_repeat_set_without_cache_is_not_scored(tmp_path):
    evaluator = InterruptOnceEvaluator()
    environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=3,
            evaluator_failure_mode="raise",
        ),
    )
    with pytest.raises(RuntimeError, match="every requested repetition"):
        environment.step_from_result(
            context=Context("point", "happiness"),
            optimisation_result=make_result(tmp_path / "clip"),
        )


def test_training_and_validation_caches_are_independent(tmp_path):
    evaluator = CountingEvaluator()
    context = Context("point", "happiness")
    result = make_result(tmp_path / "clip")
    training_environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(repeat_evaluations=2),
        observation_cache=PerceptualObservationCache(tmp_path / "cache" / "training"),
    )
    validation_environment = PerceptualBanditEnvironment(
        evaluator=evaluator,
        reward_config=EnvironmentRewardConfig(repeat_evaluations=3),
        observation_cache=PerceptualObservationCache(tmp_path / "cache" / "validation"),
    )

    training_environment.step_from_result(
        context=context,
        optimisation_result=result,
    )
    validation_environment.step_from_result(
        context=context,
        optimisation_result=result,
    )
    assert evaluator.calls == 5


def test_context_result_mismatch_is_rejected(tmp_path):
    environment = PerceptualBanditEnvironment(
        evaluator=InterruptOnceEvaluator(),
        reward_config=EnvironmentRewardConfig(repeat_evaluations=1),
    )
    with pytest.raises(ValueError, match="gesture does not match"):
        environment.step_from_result(
            context=Context("wave", "happiness"),
            optimisation_result=make_result(tmp_path / "clip"),
        )
