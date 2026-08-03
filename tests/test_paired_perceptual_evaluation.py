import json
from pathlib import Path

import numpy as np
import pytest

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult, build_reference_motion
from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    PerceptualBanditEnvironment,
    PerceptualEvaluation,
)
from laban_rl.perceptual_bandit.evaluation_cache import (
    CacheCompatibilityError,
    EvaluatorCacheIdentity,
    PerceptualObservationCache,
)
from laban_rl.perceptual_bandit.experiment import (
    ExperimentCase,
    cases_from_matrix,
    paired_comparison_summary,
    run_paired_experiment,
)
from laban_rl.perceptual_bandit.gemini_evaluator import _retry_message
from laban_rl.perceptual_bandit.scoring import (
    score_perceptual_observations,
    test_retest_reliability as compute_test_retest_reliability,
)


PROBABILITIES = {
    "anger": 0.05,
    "disgust": 0.05,
    "fear": 0.05,
    "happiness": 0.70,
    "sadness": 0.05,
    "surprise": 0.10,
}


def make_result(path: Path, *, offset: float = 0.0) -> LabanOptimisationResult:
    profile = {key: 0.5 for key in FEATURE_KEYS}
    q_ref = np.zeros((20, 2), dtype=float)
    q_var = q_ref + offset
    return LabanOptimisationResult(
        gesture="point",
        target_state="happiness",
        requested_profile=dict(profile),
        achieved_profile=dict(profile),
        achieved_profile_clipped=dict(profile),
        inner_reward=0.0,
        inner_loss=0.0,
        action_coefficients=np.zeros(4),
        q_ref=q_ref,
        q_var=q_var,
        output_dir=path,
        raw_result={
            "reward_info": {
                "path_length_ratio": 1.0,
                "joint_limit_error": 0.0,
            }
        },
    )


class CountingEvaluator:
    def __init__(self, *, schema="v1", fail_call=None):
        self.schema = schema
        self.fail_call = fail_call
        self.calls = 0

    def cache_identity(self):
        return {
            "provider": "test",
            "model": "fixed",
            "prompt_version": "prompt-v1",
            "schema_version": self.schema,
            "settings": {"temperature": 0.2},
        }

    def evaluate(self, context, optimisation_result):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("interrupted")
        delta = self.calls * 0.01
        return PerceptualEvaluation(
            affect_ratings={
                "valence": 0.8 + delta,
                "arousal": 0.7,
                "dominance": 0.6,
            },
            probabilities=dict(PROBABILITIES),
            confidence=0.75,
            perceived_state="happiness",
            reasoning_summary="fixed test observation",
        )


class SequenceEvaluator:
    def __init__(self, observations):
        self.observations = list(observations)
        self.index = 0

    def evaluate(self, context, optimisation_result):
        observation = self.observations[self.index]
        self.index += 1
        return observation


def test_cache_reuses_repeats_and_invalidates_schema(tmp_path):
    cache = PerceptualObservationCache(tmp_path / "cache")
    context = Context("point", "happiness")
    result = make_result(tmp_path / "clip")
    evaluator = CountingEvaluator()

    first = cache.collect(
        context=context, result=result, evaluator=evaluator, repeats=3
    )
    second = cache.collect(
        context=context, result=result, evaluator=evaluator, repeats=3
    )
    extended = cache.collect(
        context=context, result=result, evaluator=evaluator, repeats=5
    )

    assert first == second
    assert extended[:3] == first
    assert evaluator.calls == 5

    incompatible = CountingEvaluator(schema="v2")
    cache.collect(
        context=context, result=result, evaluator=incompatible, repeats=1
    )
    assert incompatible.calls == 1

    changed_profile = make_result(tmp_path / "clip-two")
    changed_profile.achieved_profile["weight"] = 0.75
    assert cache.cache_key(
        context=context,
        result=result,
        evaluator_identity=EvaluatorCacheIdentity.from_evaluator(evaluator),
    ) != cache.cache_key(
        context=context,
        result=changed_profile,
        evaluator_identity=EvaluatorCacheIdentity.from_evaluator(evaluator),
    )


def test_gemini_retry_message_is_windows_console_safe():
    message = _retry_message(2.0, 0, 5)

    assert "Retrying in 2.0s" in message
    message.encode("cp1252")


def test_explicit_matrix_preserves_pilot_layers_and_pair_settings():
    profile = {key: 0.5 for key in FEATURE_KEYS}
    config = {
        "cases": [
            {
                "id": "wave-anger-projected",
                "gesture": "wave",
                "target": {"state": "anger"},
                "seed": 7,
                "target_layers": {
                    "original_affect_derived_laban_target": profile,
                    "projected_feasible_laban_target": profile,
                },
                "optimizer_overrides": {"maxiter": 75},
                "candidate_profiles": {
                    "reference": {
                        "motion_source": "reference",
                        "profile": profile,
                    },
                    "styled": {"profile": profile},
                },
            }
        ]
    }

    cases = cases_from_matrix(config)

    assert [case.candidate_name for case in cases] == ["reference", "styled"]
    assert cases[0].motion_source == "reference"
    assert cases[0].require_feature_match is False
    assert cases[1].require_feature_match is True
    assert cases[1].optimizer_overrides == {"maxiter": 75}
    assert cases[1].target_layers["projected_feasible_laban_target"] == profile


def test_reference_motion_is_an_exact_matched_baseline(tmp_path):
    result = build_reference_motion(
        gesture="wave",
        target_state="surprise",
        out_dir=tmp_path / "reference",
    )

    assert np.array_equal(result.q_ref, result.q_var)
    assert result.raw_result["motion_source"] == "reference"
    assert result.raw_result["reward_info"]["path_length_ratio"] == 1.0


def test_cache_rejects_tampered_entry_and_secret_identity(tmp_path):
    cache = PerceptualObservationCache(tmp_path / "cache")
    context = Context("point", "happiness")
    result = make_result(tmp_path / "clip")
    evaluator = CountingEvaluator()
    cache.collect(context=context, result=result, evaluator=evaluator, repeats=1)
    identity = EvaluatorCacheIdentity.from_evaluator(evaluator)
    key = cache.cache_key(
        context=context, result=result, evaluator_identity=identity
    )
    path = cache.root / key[:2] / f"{key}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["identity"]["evaluator"]["schema_version"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CacheCompatibilityError, match="incompatible"):
        cache.collect(
            context=context, result=result, evaluator=evaluator, repeats=1
        )

    with pytest.raises(ValueError, match="must not contain secrets"):
        EvaluatorCacheIdentity(
            provider="test",
            model="fixed",
            prompt_version="v1",
            schema_version="v1",
            settings={"api_key": "do-not-cache"},
        ).to_dict()


def test_pure_scoring_matches_environment_reward(tmp_path):
    observations = [
        PerceptualEvaluation(
            affect_ratings={
                "valence": 0.80,
                "arousal": 0.70,
                "dominance": 0.60,
            },
            probabilities=dict(PROBABILITIES),
            confidence=0.8,
        ),
        PerceptualEvaluation(
            affect_ratings={
                "valence": 0.90,
                "arousal": 0.80,
                "dominance": 0.70,
            },
            probabilities=dict(PROBABILITIES),
            confidence=0.7,
        ),
    ]
    config = EnvironmentRewardConfig(
        repeat_evaluations=2,
        realisation_penalty_weight=0.0,
    )
    context = Context("point", "happiness")
    environment_result = PerceptualBanditEnvironment(
        evaluator=SequenceEvaluator(observations),
        reward_config=config,
    ).step_from_result(
        context=context,
        optimisation_result=make_result(tmp_path / "clip"),
    )
    pure = score_perceptual_observations(
        observations,
        target_vad=context.target_vad or {},
        target_state=context.target_state,
        reward_config=config,
    )

    assert environment_result.outer_reward == pytest.approx(
        pure["mean_vad_reward"]
    )
    assert environment_result.categorical_reward == pytest.approx(
        pure["categorical_reward"]
    )
    assert environment_result.per_axis_vad_std == pure["per_axis_vad_std"]
    assert environment_result.mean_confidence == pytest.approx(0.75)


def test_reliability_reports_low_repeat_and_degenerate_cases():
    observation = {
        "affect_ratings": {
            "valence": 0.5,
            "arousal": 0.5,
            "dominance": 0.5,
        }
    }
    low = compute_test_retest_reliability([[observation], [observation]])
    degenerate = compute_test_retest_reliability(
        [[observation, observation], [observation, observation]]
    )

    assert low["status"] == "insufficient_repeats"
    assert all(value is None for value in low["icc"].values())
    assert degenerate["status"] == "partial"
    assert all(value is None for value in degenerate["icc"].values())
    assert set(degenerate["axis_status"].values()) == {
        "degenerate_zero_variance"
    }


def test_incomplete_optional_categories_are_not_rankable():
    config = EnvironmentRewardConfig(repeat_evaluations=2)
    observations = [
        PerceptualEvaluation(
            affect_ratings={
                "valence": 0.8,
                "arousal": 0.7,
                "dominance": 0.6,
            },
            probabilities=dict(PROBABILITIES),
        ),
        PerceptualEvaluation(
            affect_ratings={
                "valence": 0.8,
                "arousal": 0.7,
                "dominance": 0.6,
            }
        ),
    ]

    score = score_perceptual_observations(
        observations,
        target_vad=Context("point", "happiness").target_vad or {},
        target_state="happiness",
        reward_config=config,
    )

    assert score["categorical_coverage"] == pytest.approx(0.5)
    assert score["categorical_complete"] is False
    assert score["categorical_reward"] is None


def test_interrupted_matrix_resumes_from_repeat_cache(tmp_path):
    context = Context("point", "happiness")
    cases = [
        ExperimentCase(
            candidate_id=f"candidate-{index}",
            candidate_name=f"profile-{index}",
            context=context,
            profile={key: 0.5 for key in FEATURE_KEYS},
            seed=7,
        )
        for index in range(2)
    ]
    inner_calls = []

    def inner_runner(case, out_dir):
        inner_calls.append(case.candidate_id)
        index = int(case.candidate_id.rsplit("-", 1)[1])
        return make_result(out_dir, offset=0.01 * (index + 1))

    cache = PerceptualObservationCache(tmp_path / "cache")
    config = EnvironmentRewardConfig(repeat_evaluations=2)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_paired_experiment(
            cases,
            inner_runner=inner_runner,
            evaluator=CountingEvaluator(fail_call=4),
            cache=cache,
            repeats=2,
            reward_config=config,
            out_dir=tmp_path / "run",
            runner_identity={"optimizer": "test-v1"},
            save_plots=False,
        )

    resumed_evaluator = CountingEvaluator()
    result = run_paired_experiment(
        cases,
        inner_runner=inner_runner,
        evaluator=resumed_evaluator,
        cache=cache,
        repeats=2,
        reward_config=config,
        out_dir=tmp_path / "run",
        runner_identity={"optimizer": "test-v1"},
        save_plots=False,
    )

    assert result["status"] == "complete"
    assert len(result["records"]) == 2
    assert resumed_evaluator.calls == 1
    assert (tmp_path / "run" / "paired_results.csv").exists()

    with pytest.raises(ValueError, match="incompatible"):
        run_paired_experiment(
            cases,
            inner_runner=inner_runner,
            evaluator=CountingEvaluator(),
            cache=cache,
            repeats=2,
            reward_config=config,
            out_dir=tmp_path / "run",
            runner_identity={"optimizer": "test-v2"},
            save_plots=False,
        )


def test_matrix_skips_physically_invalid_candidate_evaluation(tmp_path):
    context = Context("point", "happiness")
    case = ExperimentCase(
        candidate_id="invalid",
        candidate_name="invalid",
        context=context,
        profile={key: 0.5 for key in FEATURE_KEYS},
        seed=7,
    )
    evaluator = CountingEvaluator()

    def inner_runner(case, out_dir):
        result = make_result(out_dir)
        result.raw_result["reward_info"]["path_length_ratio"] = 2.0
        return result

    payload = run_paired_experiment(
        [case],
        inner_runner=inner_runner,
        evaluator=evaluator,
        cache=PerceptualObservationCache(tmp_path / "cache"),
        repeats=2,
        reward_config=EnvironmentRewardConfig(repeat_evaluations=2),
        out_dir=tmp_path / "invalid-run",
        runner_identity={"optimizer": "test-v1"},
        save_plots=False,
    )

    assert evaluator.calls == 0
    assert payload["records"][0]["observations"] == []
    assert payload["records"][0]["reliability"] is None


def test_strict_matrix_skips_feature_miss_but_keeps_reference(tmp_path):
    context = Context("point", "happiness")
    profile = {key: 0.5 for key in FEATURE_KEYS}
    cases = [
        ExperimentCase(
            candidate_id="styled",
            candidate_name="styled",
            context=context,
            profile=profile,
            seed=7,
        ),
        ExperimentCase(
            candidate_id="reference",
            candidate_name="reference",
            context=context,
            profile=profile,
            seed=7,
            motion_source="reference",
            require_feature_match=False,
        ),
    ]
    evaluator = CountingEvaluator()

    def inner_runner(case, out_dir):
        result = make_result(out_dir)
        result.achieved_profile["weight"] = 0.75
        return result

    payload = run_paired_experiment(
        cases,
        inner_runner=inner_runner,
        evaluator=evaluator,
        cache=PerceptualObservationCache(tmp_path / "cache"),
        repeats=2,
        reward_config=EnvironmentRewardConfig(
            repeat_evaluations=2,
            reject_excessive_feature_error=True,
        ),
        out_dir=tmp_path / "strict-run",
        runner_identity={"optimizer": "test-v1"},
        save_plots=False,
    )

    assert evaluator.calls == 2
    assert payload["records"][0]["observations"] == []
    assert len(payload["records"][1]["observations"]) == 2
    assert payload["records"][1]["feasibility"]["feasible"] is True
    assert payload["records"][1]["vad_score"] == pytest.approx(
        payload["records"][1]["reliability"]["mean_vad_reward"]
    )


def test_reliability_deduplicates_identical_cached_clips(tmp_path):
    context = Context("point", "happiness")
    profile = {key: 0.5 for key in FEATURE_KEYS}
    cases = [
        ExperimentCase(
            candidate_id=f"reference-{index}",
            candidate_name="reference",
            context=context,
            profile=profile,
            seed=7,
            motion_source="reference",
            require_feature_match=False,
        )
        for index in range(2)
    ]
    evaluator = CountingEvaluator()

    payload = run_paired_experiment(
        cases,
        inner_runner=lambda case, out_dir: make_result(out_dir),
        evaluator=evaluator,
        cache=PerceptualObservationCache(tmp_path / "cache"),
        repeats=2,
        reward_config=EnvironmentRewardConfig(repeat_evaluations=2),
        out_dir=tmp_path / "deduplicated-run",
        runner_identity={"optimizer": "test-v1"},
        save_plots=False,
    )

    assert evaluator.calls == 2
    assert (
        payload["records"][0]["observation_cache_key"]
        == payload["records"][1]["observation_cache_key"]
    )
    assert payload["test_retest_reliability"]["clip_count"] == 1


def test_limited_gemini_pilot_matrix_cardinality_and_metadata():
    path = Path("configs/limited_gemini_pilot.json")
    matrix = json.loads(path.read_text(encoding="utf-8"))
    cases = cases_from_matrix(matrix)

    assert len(matrix["cases"]) == 9
    assert len(cases) == 18
    assert {case.candidate_name for case in cases} == {"reference", "styled"}
    assert matrix["unique_video_count"] == 15
    assert matrix["planned_evaluator_calls"] == 45
    assert all(
        "original_affect_derived_laban_target" in case.target_layers
        for case in cases
    )
    beckon_styled = next(
        case
        for case in cases
        if case.candidate_id.startswith("beckon-disgust")
        and case.candidate_name == "styled"
    )
    assert beckon_styled.optimizer_overrides["flow_boundness_target_weight"] == 0.2


def test_paired_ranking_compares_order_and_selected_identity():
    context = Context("point", "happiness").to_dict()
    records = [
        {
            "candidate_id": "a",
            "context": context,
            "seed": 7,
            "vad_score": 0.9,
            "categorical_score": 0.2,
            "feasibility": {"feasible": True},
            "reliability": {"categorical_complete": True},
        },
        {
            "candidate_id": "b",
            "context": context,
            "seed": 7,
            "vad_score": 0.7,
            "categorical_score": 0.8,
            "feasibility": {"feasible": True},
            "reliability": {"categorical_complete": True},
        },
    ]

    summary = paired_comparison_summary(records)
    group = summary["groups"][0]

    assert group["pairwise_ranking_agreement"] == 0.0
    assert group["vad_selected"] == "a"
    assert group["categorical_selected"] == "b"
    assert group["same_selected_candidate"] is False
    assert "not directly comparable" in summary["reward_scale_note"]
