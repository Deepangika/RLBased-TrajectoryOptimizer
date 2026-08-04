import csv
import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from laban_rl.perceptual_bandit.baseline import guard_baseline_output_path
from laban_rl.perceptual_bandit.outer_learning import (
    ContextualOuterLearner,
    LatentGaussianCEMDistribution,
    LatentSample,
    OuterLearningContext,
    classify_outcome,
    penalized_vad_reward,
    require_reward_consumption,
    stopping_reason,
    weighted_vad_reward,
)
from laban_rl.perceptual_bandit.selection import strict_realisability


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluation"
    / "run_outer_learning_experiment.py"
)
_RUNNER_SPEC = importlib.util.spec_from_file_location("outer_learning_runner", RUNNER_PATH)
assert _RUNNER_SPEC is not None
assert _RUNNER_SPEC.loader is not None
outer_learning_runner = importlib.util.module_from_spec(_RUNNER_SPEC)
_RUNNER_SPEC.loader.exec_module(outer_learning_runner)


def _initial_profile(value: float = 0.55) -> dict[str, float]:
    return {
        "weight": value,
        "time": value,
        "flow_boundness": value,
        "space_indirectness": value,
        "shape_arcness": value,
    }


def _distribution(seed: int = 7) -> LatentGaussianCEMDistribution:
    return LatentGaussianCEMDistribution(
        initial_action_mean=_initial_profile(),
        initial_covariance=np.array(
            [
                [0.10, 0.05, 0.02, 0.00, 0.01],
                [0.05, 0.10, 0.03, 0.00, 0.01],
                [0.02, 0.03, 0.10, 0.02, 0.00],
                [0.00, 0.00, 0.02, 0.10, 0.04],
                [0.01, 0.01, 0.00, 0.04, 0.10],
            ],
            dtype=float,
        ),
        seed=seed,
    )


def _completed_samples(
    distribution: LatentGaussianCEMDistribution,
    *,
    round_index: int,
    rewards: list[float],
    feasible: list[bool] | None = None,
) -> list[LatentSample]:
    samples = distribution.sample_batch(
        len(rewards),
        round_index=round_index,
        prefix="test",
    )
    if feasible is None:
        feasible = [True] * len(rewards)
    completed = []
    for sample, reward, allowed in zip(samples, rewards, feasible):
        completed.append(
            LatentSample(
                sample_id=sample.sample_id,
                round_index=sample.round_index,
                action=sample.action,
                latent=sample.latent,
                reward=reward,
                feasible=allowed,
                evaluation_consumed=True,
            )
        )
    return completed


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_context_distributions_are_independent():
    first = _distribution(seed=7)
    second = _distribution(seed=11)
    learner = ContextualOuterLearner(
        {
            "wave::anger": first,
            "point::anger": second,
        }
    )
    before = second.mean_action()
    learner.distribution(OuterLearningContext("wave", "anger")).update(
        _completed_samples(first, round_index=1, rewards=[0.9, 0.8, 0.7, 0.6]),
        round_index=1,
        elite_count=2,
    )
    assert learner.distribution(OuterLearningContext("point", "anger")).mean_action() == before


def test_sampling_stays_strictly_inside_unit_interval():
    samples = _distribution().sample_batch(256, round_index=1)
    for sample in samples:
        assert all(0.0 < value < 1.0 for value in sample.action.values())


def test_full_covariance_sampling_produces_correlation():
    distribution = _distribution()
    samples = distribution.sample_batch(512, round_index=1)
    values = np.asarray(
        [[sample.action["weight"], sample.action["time"]] for sample in samples],
        dtype=float,
    )
    correlation = np.corrcoef(values.T)[0, 1]
    assert correlation > 0.15


def test_covariance_update_stays_symmetric_positive_definite():
    distribution = _distribution()
    distribution.update(
        _completed_samples(
            distribution,
            round_index=1,
            rewards=[0.9, 0.8, 0.7, 0.6, 0.5],
        ),
        round_index=1,
        elite_count=3,
    )
    assert np.allclose(distribution.covariance, distribution.covariance.T)
    assert np.all(np.linalg.eigvalsh(distribution.covariance) > 0.0)


def test_low_sample_fallback_uses_diagonal_covariance():
    distribution = _distribution()
    distribution.update(
        _completed_samples(
            distribution,
            round_index=1,
            rewards=[0.9, 0.8],
        ),
        round_index=1,
        elite_count=2,
    )
    assert distribution.last_update.covariance_source == "diagonal_fallback"


def test_elite_selection_changes_mean_and_covariance():
    distribution = _distribution()
    before_mean = distribution.mean_action()
    before_covariance = distribution.covariance.copy()
    distribution.update(
        _completed_samples(
            distribution,
            round_index=1,
            rewards=[0.1, 0.2, 0.9, 0.8, 0.3],
        ),
        round_index=1,
        elite_count=2,
    )
    assert distribution.mean_action() != before_mean
    assert not np.allclose(distribution.covariance, before_covariance)


def test_reward_consumption_is_mandatory():
    distribution = _distribution()
    samples = distribution.sample_batch(3, round_index=1)
    with pytest.raises(RuntimeError, match="not consumed"):
        require_reward_consumption(samples)


def test_weighted_reward_matches_specification():
    observed = {"valence": 0.2, "arousal": 0.8, "dominance": 0.6}
    target = {"valence": 0.1, "arousal": 0.9, "dominance": 0.5}
    reward = weighted_vad_reward(observed, target)
    assert reward == pytest.approx(1.0 - (0.2 * 0.1 + 0.4 * 0.1 + 0.4 * 0.1))
    penalized = penalized_vad_reward(observed, target, realisation_error=0.2)
    assert penalized == pytest.approx(reward - 0.25 * 0.2)


def test_plateau_and_convergence_stopping_rules():
    assert (
        stopping_reason(
            round_index=5,
            minimum_rounds=5,
            maximum_rounds=8,
            plateau_rounds=3,
            plateau_patience=3,
            current_action_std=0.10,
            maximum_action_std_for_convergence=0.04,
            feasible_candidates=3,
        )
        == "reward_plateau"
    )
    assert (
        stopping_reason(
            round_index=5,
            minimum_rounds=5,
            maximum_rounds=8,
            plateau_rounds=0,
            plateau_patience=3,
            current_action_std=0.02,
            maximum_action_std_for_convergence=0.04,
            feasible_candidates=3,
        )
        == "distribution_converged"
    )
    assert (
        stopping_reason(
            round_index=8,
            minimum_rounds=5,
            maximum_rounds=8,
            plateau_rounds=0,
            plateau_patience=3,
            current_action_std=0.10,
            maximum_action_std_for_convergence=0.04,
            feasible_candidates=0,
        )
        == "no_feasible_candidates"
    )


def test_distribution_convergence_does_not_imply_success():
    outcome = classify_outcome(
        valid_realisation=True,
        validated_vad_improvement_over_baseline=0.01,
        learned_preference_wins_over_baseline=2,
        paired_repeats=5,
        in_target_quadrant=True,
        stable_reward=False,
        converged=True,
    )
    assert outcome == "converged_unsuccessfully"


def test_partial_outcomes_are_not_labelled_full_success():
    assert (
        classify_outcome(
            valid_realisation=True,
            validated_vad_improvement_over_baseline=0.06,
            learned_preference_wins_over_baseline=2,
            paired_repeats=5,
            in_target_quadrant=False,
            stable_reward=True,
            converged=False,
        )
        == "vad_improvement_without_quadrant_match"
    )
    assert (
        classify_outcome(
            valid_realisation=False,
            validated_vad_improvement_over_baseline=0.20,
            learned_preference_wins_over_baseline=5,
            paired_repeats=5,
            in_target_quadrant=True,
            stable_reward=True,
            converged=False,
        )
        == "feasible_only"
    )


def test_mean_smoothing_and_covariance_shrinkage_apply():
    distribution = LatentGaussianCEMDistribution(
        initial_action_mean=_initial_profile(0.55),
        initial_covariance=np.eye(5) * 0.2,
        mean_smoothing=0.5,
        covariance_shrinkage=0.8,
        seed=7,
    )
    before_mu = distribution.mu_z.copy()
    elites = _completed_samples(
        distribution,
        round_index=1,
        rewards=[0.9, 0.8, 0.7, 0.6],
    )
    selected_latent_mean = np.mean(
        np.asarray([sample.latent for sample in elites[:2]], dtype=float),
        axis=0,
    )
    distribution.update(elites, round_index=1, elite_count=2)
    after_mu = distribution.mu_z.copy()
    assert np.all(
        np.abs(after_mu - selected_latent_mean)
        < np.abs(before_mu - selected_latent_mean) + 1e-8
    )
    corr = np.asarray(distribution.last_update.correlation_matrix, dtype=float)
    assert np.all(np.abs(corr - np.eye(5)) <= 1.0)


def test_mock_correlated_optimum_is_recovered():
    optimum = np.array([0.8, 0.75, 0.7, 0.25, 0.35], dtype=float)
    distribution = _distribution(seed=19)
    initial_distance = np.linalg.norm(
        np.asarray([distribution.mean_action()[key] for key in _initial_profile()], dtype=float) - optimum
    )
    for round_index in range(1, 7):
        samples = distribution.sample_batch(18, round_index=round_index)
        completed = []
        for sample in samples:
            vector = np.asarray([sample.action[key] for key in _initial_profile()], dtype=float)
            reward = 1.0 - float(np.mean(np.abs(vector - optimum)))
            completed.append(
                LatentSample(
                    sample_id=sample.sample_id,
                    round_index=sample.round_index,
                    action=sample.action,
                    latent=sample.latent,
                    reward=reward,
                    feasible=True,
                    evaluation_consumed=True,
                )
            )
        distribution.update(completed, round_index=round_index, elite_count=6)
    final_distance = np.linalg.norm(
        np.asarray([distribution.mean_action()[key] for key in _initial_profile()], dtype=float) - optimum
    )
    assert final_distance < initial_distance


def test_baseline_directory_is_protected():
    with pytest.raises(RuntimeError, match="protected baseline"):
        guard_baseline_output_path(
            Path("outputs/experiments/baseline_fixed_profiles/demo")
        )
    guard_baseline_output_path(
        Path("outputs/experiments/baseline_fixed_profiles/demo"),
        overwrite_baseline=True,
    )


def _patch_runner_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = SimpleNamespace(
        projected_feasible_initialisation=None,
        requested_optimization=_initial_profile(0.55),
        target_vad={"valence": 0.6, "arousal": 0.6, "dominance": 0.6},
        baseline_vad_reward=0.5,
        baseline_observed_vad={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
        original_affect_hypothesis=_initial_profile(0.5),
        projection_error=None,
    )
    monkeypatch.setattr(outer_learning_runner, "baseline_condition", lambda *_args, **_kwargs: baseline)
    monkeypatch.setattr(outer_learning_runner, "baseline_optimizer_overrides", lambda *_args, **_kwargs: {"seed": 7})
    monkeypatch.setattr(
        outer_learning_runner,
        "estimate_initial_covariance_from_baseline",
        lambda *_args, **_kwargs: np.eye(5, dtype=float) * 0.10,
    )


def test_synthetic_objective_distance_is_not_realisation_error():
    context = OuterLearningContext("wave", "anger")
    result = outer_learning_runner._synthetic_step(context, _initial_profile(0.2))
    assert result["valid_realisation"] is True
    assert result["physically_acceptable"] is True
    assert result["feature_realisation_acceptable"] is True
    assert result["realisation_rmse"] == pytest.approx(0.0)
    assert result["max_abs_feature_error"] == pytest.approx(0.0)
    assert float(result["synthetic_objective_distance"]) > 0.0


def test_synthetic_invalid_injection_is_explicit_and_infeasible():
    context = OuterLearningContext("wave", "anger")
    result = outer_learning_runner._synthetic_step(
        context,
        _initial_profile(0.2),
        inject_invalid=True,
    )
    feasible, _ = strict_realisability(result, tolerance=0.10)
    assert result["valid_realisation"] is False
    assert result["feature_realisation_acceptable"] is False
    assert result["physically_acceptable"] is False
    assert feasible is False


def test_feasibility_invariant_detects_contradictions():
    contradictory = {
        "valid_realisation": False,
        "physically_acceptable": False,
        "feature_realisation_acceptable": False,
        "realisation_rmse": 1.0,
        "max_abs_feature_error": 1.0,
    }
    with pytest.raises(RuntimeError, match="Feasibility invariant failed"):
        outer_learning_runner._assert_feasibility_consistency(
            feasible=True,
            result=contradictory,
            tolerance=0.10,
            source="test-contradiction",
        )


def test_no_feasible_outcomes_are_not_marked_successful():
    assert outer_learning_runner._successful_outcome("no_feasible_candidates") is False
    assert outer_learning_runner._successful_outcome("evaluator_failure") is False
    assert outer_learning_runner._successful_outcome("converged_unsuccessfully") is False


def test_no_feasible_stage_cannot_report_all_successful(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "defaults": {
                    "rounds_min": 1,
                    "rounds_max": 1,
                    "samples_per_round": 2,
                    "elite_count": 1,
                    "candidate_vlm_repeats": 1,
                    "validation_top_k": 1,
                    "validation_vlm_repeats": 1,
                    "paired_validation_repeats": 1,
                },
                "stages": {
                    "mock": {
                        "contexts": [{"gesture": "wave", "target_state": "anger"}]
                    },
                    "stage_a": {"contexts": []},
                    "stage_b": {"contexts": []},
                    "full": {"contexts": []},
                },
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "stage_fail"
    exit_code = outer_learning_runner.main(
        [
            "--config",
            str(config_path),
            "--stage",
            "mock",
            "--evaluator",
            "synthetic_invalid",
            "--out",
            str(out_dir),
            "--overwrite",
        ]
    )
    assert exit_code == 0
    stage_payload = json.loads((out_dir / "mock" / "stage_status.json").read_text(encoding="utf-8"))
    assert stage_payload["all_successful"] is False
    assert stage_payload["contexts"][0]["outcome"] == "no_feasible_candidates"


def test_no_valid_candidate_skips_selection_pairing_and_final_motion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "invalid_context"
    result = outer_learning_runner.run_context(
        context=context,
        out_dir=out_dir,
        evaluator_name="synthetic_invalid",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=1,
        rounds_max=2,
        samples_per_round=4,
        elite_count=2,
        candidate_vlm_repeats=1,
        validation_top_k=2,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
    )
    assert result["selection_status"] == "no_valid_candidate"
    assert result["outcome"] == "no_feasible_candidates"
    assert not (out_dir / "selected_validated_profile.json").exists()
    assert not (out_dir / "paired_learned_vs_baseline.json").exists()
    assert not (out_dir / "paired_learned_vs_reference.json").exists()
    assert not (out_dir / "final_motion.mp4").exists()
    assert not (out_dir / "final_motion.gif").exists()


def test_invalid_validation_candidate_is_not_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    call_state = {"validation_rank": 0}

    def fake_evaluate(_environment, _context, profile, out_dir, *, synthetic, synthetic_invalid=False):
        if "validation" not in Path(out_dir).parts:
            return {
                "outer_reward": 0.9 if "sample_000" in str(out_dir) else 0.8,
                "mean_observed_vad": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
                "mean_vad_reward": 0.8,
                "valid_realisation": True,
                "physically_acceptable": True,
                "feature_realisation_acceptable": True,
                "realisation_rmse": 0.0,
                "max_abs_feature_error": 0.0,
                "synthetic_objective_distance": 0.2,
            }, None
        call_state["validation_rank"] += 1
        if call_state["validation_rank"] == 1:
            return {
                "outer_reward": 0.99,
                "mean_observed_vad": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
                "mean_vad_reward": 0.99,
                "valid_realisation": False,
                "physically_acceptable": False,
                "feature_realisation_acceptable": False,
                "realisation_rmse": 1.0,
                "max_abs_feature_error": 1.0,
                "synthetic_objective_distance": 0.1,
            }, None
        return {
            "outer_reward": 0.7,
            "mean_observed_vad": {"valence": 0.6, "arousal": 0.6, "dominance": 0.6},
            "mean_vad_reward": 0.7,
            "valid_realisation": True,
            "physically_acceptable": True,
            "feature_realisation_acceptable": True,
            "realisation_rmse": 0.0,
            "max_abs_feature_error": 0.0,
            "synthetic_objective_distance": 0.3,
        }, None

    monkeypatch.setattr(outer_learning_runner, "_evaluate_profile", fake_evaluate)
    out_dir = tmp_path / "validation_filter"
    result = outer_learning_runner.run_context(
        context=context,
        out_dir=out_dir,
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=1,
        rounds_max=1,
        samples_per_round=2,
        elite_count=1,
        candidate_vlm_repeats=1,
        validation_top_k=2,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
    )
    assert result["selection_status"] == "selected"
    validation_payload = json.loads((out_dir / "independent_validation.json").read_text(encoding="utf-8"))
    assert validation_payload["candidates"][0]["feasible"] is False
    selected_payload = json.loads((out_dir / "selected_validated_profile.json").read_text(encoding="utf-8"))
    assert selected_payload["feasible"] is True
    assert selected_payload["validation_reward"] == pytest.approx(0.7)


def test_multi_round_synthetic_convergence_and_context_independence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    contexts = [OuterLearningContext("wave", "anger"), OuterLearningContext("beckon", "fear")]
    seeds = [7, 17, 27]
    for context in contexts:
        for seed in seeds:
            out_dir = tmp_path / context.gesture / context.target_state / f"seed_{seed}"
            result = outer_learning_runner.run_context(
                context=context,
                out_dir=out_dir,
                evaluator_name="synthetic",
                model="gemini-2.5-flash",
                temperature=0.2,
                rounds_min=5,
                rounds_max=12,
                samples_per_round=24,
                elite_count=8,
                candidate_vlm_repeats=1,
                validation_top_k=3,
                validation_vlm_repeats=1,
                paired_validation_repeats=1,
                covariance_shrinkage=0.5,
                covariance_smoothing=0.3,
                mean_smoothing=0.4,
                minimum_eigenvalue=1e-3,
                maximum_eigenvalue=1.0,
                covariance_history_rounds=3,
                round_weight_decay=0.5,
                plateau_patience=3,
                minimum_reward_improvement=0.01,
                maximum_action_std_for_convergence=0.04,
                realisation_penalty_weight=0.25,
                max_feature_error_threshold=0.10,
                seed=seed,
            )
            stop_payload = json.loads((out_dir / "stopping_reason.json").read_text(encoding="utf-8"))
            assert stop_payload["rounds_completed"] >= 5
            assert stop_payload["stopping_reason"] in {
                "distribution_converged",
                "reward_plateau",
                "maximum_budget_reached",
                "no_feasible_candidates",
            }
            round_history = np.genfromtxt(out_dir / "round_history.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
            assert len(round_history) >= 5
            assert float(round_history["best_reward"][-1]) >= float(round_history["best_reward"][0])
            mean_history = np.genfromtxt(out_dir / "mean_history.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
            assert len(mean_history) >= 5
            covariance_payload = np.load(out_dir / "covariance_history.npz")
            assert len(covariance_payload.files) >= 5
            round_keys = sorted(covariance_payload.files)
            first = np.asarray(covariance_payload[round_keys[0]], dtype=float)
            first_snapshot = first.copy()
            last = np.asarray(covariance_payload[round_keys[-1]], dtype=float)
            assert not np.allclose(first, last)
            assert abs(float(last[0, 1]) - float(first[0, 1])) > 1e-6
            assert np.all(np.linalg.eigvalsh(last) > 0.0)
            _ = np.asarray(covariance_payload[round_keys[-1]], dtype=float)
            assert np.allclose(first_snapshot, np.asarray(covariance_payload[round_keys[0]], dtype=float))

            initial_profile_payload = json.loads((out_dir / "initial_profile.json").read_text(encoding="utf-8"))
            learned_profile_payload = json.loads((out_dir / "learned_profile.json").read_text(encoding="utf-8"))
            initial_covariance = np.asarray(initial_profile_payload["initial_cem_covariance"], dtype=float)
            learned_covariance = np.asarray(learned_profile_payload["learned_cem_covariance"], dtype=float)
            assert np.allclose(initial_covariance, np.eye(5, dtype=float) * 0.10)
            assert not np.allclose(initial_covariance, learned_covariance)
            assert not np.allclose(initial_covariance, last)

            selected = json.loads((out_dir / "selected_validated_profile.json").read_text(encoding="utf-8"))
            initial = json.loads((out_dir / "initial_profile.json").read_text(encoding="utf-8"))
            optimum = outer_learning_runner._synthetic_hidden_profile(context)
            initial_distance = outer_learning_runner._synthetic_objective_distance(initial["initial_cem_mean"], optimum)
            final_distance = outer_learning_runner._synthetic_objective_distance(selected["profile"], optimum)
            assert final_distance <= initial_distance
            assert result["comparison_status"] == "not_comparable_in_synthetic_mode"
            assert result["improvement_over_baseline"] is None
            assert result["initial_synthetic_objective_distance"] is not None
            assert result["final_synthetic_objective_distance"] is not None
            assert result["synthetic_distance_reduction"] is not None
            assert result["synthetic_distance_reduction"] >= 0.0
            assert result["outcome"] in {
                "synthetic_objective_improvement",
                "synthetic_optimum_recovered",
                "synthetic_converged_unsuccessfully",
            }

    wave_seed7 = json.loads((tmp_path / "wave" / "anger" / "seed_7" / "learned_profile.json").read_text(encoding="utf-8"))
    beckon_seed7 = json.loads((tmp_path / "beckon" / "fear" / "seed_7" / "learned_profile.json").read_text(encoding="utf-8"))
    assert wave_seed7["learned_cem_mean"] != beckon_seed7["learned_cem_mean"]


def test_sample_history_and_selected_profile_feasibility_fields_are_consistent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "consistency"
    outer_learning_runner.run_context(
        context=context,
        out_dir=out_dir,
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=2,
        rounds_max=3,
        samples_per_round=6,
        elite_count=3,
        candidate_vlm_repeats=1,
        validation_top_k=2,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
    )
    sample_rows = _read_csv_rows(out_dir / "sample_history.csv")
    for row in sample_rows:
        payload = {
            "valid_realisation": row["valid_realisation"].lower() == "true",
            "physically_acceptable": row["physically_acceptable"].lower() == "true",
            "feature_realisation_acceptable": row["feature_realisation_acceptable"].lower() == "true",
            "realisation_rmse": float(row["realisation_rmse"]),
            "max_abs_feature_error": float(row["max_abs_feature_error"]),
        }
        canonical, _ = strict_realisability(payload, tolerance=0.10)
        assert canonical == (row["feasible"].lower() == "true")

    selected = json.loads((out_dir / "selected_validated_profile.json").read_text(encoding="utf-8"))
    canonical, _ = strict_realisability(selected["result"], tolerance=0.10)
    assert canonical is True
    assert selected["feasible"] is True


def test_resume_recovers_completed_candidates_without_recomputing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "resume_recovery"
    original = outer_learning_runner._evaluate_candidate_payload
    state = {"failed": False, "calls": 0}

    def flaky(payload):
        state["calls"] += 1
        if (
            not state["failed"]
            and int(payload["round_index"]) == 1
            and int(payload["sample_index"]) == 2
        ):
            state["failed"] = True
            raise RuntimeError("intentional worker failure")
        return original(payload)

    monkeypatch.setattr(outer_learning_runner, "_evaluate_candidate_payload", flaky)
    first = outer_learning_runner.run_context(
        context=context,
        out_dir=out_dir,
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=1,
        rounds_max=1,
        samples_per_round=4,
        elite_count=2,
        candidate_vlm_repeats=1,
        validation_top_k=1,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
        workers=1,
        resume=False,
    )
    assert first["selection_status"] == "evaluator_failure"
    monkeypatch.setattr(outer_learning_runner, "_evaluate_candidate_payload", original)
    resumed = outer_learning_runner.run_context(
        context=context,
        out_dir=out_dir,
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=1,
        rounds_max=1,
        samples_per_round=4,
        elite_count=2,
        candidate_vlm_repeats=1,
        validation_top_k=1,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
        workers=1,
        resume=True,
    )
    assert resumed["resume_events"] >= 1
    assert resumed["recovered_candidates"] >= 1
    assert resumed["selection_status"] == "selected"


def test_resume_recomputes_malformed_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "resume_malformed"
    outer_learning_runner.run_context(
        context=context,
        out_dir=out_dir,
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=1,
        rounds_max=1,
        samples_per_round=3,
        elite_count=2,
        candidate_vlm_repeats=1,
        validation_top_k=1,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
        workers=1,
        resume=False,
    )
    damaged = out_dir / "checkpoint" / "candidates" / "round_001" / "sample_001.json"
    payload = json.loads(damaged.read_text(encoding="utf-8"))
    payload.pop("status")
    damaged.write_text(json.dumps(payload), encoding="utf-8")
    resumed = outer_learning_runner.run_context(
        context=context,
        out_dir=out_dir,
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=1,
        rounds_max=1,
        samples_per_round=3,
        elite_count=2,
        candidate_vlm_repeats=1,
        validation_top_k=1,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
        workers=1,
        resume=True,
    )
    assert resumed["recomputed_candidates"] >= 1


def test_resume_rejects_mismatched_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "resume_mismatch"
    outer_learning_runner.run_context(
        context=context,
        out_dir=out_dir,
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=1,
        rounds_max=1,
        samples_per_round=3,
        elite_count=2,
        candidate_vlm_repeats=1,
        validation_top_k=1,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
        workers=1,
        resume=False,
    )
    with pytest.raises(RuntimeError, match="saved run settings differ"):
        outer_learning_runner.run_context(
            context=context,
            out_dir=out_dir,
            evaluator_name="synthetic",
            model="gemini-2.5-flash",
            temperature=0.2,
            rounds_min=1,
            rounds_max=1,
            samples_per_round=3,
            elite_count=3,
            candidate_vlm_repeats=1,
            validation_top_k=1,
            validation_vlm_repeats=1,
            paired_validation_repeats=1,
            covariance_shrinkage=0.5,
            covariance_smoothing=0.3,
            mean_smoothing=0.4,
            minimum_eigenvalue=1e-3,
            maximum_eigenvalue=1.0,
            covariance_history_rounds=3,
            round_weight_decay=0.5,
            plateau_patience=3,
            minimum_reward_improvement=0.01,
            maximum_action_std_for_convergence=0.04,
            realisation_penalty_weight=0.25,
            max_feature_error_threshold=0.10,
            seed=7,
            workers=1,
            resume=True,
        )


def test_workers_produce_equivalent_results_in_synthetic_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    common = dict(
        context=context,
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=2,
        rounds_max=3,
        samples_per_round=6,
        elite_count=3,
        candidate_vlm_repeats=1,
        validation_top_k=2,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
        resume=False,
    )
    out_seq = tmp_path / "workers_1"
    out_par = tmp_path / "workers_2"
    seq = outer_learning_runner.run_context(out_dir=out_seq, workers=1, **common)
    par = outer_learning_runner.run_context(out_dir=out_par, workers=2, **common)
    assert seq["outcome"] == par["outcome"]
    assert seq["selection_status"] == par["selection_status"]
    assert seq["final_synthetic_objective_distance"] == pytest.approx(par["final_synthetic_objective_distance"])
    seq_profile = json.loads((out_seq / "learned_profile.json").read_text(encoding="utf-8"))
    par_profile = json.loads((out_par / "learned_profile.json").read_text(encoding="utf-8"))
    assert seq_profile["learned_cem_mean"] == pytest.approx(par_profile["learned_cem_mean"])
    assert np.allclose(
        np.asarray(seq_profile["learned_cem_covariance"], dtype=float),
        np.asarray(par_profile["learned_cem_covariance"], dtype=float),
    )


def test_mock_mode_never_instantiates_gemini_evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _fail_if_called(*_args, **_kwargs):
        raise AssertionError("Gemini evaluator must not be instantiated in mock mode.")

    monkeypatch.setattr(outer_learning_runner, "GeminiProVideoEvaluator", _fail_if_called)
    environment = outer_learning_runner._make_environment(
        evaluator_name="mock",
        model="gemini-2.5-flash",
        temperature=0.2,
        repeats=1,
        optimiser_overrides={"seed": 7},
        observation_cache=tmp_path / "mock_no_gemini_cache",
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        evaluator_seed=7,
    )
    assert environment.evaluator.__class__.__name__ == "MockNoisyPerceptualEvaluator"


def test_resume_on_fresh_context_starts_normally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    result = outer_learning_runner.run_context(
        context=context,
        out_dir=tmp_path / "resume_fresh",
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=1,
        rounds_max=1,
        samples_per_round=3,
        elite_count=2,
        candidate_vlm_repeats=1,
        validation_top_k=1,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
        workers=1,
        resume=True,
    )
    assert result["selection_status"] == "selected"
    assert result["resume_events"] == 0
    assert result["recovered_candidates"] == 0


# ---------------------------------------------------------------------------
# Robust-elite ranking and outcome-taxonomy regression tests
# ---------------------------------------------------------------------------

from laban_rl.perceptual_bandit.outer_learning import robust_rank_key


def _context_kwargs(out_dir, **overrides):
    kwargs = dict(
        evaluator_name="synthetic",
        model="gemini-2.5-flash",
        temperature=0.2,
        rounds_min=1,
        rounds_max=1,
        samples_per_round=3,
        elite_count=2,
        candidate_vlm_repeats=1,
        validation_top_k=2,
        validation_vlm_repeats=1,
        paired_validation_repeats=1,
        covariance_shrinkage=0.5,
        covariance_smoothing=0.3,
        mean_smoothing=0.4,
        minimum_eigenvalue=1e-3,
        maximum_eigenvalue=1.0,
        covariance_history_rounds=3,
        round_weight_decay=0.5,
        plateau_patience=3,
        minimum_reward_improvement=0.01,
        maximum_action_std_for_convergence=0.04,
        realisation_penalty_weight=0.25,
        max_feature_error_threshold=0.10,
        seed=7,
        workers=1,
        resume=False,
    )
    kwargs["out_dir"] = out_dir
    kwargs.update(overrides)
    return kwargs


def _ranked_sample(base, *, reward, feasible=True, robust=True, max_error=0.05, rmse=0.0, index=0):
    return LatentSample(
        sample_id=base.sample_id,
        round_index=base.round_index,
        action=base.action,
        latent=base.latent,
        reward=reward,
        feasible=feasible,
        evaluation_consumed=True,
        metadata={
            "robustly_feasible": robust,
            "max_abs_feature_error": max_error,
            "realisation_rmse": rmse,
            "sample_index": index,
        },
    )


def test_formal_feasibility_tolerance_unchanged_by_robust_margin():
    marginal_result = {
        "valid_realisation": True,
        "physically_acceptable": True,
        "feature_realisation_acceptable": True,
        "realisation_rmse": 0.05,
        "max_abs_feature_error": 0.09,
    }
    feasible, _ = strict_realisability(marginal_result, tolerance=0.10)
    assert feasible is True
    infeasible_result = dict(marginal_result, max_abs_feature_error=0.11)
    feasible, _ = strict_realisability(infeasible_result, tolerance=0.10)
    assert feasible is False
    fields = outer_learning_runner._robustness_fields(
        feasible=True,
        result_dict=marginal_result,
        tolerance=0.10,
        robust_margin=0.08,
        sample_index=0,
    )
    assert fields["strictly_feasible"] is True
    assert fields["robustly_feasible"] is False
    assert fields["ranking_category"] == "marginal_feasible"
    assert fields["distance_to_official_tolerance"] == pytest.approx(0.01)
    assert fields["distance_to_robust_margin"] == pytest.approx(-0.01)
    robust_fields = outer_learning_runner._robustness_fields(
        feasible=True,
        result_dict=dict(marginal_result, max_abs_feature_error=0.07),
        tolerance=0.10,
        robust_margin=0.08,
        sample_index=1,
    )
    assert robust_fields["robustly_feasible"] is True
    assert robust_fields["ranking_category"] == "robust_feasible"


def test_robust_feasible_outranks_marginal_feasible():
    distribution = _distribution()
    bases = distribution.sample_batch(2, round_index=1, prefix="rank")
    marginal_high = _ranked_sample(bases[0], reward=0.95, robust=False, max_error=0.09, index=0)
    robust_low = _ranked_sample(bases[1], reward=0.60, robust=True, max_error=0.05, index=1)
    ranked = sorted([marginal_high, robust_low], key=robust_rank_key)
    assert ranked[0].sample_id == robust_low.sample_id
    elites = distribution.update([marginal_high, robust_low], round_index=1, elite_count=1)
    assert elites[0].sample_id == robust_low.sample_id


def test_marginal_feasible_outranks_infeasible_high_reward():
    distribution = _distribution()
    bases = distribution.sample_batch(2, round_index=1, prefix="rank")
    infeasible_high = _ranked_sample(bases[0], reward=0.99, feasible=False, robust=False, max_error=0.5, index=0)
    marginal = _ranked_sample(bases[1], reward=0.40, robust=False, max_error=0.09, index=1)
    elites = distribution.update([infeasible_high, marginal], round_index=1, elite_count=2)
    assert [elite.sample_id for elite in elites] == [marginal.sample_id]


def test_marginal_candidates_fill_elite_slots_when_robust_insufficient():
    distribution = _distribution()
    bases = distribution.sample_batch(4, round_index=1, prefix="rank")
    robust = _ranked_sample(bases[0], reward=0.50, robust=True, max_error=0.04, index=0)
    marginal_a = _ranked_sample(bases[1], reward=0.90, robust=False, max_error=0.09, index=1)
    marginal_b = _ranked_sample(bases[2], reward=0.80, robust=False, max_error=0.095, index=2)
    marginal_c = _ranked_sample(bases[3], reward=0.70, robust=False, max_error=0.085, index=3)
    elites = distribution.update(
        [robust, marginal_a, marginal_b, marginal_c],
        round_index=1,
        elite_count=3,
    )
    ids = [elite.sample_id for elite in elites]
    assert ids == [robust.sample_id, marginal_a.sample_id, marginal_b.sample_id]


def test_robust_ranking_is_deterministic():
    distribution = _distribution()
    bases = distribution.sample_batch(3, round_index=1, prefix="rank")
    tied_a = _ranked_sample(bases[0], reward=0.70, robust=True, max_error=0.05, rmse=0.01, index=0)
    tied_b = _ranked_sample(bases[1], reward=0.70, robust=True, max_error=0.05, rmse=0.01, index=1)
    other = _ranked_sample(bases[2], reward=0.60, robust=True, max_error=0.05, index=2)
    first = sorted([tied_b, other, tied_a], key=robust_rank_key)
    second = sorted([tied_a, tied_b, other], key=robust_rank_key)
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert [item.sample_id for item in first] == [
        tied_a.sample_id,
        tied_b.sample_id,
        other.sample_id,
    ]


def _feature_error_evaluate_factory(training_errors, validation_error):
    """Fake _evaluate_profile with configurable max feature errors."""
    state = {"training_calls": 0, "validation_calls": 0}

    def fake_evaluate(_environment, _context, profile, out_dir, *, synthetic, synthetic_invalid=False):
        if "validation" in Path(out_dir).parts:
            state["validation_calls"] += 1
            error = validation_error
            reward = 0.75
        else:
            error = training_errors[state["training_calls"] % len(training_errors)]
            state["training_calls"] += 1
            reward = 0.9 - 0.01 * state["training_calls"]
        feasible = error <= 0.10
        return {
            "outer_reward": reward,
            "mean_observed_vad": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            "mean_vad_reward": reward,
            "valid_realisation": feasible,
            "physically_acceptable": feasible,
            "feature_realisation_acceptable": feasible,
            "realisation_rmse": error,
            "max_abs_feature_error": error,
            "synthetic_objective_distance": 0.2,
        }, None

    return fake_evaluate, state


def test_no_validation_feasible_candidate_outcome(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    fake_evaluate, state = _feature_error_evaluate_factory(
        training_errors=[0.05, 0.09],
        validation_error=0.13,
    )
    monkeypatch.setattr(outer_learning_runner, "_evaluate_profile", fake_evaluate)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "validation_all_fail"
    result = outer_learning_runner.run_context(
        context=context,
        **_context_kwargs(out_dir, samples_per_round=2, elite_count=1),
    )
    assert state["validation_calls"] >= 1
    assert result["selection_status"] == "no_valid_candidate"
    assert result["outcome"] == "no_validation_feasible_candidate"
    assert outer_learning_runner._successful_outcome(result["outcome"]) is False
    assert not (out_dir / "selected_validated_profile.json").exists()
    assert not (out_dir / "paired_learned_vs_baseline.json").exists()
    assert not (out_dir / "paired_learned_vs_reference.json").exists()
    assert not (out_dir / "final_motion.mp4").exists()
    assert not (out_dir / "final_motion.gif").exists()


def test_zero_feasible_training_still_reports_no_feasible_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "no_training_feasible"
    result = outer_learning_runner.run_context(
        context=context,
        **_context_kwargs(out_dir, evaluator_name="synthetic_invalid", samples_per_round=2, elite_count=1),
    )
    assert result["selection_status"] == "no_valid_candidate"
    assert result["outcome"] == "no_feasible_candidates"


def test_robustness_fields_are_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    fake_evaluate, _state = _feature_error_evaluate_factory(
        training_errors=[0.05, 0.09],
        validation_error=0.05,
    )
    monkeypatch.setattr(outer_learning_runner, "_evaluate_profile", fake_evaluate)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "robustness_fields"
    result = outer_learning_runner.run_context(
        context=context,
        **_context_kwargs(out_dir, samples_per_round=2, elite_count=1),
    )
    rows = _read_csv_rows(out_dir / "sample_history.csv")
    assert {"strictly_feasible", "robustly_feasible", "ranking_category", "distance_to_official_tolerance", "distance_to_robust_margin"}.issubset(rows[0])
    categories = {row["ranking_category"] for row in rows}
    assert categories == {"robust_feasible", "marginal_feasible"}
    for row in rows:
        assert row["strictly_feasible"] == "True"
        if row["ranking_category"] == "robust_feasible":
            assert row["robustly_feasible"] == "True"
        else:
            assert row["robustly_feasible"] == "False"
    validation_payload = json.loads((out_dir / "independent_validation.json").read_text(encoding="utf-8"))
    candidate = validation_payload["candidates"][0]
    assert candidate["training_seed"] is not None
    assert candidate["validation_seed"] == 7
    assert candidate["training_max_abs_feature_error"] is not None
    assert candidate["validation_max_abs_feature_error"] == pytest.approx(0.05)
    assert candidate["feasibility_survived_seed_change"] is True
    assert candidate["max_feature_error_change"] is not None
    # Robust training candidate must be validated ahead of the marginal one.
    assert candidate["training_ranking_category"] == "robust_feasible"
    assert result["selection_status"] == "selected"


def test_independent_validation_cannot_be_bypassed_by_robust_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    # Training candidates are all robust, but independent validation fails.
    fake_evaluate, state = _feature_error_evaluate_factory(
        training_errors=[0.02],
        validation_error=0.20,
    )
    monkeypatch.setattr(outer_learning_runner, "_evaluate_profile", fake_evaluate)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "no_bypass"
    result = outer_learning_runner.run_context(
        context=context,
        **_context_kwargs(out_dir, samples_per_round=2, elite_count=1),
    )
    assert state["validation_calls"] >= 1
    assert result["selection_status"] == "no_valid_candidate"
    assert result["outcome"] == "no_validation_feasible_candidate"
    assert not (out_dir / "selected_validated_profile.json").exists()


def test_resume_rejects_changed_robust_margin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    out_dir = tmp_path / "resume_robust_margin"
    outer_learning_runner.run_context(
        context=context,
        **_context_kwargs(out_dir, robust_elite_max_feature_error=0.08),
    )
    with pytest.raises(RuntimeError, match="saved run settings differ"):
        outer_learning_runner.run_context(
            context=context,
            **_context_kwargs(out_dir, robust_elite_max_feature_error=0.06, resume=True),
        )


def test_robust_margin_cannot_exceed_official_tolerance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _patch_runner_baseline(monkeypatch)
    context = OuterLearningContext("wave", "anger")
    with pytest.raises(ValueError, match="robust_elite_max_feature_error"):
        outer_learning_runner.run_context(
            context=context,
            **_context_kwargs(tmp_path / "bad_margin", robust_elite_max_feature_error=0.20),
        )


def test_no_validation_feasible_outcome_is_not_successful():
    assert outer_learning_runner._successful_outcome("no_validation_feasible_candidate") is False
