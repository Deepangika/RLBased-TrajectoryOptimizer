"""Tests for the opt-in reference-relative perceptual diagnostic."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from laban_rl.config import FEATURE_KEYS
from laban_rl.optimiser_api import LabanOptimisationResult
from laban_rl.perceptual_bandit.environment import Context
from laban_rl.perceptual_bandit.paired_preference import (
    DIAGNOSTIC_PROMPT_VERSION,
    DIAGNOSTIC_SCHEMA_VERSION,
    GeminiPairedReferenceDiagnosticEvaluator,
    MockPairedPreferenceEvaluator,
    MockPairedReferenceDiagnosticEvaluator,
    PairedPreferenceCache,
    PreferencePair,
    run_paired_preference_experiment,
)
from laban_rl.perceptual_bandit.reference_relative import (
    REFERENCE_RELATIVE_CSV_FIELDS,
    REFERENCE_RELATIVE_INTERPRETATION,
    ReferenceRelativeThresholds,
    aggregate_paired_diagnostic,
    bootstrap_shift_interval,
    classify_reference_relative_outcome,
    compute_reference_relative_metrics,
    reference_relative_csv_row,
    summarise_vad_observations,
)


VAD = ("valence", "arousal", "dominance")


def _vad(v, a, d):
    return {"valence": v, "arousal": a, "dominance": d}


def _obs(v, a, d):
    return {"affect_ratings": _vad(v, a, d)}


def make_result(path: Path, *, offset: float = 0.0) -> LabanOptimisationResult:
    profile = {key: 0.5 for key in FEATURE_KEYS}
    q_ref = np.zeros((20, 2), dtype=float)
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
        q_var=q_ref + offset,
        output_dir=path,
        raw_result={
            "reward_info": {
                "path_length_ratio": 1.0,
                "joint_limit_error": 0.0,
            }
        },
    )


def _assert_json_safe(payload):
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


# ---------------------------------------------------------------------------
# summarise_vad_observations
# ---------------------------------------------------------------------------


def test_summarise_vad_observations_mean_sd_and_validity():
    summary = summarise_vad_observations(
        [
            _obs(0.2, 0.4, 0.6),
            _obs(0.4, 0.6, 0.8),
            {"affect_ratings": {"valence": 0.5}},  # missing axes -> invalid
            {"affect_ratings": _vad(float("nan"), 0.5, 0.5)},  # non-finite
            {},  # no ratings at all
        ]
    )
    assert summary["valid_count"] == 2
    assert summary["invalid_count"] == 3
    assert summary["vad_mean"] == pytest.approx(_vad(0.3, 0.5, 0.7))
    expected_sd = float(np.std([0.2, 0.4], ddof=1))
    for axis in VAD:
        assert summary["vad_sd"][axis] == pytest.approx(expected_sd)
    _assert_json_safe(summary)


def test_summarise_vad_observations_all_invalid_returns_null_summary():
    summary = summarise_vad_observations([{}, {"affect_ratings": None}])
    assert summary["vad_mean"] is None
    assert summary["vad_sd"] is None
    assert summary["valid_count"] == 0
    assert summary["invalid_count"] == 2


def test_summarise_single_observation_has_zero_sd():
    summary = summarise_vad_observations([_obs(0.3, 0.3, 0.3)])
    assert summary["valid_count"] == 1
    assert all(summary["vad_sd"][axis] == 0.0 for axis in VAD)


# ---------------------------------------------------------------------------
# compute_reference_relative_metrics
# ---------------------------------------------------------------------------


def test_metrics_positive_progress_and_alignment():
    metrics = compute_reference_relative_metrics(
        reference_vad=_vad(0.2, 0.2, 0.2),
        candidate_vad=_vad(0.4, 0.4, 0.4),
        target_vad=_vad(0.6, 0.6, 0.6),
        reference_sd=_vad(0.05, 0.05, 0.05),
    )
    assert metrics["directional_alignment"] == pytest.approx(1.0)
    assert metrics["normalised_progress"] == pytest.approx(0.5)
    assert metrics["observed_shift"] == pytest.approx(_vad(0.2, 0.2, 0.2))
    assert metrics["required_shift"] == pytest.approx(_vad(0.4, 0.4, 0.4))
    for axis in VAD:
        reduction = metrics["dimension_error_reduction"][axis]
        assert reduction["error_reduction"] == pytest.approx(0.2)
        standardised = metrics["standardised_shift"][axis]
        assert standardised["value"] == pytest.approx(0.2 / 0.05)
        assert standardised["null_reason"] is None
    assert metrics["interpretation"] == REFERENCE_RELATIVE_INTERPRETATION
    _assert_json_safe(metrics)


def test_metrics_negative_progress_when_candidate_moves_away():
    metrics = compute_reference_relative_metrics(
        reference_vad=_vad(0.4, 0.4, 0.4),
        candidate_vad=_vad(0.2, 0.2, 0.2),
        target_vad=_vad(0.6, 0.6, 0.6),
    )
    assert metrics["directional_alignment"] == pytest.approx(-1.0)
    assert metrics["normalised_progress"] == pytest.approx(-1.0)


def test_metrics_zero_shift_yields_null_alignment_with_reason():
    metrics = compute_reference_relative_metrics(
        reference_vad=_vad(0.4, 0.4, 0.4),
        candidate_vad=_vad(0.4, 0.4, 0.4),
        target_vad=_vad(0.6, 0.6, 0.6),
    )
    assert metrics["directional_alignment"] is None
    assert (
        metrics["directional_alignment_null_reason"]
        == "observed_shift_zero_length"
    )
    assert metrics["normalised_progress"] == pytest.approx(0.0)
    _assert_json_safe(metrics)


def test_metrics_reference_on_target_yields_null_progress_with_reason():
    metrics = compute_reference_relative_metrics(
        reference_vad=_vad(0.6, 0.6, 0.6),
        candidate_vad=_vad(0.5, 0.5, 0.5),
        target_vad=_vad(0.6, 0.6, 0.6),
    )
    assert metrics["normalised_progress"] is None
    assert (
        metrics["normalised_progress_null_reason"]
        == "reference_target_distance_zero"
    )
    assert (
        metrics["directional_alignment_null_reason"]
        == "required_shift_zero_length_reference_on_target"
    )


def test_metrics_zero_or_missing_reference_sd_gives_null_standardised_shift():
    metrics = compute_reference_relative_metrics(
        reference_vad=_vad(0.2, 0.2, 0.2),
        candidate_vad=_vad(0.4, 0.4, 0.4),
        target_vad=_vad(0.6, 0.6, 0.6),
        reference_sd=_vad(0.0, 0.0, 0.0),
    )
    for axis in VAD:
        assert metrics["standardised_shift"][axis]["value"] is None
        assert (
            metrics["standardised_shift"][axis]["null_reason"]
            == "reference_sd_zero_or_invalid"
        )
    no_sd = compute_reference_relative_metrics(
        reference_vad=_vad(0.2, 0.2, 0.2),
        candidate_vad=_vad(0.4, 0.4, 0.4),
        target_vad=_vad(0.6, 0.6, 0.6),
    )
    for axis in VAD:
        assert (
            no_sd["standardised_shift"][axis]["null_reason"]
            == "reference_sd_unavailable"
        )


# ---------------------------------------------------------------------------
# bootstrap_shift_interval
# ---------------------------------------------------------------------------


def test_bootstrap_requires_two_valid_observations_per_side():
    result = bootstrap_shift_interval(
        reference_observations=[_obs(0.2, 0.2, 0.2)],
        candidate_observations=[_obs(0.4, 0.4, 0.4), _obs(0.5, 0.5, 0.5)],
        seed=7,
    )
    assert result["status"] == "not_computed"
    assert result["intervals"] is None
    assert result["null_reason"] == (
        "fewer_than_two_valid_observations_on_one_side"
    )


def test_bootstrap_interval_brackets_true_shift_and_is_deterministic():
    reference = [_obs(0.2, 0.2, 0.2), _obs(0.22, 0.22, 0.22), _obs(0.18, 0.18, 0.18)]
    candidate = [_obs(0.5, 0.5, 0.5), _obs(0.52, 0.52, 0.52), _obs(0.48, 0.48, 0.48)]
    first = bootstrap_shift_interval(
        reference_observations=reference,
        candidate_observations=candidate,
        seed=7,
        n_bootstrap=500,
    )
    second = bootstrap_shift_interval(
        reference_observations=reference,
        candidate_observations=candidate,
        seed=7,
        n_bootstrap=500,
    )
    assert first == second
    assert first["status"] == "computed"
    for axis in VAD:
        interval = first["intervals"][axis]
        assert interval["lower"] <= 0.3 <= interval["upper"]
        assert interval["lower"] > 0.0  # shift clearly positive
    _assert_json_safe(first)


# ---------------------------------------------------------------------------
# aggregate_paired_diagnostic
# ---------------------------------------------------------------------------


def test_aggregate_paired_diagnostic_counts_and_rates():
    observations = [
        {
            "choice": "styled",
            "same_gesture": True,
            "valence_higher": "styled",
            "arousal_higher": "similar",
            "dominance_higher": "reference",
        },
        {
            "choice": "reference",
            "same_gesture": False,
            "valence_higher": "reference",
            "arousal_higher": "reference",
            "dominance_higher": "reference",
        },
        {
            "choice": "neither",
            "same_gesture": True,
            "valence_higher": "similar",
            "arousal_higher": "similar",
            "dominance_higher": "similar",
        },
        {"choice": "banana"},  # invalid choice ignored
    ]
    summary = aggregate_paired_diagnostic(observations)
    assert summary["candidate_preferred"] == 1
    assert summary["reference_preferred"] == 1
    assert summary["neither"] == 1
    assert summary["valid_responses"] == 3
    assert summary["invalid_responses"] == 1
    assert summary["candidate_win_rate_excluding_neither"] == pytest.approx(0.5)
    assert summary["candidate_win_rate_including_neither"] == pytest.approx(1 / 3)
    assert summary["gesture_preservation_agreement"] == pytest.approx(2 / 3)
    assert summary["dimension_level_preferences"]["valence"] == {
        "styled": 1,
        "reference": 1,
        "similar": 1,
    }
    _assert_json_safe(summary)


def test_aggregate_paired_diagnostic_empty_gives_null_rates():
    summary = aggregate_paired_diagnostic([])
    assert summary["candidate_win_rate_excluding_neither"] is None
    assert summary["candidate_win_rate_including_neither"] is None
    assert summary["gesture_preservation_agreement"] is None


# ---------------------------------------------------------------------------
# classify_reference_relative_outcome — all six labels
# ---------------------------------------------------------------------------


THRESHOLDS = ReferenceRelativeThresholds()


def _metrics(reference, candidate, target, sd=None):
    return compute_reference_relative_metrics(
        reference_vad=_vad(*reference),
        candidate_vad=_vad(*candidate),
        target_vad=_vad(*target),
        reference_sd=_vad(*sd) if sd is not None else None,
    )


def test_classify_insufficient_valid_evaluations():
    outcome = classify_reference_relative_outcome(
        metrics=None,
        thresholds=THRESHOLDS,
        reference_valid_count=1,
        candidate_valid_count=3,
    )
    assert outcome["label"] == "insufficient_valid_evaluations"
    assert "minimum_required=2" in outcome["reason"]


def test_classify_reference_already_near_target():
    metrics = _metrics((0.58, 0.6, 0.6), (0.7, 0.7, 0.7), (0.6, 0.6, 0.6))
    outcome = classify_reference_relative_outcome(
        metrics=metrics,
        thresholds=THRESHOLDS,
        reference_valid_count=3,
        candidate_valid_count=3,
    )
    assert outcome["label"] == "reference_already_near_target"


def test_classify_moved_away_from_target():
    metrics = _metrics((0.4, 0.4, 0.4), (0.2, 0.2, 0.2), (0.8, 0.8, 0.8))
    outcome = classify_reference_relative_outcome(
        metrics=metrics,
        thresholds=THRESHOLDS,
        reference_valid_count=3,
        candidate_valid_count=3,
    )
    assert outcome["label"] == "moved_away_from_target"


def test_classify_no_meaningful_perceptual_shift():
    # Tiny shift inside reference variability, no decisive paired preference.
    metrics = _metrics(
        (0.4, 0.4, 0.4),
        (0.41, 0.41, 0.41),
        (0.8, 0.8, 0.8),
        sd=(0.1, 0.1, 0.1),
    )
    outcome = classify_reference_relative_outcome(
        metrics=metrics,
        thresholds=THRESHOLDS,
        reference_valid_count=3,
        candidate_valid_count=3,
        reference_sd=_vad(0.1, 0.1, 0.1),
        paired_summary={"candidate_win_rate_excluding_neither": 0.5},
    )
    assert outcome["label"] == "no_meaningful_perceptual_shift"


def test_classify_reached_or_improved_toward_target():
    metrics = _metrics(
        (0.2, 0.2, 0.2),
        (0.75, 0.75, 0.75),
        (0.8, 0.8, 0.8),
        sd=(0.02, 0.02, 0.02),
    )
    outcome = classify_reference_relative_outcome(
        metrics=metrics,
        thresholds=THRESHOLDS,
        reference_valid_count=3,
        candidate_valid_count=3,
        reference_sd=_vad(0.02, 0.02, 0.02),
    )
    assert outcome["label"] == "reached_or_improved_toward_target"


def test_classify_directional_improvement_but_target_not_reached():
    # Meaningful, positively aligned shift that leaves the candidate exactly
    # as far from the target as the reference was: not an improvement in
    # distance, but not a regression either.
    metrics = _metrics(
        (0.2, 0.2, 0.2),
        (0.8, 0.8, 0.2),
        (0.8, 0.2, 0.2),
        sd=(0.01, 0.01, 0.01),
    )
    assert metrics["normalised_progress"] == pytest.approx(0.0)
    assert metrics["directional_alignment"] > 0.0
    outcome = classify_reference_relative_outcome(
        metrics=metrics,
        thresholds=THRESHOLDS,
        reference_valid_count=3,
        candidate_valid_count=3,
        reference_sd=_vad(0.01, 0.01, 0.01),
    )
    assert outcome["label"] == "directional_improvement_but_target_not_reached"


# ---------------------------------------------------------------------------
# CSV row
# ---------------------------------------------------------------------------


def test_csv_row_contains_every_field_and_is_json_safe():
    metrics = _metrics((0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.6, 0.6, 0.6))
    paired = aggregate_paired_diagnostic(
        [{"choice": "styled", "same_gesture": True}]
    )
    row = reference_relative_csv_row(
        gesture="point",
        target_state="happiness",
        metrics=metrics,
        paired_summary=paired,
        outcome_label="reached_or_improved_toward_target",
    )
    assert list(row) == REFERENCE_RELATIVE_CSV_FIELDS
    assert row["gesture"] == "point"
    assert row["paired_candidate_preferred"] == 1
    assert row["reference_relative_outcome"] == (
        "reached_or_improved_toward_target"
    )
    _assert_json_safe(row)


def test_csv_row_handles_missing_metrics_and_paired_summary():
    row = reference_relative_csv_row(
        gesture="wave",
        target_state="sadness",
        metrics=None,
        paired_summary=None,
        outcome_label="insufficient_valid_evaluations",
    )
    assert list(row) == REFERENCE_RELATIVE_CSV_FIELDS
    assert row["candidate_target_distance"] is None
    assert row["paired_candidate_preferred"] is None


# ---------------------------------------------------------------------------
# Diagnostic evaluators
# ---------------------------------------------------------------------------


def test_diagnostic_prompt_is_neutral_and_versions_are_distinct():
    prompt = GeminiPairedReferenceDiagnosticEvaluator._prompt(
        Context("wave", "sadness")
    )
    lowered = prompt.lower()
    assert "valence_higher" in prompt
    assert "same_gesture" in prompt
    assert "noticeable_difference" in prompt
    for banned in ("reference", "styled", "optimis", "optimiz", "reward",
                   "target vad", "laban"):
        assert banned not in lowered
    assert DIAGNOSTIC_PROMPT_VERSION != "blinded-preference-ab-v1"
    assert DIAGNOSTIC_SCHEMA_VERSION != "paired-preference-v1"


def test_mock_diagnostic_evaluator_extends_base_choice_with_dimensions(tmp_path):
    context = Context("point", "happiness")
    reference = make_result(tmp_path / "reference")
    styled = make_result(tmp_path / "styled")
    styled.achieved_profile["weight"] = 0.9
    evaluator = MockPairedReferenceDiagnosticEvaluator()
    observation = evaluator.evaluate_pair(
        context,
        reference,
        styled,
        repeat_index=0,
        pair_hash="deadbeef",
    )
    base = MockPairedPreferenceEvaluator().evaluate_pair(
        context,
        reference,
        styled,
        repeat_index=0,
        pair_hash="deadbeef",
    )
    assert observation["choice"] == base["choice"]
    # styled profile mean is higher, so every dimension resolves to styled
    assert observation["valence_higher"] == "styled"
    assert observation["arousal_higher"] == "styled"
    assert observation["dominance_higher"] == "styled"
    assert observation["same_gesture"] is True
    assert 0.0 <= observation["noticeable_difference"] <= 1.0
    displayed = (observation["displayed_a"], observation["displayed_b"])
    expected_raw = "A" if displayed[0] == "styled" else "B"
    assert observation["raw_valence_higher"] == expected_raw


def test_mock_diagnostic_similar_profiles_report_similar_dimensions(tmp_path):
    context = Context("point", "happiness")
    reference = make_result(tmp_path / "reference")
    styled = make_result(tmp_path / "styled")
    evaluator = MockPairedReferenceDiagnosticEvaluator()
    observation = evaluator.evaluate_pair(
        context,
        reference,
        styled,
        repeat_index=0,
        pair_hash="deadbeef",
    )
    assert observation["valence_higher"] == "similar"
    assert observation["raw_valence_higher"] == "similar"


def test_diagnostic_observations_pass_through_paired_cache(tmp_path):
    context = Context("point", "happiness")
    reference = make_result(tmp_path / "reference")
    styled = make_result(tmp_path / "styled")
    styled.achieved_profile["weight"] = 0.9
    pair = PreferencePair(
        pair_id="point-happiness-reference-diagnostic",
        context=context,
        reference_result=reference,
        styled_result=styled,
        target_layers={},
    )
    cache = PairedPreferenceCache(tmp_path / "rd-cache")
    first = run_paired_preference_experiment(
        [pair],
        evaluator=MockPairedReferenceDiagnosticEvaluator(),
        cache=cache,
        repeats=3,
        out_path=tmp_path / "diagnostic.json",
    )
    observations = first["records"][0]["observations"]
    assert len(observations) == 3
    for observation in observations:
        assert observation["valence_higher"] in ("styled", "reference", "similar")
        assert isinstance(observation["same_gesture"], bool)
        assert "noticeable_difference" in observation

    # Cached rows (including the diagnostic extras) are reused on resume.
    class ExplodingEvaluator(MockPairedReferenceDiagnosticEvaluator):
        def evaluate_pair(self, *args, **kwargs):
            raise AssertionError("cache should have satisfied all repeats")

    resumed = run_paired_preference_experiment(
        [pair],
        evaluator=ExplodingEvaluator(),
        cache=cache,
        repeats=3,
        out_path=tmp_path / "diagnostic.json",
    )
    assert resumed["records"][0]["observations"] == observations


def test_diagnostic_cache_identity_is_distinct_from_preference_identity():
    diagnostic = MockPairedReferenceDiagnosticEvaluator().cache_identity()
    preference = MockPairedPreferenceEvaluator().cache_identity()
    assert diagnostic["provider"] != preference["provider"]
    assert diagnostic["prompt_version"] != preference["prompt_version"]
    assert diagnostic["schema_version"] != preference["schema_version"]


def test_summary_from_diagnostic_run_aggregates_cleanly(tmp_path):
    context = Context("point", "happiness")
    reference = make_result(tmp_path / "reference")
    styled = make_result(tmp_path / "styled")
    styled.achieved_profile["weight"] = 0.9
    pair = PreferencePair(
        pair_id="point-happiness-reference-diagnostic",
        context=context,
        reference_result=reference,
        styled_result=styled,
        target_layers={},
    )
    payload = run_paired_preference_experiment(
        [pair],
        evaluator=MockPairedReferenceDiagnosticEvaluator(),
        cache=PairedPreferenceCache(tmp_path / "rd-cache"),
        repeats=4,
        out_path=tmp_path / "diagnostic.json",
    )
    summary = aggregate_paired_diagnostic(payload["records"][0]["observations"])
    assert summary["valid_responses"] == 4
    assert summary["invalid_responses"] == 0
    _assert_json_safe(summary)
