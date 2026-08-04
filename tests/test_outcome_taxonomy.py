"""Regression tests for the corrected (v2) outcome taxonomy."""

from __future__ import annotations

import pytest

from laban_rl.perceptual_bandit.outcome_taxonomy import (
    TAXONOMY_VERSION,
    assess_perceptual_outcome,
    classify_paired_preference,
)


def _paired(styled: int, reference: int, neither: int) -> dict:
    return {
        "records": [
            {"choice_counts": {"styled": styled, "reference": reference, "neither": neither}}
        ]
    }


class TestClassifyPairedPreference:
    @pytest.mark.parametrize(
        "wins,expected",
        [
            (0, "no_preference_improvement"),
            (1, "no_preference_improvement"),
            (2, "weak_or_inconclusive_preference"),
            (3, "weak_or_inconclusive_preference"),
            (4, "strong_preference_improvement"),
            (5, "strong_preference_improvement"),
        ],
    )
    def test_five_repeat_thresholds(self, wins: int, expected: str) -> None:
        assert classify_paired_preference(wins, 5 - wins, 0, 5) == expected

    def test_tie_is_inconclusive(self) -> None:
        assert classify_paired_preference(2, 2, 0, 4) == "weak_or_inconclusive_preference"

    def test_neither_heavy_is_inconclusive(self) -> None:
        # 2 styled / 1 reference / 2 neither: no clear majority.
        assert classify_paired_preference(2, 1, 2, 5) == "weak_or_inconclusive_preference"

    def test_missing_results_not_evaluated(self) -> None:
        assert classify_paired_preference(None, None, None, None) == "not_evaluated"
        assert classify_paired_preference(3, 2, 0, 0) == "not_evaluated"

    def test_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            classify_paired_preference(3, 3, 0, 5)


class TestAssessPerceptualOutcome:
    def test_execution_success_without_perceptual_improvement(self) -> None:
        # Mirrors the live wave-sadness run: negative reward delta, 2/5 paired.
        out = assess_perceptual_outcome(
            execution_success=True,
            selection_success=True,
            selected_validation_reward=0.648,
            baseline_validation_reward=0.686,
            paired_vs_baseline=_paired(2, 1, 2),
            paired_vs_reference=_paired(1, 4, 0),
        )
        assert out["taxonomy_version"] == TAXONOMY_VERSION
        assert out["execution_success"] is True
        assert out["selection_success"] is True
        assert out["absolute_reward_improvement"] is False
        assert out["paired_improvement_vs_baseline"] == "weak_or_inconclusive_preference"
        assert out["paired_improvement_vs_reference"] == "no_preference_improvement"
        assert out["perceptual_improvement"] is False
        assert out["perceptual_outcome"] == "no_perceptual_improvement"

    def test_paired_only_improvement(self) -> None:
        # Mirrors the live beckon-fear run: negative reward delta, 5/5 paired.
        out = assess_perceptual_outcome(
            execution_success=True,
            selection_success=True,
            selected_validation_reward=0.790,
            baseline_validation_reward=0.886,
            paired_vs_baseline=_paired(5, 0, 0),
            paired_vs_reference=_paired(3, 1, 1),
        )
        assert out["absolute_reward_improvement"] is False
        assert out["paired_improvement_vs_baseline"] == "strong_preference_improvement"
        assert out["perceptual_improvement"] is True
        assert out["perceptual_outcome"] == "paired_only_improvement"

    def test_absolute_and_paired_improvement(self) -> None:
        out = assess_perceptual_outcome(
            execution_success=True,
            selection_success=True,
            selected_validation_reward=0.9,
            baseline_validation_reward=0.8,
            paired_vs_baseline=_paired(4, 1, 0),
            paired_vs_reference=_paired(4, 1, 0),
        )
        assert out["perceptual_outcome"] == "absolute_and_paired_improvement"
        assert out["perceptual_improvement"] is True

    def test_selection_without_perceptual_claims(self) -> None:
        out = assess_perceptual_outcome(
            execution_success=True,
            selection_success=True,
            selected_validation_reward=None,
            baseline_validation_reward=None,
            paired_vs_baseline=None,
            paired_vs_reference=None,
        )
        assert out["perceptual_improvement"] is None
        assert out["perceptual_outcome"] == "not_evaluated"

    def test_no_valid_selection_blocks_perceptual_claims(self) -> None:
        out = assess_perceptual_outcome(
            execution_success=True,
            selection_success=False,
            selected_validation_reward=None,
            baseline_validation_reward=0.8,
            paired_vs_baseline=None,
            paired_vs_reference=None,
        )
        assert out["selection_success"] is False
        assert out["perceptual_improvement"] is False
        assert out["perceptual_outcome"] == "no_valid_selection"

    def test_non_comparable_mode_never_reports_absolute_delta(self) -> None:
        out = assess_perceptual_outcome(
            execution_success=True,
            selection_success=True,
            selected_validation_reward=0.9,
            baseline_validation_reward=0.5,
            paired_vs_baseline=None,
            paired_vs_reference=None,
            comparison_status="not_comparable_in_synthetic_mode",
        )
        assert out["absolute_reward_delta_vs_baseline"] is None
        assert out["absolute_reward_improvement"] is None

    def test_raw_counts_recorded_beside_labels(self) -> None:
        out = assess_perceptual_outcome(
            execution_success=True,
            selection_success=True,
            selected_validation_reward=0.7,
            baseline_validation_reward=0.8,
            paired_vs_baseline=_paired(2, 1, 2),
            paired_vs_reference=_paired(1, 4, 0),
        )
        assert out["paired_vs_baseline_counts"] == {
            "available": True,
            "learned_wins": 2,
            "opponent_wins": 1,
            "neither": 2,
            "repeats": 5,
        }
        assert out["paired_vs_reference_counts"]["learned_wins"] == 1
