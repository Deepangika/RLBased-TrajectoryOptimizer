"""Tests for the offline live-pilot diagnostic script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

import analyse_live_pilot_diagnostics as diag  # noqa: E402

BECKON_RUN = ROOT / "outputs" / "experiments" / "outer_learning_beckon_fear_live_flash_v1"
WAVE_RUN = ROOT / "outputs" / "experiments" / "outer_learning_wave_sadness_live_flash_v1"
BASELINE_ROOT = ROOT / "outputs" / "experiments" / "baseline_fixed_profiles"

ARTIFACTS_PRESENT = (
    (BECKON_RUN / "live_stage_a" / "stage_status.json").exists()
    and (WAVE_RUN / "live_stage_a" / "stage_status.json").exists()
    and (BASELINE_ROOT / "baseline_manifest.json").exists()
)


class TestLanguageCoding:
    def test_keywords_map_to_codes(self) -> None:
        codes = diag.code_text(
            "The arm moves in an erratic, trembling way that looks distorted."
        )
        assert "erratic_jittery" in codes
        assert "distortion_incoherence" in codes

    def test_no_match_returns_empty(self) -> None:
        assert diag.code_text("The clip shows an arm.") == []


class TestMotionMetrics:
    def test_identical_trajectories_have_zero_deviation(self) -> None:
        t = np.linspace(0.0, 1.0, 160)
        q = np.stack([np.sin(t), np.cos(t)], axis=1)
        metrics = diag.motion_metrics(q, q)
        assert metrics["mean_abs_deviation_from_reference"] == 0.0
        assert metrics["endpoint_error"] == 0.0
        assert metrics["rms_joint_velocity"] > 0.0

    def test_noisier_trajectory_has_higher_jerk(self) -> None:
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 1.0, 160)
        smooth = np.stack([np.sin(t), np.cos(t)], axis=1)
        noisy = smooth + rng.normal(scale=0.02, size=smooth.shape)
        smooth_metrics = diag.motion_metrics(smooth, smooth)
        noisy_metrics = diag.motion_metrics(smooth, noisy)
        assert noisy_metrics["rms_joint_jerk"] > smooth_metrics["rms_joint_jerk"]


@pytest.mark.skipif(not ARTIFACTS_PRESENT, reason="live pilot artifacts not present")
class TestEndToEnd:
    def test_diagnostic_runs_offline_and_passes_integrity(self, tmp_path: Path) -> None:
        exit_code = diag.main(
            [
                "--beckon-fear-run", str(BECKON_RUN),
                "--wave-sadness-run", str(WAVE_RUN),
                "--baseline-root", str(BASELINE_ROOT),
                "--out", str(tmp_path / "diag"),
            ]
        )
        assert exit_code == 0
        payload = json.loads(
            (tmp_path / "diag" / "diagnostics.json").read_text(encoding="utf-8")
        )
        assert payload["api_calls_made"] == 0
        assert payload["integrity"] == "all_checks_passed"
        agreement = {row["context"]: row for row in payload["metric_agreement"]}
        assert agreement["beckon-fear"]["corrected_outcome"] == "paired_only_improvement"
        assert (
            agreement["wave-sadness"]["corrected_outcome"] == "no_perceptual_improvement"
        )
        assert (tmp_path / "diag" / "diagnostic_report.md").exists()
        assert len(list((tmp_path / "diag" / "plots").glob("*.png"))) == 9
