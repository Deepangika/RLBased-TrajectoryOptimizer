import json
from pathlib import Path

import pytest

from laban_rl.config import FEATURE_KEYS, GESTURE_TYPES
from laban_rl.hygiene import credential_presence_check
from laban_rl.perceptual_bandit.ablation import rescore_paired_ablation
from laban_rl.perceptual_bandit.compatibility import (
    CheckpointCompatibilityError,
    checkpoint_metadata,
    migrate_metadata_only_checkpoint,
    validate_checkpoint,
)
from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    PerceptualEvaluation,
)
from laban_rl.perceptual_bandit.scoring import score_perceptual_observations
from laban_rl.readiness import audit_readiness


VAD = {"valence": 0.8, "arousal": 0.7, "dominance": 0.6}
PROBABILITIES = {
    "anger": 0.05,
    "disgust": 0.05,
    "fear": 0.05,
    "happiness": 0.7,
    "sadness": 0.05,
    "surprise": 0.1,
}


def test_ambiguity_is_distinct_from_missing_and_complete_categories():
    observations = [
        PerceptualEvaluation(
            affect_ratings=VAD,
            probabilities=PROBABILITIES,
            category_status="complete",
        ),
        PerceptualEvaluation(
            affect_ratings=VAD,
            probabilities=PROBABILITIES,
            category_status="ambiguous",
            perceived_state="ambiguous",
            category_intensities={
                label: 0.5 for label in PROBABILITIES
            },
        ),
        PerceptualEvaluation(
            affect_ratings=VAD,
            category_status="missing",
            perceived_state="none",
        ),
    ]
    score = score_perceptual_observations(
        observations,
        target_vad=Context("point", "happiness").target_vad or {},
        target_state="happiness",
        reward_config=EnvironmentRewardConfig(repeat_evaluations=3),
    )

    assert score["categorical_distribution_coverage"] == pytest.approx(2 / 3)
    assert score["categorical_unambiguous_coverage"] == pytest.approx(1 / 3)
    assert score["ambiguous_category_count"] == 1
    assert score["missing_category_count"] == 1
    assert score["categorical_complete"] is False
    assert score["categorical_reward"] is None


def _paired_payload():
    context = Context("point", "happiness").to_dict()
    records = []
    for candidate, valence, rmse in (("a", 0.9, 0.05), ("b", 0.5, 0.01)):
        observations = [
            {
                "affect_ratings": {
                    "valence": valence,
                    "arousal": 0.75,
                    "dominance": 0.65,
                },
                "probabilities": PROBABILITIES,
                "category_status": "complete",
            }
            for _ in range(2)
        ]
        records.append(
            {
                "candidate_id": candidate,
                "context": context,
                "seed": 7,
                "observations": observations,
                "feasibility": {
                    "valid_realisation": True,
                    "physically_acceptable": True,
                    "path_length_ratio": 1.0,
                    "joint_limit_error": 0.0,
                    "realisation_rmse": rmse,
                    "max_abs_feature_error": 0.05,
                },
            }
        )
    return {
        "records": records,
        "test_retest_reliability": {
            "status": "ok",
            "repeat_counts": [2, 2],
        },
    }


def test_ablation_rescores_cache_and_marks_inner_reruns():
    output = rescore_paired_ablation(
        _paired_payload(),
        [
            {
                "name": "vad-heavy",
                "settings": {
                    "valence_weight": 0.8,
                    "arousal_weight": 0.1,
                    "dominance_weight": 0.1,
                    "realisation_penalty_weight": 0.0,
                },
            },
            {"name": "optimizer-change", "settings": {"maxiter": 100}},
        ],
    )

    first = output["configurations"][0]
    ranked = sorted(first["records"], key=lambda row: row["vad_score_rank"])
    assert ranked[0]["candidate_id"] == "a"
    assert first["requires_inner_optimizer_rerun"] is False
    assert output["configurations"][1]["requires_inner_optimizer_rerun"] is True
    assert "not equivalent" in output["reward_scale_note"]


def test_ablation_handles_unavailable_raw_feasibility_measurements():
    payload = _paired_payload()
    payload["records"][0]["feasibility"]["path_length_ratio"] = None
    payload["records"][0]["feasibility"]["joint_limit_error"] = None

    output = rescore_paired_ablation(
        payload,
        [{"settings": {"minimum_path_length_ratio": 0.8}}],
    )

    first = output["configurations"][0]["records"][0]
    assert first["requires_inner_optimizer_rerun"] is True
    assert "raw feasibility" in first["rerun_reason"]


def test_stability_penalty_requires_measured_repeatability():
    with pytest.raises(ValueError, match="measured multi-clip"):
        rescore_paired_ablation(
            {"records": [], "test_retest_reliability": {"status": "insufficient_clips"}},
            [{"settings": {"stability_penalty_weight": 0.2}}],
        )


def test_checkpoint_metadata_migration_and_incompatible_policy():
    context = Context("point", target_vad=VAD).to_dict()
    legacy_cem = {
        "cem_state": {},
        "context": context,
        "resume_config": {},
    }
    migrated, changed = migrate_metadata_only_checkpoint(legacy_cem)
    assert changed is True
    validate_checkpoint(migrated, expected_kind="cem", context=context)

    with pytest.raises(CheckpointCompatibilityError, match="observation-schema"):
        migrate_metadata_only_checkpoint({"policy_state_dict": {}})

    policy = {
        "metadata": {
            **checkpoint_metadata("learned_policy", context=context),
            "observation_size": 1,
        }
    }
    with pytest.raises(CheckpointCompatibilityError, match="observation shape"):
        validate_checkpoint(
            policy, expected_kind="learned_policy", context=context
        )


def test_credential_check_reports_location_without_value(tmp_path, monkeypatch):
    secret = "AIza" + "A" * 24
    tracked = tmp_path / "tracked.txt"
    tracked.write_text(f"credential={secret}", encoding="utf-8")
    monkeypatch.setattr(
        "laban_rl.hygiene._tracked_files", lambda root: [tracked]
    )

    result = credential_presence_check(tmp_path)

    assert result["safe"] is False
    assert result["tracked_credential_findings"] == [
        {"path": "tracked.txt", "type": "google_api_key"}
    ]
    assert secret not in json.dumps(result)


def _write_readiness_fixture(root: Path):
    calibration = {
        gesture: {
            feature: {"min": 0.0, "max": 1.0}
            for feature in FEATURE_KEYS
        }
        for gesture in GESTURE_TYPES
    }
    (root / "calibration.json").write_text(json.dumps(calibration))
    matrix = {
        "gestures": ["point"],
        "targets": [{"state": "happiness"}],
        "seeds": [7],
        "candidate_profiles": {
            "candidate": {feature: 0.5 for feature in FEATURE_KEYS}
        },
    }
    (root / "matrix.json").write_text(json.dumps(matrix))
    case_id = "point__happiness__seed-7__candidate"
    paired = {
        "records": [
            {
                "candidate_id": case_id,
                "context": Context("point", "happiness").to_dict(),
                "reliability": {
                    "repeat_reliability": {"repeat_count": 2},
                    "categorical_distribution_coverage": 1.0,
                    "categorical_unambiguous_coverage": 0.5,
                    "ambiguous_category_count": 1,
                    "missing_category_count": 0,
                },
            }
        ],
        "test_retest_reliability": {"status": "ok"},
    }
    (root / "paired.json").write_text(json.dumps(paired))
    (root / "real.csv").write_text("gesture,rate\npoint,1\n")
    cache = root / "cache" / "ab"
    cache.mkdir(parents=True)
    (cache / "entry.json").write_text(
        json.dumps({"identity": {"cache_format_version": 2}})
    )
    (root / ".gitignore").write_text(
        "\n".join(
            (
                "outputs/",
                "*cache*/",
                ".env",
                "*.key",
                "*.pem",
                "*.mp4",
                "*.gif",
                "generated_calibration/",
            )
        )
    )
    return {
        "tests": {"status": "passed"},
        "calibration": "calibration.json",
        "matrix": "matrix.json",
        "paired_results": "paired.json",
        "realisability_summary": "real.csv",
        "cache": "cache",
        "minimum_repeats": 2,
    }


def test_readiness_audit_pass_and_fail(tmp_path, monkeypatch):
    manifest = _write_readiness_fixture(tmp_path)
    monkeypatch.setattr(
        "laban_rl.readiness.credential_presence_check",
        lambda root: {
            "safe": True,
            "configured_environment_variables": [],
            "tracked_credential_findings": [],
            "local_sensitive_files": [],
        },
    )

    passed = audit_readiness(tmp_path, manifest)
    assert passed["ready"] is True

    manifest["tests"]["status"] = "failed"
    failed = audit_readiness(tmp_path, manifest)
    assert failed["ready"] is False
    assert "tests" in failed["blocking_failures"]

    (tmp_path / ".gitignore").unlink()
    missing_ignore = audit_readiness(tmp_path, manifest)
    assert "artifact_ignores" in missing_ignore["blocking_failures"]
