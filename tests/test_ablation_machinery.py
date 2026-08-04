"""Regression tests for the 2x2 ablation machinery.

Covers the recalibrated-named Context mode, per-context ablation knobs on
OuterLearningContext, config parsing, override merging semantics, JSON-stable
checkpoint serialisation, and the frozen recalibrated-target artifact.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.affect import target_vad
from laban_rl.perceptual_bandit.environment import Context
from laban_rl.perceptual_bandit.outer_learning import OuterLearningContext

sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))
import run_outer_learning_experiment as runner

RECAL = {"valence": 0.30, "arousal": 0.755, "dominance": 0.30}


class TestRecalibratedContextMode:
    def test_named_mode_unchanged(self) -> None:
        context = Context(gesture="beckon", target_state="fear")
        assert context.target_mode == "named"
        assert context.key == "beckon::fear"
        assert dict(context.target_vad) == target_vad("fear")

    def test_recalibrated_mode_keeps_state_and_overrides_vad(self) -> None:
        context = Context(
            gesture="beckon", target_state="fear", target_vad=RECAL
        )
        assert context.target_mode == "recalibrated"
        assert context.target_state == "fear"
        assert context.target_label == "fear"
        assert dict(context.target_vad) == RECAL

    def test_recalibrated_key_is_isolated_from_named_key(self) -> None:
        named = Context(gesture="beckon", target_state="fear")
        recal = Context(
            gesture="beckon", target_state="fear", target_vad=RECAL
        )
        assert named.key != recal.key
        assert recal.key.startswith("beckon::fear@")

    def test_anchor_vad_with_state_resolves_to_named(self) -> None:
        context = Context(
            gesture="beckon",
            target_state="fear",
            target_vad=target_vad("fear"),
        )
        assert context.target_mode == "named"

    def test_recalibrated_round_trip(self) -> None:
        original = Context(
            gesture="wave",
            target_state="sadness",
            target_vad={"valence": 0.49, "arousal": 0.40, "dominance": 0.30},
        )
        restored = Context.from_dict(original.to_dict())
        assert restored == original
        assert restored.target_mode == "recalibrated"

    def test_recalibrated_metadata_requires_both_fields(self) -> None:
        with pytest.raises(ValueError):
            Context.from_dict(
                {
                    "gesture": "wave",
                    "target_mode": "recalibrated",
                    "target_state": None,
                    "target_vad": RECAL,
                }
            )
        with pytest.raises(ValueError):
            Context.from_dict(
                {
                    "gesture": "wave",
                    "target_mode": "recalibrated",
                    "target_state": "sadness",
                    "target_vad": None,
                }
            )

    def test_direct_vad_mode_unchanged(self) -> None:
        context = Context(
            gesture="wave",
            target_vad={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
        )
        assert context.target_mode == "vad"
        assert context.target_state is None


class TestOuterLearningContextKnobs:
    def test_defaults_are_none_and_key_unchanged(self) -> None:
        context = OuterLearningContext(gesture="beckon", target_state="fear")
        assert context.target_vad_override is None
        assert context.extra_optimiser_overrides is None
        assert context.key == "beckon::fear"

    def test_key_stable_under_overrides_for_matched_seeds(self) -> None:
        plain = OuterLearningContext(gesture="beckon", target_state="fear")
        loaded = OuterLearningContext(
            gesture="beckon",
            target_state="fear",
            target_vad_override=OuterLearningContext.freeze_mapping(RECAL),
            extra_optimiser_overrides=OuterLearningContext.freeze_mapping(
                {"smooth_weight": 1.0}
            ),
        )
        # Candidate seeds derive from context.key, so matched seeds across
        # ablation conditions require the key to ignore the knobs.
        assert loaded.key == plain.key

    def test_freeze_mapping_is_deterministic(self) -> None:
        first = OuterLearningContext.freeze_mapping(
            {"b": 2.0, "a": 1.0, "c": 3.0}
        )
        second = OuterLearningContext.freeze_mapping(
            {"c": 3.0, "a": 1.0, "b": 2.0}
        )
        assert first == second
        assert first == (("a", 1.0), ("b", 2.0), ("c", 3.0))

    def test_dict_views(self) -> None:
        context = OuterLearningContext(
            gesture="wave",
            target_state="sadness",
            target_vad_override=OuterLearningContext.freeze_mapping(RECAL),
        )
        assert context.target_vad_override_dict == RECAL
        assert context.extra_optimiser_overrides_dict is None


class TestRunnerAblationPlumbing:
    def test_contexts_from_stage_parses_knobs(self) -> None:
        stage = {
            "contexts": [
                {
                    "gesture": "beckon",
                    "target_state": "fear",
                    "target_vad_override": RECAL,
                    "extra_optimiser_overrides": {"smooth_weight": 1.0},
                },
                {"gesture": "wave", "target_state": "sadness"},
            ]
        }
        contexts = runner._contexts_from_stage(stage)
        assert contexts[0].target_vad_override_dict == RECAL
        assert contexts[0].extra_optimiser_overrides_dict == {
            "smooth_weight": 1.0
        }
        assert contexts[1].target_vad_override is None
        assert contexts[1].extra_optimiser_overrides is None

    def test_target_context_uses_recalibrated_mode(self) -> None:
        context = OuterLearningContext(
            gesture="beckon",
            target_state="fear",
            target_vad_override=OuterLearningContext.freeze_mapping(RECAL),
        )
        target_context = runner._target_context_for(context)
        assert target_context.target_mode == "recalibrated"
        assert dict(target_context.target_vad) == RECAL
        assert target_context.target_state == "fear"

    def test_target_context_without_override_is_named(self) -> None:
        context = OuterLearningContext(gesture="wave", target_state="sadness")
        target_context = runner._target_context_for(context)
        assert target_context.target_mode == "named"
        assert dict(target_context.target_vad) == target_vad("sadness")

    def test_context_payload_survives_json_round_trip(self) -> None:
        context = OuterLearningContext(
            gesture="beckon",
            target_state="fear",
            target_vad_override=OuterLearningContext.freeze_mapping(RECAL),
            extra_optimiser_overrides=OuterLearningContext.freeze_mapping(
                {"smooth_weight": 1.0}
            ),
        )
        payload = runner._context_payload(context)
        reloaded = json.loads(json.dumps(payload))
        assert reloaded == payload
        restored = runner._outer_context_from_payload(reloaded)
        assert restored == context

    def test_checkpoint_settings_mismatch_on_different_knobs(self) -> None:
        base = dict(
            evaluator_name="mock",
            model="gemini-2.5-flash",
            temperature=0.2,
            rounds_min=5,
            rounds_max=5,
            samples_per_round=8,
            elite_count=3,
            candidate_vlm_repeats=2,
            validation_top_k=4,
            validation_vlm_repeats=5,
            paired_validation_repeats=5,
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
            robust_elite_max_feature_error=0.08,
            seed=7,
        )
        plain = runner._checkpoint_settings_payload(
            context=OuterLearningContext(gesture="beckon", target_state="fear"),
            **base,
        )
        recalibrated = runner._checkpoint_settings_payload(
            context=OuterLearningContext(
                gesture="beckon",
                target_state="fear",
                target_vad_override=OuterLearningContext.freeze_mapping(RECAL),
            ),
            **base,
        )
        assert plain != recalibrated
        # JSON round trips compare equal, so identical resumed settings match.
        assert json.loads(json.dumps(recalibrated)) == recalibrated


class TestFrozenRecalibratedTargets:
    ARTIFACT = ROOT / "configs" / "ablations" / "recalibrated_targets_v1.json"

    @pytest.fixture()
    def artifact(self) -> dict:
        if not self.ARTIFACT.exists():
            pytest.skip("Frozen recalibrated targets artifact absent.")
        return json.loads(self.ARTIFACT.read_text(encoding="utf-8"))

    def test_artifact_is_frozen_with_provenance(self, artifact: dict) -> None:
        assert artifact["status"] == "frozen"
        assert artifact["evidence"]["predates_outer_learning"] is True
        assert artifact["evidence"]["excludes_learned_candidates"] is True
        assert "paired_results.json" in artifact["evidence"]["source_artifact"]

    def test_targets_inside_reachable_envelope(self, artifact: dict) -> None:
        for context in artifact["contexts"]:
            recal = context["recalibrated_target_vad"]
            for axis, bounds in context["reachable_envelope"].items():
                assert bounds["p10"] - 1e-6 <= recal[axis] <= bounds["p90"] + 1e-6

    def test_targets_differ_from_canonical(self, artifact: dict) -> None:
        for context in artifact["contexts"]:
            assert (
                context["recalibrated_target_vad"]
                != context["canonical_target_vad"]
            )
            assert context["axes_moved"]

    def test_condition_b_configs_match_frozen_targets(
        self, artifact: dict
    ) -> None:
        targets = {
            (c["gesture"], c["target_state"]): c["recalibrated_target_vad"]
            for c in artifact["contexts"]
        }
        for gesture, state in (("beckon", "fear"), ("wave", "sadness")):
            config_path = (
                ROOT / "configs" / "ablations"
                / f"ablation_{gesture}_{state}_B_v1.json"
            )
            if not config_path.exists():
                pytest.skip("Condition B config absent.")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            context = config["stages"]["stage_a"]["contexts"][0]
            assert context["target_vad_override"] == targets[(gesture, state)]
            assert "extra_optimiser_overrides" not in context
