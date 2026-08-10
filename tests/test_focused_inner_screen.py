from types import SimpleNamespace

import pytest

from scripts.direct_laban_feature_optimizer_spatiotemporal_final import (
    compose_loss_components,
)
from scripts.evaluation.run_focused_inner_screen import (
    aggregate,
    build_cases,
    write_json_atomic,
)
from scripts.evaluation.project_wave_feasible_targets import (
    generate_candidates,
    profile_distance,
    select_projection,
    validate_publication_scope,
)
from scripts.debug.evaluate_fixed_profile_holdout import recorded_optimizer_overrides
from scripts.train_cem_contextual_bandit import (
    _build_optimiser_overrides,
    load_projected_initial_profile,
)


def test_focused_screen_grid_sizes_and_order():
    fear = build_cases("beckon-fear")
    wave = build_cases("wave")

    assert len(fear) == 24
    assert {case["state"] for case in fear} == {"fear"}
    assert {case["timing_basis"] for case in fear} == {5, 6}
    assert len(wave) == 144
    assert {case["state"] for case in wave} == {
        "anger",
        "disgust",
        "fear",
        "sadness",
    }


def test_aggregate_prioritises_seed_robust_feasibility():
    base = {
        "gesture": "beckon",
        "state": "fear",
        "flow_target_weight": 0.1,
        "maxiter": 45,
        "popsize": 5,
        "rmse": 0.05,
        "flow_abs_error": 0.04,
        "time_abs_error": 0.03,
        "shape_abs_error": 0.02,
        "path_length_ratio": 1.0,
        "nearest_path_mse": 0.001,
        "nearest_path_max_dist": 0.01,
        "endpoint_error": 0.01,
        "direction_error": 0.01,
        "smoothness_error": 0.01,
        "joint_limit_error": 0.0,
    }
    rows = []
    for timing_basis, feasible_count in ((5, 2), (6, 3)):
        for index in range(3):
            rows.append(
                {
                    **base,
                    "timing_basis": timing_basis,
                    "strictly_feasible": index < feasible_count,
                    "max_abs_feature_error": 0.08 + 0.01 * index,
                }
            )

    summaries = aggregate(rows)

    assert summaries[0]["timing_basis"] == 6
    assert summaries[0]["all_seeds_strictly_feasible"] is True
    assert summaries[0]["minimum_path_length_ratio"] == pytest.approx(1.0)
    assert summaries[0]["maximum_path_length_ratio"] == pytest.approx(1.0)


def test_legacy_target_tracking_weights_do_not_duplicate_lma_loss():
    args = SimpleNamespace(
        nearest_path_weight=0.0,
        smooth_weight=0.0,
        minimum_path_length_ratio=0.70,
        maximum_path_length_ratio=1.30,
        endpoint_tolerance=0.08,
        direction_tolerance=0.25,
        constraint_penalty_weight=100.0,
    )
    components = compose_loss_components(
        feature_loss=0.2,
        smoothness_error=0.0,
        joint_limit_error=0.0,
        terms={
            "nearest_path_mse": 0.0,
            "endpoint_error": 0.0,
            "direction_error": 0.0,
            "path_length_ratio": 1.0,
        },
        args=args,
    )

    assert components["objective_lma"] == pytest.approx(0.2)
    assert components["objective_loss"] == pytest.approx(0.2)
    assert components["constraint_penalty"] == pytest.approx(0.0)
    assert components["total_loss"] == pytest.approx(0.2)


def test_atomic_json_write_replaces_existing_result(tmp_path):
    path = tmp_path / "raw_results.json"
    path.write_text("old", encoding="utf-8")

    write_json_atomic(path, [{"case_id": "new"}])

    assert path.read_text(encoding="utf-8").startswith("[")
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    ("target_state", "timing_basis"),
    (("fear", 5), ("surprise", 6)),
)
def test_beckon_uses_seed_robust_screen_defaults(target_state, timing_basis):
    args = SimpleNamespace(
        maxiter=45,
        popsize=5,
        local_maxiter=100,
        de_mutation=0.5,
        de_recombination=0.65,
        seed=7,
        target_state=target_state,
    )

    overrides = _build_optimiser_overrides(args, "beckon")

    assert overrides["maxiter"] == 75
    assert overrides["popsize"] == 8
    assert overrides["n_timing_basis"] == timing_basis
    assert overrides["time_target_weight"] == 0.10
    assert overrides["flow_boundness_target_weight"] == 0.10
    assert ("shape_arcness_target_weight" in overrides) == (
        target_state == "fear"
    )


@pytest.mark.parametrize("target_state", ("anger", "disgust", "fear", "sadness"))
def test_difficult_wave_targets_use_projection_validation_budget(target_state):
    args = SimpleNamespace(
        maxiter=45,
        popsize=5,
        local_maxiter=100,
        de_mutation=0.5,
        de_recombination=0.65,
        seed=7,
        target_state=target_state,
        wave_flow_target_weight=0.75,
    )

    overrides = _build_optimiser_overrides(args, "wave")

    assert overrides["maxiter"] == 75
    assert overrides["popsize"] == 8
    assert overrides["n_timing_basis"] == 6
    assert overrides["flow_boundness_target_weight"] == 1.5


def test_projection_candidates_are_deterministic_and_include_original():
    original = {
        "weight": 0.9,
        "time": 0.8,
        "flow_boundness": 0.7,
        "space_indirectness": 0.2,
        "shape_arcness": 0.1,
    }
    reference = {key: 0.5 for key in original}

    first = generate_candidates(original, reference, count=128, seed=7)
    second = generate_candidates(original, reference, count=128, seed=7)

    assert first == second
    assert len(first) == 128
    assert first[0] == original
    assert profile_distance(first[0], original) == 0.0


def test_projection_selects_nearest_three_seed_feasible_profile():
    candidates = [
        {
            "candidate_profile": {"weight": 0.4},
            "strictly_feasible_seeds": 3,
            "equal_weight_distance_to_original": 0.2,
            "worst_max_abs_feature_error": 0.08,
            "mean_nearest_path_mse": 0.01,
        },
        {
            "candidate_profile": {"weight": 0.5},
            "strictly_feasible_seeds": 2,
            "equal_weight_distance_to_original": 0.1,
            "worst_max_abs_feature_error": 0.05,
            "mean_nearest_path_mse": 0.01,
        },
    ]

    selected = select_projection(candidates)

    assert selected is candidates[0]


def test_wave_projection_preserves_original_target_metadata(tmp_path):
    from laban_rl.targets import TARGET_PROFILES

    projected = {
        "weight": 0.8,
        "time": 0.75,
        "flow_boundness": 0.7,
        "space_indirectness": 0.2,
        "shape_arcness": 0.2,
    }
    config = {
        "format_version": 1,
        "gesture": "wave",
        "states": {
            "anger": {
                "original_affect_target": TARGET_PROFILES["anger"],
                "projected_feasible_target": projected,
                "equal_weight_distance_to_original": 0.1,
                "validation": {
                    "strictly_feasible_seeds": 3,
                    "worst_max_abs_feature_error": 0.09,
                    "minimum_path_length_ratio": 0.95,
                    "maximum_path_length_ratio": 1.05,
                },
            }
        },
    }
    path = tmp_path / "projections.json"
    path.write_text(__import__("json").dumps(config), encoding="utf-8")

    profile, metadata = load_projected_initial_profile(
        "wave", "anger", path=path
    )

    assert profile == projected
    assert metadata["original_affect_target"] == TARGET_PROFILES["anger"]
    assert metadata["projected_feasible_target"] == projected


def test_partial_projection_cannot_replace_production_config():
    from scripts.evaluation.project_wave_feasible_targets import PRODUCTION_CONFIG

    with pytest.raises(ValueError, match="requires all three states"):
        validate_publication_scope(["anger"], PRODUCTION_CONFIG)


def test_partial_projection_can_write_diagnostic_config(tmp_path):
    validate_publication_scope(["anger"], tmp_path / "anger_projection.json")


def test_legacy_holdout_requires_recorded_effective_overrides():
    with pytest.raises(RuntimeError, match="cannot be reproduced"):
        recorded_optimizer_overrides({"inner_maxiter": 45})

    overrides = {"maxiter": 75, "popsize": 8}
    assert recorded_optimizer_overrides(
        {"inner_optimizer_overrides": overrides}
    ) == overrides
