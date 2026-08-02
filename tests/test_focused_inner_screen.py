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


def test_target_tracking_weights_contribute_to_inner_loss():
    args = SimpleNamespace(
        weight_target_weight=0.0,
        time_target_weight=0.0,
        flow_boundness_target_weight=1.5,
        space_indirectness_target_weight=0.0,
        shape_arcness_target_weight=0.0,
        out_of_range_weight=0.0,
        preserve_weight=0.0,
        nearest_path_weight=0.0,
        endpoint_weight=0.0,
        direction_weight=0.0,
        path_length_weight=0.0,
        detour_weight=0.0,
        max_dev_weight=0.0,
        smooth_weight=0.0,
        joint_limit_weight=0.0,
        coeff_weight=0.0,
        time_coeff_weight=0.0,
        time_warp_weight=0.0,
        time_roughness_weight=0.0,
    )
    components = compose_loss_components(
        feature_loss=0.2,
        target_tracking_errors={
            "weight": 0.0,
            "time": 0.0,
            "flow_boundness": 0.04,
            "space_indirectness": 0.0,
            "shape_arcness": 0.0,
        },
        out_of_range_error=0.0,
        joint_preservation_error=0.0,
        smoothness_error=0.0,
        joint_limit_error=0.0,
        spatial_coeff_error=0.0,
        timing_coeff_error=0.0,
        terms={
            "nearest_path_mse": 0.0,
            "endpoint_error": 0.0,
            "direction_error": 0.0,
            "path_length_error": 0.0,
            "detour_error": 0.0,
            "max_dev_error": 0.0,
            "time_warp_deviation": 0.0,
            "time_roughness": 0.0,
        },
        args=args,
    )

    assert components["target_flow_boundness"] == pytest.approx(0.06)
    assert components["total_loss"] == pytest.approx(0.26)


def test_atomic_json_write_replaces_existing_result(tmp_path):
    path = tmp_path / "raw_results.json"
    path.write_text("old", encoding="utf-8")

    write_json_atomic(path, [{"case_id": "new"}])

    assert path.read_text(encoding="utf-8").startswith("[")
    assert not list(tmp_path.glob("*.tmp"))
