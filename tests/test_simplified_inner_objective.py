"""Regression tests for the compact inner-optimiser objective."""
from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "direct_laban_feature_optimizer_spatiotemporal_final.py"
SPEC = importlib.util.spec_from_file_location("direct_inner_optimizer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _args() -> Namespace:
    return Namespace(
        nearest_path_weight=1.5,
        smooth_weight=0.01,
        minimum_path_length_ratio=0.70,
        maximum_path_length_ratio=1.30,
        endpoint_tolerance=0.08,
        direction_tolerance=0.25,
        constraint_penalty_weight=100.0,
    )


def _terms(**overrides: float) -> dict[str, float]:
    values = {
        "nearest_path_mse": 0.02,
        "path_length_ratio": 1.0,
        "endpoint_error": 0.04,
        "direction_error": 0.10,
    }
    values.update(overrides)
    return values


def test_compact_objective_contains_only_lma_path_and_jerk() -> None:
    result = MODULE.compose_loss_components(
        feature_loss=0.20,
        smoothness_error=2.0,
        joint_limit_error=0.0,
        terms=_terms(),
        args=_args(),
    )
    assert result["objective_lma"] == 0.20
    assert result["objective_path"] == 0.03
    assert result["objective_jerk"] == 0.02
    assert result["objective_loss"] == 0.25
    assert result["constraint_penalty"] == 0.0
    assert result["total_loss"] == result["objective_loss"]


def test_constraint_violations_are_separate_from_objective() -> None:
    result = MODULE.compose_loss_components(
        feature_loss=0.20,
        smoothness_error=2.0,
        joint_limit_error=0.01,
        terms=_terms(
            path_length_ratio=1.40,
            endpoint_error=0.18,
            direction_error=0.35,
        ),
        args=_args(),
    )
    assert result["objective_loss"] == 0.25
    assert result["constraint_path_length"] > 0.0
    assert result["constraint_endpoint"] > 0.0
    assert result["constraint_direction"] > 0.0
    assert result["constraint_joint_limits"] > 0.0
    assert result["constraint_penalty"] > 0.0
    assert result["total_loss"] > result["objective_loss"]
