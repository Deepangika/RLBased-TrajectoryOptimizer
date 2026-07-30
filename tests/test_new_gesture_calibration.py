from argparse import Namespace
import json

import numpy as np

import robust_laban_normalisation_balanced_3gestures as laban
from laban_rl.config import FEATURE_KEYS, JointLimits
from laban_rl.io_utils import load_ranges_or_default
from laban_rl.targets import TARGET_PROFILES
from laban_rl.trajectories import make_reference_trajectory
from scripts.calibrate_new_gesture_ranges import (
    NEW_GESTURES,
    calibrate_new_gesture_ranges,
    update_config,
)
from scripts.direct_laban_feature_optimizer_spatiotemporal_final import (
    make_feature_weights,
)
from scripts.train_cem_contextual_bandit import _build_optimiser_overrides


RANGES_PATH = "configs/normalisation_ranges_balanced_3gestures_by_gesture.json"


def test_checked_in_ranges_cover_every_new_gesture():
    for gesture in NEW_GESTURES:
        ranges = load_ranges_or_default(RANGES_PATH, gesture=gesture)
        assert set(ranges) == set(FEATURE_KEYS)
        for low, high in ranges.values():
            assert np.isfinite(low)
            assert np.isfinite(high)
            assert high > low


def test_small_calibration_sweep_is_deterministic_and_nondegenerate():
    settings = {
        "n_samples": 60,
        "seed": 1234,
        "low_percentile": 5.0,
        "high_percentile": 95.0,
    }
    first = calibrate_new_gesture_ranges(**settings)
    second = calibrate_new_gesture_ranges(**settings)

    assert first == second
    for gesture in NEW_GESTURES:
        assert set(first[gesture]) == set(FEATURE_KEYS)
        assert all(high > low for low, high in first[gesture].values())


def test_config_update_preserves_legacy_ranges(tmp_path):
    with open(RANGES_PATH, encoding="utf-8") as handle:
        original = json.load(handle)
    output = tmp_path / "ranges.json"
    output.write_text(json.dumps(original), encoding="utf-8")
    calibrated = calibrate_new_gesture_ranges(n_samples=30, seed=9)

    update_config(output, calibrated)

    updated = json.loads(output.read_text(encoding="utf-8"))
    for gesture in ("balanced", "wave", "reach", "point"):
        assert updated[gesture] == original[gesture]


def test_calibrated_semantic_variants_span_affective_targets():
    arm = laban.ArmConfig()
    filter_config = laban.FilterConfig()
    limits = JointLimits()
    target = TARGET_PROFILES["happiness"]

    for gesture_index, gesture in enumerate(NEW_GESTURES):
        ranges = load_ranges_or_default(RANGES_PATH, gesture=gesture)
        rng = np.random.default_rng(700 + gesture_index)
        normalised_samples = []
        for _ in range(240):
            q, _ = laban.generate_random_gesture(gesture, arm.n_points, rng)
            assert np.all((limits.shoulder_min <= q[:, 0]) & (q[:, 0] <= limits.shoulder_max))
            assert np.all((limits.elbow_min <= q[:, 1]) & (q[:, 1] <= limits.elbow_max))
            raw = laban.compute_laban_features(q, arm, filter_config)
            normalised_samples.append(
                laban.normalise_laban_features(raw, ranges, clip=False)
            )

        for feature in FEATURE_KEYS:
            values = np.asarray([sample[feature] for sample in normalised_samples])
            assert np.all(np.isfinite(values))
            assert np.ptp(values) > 0.35
            assert np.min(np.abs(values - target[feature])) < 0.12


def test_circle_reference_arcness_is_not_clipped_to_zero():
    arm = laban.ArmConfig()
    filter_config = laban.FilterConfig()
    q_ref = make_reference_trajectory("circle", arm)
    raw = laban.compute_laban_features(q_ref, arm, filter_config)
    circle_ranges = load_ranges_or_default(RANGES_PATH, gesture="circle")
    balanced_ranges = load_ranges_or_default(RANGES_PATH)

    circle_arcness = laban.normalise_laban_features(
        raw,
        circle_ranges,
        clip=True,
    )["shape_arcness"]
    balanced_arcness = laban.normalise_laban_features(
        raw,
        balanced_ranges,
        clip=True,
    )["shape_arcness"]

    assert balanced_arcness == 0.0
    assert 0.10 < circle_arcness < 0.30


def test_pump_optimizer_adds_temporal_capacity_without_spatial_relaxation():
    args = Namespace(
        maxiter=45,
        popsize=5,
        local_maxiter=100,
        de_mutation=0.5,
        de_recombination=0.65,
        seed=17,
    )
    overrides = _build_optimiser_overrides(args, "celebratory_pump")

    assert overrides["n_timing_basis"] == 5
    assert overrides["time_scale"] == 2.0
    assert overrides["flow_weight"] == 3.0
    assert "max_delta" not in overrides
    assert "max_total_delta" not in overrides
    assert make_feature_weights(1.0, 1.0, overrides["flow_weight"])[
        "flow_boundness"
    ] == 3.0
