import json
import tempfile
from pathlib import Path

import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban

from laban_rl.config import FEATURE_KEYS
from laban_rl.io_utils import load_ranges_or_default
from laban_rl.trajectories import make_reference_trajectory
from laban_rl.features import (
    compute_raw_and_norm_features,
    feature_dict_to_array,
    target_dict_to_array,
    get_feature_mask,
)
from laban_rl.targets import TARGET_PROFILES


def test_feature_computation_returns_expected_keys():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filter_config = laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    ranges = load_ranges_or_default("configs/normalisation_ranges_balanced_3gestures.json")

    q = make_reference_trajectory("point", arm)

    raw, norm = compute_raw_and_norm_features(q, arm, filter_config, ranges)

    for key in FEATURE_KEYS:
        assert key in raw
        assert key in norm


def test_feature_dict_to_array_shape():
    features = {
        "weight": 0.1,
        "time": 0.2,
        "flow_boundness": 0.3,
        "space_indirectness": np.nan,
        "shape_arcness": 0.5,
    }

    arr = feature_dict_to_array(features)

    assert arr.shape == (5,)
    assert np.all(np.isfinite(arr))
    assert arr[3] == 0.0


def test_target_dict_to_array_shape():
    arr = target_dict_to_array(TARGET_PROFILES["confident"])

    assert arr.shape == (5,)
    assert np.all(np.isfinite(arr))


def test_load_gesture_specific_ranges():
    nested_ranges = {
        "balanced": {
            "weight": {"min": 0.01, "max": 1.0},
            "time": {"min": 0.1, "max": 10.0},
            "flow_boundness": {"min": 1.0, "max": 100.0},
            "space_indirectness": {"min": 1.0, "max": 50.0},
            "shape_arcness": {"min": 10.0, "max": 100.0},
        },
        "point": {
            "weight": {"min": 0.02, "max": 0.8},
            "time": {"min": 0.5, "max": 8.0},
            "flow_boundness": {"min": 1.5, "max": 50.0},
            "space_indirectness": {"min": 1.2, "max": 30.0},
            "shape_arcness": {"min": 12.0, "max": 80.0},
        },
        "wave": {
            "weight": {"min": 0.05, "max": 1.1},
            "time": {"min": 0.8, "max": 7.0},
            "flow_boundness": {"min": 2.0, "max": 70.0},
            "space_indirectness": {"min": 1.1, "max": 35.0},
            "shape_arcness": {"min": 14.0, "max": 95.0},
        },
    }

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump(nested_ranges, f)
        temp_path = f.name

    try:
        balanced = load_ranges_or_default(temp_path)
        assert balanced["weight"] == (0.01, 1.0)

        point = load_ranges_or_default(temp_path, gesture="point")
        assert point["weight"] == (0.02, 0.8)

        wave = load_ranges_or_default(temp_path, gesture="wave")
        assert wave["weight"] == (0.05, 1.1)
    finally:
        try:
            Path(temp_path).unlink()
        except OSError:
            pass


def test_wave_masks_space():
    features = {
        "weight": 0.1,
        "time": 0.2,
        "flow_boundness": 0.3,
        "space_indirectness": 0.4,
        "shape_arcness": 0.5,
    }

    mask = get_feature_mask("wave", features)

    assert mask.shape == (5,)
    assert mask[FEATURE_KEYS.index("space_indirectness")] == 0.0
