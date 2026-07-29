import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban

from laban_rl.io_utils import load_ranges_or_default
from laban_rl.evaluation import run_random_search
from laban_rl.visualisation import save_outputs


def test_save_outputs_creates_files(tmp_path):
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filter_config = laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    ranges = load_ranges_or_default("configs/normalisation_ranges_balanced_3gestures.json")

    result = run_random_search(
        gesture="point",
        target_name="happiness",
        n_trials=3,
        arm=arm,
        filter_config=filter_config,
        ranges=ranges,
    )

    paths = save_outputs(result, tmp_path, arm)

    expected_names = {
        "best_variant.npz",
        "best_summary.txt",
        "feature_comparison.png",
        "joint_comparison.png",
        "wrist_path_comparison.png",
        "arm_comparison.gif",
    }

    created_names = {path.name for path in paths}

    assert expected_names == created_names

    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
