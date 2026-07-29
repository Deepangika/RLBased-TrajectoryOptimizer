import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban

from laban_rl.io_utils import load_ranges_or_default
from laban_rl.evaluation import run_random_search


def test_random_search_returns_valid_result():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filter_config = laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    ranges = load_ranges_or_default("configs/normalisation_ranges_balanced_3gestures.json")

    result = run_random_search(
        gesture="point",
        target_name="happiness",
        n_trials=5,
        arm=arm,
        filter_config=filter_config,
        ranges=ranges,
    )

    assert result is not None
    assert np.isfinite(result["reward"])
    assert result["action"].shape == (6,)
    assert result["q_ref"].shape == result["q_var"].shape
    assert "style_error" in result["reward_info"]
