import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban

from laban_rl.envs import LabanTrajectoryStylerEnv
from laban_rl.io_utils import load_ranges_or_default


def test_env_single_step():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    filter_config = laban.FilterConfig(enabled=True, cutoff_hz=5.0, order=4)
    ranges = load_ranges_or_default("configs/normalisation_ranges_balanced_3gestures.json")

    env = LabanTrajectoryStylerEnv(
        gestures=["point"],
        targets=["happiness"],
        arm=arm,
        filter_config=filter_config,
        ranges=ranges,
    )

    obs, info = env.reset(seed=123)
    action = np.zeros(6)

    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert obs.shape == (26,)
    assert next_obs.shape == (26,)
    assert terminated is True
    assert truncated is False
    assert np.isfinite(reward)
    assert "var_norm" in step_info
    assert step_info["gesture_type"] == "point"
    assert step_info["target_name"] == "happiness"
