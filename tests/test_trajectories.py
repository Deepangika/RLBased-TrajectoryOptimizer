import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban
from laban_rl.trajectories import make_reference_trajectory

def test_reference_trajectory_shape():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    for gesture in ["wave", "reach", "point"]:
        q = make_reference_trajectory(gesture, arm)
        assert isinstance(q, np.ndarray)
        assert q.shape == (160, 2)
        assert np.all(np.isfinite(q))
