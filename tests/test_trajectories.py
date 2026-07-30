import numpy as np
import robust_laban_normalisation_balanced_3gestures as laban
from laban_rl.config import GESTURE_TYPES, JointLimits
from laban_rl.trajectories import make_reference_trajectory

def test_reference_trajectory_shape():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    limits = JointLimits()
    for gesture in GESTURE_TYPES:
        q = make_reference_trajectory(gesture, arm)
        assert isinstance(q, np.ndarray)
        assert q.shape == (160, 2)
        assert np.all(np.isfinite(q))
        assert np.all((limits.shoulder_min <= q[:, 0]) & (q[:, 0] <= limits.shoulder_max))
        assert np.all((limits.elbow_min <= q[:, 1]) & (q[:, 1] <= limits.elbow_max))


def test_new_gestures_are_distinct_and_non_static():
    arm = laban.ArmConfig(n_points=160, duration=2.0, l1=0.30, l2=0.25)
    trajectories = [
        make_reference_trajectory(gesture, arm)
        for gesture in ("circle", "beckon", "celebratory_pump")
    ]

    for q in trajectories:
        assert np.max(np.ptp(q, axis=0)) > 0.25

    assert not np.allclose(trajectories[0], trajectories[1])
    assert not np.allclose(trajectories[1], trajectories[2])
