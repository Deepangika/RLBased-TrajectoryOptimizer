"""
PPO training helpers for the Laban RL trajectory styler.

This file should contain training logic only.
It should not define rewards, trajectories, features, or visualisation.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

# Reduce common Windows/OpenMP issues caused by PyTorch/NumPy/Matplotlib stacks.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import robust_laban_normalisation_balanced_3gestures as laban

from .envs import LabanTrajectoryStylerEnv


def train_ppo(
    gestures: List[str],
    targets: List[str],
    timesteps: int,
    arm: laban.ArmConfig,
    filter_config: laban.FilterConfig,
    ranges: Dict[str, Tuple[float, float]],
    learning_rate: float = 1e-4,
    n_steps: int = 256,
    batch_size: int = 64,
    gamma: float = 0.95,
    ent_coef: float = 0.001,
    verbose: int = 1,
):
    """
    Train a PPO policy for the one-step Laban trajectory styling environment.

    Defaults use a slightly lower learning rate and entropy coefficient than
    the original prototype because earlier runs showed boundary action
    saturation.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is not installed. Install with:\n"
            "pip install stable-baselines3 gymnasium"
        ) from exc

    env = LabanTrajectoryStylerEnv(
        gestures=gestures,
        targets=targets,
        arm=arm,
        filter_config=filter_config,
        ranges=ranges,
    )

    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
    )

    model.learn(total_timesteps=timesteps)

    return model
