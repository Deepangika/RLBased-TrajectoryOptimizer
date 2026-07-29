"""Noise-aware Cross-Entropy Method for continuous Laban profiles."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

from laban_rl.config import FEATURE_KEYS


@dataclass(frozen=True)
class Elite:
    profile: dict[str, float]
    reward: float
    round_index: int


class CEMOptimizer:
    """Diagonal-Beta CEM with current-batch elites and smoothed updates."""

    def __init__(
        self,
        initial_profile: Mapping[str, float] | None = None,
        initial_width: float = 0.15,
        *,
        seed: int = 7,
        smoothing: float = 0.7,
        min_std: float = 0.03,
        max_std: float = 0.40,
        min_elites: int = 3,
    ) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1].")
        if not 0.0 < min_std <= max_std:
            raise ValueError("Require 0 < min_std <= max_std.")
        if min_elites < 2:
            raise ValueError("min_elites must be at least 2.")

        self.feature_keys = tuple(FEATURE_KEYS)
        source = initial_profile or {key: 0.5 for key in self.feature_keys}
        self.beta_params = {
            key: {
                "mean": float(np.clip(source.get(key, 0.5), 0.01, 0.99)),
                "std": float(np.clip(initial_width, min_std, max_std)),
            }
            for key in self.feature_keys
        }
        self.smoothing = float(smoothing)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.min_elites = int(min_elites)
        self.elites: list[Elite] = []
        self.round_counter = 0
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _beta_parameters(mean: float, std: float) -> tuple[float, float] | None:
        variance = std**2
        maximum_variance = mean * (1.0 - mean)
        if variance <= 1e-12 or variance >= maximum_variance:
            return None
        concentration = maximum_variance / variance - 1.0
        return (
            float(np.clip(mean * concentration, 0.1, 1000.0)),
            float(np.clip((1.0 - mean) * concentration, 0.1, 1000.0)),
        )

    def sample_batch(self, batch_size: int) -> list[dict[str, float]]:
        if batch_size < self.min_elites:
            raise ValueError(
                f"batch_size={batch_size} must be at least min_elites={self.min_elites}."
            )
        batch = []
        for _ in range(batch_size):
            profile = {}
            for feature in self.feature_keys:
                mean = self.beta_params[feature]["mean"]
                std = self.beta_params[feature]["std"]
                params = self._beta_parameters(mean, std)
                if params is None:
                    value = self.rng.normal(mean, std)
                else:
                    value = self.rng.beta(*params)
                profile[feature] = float(np.clip(value, 0.0, 1.0))
            batch.append(profile)
        return batch

    def update_elites(
        self,
        candidates: Sequence[tuple[Mapping[str, float], float]],
        elite_fraction: float = 0.4,
    ) -> None:
        """Fit to current-round elites only; historical winners are not recycled."""
        if not candidates:
            raise ValueError("At least one candidate is required.")
        if not 0.0 < elite_fraction <= 1.0:
            raise ValueError("elite_fraction must be in (0, 1].")
        if any(not np.isfinite(float(reward)) for _, reward in candidates):
            raise ValueError("All candidate rewards must be finite.")

        elite_count = max(self.min_elites, int(np.ceil(len(candidates) * elite_fraction)))
        elite_count = min(elite_count, len(candidates))
        ranked = sorted(candidates, key=lambda item: float(item[1]), reverse=True)
        selected = ranked[:elite_count]
        self.elites = [
            Elite(dict(profile), float(reward), self.round_counter + 1)
            for profile, reward in selected
        ]

        alpha = self.smoothing
        for feature in self.feature_keys:
            values = np.asarray([profile[feature] for profile, _ in selected], dtype=float)
            elite_mean = float(np.mean(values))
            elite_std = float(np.std(values, ddof=0))
            old = self.beta_params[feature]
            old["mean"] = float(np.clip((1.0 - alpha) * old["mean"] + alpha * elite_mean, 0.01, 0.99))
            old["std"] = float(np.clip((1.0 - alpha) * old["std"] + alpha * elite_std,
                                       self.min_std, self.max_std))
        self.round_counter += 1

    def decay_exploration(self, rate: float) -> None:
        if not 0.0 < rate <= 1.0:
            raise ValueError("Exploration decay rate must be in (0, 1].")
        for params in self.beta_params.values():
            params["std"] = float(np.clip(params["std"] * rate, self.min_std, self.max_std))

    def get_mean_profile(self) -> dict[str, float]:
        return {feature: params["mean"] for feature, params in self.beta_params.items()}

    def get_std_profile(self) -> dict[str, float]:
        return {feature: params["std"] for feature, params in self.beta_params.items()}

    def get_best_elite(self) -> Elite | None:
        return max(self.elites, key=lambda elite: elite.reward) if self.elites else None

    def diagnostics(self) -> dict[str, float | int | dict[str, float]]:
        """Return generation-level quantities required for convergence plots."""
        std_profile = self.get_std_profile()
        rewards = np.asarray([elite.reward for elite in self.elites], dtype=float)
        return {
            "round": int(self.round_counter),
            "mean_profile_std": float(np.mean(list(std_profile.values()))),
            "min_profile_std": float(np.min(list(std_profile.values()))),
            "max_profile_std": float(np.max(list(std_profile.values()))),
            "log_search_volume": float(
                np.sum(np.log(np.maximum(list(std_profile.values()), 1e-12)))
            ),
            "best_elite_reward": (
                float(np.max(rewards)) if rewards.size else float("nan")
            ),
            "mean_elite_reward": (
                float(np.mean(rewards)) if rewards.size else float("nan")
            ),
            "elite_reward_std": (
                float(np.std(rewards, ddof=1)) if rewards.size > 1 else 0.0
            ),
            "profile_mean": self.get_mean_profile(),
            "profile_std": std_profile,
        }

    def state_dict(self) -> dict:
        return {
            "beta_params": self.beta_params,
            "elites": [asdict(elite) for elite in self.elites],
            "round_counter": self.round_counter,
            "rng_state": self.rng.bit_generator.state,
            "smoothing": self.smoothing,
            "min_std": self.min_std,
            "max_std": self.max_std,
            "min_elites": self.min_elites,
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.beta_params = {
            key: {"mean": float(value["mean"]), "std": float(value["std"])}
            for key, value in state["beta_params"].items()
        }
        self.elites = [Elite(**elite) for elite in state.get("elites", [])]
        self.round_counter = int(state.get("round_counter", 0))
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
