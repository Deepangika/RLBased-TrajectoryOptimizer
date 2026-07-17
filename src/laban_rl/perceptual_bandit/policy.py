"""Beta-distribution continuous contextual-bandit policy.

Drop into:
    src/laban_rl/perceptual_bandit/policy.py

Why Beta instead of the previous logistic-normal policy:
- actions are naturally bounded inside (0, 1);
- exploration is not distorted by the sigmoid near 0 or 1;
- the policy mean is directly interpretable as the expected Laban profile;
- concentration controls how tightly actions are sampled around that mean.

Each (gesture, target_state) context has its own learnable:
- 5-D mean profile;
- 5-D concentration values.

The first reward for each context initializes the baseline and does not update
the policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta

FEATURE_KEYS: tuple[str, ...] = (
    "weight",
    "time",
    "flow_boundness",
    "space_indirectness",
    "shape_arcness",
)

DEFAULT_GESTURES: tuple[str, ...] = (
    "wave",
    "reach",
    "point",
)

DEFAULT_STATES: tuple[str, ...] = (
    "confident",
    "calm",
    "hesitant",
    "friendly",
    "confused",
    "angry",
)


@dataclass(frozen=True)
class BanditContext:
    gesture: str
    target_state: str

    @property
    def key(self) -> str:
        return f"{self.gesture}::{self.target_state}"


@dataclass
class PolicySample:
    context: BanditContext
    action_tensor: torch.Tensor
    action_profile: dict[str, float]
    log_prob: torch.Tensor
    entropy: torch.Tensor


def _logit(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x) - torch.log1p(-x)


def _validate_profile(
    profile: Mapping[str, float],
) -> np.ndarray:
    missing = [key for key in FEATURE_KEYS if key not in profile]
    if missing:
        raise ValueError(
            f"Profile is missing required keys: {missing}"
        )

    values = np.asarray(
        [float(profile[key]) for key in FEATURE_KEYS],
        dtype=np.float32,
    )

    if not np.all(np.isfinite(values)):
        raise ValueError("Profile contains NaN or infinite values.")

    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError(
            "Policy initialization values must lie strictly inside (0, 1)."
        )

    return values


def _concentration_from_action_std(
    mean: torch.Tensor,
    action_std: float,
    *,
    min_concentration: float,
    max_concentration: float,
) -> torch.Tensor:
    """Choose Beta concentration to approximately match action-space std.

    For Beta(alpha, beta), with:
        mean = alpha / (alpha + beta)
        concentration = alpha + beta = kappa

    variance is:
        mean * (1 - mean) / (kappa + 1)

    Therefore:
        kappa = mean * (1 - mean) / variance - 1

    A single requested action_std cannot be achieved exactly near boundaries
    when the support is bounded. We clamp concentration to keep the policy
    numerically stable and the distribution well-behaved.
    """
    variance = float(action_std) ** 2

    kappa = mean * (1.0 - mean) / variance - 1.0

    return torch.clamp(
        kappa,
        min=min_concentration,
        max=max_concentration,
    )


class ContinuousContextualBanditPolicy(nn.Module):
    """Independent Beta policy head for every context."""

    def __init__(
        self,
        *,
        gestures: Sequence[str] = DEFAULT_GESTURES,
        states: Sequence[str] = DEFAULT_STATES,
        informed_initial_profiles: Mapping[
            str, Mapping[str, float]
        ] | None = None,
        default_initial_profile: Mapping[str, float] | None = None,
        initial_action_std: float = 0.08,
        min_concentration: float = 4.0,
        max_concentration: float = 200.0,
        min_alpha_beta: float = 1.05,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        if initial_action_std <= 0.0:
            raise ValueError("initial_action_std must be positive.")
        if min_concentration <= 0.0:
            raise ValueError("min_concentration must be positive.")
        if max_concentration <= min_concentration:
            raise ValueError(
                "max_concentration must exceed min_concentration."
            )
        if min_alpha_beta <= 0.0:
            raise ValueError("min_alpha_beta must be positive.")

        self.device = torch.device(device)
        self.min_concentration = float(min_concentration)
        self.max_concentration = float(max_concentration)
        self.min_alpha_beta = float(min_alpha_beta)

        self.contexts = tuple(
            BanditContext(gesture, state)
            for gesture in gestures
            for state in states
        )
        self.context_to_index = {
            context.key: index
            for index, context in enumerate(self.contexts)
        }

        n_contexts = len(self.contexts)
        n_features = len(FEATURE_KEYS)

        if default_initial_profile is None:
            default_values = np.full(
                n_features,
                0.5,
                dtype=np.float32,
            )
        else:
            default_values = _validate_profile(
                default_initial_profile
            )

        initial_means = np.tile(
            default_values[None, :],
            (n_contexts, 1),
        )

        if informed_initial_profiles is not None:
            for context_key, profile in informed_initial_profiles.items():
                if context_key not in self.context_to_index:
                    raise KeyError(
                        f"Unknown context key {context_key!r}."
                    )

                initial_means[
                    self.context_to_index[context_key]
                ] = _validate_profile(profile)

        mean_tensor = torch.tensor(
            initial_means,
            dtype=torch.float32,
            device=self.device,
        )

        initial_concentration = _concentration_from_action_std(
            mean_tensor,
            initial_action_std,
            min_concentration=self.min_concentration,
            max_concentration=self.max_concentration,
        )

        # Mean stays in (0, 1) through sigmoid.
        self.mean_logits = nn.Parameter(_logit(mean_tensor))

        # Concentration stays positive through exp, then is clamped in use.
        self.log_concentration = nn.Parameter(
            torch.log(initial_concentration)
        )

        self.to(self.device)

    def _context_index(
        self,
        context: BanditContext,
    ) -> int:
        try:
            return self.context_to_index[context.key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown context {context.key!r}."
            ) from exc

    def _mean_and_concentration(
        self,
        context: BanditContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        index = self._context_index(context)

        mean = torch.sigmoid(self.mean_logits[index])

        concentration = torch.exp(
            self.log_concentration[index]
        )
        concentration = torch.clamp(
            concentration,
            min=self.min_concentration,
            max=self.max_concentration,
        )

        return mean, concentration

    def distribution(
        self,
        context: BanditContext,
    ) -> Beta:
        mean, concentration = self._mean_and_concentration(
            context
        )

        alpha = torch.clamp(
            mean * concentration,
            min=self.min_alpha_beta,
        )
        beta = torch.clamp(
            (1.0 - mean) * concentration,
            min=self.min_alpha_beta,
        )

        return Beta(alpha, beta)

    def sample_action(
        self,
        context: BanditContext,
    ) -> PolicySample:
        distribution = self.distribution(context)

        # Standard REINFORCE score-function sample.
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum()
        entropy = distribution.entropy().sum()

        profile = {
            key: float(value)
            for key, value in zip(
                FEATURE_KEYS,
                action.detach().cpu().tolist(),
            )
        }

        return PolicySample(
            context=context,
            action_tensor=action,
            action_profile=profile,
            log_prob=log_prob,
            entropy=entropy,
        )

    @torch.no_grad()
    def mean_profile(
        self,
        context: BanditContext,
    ) -> dict[str, float]:
        mean, _ = self._mean_and_concentration(context)

        return {
            key: float(value)
            for key, value in zip(
                FEATURE_KEYS,
                mean.cpu().tolist(),
            )
        }

    @torch.no_grad()
    def concentration_profile(
        self,
        context: BanditContext,
    ) -> dict[str, float]:
        _, concentration = self._mean_and_concentration(context)

        return {
            key: float(value)
            for key, value in zip(
                FEATURE_KEYS,
                concentration.cpu().tolist(),
            )
        }

    @torch.no_grad()
    def approximate_action_std_profile(
        self,
        context: BanditContext,
    ) -> dict[str, float]:
        mean, concentration = self._mean_and_concentration(context)

        variance = (
            mean * (1.0 - mean)
            / (concentration + 1.0)
        )
        std = torch.sqrt(torch.clamp(variance, min=0.0))

        return {
            key: float(value)
            for key, value in zip(
                FEATURE_KEYS,
                std.cpu().tolist(),
            )
        }


class ContextRewardBaseline:
    """Per-context exponential moving-average reward baseline."""

    def __init__(
        self,
        momentum: float = 0.50,
    ) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1).")

        self.momentum = float(momentum)
        self.values: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def has_value(
        self,
        context: BanditContext,
    ) -> bool:
        return context.key in self.values

    def value(
        self,
        context: BanditContext,
    ) -> float:
        return float(self.values.get(context.key, 0.0))

    def update(
        self,
        context: BanditContext,
        reward: float,
    ) -> float:
        reward = float(reward)
        key = context.key

        if key not in self.values:
            new_value = reward
        else:
            new_value = (
                self.momentum * self.values[key]
                + (1.0 - self.momentum) * reward
            )

        self.values[key] = float(new_value)
        self.counts[key] = self.counts.get(key, 0) + 1

        return self.values[key]

    def state_dict(self) -> dict:
        return {
            "momentum": self.momentum,
            "values": dict(self.values),
            "counts": dict(self.counts),
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.momentum = float(state["momentum"])
        self.values = {
            str(key): float(value)
            for key, value in state["values"].items()
        }
        self.counts = {
            str(key): int(value)
            for key, value in state["counts"].items()
        }


def reinforce_update(
    *,
    policy: ContinuousContextualBanditPolicy,
    optimizer: torch.optim.Optimizer,
    sample: PolicySample,
    reward: float,
    baseline: ContextRewardBaseline,
    max_grad_norm: float | None = 5.0,
    entropy_weight: float = 0.01,
) -> dict[str, float | bool]:
    """Perform one online REINFORCE update.

    entropy_weight: coefficient for the entropy bonus (encourages exploration).
    A small positive value (0.005-0.05) reduces the variance-collapse problem
    where the Beta distribution concentrates too early on a bad region.
    """
    reward = float(reward)

    if not baseline.has_value(sample.context):
        baseline_after = baseline.update(
            sample.context,
            reward,
        )

        return {
            "reward": reward,
            "baseline_before": reward,
            "baseline_after": baseline_after,
            "advantage": 0.0,
            "policy_loss": 0.0,
            "grad_norm": 0.0,
            "policy_updated": False,
        }

    baseline_before = baseline.value(sample.context)
    advantage = reward - baseline_before
    # Subtract entropy bonus so that maximising entropy reduces loss.
    loss = -sample.log_prob * advantage - entropy_weight * sample.entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_grad_norm,
        )
        grad_norm_value = float(grad_norm)
    else:
        grad_norm_value = float("nan")

    optimizer.step()

    baseline_after = baseline.update(
        sample.context,
        reward,
    )

    return {
        "reward": reward,
        "baseline_before": float(baseline_before),
        "baseline_after": float(baseline_after),
        "advantage": float(advantage),
        "entropy": float(sample.entropy.detach().cpu()),
        "policy_loss": float(loss.detach().cpu()),
        "grad_norm": grad_norm_value,
        "policy_updated": True,
    }
