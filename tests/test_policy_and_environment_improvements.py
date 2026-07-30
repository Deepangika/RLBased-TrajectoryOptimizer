"""Tests for policy learning improvements: RMSE gating, entropy decay, reward logging."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

for candidate in [PROJECT_ROOT, SRC_DIR]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from laban_rl.perceptual_bandit.environment import (
    Context,
    EnvironmentRewardConfig,
    MockNoisyPerceptualEvaluator,
)
from laban_rl.perceptual_bandit.policy import (
    BanditContext,
    ContextRewardBaseline,
    ContinuousContextualBanditPolicy,
    reinforce_update,
)


class TestRewardMarginMode:
    """Test raw vs clipped margin modes."""

    def test_reward_margin_mode_raw(self):
        config = EnvironmentRewardConfig(
            reward_margin_mode="raw",
            clip_negative_margin=False,
        )
        config.validate()
        assert config.reward_margin_mode == "raw"
        assert not config.clip_negative_margin

    def test_reward_margin_mode_clipped(self):
        config = EnvironmentRewardConfig(
            reward_margin_mode="clipped",
        )
        config.validate()
        assert config.reward_margin_mode == "clipped"

    def test_invalid_reward_margin_mode(self):
        config = EnvironmentRewardConfig(
            reward_margin_mode="invalid",
        )
        with pytest.raises(ValueError, match="reward_margin_mode"):
            config.validate()

    def test_vad_is_default_reward_mode(self):
        config = EnvironmentRewardConfig()
        config.validate()
        assert config.perceptual_reward_mode == "vad"


class TestEntropyInUpdate:
    """Test entropy computation and bonus in policy updates."""

    def test_entropy_in_policy_sample(self):
        """Policy sample should include entropy from distribution."""
        policy = ContinuousContextualBanditPolicy()
        context = BanditContext("point", "happiness")
        sample = policy.sample_action(context)

        # Entropy should be computed (can be negative in log space for Beta).
        assert hasattr(sample, "entropy")
        entropy_val = float(sample.entropy.detach().cpu())
        # Beta entropy is computed via digamma function and can be negative,
        # but should be finite.
        assert np.isfinite(entropy_val)


class TestInformedProfiles:
    """Test that informed profiles cover the configured gesture/emotion states."""

    def test_all_12_profiles_present(self):
        """All combinations of (wave|reach|point) x EMOTION_STATES should exist."""
        from scripts.train_cem_contextual_bandit import INFORMED_PROFILES
        from laban_rl.config import EMOTION_STATES, GESTURE_TYPES

        expected_keys = [
            f"{gesture}::{state}"
            for gesture in GESTURE_TYPES
            for state in EMOTION_STATES
        ]

        for key in expected_keys:
            assert key in INFORMED_PROFILES, f"Missing informed profile: {key}"

        # The dict may include aliases/extra priors, but all configured states must exist.
        assert len(INFORMED_PROFILES) >= len(expected_keys)

    def test_profile_values_in_range(self):
        """All profile values should be in (0, 1)."""
        from scripts.train_cem_contextual_bandit import INFORMED_PROFILES

        for key, profile in INFORMED_PROFILES.items():
            for feature, value in profile.items():
                assert (
                    0.0 < value < 1.0
                ), f"{key}::{feature} value {value} not in (0, 1)"


class TestBetaPolicyPreservation:
    """Verify Beta policy is intact and working correctly."""

    def test_beta_policy_bounded_sampling(self):
        """Beta policy samples should be in (0, 1)."""
        policy = ContinuousContextualBanditPolicy()
        context = BanditContext("point", "happiness")

        for _ in range(10):
            sample = policy.sample_action(context)
            action = sample.action_tensor
            # All dimensions should be in (0, 1)
            assert torch.all(action > 0.0)
            assert torch.all(action < 1.0)

    def test_beta_policy_mean_profile(self):
        """Mean profile should be obtainable and in bounds."""
        policy = ContinuousContextualBanditPolicy()
        context = BanditContext("point", "happiness")
        mean = policy.mean_profile(context)

        assert len(mean) == 5
        for key, value in mean.items():
            assert 0.0 < value < 1.0

    def test_beta_policy_std_profile(self):
        """Approximate std profile should be obtainable and non-negative."""
        policy = ContinuousContextualBanditPolicy()
        context = BanditContext("point", "happiness")
        std = policy.approximate_action_std_profile(context)

        assert len(std) == 5
        for key, value in std.items():
            assert value >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
