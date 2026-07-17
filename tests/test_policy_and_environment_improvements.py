"""Tests for policy learning improvements: RMSE gating, entropy decay, reward logging."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


class TestRMSEGating:
    """Test RMSE-based policy update gating."""

    def test_update_with_low_rmse(self):
        """Update should proceed normally when RMSE is low."""
        policy = ContinuousContextualBanditPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        baseline = ContextRewardBaseline()

        context = BanditContext("point", "confident")
        sample = policy.sample_action(context)

        # Initialize baseline
        baseline.update(context, 0.5)

        # Update with low RMSE (should proceed)
        stats = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            sample=sample,
            reward=0.5,
            baseline=baseline,
            realisation_rmse=0.02,
            rmse_normal_threshold=0.05,
            rmse_skip_threshold=0.10,
            skip_high_rmse_updates=True,
        )

        assert stats["policy_updated"]
        assert not stats["rmse_skipped"]

    def test_skip_update_with_high_rmse(self):
        """Update should skip when RMSE exceeds skip threshold."""
        policy = ContinuousContextualBanditPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        baseline = ContextRewardBaseline()

        context = BanditContext("point", "confident")
        sample = policy.sample_action(context)

        # Initialize baseline
        baseline.update(context, 0.5)

        # Update with high RMSE (should skip)
        stats = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            sample=sample,
            reward=0.5,
            baseline=baseline,
            realisation_rmse=0.15,
            rmse_normal_threshold=0.05,
            rmse_skip_threshold=0.10,
            skip_high_rmse_updates=True,
        )

        assert not stats["policy_updated"]
        assert stats["rmse_skipped"]

    def test_rmse_gating_disabled(self):
        """When disabled, high RMSE should not skip update."""
        policy = ContinuousContextualBanditPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        baseline = ContextRewardBaseline()

        context = BanditContext("point", "confident")
        sample = policy.sample_action(context)

        # Initialize baseline
        baseline.update(context, 0.5)

        # Update with high RMSE but skip disabled
        stats = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            sample=sample,
            reward=0.5,
            baseline=baseline,
            realisation_rmse=0.15,
            rmse_normal_threshold=0.05,
            rmse_skip_threshold=0.10,
            skip_high_rmse_updates=False,
        )

        # Should still update when gating is disabled
        assert stats["policy_updated"]
        assert not stats["rmse_skipped"]


class TestEntropyInUpdate:
    """Test entropy computation and bonus in policy updates."""

    def test_entropy_in_policy_sample(self):
        """Policy sample should include entropy from distribution."""
        policy = ContinuousContextualBanditPolicy()
        context = BanditContext("point", "confident")
        sample = policy.sample_action(context)

        # Entropy should be computed (can be negative in log space for Beta).
        assert hasattr(sample, "entropy")
        entropy_val = float(sample.entropy.detach().cpu())
        # Beta entropy is computed via digamma function and can be negative,
        # but should be finite.
        assert np.isfinite(entropy_val)

    def test_entropy_weight_zero(self):
        """When entropy_weight=0, entropy should not affect loss."""
        policy = ContinuousContextualBanditPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        baseline = ContextRewardBaseline()

        context = BanditContext("point", "confident")
        sample = policy.sample_action(context)
        baseline.update(context, 0.5)

        stats = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            sample=sample,
            reward=0.7,
            baseline=baseline,
            entropy_weight=0.0,
        )

        # entropy_weight should be logged as 0.0
        assert stats["effective_entropy_weight"] == 0.0

    def test_entropy_weight_nonzero(self):
        """When entropy_weight > 0, should be included in update."""
        policy = ContinuousContextualBanditPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        baseline = ContextRewardBaseline()

        context = BanditContext("point", "confident")
        sample = policy.sample_action(context)
        baseline.update(context, 0.5)

        stats = reinforce_update(
            policy=policy,
            optimizer=optimizer,
            sample=sample,
            reward=0.7,
            baseline=baseline,
            entropy_weight=0.02,
        )

        # entropy_weight should be logged
        assert stats["effective_entropy_weight"] == 0.02


class TestInformedProfiles:
    """Test that all 12 gesture-state informed profiles exist."""

    def test_all_12_profiles_present(self):
        """All combinations of (wave|reach|point) x (confident|calm|hesitant|friendly) exist."""
        from scripts.train_real_continuous_contextual_bandit import INFORMED_PROFILES

        expected_keys = [
            f"{gesture}::{state}"
            for gesture in ["wave", "reach", "point"]
            for state in ["confident", "calm", "hesitant", "friendly"]
        ]

        for key in expected_keys:
            assert key in INFORMED_PROFILES, f"Missing informed profile: {key}"

        # Should have exactly 12 profiles
        assert len(INFORMED_PROFILES) == 12

    def test_profile_values_in_range(self):
        """All profile values should be in (0, 1)."""
        from scripts.train_real_continuous_contextual_bandit import INFORMED_PROFILES

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
        context = BanditContext("point", "confident")

        for _ in range(10):
            sample = policy.sample_action(context)
            action = sample.action_tensor
            # All dimensions should be in (0, 1)
            assert torch.all(action > 0.0)
            assert torch.all(action < 1.0)

    def test_beta_policy_mean_profile(self):
        """Mean profile should be obtainable and in bounds."""
        policy = ContinuousContextualBanditPolicy()
        context = BanditContext("point", "confident")
        mean = policy.mean_profile(context)

        assert len(mean) == 5
        for key, value in mean.items():
            assert 0.0 < value < 1.0

    def test_beta_policy_std_profile(self):
        """Approximate std profile should be obtainable and non-negative."""
        policy = ContinuousContextualBanditPolicy()
        context = BanditContext("point", "confident")
        std = policy.approximate_action_std_profile(context)

        assert len(std) == 5
        for key, value in std.items():
            assert value >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
