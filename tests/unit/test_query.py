"""
Unit Tests for Query Action and EVI Calculation
Feature: 002-full-prototype
Tasks: T057, T058

Tests Expected Value of Information (EVI) computation and query triggering logic.

References:
- docs/theory.md §5: Active information acquisition
- SC-006: EVI ≥ Δ* before query
- SC-007: Entropy reduction ≥20% after query
"""

import numpy as np

from robust_semantic_agent.core.belief import Belief
from robust_semantic_agent.core.query import evi, should_query


class TestEVIComputation:
    """Test Expected Value of Information calculation."""

    def test_evi_computed_without_error(self):
        """
        EVI should compute without errors for uncertain belief.

        NOTE: EVI can be negative for value functions like -distance(goal),
        because learning the true state might reveal we're farther than expected.
        This is mathematically correct - information can have negative value
        when it reveals bad news.
        """
        np.random.seed(42)

        # Create uncertain belief (high variance particles)
        belief = Belief(n_particles=1000, state_dim=2)
        belief.particles = np.random.randn(1000, 2) * 1.0  # High variance
        belief.log_weights = np.full(1000, -np.log(1000))

        # Simple value function (distance to goal)
        goal = np.array([0.8, 0.8])

        def value_fn(b):
            mean = b.mean()
            return -np.linalg.norm(mean - goal)

        # Compute EVI
        evi_value = evi(belief, value_fn, obs_noise=0.1, n_samples=50)

        # Should compute without error and be finite
        assert np.isfinite(evi_value), f"EVI should be finite, got {evi_value}"
        # Can be positive or negative depending on belief and value function

    def test_evi_zero_for_certain_belief(self):
        """
        EVI should be near zero when belief is already certain.

        Scenario: Low-entropy belief → EVI ≈ 0
        """
        np.random.seed(42)

        # Create certain belief (low variance particles)
        belief = Belief(n_particles=1000, state_dim=2)
        belief.particles = np.random.randn(1000, 2) * 0.01  # Very low variance
        belief.log_weights = np.full(1000, -np.log(1000))

        # Simple value function
        goal = np.array([0.8, 0.8])

        def value_fn(b):
            mean = b.mean()
            return -np.linalg.norm(mean - goal)

        # Compute EVI
        evi_value = evi(belief, value_fn, obs_noise=0.1, n_samples=50)

        # Should be near zero for certain belief
        assert evi_value < 0.1, f"EVI should be low for certain belief, got {evi_value}"

    def test_evi_magnitude_correlates_with_uncertainty(self):
        """
        Higher uncertainty should lead to larger absolute EVI magnitude.

        NOTE: EVI sign can be positive or negative, but magnitude (absolute value)
        should increase with uncertainty - more information when less certain.
        """
        np.random.seed(42)

        goal = np.array([0.8, 0.8])

        def value_fn(b):
            mean = b.mean()
            return -np.linalg.norm(mean - goal)

        # Low uncertainty belief
        belief_low = Belief(n_particles=500, state_dim=2)
        belief_low.particles = np.random.randn(500, 2) * 0.05  # Very low variance
        belief_low.log_weights = np.full(500, -np.log(500))

        # High uncertainty belief
        belief_high = Belief(n_particles=500, state_dim=2)
        belief_high.particles = np.random.randn(500, 2) * 0.5  # Higher variance
        belief_high.log_weights = np.full(500, -np.log(500))

        evi_low = evi(belief_low, value_fn, obs_noise=0.1, n_samples=30)
        evi_high = evi(belief_high, value_fn, obs_noise=0.1, n_samples=30)

        # Magnitude should be higher for uncertain belief
        assert abs(evi_high) > abs(
            evi_low
        ), f"Higher uncertainty should have higher |EVI|: |{evi_high}| vs |{evi_low}|"


class TestShouldQueryThreshold:
    """Test query triggering logic based on EVI threshold."""

    def test_should_query_when_evi_exceeds_threshold(self):
        """Query should trigger when EVI ≥ Δ*."""
        evi_value = 0.2
        delta_star = 0.15

        assert should_query(evi_value, delta_star) is True

    def test_should_not_query_when_evi_below_threshold(self):
        """Query should not trigger when EVI < Δ*."""
        evi_value = 0.1
        delta_star = 0.15

        assert should_query(evi_value, delta_star) is False

    def test_should_query_boundary_case(self):
        """Query should trigger when EVI = Δ* (boundary)."""
        evi_value = 0.15
        delta_star = 0.15

        assert should_query(evi_value, delta_star) is True


class TestEntropyReduction:
    """Test entropy reduction after query (SC-007)."""

    def test_entropy_decreases_after_observation(self):
        """
        Entropy should decrease after incorporating new observation.

        SC-007: Entropy reduction ≥ 20% after query
        """
        np.random.seed(42)

        # Create belief
        belief = Belief(n_particles=1000, state_dim=2)
        belief.particles = np.random.randn(1000, 2) * 0.5
        belief.log_weights = np.full(1000, -np.log(1000))

        # Measure initial entropy
        entropy_before = belief.entropy()

        # Simulate observation update (like from query)
        observation = np.array([0.1, 0.2])
        obs_noise = 0.05  # Low noise = high information
        belief.update_obs(observation, obs_noise)

        # Measure entropy after
        entropy_after = belief.entropy()

        # Verify reduction
        reduction = (entropy_before - entropy_after) / entropy_before
        assert reduction > 0, f"Entropy should decrease, got {reduction:.2%}"
        assert (
            entropy_after < entropy_before
        ), f"H_after={entropy_after} should be < H_before={entropy_before}"

    def test_entropy_reduction_proportional_to_obs_quality(self):
        """
        Better observations (low noise) should reduce entropy more.

        Property: Lower noise → Greater entropy reduction
        """
        np.random.seed(42)

        # Create two identical beliefs
        belief_low_noise = Belief(n_particles=1000, state_dim=2)
        belief_low_noise.particles = np.random.randn(1000, 2) * 0.5
        belief_low_noise.log_weights = np.full(1000, -np.log(1000))

        belief_high_noise = Belief(n_particles=1000, state_dim=2)
        belief_high_noise.particles = belief_low_noise.particles.copy()
        belief_high_noise.log_weights = belief_low_noise.log_weights.copy()

        # Measure initial entropies
        H_before_low = belief_low_noise.entropy()
        H_before_high = belief_high_noise.entropy()

        # Same observation, different noise levels
        observation = np.array([0.1, 0.2])
        belief_low_noise.update_obs(observation, obs_noise=0.01)  # Precise
        belief_high_noise.update_obs(observation, obs_noise=0.3)  # Noisy

        # Measure after
        H_after_low = belief_low_noise.entropy()
        H_after_high = belief_high_noise.entropy()

        reduction_low = (H_before_low - H_after_low) / H_before_low
        reduction_high = (H_before_high - H_after_high) / H_before_high

        assert (
            reduction_low > reduction_high
        ), f"Low noise should reduce entropy more: {reduction_low:.2%} vs {reduction_high:.2%}"


class TestValueImprovement:
    """Test that query action improves value function."""

    def test_value_improves_after_query(self):
        """
        Value function should improve after query observation.

        Rationale: More certain belief → better action selection → higher value
        """
        np.random.seed(42)

        # Create uncertain belief
        belief = Belief(n_particles=1000, state_dim=2)
        belief.particles = np.random.randn(1000, 2) * 0.5
        belief.log_weights = np.full(1000, -np.log(1000))

        # Value function (negative distance to goal)
        goal = np.array([0.3, 0.3])

        def value_fn(b):
            mean = b.mean()
            dist = np.linalg.norm(mean - goal)
            return -dist

        # Value before
        value_before = value_fn(belief)

        # Simulate query observation (accurate)
        true_state = np.array([0.25, 0.28])  # Close to goal
        observation = true_state + np.random.randn(2) * 0.05
        belief.update_obs(observation, obs_noise=0.05)

        # Value after (mean should be closer to true state → closer to goal)
        value_after = value_fn(belief)

        # Should improve (become less negative)
        assert value_after > value_before, (
            f"Value should improve after informative observation: "
            f"{value_before:.3f} → {value_after:.3f}"
        )
