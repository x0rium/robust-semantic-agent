"""
Unit Tests: Belief Class (Particle Filter)
Feature: 002-full-prototype
Tasks: T018, T019, T020

Tests MUST FAIL initially (TDD Principle V: NON-NEGOTIABLE)
"""

import numpy as np
import pytest


def total_variation_distance(log_weights1, log_weights2):
    """Helper: Compute TV distance between two distributions."""
    w1 = np.exp(log_weights1 - np.max(log_weights1))
    w1 /= np.sum(w1)
    w2 = np.exp(log_weights2 - np.max(log_weights2))
    w2 /= np.sum(w2)
    return 0.5 * np.sum(np.abs(w1 - w2))


@pytest.mark.unit
class TestBeliefObservationUpdate:
    """T018: Test observation likelihood weighting."""

    def test_update_obs_shifts_mean_toward_observation(self):
        """Observation update should shift belief mean toward observed value."""
        from robust_semantic_agent.core.belief import Belief

        # Initialize uniform belief
        belief = Belief(n_particles=1000, state_dim=2)
        belief.particles = np.random.randn(1000, 2) * 2.0  # Wide initial spread

        initial_mean = belief.mean()

        # Observe at specific location
        observation = np.array([1.5, -0.5])
        obs_noise = 0.2

        belief.update_obs(observation, obs_noise)

        updated_mean = belief.mean()

        # Mean should move toward observation
        dist_initial = np.linalg.norm(initial_mean - observation)
        dist_updated = np.linalg.norm(updated_mean - observation)

        assert (
            dist_updated < dist_initial
        ), "Observation update should shift mean toward observation"
        assert dist_updated < 0.5, "Updated mean should be close to observation"

    def test_update_obs_reduces_ess(self):
        """Informative observation should reduce effective sample size."""
        from robust_semantic_agent.core.belief import Belief

        belief = Belief(n_particles=1000, state_dim=2)
        belief.particles = np.random.randn(1000, 2) * 2.0

        ess_before = belief.ess()

        # Very informative observation (low noise)
        observation = np.array([0.0, 0.0])
        obs_noise = 0.1

        belief.update_obs(observation, obs_noise)

        ess_after = belief.ess()

        assert ess_after < ess_before, "Informative observation should reduce ESS"
        assert ess_after < 0.5 * belief.n_particles, "ESS should drop significantly"


@pytest.mark.unit
class TestBeliefCommutativity:
    """T019: Test commutativity of observation and message updates (FR-002)."""

    def test_obs_message_commutativity_tv_distance(self):
        """
        Updates must commute: β(obs→msg) ≈ β(msg→obs)
        FR-002: Total variation distance ≤ 1e-6
        """
        from robust_semantic_agent.core.belief import Belief
        from robust_semantic_agent.core.messages import Message, SourceTrust
        from robust_semantic_agent.core.semantics import BelnapValue

        np.random.seed(42)

        # Initial belief
        belief_a = Belief(n_particles=5000, state_dim=2)
        belief_a.particles = np.random.randn(5000, 2)
        belief_a.log_weights = np.full(5000, -np.log(5000))

        belief_b = Belief(n_particles=5000, state_dim=2)
        belief_b.particles = belief_a.particles.copy()
        belief_b.log_weights = belief_a.log_weights.copy()

        # Observation
        observation = np.array([0.5, 0.3])
        obs_noise = 0.1

        # Message
        def claim_fn(particles):
            return particles[:, 0] > 0.0  # Claim: x[0] > 0

        message = Message(
            claim="x[0] > 0",
            source="source_1",
            value=BelnapValue.TRUE,
            A_c=claim_fn,
        )
        source_trust = SourceTrust(r_s=0.8)

        # Order 1: observation → message
        belief_a.update_obs(observation, obs_noise)
        belief_a.apply_message(message, source_trust)

        # Order 2: message → observation
        belief_b.apply_message(message, source_trust)
        belief_b.update_obs(observation, obs_noise)

        # Compute TV distance
        tv_dist = total_variation_distance(belief_a.log_weights, belief_b.log_weights)

        assert tv_dist <= 1e-6, f"Commutativity violated: TV distance = {tv_dist:.2e} > 1e-6"


@pytest.mark.unit
class TestBeliefResampling:
    """T020: Test ESS computation and systematic resampling."""

    def test_eff_sample_size_uniform_weights(self):
        """ESS should equal N for uniform weights."""
        from robust_semantic_agent.core.belief import Belief

        belief = Belief(n_particles=1000, state_dim=2)
        # Uniform weights (default initialization)

        ess = belief.ess()

        assert abs(ess - 1000) < 1.0, f"ESS should be ~N for uniform weights, got {ess}"

    def test_eff_sample_size_degenerate_weights(self):
        """ESS should be ~1 when one particle has all weight."""
        from robust_semantic_agent.core.belief import Belief

        belief = Belief(n_particles=1000, state_dim=2)

        # Make weights degenerate (one particle dominant)
        belief.log_weights = np.full(1000, -1000.0)  # Very small
        belief.log_weights[0] = 0.0  # One large weight
        belief._normalize_log_weights()

        ess = belief.ess()

        assert ess < 10, f"ESS should be ~1 for degenerate weights, got {ess}"

    def test_resample_restores_ess(self):
        """Resampling should restore ESS to ~N."""
        from robust_semantic_agent.core.belief import Belief

        belief = Belief(n_particles=1000, state_dim=2)
        belief.particles = np.random.randn(1000, 2)

        # Create low ESS via informative observation
        observation = np.array([0.0, 0.0])
        belief.update_obs(observation, obs_noise=0.05)

        ess_before = belief.ess()
        assert ess_before < 500, "ESS should be low before resampling"

        # Resample
        belief.resample()

        ess_after = belief.ess()

        assert (
            ess_after > 0.9 * belief.n_particles
        ), f"ESS should be ~N after resampling, got {ess_after}"

    def test_resample_preserves_distribution(self):
        """Resampling should preserve weighted mean (approximately)."""
        from robust_semantic_agent.core.belief import Belief

        belief = Belief(n_particles=5000, state_dim=2)
        belief.particles = np.random.randn(5000, 2)

        # Create non-uniform weights
        observation = np.array([1.0, 0.5])
        belief.update_obs(observation, obs_noise=0.2)

        mean_before = belief.mean()

        belief.resample()

        mean_after = belief.mean()

        # Means should be close (within sampling error)
        diff = np.linalg.norm(mean_after - mean_before)
        assert diff < 0.1, f"Resampling should preserve mean, got diff={diff:.3f}"
