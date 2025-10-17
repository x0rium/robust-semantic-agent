"""
Unit Tests: Credal Sets
Feature: 002-full-prototype
Task: T045, T046

Tests credal set creation, lower expectation, and posterior diversity.

References:
- docs/theory.md §3.3: Credal sets for v=⊤
- FR-005: Lower expectation monotonicity
- SC-004: Credal set coherence
"""

import numpy as np

from robust_semantic_agent.core.belief import Belief
from robust_semantic_agent.core.credal import CredalSet


class TestCredalSetCreation:
    """
    Test credal set creation and basic operations.

    References:
        - Task T046: Credal set creation tests
        - FR-005: CredalSet class specification
    """

    def test_credal_set_initialization(self):
        """CredalSet can be initialized with K posteriors"""
        # Create 3 simple posteriors (each is a Belief)
        posteriors = []
        for i in range(3):
            belief = Belief(n_particles=100, state_dim=2)
            belief.particles = np.random.randn(100, 2) + i  # Shift means
            posteriors.append(belief)

        credal = CredalSet(posteriors=posteriors)
        assert credal.K == 3
        assert len(credal.posteriors) == 3

    def test_credal_set_add_posterior(self):
        """Can add posteriors to existing credal set"""
        credal = CredalSet(posteriors=[])
        assert credal.K == 0

        belief1 = Belief(n_particles=50, state_dim=2)
        credal.add_posterior(belief1)
        assert credal.K == 1

        belief2 = Belief(n_particles=50, state_dim=2)
        credal.add_posterior(belief2)
        assert credal.K == 2

    def test_credal_set_empty_initialization(self):
        """CredalSet can be initialized empty"""
        credal = CredalSet()
        assert credal.K == 0
        assert len(credal.posteriors) == 0


class TestPosteriorDiversity:
    """
    Test that credal set posteriors are diverse (not identical).

    For v=⊤, we should create K extreme posteriors from logit interval Λ_s.

    References:
        - Task T046: Posterior diversity tests
        - docs/theory.md: Logit interval [-λ_s, +λ_s] for v=⊤
    """

    def test_posteriors_have_different_means(self):
        """K posteriors should have different means"""
        # Create posteriors with different mean shifts
        posteriors = []
        for i in range(5):
            belief = Belief(n_particles=100, state_dim=2)
            # Shift particles to create different means
            belief.particles = np.random.randn(100, 2) + np.array([i * 0.5, 0])
            posteriors.append(belief)

        credal = CredalSet(posteriors=posteriors)

        # Compute means of each posterior
        means = [p.mean() for p in credal.posteriors]

        # Verify they are not all identical
        # (at least two should differ by > 0.1)
        max_diff = 0
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                diff = np.linalg.norm(means[i] - means[j])
                max_diff = max(max_diff, diff)

        assert max_diff > 0.1, "Posteriors should be diverse (different means)"

    def test_posteriors_are_independent_objects(self):
        """Each posterior should be an independent Belief object"""
        belief1 = Belief(n_particles=50, state_dim=2)
        belief2 = Belief(n_particles=50, state_dim=2)

        credal = CredalSet(posteriors=[belief1, belief2])

        # Modifying one should not affect the other
        original_mean_0 = credal.posteriors[0].mean().copy()
        credal.posteriors[1].particles += 10.0  # Modify second

        # First should be unchanged
        assert np.allclose(credal.posteriors[0].mean(), original_mean_0)


class TestLowerExpectation:
    """
    Test lower expectation computation.

    Lower expectation: 𝔼_[f] = min_{P ∈ Γ} 𝔼_P[f(x)]

    For credal set with K posteriors, this is the minimum expected value.

    References:
        - Task T045: Lower expectation monotonicity (SC-004)
        - FR-005: Lower expectation ≤ any extreme posterior expectation
    """

    def test_lower_expectation_simple_function(self):
        """Lower expectation of simple function f(x) = x[0]"""
        # Create posteriors with different x[0] means
        posteriors = []
        means_x0 = [0.0, 1.0, 2.0]  # Three different means for x[0]

        for mean_val in means_x0:
            belief = Belief(n_particles=100, state_dim=2)
            belief.particles = np.random.randn(100, 2) * 0.1  # Small variance
            belief.particles[:, 0] += mean_val  # Shift x[0]
            posteriors.append(belief)

        credal = CredalSet(posteriors=posteriors)

        # Function: f(x) = x[0]
        def f(x):
            return x[0]

        lower_exp = credal.lower_expectation(f)

        # Should be close to min(means_x0) = 0.0
        assert lower_exp < 0.3, f"Lower expectation {lower_exp} should be close to 0.0"
        assert lower_exp > -0.3, f"Lower expectation {lower_exp} should be positive"

    def test_lower_expectation_monotonicity(self):
        """
        SC-004: Lower expectation ≤ expectation of any extreme posterior.

        𝔼_[f] ≤ 𝔼_P[f] for all P ∈ Γ
        """
        # Create 5 posteriors
        posteriors = []
        for i in range(5):
            belief = Belief(n_particles=100, state_dim=2)
            belief.particles = np.random.randn(100, 2) + i
            posteriors.append(belief)

        credal = CredalSet(posteriors=posteriors)

        # Function: f(x) = ||x||^2
        def f(x):
            return np.dot(x, x)

        lower_exp = credal.lower_expectation(f)

        # Compute expectation for each posterior
        for i, belief in enumerate(credal.posteriors):
            # Expected value: E[f(x)] ≈ mean over particles
            particles = belief.particles
            weights = np.exp(belief.log_weights - np.max(belief.log_weights))
            weights /= np.sum(weights)
            expected = np.sum([w * f(p) for w, p in zip(weights, particles, strict=False)])

            # Lower expectation should be ≤ this
            assert (
                lower_exp <= expected + 1e-6
            ), f"Lower expectation {lower_exp} exceeds posterior {i} expectation {expected}"

    def test_lower_expectation_constant_function(self):
        """Lower expectation of constant function f(x) = c should be c"""
        posteriors = []
        for i in range(3):
            belief = Belief(n_particles=50, state_dim=2)
            belief.particles = np.random.randn(50, 2) * (i + 1)  # Different variances
            posteriors.append(belief)

        credal = CredalSet(posteriors=posteriors)

        # Constant function
        def f(x):
            return 5.0

        lower_exp = credal.lower_expectation(f)
        assert (
            np.abs(lower_exp - 5.0) < 1e-6
        ), "Lower expectation of constant should equal the constant"

    def test_lower_expectation_linear_function(self):
        """Lower expectation of linear function"""
        # Create posteriors with different means
        posteriors = []
        np.random.seed(42)
        for i in range(4):
            belief = Belief(n_particles=100, state_dim=2)
            belief.particles = np.random.randn(100, 2)
            belief.particles += np.array([i, -i])  # Different mean shifts
            posteriors.append(belief)

        credal = CredalSet(posteriors=posteriors)

        # Linear function: f(x) = 2*x[0] - 3*x[1]
        def f(x):
            return 2 * x[0] - 3 * x[1]

        lower_exp = credal.lower_expectation(f)

        # Verify it's a real number
        assert np.isfinite(lower_exp)

        # Verify monotonicity
        for belief in credal.posteriors:
            particles = belief.particles
            weights = np.exp(belief.log_weights - np.max(belief.log_weights))
            weights /= np.sum(weights)
            expected = np.sum([w * f(p) for w, p in zip(weights, particles, strict=False)])
            assert lower_exp <= expected + 1e-5


class TestCredalSetEdgeCases:
    """
    Test edge cases and error handling for credal sets.

    References:
        - Task T046: Robustness tests
    """

    def test_single_posterior_credal_set(self):
        """Credal set with K=1 should work (degenerate case)"""
        belief = Belief(n_particles=50, state_dim=2)
        belief.particles = np.random.randn(50, 2)

        credal = CredalSet(posteriors=[belief])
        assert credal.K == 1

        def f(x):
            return x[0]

        # Should equal the single posterior's expectation
        lower_exp = credal.lower_expectation(f)
        particles = belief.particles
        weights = np.exp(belief.log_weights - np.max(belief.log_weights))
        weights /= np.sum(weights)
        expected = np.sum([w * f(p) for w, p in zip(weights, particles, strict=False)])

        assert np.abs(lower_exp - expected) < 1e-5

    def test_credal_set_with_identical_posteriors(self):
        """If all posteriors identical, lower expectation = their expectation"""
        # Create 3 identical posteriors
        np.random.seed(42)
        particles = np.random.randn(100, 2)

        posteriors = []
        for _ in range(3):
            belief = Belief(n_particles=100, state_dim=2)
            belief.particles = particles.copy()  # Same particles
            posteriors.append(belief)

        credal = CredalSet(posteriors=posteriors)

        def f(x):
            return x[0] + x[1]

        lower_exp = credal.lower_expectation(f)

        # Compute expected value from first posterior
        particles = credal.posteriors[0].particles
        weights = np.exp(
            credal.posteriors[0].log_weights - np.max(credal.posteriors[0].log_weights)
        )
        weights /= np.sum(weights)
        expected = np.sum([w * f(p) for w, p in zip(weights, particles, strict=False)])

        # Should be approximately equal
        assert np.abs(lower_exp - expected) < 1e-4
