"""
Credal Sets for Contradictory Information
Feature: 002-full-prototype
Task: T050

Implements credal set representation for v=⊤ (contradiction) messages.

When receiving contradictory information (Belnap value ⊤), instead of a single
posterior belief, we maintain a credal set Γ = {P_1, ..., P_K} of extreme
posteriors generated from logit interval Λ_s = [-λ_s, +λ_s].

Decision-making uses lower expectation: 𝔼_[f] = min_{P ∈ Γ} 𝔼_P[f(x)]

References:
- docs/theory.md §3.3: Credal sets
- FR-005: Lower expectation monotonicity
- SC-004: Credal set coherence
"""

from collections.abc import Callable

import numpy as np


class CredalSet:
    """
    Credal set: ensemble of K extreme posterior beliefs.

    Represents imprecise probability when receiving contradictory
    information (v=⊤). Instead of single belief β(x), maintain
    credal set Γ = {β_1(x), ..., β_K(x)}.

    Attributes:
        posteriors: List of Belief objects (K extreme posteriors)
        K: Number of posteriors in credal set

    Methods:
        add_posterior(belief): Add posterior to credal set
        lower_expectation(f): Compute 𝔼_[f] = min_{P ∈ Γ} 𝔼_P[f]
        mean(): Conservative mean estimate (using lower expectation)

    References:
        - Task T050: CredalSet implementation
        - FR-005: Lower expectation for robust decisions
        - SC-004: Coherence validation
    """

    def __init__(self, posteriors: list = None):
        """
        Initialize credal set.

        Args:
            posteriors: List of Belief objects (extreme posteriors)
                       If None, initialize empty credal set
        """
        if posteriors is None:
            posteriors = []

        self.posteriors = posteriors
        self.K = len(posteriors)

    def add_posterior(self, belief) -> None:
        """
        Add posterior to credal set.

        Args:
            belief: Belief object to add
        """
        self.posteriors.append(belief)
        self.K = len(self.posteriors)

    def lower_expectation(self, f: Callable[[np.ndarray], float]) -> float:
        """
        Compute lower expectation (robust/conservative estimate).

        𝔼_[f] = min_{P ∈ Γ} 𝔼_P[f(x)]

        This is the worst-case expected value across all extreme posteriors.
        Ensures robustness when facing contradictory information.

        Args:
            f: Function x → ℝ to compute expectation of

        Returns:
            Lower expectation value

        References:
            - FR-005: Lower expectation monotonicity
            - Ensures 𝔼_[f] ≤ 𝔼_P[f] for all P ∈ Γ
        """
        if self.K == 0:
            raise ValueError("Cannot compute lower expectation on empty credal set")

        expectations = []

        for belief in self.posteriors:
            # Compute E_P[f(x)] for this posterior P
            particles = belief.particles
            log_weights = belief.log_weights

            # Normalize weights
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= np.sum(weights)

            # Expected value: sum_i w_i * f(x_i)
            expected = 0.0
            for w, x in zip(weights, particles, strict=False):
                expected += w * f(x)

            expectations.append(expected)

        # Return minimum (lower bound)
        return min(expectations)

    def mean(self) -> np.ndarray:
        """
        Conservative mean estimate using lower expectation.

        For each dimension d, compute 𝔼_[x_d] = min_P 𝔼_P[x_d]

        Returns:
            Mean vector (state_dim,)

        Note:
            This is conservative - may be pessimistic. Alternative:
            - Mean of means: average all posterior means
            - Centroid: geometric center of credal set
        """
        if self.K == 0:
            raise ValueError("Cannot compute mean of empty credal set")

        state_dim = self.posteriors[0].state_dim

        # For each dimension, compute lower expectation
        mean = np.zeros(state_dim)
        for d in range(state_dim):
            # Function: f(x) = x[d]
            def f_d(x):
                return x[d]

            mean[d] = self.lower_expectation(f_d)

        return mean

    def variance(self) -> np.ndarray:
        """
        Upper variance estimate (maximum across posteriors).

        For credal sets, variance is an interval [var_, var̄].
        We return the upper bound (conservative for uncertainty).

        Returns:
            Variance vector (state_dim,) - upper bound
        """
        if self.K == 0:
            raise ValueError("Cannot compute variance of empty credal set")

        state_dim = self.posteriors[0].state_dim

        # Maximum variance across all posteriors (conservative)
        max_var = np.zeros(state_dim)

        for belief in self.posteriors:
            particles = belief.particles
            weights = np.exp(belief.log_weights - np.max(belief.log_weights))
            weights /= np.sum(weights)

            # Variance: E[x^2] - E[x]^2
            mean = np.average(particles, axis=0, weights=weights)
            var = np.average((particles - mean) ** 2, axis=0, weights=weights)

            max_var = np.maximum(max_var, var)

        return max_var

    def __repr__(self) -> str:
        return f"CredalSet(K={self.K} posteriors)"

    def __len__(self) -> int:
        return self.K


# Helper functions for credal set creation


def create_credal_from_logit_interval(
    base_belief,
    A_c: Callable[[np.ndarray], np.ndarray],
    lambda_s: float,
    K: int = 5,
) -> CredalSet:
    """
    Create credal set from logit interval Λ_s = [-λ_s, +λ_s].

    For contradiction (v=⊤), source provides interval of logit multipliers.
    Generate K extreme posteriors by sampling from this interval.

    Args:
        base_belief: Belief object to extend with credal set
        A_c: Claim indicator function x → {0, 1}
        lambda_s: Logit bound (from source trust)
        K: Number of extreme posteriors to generate

    Returns:
        CredalSet with K posteriors

    Algorithm:
        For k = 1..K:
        1. Select logit value from interval: λ_k ∈ [-λ_s, +λ_s]
        2. Create posterior: β_k(x) ∝ exp(λ_k · A_c(x)) · β(x)
        3. Add to credal set Γ

    References:
        - docs/theory.md §3.3: Credal set expansion
        - Task T051: Belief extension for v=⊤
    """
    from robust_semantic_agent.core.belief import Belief

    posteriors = []

    # Generate K extreme posteriors spanning the logit interval
    for k in range(K):
        # Select logit value: evenly spaced from -λ_s to +λ_s
        if K == 1:
            logit_value = 0.0
        else:
            # Linear interpolation: -λ_s + (2λ_s * k / (K-1))
            logit_value = -lambda_s + (2 * lambda_s * k / (K - 1))

        # Create new belief with same particles
        belief_k = Belief(
            n_particles=base_belief.n_particles,
            state_dim=base_belief.state_dim,
            resample_threshold=base_belief.resample_threshold,
        )

        # Copy particles and weights
        belief_k.particles = base_belief.particles.copy()
        belief_k.log_weights = base_belief.log_weights.copy()

        # Apply logit multiplier: log w_k = log w + λ_k · A_c(x)
        claim_satisfied = A_c(belief_k.particles)  # Shape: (n_particles,)

        # Logit multiplier: +λ_k for A_c(x)=1, -λ_k for A_c(x)=0
        # This creates "extreme" posteriors favoring/disfavoring the claim
        log_mult = np.where(claim_satisfied, logit_value, -logit_value)

        belief_k.log_weights += log_mult

        # Normalize
        belief_k._normalize_log_weights()

        posteriors.append(belief_k)

    return CredalSet(posteriors=posteriors)
