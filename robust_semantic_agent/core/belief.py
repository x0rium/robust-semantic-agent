"""
Particle Filter Belief Tracking
Feature: 002-full-prototype
Tasks: T028, T029

Implements:
- Belief class with particle representation
- Observation updates with log-space likelihood weighting
- Message updates with soft-likelihood multipliers
- Systematic resampling with ESS monitoring

References:
- docs/theory.md §1: Belief-MDP formulation
- exploration/001_particle_filter.py: Verified MWE
- FR-002: Commutative update requirement (TV ≤ 1e-6)
"""

import numpy as np
from scipy.stats import norm


class Belief:
    """
    Particle filter belief tracking for POMDP.

    Represents belief β(x) as weighted particles in log-space.

    Attributes:
        particles: (n_particles, state_dim) array of particle positions
        log_weights: (n_particles,) array of log-probabilities
        n_particles: Number of particles
        state_dim: State space dimensionality
        resample_threshold: ESS threshold for triggering resampling (as fraction of N)

    Implements Theorem 1 (theory.md): Belief-MDP equivalence
    """

    def __init__(
        self,
        n_particles: int = 5000,
        state_dim: int = 2,
        resample_threshold: float = 0.5,
    ):
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.resample_threshold = resample_threshold

        # Initialize uniform distribution
        self.particles = np.zeros((n_particles, state_dim))
        self.log_weights = np.full(n_particles, -np.log(n_particles))

        # Credal set for contradictory information (v=⊤)
        self.credal_set = None  # Optional CredalSet object

    def update_obs(self, observation: np.ndarray, obs_noise: float) -> None:
        """
        Update belief with observation using Gaussian likelihood.

        β̃(x) ∝ G(o|x) · β(x)

        Args:
            observation: Observed value (state_dim,)
            obs_noise: Observation noise standard deviation

        Updates:
            log_weights via likelihood weighting

        References:
            - FR-001: Observation kernel G
            - exploration/001_particle_filter.py: Validated implementation
        """
        # Compute log-likelihood G(o|x) = N(o; x, σ²)
        # For multivariate: sum log-likelihoods across dimensions
        log_likelihood = np.sum(
            norm.logpdf(observation, loc=self.particles, scale=obs_noise), axis=1
        )

        # Update weights in log-space
        self.log_weights += log_likelihood

        # Normalize
        self._normalize_log_weights()

    def apply_message(self, message, source_trust) -> None:
        """
        Apply message update with soft-likelihood multiplier.

        β'(x) ∝ M_{c,s,v}(x) · β(x)

        Args:
            message: Message object with claim, source, value, A_c
            source_trust: SourceTrust object with reliability r_s

        Updates:
            log_weights via message multiplier

        References:
            - FR-003: Message integration
            - FR-002: Commutativity with observations (TV ≤ 1e-6)
        """
        from .semantics import BelnapValue

        # Get logit from source reliability
        lambda_s = source_trust.logit()

        # Evaluate claim on particles
        claim_satisfied = message.A_c(self.particles)  # Boolean array (n_particles,)

        # Compute log-multiplier based on Belnap status
        if message.value == BelnapValue.TRUE:
            # Support claim: +λ_s where true, -λ_s where false
            log_mult = np.where(claim_satisfied, lambda_s, -lambda_s)
        elif message.value == BelnapValue.FALSE:
            # Countersupport claim: -λ_s where true, +λ_s where false
            log_mult = np.where(claim_satisfied, -lambda_s, lambda_s)
        elif message.value == BelnapValue.NEITHER:
            # No information: neutral multiplier
            log_mult = np.zeros(self.n_particles)
        else:  # BelnapValue.BOTH - contradiction (v=⊤)
            # Task T051: Expand belief to credal set
            # Logit interval Λ_s = [-λ_s, +λ_s] → K extreme posteriors
            from .credal import create_credal_from_logit_interval

            # Get K from config (default 5)
            K = 5  # TODO: Read from config

            # Create credal set with K extreme posteriors
            self.credal_set = create_credal_from_logit_interval(
                base_belief=self,
                A_c=message.A_c,
                lambda_s=lambda_s,
                K=K,
            )

            # For base belief, apply neutral multiplier (central estimate)
            log_mult = np.zeros(self.n_particles)

        # Update weights in log-space
        self.log_weights += log_mult

        # Normalize
        self._normalize_log_weights()

    def resample(self) -> None:
        """
        Systematic resampling (low variance).

        Restores ESS to ~N by resampling particles according to weights.

        References:
            - exploration/001_particle_filter.py: Validated algorithm
        """
        # Normalize weights to probabilities
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights /= np.sum(weights)

        # Systematic resampling
        cumsum = np.cumsum(weights)
        positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        indices = np.searchsorted(cumsum, positions)

        # Resample particles
        self.particles = self.particles[indices].copy()

        # Reset weights to uniform
        self.log_weights = np.full(self.n_particles, -np.log(self.n_particles))

        # Add small jitter to maintain diversity
        self.particles += np.random.randn(self.n_particles, self.state_dim) * 0.01

    def ess(self) -> float:
        """
        Compute Effective Sample Size.

        ESS = 1 / Σ(w_i²)

        Returns:
            Effective sample size ∈ [1, N]
        """
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights /= np.sum(weights)
        return 1.0 / np.sum(weights**2)

    def mean(self) -> np.ndarray:
        """
        Compute weighted mean of particles.

        Returns:
            Mean state estimate (state_dim,)
        """
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights /= np.sum(weights)
        return np.average(self.particles, weights=weights, axis=0)

    def covariance(self) -> np.ndarray:
        """
        Compute weighted covariance of particles.

        Returns:
            Covariance matrix (state_dim, state_dim)
        """
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights /= np.sum(weights)
        mean = self.mean()
        diff = self.particles - mean
        return np.average(diff[:, :, None] * diff[:, None, :], weights=weights, axis=0)

    def entropy(self) -> float:
        """
        Compute Shannon entropy of belief.

        H(β) = -Σ w_i log(w_i)

        Returns:
            Entropy in nats

        References:
            - Task T062: Query action implementation
        """
        weights = np.exp(self.log_weights - np.max(self.log_weights))
        weights /= np.sum(weights)

        # Avoid log(0)
        weights = weights[weights > 1e-12]

        return -np.sum(weights * np.log(weights))

    def _normalize_log_weights(self) -> None:
        """
        Normalize log-weights using log-sum-exp trick.

        Prevents numerical overflow/underflow.

        References:
            - docs/verified-apis.md: Log-PF algorithm
        """
        log_w_max = np.max(self.log_weights)
        log_sum = log_w_max + np.log(np.sum(np.exp(self.log_weights - log_w_max)))
        self.log_weights -= log_sum

    def __repr__(self) -> str:
        return (
            f"Belief(n_particles={self.n_particles}, "
            f"state_dim={self.state_dim}, "
            f"ESS={self.ess():.1f}, "
            f"mean={self.mean()})"
        )
