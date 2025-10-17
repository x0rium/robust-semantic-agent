"""
CVaR (Conditional Value at Risk) Implementation
Feature: 002-full-prototype
Tasks: T032, T033

Implements:
- cvar(): Sort-and-average algorithm for empirical samples
- cvar_weighted(): Weighted CVaR for particle beliefs
- RiskBellman: Risk-aware Bellman operator

References:
- docs/theory.md §4: Risk measures
- exploration/002_cvar.py: Verified MWE
- SC-005: CVaR analytical validation
"""

from collections.abc import Callable

import numpy as np


def cvar(values: np.ndarray, alpha: float = 0.10) -> float:
    """
    Compute CVaR@α from empirical samples (sort-and-average).

    CVaR@α = mean of worst α-fraction of outcomes

    Args:
        values: Array of outcome values (n,) - lower is worse for losses
        alpha: Tail risk level ∈ (0, 1] (default 0.10 = focus on worst 10%)

    Returns:
        CVaR value (expected value in worst α-tail)

    Example:
        >>> returns = np.array([-10, -5, -3, -1, 0, 2, 5, 10])
        >>> cvar(returns, alpha=0.25)  # Worst 25%
        -7.0  # Mean of [-10, -5]

    References:
        - SC-005: Gaussian/Uniform analytical validation
        - docs/verified-apis.md: Algorithm specification
    """
    n = len(values)
    cutoff_idx = max(1, int(np.ceil(alpha * n)))

    sorted_values = np.sort(values)  # Ascending order (worst first)
    return np.mean(sorted_values[:cutoff_idx])


def cvar_weighted(log_weights: np.ndarray, values: np.ndarray, alpha: float = 0.10) -> float:
    """
    Compute CVaR@α from log-weighted particles (for belief integration).

    Args:
        log_weights: Log-probabilities (n,) from particle filter
        values: Outcome values (n,) for each particle
        alpha: Tail risk level ∈ (0, 1]

    Returns:
        Weighted CVaR estimate

    Example:
        >>> log_weights = belief.log_weights
        >>> rewards = compute_rewards(belief.particles, action)
        >>> risk = cvar_weighted(log_weights, rewards, alpha=0.1)

    References:
        - FR-006: CVaR requirement
        - docs/theory.md §4.1: CVaR operator
    """
    # Normalize weights stably
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)

    # Sort by value (ascending)
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Find α-quantile cutoff
    cumsum = np.cumsum(sorted_weights)
    cutoff_idx = np.searchsorted(cumsum, alpha, side="right")

    if cutoff_idx == 0:
        cutoff_idx = 1

    # Weighted average of tail
    tail_weights = sorted_weights[:cutoff_idx]
    tail_values = sorted_values[:cutoff_idx]

    if np.sum(tail_weights) > 1e-12:
        return np.average(tail_values, weights=tail_weights)
    else:
        return sorted_values[0]  # Worst case fallback


class RiskBellman:
    """
    Risk-aware Bellman operator using CVaR.

    Implements: T_ρ V(b) = max_u CVaR_α(r(b,u) + γ V(b'))

    Attributes:
        alpha: CVaR risk level ∈ (0, 1]
        gamma: Discount factor ∈ [0, 1]

    References:
        - docs/theory.md §4.2: Risk-sensitive value iteration
        - FR-006: CVaR-based decision making
    """

    def __init__(self, alpha: float = 0.10, gamma: float = 0.98):
        """
        Initialize risk-aware Bellman operator.

        Args:
            alpha: CVaR level (lower = more risk-averse)
            gamma: Discount factor
        """
        self.alpha = alpha
        self.gamma = gamma

    def backup(
        self,
        belief,
        action: np.ndarray,
        reward_fn: Callable,
        transition_fn: Callable,
        value_fn: Callable,
        n_samples: int = 100,
    ) -> float:
        """
        Compute CVaR Bellman backup for belief-action pair.

        T_ρ V(b,u) = CVaR_α(r + γ V(b'))

        Args:
            belief: Belief object with particles and log_weights
            action: Control input (m,)
            reward_fn: (state, action) → reward
            transition_fn: (state, action) → next_state
            value_fn: (next_belief) → value
            n_samples: Monte Carlo samples for expectation

        Returns:
            CVaR value estimate

        References:
            - Task T033: RiskBellman implementation
        """
        # Sample particles from belief
        weights = np.exp(belief.log_weights - np.max(belief.log_weights))
        weights /= np.sum(weights)

        indices = np.random.choice(len(belief.particles), size=n_samples, replace=True, p=weights)

        sampled_particles = belief.particles[indices]

        # Compute returns for each sample
        returns = []
        for x in sampled_particles:
            # Immediate reward
            r = reward_fn(x, action)

            # Next state
            x_next = transition_fn(x, action)

            # Next value (simplified: assume deterministic)
            v_next = value_fn(x_next)

            # Total return
            returns.append(r + self.gamma * v_next)

        # CVaR of returns
        return cvar(np.array(returns), self.alpha)

    def __repr__(self) -> str:
        return f"RiskBellman(α={self.alpha}, γ={self.gamma})"
