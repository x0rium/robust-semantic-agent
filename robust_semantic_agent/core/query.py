"""
Query Action and Expected Value of Information (EVI)
Feature: 002-full-prototype
Tasks: T060, T061

Implements active information acquisition via query action.

Key Concepts:
- EVI: Expected Value of Information
- Query action: Request additional observation at cost c
- Trigger when EVI ≥ Δ* (minimum expected regret threshold)

References:
- docs/theory.md §5: Active information acquisition
- SC-006: EVI ≥ Δ* before query
- SC-007: Entropy reduction ≥ 20% after query
"""

from collections.abc import Callable

import numpy as np

from .belief import Belief


def evi(
    belief: Belief,
    value_fn: Callable[[Belief], float],
    obs_noise: float,
    n_samples: int = 100,
) -> float:
    """
    Compute Expected Value of Information (EVI).

    EVI = 𝔼_o[V(β_post(o))] - V(β)

    Where:
    - β: Current belief
    - o: Potential observation sampled from belief
    - β_post(o): Posterior belief after incorporating observation o
    - V(β): Value function over beliefs

    Algorithm:
    1. Sample potential observations from current belief
    2. For each observation, compute posterior belief
    3. Evaluate value function on each posterior
    4. Take expectation over observations
    5. Subtract current value

    Args:
        belief: Current belief state
        value_fn: Value function V(β) → ℝ
        obs_noise: Observation noise standard deviation
        n_samples: Number of observation samples for expectation

    Returns:
        EVI value (positive → information is valuable)

    References:
        - docs/theory.md §5.1: EVI computation
        - SC-006: EVI ≥ Δ* triggers query

    Example:
        >>> goal = np.array([0.8, 0.8])
        >>> def value_fn(b):
        ...     mean = b.mean()
        ...     return -np.linalg.norm(mean - goal)
        >>> evi_value = evi(belief, value_fn, obs_noise=0.1, n_samples=50)
    """
    # Current value
    V_current = value_fn(belief)

    # Sample potential observations from belief
    # Draw particles according to weights
    weights = np.exp(belief.log_weights - np.max(belief.log_weights))
    weights /= np.sum(weights)

    # Sample particle indices
    indices = np.random.choice(belief.n_particles, size=n_samples, replace=True, p=weights)
    sampled_states = belief.particles[indices]

    # Generate noisy observations from sampled states
    observations = sampled_states + np.random.randn(n_samples, belief.state_dim) * obs_noise

    # Compute posterior values for each observation
    posterior_values = []

    for obs in observations:
        # Create posterior belief by updating with observation
        posterior = Belief(
            n_particles=belief.n_particles,
            state_dim=belief.state_dim,
            resample_threshold=belief.resample_threshold,
        )
        posterior.particles = belief.particles.copy()
        posterior.log_weights = belief.log_weights.copy()

        # Update with observation
        posterior.update_obs(obs, obs_noise)

        # Evaluate value function
        V_post = value_fn(posterior)
        posterior_values.append(V_post)

    # Expected value of posterior
    V_expected_post = np.mean(posterior_values)

    # EVI = Expected improvement
    evi_value = V_expected_post - V_current

    return evi_value


def should_query(evi_value: float, delta_star: float) -> bool:
    """
    Decide whether to trigger query action based on EVI threshold.

    Query triggers when EVI ≥ Δ* (expected value of information exceeds
    minimum regret reduction threshold).

    Args:
        evi_value: Computed EVI from evi() function
        delta_star: Minimum EVI threshold (regret threshold)

    Returns:
        True if query should be triggered, False otherwise

    References:
        - docs/theory.md §5.2: Query decision rule
        - SC-006: EVI ≥ Δ* requirement

    Example:
        >>> evi_val = 0.2
        >>> delta_star = 0.15
        >>> should_query(evi_val, delta_star)
        True
    """
    return evi_value >= delta_star


def compute_query_observation(env, obs_noise: float) -> np.ndarray:
    """
    Request additional observation from environment (query action).

    This function is called when query action triggers. It returns
    a more precise observation of the current state at cost c.

    Args:
        env: Environment instance with .state attribute
        obs_noise: Observation noise for query (can be lower than normal)

    Returns:
        Observation array (state_dim,)

    Note:
        Query observations may have lower noise than standard observations,
        providing more information at a cost.

    References:
        - docs/theory.md §5.3: Query observation model
    """
    # Return noisy observation of true state
    noise = np.random.randn(env.state.shape[0]) * obs_noise
    return env.state + noise


def __repr__() -> str:
    return "Query module: evi(), should_query(), compute_query_observation()"
