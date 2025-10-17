# Particle Filter Best Practices for RSA POMDP Belief Tracking

## Research Summary

Based on research of current best practices (2024), here are recommendations for implementing particle filters in the Robust Semantic Agent project.

## Recommended Approach

### 1. Particle Representation

**Data Structure**: Use NumPy (n, d) matrix where:
- n = number of particles (recommend 1000-10000 for 2D continuous spaces)
- d = state dimensionality
- Store weights separately as (n,) vector

**Benefits**:
- Enables vectorized operations across all particles
- Memory-efficient
- Compatible with scipy.stats distributions

### 2. Weight Updates in Log-Space (Critical for Numerical Stability)

**Key insight**: Particle weights can approach zero, causing underflow. The Log-PF approach (Gentner et al., 2018) solves this by:

1. Store log-weights: `log_w` instead of `w`
2. Update in log-space: `log_w += log(likelihood)`
3. Normalize using log-sum-exp trick:
   ```python
   log_w_max = np.max(log_w)
   log_w_normalized = log_w - (log_w_max + np.log(np.sum(np.exp(log_w - log_w_max))))
   ```

**Jacobian logarithm** enables computing sums in log-space without converting back:
`log(exp(a) + exp(b)) = max(a,b) + log(1 + exp(-|a-b|))`

### 3. Resampling Strategy

**When to resample**: Check Effective Sample Size (ESS)
```python
ESS = 1.0 / np.sum(weights**2)
if ESS < n_particles / 2:  # Common threshold: 50% of particles
    resample()
```

**Algorithm**: Systematic resampling (preferred over multinomial)
- Lower variance than multinomial
- More deterministic, better coverage
- O(n) complexity
- Preserves particles if weights are uniform

### 4. Message Integration for RSA

For soft-likelihood message multipliers M(x) from sources:

1. **Treat messages as additional likelihoods**:
   ```python
   # Observation update
   log_w += np.log(observation_likelihood(particles, observation))

   # Message update (commutative with observations)
   log_w += np.log(message_multiplier(particles, claim, source, status))
   ```

2. **Ensure commutativity** (RSA requirement):
   - Messages and observations are conditionally independent given state
   - Update order should not affect result (test with TV distance ≤ 1e-6)
   - Apply normalization after all updates complete

### 5. Key Numerical Stability Considerations

1. **Never multiply small probabilities directly** - always use log-space
2. **Log-sum-exp trick** for normalization prevents overflow/underflow
3. **Add small epsilon** (1e-300) before taking log if zeros possible
4. **Monitor ESS** - low ESS indicates particle degeneracy
5. **Particle diversity** - add small jitter during resampling to prevent collapse

## Example Code Snippet

```python
import numpy as np
from scipy.stats import norm, multivariate_normal

class ParticleBeliefTracker:
    """Particle filter for POMDP belief tracking with message integration."""

    def __init__(self, n_particles, state_dim, resample_threshold=0.5):
        self.n = n_particles
        self.d = state_dim
        self.threshold = resample_threshold

        # Particles: (n, d) matrix
        self.particles = np.zeros((n_particles, state_dim))

        # Log-weights: (n,) vector
        self.log_weights = np.full(n_particles, -np.log(n_particles))

    def predict(self, dynamics_fn, process_noise_std):
        """Propagate particles forward using dynamics model."""
        # Apply deterministic dynamics
        self.particles = dynamics_fn(self.particles)

        # Add process noise (prevents degeneracy)
        noise = np.random.randn(self.n, self.d) * process_noise_std
        self.particles += noise

    def update_observation(self, observation, likelihood_fn):
        """Update weights based on observation likelihood (in log-space)."""
        # Compute log-likelihood for each particle
        log_likelihoods = likelihood_fn(self.particles, observation)

        # Update log-weights
        self.log_weights += log_likelihoods

        # Normalize using log-sum-exp trick
        self._normalize_log_weights()

    def update_message(self, claim, source, status, message_fn):
        """
        Incorporate soft-likelihood message from source.
        Commutative with observation updates (order-independent).
        """
        # Compute log-multiplier M_{c,s,v}(x) for each particle
        log_multipliers = message_fn(self.particles, claim, source, status)

        # Update log-weights (same as observation)
        self.log_weights += log_multipliers

        # Normalize
        self._normalize_log_weights()

    def _normalize_log_weights(self):
        """Normalize log-weights using log-sum-exp trick for numerical stability."""
        # Shift by max to prevent overflow
        log_w_max = np.max(self.log_weights)
        log_sum = log_w_max + np.log(np.sum(np.exp(self.log_weights - log_w_max)))

        # Normalize
        self.log_weights -= log_sum

    def check_and_resample(self):
        """Check ESS and resample if below threshold."""
        # Convert to linear weights for ESS calculation
        weights = np.exp(self.log_weights)
        ess = 1.0 / np.sum(weights**2)

        if ess < self.threshold * self.n:
            self._systematic_resample(weights)
            return True
        return False

    def _systematic_resample(self, weights):
        """Systematic resampling (low variance)."""
        n = self.n

        # Cumulative sum of weights
        cumsum = np.cumsum(weights)

        # Start at random position in [0, 1/n]
        positions = (np.arange(n) + np.random.uniform()) / n

        # Find indices
        indices = np.searchsorted(cumsum, positions)

        # Resample particles
        self.particles = self.particles[indices]

        # Reset weights to uniform (in log-space)
        self.log_weights = np.full(n, -np.log(n))

        # Add small jitter to maintain diversity
        jitter = np.random.randn(n, self.d) * 0.01
        self.particles += jitter

    def get_mean_estimate(self):
        """Return weighted mean of particles."""
        weights = np.exp(self.log_weights)
        return np.average(self.particles, weights=weights, axis=0)

    def total_variation_distance(self, other_belief):
        """
        Compute TV distance to another belief (for testing commutativity).
        For particles, approximate using weighted samples.
        """
        # Sort particles and weights for comparison
        idx1 = np.argsort(self.particles[:, 0])
        idx2 = np.argsort(other_belief.particles[:, 0])

        w1 = np.exp(self.log_weights[idx1])
        w2 = np.exp(other_belief.log_weights[idx2])

        # TV distance approximation
        tv = 0.5 * np.sum(np.abs(w1 - w2))
        return tv


# Example usage for RSA
def example_likelihood(particles, observation):
    """
    Observation likelihood: G(o|x)
    Example: Gaussian sensor model
    """
    # Distance from observation
    distances = np.linalg.norm(particles - observation, axis=1)

    # Log-likelihood (Gaussian)
    sensor_std = 0.5
    log_lik = -0.5 * (distances / sensor_std)**2
    return log_lik


def example_message_multiplier(particles, claim, source, status):
    """
    Soft-likelihood message multiplier: M_{c,s,v}(x)
    Based on source trust r_s and claim region A_c
    """
    # For status v=true: multiply by exp(λ_s) if x in A_c, else exp(-λ_s)
    # For status v=false: reverse
    # For status v=contradiction (⊤): return ensemble (credal set)

    # Example: claim is "x_0 > threshold"
    claim_satisfied = particles[:, 0] > claim['threshold']

    # Source trust -> logit
    r_s = source['reliability']
    lambda_s = np.log(r_s / (1 - r_s))

    # Compute log-multiplier
    if status == 'true':
        log_mult = np.where(claim_satisfied, lambda_s, -lambda_s)
    elif status == 'false':
        log_mult = np.where(claim_satisfied, -lambda_s, lambda_s)
    else:  # contradiction - neutral (or handle with credal set)
        log_mult = np.zeros(len(particles))

    return log_mult


# Test commutativity (RSA requirement)
def test_commutativity():
    """Verify observation and message updates are commutative."""
    np.random.seed(42)

    # Create two identical beliefs
    belief1 = ParticleBeliefTracker(n_particles=1000, state_dim=2)
    belief2 = ParticleBeliefTracker(n_particles=1000, state_dim=2)

    # Same initial particles
    initial_particles = np.random.randn(1000, 2)
    belief1.particles = initial_particles.copy()
    belief2.particles = initial_particles.copy()

    observation = np.array([1.0, 0.5])
    claim = {'threshold': 0.5}
    source = {'reliability': 0.8}

    # Order 1: observation then message
    belief1.update_observation(observation, example_likelihood)
    belief1.update_message(claim, source, 'true', example_message_multiplier)

    # Order 2: message then observation
    belief2.update_message(claim, source, 'true', example_message_multiplier)
    belief2.update_observation(observation, example_likelihood)

    # Check TV distance
    tv = belief1.total_variation_distance(belief2)
    print(f"Total Variation Distance: {tv:.10f}")
    assert tv < 1e-6, f"Commutativity violated: TV = {tv}"
    print("✓ Commutativity test passed")


if __name__ == "__main__":
    test_commutativity()
```

## Authoritative Sources

1. **Log-PF Paper** (Numerical Stability):
   - Gentner et al. (2018), "Log-PF: Particle Filtering in Logarithm Domain"
   - Journal of Electrical and Computer Engineering
   - https://onlinelibrary.wiley.com/doi/10.1155/2018/5763461
   - Key contribution: Jacobian logarithm for weight normalization

2. **Python Implementation Guide** (Practical Tutorial):
   - Labbe, "Kalman and Bayesian Filters in Python"
   - Chapter 12: Particle Filters
   - https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
   - Comprehensive notebook with working code examples

3. **POMDP-Specific Implementation** (Belief Tracking):
   - pomdp-py library by H2R Lab
   - https://h2r.github.io/pomdp-py/
   - Production-quality particle filter for belief-MDP planning
   - Used in robotics research, includes WeightedParticles class

## Additional Recommendations for RSA

1. **Particle count**: Start with 5000 particles for 2D continuous space (as per CLAUDE.md)
2. **Resample threshold**: 50% ESS is standard, tune based on performance
3. **Process noise**: Add small jitter (σ ≈ 0.01) to prevent particle collapse
4. **Credal sets**: For contradictions (v=⊤), maintain ensemble of K=10 extreme posteriors
5. **Performance**: Target 30+ Hz with 10k particles (vectorized NumPy operations essential)

## Testing Checklist

- [ ] Commutativity: TV distance ≤ 1e-6 for observation/message order permutations
- [ ] ESS monitoring: Resample triggers when ESS < threshold
- [ ] No underflow: Log-weights never produce NaN or -inf
- [ ] Particle diversity: Std dev of particles doesn't collapse to zero
- [ ] Credal set monotonicity: Lower expectation ≤ any extreme posterior
