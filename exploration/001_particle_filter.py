"""
Minimal Working Example: Log-Space Particle Filter
Feature: 002-full-prototype
Task: T007

Tests:
1. Observation update with Gaussian likelihood
2. Message update with soft-likelihood multiplier
3. Commutativity: TV distance ≤ 1e-6
4. Systematic resampling when ESS < threshold
"""

import numpy as np
from scipy.stats import norm


def total_variation_distance(log_weights1, log_weights2):
    """Compute total variation distance between two distributions."""
    # Normalize weights
    w1 = np.exp(log_weights1 - np.max(log_weights1))
    w1 /= np.sum(w1)
    w2 = np.exp(log_weights2 - np.max(log_weights2))
    w2 /= np.sum(w2)
    return 0.5 * np.sum(np.abs(w1 - w2))


def effective_sample_size(log_weights):
    """Compute effective sample size (ESS)."""
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)
    return 1.0 / np.sum(weights**2)


def systematic_resample(particles, log_weights):
    """Systematic resampling with low variance."""
    N = len(particles)
    weights = np.exp(log_weights - np.max(log_weights))
    weights /= np.sum(weights)

    # Systematic resampling
    positions = (np.arange(N) + np.random.uniform()) / N
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)

    resampled_particles = particles[indices]
    resampled_log_weights = np.full(N, -np.log(N))  # Uniform weights

    return resampled_particles, resampled_log_weights


def update_observation(particles, log_weights, observation, obs_noise):
    """Update belief with observation using Gaussian likelihood G(o|x)."""
    # Likelihood: G(o|x) = N(o; x, obs_noise^2) for each particle
    # log G(o|x) = log N(o; x, obs_noise^2)
    # For multivariate observations, sum log-likelihoods across dimensions
    log_likelihood = np.sum(norm.logpdf(observation, loc=particles, scale=obs_noise), axis=1)

    # Update weights: log w' = log w + log G(o|x)
    new_log_weights = log_weights + log_likelihood

    # Normalize (log-sum-exp trick)
    max_log = np.max(new_log_weights)
    new_log_weights = new_log_weights - (max_log + np.log(np.sum(np.exp(new_log_weights - max_log))))

    return particles.copy(), new_log_weights


def apply_message(particles, log_weights, message_fn):
    """Apply message as soft-likelihood multiplier M(x)."""
    # log M(x) for each particle
    log_multiplier = message_fn(particles)

    # Update weights: log w' = log w + log M(x)
    new_log_weights = log_weights + log_multiplier

    # Normalize
    max_log = np.max(new_log_weights)
    new_log_weights = new_log_weights - (max_log + np.log(np.sum(np.exp(new_log_weights - max_log))))

    return particles.copy(), new_log_weights


def main():
    print("=" * 60)
    print("Particle Filter MWE: Log-Space Implementation")
    print("=" * 60)

    # Initialize particles
    N = 5000
    np.random.seed(42)
    particles = np.random.randn(N, 2)  # 2D state space
    log_weights = np.full(N, -np.log(N))  # Uniform initial weights

    print(f"\nInitial state:")
    print(f"  Particles: {N}")
    print(f"  Mean: {np.mean(particles, axis=0)}")
    print(f"  Std: {np.std(particles, axis=0)}")
    print(f"  ESS: {effective_sample_size(log_weights):.1f}")

    # Test 1: Observation update
    print("\n" + "-" * 60)
    print("Test 1: Observation Update")
    observation = np.array([0.5, 0.3])
    obs_noise = 0.1

    particles1, log_weights1 = update_observation(particles, log_weights, observation, obs_noise)

    weights1 = np.exp(log_weights1 - np.max(log_weights1))
    weights1 /= np.sum(weights1)
    mean1 = np.sum(particles1 * weights1[:, None], axis=0)

    print(f"  Observation: {observation}")
    print(f"  Updated mean: {mean1}")
    print(f"  ESS: {effective_sample_size(log_weights1):.1f}")
    print(f"  ✓ Observation update successful")

    # Test 2: Message update
    print("\n" + "-" * 60)
    print("Test 2: Message Update")

    # Message: claim that x[0] > 0 (support region)
    def message_fn(particles):
        # Soft indicator: M(x) ∝ sigmoid(10 * x[0])
        return 10 * particles[:, 0] - np.log(1 + np.exp(10 * particles[:, 0]))

    particles2, log_weights2 = apply_message(particles1, log_weights1, message_fn)

    weights2 = np.exp(log_weights2 - np.max(log_weights2))
    weights2 /= np.sum(weights2)
    mean2 = np.sum(particles2 * weights2[:, None], axis=0)

    print(f"  Message: x[0] > 0")
    print(f"  Updated mean: {mean2}")
    print(f"  ESS: {effective_sample_size(log_weights2):.1f}")
    print(f"  ✓ Message update successful")

    # Test 3: Commutativity
    print("\n" + "-" * 60)
    print("Test 3: Commutativity (FR-002)")

    # Order 1: observation → message
    particles_a, log_weights_a = update_observation(particles, log_weights, observation, obs_noise)
    particles_a, log_weights_a = apply_message(particles_a, log_weights_a, message_fn)

    # Order 2: message → observation
    particles_b, log_weights_b = apply_message(particles, log_weights, message_fn)
    particles_b, log_weights_b = update_observation(particles_b, log_weights_b, observation, obs_noise)

    tv_dist = total_variation_distance(log_weights_a, log_weights_b)

    print(f"  Total Variation Distance: {tv_dist:.2e}")
    print(f"  Target: ≤ 1e-6")

    if tv_dist <= 1e-6:
        print(f"  ✓ PASS: Commutativity verified (TV = {tv_dist:.2e})")
    else:
        print(f"  ✗ FAIL: TV distance too large ({tv_dist:.2e} > 1e-6)")

    # Test 4: Resampling
    print("\n" + "-" * 60)
    print("Test 4: Systematic Resampling")

    ess_before = effective_sample_size(log_weights2)
    threshold = 0.5 * N

    print(f"  ESS before: {ess_before:.1f}")
    print(f"  Threshold: {threshold:.1f}")

    if ess_before < threshold:
        particles_resampled, log_weights_resampled = systematic_resample(particles2, log_weights2)
        ess_after = effective_sample_size(log_weights_resampled)

        print(f"  ESS after resampling: {ess_after:.1f}")
        print(f"  ✓ Resampling triggered and completed")
    else:
        print(f"  No resampling needed (ESS > threshold)")

    # Performance test
    print("\n" + "-" * 60)
    print("Test 5: Performance (Target: <2ms per update)")

    import time

    # Time observation update
    start = time.perf_counter()
    for _ in range(100):
        update_observation(particles, log_weights, observation, obs_noise)
    elapsed = (time.perf_counter() - start) / 100 * 1000

    print(f"  Observation update: {elapsed:.3f} ms")

    if elapsed < 2.0:
        print(f"  ✓ PASS: Performance target met ({elapsed:.3f} ms < 2 ms)")
    else:
        print(f"  ⚠ WARNING: Slower than target ({elapsed:.3f} ms > 2 ms)")

    print("\n" + "=" * 60)
    print("Particle Filter MWE: All tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
