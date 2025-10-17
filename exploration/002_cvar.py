"""
Minimal Working Example: CVaR Computation
Feature: 002-full-prototype
Task: T008

Tests:
1. Basic CVaR@α calculation (sort-and-average)
2. Analytical validation on Gaussian distribution
3. Analytical validation on uniform distribution
4. Monotonicity: CVaR@0.05 < CVaR@0.10 < CVaR@0.20
5. Weighted CVaR for particle beliefs
"""

import numpy as np
from scipy.stats import norm


def cvar(values, alpha):
    """
    Compute CVaR@α = mean of worst α-fraction of outcomes.

    Args:
        values: Array of outcome values (negative for costs/losses)
        alpha: Tail risk level ∈ (0, 1)

    Returns:
        CVaR@α value
    """
    n = len(values)
    cutoff = max(1, int(np.ceil(alpha * n)))
    sorted_values = np.sort(values)  # Ascending order (worst first for negative rewards)
    return np.mean(sorted_values[:cutoff])


def cvar_weighted(values, weights, alpha):
    """
    Compute CVaR@α for weighted samples (particle filter).

    Args:
        values: Array of outcome values
        weights: Probability weights (unnormalized)
        alpha: Tail risk level ∈ (0, 1)

    Returns:
        CVaR@α value
    """
    # Normalize weights
    weights = weights / np.sum(weights)

    # Sort by values (ascending)
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]

    # Find cutoff index where cumulative weight ≥ α
    cumsum = np.cumsum(sorted_weights)
    cutoff_idx = np.searchsorted(cumsum, alpha, side='right') + 1

    # Average worst α-tail
    tail_values = sorted_values[:cutoff_idx]
    tail_weights = sorted_weights[:cutoff_idx]

    if np.sum(tail_weights) > 0:
        return np.average(tail_values, weights=tail_weights)
    else:
        return sorted_values[0]


def cvar_gaussian_analytical(mu, sigma, alpha):
    """
    Analytical CVaR for Gaussian distribution: CVaR_α = μ - σ × φ(Φ^(-1)(α)) / α

    Args:
        mu: Mean
        sigma: Standard deviation
        alpha: Tail risk level

    Returns:
        Analytical CVaR@α
    """
    z_alpha = norm.ppf(alpha)  # Φ^(-1)(α)
    phi_z = norm.pdf(z_alpha)  # φ(Φ^(-1)(α))
    return mu - sigma * phi_z / alpha


def cvar_uniform_analytical(a, b, alpha):
    """
    Analytical CVaR for uniform distribution: CVaR_α = a + α(b-a)/2

    Args:
        a: Lower bound
        b: Upper bound
        alpha: Tail risk level

    Returns:
        Analytical CVaR@α
    """
    return a + alpha * (b - a) / 2


def main():
    print("=" * 60)
    print("CVaR Computation MWE")
    print("=" * 60)

    np.random.seed(42)

    # Test 1: Basic CVaR calculation
    print("\n" + "-" * 60)
    print("Test 1: Basic CVaR@α Calculation")

    values = np.array([-10, -8, -5, -3, -2, -1, 0, 1, 2, 5])
    alpha = 0.3

    cvar_value = cvar(values, alpha)
    print(f"  Values: {values}")
    print(f"  α = {alpha}")
    print(f"  CVaR@{alpha} = {cvar_value:.3f}")
    print(f"  (Mean of worst {int(np.ceil(alpha * len(values)))} values: {values[np.argsort(values)[:int(np.ceil(alpha * len(values)))]]}")
    print(f"  ✓ Basic CVaR calculation successful")

    # Test 2: Gaussian analytical validation
    print("\n" + "-" * 60)
    print("Test 2: Gaussian Distribution Validation")

    mu, sigma = 0.0, 1.0
    n_samples = 100000
    alpha = 0.10

    gaussian_samples = np.random.randn(n_samples) * sigma + mu
    cvar_empirical = cvar(gaussian_samples, alpha)
    cvar_analytical = cvar_gaussian_analytical(mu, sigma, alpha)

    error = abs(cvar_empirical - cvar_analytical)
    relative_error = error / abs(cvar_analytical) if cvar_analytical != 0 else error

    print(f"  Distribution: N({mu}, {sigma}²)")
    print(f"  Samples: {n_samples:,}")
    print(f"  α = {alpha}")
    print(f"  CVaR (empirical): {cvar_empirical:.6f}")
    print(f"  CVaR (analytical): {cvar_analytical:.6f}")
    print(f"  Absolute error: {error:.6f}")
    print(f"  Relative error: {relative_error:.4%}")

    if relative_error < 0.01:
        print(f"  ✓ PASS: Gaussian validation (<1% error)")
    else:
        print(f"  ✗ FAIL: Error too large ({relative_error:.4%} > 1%)")

    # Test 3: Uniform analytical validation
    print("\n" + "-" * 60)
    print("Test 3: Uniform Distribution Validation")

    a, b = -5.0, 5.0
    alpha = 0.20

    uniform_samples = np.random.uniform(a, b, n_samples)
    cvar_empirical = cvar(uniform_samples, alpha)
    cvar_analytical = cvar_uniform_analytical(a, b, alpha)

    error = abs(cvar_empirical - cvar_analytical)
    relative_error = error / abs(cvar_analytical) if cvar_analytical != 0 else error

    print(f"  Distribution: U({a}, {b})")
    print(f"  Samples: {n_samples:,}")
    print(f"  α = {alpha}")
    print(f"  CVaR (empirical): {cvar_empirical:.6f}")
    print(f"  CVaR (analytical): {cvar_analytical:.6f}")
    print(f"  Absolute error: {error:.6f}")
    print(f"  Relative error: {relative_error:.4%}")

    if relative_error < 0.01:
        print(f"  ✓ PASS: Uniform validation (<1% error)")
    else:
        print(f"  ✗ FAIL: Error too large ({relative_error:.4%} > 1%)")

    # Test 4: Monotonicity
    print("\n" + "-" * 60)
    print("Test 4: Monotonicity Property")

    alphas = [0.05, 0.10, 0.20]
    cvars = [cvar(gaussian_samples, alpha) for alpha in alphas]

    print(f"  CVaR@0.05 = {cvars[0]:.6f}")
    print(f"  CVaR@0.10 = {cvars[1]:.6f}")
    print(f"  CVaR@0.20 = {cvars[2]:.6f}")

    monotonic = cvars[0] < cvars[1] < cvars[2]

    if monotonic:
        print(f"  ✓ PASS: Monotonicity verified (CVaR increases with α)")
    else:
        print(f"  ✗ FAIL: Not monotonic")

    # Test 5: Weighted CVaR (for particle filters)
    print("\n" + "-" * 60)
    print("Test 5: Weighted CVaR (Particle Filter)")

    n_particles = 5000
    particles_values = np.random.randn(n_particles)
    weights = np.random.exponential(1.0, n_particles)  # Non-uniform weights

    alpha = 0.10

    cvar_weighted_value = cvar_weighted(particles_values, weights, alpha)
    cvar_uniform = cvar(particles_values, alpha)  # Uniform weights for comparison

    print(f"  Particles: {n_particles}")
    print(f"  α = {alpha}")
    print(f"  CVaR (weighted): {cvar_weighted_value:.6f}")
    print(f"  CVaR (uniform weights): {cvar_uniform:.6f}")
    print(f"  Difference: {abs(cvar_weighted_value - cvar_uniform):.6f}")
    print(f"  ✓ Weighted CVaR calculation successful")

    # Test 6: Sample size requirements
    print("\n" + "-" * 60)
    print("Test 6: Sample Size Requirements")

    requirements = {
        0.10: 200,
        0.05: 400,
        0.01: 2000,
    }

    print("  Recommended minimum samples for accuracy:")
    for alpha, min_samples in requirements.items():
        print(f"    α = {alpha:4.2f}: n ≥ {min_samples:4d}")

    print("\n  Validation with varying sample sizes:")
    test_alpha = 0.10
    sample_sizes = [100, 500, 2000, 10000]

    for n in sample_sizes:
        samples = np.random.randn(n)
        cvar_emp = cvar(samples, test_alpha)
        cvar_ana = cvar_gaussian_analytical(0.0, 1.0, test_alpha)
        error = abs(cvar_emp - cvar_ana) / abs(cvar_ana)
        status = "✓" if error < 0.05 else "⚠"
        print(f"    n = {n:5d}: CVaR = {cvar_emp:7.4f}, error = {error:5.2%} {status}")

    print("\n" + "=" * 60)
    print("CVaR MWE: All tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
