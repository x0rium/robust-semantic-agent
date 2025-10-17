"""
Unit Tests: CVaR Risk Measure
Feature: 002-full-prototype
Tasks: T021, T022, T023

Tests MUST FAIL initially (TDD Principle V: NON-NEGOTIABLE)
"""

import numpy as np
import pytest
from scipy.stats import norm


@pytest.mark.unit
class TestCVaRGaussianAnalytical:
    """T021: Validate CVaR against Gaussian analytical formula (SC-005)."""

    def test_cvar_gaussian_analytical_match(self):
        """
        CVaR@α for Gaussian should match analytical formula within <1% error.
        Formula: CVaR_α = μ - σ × φ(Φ^(-1)(α)) / (1-α)
        """
        from robust_semantic_agent.risk.cvar import cvar

        np.random.seed(42)

        mu, sigma = 0.0, 1.0
        alpha = 0.10
        n_samples = 100000

        # Generate Gaussian samples
        samples = np.random.randn(n_samples) * sigma + mu

        # Empirical CVaR
        cvar_empirical = cvar(samples, alpha)

        # Analytical CVaR for Gaussian
        z_alpha = norm.ppf(alpha)
        phi_z = norm.pdf(z_alpha)
        cvar_analytical = mu - sigma * phi_z / alpha

        # Error
        error = abs(cvar_empirical - cvar_analytical)
        relative_error = error / abs(cvar_analytical) if cvar_analytical != 0 else error

        assert relative_error < 0.01, (
            f"CVaR Gaussian validation failed: "
            f"empirical={cvar_empirical:.6f}, analytical={cvar_analytical:.6f}, "
            f"error={relative_error:.4%} > 1%"
        )


@pytest.mark.unit
class TestCVaRUniformAnalytical:
    """T022: Validate CVaR against Uniform analytical formula."""

    def test_cvar_uniform_analytical_match(self):
        """
        CVaR@α for Uniform should match analytical formula.
        Formula: CVaR_α = a + α × (b-a) / 2
        """
        from robust_semantic_agent.risk.cvar import cvar

        np.random.seed(42)

        a, b = -5.0, 5.0
        alpha = 0.20
        n_samples = 50000

        # Generate Uniform samples
        samples = np.random.uniform(a, b, n_samples)

        # Empirical CVaR
        cvar_empirical = cvar(samples, alpha)

        # Analytical CVaR for Uniform
        cvar_analytical = a + alpha * (b - a) / 2

        # Error
        error = abs(cvar_empirical - cvar_analytical)
        relative_error = error / abs(cvar_analytical) if cvar_analytical != 0 else error

        assert relative_error < 0.02, (
            f"CVaR Uniform validation failed: "
            f"empirical={cvar_empirical:.6f}, analytical={cvar_analytical:.6f}, "
            f"error={relative_error:.4%} > 2%"
        )


@pytest.mark.unit
class TestCVaRMonotonicity:
    """T023: Verify monotonicity property: α1 < α2 → CVaR@α1 ≤ CVaR@α2."""

    def test_cvar_monotonicity_increasing_alpha(self):
        """CVaR should increase (become less risk-averse) as α increases."""
        from robust_semantic_agent.risk.cvar import cvar

        np.random.seed(42)

        samples = np.random.randn(10000)

        cvar_005 = cvar(samples, alpha=0.05)
        cvar_010 = cvar(samples, alpha=0.10)
        cvar_020 = cvar(samples, alpha=0.20)

        # For negative rewards (losses), CVaR increases toward 0 as α increases
        assert cvar_005 <= cvar_010 <= cvar_020, (
            f"Monotonicity violated: CVaR@0.05={cvar_005:.3f}, "
            f"CVaR@0.10={cvar_010:.3f}, CVaR@0.20={cvar_020:.3f}"
        )

    def test_cvar_extreme_alpha_values(self):
        """Test edge cases: α→0 (worst case) and α=1 (mean)."""
        from robust_semantic_agent.risk.cvar import cvar

        np.random.seed(42)

        samples = np.random.randn(10000)

        # α ≈ 0: Should be close to minimum (worst single outcome)
        # NOTE: With finite samples, CVaR@α is average of worst α-fraction
        # Not exactly the minimum, but close. Allow tolerance for small sample effects
        cvar_extreme = cvar(samples, alpha=0.001)
        min_sample = np.min(samples)

        # CVaR@0.001 is mean of worst 0.1% ≈ 10 samples out of 10000
        # Should be close to, but >= minimum (since it's an average)
        assert (
            cvar_extreme <= min_sample + 0.5
        ), f"CVaR@0.001 should be close to min, got {cvar_extreme:.3f} vs min={min_sample:.3f}"
        assert (
            cvar_extreme >= min_sample - 0.01
        ), f"CVaR@0.001 should be >= min, got {cvar_extreme:.3f} vs min={min_sample:.3f}"

        # α = 1: Should equal mean
        cvar_all = cvar(samples, alpha=1.0)
        mean_sample = np.mean(samples)

        assert (
            abs(cvar_all - mean_sample) < 0.01
        ), f"CVaR@1.0 should equal mean, got {cvar_all:.3f} vs mean={mean_sample:.3f}"


@pytest.mark.unit
class TestCVaRWeighted:
    """Test weighted CVaR for particle belief integration."""

    def test_cvar_weighted_particles(self):
        """Weighted CVaR should handle log-space particle weights."""
        from robust_semantic_agent.risk.cvar import cvar_weighted

        np.random.seed(42)

        n_particles = 5000
        particles_values = np.random.randn(n_particles)

        # Non-uniform log-weights (simulate belief after observation)
        log_weights = np.random.randn(n_particles)
        log_weights -= np.max(log_weights)  # Normalize

        alpha = 0.10

        cvar_value = cvar_weighted(log_weights, particles_values, alpha)

        # Should be in reasonable range (not NaN, not infinite)
        assert np.isfinite(cvar_value), f"CVaR should be finite, got {cvar_value}"
        assert cvar_value < np.max(particles_values), "CVaR should be less than max value"
        assert cvar_value > np.min(particles_values), "CVaR should be greater than min value"
