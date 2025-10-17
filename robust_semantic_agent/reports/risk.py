"""
Risk Analysis Reports
Feature: 002-full-prototype
Tasks: T076, T077

Generate CVaR curves and tail distribution visualizations.

References:
- docs/theory.md §3: Risk measures
- SC-010: Risk-averse CVaR ≥ baseline
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..risk.cvar import cvar


def generate_cvar_curves(
    episodes: list[dict], alphas: np.ndarray, output_path: str, baseline_episodes: list[dict] = None
) -> dict:
    """
    Generate CVaR curves showing risk sensitivity across α values.

    Task T076: CVaR curve visualization for risk analysis.

    Plots CVaR@α for various α ∈ (0, 1], comparing RSA agent vs baseline.

    Args:
        episodes: List of episode dicts with 'total_return' key
        alphas: Array of CVaR risk levels (e.g., [0.05, 0.1, 0.2, ..., 1.0])
        output_path: Path to save figure
        baseline_episodes: Optional baseline episodes for comparison

    Returns:
        Dict with CVaR values at each alpha

    References:
        - SC-010: Risk-averse CVaR ≥ baseline
        - docs/theory.md §3.1: CVaR definition

    Example:
        >>> alphas = np.linspace(0.05, 1.0, 20)
        >>> results = generate_cvar_curves(episodes, alphas, "reports/cvar.png")
    """
    # Extract returns
    returns = np.array([ep["total_return"] for ep in episodes])

    # Compute CVaR for each alpha
    cvar_values = []
    for alpha in alphas:
        cvar_val = cvar(returns, alpha)
        cvar_values.append(cvar_val)

    cvar_values = np.array(cvar_values)

    # Compute baseline if provided
    baseline_cvar = None
    if baseline_episodes:
        baseline_returns = np.array([ep["total_return"] for ep in baseline_episodes])
        baseline_cvar = []
        for alpha in alphas:
            baseline_cvar.append(cvar(baseline_returns, alpha))
        baseline_cvar = np.array(baseline_cvar)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # RSA agent
    ax.plot(alphas, cvar_values, "b-", linewidth=2, label="RSA Agent", marker="o")

    # Baseline
    if baseline_cvar is not None:
        ax.plot(alphas, baseline_cvar, "r--", linewidth=2, label="Baseline", marker="s")

        # Highlight where RSA > baseline (risk-averse)
        better_mask = cvar_values > baseline_cvar
        if np.any(better_mask):
            ax.fill_between(
                alphas,
                cvar_values,
                baseline_cvar,
                where=better_mask,
                alpha=0.3,
                color="green",
                label="RSA Better (Risk-Averse)",
            )

    ax.set_xlabel("Risk Level α", fontsize=12)
    ax.set_ylabel("CVaR@α (Expected Return in Worst α-Tail)", fontsize=12)
    ax.set_title("CVaR Risk Profile", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Return results
    results = {
        "alphas": alphas.tolist(),
        "cvar_values": cvar_values.tolist(),
    }
    if baseline_cvar is not None:
        results["baseline_cvar"] = baseline_cvar.tolist()

    return results


def generate_tail_distributions(
    episodes: list[dict], output_path: str, baseline_episodes: list[dict] = None
) -> None:
    """
    Generate tail distribution comparison (histogram + CDF).

    Task T077: Visualize empirical return distributions.

    Shows full distribution with focus on tail (worst outcomes).

    Args:
        episodes: List of episode dicts with 'total_return'
        output_path: Path to save figure
        baseline_episodes: Optional baseline for comparison

    References:
        - SC-010: Tail risk comparison
        - docs/theory.md §3.1: CVaR tail expectations

    Example:
        >>> generate_tail_distributions(episodes, "reports/tails.png")
    """
    returns = np.array([ep["total_return"] for ep in episodes])

    baseline_returns = None
    if baseline_episodes:
        baseline_returns = np.array([ep["total_return"] for ep in baseline_episodes])

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Histogram
    ax1 = axes[0]
    bins = 30

    ax1.hist(returns, bins=bins, alpha=0.7, label="RSA Agent", color="blue", density=True)
    if baseline_returns is not None:
        ax1.hist(
            baseline_returns, bins=bins, alpha=0.7, label="Baseline", color="red", density=True
        )

    ax1.set_xlabel("Total Return", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Return Distribution", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Empirical CDF (focus on tail)
    ax2 = axes[1]

    # Sort returns
    sorted_returns = np.sort(returns)
    cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    ax2.plot(sorted_returns, cdf, "b-", linewidth=2, label="RSA Agent")

    if baseline_returns is not None:
        sorted_baseline = np.sort(baseline_returns)
        cdf_baseline = np.arange(1, len(sorted_baseline) + 1) / len(sorted_baseline)
        ax2.plot(sorted_baseline, cdf_baseline, "r--", linewidth=2, label="Baseline")

    # Highlight worst 10% (α=0.1 tail)
    ax2.axhline(0.1, color="gray", linestyle=":", linewidth=1, label="α=0.1 Tail")
    ax2.fill_between(
        sorted_returns[cdf <= 0.1], 0, 0.1, alpha=0.3, color="orange", label="Worst 10%"
    )

    ax2.set_xlabel("Total Return", fontsize=12)
    ax2.set_ylabel("Cumulative Probability", fontsize=12)
    ax2.set_title("Empirical CDF (Tail Focus)", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 0.5])  # Focus on lower tail

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def __repr__() -> str:
    return "Risk reporting: generate_cvar_curves(), generate_tail_distributions()"
