"""
Credal Set Visualization
Feature: 002-full-prototype
Task: T080

Generate posterior ensemble plots for credal sets (v=⊤ contradictions).

References:
- docs/theory.md §2: Credal sets for contradictions
- SC-004: Lower expectation monotonicity
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_posterior_ensemble_plot(
    credal_set,
    output_path: str,
    feature_indices: tuple = (0, 1),
    title: str = "Credal Set: Posterior Ensemble",
) -> None:
    """
    Visualize credal set as ensemble of K extreme posteriors.

    Task T080: Plot particle distributions for all K posteriors in credal set.

    Shows how belief expands into credal set when receiving v=⊤ (contradiction).
    Each posterior represents an extreme interpretation of contradictory information.

    Args:
        credal_set: CredalSet object with posteriors attribute
        output_path: Path to save figure
        feature_indices: Which state dimensions to plot (default: (0, 1) for 2D)
        title: Plot title

    References:
        - SC-004: Lower expectation ≤ any posterior
        - docs/theory.md §2.4: Credal set expansion

    Example:
        >>> from robust_semantic_agent.core.credal import CredalSet
        >>> credal_set = CredalSet(posteriors=[...])
        >>> generate_posterior_ensemble_plot(credal_set, "reports/credal.png")
    """
    if credal_set is None or len(credal_set.posteriors) == 0:
        # Nothing to plot
        return

    K = credal_set.K
    posteriors = credal_set.posteriors

    # Determine grid layout
    n_cols = min(3, K)
    n_rows = (K + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # Flatten axes for easier indexing
    if K == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # Plot each posterior
    for k in range(K):
        ax = axes[k]
        posterior = posteriors[k]

        # Get particles and weights
        particles = posterior.particles
        weights = np.exp(posterior.log_weights - np.max(posterior.log_weights))
        weights /= np.sum(weights)

        # Extract feature dimensions
        x_feat = particles[:, feature_indices[0]]
        y_feat = particles[:, feature_indices[1]]

        # Scatter plot weighted by particle weight
        scatter = ax.scatter(
            x_feat,
            y_feat,
            c=weights,
            s=10,
            alpha=0.6,
            cmap="viridis",
            vmin=0,
            vmax=weights.max() if weights.max() > 0 else 1,
        )

        # Compute and plot mean
        mean = posterior.mean()
        ax.plot(
            mean[feature_indices[0]], mean[feature_indices[1]], "r*", markersize=15, label="Mean"
        )

        ax.set_title(f"Posterior {k+1}/{K}", fontsize=12)
        ax.set_xlabel(f"State Dim {feature_indices[0]}")
        ax.set_ylabel(f"State Dim {feature_indices[1]}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Colorbar
        plt.colorbar(scatter, ax=ax, label="Particle Weight")

    # Hide empty subplots
    for k in range(K, len(axes)):
        axes[k].axis("off")

    fig.suptitle(title, fontsize=16)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def __repr__() -> str:
    return "Credal reporting: generate_posterior_ensemble_plot()"
