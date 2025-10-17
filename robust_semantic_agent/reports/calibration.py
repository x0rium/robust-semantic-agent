"""
Calibration Metrics and Visualization
Feature: 002-full-prototype
Tasks: T070, T071, T072, T073

Implements:
- ECE (Expected Calibration Error)
- Brier score
- Reliability diagrams
- ROC curves

References:
- docs/theory.md §6: Calibration
- SC-008: ECE ≤ 0.05
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_ece(predictions: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ (n_b / N) · |acc_b - conf_b|

    Where:
    - n_b: Number of samples in bin b
    - N: Total samples
    - acc_b: Accuracy in bin b (fraction of positives)
    - conf_b: Average confidence in bin b

    Args:
        predictions: Predicted probabilities (N,) in [0, 1]
        outcomes: Binary outcomes (N,) in {0, 1}
        n_bins: Number of bins for binning predictions

    Returns:
        ECE value in [0, 1]

    References:
        - SC-008: ECE ≤ 0.05 requirement
        - Naeini et al. (2015): Obtaining Well Calibrated Probabilities

    Example:
        >>> predictions = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        >>> outcomes = np.array([1, 1, 0, 0, 0])
        >>> ece = compute_ece(predictions, outcomes, n_bins=5)
    """
    predictions = np.asarray(predictions)
    outcomes = np.asarray(outcomes)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predictions, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    N = len(predictions)

    for b in range(n_bins):
        # Get samples in bin
        mask = bin_indices == b
        n_b = np.sum(mask)

        if n_b == 0:
            continue

        # Accuracy in bin (fraction of positive outcomes)
        acc_b = np.mean(outcomes[mask])

        # Average confidence in bin
        conf_b = np.mean(predictions[mask])

        # Add weighted contribution
        ece += (n_b / N) * np.abs(acc_b - conf_b)

    return ece


def compute_brier(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error for probabilities).

    BS = (1/N) Σ (p_i - o_i)²

    Where:
    - p_i: Predicted probability for sample i
    - o_i: Actual outcome (0 or 1)

    Args:
        predictions: Predicted probabilities (N,) in [0, 1]
        outcomes: Binary outcomes (N,) in {0, 1}

    Returns:
        Brier score in [0, 1] (lower is better)

    References:
        - Brier (1950): Verification of forecasts
        - Murphy (1973): A new vector partition of the probability score

    Example:
        >>> predictions = np.array([0.9, 0.1, 0.6])
        >>> outcomes = np.array([1, 0, 1])
        >>> brier = compute_brier(predictions, outcomes)
    """
    predictions = np.asarray(predictions)
    outcomes = np.asarray(outcomes)

    # Mean squared error
    brier_score = np.mean((predictions - outcomes) ** 2)

    return brier_score


def generate_reliability_diagram(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    output_path: str,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
) -> None:
    """
    Generate reliability diagram (calibration plot).

    Plots predicted probability vs observed frequency in bins.
    Perfect calibration → points lie on diagonal.

    Args:
        predictions: Predicted probabilities (N,)
        outcomes: Binary outcomes (N,)
        output_path: Path to save figure
        n_bins: Number of bins
        title: Plot title

    References:
        - DeGroot & Fienberg (1983): Comparison of probability forecasters
        - SC-008: Calibration visualization

    Example:
        >>> generate_reliability_diagram(
        ...     predictions, outcomes,
        ...     "reports/reliability.png", n_bins=10
        ... )
    """
    predictions = np.asarray(predictions)
    outcomes = np.asarray(outcomes)

    # Create bins
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(predictions, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute accuracy and confidence per bin
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for b in range(n_bins):
        mask = bin_indices == b
        n_b = np.sum(mask)

        if n_b == 0:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            bin_counts.append(0)
        else:
            bin_accuracies.append(np.mean(outcomes[mask]))
            bin_confidences.append(np.mean(predictions[mask]))
            bin_counts.append(n_b)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

    # Calibration curve
    valid_mask = ~np.isnan(bin_accuracies)
    ax.plot(
        np.array(bin_confidences)[valid_mask],
        np.array(bin_accuracies)[valid_mask],
        "o-",
        markersize=8,
        linewidth=2,
        label="Model",
    )

    # Bar chart of sample counts
    ax2 = ax.twinx()
    ax2.bar(
        bin_centers, bin_counts, width=1 / n_bins, alpha=0.3, color="gray", label="Sample Count"
    )
    ax2.set_ylabel("Count", fontsize=12)

    ax.set_xlabel("Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_roc_curve(
    predictions: np.ndarray, outcomes: np.ndarray, output_path: str, title: str = "ROC Curve"
) -> float:
    """
    Generate Receiver Operating Characteristic (ROC) curve.

    Plots True Positive Rate vs False Positive Rate at various thresholds.

    Args:
        predictions: Predicted probabilities (N,)
        outcomes: Binary outcomes (N,)
        output_path: Path to save figure
        title: Plot title

    Returns:
        AUC (Area Under Curve) score

    References:
        - Fawcett (2006): An introduction to ROC analysis
        - SC-008: Classification performance visualization

    Example:
        >>> auc = generate_roc_curve(
        ...     predictions, outcomes, "reports/roc.png"
        ... )
    """
    predictions = np.asarray(predictions)
    outcomes = np.asarray(outcomes)

    # Sort by prediction (descending)
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_outcomes = outcomes[sorted_indices]

    # Compute TPR and FPR at each threshold
    n_positives = np.sum(outcomes == 1)
    n_negatives = np.sum(outcomes == 0)

    tpr = []
    fpr = []

    for i in range(len(sorted_predictions) + 1):
        if i == 0:
            # Threshold = 1.0 (predict all negative)
            tp = 0
            fp = 0
        else:
            # Predict first i samples as positive
            predicted_positives = sorted_outcomes[:i]
            tp = np.sum(predicted_positives == 1)
            fp = np.sum(predicted_positives == 0)

        tpr_val = tp / n_positives if n_positives > 0 else 0
        fpr_val = fp / n_negatives if n_negatives > 0 else 0

        tpr.append(tpr_val)
        fpr.append(fpr_val)

    tpr = np.array(tpr)
    fpr = np.array(fpr)

    # Compute AUC using trapezoidal rule
    auc = np.trapz(tpr, fpr)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # ROC curve
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {auc:.3f})")

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return auc


def __repr__() -> str:
    return (
        "Calibration module: compute_ece(), compute_brier(), "
        "generate_reliability_diagram(), generate_roc_curve()"
    )
