"""
CLI: Calibration Script
Feature: 002-full-prototype
Task: T074

Auto-calibrate semantic thresholds τ and τ' to achieve target ECE.

Usage:
    python -m robust_semantic_agent.cli.calibrate --target-ece 0.05 --output reports/calibration

References:
- SC-008: ECE ≤ 0.05
- docs/theory.md §6: Calibration
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ..core.semantics import calibrate_thresholds
from ..reports.calibration import (
    compute_brier,
    generate_reliability_diagram,
    generate_roc_curve,
)


def load_episodes_from_jsonl(file_path: str) -> list:
    """
    Load episode data from JSONL file.

    Expected format: Each line contains episode data with claims/statuses.
    For calibration, we need: s_c, s_bar_c, ground_truth

    Args:
        file_path: Path to JSONL file

    Returns:
        List of episode dicts
    """
    episodes = []

    with open(file_path) as f:
        for line in f:
            ep_data = json.loads(line)
            # Extract claim data if available
            # For now, generate synthetic data from episode steps
            episodes.append(ep_data)

    return episodes


def generate_synthetic_calibration_data(n_samples: int = 500) -> list:
    """
    Generate synthetic calibration data for testing.

    Args:
        n_samples: Number of samples to generate

    Returns:
        List of dicts with s_c, s_bar_c, ground_truth
    """
    np.random.seed(42)

    # Generate well-separated data
    s_c = np.concatenate(
        [
            np.random.beta(5, 2, n_samples // 2),  # High support for TRUE
            np.random.beta(2, 5, n_samples // 2),  # Low support for FALSE
        ]
    )
    s_bar_c = np.concatenate(
        [
            np.random.beta(2, 5, n_samples // 2),  # Low countersupport for TRUE
            np.random.beta(5, 2, n_samples // 2),  # High countersupport for FALSE
        ]
    )
    ground_truth = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    # Shuffle
    indices = np.random.permutation(n_samples)
    s_c = s_c[indices]
    s_bar_c = s_bar_c[indices]
    ground_truth = ground_truth[indices]

    episodes = []
    for i in range(n_samples):
        episodes.append(
            {
                "s_c": float(s_c[i]),
                "s_bar_c": float(s_bar_c[i]),
                "ground_truth": int(ground_truth[i]),
            }
        )

    return episodes


def main():
    """Main calibration execution."""
    parser = argparse.ArgumentParser(description="Calibrate semantic thresholds")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to episode data (JSONL). If not provided, uses synthetic data.",
    )
    parser.add_argument(
        "--target-ece",
        type=float,
        default=0.05,
        help="Target Expected Calibration Error (default: 0.05)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/calibration",
        help="Output directory for reports and visualizations",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of synthetic samples if no input provided (default: 500)",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load or generate data
    if args.input:
        logger.info(f"Loading episodes from {args.input}")
        episodes = load_episodes_from_jsonl(args.input)
    else:
        logger.info(f"Generating {args.n_samples} synthetic calibration samples")
        episodes = generate_synthetic_calibration_data(args.n_samples)

    logger.info(f"Loaded {len(episodes)} episodes for calibration")

    # Calibrate thresholds
    logger.info("Running threshold calibration...")
    tau_opt, tau_prime_opt, ece_before, ece_after = calibrate_thresholds(
        episodes, target_ece=args.target_ece
    )

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract predictions and outcomes for visualization
    s_c = np.array([ep["s_c"] for ep in episodes])
    s_bar_c = np.array([ep["s_bar_c"] for ep in episodes])
    ground_truth = np.array([ep["ground_truth"] for ep in episodes])

    # Convert to predictions using calibrated thresholds
    from ..core.semantics import BelnapValue, status

    predictions_before = []
    predictions_after = []

    for i in range(len(episodes)):
        # Before calibration
        v_before = status(s_c[i], s_bar_c[i], tau=0.7, tau_prime=0.3)
        if v_before == BelnapValue.TRUE:
            predictions_before.append(0.9)
        elif v_before == BelnapValue.FALSE:
            predictions_before.append(0.1)
        else:
            predictions_before.append(0.5)

        # After calibration
        v_after = status(s_c[i], s_bar_c[i], tau=tau_opt, tau_prime=tau_prime_opt)
        if v_after == BelnapValue.TRUE:
            predictions_after.append(0.9)
        elif v_after == BelnapValue.FALSE:
            predictions_after.append(0.1)
        else:
            predictions_after.append(0.5)

    predictions_before = np.array(predictions_before)
    predictions_after = np.array(predictions_after)

    # Compute metrics
    brier_before = compute_brier(predictions_before, ground_truth)
    brier_after = compute_brier(predictions_after, ground_truth)

    # Generate visualizations
    logger.info("Generating reliability diagrams...")
    generate_reliability_diagram(
        predictions_before,
        ground_truth,
        output_path=str(output_dir / "reliability_before.png"),
        title="Reliability Diagram (Before Calibration)",
    )
    generate_reliability_diagram(
        predictions_after,
        ground_truth,
        output_path=str(output_dir / "reliability_after.png"),
        title="Reliability Diagram (After Calibration)",
    )

    logger.info("Generating ROC curves...")
    auc_before = generate_roc_curve(
        predictions_before,
        ground_truth,
        output_path=str(output_dir / "roc_before.png"),
        title="ROC Curve (Before Calibration)",
    )
    auc_after = generate_roc_curve(
        predictions_after,
        ground_truth,
        output_path=str(output_dir / "roc_after.png"),
        title="ROC Curve (After Calibration)",
    )

    # Save calibration results
    results = {
        "tau_optimal": float(tau_opt),
        "tau_prime_optimal": float(tau_prime_opt),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "brier_before": float(brier_before),
        "brier_after": float(brier_after),
        "auc_before": float(auc_before),
        "auc_after": float(auc_after),
        "n_samples": len(episodes),
        "target_ece": args.target_ece,
    }

    results_file = output_dir / "calibration_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Calibration Summary")
    print("=" * 60)
    print(f"Samples: {len(episodes)}")
    print("\nOptimal Thresholds:")
    print(f"  τ  (TRUE threshold):  {tau_opt:.4f}")
    print(f"  τ' (FALSE threshold): {tau_prime_opt:.4f}")
    print("\nExpected Calibration Error (ECE):")
    print(f"  Before: {ece_before:.4f}")
    print(f"  After:  {ece_after:.4f}")
    print(f"  Improvement: {(ece_before - ece_after) / ece_before * 100:.1f}%")
    print("\nBrier Score:")
    print(f"  Before: {brier_before:.4f}")
    print(f"  After:  {brier_after:.4f}")
    print("\nAUC:")
    print(f"  Before: {auc_before:.4f}")
    print(f"  After:  {auc_after:.4f}")
    print(f"\nSC-008 (ECE ≤ {args.target_ece}): ", end="")
    print("✅ PASS" if ece_after <= args.target_ece else "❌ FAIL")
    print(f"\nReports saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
