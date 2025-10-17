"""
CLI: Evaluation Script
Feature: 002-full-prototype
Task: T081

Generate comprehensive reports from logged episodes.

Usage:
    python -m robust_semantic_agent.cli.evaluate --runs-dir runs/20240101_120000 --output reports

References:
- SC-001 through SC-013: All success criteria
- User Story 5: Performance monitoring
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from ..reports.risk import generate_cvar_curves, generate_tail_distributions
from ..reports.safety import compute_violation_rates, generate_barrier_traces


def load_episodes_from_jsonl(file_path: Path) -> list:
    """Load episodes from JSONL file."""
    episodes = []

    with open(file_path) as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    return episodes


def main():
    """Main evaluation execution."""
    parser = argparse.ArgumentParser(description="Generate RSA evaluation reports")
    parser.add_argument(
        "--runs-dir",
        type=str,
        required=True,
        help="Directory containing episode logs (JSONL files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports (default: reports/)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Optional baseline runs directory for comparison",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load episodes
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    # Find episodes.jsonl
    episodes_file = runs_dir / "episodes.jsonl"
    if not episodes_file.exists():
        raise FileNotFoundError(f"Episodes file not found: {episodes_file}")

    logger.info(f"Loading episodes from {episodes_file}")
    episodes = load_episodes_from_jsonl(episodes_file)
    logger.info(f"Loaded {len(episodes)} episodes")

    # Load baseline if provided
    baseline_episodes = None
    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        baseline_file = baseline_dir / "episodes.jsonl"
        if baseline_file.exists():
            logger.info(f"Loading baseline from {baseline_file}")
            baseline_episodes = load_episodes_from_jsonl(baseline_file)
            logger.info(f"Loaded {len(baseline_episodes)} baseline episodes")

    # Create output directories
    output_dir = Path(args.output)
    risk_dir = output_dir / "risk"
    safety_dir = output_dir / "safety"
    credal_dir = output_dir / "credal"
    calibration_dir = output_dir / "calibration"

    for dir_path in [risk_dir, safety_dir, credal_dir, calibration_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # === Risk Reports (T076, T077) ===
    logger.info("Generating risk reports...")

    # CVaR curves
    alphas = np.linspace(0.05, 1.0, 20)
    cvar_results = generate_cvar_curves(
        episodes,
        alphas,
        output_path=str(risk_dir / "cvar_curves.png"),
        baseline_episodes=baseline_episodes,
    )

    # Tail distributions
    generate_tail_distributions(
        episodes,
        output_path=str(risk_dir / "tail_distributions.png"),
        baseline_episodes=baseline_episodes,
    )

    # Save CVaR results
    with open(risk_dir / "cvar_results.json", "w") as f:
        json.dump(cvar_results, f, indent=2)

    # === Safety Reports (T078, T079) ===
    logger.info("Generating safety reports...")

    # Barrier traces
    generate_barrier_traces(
        episodes, output_path=str(safety_dir / "barrier_traces.png"), max_episodes=10
    )

    # Violation rates
    violation_stats = compute_violation_rates(episodes)

    # Save violation stats
    with open(safety_dir / "violation_stats.json", "w") as f:
        json.dump(violation_stats, f, indent=2)

    # === Credal Set Reports (T080) ===
    logger.info("Generating credal set reports...")

    # Extract credal sets from episodes (if any)
    credal_sets_found = 0
    for ep_idx, episode in enumerate(episodes[:5]):  # First 5 episodes
        if "steps" not in episode:
            continue

        for step_idx, step in enumerate(episode["steps"]):
            info = step.get("info", {})
            if info.get("credal_set_active", False):
                credal_sets_found += 1
                # Note: Would need to save credal set data in episode logs
                # For now, skip visualization if not available
                break

    if credal_sets_found > 0:
        logger.info(f"Found {credal_sets_found} credal set activations")
    else:
        logger.warning("No credal sets found in episodes")

    # === Summary Statistics ===
    logger.info("Computing summary statistics...")

    # Extract metrics
    returns = [ep.get("total_return", 0.0) for ep in episodes]
    goal_successes = sum(1 for ep in episodes if ep.get("goal_reached", False))
    total_steps = violation_stats["total_steps"]

    # Compute CVaR@0.1 for SC-010
    cvar_01_rsa = None
    cvar_01_baseline = None
    if len(returns) > 0:
        from ..risk.cvar import cvar

        cvar_01_rsa = cvar(np.array(returns), alpha=0.1)

    if baseline_episodes:
        baseline_returns = [ep.get("total_return", 0.0) for ep in baseline_episodes]
        if len(baseline_returns) > 0:
            cvar_01_baseline = cvar(np.array(baseline_returns), alpha=0.1)

    # SC-010: Risk-averse CVaR check
    sc010_pass = False
    if cvar_01_rsa is not None and cvar_01_baseline is not None:
        sc010_pass = cvar_01_rsa >= cvar_01_baseline

    # Create summary
    summary = {
        "episodes": len(episodes),
        "total_steps": total_steps,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "std_return": float(np.std(returns)) if returns else 0.0,
        "goal_success_rate": goal_successes / len(episodes) if episodes else 0.0,
        "cvar_01": float(cvar_01_rsa) if cvar_01_rsa is not None else None,
        "cvar_01_baseline": float(cvar_01_baseline) if cvar_01_baseline is not None else None,
        "success_criteria": {
            "SC-001 (zero violations)": violation_stats["sc001_pass"],
            "SC-002 (filter ≥1%)": violation_stats["sc002_pass"],
            "SC-010 (CVaR ≥ baseline)": sc010_pass if cvar_01_baseline else None,
        },
        "safety": violation_stats,
    }

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # === Print Summary ===
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {len(episodes)}")
    print(f"Total Steps: {total_steps}")
    print("\nPerformance:")
    print(f"  Mean Return: {summary['mean_return']:.2f} ± {summary['std_return']:.2f}")
    print(f"  Goal Success Rate: {summary['goal_success_rate']:.1%}")
    print(f"  CVaR@0.1: {cvar_01_rsa:.2f}" if cvar_01_rsa else "  CVaR@0.1: N/A")

    print("\nSafety:")
    print(f"  Violations: {violation_stats['violations']}")
    print(f"  Violation Rate: {violation_stats['violation_rate']:.4%}")
    print(
        f"  Filter Activations: {violation_stats['filter_activations']} ({violation_stats['filter_activation_rate']:.2%})"
    )

    print("\nSuccess Criteria:")
    for criterion, passed in summary["success_criteria"].items():
        if passed is None:
            status = "⚠️  N/A"
        elif passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"  {criterion}: {status}")

    print(f"\nReports saved to: {output_dir}")
    print("  - risk/cvar_curves.png")
    print("  - risk/tail_distributions.png")
    print("  - safety/barrier_traces.png")
    print("  - safety/violation_stats.json")
    print("  - summary.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
