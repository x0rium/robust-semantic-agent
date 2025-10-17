"""
Safety Analysis Reports
Feature: 002-full-prototype
Tasks: T078, T079

Generate barrier function traces and violation rate analysis.

References:
- docs/theory.md §4: Safety via CBF
- SC-001: Zero violations
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_barrier_traces(episodes: list[dict], output_path: str, max_episodes: int = 5) -> None:
    """
    Generate barrier function B(x) traces over time.

    Task T078: Visualize barrier evolution and safety margin.

    Shows B(x) values at each timestep for multiple episodes.
    Safety constraint: B(x) ≤ 0 for all x ∈ S (safe set).

    Args:
        episodes: List of episode dicts with step-level info containing 'barrier_value'
        output_path: Path to save figure
        max_episodes: Maximum number of episodes to plot (avoid clutter)

    References:
        - SC-001: Zero violations (B(x) > 0 → violation)
        - docs/theory.md §4.1: CBF definition

    Example:
        >>> generate_barrier_traces(episodes, "reports/barriers.png", max_episodes=10)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    violation_count = 0
    total_steps = 0

    # Plot first max_episodes
    for ep_idx, episode in enumerate(episodes[:max_episodes]):
        if "steps" not in episode:
            continue

        steps = episode["steps"]
        timesteps = list(range(len(steps)))

        # Extract barrier values
        barrier_values = []
        for step in steps:
            # Get barrier value from step info
            info = step.get("info", {})
            # Compute barrier value from true state if not stored
            if "barrier_value" in info:
                B = info["barrier_value"]
            else:
                # Compute from state: B(x) = r² - ||x - center||² where r = obstacle radius
                state = step.get("state", np.array([0, 0]))
                # Assume circular obstacle at origin with radius 0.3
                B = 0.3**2 - np.linalg.norm(state - np.array([0, 0])) ** 2

            barrier_values.append(B)

            # Check violation
            if B > 0:
                violation_count += 1

            total_steps += 1

        # Plot
        color = "blue" if max(barrier_values) <= 0 else "red"
        alpha = 0.7 if ep_idx < 3 else 0.3  # Emphasize first few
        ax.plot(timesteps, barrier_values, color=color, alpha=alpha, linewidth=1.5)

    # Safety threshold
    ax.axhline(0, color="black", linestyle="--", linewidth=2, label="Safety Threshold (B=0)")

    # Shade unsafe region
    ylim = ax.get_ylim()
    ax.fill_between(
        [0, max(timesteps) if episodes else 1],
        0,
        ylim[1],
        alpha=0.2,
        color="red",
        label="Unsafe Region (B>0)",
    )

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Barrier Function B(x)", fontsize=12)
    ax.set_title(
        f"Barrier Function Traces ({min(len(episodes), max_episodes)} episodes)", fontsize=14
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_violation_rates(episodes: list[dict]) -> dict:
    """
    Compute safety violation statistics.

    Task T079: Analyze violation rates across episodes.

    Returns detailed breakdown of violations, filter activations, and safety margins.

    Args:
        episodes: List of episode dicts

    Returns:
        Dict with keys:
        - total_steps: Total timesteps
        - violations: Number of safety violations
        - violation_rate: Violations per step
        - episodes_with_violations: Count of episodes with ≥1 violation
        - filter_activations: Number of CBF-QP filter activations
        - filter_activation_rate: Filter activations per step

    References:
        - SC-001: Zero violations requirement
        - SC-002: Filter activation ≥1%

    Example:
        >>> stats = compute_violation_rates(episodes)
        >>> print(f"Violation rate: {stats['violation_rate']:.4f}")
    """
    total_steps = 0
    violations = 0
    episodes_with_violations = 0
    filter_activations = 0

    for episode in episodes:
        if "steps" not in episode:
            continue

        episode_had_violation = False

        for step in episode["steps"]:
            total_steps += 1

            info = step.get("info", {})

            # Check violation
            if info.get("violated_safety", False):
                violations += 1
                episode_had_violation = True

            # Check filter activation
            if info.get("safety_filter_active", False):
                filter_activations += 1

        if episode_had_violation:
            episodes_with_violations += 1

    # Compute rates
    violation_rate = violations / total_steps if total_steps > 0 else 0.0
    filter_activation_rate = filter_activations / total_steps if total_steps > 0 else 0.0

    results = {
        "total_steps": total_steps,
        "violations": violations,
        "violation_rate": violation_rate,
        "episodes_with_violations": episodes_with_violations,
        "episodes_total": len(episodes),
        "filter_activations": filter_activations,
        "filter_activation_rate": filter_activation_rate,
        # Success criteria checks
        "sc001_pass": violations == 0,  # SC-001: Zero violations
        "sc002_pass": filter_activation_rate >= 0.01,  # SC-002: ≥1% filter activation
    }

    return results


def __repr__() -> str:
    return "Safety reporting: generate_barrier_traces(), compute_violation_rates()"
