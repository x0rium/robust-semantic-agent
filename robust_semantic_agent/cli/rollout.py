"""
CLI: Rollout Script
Feature: 002-full-prototype
Task: T041

Run agent rollouts with visualization and logging.

Usage:
    python -m robust_semantic_agent.cli.rollout --config configs/default.yaml --episodes 10

References:
- FR-013: Episode logging
- quickstart.md: Demo rollout examples
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core.config import Configuration
from ..core.episode import Episode
from ..envs.forbidden_circle.env import ForbiddenCircleEnv
from ..policy.agent import Agent


def main():
    """Main rollout execution."""
    parser = argparse.ArgumentParser(description="Run RSA agent rollouts")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--render", action="store_true", help="Enable visualization (future)")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="runs",
        help="Directory for episode logs",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--enable-query",
        action="store_true",
        help="Enable query action (Task T065)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = Configuration.from_yaml(args.config)

    # Override query setting if specified
    if args.enable_query:
        config.query.enabled = True
        logger.info("Query action enabled via --enable-query flag")

    config.validate()

    # Set random seed
    np.random.seed(config.seed)

    # Create environment and agent
    env = ForbiddenCircleEnv(config)
    agent = Agent(config)

    # Prepare log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "episodes.jsonl"

    logger.info(f"Starting {args.episodes} episodes")
    logger.info(f"Logging to {log_file}")

    # Statistics
    total_returns = []
    goal_successes = 0
    safety_violations = 0
    total_steps = 0
    filter_activations = 0
    query_triggers = 0  # Task T065: Query statistics
    entropy_reductions = []

    # Run episodes
    for ep in range(args.episodes):
        episode = Episode(episode_id=ep, config_hash=str(config.seed))

        obs = env.reset()
        agent.reset()
        done = False

        while not done:
            # Agent acts (Task T065: pass env for query action)
            action, info = agent.act(obs, env=env)

            # Environment steps
            obs_next, reward, done, env_info = env.step(action)

            # Record step
            episode.add_step(
                state=env_info["true_state"],
                action=action,
                observation=obs,
                reward=reward,
                info={**info, **env_info},
            )

            # Update statistics
            total_steps += 1
            if info.get("safety_filter_active", False):
                filter_activations += 1
            if env_info.get("violated_safety", False):
                safety_violations += 1

            # Task T065: Query statistics
            if info.get("query_triggered", False):
                query_triggers += 1
                # Compute entropy reduction
                H_before = info.get("entropy_before_query")
                H_after = info.get("entropy_after_query")
                if H_before and H_after:
                    reduction = (H_before - H_after) / H_before
                    entropy_reductions.append(reduction)

            obs = obs_next

        # Episode complete
        total_returns.append(episode.total_return)
        if env_info.get("goal_reached", False):
            goal_successes += 1

        # Save episode
        episode.save(log_file)

        # Print progress
        logger.info(
            f"Episode {ep+1}/{args.episodes}: "
            f"Return={episode.total_return:.2f}, "
            f"Steps={len(episode.steps)}, "
            f"Goal={'✓' if env_info.get('goal_reached') else '✗'}"
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Rollout Summary")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Average Return: {np.mean(total_returns):.2f} ± {np.std(total_returns):.2f}")
    print(f"Goal Success Rate: {goal_successes / args.episodes:.1%}")
    print(f"Safety Violations: {safety_violations} / {total_steps} steps")
    print(
        f"Filter Activation Rate: {filter_activations / total_steps:.2%}"
        if total_steps > 0
        else "N/A"
    )

    # Task T065: Query statistics
    if config.query.enabled:
        print("\nQuery Action Statistics:")
        print(
            f"  Queries Triggered: {query_triggers} / {total_steps} steps ({query_triggers/total_steps:.2%})"
        )
        if entropy_reductions:
            mean_reduction = np.mean(entropy_reductions)
            print(f"  Average Entropy Reduction: {mean_reduction:.2%}")
            print(f"  SC-007 (≥20% reduction): {'✅ PASS' if mean_reduction >= 0.20 else '❌ FAIL'}")

    print(f"\nLogs saved to: {log_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
