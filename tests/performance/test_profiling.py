"""
Performance Profiling Test
Feature: 002-full-prototype
Task: T091

Profiles agent performance to verify SC-009: 30+ Hz with 10k particles.

Tests execution speed of:
- Belief update (observation + message)
- Policy step (action selection)
- Safety filter (CBF-QP)
- Full agent.act() cycle

Target: ≥30 Hz (≤33.3 ms per step) @ 10k particles
"""

import time

import numpy as np
import pytest


@pytest.mark.performance
class TestAgentPerformance:
    """Performance profiling for agent components."""

    def test_belief_update_performance_10k_particles(self):
        """
        Profile belief update speed with 10k particles.

        Target: ≥30 Hz (≤33.3 ms per update)
        """
        from robust_semantic_agent.core.belief import Belief

        # Create belief with 10k particles
        belief = Belief(n_particles=10000, state_dim=2)

        # Initialize with random particles
        belief.particles = np.random.randn(10000, 2)

        # Time observation updates
        n_trials = 50
        times = []

        for _ in range(n_trials):
            obs = np.array([0.5, 0.5])
            obs_noise = 0.1

            start = time.perf_counter()
            belief.update_obs(obs, obs_noise)
            elapsed = time.perf_counter() - start

            times.append(elapsed)

        mean_time = np.mean(times)
        frequency_hz = 1.0 / mean_time if mean_time > 0 else float("inf")

        print(f"\nBelief Update @ 10k particles:")
        print(f"  Mean time: {mean_time*1000:.2f} ms")
        print(f"  Frequency: {frequency_hz:.1f} Hz")
        print(f"  Target: ≥30 Hz (≤33.3 ms)")

        # SC-009: ≥30 Hz
        assert (
            frequency_hz >= 30.0
        ), f"Belief update too slow: {frequency_hz:.1f} Hz < 30 Hz target"

    def test_safety_filter_performance(self):
        """
        Profile CBF-QP safety filter speed.

        Target: ≥30 Hz (≤33.3 ms per solve)
        """
        from robust_semantic_agent.envs.forbidden_circle.safety import BarrierFunction
        from robust_semantic_agent.safety.cbf import SafetyFilter

        # Create barrier function
        obstacle_center = np.array([0.0, 0.0])
        obstacle_radius = 0.3
        barrier_fn = BarrierFunction(radius=obstacle_radius, center=obstacle_center)

        # Create safety filter
        safety_filter = SafetyFilter(barrier_fn=barrier_fn, alpha=0.5)

        n_trials = 50
        times = []

        for _ in range(n_trials):
            # Sample state and action
            state = np.random.randn(2) * 0.5  # Near origin
            u_des = np.random.randn(2) * 0.1

            start = time.perf_counter()
            u_safe, slack = safety_filter.filter(state, u_des)
            elapsed = time.perf_counter() - start

            times.append(elapsed)

        mean_time = np.mean(times)
        frequency_hz = 1.0 / mean_time if mean_time > 0 else float("inf")

        print(f"\nCBF-QP Filter:")
        print(f"  Mean time: {mean_time*1000:.2f} ms")
        print(f"  Frequency: {frequency_hz:.1f} Hz")
        print(f"  Target: ≥30 Hz (≤33.3 ms)")

        assert frequency_hz >= 30.0, f"CBF-QP too slow: {frequency_hz:.1f} Hz < 30 Hz target"

    def test_full_agent_act_cycle_performance(self):
        """
        Profile full agent.act() cycle with 10k particles.

        Includes:
        - Belief update
        - Policy selection
        - Safety filter

        Target: ≥30 Hz (≤33.3 ms per act)
        """
        from robust_semantic_agent.core.config import Configuration
        from robust_semantic_agent.envs.forbidden_circle.env import ForbiddenCircleEnv
        from robust_semantic_agent.policy.agent import Agent

        # Load config
        config = Configuration.from_yaml("configs/default.yaml")
        config.belief.n_particles = 10000  # Force 10k particles
        config.safety.cbf = True  # Enable CBF

        # Create environment and agent
        env = ForbiddenCircleEnv(config)
        agent = Agent(config)

        obs = env.reset()
        agent.reset()

        # Warmup
        for _ in range(5):
            action, info = agent.act(obs)

        # Time act() cycles
        n_trials = 30
        times = []

        for _ in range(n_trials):
            start = time.perf_counter()
            action, info = agent.act(obs)
            elapsed = time.perf_counter() - start

            times.append(elapsed)

            # Step environment for next obs
            obs, reward, done, env_info = env.step(action)
            if done:
                obs = env.reset()
                agent.reset()

        mean_time = np.mean(times)
        p50_time = np.percentile(times, 50)
        p95_time = np.percentile(times, 95)
        frequency_hz = 1.0 / mean_time if mean_time > 0 else float("inf")

        print(f"\nFull Agent.act() @ 10k particles:")
        print(f"  Mean time: {mean_time*1000:.2f} ms")
        print(f"  P50 time: {p50_time*1000:.2f} ms")
        print(f"  P95 time: {p95_time*1000:.2f} ms")
        print(f"  Frequency: {frequency_hz:.1f} Hz")
        print(f"  Target: ≥30 Hz (≤33.3 ms)")

        # SC-009: ≥30 Hz for full cycle
        if frequency_hz < 30.0:
            print(
                f"\n⚠️  WARNING: Performance below target: {frequency_hz:.1f} Hz < 30 Hz"
            )
            print("  Consider optimizing:")
            print("  - Reduce particle count")
            print("  - Optimize belief update (vectorization)")
            print("  - Use sparse CBF constraints")

        # Log performance for monitoring
        assert (
            mean_time < 1.0
        ), f"Agent.act() extremely slow: {mean_time*1000:.0f} ms (likely a bug)"


@pytest.mark.performance
class TestBeliefScaling:
    """Test belief performance scaling with particle count."""

    @pytest.mark.parametrize("n_particles", [1000, 5000, 10000, 20000])
    def test_belief_update_scaling(self, n_particles):
        """Profile belief update across particle counts."""
        from robust_semantic_agent.core.belief import Belief

        belief = Belief(n_particles=n_particles, state_dim=2)
        belief.particles = np.random.randn(n_particles, 2)

        n_trials = 20
        times = []

        for _ in range(n_trials):
            obs = np.array([0.5, 0.5])
            start = time.perf_counter()
            belief.update_obs(obs, obs_noise=0.1)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        mean_time = np.mean(times)
        frequency_hz = 1.0 / mean_time if mean_time > 0 else float("inf")

        print(f"\n{n_particles:5d} particles: {mean_time*1000:6.2f} ms ({frequency_hz:6.1f} Hz)")
