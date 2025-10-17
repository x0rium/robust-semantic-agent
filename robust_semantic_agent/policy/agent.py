"""
Agent: Integrated Belief Tracking + Safety + Policy
Feature: 002-full-prototype
Task: T038

Coordinates belief updates, safety filtering, and action selection.

References:
- docs/theory.md: Full system integration
- FR-001 through FR-007: Core requirements
"""

import logging
from typing import Any

import numpy as np

from ..core.belief import Belief
from ..envs.forbidden_circle.safety import BarrierFunction
from ..policy.planner import Policy
from ..safety.cbf import SafetyFilter


class Agent:
    """
    Integrated agent with belief tracking, safety, and policy.

    Components:
    - Belief: Particle filter for state estimation
    - SafetyFilter: CBF-QP for control safety
    - Policy: Action selection strategy

    Methods:
        reset(): Initialize new episode
        act(observation): Select safe action based on belief

    References:
        - Task T038: Agent implementation
        - SC-001: Zero violations
        - SC-002: Filter activation rate
    """

    def __init__(self, config):
        """
        Initialize agent from configuration.

        Args:
            config: Configuration object

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Validate configuration (production safety)
        self._validate_config()

        # Initialize belief tracker
        self.belief = Belief(
            n_particles=config.belief.particles,
            state_dim=config.env.state_dim,
            resample_threshold=config.belief.resample_threshold,
        )

        # Initialize safety filter (if enabled)
        if config.safety.cbf:
            barrier_fn = BarrierFunction(
                radius=config.env.obstacle_radius,
                center=np.array(config.env.obstacle_center),
            )
            self.safety_filter = SafetyFilter(
                barrier_fn=barrier_fn,
                alpha=config.safety.barrier_alpha,
                slack_penalty=config.safety.slack_penalty,
                max_iter=config.safety.qp_max_iter,
            )
        else:
            self.safety_filter = None

        # Initialize policy
        self.policy = Policy(goal=np.array(config.env.goal_region), gain=1.0)

        # Episode state
        self.timestep = 0

    def reset(self) -> None:
        """
        Reset agent for new episode.

        Reinitialize belief and counters.
        """
        # Reset belief to uniform
        self.belief = Belief(
            n_particles=self.config.belief.particles,
            state_dim=self.config.env.state_dim,
            resample_threshold=self.config.belief.resample_threshold,
        )

        # Initialize particles randomly
        self.belief.particles = (
            np.random.randn(self.belief.n_particles, self.belief.state_dim) * 0.5
        )

        self.timestep = 0

    def act(self, observation: np.ndarray, env=None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Select action based on observation.

        Pipeline:
        1. Update belief with observation
        2. (Optional) Compute EVI and check query trigger
        3. Select nominal action from policy
        4. Apply safety filter (if enabled)
        5. Return safe action + info

        Args:
            observation: Noisy state observation (state_dim,)
            env: Environment instance (needed for query action)

        Returns:
            action: Safe control input (action_dim,)
            info: Dict with belief_mean, safety_filter_active, query_triggered, evi, etc.

        Raises:
            ValueError: If observation is invalid
            RuntimeError: If safety filter fails critically
        """
        # Production input validation
        if observation is None:
            raise ValueError("Observation cannot be None")

        if not isinstance(observation, np.ndarray):
            try:
                observation = np.array(observation, dtype=float)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid observation type: {type(observation)}") from e

        if observation.shape[0] != self.config.env.state_dim:
            raise ValueError(
                f"Observation dimension mismatch: got {observation.shape[0]}, "
                f"expected {self.config.env.state_dim}"
            )

        if not np.all(np.isfinite(observation)):
            raise ValueError(
                f"Observation contains invalid values: {observation}"
            )
        # Update belief with observation
        obs_noise = self.config.env.observation_noise
        self.belief.update_obs(observation, obs_noise)

        # Check and resample if needed
        if self.belief.ess() < self.config.belief.resample_threshold * self.belief.n_particles:
            self.belief.resample()

        # Get belief mean estimate
        belief_mean = self.belief.mean()

        # Task T063: Query action with EVI decision rule
        query_triggered = False
        evi_value = 0.0
        entropy_before_query = None
        entropy_after_query = None

        if self.config.query.enabled and env is not None:
            # Measure entropy before potential query
            entropy_before_query = self.belief.entropy()

            # Define value function: negative distance to goal
            goal = np.array(self.config.env.goal_region)

            def value_fn(b):
                mean = b.mean()
                return -np.linalg.norm(mean - goal)

            # Compute EVI
            from ..core.query import evi, should_query

            evi_value = evi(
                self.belief,
                value_fn,
                obs_noise=obs_noise * 0.5,  # Query has lower noise
                n_samples=50,
            )

            # Check if should query
            if should_query(evi_value, self.config.query.delta_star):
                query_triggered = True

                # Execute query (get additional observation)
                from ..core.query import compute_query_observation

                query_obs = compute_query_observation(env, obs_noise * 0.5)

                # Update belief with query observation
                self.belief.update_obs(query_obs, obs_noise * 0.5)

                # Resample if needed
                if (
                    self.belief.ess()
                    < self.config.belief.resample_threshold * self.belief.n_particles
                ):
                    self.belief.resample()

                # Measure entropy after query
                entropy_after_query = self.belief.entropy()

                # Update belief mean after query
                belief_mean = self.belief.mean()

        # Select nominal action from policy
        u_desired = self.policy.select_action(self.belief)

        # Apply safety filter with production error handling
        safety_filter_active = False
        slack = 0.0
        safety_filter_error = None

        if self.safety_filter is not None:
            try:
                u_safe, slack = self.safety_filter.filter(belief_mean, u_desired)

                # Production check: Verify output is valid
                if not np.all(np.isfinite(u_safe)):
                    raise RuntimeError(f"Safety filter returned invalid action: {u_safe}")

                # Check if filter modified action
                if np.linalg.norm(u_safe - u_desired) > 1e-4:
                    safety_filter_active = True

                action = u_safe

            except Exception as e:
                # Production fallback: If CBF-QP fails, log and use zero action (safe stop)
                self.logger.error(
                    f"Safety filter failed at timestep {self.timestep}: {e}. "
                    "Using emergency stop (zero action)."
                )
                safety_filter_error = str(e)
                action = np.zeros_like(u_desired)  # Emergency stop
                safety_filter_active = True  # Mark as activated (emergency mode)
        else:
            action = u_desired

        # Increment timestep
        self.timestep += 1

        # Prepare info dict
        info = {
            "belief_mean": belief_mean,
            "belief_ess": self.belief.ess(),
            "safety_filter_active": safety_filter_active,
            "slack": slack,
            "u_desired": u_desired,
            "u_safe": action,
            # Production monitoring
            "safety_filter_error": safety_filter_error,
            "timestep": self.timestep,
            # Task T055: Credal set info for US2
            "credal_set_active": self.belief.credal_set is not None,
            "credal_set_K": self.belief.credal_set.K if self.belief.credal_set else 0,
            # Task T063: Query action info for US3
            "query_triggered": query_triggered,
            "evi": evi_value,
            "entropy_before_query": entropy_before_query,
            "entropy_after_query": entropy_after_query,
        }

        return action, info

    def update_belief_with_message(self, message, source_trust=None) -> None:
        """
        Update belief with semantic message.

        Args:
            message: Message object with claim, source, value, A_c
            source_trust: Optional SourceTrust object. If None, uses config default.

        References:
            - theory.md §3: Message integration
            - Task T051: Credal set expansion for v=⊤
        """
        from ..core.messages import SourceTrust

        # Use provided source trust or create from config
        if source_trust is None:
            # Safely get trust_init from config with fallback chain
            r_s_init = 0.7  # Default fallback
            if hasattr(self.config, 'credal'):
                r_s_init = getattr(self.config.credal, 'trust_init', 0.7)
            source_trust = SourceTrust(r_s=r_s_init)

        # Apply message to belief
        self.belief.apply_message(message, source_trust)

        # Resample if needed after message update
        if self.belief.ess() < self.config.belief.resample_threshold * self.belief.n_particles:
            self.belief.resample()

    def _validate_config(self) -> None:
        """
        Validate configuration for production safety.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate belief parameters
        if not hasattr(self.config, 'belief'):
            raise ValueError("Config missing 'belief' section")

        if self.config.belief.particles < 100:
            raise ValueError(
                f"Too few particles: {self.config.belief.particles}. "
                "Minimum 100 for production."
            )

        if self.config.belief.particles > 100000:
            self.logger.warning(
                f"Very large particle count ({self.config.belief.particles}). "
                "May impact performance."
            )

        if not 0.1 <= self.config.belief.resample_threshold <= 0.9:
            raise ValueError(
                f"Invalid resample_threshold: {self.config.belief.resample_threshold}. "
                "Must be in [0.1, 0.9]."
            )

        # Validate environment parameters
        if not hasattr(self.config, 'env'):
            raise ValueError("Config missing 'env' section")

        if self.config.env.state_dim < 1:
            raise ValueError(f"Invalid state_dim: {self.config.env.state_dim}")

        if self.config.env.observation_noise <= 0:
            raise ValueError(
                f"Invalid observation_noise: {self.config.env.observation_noise}. "
                "Must be positive."
            )

        # Validate safety parameters if CBF enabled
        if hasattr(self.config, 'safety') and self.config.safety.cbf:
            if self.config.safety.barrier_alpha <= 0:
                raise ValueError(
                    f"Invalid barrier_alpha: {self.config.safety.barrier_alpha}. "
                    "Must be positive."
                )

            if self.config.safety.slack_penalty < 1.0:
                self.logger.warning(
                    f"Low slack_penalty ({self.config.safety.slack_penalty}). "
                    "Recommend >= 100 for hard constraints."
                )

        # Validate credal parameters (optional section)
        if hasattr(self.config, 'credal'):
            trust_init = getattr(self.config.credal, 'trust_init', 0.7)
            if not 0.0 < trust_init < 1.0:
                raise ValueError(
                    f"Invalid trust_init: {trust_init}. Must be in (0, 1)."
                )
            self.logger.debug(f"Credal settings: trust_init={trust_init}")

        # Validate query parameters if enabled
        if hasattr(self.config, 'query') and self.config.query.enabled:
            if self.config.query.cost < 0:
                raise ValueError(
                    f"Invalid query cost: {self.config.query.cost}. Must be >= 0."
                )

            if self.config.query.delta_star <= 0:
                raise ValueError(
                    f"Invalid delta_star: {self.config.query.delta_star}. "
                    "Must be positive."
                )

        self.logger.info("✓ Configuration validated successfully")

    def __repr__(self) -> str:
        return (
            f"Agent(particles={self.belief.n_particles}, "
            f"cbf={'enabled' if self.safety_filter else 'disabled'})"
        )
