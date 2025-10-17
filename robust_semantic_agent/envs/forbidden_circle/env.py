"""
Forbidden Circle Navigation Environment
Feature: 002-full-prototype
Task: T036

2D navigation with circular forbidden zone and noisy beacon observations.

Dynamics: ẋ = u (2D integrator)
Observations: Noisy beacons
Safe set: S = {x: ||x - center|| ≥ radius}

References:
- docs/theory.md §1.1: Dynamics and observations
- FR-001: Observation kernel G
"""

import logging

import numpy as np


class ForbiddenCircleEnv:
    """
    2D navigation environment with forbidden circular zone.

    State space: R²
    Action space: R² (velocity commands)
    Observation: Noisy position (Gaussian)

    Attributes:
        obstacle_radius: Forbidden zone radius
        obstacle_center: Forbidden zone center
        goal_region: Goal position
        goal_radius: Goal tolerance
        obs_noise: Observation noise std dev
        max_action: Action clipping limit
        dt: Timestep duration

    Methods:
        reset(): Initialize episode
        step(action): Execute action, return (obs, reward, done, info)

    References:
        - SC-001: Zero violations requirement
        - Task T042: Integration test
    """

    def __init__(self, config=None):
        """
        Initialize environment from configuration.

        Args:
            config: Configuration object (or None for defaults)
        """
        if config is None:
            # Default configuration
            self.obstacle_radius = 0.3
            self.obstacle_center = np.array([0.0, 0.0])
            self.goal_region = np.array([0.8, 0.8])
            self.goal_radius = 0.1
            self.obs_noise = 0.1
            self.max_action = 0.15
            self.dt = 0.1
        else:
            env_cfg = config.env
            self.obstacle_radius = env_cfg.obstacle_radius
            self.obstacle_center = np.array(env_cfg.obstacle_center)
            self.goal_region = np.array(env_cfg.goal_region)
            self.goal_radius = env_cfg.goal_radius
            self.obs_noise = env_cfg.observation_noise
            self.max_action = env_cfg.max_action
            self.dt = 0.1

        # State
        self.state = None
        self.timestep = 0
        self.max_timesteps = 50

        # Task T054: Gossip source for contradiction testing
        self.enable_gossip_source = False  # Enable for US2 testing
        self.gossip_messages = []

        self.logger = logging.getLogger(__name__)

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation (noisy position)
        """
        # Random initial state (far from obstacle and goal)
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.5, 1.0)
        self.state = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        # Ensure not in obstacle
        while self._is_in_obstacle(self.state):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0.5, 1.0)
            self.state = np.array([radius * np.cos(angle), radius * np.sin(angle)])

        self.timestep = 0

        # Return noisy observation
        return self._get_observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute action and return transition.

        Args:
            action: Control input (2,)

        Returns:
            observation: Noisy state observation
            reward: Step reward
            done: Episode termination flag
            info: Additional information dict
        """
        # Clip action
        action = np.clip(action, -self.max_action, self.max_action)

        # Dynamics: x+ = x + u*dt
        self.state = self.state + action * self.dt

        # Increment timestep
        self.timestep += 1

        # Compute reward
        reward = self._compute_reward()

        # Check termination
        done = False
        goal_reached = False
        violated_safety = False

        # Check goal
        if self._is_at_goal(self.state):
            done = True
            goal_reached = True
            reward += 10.0  # Goal bonus

        # Check obstacle violation
        if self._is_in_obstacle(self.state):
            violated_safety = True
            reward -= 10.0  # Penalty

        # Check timeout
        if self.timestep >= self.max_timesteps:
            done = True

        # Get observation
        obs = self._get_observation()

        # Info dict
        info = {
            "true_state": self.state.copy(),
            "goal_reached": goal_reached,
            "violated_safety": violated_safety,
            "timestep": self.timestep,
        }

        return obs, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Generate noisy observation of current state."""
        noise = np.random.randn(2) * self.obs_noise
        return self.state + noise

    def _compute_reward(self) -> float:
        """Compute step reward (distance to goal)."""
        dist_to_goal = np.linalg.norm(self.state - self.goal_region)
        return -dist_to_goal  # Negative distance (minimize)

    def _is_at_goal(self, state: np.ndarray) -> bool:
        """Check if state is within goal region."""
        dist = np.linalg.norm(state - self.goal_region)
        return dist <= self.goal_radius

    def _is_in_obstacle(self, state: np.ndarray) -> bool:
        """Check if state is inside forbidden zone."""
        dist = np.linalg.norm(state - self.obstacle_center)
        return dist < self.obstacle_radius

    def get_messages(self):
        """
        Get exogenous messages from gossip source.

        Task T054: Gossip source emitting v=⊤ (contradiction) messages.

        Returns:
            List of Message objects

        Example gossip messages:
        - "Agent is north of center" (v=TRUE)
        - "Agent is south of center" (v=FALSE)
        - Combined → v=BOTH (contradiction)
        """
        from ...core.messages import Message
        from ...core.semantics import BelnapValue

        if not self.enable_gossip_source:
            return []

        messages = []

        # Emit contradictory message with some probability
        if np.random.rand() < 0.1:  # 10% chance per step
            # Create contradictory message about location
            # Claim: "Agent is in northern half"
            message = Message(
                claim="location_north",
                source="gossip",
                value=BelnapValue.BOTH,  # ⊤ = contradiction
                A_c=lambda x: x[:, 1] > 0.0,  # y > 0
            )
            messages.append(message)

        return messages

    def __repr__(self) -> str:
        return (
            f"ForbiddenCircleEnv(obstacle_r={self.obstacle_radius}, "
            f"goal={self.goal_region}, obs_noise={self.obs_noise})"
        )
