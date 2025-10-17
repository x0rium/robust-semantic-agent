"""
Policy Planning Module
Feature: 002-full-prototype
Task: T037

Simple greedy policy for MVP navigation.

References:
- docs/theory.md §4: Risk-aware policy
- FR-006: CVaR-based decision making
"""

import numpy as np


class Policy:
    """
    Simple greedy policy for navigation.

    For MVP: Navigate toward goal with proportional control.
    Future: Value iteration, Perseus-PBVI, or actor-critic.

    Attributes:
        goal: Goal position (2,)
        gain: Proportional control gain

    Methods:
        select_action(belief): Choose control based on belief

    References:
        - Task T037: Simple policy implementation
        - Task T090: Advanced policy training (optional)
    """

    def __init__(self, goal: np.ndarray, gain: float = 1.0):
        """
        Initialize policy.

        Args:
            goal: Goal position (2,)
            gain: Proportional gain for control
        """
        self.goal = np.array(goal)
        self.gain = gain

    def select_action(self, belief) -> np.ndarray:
        """
        Select action based on current belief.

        Simple strategy: Navigate toward goal from belief mean.

        Task T053: Handles both Belief and CredalSet.
        If credal set exists (v=⊤ message received), uses conservative
        lower expectation for robust decision making.

        Args:
            belief: Belief object (may contain credal_set attribute)
                   OR CredalSet object directly

        Returns:
            action: Control input (2,)

        Future:
            - Risk-aware planning with CVaR
            - Value iteration in belief space
            - Actor-critic with CVaR objective
        """
        # Check if belief has credal set (from v=⊤ message)
        if hasattr(belief, "credal_set") and belief.credal_set is not None:
            # Use credal set's conservative mean (lower expectation)
            state_estimate = belief.credal_set.mean()
        else:
            # Standard belief or CredalSet directly
            state_estimate = belief.mean()

        # Proportional control toward goal
        direction = self.goal - state_estimate
        distance = np.linalg.norm(direction)

        if distance < 1e-6:
            return np.zeros(2)

        # Normalize and scale
        action = self.gain * direction / distance

        return action

    def __repr__(self) -> str:
        return f"Policy(goal={self.goal}, gain={self.gain})"
