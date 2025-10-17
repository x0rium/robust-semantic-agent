"""
Control Barrier Function (CBF) Safety Filter
Feature: 002-full-prototype
Task: T035

Implements QP-based safety filter using cvxpy + OSQP solver.

Minimizes deviation from desired control while enforcing CBF constraint:
    minimize    ||u - u_des||²
    subject to  ∇h(x)·u ≥ -α·h(x) - slack

References:
- docs/theory.md §4.3: CBF-QP formulation
- exploration/003_qp_solver.py: Verified MWE
- FR-007: Supermartingale constraint
"""

import logging

import cvxpy as cp
import numpy as np


class SafetyFilter:
    """
    CBF-QP safety filter using OSQP solver.

    Solves quadratic program to find safe control input.

    Attributes:
        barrier_fn: BarrierFunction object
        alpha: CBF class-K parameter (> 0)
        slack_penalty: Penalty for slack variable (infeasibility handling)
        max_iter: Maximum QP solver iterations
        verbose: Solver verbosity flag

    Methods:
        filter(x, u_desired): Project desired control onto safe set

    References:
        - SC-001: Zero violations requirement
        - SC-002: ≥1% filter activations
    """

    def __init__(
        self,
        barrier_fn,
        alpha: float = 0.5,
        slack_penalty: float = 1000.0,
        max_iter: int = 200,
        verbose: bool = False,
    ):
        """
        Initialize CBF-QP safety filter.

        Args:
            barrier_fn: BarrierFunction instance
            alpha: Class-K parameter (controls conservativeness)
            slack_penalty: Penalty weight for constraint relaxation
            max_iter: OSQP max iterations
            verbose: Print solver output
        """
        self.barrier_fn = barrier_fn
        self.alpha = alpha
        self.slack_penalty = slack_penalty
        self.max_iter = max_iter
        self.verbose = verbose

        self.logger = logging.getLogger(__name__)

    def filter(self, x: np.ndarray, u_desired: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Project desired control onto safe set via QP.

        minimize    ||u - u_desired||²  + penalty × slack
        subject to  ∇h(x)·u ≥ -α·h(x) - slack
                    slack ≥ 0

        Args:
            x: Current state (2,)
            u_desired: Nominal control input (2,)

        Returns:
            u_safe: Safe control input (2,)
            slack: Slack variable value (0 if feasible)

        References:
            - docs/verified-apis.md: OSQP solver configuration
            - Task T025: QP solve tests
        """
        m = len(u_desired)

        # Decision variables
        u = cp.Variable(m)
        slack = cp.Variable(nonneg=True)

        # Objective: minimize deviation + penalize slack
        objective = cp.Minimize(cp.sum_squares(u - u_desired) + self.slack_penalty * slack)

        # CBF constraint
        h_x = self.barrier_fn.evaluate(x)
        dh_dx = self.barrier_fn.gradient(x)

        # For 2D integrator: Lfh = 0, Lgh = ∇h
        # Constraint: ḣ = ∇h(x)·u ≥ -α·h(x) - slack
        # This ensures h(x) remains non-negative (safe set)
        constraints = [dh_dx @ u >= -self.alpha * h_x - slack]

        # Solve QP
        prob = cp.Problem(objective, constraints)

        try:
            prob.solve(
                solver=cp.OSQP,
                warm_start=True,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                self.logger.warning(f"QP solver status: {prob.status}")

            u_safe = u.value
            slack_value = slack.value

            # Log if slack was used
            if slack_value > 1e-5:
                self.logger.info(f"CBF relaxed by slack={slack_value:.6f} at state {x}")

            return u_safe, slack_value

        except Exception as e:
            self.logger.error(f"QP solve failed: {e}")
            # Fallback: return zero control
            return np.zeros(m), float("inf")

    def __repr__(self) -> str:
        return f"SafetyFilter(α={self.alpha}, slack_penalty={self.slack_penalty})"
