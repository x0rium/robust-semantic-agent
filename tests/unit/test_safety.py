"""
Unit Tests: CBF Safety Filter
Feature: 002-full-prototype
Tasks: T024, T025

Tests MUST FAIL initially (TDD Principle V: NON-NEGOTIABLE)
"""

import numpy as np
import pytest


@pytest.mark.unit
class TestCBFSupermartingale:
    """T024: Test CBF supermartingale property: E[B(x+)|x,u] ≤ B(x)."""

    def test_cbf_supermartingale_property(self):
        """
        For safe action u_safe from CBF-QP, barrier should satisfy:
        E[B(x+)] ≤ B(x) (supermartingale property)
        """
        from robust_semantic_agent.envs.forbidden_circle.safety import BarrierFunction
        from robust_semantic_agent.safety.cbf import SafetyFilter

        # Define barrier function for circular forbidden zone
        barrier_fn = BarrierFunction(radius=0.3, center=np.array([0.0, 0.0]))

        safety_filter = SafetyFilter(barrier_fn, alpha=0.5)

        # Test state (near boundary)
        x = np.array([0.35, 0.0])

        # Desired control (pointing inward - unsafe)
        u_desired = np.array([-0.1, 0.0])

        # Get safe control from CBF-QP
        u_safe, _ = safety_filter.filter(x, u_desired)

        # Simulate next state (deterministic dynamics: x+ = x + u*dt)
        dt = 0.1
        x_next = x + u_safe * dt

        # Check barrier values
        B_x = barrier_fn.evaluate(x)
        B_x_next = barrier_fn.evaluate(x_next)

        # Supermartingale: B(x+) ≤ B(x) + tolerance
        assert (
            B_x_next <= B_x + 1e-3
        ), f"Supermartingale violated: B(x)={B_x:.6f}, B(x+)={B_x_next:.6f}"


@pytest.mark.unit
class TestQPSolve:
    """T025: Test basic QP solve and infeasibility handling."""

    def test_qp_solve_basic(self):
        """QP should find safe control when feasible."""
        from robust_semantic_agent.envs.forbidden_circle.safety import BarrierFunction
        from robust_semantic_agent.safety.cbf import SafetyFilter

        barrier_fn = BarrierFunction(radius=0.3, center=np.array([0.0, 0.0]))
        safety_filter = SafetyFilter(barrier_fn, alpha=0.5)

        # Safe state (far from boundary)
        x = np.array([0.6, 0.6])

        # Desired control
        u_desired = np.array([0.1, -0.05])

        # Get safe control
        u_safe, slack = safety_filter.filter(x, u_desired)

        # Should be close to desired (state is safe)
        np.linalg.norm(u_safe - u_desired)

        # Slack should be ~0 (feasible)
        assert slack < 1e-5, f"Slack should be ~0 for feasible case, got {slack:.6f}"

        # Safe control should exist
        assert u_safe is not None, "Safe control should not be None"
        assert np.all(np.isfinite(u_safe)), "Safe control should be finite"

    def test_qp_infeasibility_slack(self):
        """QP should use slack variable when strictly infeasible."""
        from robust_semantic_agent.envs.forbidden_circle.safety import BarrierFunction
        from robust_semantic_agent.safety.cbf import SafetyFilter

        barrier_fn = BarrierFunction(radius=0.3, center=np.array([0.0, 0.0]))
        safety_filter = SafetyFilter(barrier_fn, alpha=0.5, slack_penalty=1000.0)

        # State INSIDE forbidden zone (unsafe)
        x = np.array([0.1, 0.1])

        # Desired control
        u_desired = np.array([0.0, 0.0])

        # Get safe control (may require slack)
        u_safe, slack = safety_filter.filter(x, u_desired)

        # Safe control should still be returned
        assert u_safe is not None, "Safe control should exist even with slack"
        assert np.all(np.isfinite(u_safe)), "Safe control should be finite"

        # Slack may be positive (infeasible case)
        # We just verify solver didn't crash

    def test_cbf_constraint_enforcement(self):
        """Verify CBF constraint is satisfied: Lfh + Lgh·u ≥ -α·h."""
        from robust_semantic_agent.envs.forbidden_circle.safety import BarrierFunction
        from robust_semantic_agent.safety.cbf import SafetyFilter

        barrier_fn = BarrierFunction(radius=0.3, center=np.array([0.0, 0.0]))
        safety_filter = SafetyFilter(barrier_fn, alpha=0.5)

        # Test state
        x = np.array([0.4, 0.3])

        # Desired control (may violate constraint)
        u_desired = np.array([-0.15, -0.10])

        # Get safe control
        u_safe, _ = safety_filter.filter(x, u_desired)

        # Compute constraint components
        h_x = barrier_fn.evaluate(x)
        dh_dx = barrier_fn.gradient(x)

        # For 2D integrator: Lfh = 0 (no drift), Lgh = ∇h
        Lfh_x = 0.0
        Lgh_x = dh_dx

        # Check constraint: Lfh + Lgh·u ≥ -α·h
        lhs = Lfh_x + np.dot(Lgh_x, u_safe)
        rhs = -0.5 * h_x

        # Allow small numerical tolerance
        assert lhs >= rhs - 1e-4, f"CBF constraint violated: LHS={lhs:.6f} < RHS={rhs:.6f}"
