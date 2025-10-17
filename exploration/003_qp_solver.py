"""
Minimal Working Example: cvxpy QP Solver for CBF Safety Filter
Feature: 002-full-prototype
Task: T009

Tests:
1. Basic QP formulation and solve
2. CBF constraint: Lfh(x) + Lgh(x)·u ≥ -α·h(x)
3. Infeasibility handling with slack variable
4. Warm-start performance comparison
5. OSQP solver performance (target: 1-10ms)
"""

import numpy as np
import cvxpy as cp
import time


def cbf_safety_filter(x, u_desired, barrier_fn, barrier_grad, alpha=0.5, slack_penalty=1000.0):
    """
    CBF-QP safety filter: min ||u - u_desired||² s.t. CBF constraint.

    Args:
        x: Current state (2D position)
        u_desired: Nominal control input
        barrier_fn: Barrier function h(x) (safe if h(x) ≤ 0)
        barrier_grad: Gradient ∇h(x)
        alpha: Class-K function parameter
        slack_penalty: Penalty for constraint relaxation

    Returns:
        u_safe: Safe control input
        slack_value: Constraint relaxation (0 if feasible)
    """
    m = len(u_desired)

    # Decision variables
    u = cp.Variable(m)
    slack = cp.Variable(nonneg=True)

    # Objective: minimize deviation from desired control + slack penalty
    objective = cp.Minimize(cp.sum_squares(u - u_desired) + slack_penalty * slack)

    # CBF constraint: Lfh(x) + Lgh(x)·u ≥ -α·h(x) - slack
    # For simplicity, assume Lfh = 0 (no drift in h)
    # Lgh(x) = ∇h(x)·G(x) where G(x) is control matrix
    # For 2D integrator: ẋ = u, so Lgh(x) = ∇h(x)

    h_x = barrier_fn(x)
    dh_dx = barrier_grad(x)

    # CBF constraint with slack
    constraints = [dh_dx @ u >= -alpha * h_x - slack]

    # Formulate and solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.OSQP, warm_start=False, verbose=False)

    if prob.status != cp.OPTIMAL:
        print(f"  ⚠ Warning: QP status = {prob.status}")

    return u.value, slack.value


def main():
    print("=" * 60)
    print("cvxpy QP Solver MWE: CBF Safety Filter")
    print("=" * 60)

    # Define barrier function for circular forbidden zone
    # Safe set: S = {x: ||x|| ≥ r} (outside circle of radius r)
    # Barrier: h(x) = r² - ||x||² (safe if h ≤ 0)

    r = 0.3  # Forbidden zone radius

    def barrier_fn(x):
        """h(x) = r² - ||x||²"""
        return r**2 - np.dot(x, x)

    def barrier_grad(x):
        """∇h(x) = -2x"""
        return -2 * x

    # Test 1: Basic QP solve (safe state)
    print("\n" + "-" * 60)
    print("Test 1: Basic QP Solve (Safe State)")

    x_safe = np.array([0.5, 0.5])  # Outside forbidden zone
    u_desired = np.array([0.1, -0.05])

    h_x = barrier_fn(x_safe)
    print(f"  State: {x_safe}")
    print(f"  Barrier value: h(x) = {h_x:.6f} (safe if ≤ 0)")
    print(f"  Desired control: {u_desired}")

    u_safe, slack = cbf_safety_filter(x_safe, u_desired, barrier_fn, barrier_grad)

    print(f"  Safe control: {u_safe}")
    print(f"  Slack: {slack:.6f}")
    print(f"  Deviation: ||u_safe - u_desired|| = {np.linalg.norm(u_safe - u_desired):.6f}")

    if slack < 1e-6:
        print(f"  ✓ PASS: Feasible solution (slack ≈ 0)")
    else:
        print(f"  ⚠ Warning: Constraint relaxed (slack = {slack:.6f})")

    # Test 2: CBF constraint enforcement (unsafe desired control)
    print("\n" + "-" * 60)
    print("Test 2: CBF Constraint Enforcement")

    x_near_boundary = np.array([0.35, 0.0])  # Close to forbidden zone
    u_inward = np.array([-0.15, 0.0])  # Points toward forbidden zone

    h_x = barrier_fn(x_near_boundary)
    print(f"  State: {x_near_boundary}")
    print(f"  Barrier value: h(x) = {h_x:.6f}")
    print(f"  Desired control (inward): {u_inward}")

    u_safe, slack = cbf_safety_filter(x_near_boundary, u_inward, barrier_fn, barrier_grad, alpha=0.5)

    print(f"  Safe control: {u_safe}")
    print(f"  Slack: {slack:.6f}")

    # Verify CBF constraint
    dh_dx = barrier_grad(x_near_boundary)
    lhs = np.dot(dh_dx, u_safe)
    rhs = -0.5 * h_x

    print(f"  Constraint check:")
    print(f"    LHS: ∇h·u = {lhs:.6f}")
    print(f"    RHS: -α·h(x) = {rhs:.6f}")
    print(f"    Satisfied: {lhs:.6f} ≥ {rhs:.6f} ? {lhs >= rhs - 1e-6}")

    if lhs >= rhs - 1e-6:
        print(f"  ✓ PASS: CBF constraint satisfied")
    else:
        print(f"  ✗ FAIL: CBF constraint violated")

    # Test 3: Infeasibility handling
    print("\n" + "-" * 60)
    print("Test 3: Infeasibility Handling (Slack Variable)")

    x_inside = np.array([0.1, 0.1])  # Inside forbidden zone
    u_desired = np.array([0.0, 0.0])

    h_x = barrier_fn(x_inside)
    print(f"  State (inside forbidden zone): {x_inside}")
    print(f"  Barrier value: h(x) = {h_x:.6f} (UNSAFE: h > 0)")
    print(f"  Desired control: {u_desired}")

    u_safe, slack = cbf_safety_filter(x_inside, u_desired, barrier_fn, barrier_grad, alpha=0.5, slack_penalty=1000.0)

    print(f"  Safe control: {u_safe}")
    print(f"  Slack: {slack:.6f}")

    if slack > 1e-6:
        print(f"  ✓ Infeasibility detected, slack activated ({slack:.6f} > 0)")
    else:
        print(f"  Feasible solution found")

    # Test 4: Warm-start performance
    print("\n" + "-" * 60)
    print("Test 4: Warm-Start Performance")

    x_test = np.array([0.4, 0.3])
    u_test = np.array([0.1, 0.1])

    # Cold start
    start = time.perf_counter()
    for _ in range(100):
        u = cp.Variable(2)
        slack = cp.Variable(nonneg=True)
        h_x = barrier_fn(x_test)
        dh_dx = barrier_grad(x_test)
        objective = cp.Minimize(cp.sum_squares(u - u_test) + 1000.0 * slack)
        constraints = [dh_dx @ u >= -0.5 * h_x - slack]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, warm_start=False, verbose=False)
    elapsed_cold = (time.perf_counter() - start) / 100 * 1000

    # Warm start
    u = cp.Variable(2)
    slack = cp.Variable(nonneg=True)
    h_x = barrier_fn(x_test)
    dh_dx = barrier_grad(x_test)
    objective = cp.Minimize(cp.sum_squares(u - u_test) + 1000.0 * slack)
    constraints = [dh_dx @ u >= -0.5 * h_x - slack]
    prob = cp.Problem(objective, constraints)

    start = time.perf_counter()
    for _ in range(100):
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
    elapsed_warm = (time.perf_counter() - start) / 100 * 1000

    print(f"  Cold start: {elapsed_cold:.3f} ms")
    print(f"  Warm start: {elapsed_warm:.3f} ms")
    print(f"  Speedup: {elapsed_cold / elapsed_warm:.2f}x")

    if elapsed_warm < 10.0:
        print(f"  ✓ PASS: Warm-start time < 10ms target")
    else:
        print(f"  ⚠ Warning: Slower than target ({elapsed_warm:.3f} ms > 10 ms)")

    # Test 5: Overall performance benchmark
    print("\n" + "-" * 60)
    print("Test 5: Performance Benchmark (Target: 1-10ms)")

    n_trials = 1000
    times = []

    for _ in range(n_trials):
        x_random = np.random.randn(2)
        u_random = np.random.randn(2) * 0.1

        start = time.perf_counter()
        cbf_safety_filter(x_random, u_random, barrier_fn, barrier_grad)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times = np.array(times)
    print(f"  Trials: {n_trials}")
    print(f"  Mean: {np.mean(times):.3f} ms")
    print(f"  Median: {np.median(times):.3f} ms")
    print(f"  Std: {np.std(times):.3f} ms")
    print(f"  Min: {np.min(times):.3f} ms")
    print(f"  Max: {np.max(times):.3f} ms")
    print(f"  95th percentile: {np.percentile(times, 95):.3f} ms")

    if np.percentile(times, 95) < 10.0:
        print(f"  ✓ PASS: 95th percentile < 10ms")
    else:
        print(f"  ⚠ Warning: Some solves exceed 10ms target")

    print("\n" + "=" * 60)
    print("cvxpy QP Solver MWE: All tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
