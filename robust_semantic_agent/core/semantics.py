"""
Belnap 4-Valued Logic and Semantic Layer
Feature: 002-full-prototype
Tasks: T027, T048, T049

Implements:
- BelnapValue enum with 2-bit encoding
- Bilattice operations (truth and knowledge lattices)
- Status assignment from support/countersupport thresholds

References:
- docs/theory.md §2: Semantic layer specification
- exploration/004_belnap.py: Verified MWE
"""

from enum import IntEnum

import numpy as np


class BelnapValue(IntEnum):
    """
    Belnap 4-valued logic with 2-bit encoding.

    Encoding: (falsity_bit, truth_bit)
    - NEITHER (⊥): 0b00 - no information
    - TRUE (t):    0b01 - only supports truth
    - FALSE (f):   0b10 - only supports falsity
    - BOTH (⊤):    0b11 - contradiction

    Bilattice structure:
    - Truth order (≤_t): f ≤ ⊥ ≤ t and f ≤ ⊤ ≤ t
    - Knowledge order (≤_k): ⊥ ≤ t ≤ ⊤ and ⊥ ≤ f ≤ ⊤
    """

    NEITHER = 0b00  # ⊥
    TRUE = 0b01  # t
    FALSE = 0b10  # f
    BOTH = 0b11  # ⊤

    def __str__(self) -> str:
        symbols = {
            BelnapValue.NEITHER: "⊥",
            BelnapValue.TRUE: "t",
            BelnapValue.FALSE: "f",
            BelnapValue.BOTH: "⊤",
        }
        return symbols[self]


# Truth lattice operations (≤_t)


def and_t(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """
    Truth-preserving conjunction (∧).
    min on truth, max on falsity.
    """
    t_bit = min(x & 0b01, y & 0b01)
    f_bit = max((x & 0b10) >> 1, (y & 0b10) >> 1)
    return BelnapValue((f_bit << 1) | t_bit)


def or_t(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """
    Truth-preserving disjunction (∨).
    max on truth, min on falsity.
    """
    t_bit = max(x & 0b01, y & 0b01)
    f_bit = min((x & 0b10) >> 1, (y & 0b10) >> 1)
    return BelnapValue((f_bit << 1) | t_bit)


def not_t(x: BelnapValue) -> BelnapValue:
    """
    Negation (¬): swap truth and falsity bits.
    Involution: ¬¬x = x
    """
    return BelnapValue(((x & 0b01) << 1) | ((x & 0b10) >> 1))


# Knowledge lattice operations (≤_k)


def consensus(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """
    Consensus (⊗): bitwise AND.
    Agree only on shared information.
    """
    return BelnapValue(x & y)


def gullibility(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """
    Gullibility (⊕): bitwise OR.
    Accept all information (may lead to contradiction).
    """
    return BelnapValue(x | y)


# Status assignment for RSA semantic layer


def status(s_c: float, s_bar_c: float, tau: float = 0.68, tau_prime: float = 0.32) -> BelnapValue:
    """
    Assign Belnap status based on support/countersupport thresholds.

    Args:
        s_c: Support for claim c ∈ [0, 1]
        s_bar_c: Countersupport (support for ¬c) ∈ [0, 1]
        tau: High threshold (> 0.5) for strong evidence
        tau_prime: Low threshold (< 0.5) for weak evidence

    Returns:
        BelnapValue status:
        - TRUE: High support, low countersupport
        - FALSE: Low support, high countersupport
        - BOTH: High support AND high countersupport (contradiction → credal set)
        - NEITHER: Insufficient evidence

    References:
        - docs/theory.md §2.3: Status assignment
        - FR-004: Belnap status specification
    """
    if s_c >= tau and s_bar_c < tau_prime:
        return BelnapValue.TRUE
    elif s_bar_c >= tau and s_c < tau_prime:
        return BelnapValue.FALSE
    elif s_c >= tau and s_bar_c >= tau:
        return BelnapValue.BOTH  # Contradiction → triggers credal set expansion
    else:
        return BelnapValue.NEITHER  # Insufficient evidence


def calibrate_thresholds(
    episodes: list, cost_matrix: np.ndarray = None, target_ece: float = 0.05
) -> tuple:
    """
    Auto-calibrate thresholds τ and τ' to minimize ECE with cost penalties.

    Task T069: Threshold calibration for semantic status assignment.

    Algorithm:
    1. Grid search over (τ, τ') space
    2. For each pair, compute status predictions
    3. Evaluate ECE + cost-weighted error
    4. Select thresholds that minimize objective

    Args:
        episodes: List of dicts with keys:
                  - 's_c': support score
                  - 's_bar_c': countersupport score
                  - 'ground_truth': actual outcome (0 or 1)
        cost_matrix: 2x2 cost matrix [[TN, FP], [FN, TP]]
                     Default: [[0, 1], [1, 0]] (balanced)
        target_ece: Target ECE threshold (SC-008: 0.05)

    Returns:
        tau_opt: Optimal τ threshold (for TRUE)
        tau_prime_opt: Optimal τ' threshold (for FALSE)
        ece_before: ECE before calibration (using default τ=0.7, τ'=0.3)
        ece_after: ECE after calibration

    References:
        - docs/theory.md §6: Calibration
        - SC-008: ECE ≤ 0.05

    Example:
        >>> episodes = [
        ...     {'s_c': 0.8, 's_bar_c': 0.2, 'ground_truth': 1},
        ...     {'s_c': 0.3, 's_bar_c': 0.7, 'ground_truth': 0},
        ... ]
        >>> tau, tau_prime, ece_before, ece_after = calibrate_thresholds(episodes)
    """
    from ..reports.calibration import compute_ece

    if cost_matrix is None:
        cost_matrix = np.array([[0, 1], [1, 0]])  # Balanced: FP=FN=1

    # Extract data
    s_c = np.array([ep["s_c"] for ep in episodes])
    s_bar_c = np.array([ep["s_bar_c"] for ep in episodes])
    ground_truth = np.array([ep["ground_truth"] for ep in episodes])

    # Compute ECE before calibration (default thresholds)
    tau_default = 0.7
    tau_prime_default = 0.3

    predictions_before = []
    for i in range(len(episodes)):
        v = status(s_c[i], s_bar_c[i], tau_default, tau_prime_default)
        # Convert status to probability
        if v == BelnapValue.TRUE:
            pred = 0.9
        elif v == BelnapValue.FALSE:
            pred = 0.1
        elif v == BelnapValue.NEITHER:
            pred = 0.5
        else:  # BOTH
            pred = 0.5
        predictions_before.append(pred)

    ece_before = compute_ece(np.array(predictions_before), ground_truth, n_bins=10)

    # Grid search for optimal thresholds
    tau_candidates = np.linspace(0.55, 0.95, 20)  # τ > 0.5
    tau_prime_candidates = np.linspace(0.05, 0.45, 20)  # τ' < 0.5

    best_ece = float("inf")
    best_cost = float("inf")
    tau_opt = tau_default
    tau_prime_opt = tau_prime_default

    for tau in tau_candidates:
        for tau_prime in tau_prime_candidates:
            # Ensure τ' < 0.5 < τ
            if tau_prime >= 0.5 or tau <= 0.5:
                continue

            # Compute predictions with these thresholds
            predictions = []
            for i in range(len(episodes)):
                v = status(s_c[i], s_bar_c[i], tau, tau_prime)
                # Convert to probability
                if v == BelnapValue.TRUE:
                    pred = 0.9
                elif v == BelnapValue.FALSE:
                    pred = 0.1
                elif v == BelnapValue.NEITHER:
                    pred = 0.5
                else:  # BOTH
                    pred = 0.5
                predictions.append(pred)

            predictions = np.array(predictions)

            # Compute ECE
            ece = compute_ece(predictions, ground_truth, n_bins=10)

            # Compute cost-weighted error
            binary_predictions = (predictions > 0.5).astype(int)
            confusion = np.zeros((2, 2))
            for pred, truth in zip(binary_predictions, ground_truth, strict=False):
                confusion[int(truth), int(pred)] += 1

            # Total cost: FP_cost * FP_count + FN_cost * FN_count
            fp_count = confusion[0, 1]  # Ground truth = 0, predicted = 1
            fn_count = confusion[1, 0]  # Ground truth = 1, predicted = 0
            total_cost = cost_matrix[0, 1] * fp_count + cost_matrix[1, 0] * fn_count

            # Objective: ECE + normalized cost
            objective = ece + 0.1 * (total_cost / len(episodes))

            # Update best
            if objective < best_cost:
                best_cost = objective
                best_ece = ece
                tau_opt = tau
                tau_prime_opt = tau_prime

    ece_after = best_ece

    return tau_opt, tau_prime_opt, ece_before, ece_after
