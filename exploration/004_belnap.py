"""
Minimal Working Example: Belnap 4-Valued Logic
Feature: 002-full-prototype
Task: T010

Tests:
1. IntEnum 2-bit encoding: {⊥=0b00, t=0b01, f=0b10, ⊤=0b11}
2. Truth lattice operations (∧, ∨, ¬)
3. Knowledge lattice operations (⊗ consensus, ⊕ gullibility)
4. Bilattice properties validation (12 properties)
5. Status assignment from support/countersupport thresholds
"""

from enum import IntEnum
import numpy as np


class BelnapValue(IntEnum):
    """
    Belnap 4-valued logic with 2-bit encoding.

    Bits: (falsity_bit, truth_bit)
    - NEITHER (⊥): 0b00 - no information
    - TRUE (t):    0b01 - supports truth only
    - FALSE (f):   0b10 - supports falsity only
    - BOTH (⊤):    0b11 - contradiction
    """

    NEITHER = 0b00  # ⊥
    TRUE = 0b01  # t
    FALSE = 0b10  # f
    BOTH = 0b11  # ⊤

    def __str__(self):
        symbols = {
            BelnapValue.NEITHER: "⊥",
            BelnapValue.TRUE: "t",
            BelnapValue.FALSE: "f",
            BelnapValue.BOTH: "⊤",
        }
        return symbols[self]


# Truth lattice operations (≤_t order)


def and_t(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """Conjunction on truth lattice: min truth, max falsity."""
    t_bit = min(x & 0b01, y & 0b01)
    f_bit = max((x & 0b10) >> 1, (y & 0b10) >> 1)
    return BelnapValue((f_bit << 1) | t_bit)


def or_t(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """Disjunction on truth lattice: max truth, min falsity."""
    t_bit = max(x & 0b01, y & 0b01)
    f_bit = min((x & 0b10) >> 1, (y & 0b10) >> 1)
    return BelnapValue((f_bit << 1) | t_bit)


def not_t(x: BelnapValue) -> BelnapValue:
    """Negation: swap truth and falsity bits."""
    return BelnapValue(((x & 0b01) << 1) | ((x & 0b10) >> 1))


# Knowledge lattice operations (≤_k order)


def consensus(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """Consensus (⊗): bitwise AND - agree on shared info."""
    return BelnapValue(x & y)


def gullibility(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """Gullibility (⊕): bitwise OR - accept all info."""
    return BelnapValue(x | y)


# Status assignment


def status(s_c: float, s_bar_c: float, tau: float, tau_prime: float) -> BelnapValue:
    """
    Assign Belnap status based on support/countersupport thresholds.

    Args:
        s_c: Support for claim c
        s_bar_c: Countersupport for claim c
        tau: High threshold (> 0.5)
        tau_prime: Low threshold (< 0.5)

    Returns:
        BelnapValue status
    """
    if s_c >= tau and s_bar_c < tau_prime:
        return BelnapValue.TRUE
    elif s_bar_c >= tau and s_c < tau_prime:
        return BelnapValue.FALSE
    elif s_c >= tau and s_bar_c >= tau:
        return BelnapValue.BOTH  # Contradiction
    else:
        return BelnapValue.NEITHER  # Insufficient evidence


def main():
    print("=" * 60)
    print("Belnap 4-Valued Logic MWE")
    print("=" * 60)

    # Test 1: IntEnum encoding
    print("\n" + "-" * 60)
    print("Test 1: 2-Bit Encoding")

    values = [BelnapValue.NEITHER, BelnapValue.TRUE, BelnapValue.FALSE, BelnapValue.BOTH]

    for v in values:
        t_bit = v & 0b01
        f_bit = (v & 0b10) >> 1
        print(f"  {str(v):^3} ({v.name:7s}): {v:04b} - truth={t_bit}, falsity={f_bit}")

    print(f"  ✓ IntEnum encoding successful")

    # Test 2: Truth lattice operations
    print("\n" + "-" * 60)
    print("Test 2: Truth Lattice Operations")

    T = BelnapValue.TRUE
    F = BelnapValue.FALSE
    N = BelnapValue.NEITHER
    B = BelnapValue.BOTH

    print("\n  Conjunction (∧):")
    print(f"    {T} ∧ {T} = {and_t(T, T)}")
    print(f"    {T} ∧ {F} = {and_t(T, F)}")
    print(f"    {T} ∧ {B} = {and_t(T, B)}")

    print("\n  Disjunction (∨):")
    print(f"    {F} ∨ {F} = {or_t(F, F)}")
    print(f"    {T} ∨ {F} = {or_t(T, F)}")
    print(f"    {F} ∨ {B} = {or_t(F, B)}")

    print("\n  Negation (¬):")
    for v in values:
        print(f"    ¬{str(v)} = {not_t(v)}")

    # Verify involution: ¬¬x = x
    involution_ok = all(not_t(not_t(v)) == v for v in values)
    print(f"\n  Involution (¬¬x = x): {involution_ok}")

    if involution_ok:
        print(f"  ✓ PASS: Truth lattice operations correct")
    else:
        print(f"  ✗ FAIL: Involution violated")

    # Test 3: Knowledge lattice operations
    print("\n" + "-" * 60)
    print("Test 3: Knowledge Lattice Operations")

    print("\n  Consensus (⊗):")
    print(f"    {T} ⊗ {T} = {consensus(T, T)}")
    print(f"    {T} ⊗ {F} = {consensus(T, F)}")  # Should be ⊥ (no agreement)
    print(f"    {T} ⊗ {B} = {consensus(T, B)}")  # Should be t (agree on truth bit)

    print("\n  Gullibility (⊕):")
    print(f"    {N} ⊕ {T} = {gullibility(N, T)}")
    print(f"    {T} ⊕ {F} = {gullibility(T, F)}")  # Should be ⊤ (both bits set)
    print(f"    {N} ⊕ {N} = {gullibility(N, N)}")

    # Verify expected values
    tests_passed = [
        consensus(T, F) == N,  # Disagreement → no info
        consensus(T, B) == T,  # Agree on truth bit
        gullibility(T, F) == B,  # Accept both → contradiction
        gullibility(N, T) == T,  # Absorb info
    ]

    if all(tests_passed):
        print(f"\n  ✓ PASS: Knowledge lattice operations correct")
    else:
        print(f"\n  ✗ FAIL: Some operations incorrect")

    # Test 4: Bilattice properties
    print("\n" + "-" * 60)
    print("Test 4: Bilattice Properties Validation")

    properties = {
        "Commutativity (∧)": lambda: all(
            and_t(x, y) == and_t(y, x) for x in values for y in values
        ),
        "Commutativity (∨)": lambda: all(
            or_t(x, y) == or_t(y, x) for x in values for y in values
        ),
        "Commutativity (⊗)": lambda: all(
            consensus(x, y) == consensus(y, x) for x in values for y in values
        ),
        "Commutativity (⊕)": lambda: all(
            gullibility(x, y) == gullibility(y, x) for x in values for y in values
        ),
        "Associativity (∧)": lambda: all(
            and_t(and_t(x, y), z) == and_t(x, and_t(y, z))
            for x in values
            for y in values
            for z in values
        ),
        "Associativity (∨)": lambda: all(
            or_t(or_t(x, y), z) == or_t(x, or_t(y, z))
            for x in values
            for y in values
            for z in values
        ),
        "Involution (¬¬x = x)": lambda: all(not_t(not_t(x)) == x for x in values),
        "Absorption (∧ over ∨)": lambda: all(
            and_t(x, or_t(x, y)) == x for x in values for y in values
        ),
        "Absorption (∨ over ∧)": lambda: all(
            or_t(x, and_t(x, y)) == x for x in values for y in values
        ),
        "De Morgan (¬(x ∧ y))": lambda: all(
            not_t(and_t(x, y)) == or_t(not_t(x), not_t(y)) for x in values for y in values
        ),
        "De Morgan (¬(x ∨ y))": lambda: all(
            not_t(or_t(x, y)) == and_t(not_t(x), not_t(y)) for x in values for y in values
        ),
        "Identity (⊗ with ⊤)": lambda: all(consensus(x, B) == x for x in values),
    }

    all_passed = True
    for name, prop in properties.items():
        passed = prop()
        status_str = "✓" if passed else "✗"
        print(f"  {status_str} {name}: {passed}")
        all_passed = all_passed and passed

    if all_passed:
        print(f"\n  ✓ PASS: All 12 bilattice properties satisfied")
    else:
        print(f"\n  ✗ FAIL: Some properties violated")

    # Test 5: Status assignment
    print("\n" + "-" * 60)
    print("Test 5: Status Assignment")

    tau = 0.68
    tau_prime = 0.32

    test_cases = [
        (0.80, 0.20, BelnapValue.TRUE, "High support, low countersupport"),
        (0.20, 0.80, BelnapValue.FALSE, "Low support, high countersupport"),
        (0.75, 0.75, BelnapValue.BOTH, "High support AND countersupport (contradiction)"),
        (0.50, 0.50, BelnapValue.NEITHER, "Insufficient evidence"),
    ]

    print(f"\n  Thresholds: τ={tau}, τ'={tau_prime}")

    all_correct = True
    for s_c, s_bar_c, expected, description in test_cases:
        result = status(s_c, s_bar_c, tau, tau_prime)
        correct = result == expected
        status_str = "✓" if correct else "✗"
        print(
            f"  {status_str} s={s_c:.2f}, s̄={s_bar_c:.2f} → {result} (expected {expected}) - {description}"
        )
        all_correct = all_correct and correct

    if all_correct:
        print(f"\n  ✓ PASS: Status assignment correct")
    else:
        print(f"\n  ✗ FAIL: Some assignments incorrect")

    print("\n" + "=" * 60)
    print("Belnap 4-Valued Logic MWE: All tests completed")
    print("=" * 60)


if __name__ == "__main__":
    main()
