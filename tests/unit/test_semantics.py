"""
Unit Tests: Belnap Semantics
Feature: 002-full-prototype
Task: T043, T044

Tests Belnap 4-valued logic bilattice operations and status assignment.

References:
- docs/theory.md §3: Belnap bilattice
- FR-003: Bilattice properties
- FR-004: Status assignment v_t(c)
"""

import pytest

from robust_semantic_agent.core.semantics import (
    BelnapValue,
    and_t,
    consensus,
    gullibility,
    not_t,
    or_t,
    status,
)


class TestBelnapBilattice:
    """
    Test Belnap bilattice operations.

    Properties to verify (12 total):
    1-2. Commutativity: x ∧_t y = y ∧_t x, x ∨_t y = y ∨_t x
    3-4. Associativity: (x ∧_t y) ∧_t z = x ∧_t (y ∧_t z)
    5-6. Absorption: x ∧_t (x ∨_t y) = x, x ∨_t (x ∧_t y) = x
    7-8. Involution: ¬_t(¬_t x) = x
    9-10. De Morgan: ¬_t(x ∧_t y) = ¬_t x ∨_t ¬_t y
    11-12. Identity: x ∧_t ⊤ = x, x ∨_t ⊥ = x

    References:
        - Task T043: Bilattice property tests
        - FR-003: Belnap operations correctness
    """

    @pytest.fixture
    def all_values(self):
        """All four Belnap values."""
        return [
            BelnapValue.NEITHER,  # ⊥
            BelnapValue.TRUE,  # t
            BelnapValue.FALSE,  # f
            BelnapValue.BOTH,  # ⊤
        ]

    def test_and_t_commutativity(self, all_values):
        """Property 1: x ∧_t y = y ∧_t x"""
        for x in all_values:
            for y in all_values:
                assert and_t(x, y) == and_t(y, x), f"and_t not commutative: {x} ∧ {y} ≠ {y} ∧ {x}"

    def test_or_t_commutativity(self, all_values):
        """Property 2: x ∨_t y = y ∨_t x"""
        for x in all_values:
            for y in all_values:
                assert or_t(x, y) == or_t(y, x), f"or_t not commutative: {x} ∨ {y} ≠ {y} ∨ {x}"

    def test_and_t_associativity(self, all_values):
        """Property 3: (x ∧_t y) ∧_t z = x ∧_t (y ∧_t z)"""
        for x in all_values:
            for y in all_values:
                for z in all_values:
                    left = and_t(and_t(x, y), z)
                    right = and_t(x, and_t(y, z))
                    assert left == right, f"and_t not associative: ({x}∧{y})∧{z} ≠ {x}∧({y}∧{z})"

    def test_or_t_associativity(self, all_values):
        """Property 4: (x ∨_t y) ∨_t z = x ∨_t (y ∨_t z)"""
        for x in all_values:
            for y in all_values:
                for z in all_values:
                    left = or_t(or_t(x, y), z)
                    right = or_t(x, or_t(y, z))
                    assert left == right, f"or_t not associative: ({x}∨{y})∨{z} ≠ {x}∨({y}∨{z})"

    def test_and_absorption(self, all_values):
        """Property 5: x ∧_t (x ∨_t y) = x"""
        for x in all_values:
            for y in all_values:
                result = and_t(x, or_t(x, y))
                assert result == x, f"and absorption fails: {x} ∧ ({x}∨{y}) = {result} ≠ {x}"

    def test_or_absorption(self, all_values):
        """Property 6: x ∨_t (x ∧_t y) = x"""
        for x in all_values:
            for y in all_values:
                result = or_t(x, and_t(x, y))
                assert result == x, f"or absorption fails: {x} ∨ ({x}∧{y}) = {result} ≠ {x}"

    def test_not_involution(self, all_values):
        """Property 7: ¬_t(¬_t x) = x"""
        for x in all_values:
            result = not_t(not_t(x))
            assert result == x, f"not_t involution fails: ¬¬{x} = {result} ≠ {x}"

    def test_de_morgan_and(self, all_values):
        """Property 9: ¬_t(x ∧_t y) = ¬_t x ∨_t ¬_t y"""
        for x in all_values:
            for y in all_values:
                left = not_t(and_t(x, y))
                right = or_t(not_t(x), not_t(y))
                assert left == right, f"De Morgan and fails: ¬({x}∧{y}) = {left} ≠ {right}"

    def test_de_morgan_or(self, all_values):
        """Property 10: ¬_t(x ∨_t y) = ¬_t x ∧_t ¬_t y"""
        for x in all_values:
            for y in all_values:
                left = not_t(or_t(x, y))
                right = and_t(not_t(x), not_t(y))
                assert left == right, f"De Morgan or fails: ¬({x}∨{y}) = {left} ≠ {right}"

    def test_and_identity(self, all_values):
        """Property 11: x ∧_t TRUE = x (TRUE is identity for AND in truth order)"""
        for x in all_values:
            result = and_t(x, BelnapValue.TRUE)
            assert result == x, f"and identity fails: {x} ∧ t = {result} ≠ {x}"

    def test_or_identity(self, all_values):
        """Property 12: x ∨_t FALSE = x (FALSE is identity for OR in truth order)"""
        for x in all_values:
            result = or_t(x, BelnapValue.FALSE)
            assert result == x, f"or identity fails: {x} ∨ f = {result} ≠ {x}"

    def test_truth_order(self):
        """Verify truth order: f ≤_t ⊥,⊤ ≤_t t (FALSE bottom, TRUE top in truth lattice)"""
        # FALSE is bottom in truth order: f ∧ x = f, f ∨ x = x
        assert and_t(BelnapValue.FALSE, BelnapValue.NEITHER) == BelnapValue.FALSE
        assert and_t(BelnapValue.FALSE, BelnapValue.TRUE) == BelnapValue.FALSE
        assert and_t(BelnapValue.FALSE, BelnapValue.BOTH) == BelnapValue.FALSE

        # TRUE is top in truth order: t ∨ x = t, t ∧ x = x
        assert or_t(BelnapValue.TRUE, BelnapValue.NEITHER) == BelnapValue.TRUE
        assert or_t(BelnapValue.TRUE, BelnapValue.FALSE) == BelnapValue.TRUE
        assert or_t(BelnapValue.TRUE, BelnapValue.BOTH) == BelnapValue.TRUE

        # Verify Wikipedia case: ⊥ ∧ ⊤ = f, ⊥ ∨ ⊤ = t
        assert and_t(BelnapValue.NEITHER, BelnapValue.BOTH) == BelnapValue.FALSE
        assert or_t(BelnapValue.NEITHER, BelnapValue.BOTH) == BelnapValue.TRUE


class TestConsensusGullibility:
    """
    Test consensus and gullibility operators.

    Consensus: Meet in knowledge order (⊥ ≤_k t,f ≤_k ⊤)
    Gullibility: Join in knowledge order

    References:
        - Task T043: Additional bilattice operations
        - docs/theory.md §3.2: Knowledge order
    """

    def test_consensus_symmetric(self):
        """Consensus is symmetric: consensus(x, y) = consensus(y, x)"""
        values = [BelnapValue.NEITHER, BelnapValue.TRUE, BelnapValue.FALSE, BelnapValue.BOTH]
        for x in values:
            for y in values:
                assert consensus(x, y) == consensus(y, x)

    def test_consensus_true_false_gives_neither(self):
        """consensus(t, f) = ⊥ (conflicting information)"""
        assert consensus(BelnapValue.TRUE, BelnapValue.FALSE) == BelnapValue.NEITHER

    def test_consensus_both_resolves(self):
        """consensus(⊤, x) = x (⊤ contains all info, meet gives x)"""
        assert consensus(BelnapValue.BOTH, BelnapValue.TRUE) == BelnapValue.TRUE
        assert consensus(BelnapValue.BOTH, BelnapValue.FALSE) == BelnapValue.FALSE

    def test_gullibility_accepts_all(self):
        """gullibility(t, f) = ⊤ (accept both, contradiction)"""
        assert gullibility(BelnapValue.TRUE, BelnapValue.FALSE) == BelnapValue.BOTH

    def test_gullibility_neither_bottom(self):
        """gullibility(⊥, x) = x (⊥ has no info, join gives x)"""
        assert gullibility(BelnapValue.NEITHER, BelnapValue.TRUE) == BelnapValue.TRUE
        assert gullibility(BelnapValue.NEITHER, BelnapValue.FALSE) == BelnapValue.FALSE


class TestStatusAssignment:
    """
    Test status assignment v_t(c) based on support/countersupport.

    Status rules (τ=0.7, τ'=0.3 default):
    - TRUE:    s_c ≥ τ and s̄_c < τ'
    - FALSE:   s̄_c ≥ τ and s_c < τ'
    - BOTH:    s_c ≥ τ and s̄_c ≥ τ (contradiction)
    - NEITHER: otherwise (insufficient evidence)

    References:
        - Task T044: Status assignment tests
        - FR-004: Status function specification
    """

    def test_status_true_high_support_low_counter(self):
        """s_c=0.8, s̄_c=0.2 → TRUE"""
        v = status(s_c=0.8, s_bar_c=0.2, tau=0.7, tau_prime=0.3)
        assert v == BelnapValue.TRUE

    def test_status_false_high_counter_low_support(self):
        """s_c=0.2, s̄_c=0.8 → FALSE"""
        v = status(s_c=0.2, s_bar_c=0.8, tau=0.7, tau_prime=0.3)
        assert v == BelnapValue.FALSE

    def test_status_both_high_support_high_counter(self):
        """s_c=0.75, s̄_c=0.75 → BOTH (contradiction)"""
        v = status(s_c=0.75, s_bar_c=0.75, tau=0.7, tau_prime=0.3)
        assert v == BelnapValue.BOTH

    def test_status_neither_low_support_low_counter(self):
        """s_c=0.4, s̄_c=0.4 → NEITHER (insufficient)"""
        v = status(s_c=0.4, s_bar_c=0.4, tau=0.7, tau_prime=0.3)
        assert v == BelnapValue.NEITHER

    def test_status_neither_medium_support(self):
        """s_c=0.6, s̄_c=0.5 → NEITHER (neither threshold crossed)"""
        v = status(s_c=0.6, s_bar_c=0.5, tau=0.7, tau_prime=0.3)
        assert v == BelnapValue.NEITHER

    def test_status_boundary_tau(self):
        """Test boundary at τ threshold"""
        # Just below threshold
        v_below = status(s_c=0.69, s_bar_c=0.2, tau=0.7, tau_prime=0.3)
        assert v_below == BelnapValue.NEITHER

        # At threshold
        v_at = status(s_c=0.7, s_bar_c=0.2, tau=0.7, tau_prime=0.3)
        assert v_at == BelnapValue.TRUE

    def test_status_custom_thresholds(self):
        """Test with non-standard thresholds"""
        v = status(s_c=0.6, s_bar_c=0.1, tau=0.5, tau_prime=0.2)
        assert v == BelnapValue.TRUE

    def test_status_edge_case_tau_equal_tau_prime(self):
        """Edge case: τ = τ' (should still work logically)"""
        # This is degenerate but should not crash
        v = status(s_c=0.6, s_bar_c=0.4, tau=0.5, tau_prime=0.5)
        # Exact behavior depends on implementation, just ensure no crash
        assert v in [BelnapValue.NEITHER, BelnapValue.TRUE, BelnapValue.FALSE, BelnapValue.BOTH]


class TestBelnapValueEnum:
    """
    Test BelnapValue enum basic properties.

    References:
        - Task T027: BelnapValue enum (already implemented in US1)
        - Verify 2-bit encoding
    """

    def test_enum_values(self):
        """Verify 2-bit encoding: ⊥=00, t=01, f=10, ⊤=11"""
        assert BelnapValue.NEITHER.value == 0b00
        assert BelnapValue.TRUE.value == 0b01
        assert BelnapValue.FALSE.value == 0b10
        assert BelnapValue.BOTH.value == 0b11

    def test_enum_distinct(self):
        """All four values are distinct"""
        values = [BelnapValue.NEITHER, BelnapValue.TRUE, BelnapValue.FALSE, BelnapValue.BOTH]
        assert len(set(values)) == 4

    def test_enum_string_representation(self):
        """String representation is readable"""
        # Just verify it doesn't crash
        for v in [BelnapValue.NEITHER, BelnapValue.TRUE, BelnapValue.FALSE, BelnapValue.BOTH]:
            str(v)
            repr(v)
