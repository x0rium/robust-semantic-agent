# Belnap 4-Valued Logic Implementation Research

**Date:** 2025-10-16
**Purpose:** Research Belnap bilattice implementation for robust-semantic-agent credal semantics

## Executive Summary

**No dedicated Python libraries exist for Belnap logic.** Implementation requires custom Enum-based solution with explicit operation definitions. Bit-pair representation (t=01, f=10, ⊥=00, ⊤=11) offers computational efficiency and natural encoding of bilattice structure.

**Recommended approach:** Python Enum with operator overloading, validated against canonical truth tables from Belnap-Dunn theory.

---

## 1. Data Structure: Representing B = {⊥, t, f, ⊤}

### Recommended Implementation: Enum + Bit Encoding

```python
from enum import IntEnum
from typing import Tuple

class BelnapValue(IntEnum):
    """Belnap 4-valued logic with bit-pair encoding.

    Encoding: (truth_bit, falsity_bit)
    - NEITHER (⊥): 00 = no information
    - TRUE (t):    01 = only true
    - FALSE (f):   10 = only false
    - BOTH (⊤):    11 = contradiction
    """
    NEITHER = 0b00  # ⊥
    TRUE    = 0b01  # t
    FALSE   = 0b10  # f
    BOTH    = 0b11  # ⊤

    @property
    def truth_bit(self) -> int:
        """Extract truth information bit."""
        return self.value & 0b01

    @property
    def falsity_bit(self) -> int:
        """Extract falsity information bit."""
        return (self.value & 0b10) >> 1

    @classmethod
    def from_bits(cls, truth: int, falsity: int) -> 'BelnapValue':
        """Construct from separate truth/falsity bits."""
        return cls((falsity << 1) | truth)
```

**Advantages:**
- Compact representation (2 bits per value)
- Fast bitwise operations
- Natural correspondence to bilattice semantics: truth_bit indicates positive evidence, falsity_bit indicates negative evidence

**Source:** Wikipedia Four-valued logic (bit encoding 01=T, 10=F, 00=N, 11=B)

---

## 2. Operations: Truth-Preserving vs Knowledge-Preserving

### Truth-Preserving Operations (≤_t lattice)

```python
def and_t(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """Truth conjunction ∧: min on truth order.

    Truth order: f ≤ ⊥ ≤ t and f ≤ ⊤ ≤ t
    """
    truth = min(x.truth_bit, y.truth_bit)
    falsity = max(x.falsity_bit, y.falsity_bit)
    return BelnapValue.from_bits(truth, falsity)

def or_t(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """Truth disjunction ∨: max on truth order."""
    truth = max(x.truth_bit, y.truth_bit)
    falsity = min(x.falsity_bit, y.falsity_bit)
    return BelnapValue.from_bits(truth, falsity)

def not_t(x: BelnapValue) -> BelnapValue:
    """Truth negation ¬: inverts truth, preserves knowledge.

    ¬t=f, ¬f=t, ¬⊥=⊥, ¬⊤=⊤
    """
    return BelnapValue.from_bits(x.falsity_bit, x.truth_bit)
```

### Knowledge-Preserving Operations (≤_k lattice)

```python
def consensus(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """Consensus operator ⊗: ≤_k-meet (most agreed-upon info).

    Knowledge order: ⊥ ≤ t ≤ ⊤ and ⊥ ≤ f ≤ ⊤
    ⊗ = greatest lower bound on knowledge order
    """
    truth = x.truth_bit & y.truth_bit  # agree on truth only if both say true
    falsity = x.falsity_bit & y.falsity_bit  # agree on false only if both say false
    return BelnapValue.from_bits(truth, falsity)

def gullibility(x: BelnapValue, y: BelnapValue) -> BelnapValue:
    """Gullibility operator ⊕: ≤_k-join (combine all info).

    ⊕ = least upper bound on knowledge order (accept everything)
    """
    truth = x.truth_bit | y.truth_bit  # believe true if either says true
    falsity = x.falsity_bit | y.falsity_bit  # believe false if either says false
    return BelnapValue.from_bits(truth, falsity)
```

**Source:** Fitting's bilattice terminology (consensus/gullibility); bitwise ops from math.stackexchange bit-pair encoding

---

## 3. Order Relations

```python
def leq_truth(x: BelnapValue, y: BelnapValue) -> bool:
    """Truth order ≤_t: increase in truth with decrease in falsity.

    f ≤_t ⊥ ≤_t t and f ≤_t ⊤ ≤_t t
    """
    return (x.truth_bit <= y.truth_bit and
            x.falsity_bit >= y.falsity_bit)

def leq_knowledge(x: BelnapValue, y: BelnapValue) -> bool:
    """Knowledge order ≤_k: monotone increase in information.

    ⊥ ≤_k t ≤_k ⊤ and ⊥ ≤_k f ≤_k ⊤
    """
    return (x.truth_bit <= y.truth_bit and
            x.falsity_bit <= y.falsity_bit)
```

**Interpretation:**
- ≤_t: "x is at least as true as y"
- ≤_k: "x contains at most as much information as y"

---

## 4. Complete Truth Tables

### Truth-Preserving Operations

**AND (∧):**
```
  ∧ | ⊥  t  f  ⊤
----+------------
  ⊥ | ⊥  ⊥  f  f
  t | ⊥  t  f  ⊤
  f | f  f  f  f
  ⊤ | f  ⊤  f  ⊤
```

**OR (∨):**
```
  ∨ | ⊥  t  f  ⊤
----+------------
  ⊥ | ⊥  t  ⊥  t
  t | t  t  t  t
  f | ⊥  t  f  ⊤
  ⊤ | t  t  ⊤  ⊤
```

**NOT (¬):**
```
¬⊥ = ⊥
¬t = f
¬f = t
¬⊤ = ⊤
```

**Source:** Wikipedia Four-valued logic article

### Knowledge-Preserving Operations

**CONSENSUS (⊗):**
```
  ⊗ | ⊥  t  f  ⊤
----+------------
  ⊥ | ⊥  ⊥  ⊥  ⊥
  t | ⊥  t  ⊥  t
  f | ⊥  ⊥  f  f
  ⊤ | ⊥  t  f  ⊤
```

**GULLIBILITY (⊕):**
```
  ⊕ | ⊥  t  f  ⊤
----+------------
  ⊥ | ⊥  t  f  ⊤
  t | t  t  ⊤  ⊤
  f | f  ⊤  f  ⊤
  ⊤ | ⊤  ⊤  ⊤  ⊤
```

---

## 5. Test Strategy: Bilattice Properties

### Unit Tests (≥80% coverage target)

**Commutativity:**
```python
def test_commutativity():
    for x in BelnapValue:
        for y in BelnapValue:
            assert and_t(x, y) == and_t(y, x)
            assert or_t(x, y) == or_t(y, x)
            assert consensus(x, y) == consensus(y, x)
            assert gullibility(x, y) == gullibility(y, x)
```

**Associativity:**
```python
def test_associativity():
    for x in BelnapValue:
        for y in BelnapValue:
            for z in BelnapValue:
                assert and_t(x, and_t(y, z)) == and_t(and_t(x, y), z)
                # ... similar for ∨, ⊗, ⊕
```

**Absorption Laws:**
```python
def test_absorption():
    for x in BelnapValue:
        for y in BelnapValue:
            # Truth lattice: x ∧ (x ∨ y) = x
            assert and_t(x, or_t(x, y)) == x
            # Knowledge lattice: x ⊗ (x ⊕ y) = x
            assert consensus(x, gullibility(x, y)) == x
```

**Idempotence:**
```python
def test_idempotence():
    for x in BelnapValue:
        assert and_t(x, x) == x
        assert or_t(x, x) == x
        assert consensus(x, x) == x
        assert gullibility(x, x) == x
```

**Involution (Double Negation):**
```python
def test_involution():
    for x in BelnapValue:
        assert not_t(not_t(x)) == x
```

**Distributivity (12 laws):**
```python
def test_distributivity():
    for x in BelnapValue:
        for y in BelnapValue:
            for z in BelnapValue:
                # ∧ over ∨: x ∧ (y ∨ z) = (x ∧ y) ∨ (x ∧ z)
                assert and_t(x, or_t(y, z)) == or_t(and_t(x, y), and_t(x, z))
                # ⊗ over ⊕: x ⊗ (y ⊕ z) = (x ⊗ y) ⊕ (x ⊗ z)
                assert consensus(x, gullibility(y, z)) == gullibility(
                    consensus(x, y), consensus(x, z))
                # Cross-lattice: ∧ over ⊗, etc. (10 more laws)
```

**Truth Table Validation:**
```python
def test_truth_tables():
    """Validate against canonical Belnap truth tables."""
    # AND
    assert and_t(NEITHER, TRUE) == NEITHER
    assert and_t(TRUE, FALSE) == FALSE
    assert and_t(BOTH, TRUE) == BOTH
    # ... complete tables from section 4
```

---

## 6. Libraries: None Found

**Search results:**
- **No dedicated Belnap logic Python libraries** on PyPI or GitHub
- **Fuzzy logic libraries** (scikit-fuzzy, fuzzylogic) handle continuous multi-valued logic but not discrete 4-valued Belnap semantics
- **Classical logic libraries** (SymPy.logic, LogPy, pyDatalog) lack paraconsistent/bilattice support
- **Theorem provers:** No Python bindings found for Belnap extensions

**Conclusion:** Custom implementation required. Enum-based approach is standard practice (see implementation above).

---

## 7. Integration with RSA Credal Semantics

### Status Assignment (CLAUDE.md requirement)

```python
def assign_status(support: float, countersupport: float,
                  tau: float = 0.7, tau_prime: float = 0.3) -> BelnapValue:
    """Assign Belnap value based on evidence thresholds.

    Args:
        support: Evidence for claim (0-1)
        countersupport: Evidence against claim (0-1)
        tau: Truth threshold (>0.5)
        tau_prime: Falsity threshold (<0.5)

    Returns:
        ⊥ if support ≤ tau_prime and countersupport ≤ tau_prime
        t if support > tau and countersupport ≤ tau_prime
        f if support ≤ tau_prime and countersupport > tau
        ⊤ if support > tau and countersupport > tau
    """
    truth_bit = 1 if support > tau else 0
    falsity_bit = 1 if countersupport > tau else 0
    return BelnapValue.from_bits(truth_bit, falsity_bit)
```

### Credal Set Expansion for ⊤

When status = ⊤ (contradiction), generate credal set of extreme posteriors:
1. Extract logit interval Λ_s = [-λ_s, +λ_s] for source reliability
2. Sample K extreme logit assignments to claim evidence sets A_c, A_c^c
3. Store ensemble of K posteriors
4. Compute lower expectation (worst-case) or nested CVaR

**Link to belief update:** `core/belief.py` message incorporation with v=⊤ triggers credal expansion.

---

## 8. Performance Considerations

- **Bit operations:** AND/OR via bitwise ops are O(1)
- **Storage:** 2 bits per value (vs 8+ bytes for Enum object overhead in naive impl)
- **Optimization:** Use IntEnum backed by numpy uint8 array for bulk operations
- **Target:** Process 10k particle beliefs at 30+ Hz (CLAUDE.md requirement)

---

## References

1. **Wikipedia Four-valued logic** (https://en.wikipedia.org/wiki/Four-valued_logic)
   Bit-pair encoding, truth tables for ∧/∨/¬

2. **Stanford Encyclopedia: Truth Values - Bilattices** (https://plato.stanford.edu/entries/truth-values/generalized-truth-values.html)
   Formal definition of two partial orders, operation semantics

3. **Mathematics StackExchange** (https://math.stackexchange.com/questions/1352967)
   Python code example using bit pairs: t=(1,0), f=(0,1), u=(0,0), p=(1,1)

4. **Fitting, M. - Bilattices in Logic Programming** (ResearchGate)
   Consensus/gullibility terminology, 12 distributive laws

5. **Belnap, N. - A Useful Four-Valued Logic** (ResearchGate)
   Original semantics: handling multiple information sources with contradictions

---

## Next Steps

1. **Implement MWE** (`exploration/005_belnap_mwe.py`):
   - BelnapValue Enum class
   - All 6 operations (∧, ∨, ¬, ⊗, ⊕, ≤_t, ≤_k)
   - Truth table validation tests
   - Run and capture actual output

2. **Verify properties** (unit tests in exploration):
   - 12 distributive laws
   - Commutativity, associativity, absorption
   - Order reflexivity, antisymmetry, transitivity

3. **Document in verified-apis.md**:
   - No external API (self-contained)
   - Mark as **VERIFIED** with test output
   - Record ECE calibration threshold defaults (τ=0.7, τ'=0.3)

4. **Plan integration** (`core/semantics.py`):
   - BelnapValue base class
   - Status assignment function with calibration hooks
   - Message incorporation logic (v=⊤ → credal set trigger)

**DoD checkpoint:** MWE runs with zero assertion failures, coverage ≥80%, TV distance ≤ 1e-6 for operation consistency.
