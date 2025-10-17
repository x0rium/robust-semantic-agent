# Theory Divergences and Implementation Gotchas

**Purpose**: Document any divergences between `docs/theory.md` formal specifications and actual implementation decisions.

**Status**: Initial placeholder - to be updated as implementation progresses.

---

## Constitution Principle III Compliance

Per Constitution Principle III (Formal Specification Alignment), this document tracks:
- Any necessary deviations from `docs/theory.md` mathematical specifications
- Rationale for each divergence
- Impact on correctness guarantees
- Mitigation strategies

**Current Status**: No divergences yet - implementation in progress.

---

## Placeholder Sections (to be filled during implementation)

### Particle Filter Belief Tracking
**Specification** (theory.md §1): TBD
**Implementation**: TBD
**Divergences**: None yet

### CVaR Risk Measure
**Specification** (theory.md §4): TBD
**Implementation**: TBD
**Divergences**: None yet

### CBF Safety Filter
**Specification** (theory.md §4): TBD
**Implementation**: TBD
**Divergences**: None yet

### Belnap Semantics
**Specification** (theory.md §2): TBD
**Implementation**: TBD
**Divergences**: None yet

---

## Common Implementation Patterns to Watch

1. **Numerical Precision**
   - Log-space operations may introduce rounding errors
   - Mitigation: Use log-sum-exp trick, monitor TV distance

2. **Discretization Effects**
   - Particle count N finite (theory assumes continuous)
   - Mitigation: Use N ≥ 5000, test commutativity empirically

3. **QP Solver Tolerances**
   - OSQP uses numerical tolerances (theory assumes exact)
   - Mitigation: Set eps_abs/eps_rel appropriately, use slack variables

4. **Calibration Convergence**
   - Threshold tuning may not achieve exact ECE target
   - Mitigation: Iterative refinement, cost-matrix weighting

---

**Last Updated**: 2025-10-16
**Next Review**: After Phase 3 (User Story 1) implementation
