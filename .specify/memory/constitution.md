# Robust Semantic Agent (RSA) Constitution

<!--
Sync Impact Report:
Version change: initial → 1.0.0
Modified principles: N/A (initial constitution)
Added sections:
  - I. Exploration-First Development
  - II. Reality-Verified Implementation
  - III. Formal Specification Alignment
  - IV. Risk-Safety-Semantics Triad
  - V. Test-Driven Development (NON-NEGOTIABLE)
  - VI. Performance & Calibration Standards
  - Mathematical Rigor
  - Security & Ethics
Removed sections: None
Templates requiring updates:
  ✅ plan-template.md - Constitution Check section references this file
  ✅ spec-template.md - Requirements section aligns with principles
  ✅ tasks-template.md - Test-first workflow matches Principle V
Follow-up TODOs: None - all placeholders filled
-->

## Core Principles

### I. Exploration-First Development

**MUST** prove feasibility before implementation. All components require:

- Minimal Working Examples (MWE) in `exploration/` with actual execution output
- MCP/agent tool verification traces documented in `exploration/003_tools.md`
- External dependency validation (numpy, scipy, cvxpy, pytorch/jax) with version pinning
- `docs/verified-apis.md` entries showing: service version, req→resp examples, limits, sources, status (verified | UNRUNNABLE)

**Rationale**: Research prototype dealing with novel integration of belief-MDP, CVaR risk, CBF safety, and Belnap semantics—assumptions must be empirically validated before committing to implementation.

### II. Reality-Verified Implementation

**MUST NOT** ship unverified code. Production code requirements:

- No mocks, fakes, or stubs outside of unit test boundaries
- All external integrations backed by contract tests or documented simulation protocol
- Hyperparameters only via YAML configs (no magic numbers)
- Reproducible runs: pinned dependencies, fixed seeds, logged configs
- When execution impossible: follow simulation protocol (dry-run, static analysis, contract tests, canary environment) and mark **UNRUNNABLE** in `docs/verified-apis.md` with applied validation steps

**Rationale**: Agent operates under uncertainty with safety constraints—unverified assumptions can lead to safety violations or incorrect risk assessments.

### III. Formal Specification Alignment

**MUST** implement according to `docs/theory.md` mathematical specification:

- Commutative belief updates: total variation distance ≤ 1e-6 between observation→message and message→observation orders
- Belnap bilattice operations follow truth (≤_t) and knowledge (≤_k) order properties
- Credal sets for v=⊤ (contradiction): lower expectation ≤ any extreme posterior
- CVaR@α matches analytical values on toy distributions
- CBF supermartingale property: 𝔼[B(x_{t+1}) | x_t, u_t] ≤ B(x_t)

**Rationale**: Theoretical guarantees (safety, calibration, risk coherence) depend on precise mathematical implementation—divergence invalidates proofs.

### IV. Risk-Safety-Semantics Triad

**MUST** maintain integration of three subsystems:

- **Risk**: CVaR@α (mandatory) or nested CVaR dynamic measure; risk-Bellman operator in belief space
- **Safety**: CBF-QP filter logging ≥1% activations in 100 episodes; zero S^c violations (or ≤α for chance constraints)
- **Semantics**: Belnap status assignment with calibrated thresholds (τ > 0.5 > τ'); ECE ≤ 0.05; EVI-based query action triggering when EVI ≥ Δ*

Each subsystem has acceptance criteria defined in CLAUDE.md §"Acceptance Criteria". Features touching one subsystem MUST verify non-regression in the other two.

**Rationale**: Agent's novel contribution is robust integration of these three dimensions—siloed development would break the core value proposition.

### V. Test-Driven Development (NON-NEGOTIABLE)

**MUST** follow strict TDD cycle:

1. Write tests capturing acceptance criteria from spec.md/theory.md
2. Verify tests FAIL (red)
3. Implement minimal code to pass (green)
4. Refactor with ≥80% coverage target maintained

**Test hierarchy**:
- Unit tests: mathematical properties (commutativity, bilattice, CVaR correctness)
- Contract tests: external system boundaries (if applicable)
- Integration tests: 100-episode scenarios validating safety/risk/query metrics

**Rationale**: Complexity of POMDP + risk + safety + semantics makes regression bugs catastrophic; TDD ensures each component's guarantees hold throughout development.

### VI. Performance & Calibration Standards

**MUST** meet quantitative targets:

- **Performance**: 30+ Hz for demo with 10k belief particles (log if degraded)
- **Calibration**: ECE ≤ 0.05 on test split; auto-tuned τ/τ' with cost-matrix penalties
- **Safety**: Zero S^c entries in 100 CBF-enabled episodes (or ≤α with chance constraints)
- **Query ROI**: ≥10% regret reduction when query action enabled; entropy reduction ≥20% post-query

Deviations require justification in complexity tracking table (plan.md) and mitigation plan.

**Rationale**: Research prototype must demonstrate feasibility at realistic scales; calibration ensures semantic layer trustworthiness; safety metrics validate theoretical guarantees.

## Mathematical Rigor

**Consistency with `docs/theory.md`**:
- All mathematical symbols, notation, and formulas in code comments MUST match theory.md definitions
- Theorems referenced in docstrings MUST cite section numbers (e.g., "implements Theorem 2: Doob supermartingale safety")
- Divergence from theory requires explicit documentation in `docs/gotchas.md` with rationale

**Type Safety**:
- Type hints required (PEP 484/585) for all public APIs
- Validate with mypy/pyright in CI
- Measure spaces (𝒳, 𝒰, 𝒪) represented by explicit types/classes

## Security & Ethics

**Defensive Use Only**:
- Agent designed for safe decision-making under uncertainty in legitimate applications
- No contribution to offensive capabilities (adversarial manipulation of beliefs, exploitation of safety filter weaknesses)
- Contradictory information (v=⊤) handling via credal sets prevents deception-based attacks

**Data Handling**:
- Source trust parameters (r_s) and belief states may encode sensitive information—treat as confidential
- Logging MUST NOT expose raw observations if domain-sensitive (configurable redaction)

## Governance

**Amendment Procedure**:
1. Propose changes via issue/PR with rationale
2. Verify impact on dependent templates (plan-template.md, spec-template.md, tasks-template.md)
3. Update constitution version (MAJOR: incompatible principle removal/redefinition; MINOR: new principle/material expansion; PATCH: clarification/typos)
4. Propagate changes to templates and update Sync Impact Report
5. Require approval from project maintainer(s) before merge

**Compliance Review**:
- All PRs MUST reference constitution principles in description
- CI enforces: test coverage ≥80%, type hints, no magic numbers, pinned dependencies
- Pre-merge checklist (CLAUDE.md §"Definition of Done") maps to constitution principles

**Complexity Justification**:
- Violations of simplicity (e.g., adding nested CVaR before CVaR baseline works, introducing 5th agent when 4-agent rule suggests refactoring) require entry in plan.md Complexity Tracking table
- Justification must show: (a) what simpler alternative was tried, (b) why it failed, (c) measurable benefit of complexity

**Runtime Guidance**:
- Use `CLAUDE.md` for AI agent development workflow (exploration→verification→implementation)
- Use `README.md` for human developer onboarding and epic planning
- Use this constitution for non-negotiable constraints and principle arbitration

**Version**: 1.0.0 | **Ratified**: 2025-10-16 | **Last Amended**: 2025-10-16
