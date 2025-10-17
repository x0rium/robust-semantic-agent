# Implementation Plan: Robust Semantic Agent Full Prototype

**Branch**: `002-full-prototype` | **Date**: 2025-10-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-full-prototype/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a full-featured prototype of the Robust Semantic Agent integrating three subsystems: (1) CVaR-based risk management in belief space, (2) CBF-QP safety filtering with zero violation guarantee, and (3) Belnap 4-valued semantic layer with credal sets for contradictory information and EVI-based query action. The agent demonstrates safe navigation in a 2D environment with forbidden zones, handling uncertainty through particle belief tracking and making risk-aware decisions under contradictory exogenous messages.

Technical approach centers on exploration-first development (MWEs in exploration/ before implementation), formal alignment with docs/theory.md mathematical specifications (commutative updates, bilattice operations, supermartingale CBF), and comprehensive test coverage (80%+) with TDD workflow. Deliverables include functional CLI scripts (train, rollout, evaluate, calibrate), YAML configuration support, JSONL episode logging, and automated calibration/risk/safety reporting.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: NumPy 1.24+, SciPy 1.10+, cvxpy 1.3+ (for QP solver), pytest 7.0+ (testing), ruff + black (linting), mypy/pyright (type checking), matplotlib 3.7+ (plotting), PyYAML 6.0+ (config)
**Storage**: File-based (JSONL for episode logs, YAML for configs, PNG/PDF for reports)
**Testing**: pytest with coverage plugin, parametrized tests for mathematical properties
**Target Platform**: Linux/macOS development environments (Python 3.11+ compatible)
**Project Type**: Single project (research prototype with CLI interface)
**Performance Goals**: 30+ Hz execution rate with 10k belief particles during rollout demonstrations
**Constraints**: Commutative update TV distance ≤ 1e-6, ECE ≤ 0.05, zero safety violations in CBF-enabled runs
**Scale/Scope**: ~8 core modules (belief, messages, credal, semantics, risk/cvar, safety/cbf, query, policy/agent), 4 CLI scripts, 6 test suites, ~3-5k LOC estimated

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Principle I: Exploration-First Development

**Status**: ⚠️ **NOT YET COMPLIANT** (will be addressed in Phase 0)

**Requirements**:
- [ ] MWE for particle filter belief updates (observation kernel G, message multipliers M)
- [ ] MWE for CVaR@α computation on sample distributions
- [ ] MWE for cvxpy QP solver (CBF constraint enforcement)
- [ ] MWE for Belnap bilattice operations (truth/knowledge orders)
- [ ] MCP/agent verification trace in `exploration/003_tools.md`
- [ ] Version-pinned dependencies in `docs/verified-apis.md` (NumPy, SciPy, cvxpy)

**Plan**: Phase 0 research will create MWEs and document findings before implementation begins.

### Principle II: Reality-Verified Implementation

**Status**: ✅ **COMPLIANT**

**Compliance**:
- No mocks in production code (only in unit test boundaries per TDD)
- Hyperparameters via YAML configs (FR-012)
- Reproducible runs with fixed seeds (configs/default.yaml: seed: 42)
- No external integrations requiring contract tests (self-contained prototype)

### Principle III: Formal Specification Alignment

**Status**: ✅ **COMPLIANT**

**Compliance**:
- FR-002: Commutative update TV distance ≤ 1e-6 (test in test_belief.py)
- FR-003: Belnap bilattice operations follow (≤_t, ≤_k) properties (test in test_semantics.py)
- FR-005: Credal set lower expectation ≤ extreme posteriors (test in test_credal.py)
- SC-005: CVaR@α matches analytical on toy distributions (test in test_risk.py)
- FR-007: CBF supermartingale 𝔼[B(x+)|x,u] ≤ B(x) (test in test_safety.py)
- FR-015: Docstrings cite theory.md theorem numbers (e.g., "implements Theorem 2")

### Principle IV: Risk-Safety-Semantics Triad

**Status**: ✅ **COMPLIANT**

**Compliance**:
- **Risk**: CVaR@α mandatory (FR-006), risk-Bellman operator, SC-005/SC-010 validation
- **Safety**: CBF-QP filter (FR-007), ≥1% activation logging (SC-002), zero violations (SC-001)
- **Semantics**: Belnap status v_t(c) (FR-004), calibration ECE ≤ 0.05 (FR-009, SC-008), EVI-based query (FR-008, SC-006/SC-007)
- Integration tests validate all three subsystems together (100-episode scenarios)

### Principle V: Test-Driven Development (NON-NEGOTIABLE)

**Status**: ✅ **COMPLIANT**

**Compliance**:
- TDD workflow enforced: write tests → verify FAIL → implement → verify PASS
- Test hierarchy: unit (mathematical properties), integration (100-episode scenarios)
- SC-012: 80% coverage target for core modules
- pytest with parametrized tests for commutativity, bilattice, CVaR correctness

### Principle VI: Performance & Calibration Standards

**Status**: ✅ **COMPLIANT**

**Compliance**:
- SC-009: 30+ Hz with 10k particles (log if degraded)
- SC-008: ECE ≤ 0.05 with auto-tuned τ/τ'
- SC-001: Zero S^c violations in 100 CBF episodes
- SC-007: ≥10% regret reduction with query, SC-006: ≥20% entropy reduction

### Mathematical Rigor

**Status**: ✅ **COMPLIANT**

**Compliance**:
- All symbols/notation match docs/theory.md (§1 dynamics, §2 semantics, §3 messages, §4 risk/safety)
- Docstrings cite theorem numbers: "Theorem 1: belief-MDP equivalence", "Theorem 2: Doob supermartingale", etc.
- Type hints (PEP 484/585) for all public APIs, validated with mypy/pyright (SC-013)
- Measure spaces 𝒳, 𝒰, 𝒪 as explicit types (e.g., StateSpace, ActionSpace classes)

### Security & Ethics

**Status**: ✅ **COMPLIANT**

**Compliance**:
- Defensive use only (safe decision-making under uncertainty)
- No offensive capabilities (adversarial belief manipulation, safety exploit tools)
- Contradictory info (v=⊤) handled via credal sets (prevents deception attacks)
- Configurable observation logging redaction (no sensitive data exposure)

### Overall Gate Decision

**Pre-Phase 0**: ⚠️ **CONDITIONAL PASS** - Proceed to Phase 0 with requirement to complete Principle I (Exploration-First) MWEs and verification before Phase 1 design.

**Post-Phase 0**: ✅ **PASS** - All research completed (research.md), MWE requirements identified, dependencies verified in docs/verified-apis.md

**Post-Phase 1**: ✅ **FINAL PASS** - Design artifacts complete:
- ✅ research.md (4 technical area research summaries with decisions)
- ✅ data-model.md (9 entity schemas with validation rules)
- ✅ quickstart.md (end-to-end user guide with performance benchmarks)
- ✅ Agent context updated (CLAUDE.md with Python 3.11+, NumPy, SciPy, cvxpy dependencies)
- N/A contracts/ (research prototype with CLI, no API contracts needed)

All constitution principles remain compliant. Ready for Phase 2 (/speckit.tasks).

## Project Structure

### Documentation (this feature)

```
specs/002-full-prototype/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command) - N/A for this prototype
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
robust-semantic-agent/
├── exploration/                  # Phase 0 MWEs (Constitution Principle I)
│   ├── 001_particle_filter.py   # Belief update with G, M multipliers
│   ├── 002_cvar.py               # CVaR@α on sample distributions
│   ├── 003_qp_solver.py          # cvxpy CBF constraint demo
│   ├── 004_belnap.py             # Bilattice operations (≤_t, ≤_k)
│   └── 003_tools.md              # MCP/agent verification trace
│
├── robust_semantic_agent/        # Main package (src/)
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── belief.py             # Belief class: particles, update_obs, apply_messages
│   │   ├── messages.py           # Message, SourceTrust dataclasses
│   │   ├── credal.py             # CredalSet: ensemble posteriors, lower expectation
│   │   ├── semantics.py          # Belnap operations, status assignment, calibration
│   │   └── query.py              # EVI/EVPI, should_query logic
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── cvar.py               # cvar(values, alpha), RiskBellman class
│   │   └── nested.py             # Nested CVaR interface (optional, future)
│   ├── safety/
│   │   ├── __init__.py
│   │   └── cbf.py                # SafetyFilter: QP projection, supermartingale checks
│   ├── policy/
│   │   ├── __init__.py
│   │   ├── planner.py            # VI/Perseus-PBVI or actor-critic
│   │   └── agent.py              # Agent: act(), learn(), filter() integration
│   ├── envs/
│   │   ├── __init__.py
│   │   └── forbidden_circle/
│   │       ├── __init__.py
│   │       ├── env.py            # 2D navigation dynamics K, observations G, render
│   │       ├── safety.py         # Barrier function B(x), CBF parameters
│   │       └── configs/
│   │           └── default.yaml  # Environment-specific config (radius, noise, goals)
│   ├── reports/
│   │   ├── __init__.py
│   │   ├── calibration.py        # ECE, Brier, ROC, reliability diagrams
│   │   ├── risk.py               # CVaR curves, tail distributions
│   │   ├── safety.py             # B(x) traces, violation rates
│   │   └── credal.py             # Posterior ensemble visualization
│   └── cli/
│       ├── __init__.py
│       ├── train.py              # Training/planning script
│       ├── rollout.py            # Rollout with visualization
│       ├── evaluate.py           # Generate reports
│       └── calibrate.py          # Threshold auto-tuning
│
├── tests/
│   ├── unit/
│   │   ├── test_belief.py        # Commutative update TV ≤ 1e-6
│   │   ├── test_semantics.py     # Belnap bilattice properties
│   │   ├── test_credal.py        # Lower expectation monotonicity
│   │   ├── test_risk.py          # CVaR@α analytical correctness
│   │   ├── test_safety.py        # CBF supermartingale property
│   │   └── test_query.py         # EVI computation
│   └── integration/
│       ├── test_navigation.py    # 100-episode P1 scenario
│       ├── test_contradictions.py # P2 credal set scenario
│       └── test_query_action.py  # P3 query ROI scenario
│
├── configs/
│   ├── default.yaml              # Master hyperparameter config
│   ├── risk.yaml                 # CVaR alpha, nested flags
│   ├── safety.yaml               # CBF enabled, QP params
│   └── thresholds.yaml           # τ, τ' (or auto=true)
│
├── docs/
│   ├── theory.md                 # (Exists) Formal mathematical spec
│   ├── verified-apis.md          # (New) Dependency versions, req→resp examples
│   ├── gotchas.md                # (New) Theory divergences if any
│   └── examples/                 # (New) Quickstart notebooks
│
├── runs/                         # Episode logs (*.jsonl), generated at runtime
├── reports/                      # Generated plots/metrics (calibration/, risk/, safety/, credal/)
│
├── pyproject.toml                # Project metadata, dependencies, build config
├── Makefile                      # Shortcuts: make install, make test, make lint, make report
├── README.md                     # (Exists) High-level overview
├── CLAUDE.md                     # (Exists) AI agent development guide
└── .specify/
    └── memory/
        └── constitution.md       # (Exists) Governance framework
```

**Structure Decision**: Single project layout chosen because this is a self-contained research prototype with CLI interface, not a web/mobile application. All components live under `robust_semantic_agent/` package with clear separation: `core/` for belief/semantics, `risk/` for CVaR, `safety/` for CBF, `policy/` for agent logic, `envs/` for demo environment, `reports/` for analysis, `cli/` for user scripts. Tests mirror this structure with `unit/` for mathematical properties and `integration/` for scenario validation.

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

**No violations requiring justification.** All constitution principles are compliant or have clear remediation path (Principle I addressed in Phase 0).
