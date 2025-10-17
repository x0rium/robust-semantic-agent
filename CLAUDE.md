# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Robust Semantic Agent (RSA)** - A research prototype implementing a belief-MDP/POMDP-based agent with:
- Dynamic risk measures (CVaR/nested CVaR)
- Safety guarantees (CBF-QP/viability)
- Belnap 4-valued logic semantic layer
- Credal sets for contradictory information
- Query action with EVI (Expected Value of Information) decision rule

**Language:** Python 3.11+
**Domain:** Reinforcement learning, decision theory under uncertainty, formal semantics

## Architecture

The system is designed around belief-space planning with multiple uncertainty layers:

### Core Components (planned in `robust_semantic_agent/`)

**Belief Management** (`core/belief.py`, `core/messages.py`, `core/credal.py`)
- Particle/grid-based belief tracking with observation updates via kernel G
- Soft-likelihood message incorporation: M_{c,s,v}(x) multipliers from sources
- Commutative update rule: observations first, then messages, preserving total variation ≤ 1e-6
- Credal sets for v=⊤ (contradiction): ensemble of K extreme posteriors with lower expectation

**Semantics** (`core/semantics.py`)
- Belnap bilattice B={⊥,t,f,⊤} with truth (≤_t) and knowledge (≤_k) orders
- Status assignment for claims c: v_t(c) based on support/countersupport and thresholds τ, τ'
- Calibration: auto-tune τ>0.5>τ' to achieve ECE ≤ 0.05 with cost-matrix penalties

**Risk** (`core/risk/`)
- `cvar.py`: CVaR@α tail risk measure (mandatory)
- `nested.py`: Nested CVaR or coherent dynamic risk interface for infinite horizon
- Risk-Bellman backup: T_ρ V(b) = max_u CVaR_α(r(b,u) + γV(b'))

**Safety** (`core/safety/`)
- `cbf.py`: QP-based safety filter using stochastic control barrier functions
- Constraint: B(x⁺) ≤ B(x), {x: B(x)≤0} ⊆ S (safe set)
- Log all corrections; must show ≥1% filter activations in 100 episodes

**Query & Active Information** (`core/query.py`)
- EVI calculation: E[V(β_post)] - V(β) for potential observations
- Action **query**: abstain + request observation at cost c
- Trigger rule: query when EVI ≥ Δ* (min expected regret threshold)

**Policy** (`core/policy/`)
- `planner.py`: VI/Perseus-PBVI or actor-critic in belief space
- `agent.py`: Main loop coordinating act(), learn(), safety_filter()

### Environment (`envs/forbidden_circle/`)
- R² navigation with forbidden circular zone S^c
- Noisy beacon observations
- "Gossip" source generating contradictory messages (v=⊤)
- Demonstrates credal set expansion and query triggering

### Reports (`reports/`)
- `calibration.py`: ECE, Brier score, ROC, reliability diagrams
- `risk.py`: Empirical tail distributions, CVaR curves vs baseline
- `safety.py`: Barrier function B(x) traces, violation rates
- `credal.py`: Visualize posterior ensemble for contradictory info

## Development Workflow

### No implementation exists yet - start with exploration phase:

1. **Exploration** (`exploration/`)
   - Create minimal working examples (MWE) for each core component
   - Document MCP/agent traces in `003_tools.md`
   - Verify external dependencies (numpy, scipy, cvxpy for QP) with actual output

2. **Verification** (`docs/verified-apis.md`)
   - Record versions, API examples, and any UNRUNNABLE simulations
   - Document gotchas in `docs/gotchas.md`

3. **Implementation** (`robust_semantic_agent/src/`)
   - Only add code after exploration proves feasibility
   - All hyperparameters via YAML configs (no magic numbers)

### Testing Requirements

**Unit tests** (≥80% coverage target):
- Commutative update: total variation distance ≤ 1e-6 between order permutations
- Credal set monotonicity: lower expectation ≤ any extreme posterior
- CVaR@α matches analytical on toy distributions
- Belnap operations follow bilattice properties

**Integration tests**:
- 100 episodes with CBF enabled → zero S^c violations (or ≤α for chance constraints)
- Query episodes: mean EVI ≥ Δ* before query, entropy reduction ≥20% after
- Calibration: ECE ≤ 0.05 on test split

### Commands

**Setup** (to be implemented):
```bash
# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies (when pyproject.toml exists)
pip install -e .

# Or use make targets (when Makefile exists)
make install
```

**Testing**:
```bash
# Run tests with coverage
pytest tests/ --cov=robust_semantic_agent --cov-report=term-missing

# Or via make
make test
```

**Linting**:
```bash
# Format code
black robust_semantic_agent/ tests/
ruff check robust_semantic_agent/ tests/

# Or via make
make lint
```

**CLI Scripts** (planned in `cli/`):
```bash
# Train/plan policy for N episodes
python -m cli.train --alpha 0.1 --delta_star 0.15 --particles 5000

# Rollout with visualization
python -m cli.rollout --config configs/default.yaml --render

# Evaluate metrics
python -m cli.evaluate --runs-dir runs/experiment_name

# Calibrate thresholds
python -m cli.calibrate --target-ece 0.05
```

## Key Implementation Details

### Message Update Commutativity
Messages treated as soft-likelihoods, conditionally independent from observations given x:
1. Apply observation: β̃(x) ∝ G(o|x)·β(x)
2. Apply message: β'(x) ∝ M_{c,s,v}(x)·β̃(x)
Order matters only numerically; enforce identical results via tests.

### Credal Sets for v=⊤
- Logit interval: Λ_s = [-λ_s, +λ_s] where λ_s = log(r_s/(1-r_s))
- Generate K extreme posteriors via varied logit assignments to A_c / A_c^c
- Store as ensemble; compute lower expectation (worst-case) or nested CVaR

### Source Trust Updates
- Beta-Bernoulli with exponential forgetting (η ∈ (0,1))
- Weight successes/failures by claim complexity and base rate
- r_s ← BetaPost(r_s; weighted evidence), then λ_s = log(r_s/(1-r_s))

### Safety Filter Runtime
- Formulate QP: min ||u - u₀||² s.t. B(x⁺) ≤ B(x)
- Solve with cvxpy/osqp (max 50 iterations, slack 1e-3)
- Log {"safety_filter": "applied", "u_raw": ..., "u_safe": ...} when correction occurs

## Configuration Files

All in `configs/`:
- `default.yaml`: seed, discount, particles, resample threshold
- `risk.yaml`: CVaR alpha, nested flag
- `safety.yaml`: CBF enabled, QP parameters
- `thresholds.yaml`: τ, τ' (or auto=true for calibration)

## Acceptance Criteria (Definition of Done)

Before merging to main:
1. ✅ Unit tests pass with coverage ≥80%
2. ✅ Commutativity test: TV distance ≤ 1e-6
3. ✅ Safety: zero S^c entries in 100 CBF-enabled episodes (or ≤α)
4. ✅ Query: EVI ≥ Δ* before trigger, positive ROI (≥10% regret reduction)
5. ✅ Calibration: ECE ≤ 0.05 on test set
6. ✅ CI green: `make test && make report` succeeds
7. ✅ README updated with run instructions

## Development Guidelines

- **No mocks in production code** - only in unit tests for external boundaries
- **Verify APIs before implementing** - use docs/verified-apis.md
- **Small commits** - each commit must pass CI, have clear message
- **No public signature changes** without migration and test updates
- **Performance target**: 30+ Hz for demo with 10k particles (log if degraded)
- **Type hints required** - PEP 484/585, validate with mypy/pyright

## Risk Operator Notes

**CVaR@α** (mandatory):
- Tail expectation beyond α-quantile
- For samples: sort values, average worst α-fraction

**Nested CVaR** (optional, future):
- Recursive composition: ρ_t(Z) = CVaR_α(Z_t + γ·ρ_{t+1}(Z_{t+1}))
- Finite horizon first; infinite via iterative operator with convergence check

## References

- Theory specification: `docs/theory.md` (formal mathematical definitions)
- Task breakdown: README.md §"План работ" (epics A-H)
- Initial issues to create: 10 tasks listed in README §"Стартовые задачи"
