# Feature Specification: Robust Semantic Agent Full Prototype

**Feature Branch**: `002-full-prototype`
**Created**: 2025-10-16
**Status**: Draft
**Input**: User description: "давай реализуем полнофункциональный прототип"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Safe Navigation with Risk-Aware Decision Making (Priority: P1)

A researcher configures and runs the agent in a 2D navigation environment with a forbidden circular zone. The agent must reach a goal position while avoiding the forbidden zone, making decisions under uncertainty from noisy beacon observations. The agent balances risk aversion (CVaR) with goal achievement.

**Why this priority**: This is the core demonstration scenario that validates all three subsystems (Risk, Safety, Semantics) working together. Without this working, the prototype cannot demonstrate its fundamental value proposition.

**Independent Test**: Can be fully tested by configuring the forbidden circle environment, running 100 episodes with CBF enabled, and verifying zero zone violations while achieving goal-reaching success rate above 80%.

**Acceptance Scenarios**:

1. **Given** the agent starts outside the forbidden zone with noisy position estimates, **When** it plans a path to the goal, **Then** it reaches the goal in at least 80% of episodes without entering the forbidden zone
2. **Given** the CBF safety filter is enabled, **When** the agent attempts an action that would violate safety constraints, **Then** the safety filter corrects the action and logs the intervention
3. **Given** observations are highly uncertain, **When** the agent updates its belief state, **Then** the belief distribution correctly reflects uncertainty (entropy above threshold) and converges as more observations arrive

---

### User Story 2 - Contradiction Handling via Credal Sets (Priority: P2)

A researcher introduces contradictory information sources (gossip source providing v=⊤ messages). The agent must recognize contradictions, expand its belief representation to a credal set, and make robust worst-case decisions using the lower expectation or nested CVaR over the ensemble of posteriors.

**Why this priority**: This validates the semantic layer's unique capability to handle contradictory information without belief explosion, which is a key differentiator of the RSA approach.

**Independent Test**: Can be tested by configuring a gossip source that emits contradictory claims (v=⊤), observing credal set expansion to K extreme posteriors, and verifying that the agent's decisions remain coherent (no crashes, maintains safety).

**Acceptance Scenarios**:

1. **Given** a claim c with support and countersupport both above threshold τ, **When** the semantic layer assigns status, **Then** it correctly returns v_t(c) = ⊤ (contradiction)
2. **Given** a message with v=⊤ arrives, **When** belief update occurs, **Then** the system creates a credal set with K extreme posteriors spanning the logit interval Λ_s
3. **Given** a credal set exists, **When** action selection occurs, **Then** the risk operator computes lower expectation or nested CVaR across the ensemble and selects an action robust to worst-case posterior

---

### User Story 3 - Active Information Acquisition via Query Action (Priority: P3)

A researcher enables the query action with cost c and threshold Δ*. When the agent encounters high uncertainty and the Expected Value of Information (EVI) exceeds Δ*, the agent chooses to abstain from acting and requests additional observations. After receiving the observation, entropy reduces and the agent proceeds with higher confidence.

**Why this priority**: This demonstrates the agent's meta-reasoning capability to recognize when it needs more information, balancing exploration cost against decision quality improvement.

**Independent Test**: Can be tested by running episodes with query enabled vs disabled, measuring EVI before query actions, entropy reduction after query, and regret improvement (≥10% reduction in regret when query enabled).

**Acceptance Scenarios**:

1. **Given** the agent's current belief has high entropy and EVI ≥ Δ*, **When** action selection occurs, **Then** the agent chooses the query action instead of a physical action
2. **Given** a query action is executed, **When** an additional observation is received, **Then** belief entropy decreases by at least 20%
3. **Given** query action is enabled across many episodes, **When** comparing regret to baseline without query, **Then** regret reduces by at least 10% (positive ROI on information acquisition)

---

### User Story 4 - Semantic Status Calibration and Reporting (Priority: P4)

A researcher runs calibration analysis to tune the thresholds τ and τ' that determine when claims are considered true/false/contradictory/unknown. The system auto-tunes these thresholds to achieve ECE ≤ 0.05 on a held-out test split, accounting for cost asymmetries (false positives vs false negatives) via a cost matrix.

**Why this priority**: Calibration ensures the semantic layer's trustworthiness and interpretability. Without proper calibration, status assignments may mislead users about the agent's confidence.

**Independent Test**: Can be tested by running calibration script on test episodes, generating reliability diagrams and ECE reports, and verifying ECE ≤ 0.05 with tuned thresholds.

**Acceptance Scenarios**:

1. **Given** a dataset of episodes with ground-truth claim outcomes, **When** calibration runs with cost matrix penalties, **Then** the system outputs τ and τ' values achieving ECE ≤ 0.05
2. **Given** calibrated thresholds are applied, **When** generating reports, **Then** reliability diagrams show predicted confidence aligned with empirical accuracy
3. **Given** the agent assigns status t (true) to a claim, **When** measuring calibration, **Then** the empirical frequency of that claim being true matches the confidence level within ±5%

---

### User Story 5 - Performance Monitoring and Risk/Safety Reports (Priority: P5)

A researcher runs evaluation after training to generate comprehensive reports on risk (CVaR curves, tail distributions), safety (barrier function traces, violation rates), and calibration (ECE, Brier, ROC). The reports provide quantitative evidence that the agent meets all acceptance criteria.

**Why this priority**: Comprehensive reporting enables validation of theoretical guarantees and provides transparency for research publication or deployment decisions.

**Independent Test**: Can be tested by running evaluation script on logged episodes and verifying all report artifacts are generated (calibration/, risk/, safety/, credal/ subdirectories with plots and metrics).

**Acceptance Scenarios**:

1. **Given** 100 logged episodes, **When** evaluation runs, **Then** safety report shows barrier function B(x) traces for all episodes and violation rate = 0% (or ≤ α for chance constraints)
2. **Given** risk-averse policy (CVaR α=0.1) and risk-neutral baseline, **When** comparing CVaR curves, **Then** risk-averse policy achieves CVaR no worse than baseline with 95% confidence
3. **Given** calibration report, **When** reviewing ECE/Brier/ROC metrics, **Then** all metrics meet targets (ECE ≤ 0.05, AUC ≥ 0.85)

---

### Edge Cases

- **Belief collapse**: What happens when all particles concentrate on a single point due to low observation noise? The resampling threshold should trigger particle diversity restoration.
- **Safety filter infeasibility**: How does the system handle cases where no safe action exists (QP infeasible)? A fallback "panic" action (e.g., stop/hover) should be triggered and logged.
- **Credal set explosion**: What happens if multiple contradictory messages arrive simultaneously? The system should cap K (ensemble size) and use beam search to maintain computational tractability.
- **Query action loops**: Can the agent get stuck repeatedly querying without taking physical actions? A query budget or timeout mechanism prevents infinite query loops.
- **Source trust drift**: How does the system handle sources whose reliability changes over time? Beta-Bernoulli updates with exponential forgetting (η parameter) adapt trust dynamically.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST implement particle-based belief tracking with observation updates via likelihood kernel G and message incorporation via soft-likelihood multipliers M_{c,s,v}
- **FR-002**: System MUST maintain commutative belief updates (observation → message vs message → observation) with total variation distance ≤ 1e-6
- **FR-003**: System MUST implement Belnap bilattice operations (∧, ∨, ⊗, ⊕, ¬) following truth (≤_t) and knowledge (≤_k) order properties
- **FR-004**: System MUST assign claim status v_t(c) ∈ {⊥, t, f, ⊤} based on support s_c and countersupport s̄_c relative to thresholds τ > 0.5 > τ'
- **FR-005**: System MUST create credal sets (ensemble of K extreme posteriors) when contradictory messages (v=⊤) arrive, using logit intervals Λ_s
- **FR-006**: System MUST compute CVaR@α for return distributions and implement risk-Bellman operator in belief space
- **FR-007**: System MUST implement CBF-QP safety filter projecting unsafe actions to safe alternatives and logging all corrections
- **FR-008**: System MUST compute Expected Value of Information (EVI) and trigger query action when EVI ≥ Δ*
- **FR-009**: System MUST auto-calibrate thresholds τ and τ' to achieve ECE ≤ 0.05 on test split using cost matrix
- **FR-010**: System MUST generate reports (calibration, risk, safety, credal) with plots and quantitative metrics
- **FR-011**: System MUST provide CLI scripts for train, rollout, evaluate, calibrate workflows
- **FR-012**: System MUST support YAML configuration for all hyperparameters (seed, discount, risk alpha, safety QP params, query cost/threshold, belief particles, credal K, thresholds)
- **FR-013**: System MUST log all episodes to JSONL format with belief states, actions, rewards, EVI, claim statuses, safety interventions
- **FR-014**: System MUST update source trust r_s via Beta-Bernoulli with exponential forgetting η
- **FR-015**: System MUST validate all mathematical operations against docs/theory.md formal specification (reference theorem numbers in docstrings)

### Key Entities

- **Belief (β_t)**: Probability distribution over state space 𝒳, represented as particle ensemble (x^(i), w^(i)) or grid approximation; updated via observation kernel G and message multipliers M
- **CredalSet (ℬ_t)**: Ensemble of K extreme posterior distributions arising from contradictory information; supports lower expectation and nested CVaR computation
- **Message (c,s,v)**: Claim c from source s with Belnap truth value v ∈ {⊥, t, f, ⊤}; induces soft-likelihood multiplier M_{c,s,v}(x) using source trust logit λ_s
- **SourceTrust (r_s)**: Reliability parameter for source s ∈ [0,1]; updated via Beta-Bernoulli; λ_s = log(r_s/(1-r_s)) is the logit-reliability
- **BarrierFunction (B)**: Mapping from state to ℝ such that {x: B(x)≤0} ⊆ Safe; maintained as supermartingal via CBF-QP
- **Policy (π)**: Mapping from belief (or credal set) to action distribution; optimizes risk-Bellman operator subject to safety constraints
- **Episode**: Sequence of (belief_t, action_t, reward_t, observation_t, messages_t, safety_diag_t, evi_t, statuses_t) logged to JSONL
- **Configuration**: YAML-serialized hyperparameters (seed, discount γ, risk α, CBF enabled, QP params, query cost c and threshold Δ*, particles N, credal K, calibration target ECE)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Agent achieves zero forbidden zone entries in 100 CBF-enabled episodes (or ≤ α violation rate for chance constraints)
- **SC-002**: Safety filter activates in at least 1% of steps across 100 episodes (demonstrates non-trivial correction)
- **SC-003**: Belief update maintains commutative property with total variation distance ≤ 1e-6 between observation-first and message-first orderings
- **SC-004**: Credal set lower expectation is less than or equal to expectation under any individual extreme posterior (monotonicity check)
- **SC-005**: CVaR@α computation matches analytical values on toy distributions (Gaussian, uniform) within 1% relative error
- **SC-006**: Query action reduces belief entropy by at least 20% on average when triggered
- **SC-007**: Enabling query action reduces regret by at least 10% compared to no-query baseline across 100 episodes
- **SC-008**: Calibration achieves ECE ≤ 0.05 on held-out test split with auto-tuned thresholds
- **SC-009**: Agent demonstrates 30+ Hz performance with 10k belief particles in demo rollout (or logs degradation with explanation)
- **SC-010**: Risk-averse policy (CVaR α=0.1) achieves CVaR no worse than risk-neutral baseline with 95% statistical confidence
- **SC-011**: All CLI commands (train, rollout, evaluate, calibrate) execute successfully and produce expected outputs (logs, reports, configs)
- **SC-012**: Test coverage reaches 80% for core modules (belief, semantics, credal, risk, safety, query)
- **SC-013**: All code passes type checking (mypy/pyright) and linting (ruff, black) in CI
- **SC-014**: Documentation (README, theory.md, CLAUDE.md) accurately reflects implemented system (no outdated information)
