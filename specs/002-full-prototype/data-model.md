# Data Model: Robust Semantic Agent Full Prototype

**Feature**: 002-full-prototype
**Date**: 2025-10-16
**Purpose**: Define core entity schemas and relationships for belief-MDP with risk, safety, and semantics

---

## Entity Catalog

### Core Entities (Belief Tracking)

1. **Belief** - Probability distribution over state space
2. **CredalSet** - Ensemble of extreme posteriors for contradictory info
3. **Message** - Exogenous claim from external source
4. **SourceTrust** - Reliability parameter for information source

### Policy & Control Entities

5. **BarrierFunction** - Safety constraint mapping
6. **Policy** - Belief-to-action mapping
7. **Agent** - Main control loop coordinating subsystems

### Environment & Execution

8. **Episode** - Logged trajectory with diagnostics
9. **Configuration** - YAML-serialized hyperparameters

---

## 1. Belief

**Description**: Particle-based representation of probability distribution β_t ∈ 𝒫(𝒳) over state space. Sufficient statistic for history of observations and messages.

### Attributes

| Attribute     | Type              | Description                                              | Constraints                     |
|---------------|-------------------|----------------------------------------------------------|---------------------------------|
| `particles`   | `np.ndarray`      | (N, state_dim) array of state samples                    | N ≥ 1000, float64               |
| `log_weights` | `np.ndarray`      | (N,) array of log-probabilities                          | float64, sum(exp(·)) ≈ 1        |
| `state_dim`   | `int`             | Dimensionality of state space                            | ≥ 1                             |
| `n_particles` | `int`             | Number of particles N                                    | Read-only, len(particles)       |
| `eff_n`       | `float`           | Effective sample size 1/Σw²                              | Computed property, ∈ [1, N]     |

### Operations

- `update_obs(observation) → Belief`: Apply observation likelihood G(o|x), return new belief
- `apply_message(message, source_trust) → Belief | CredalSet`: Apply soft-likelihood multiplier M_{c,s,v}(x)
- `sample(n) → np.ndarray`: Draw n samples from belief distribution
- `mean() → np.ndarray`: Compute expectation 𝔼[x]
- `entropy() → float`: Compute Shannon entropy H(β)
- `resample() → Belief`: Systematic resampling if ESS < threshold

### Relationships

- **Produces**: CredalSet (when applying v=⊤ message)
- **Consumes**: Message + SourceTrust
- **Used by**: Agent (for action selection), Policy (as input)

### Validation Rules

- Total variation between commutative update orders ≤ 1e-6 (FR-002)
- Log-weights must not contain NaN or Inf
- Particle array shape must match (n_particles, state_dim)

---

## 2. CredalSet

**Description**: Ensemble of K extreme posterior distributions arising from contradictory information (v=⊤). Represents imprecise probability via convex hull.

### Attributes

| Attribute       | Type                  | Description                                          | Constraints                 |
|-----------------|-----------------------|------------------------------------------------------|-----------------------------|
| `posteriors`    | `list[Belief]`        | K extreme posterior distributions                    | 2 ≤ K ≤ 20                  |
| `K`             | `int`                 | Number of extreme points in ensemble                 | Read-only, len(posteriors)  |
| `base_belief`   | `Belief`              | Pre-contradiction belief (before v=⊤ message)        | Optional, for diagnostics   |

### Operations

- `lower_expectation(f: Callable) → float`: Compute min_{β∈ℬ} 𝔼_β[f(x)] (worst-case)
- `nested_cvar(values: np.ndarray, alpha: float) → float`: CVaR across ensemble distribution
- `sample(n) → np.ndarray`: Sample from mixture of posteriors (uniform mixture)
- `add_posterior(belief: Belief) → None`: Add extreme posterior to ensemble
- `prune(max_k: int) → CredalSet`: Reduce ensemble size via beam search

### Relationships

- **Created by**: Belief.apply_message when v=⊤
- **Used by**: Agent (for robust action selection), Policy (as alternative input type)
- **Composes**: Multiple Belief instances

### Validation Rules

- Lower expectation ≤ expectation of any individual posterior (SC-004 monotonicity)
- All posteriors must have same state_dim
- Posteriors derived from same base_belief with varied logit assignments

---

## 3. Message

**Description**: Exogenous claim (c, s, v) from source s about assertion c with Belnap truth value v.

### Attributes

| Attribute | Type                               | Description                              | Constraints                    |
|-----------|------------------------------------|------------------------------------------|--------------------------------|
| `claim`   | `str`                              | Identifier for claim/assertion           | Non-empty string               |
| `source`  | `str`                              | Identifier for information source        | Non-empty string               |
| `value`   | `BelnapValue` (Enum)               | Truth status: ⊥, t, f, ⊤                 | One of 4 values                |
| `A_c`     | `Callable[[np.ndarray], bool]`     | Region indicator function for claim      | Returns boolean array          |

### Operations

- `multiplier(particles: np.ndarray, source_trust: SourceTrust) → np.ndarray`: Compute M_{c,s,v}(x) for each particle
- `logit_interval(source_trust: SourceTrust) → tuple[float, float]`: Return Λ_s = [-λ_s, +λ_s] for v=⊤

### Relationships

- **Consumed by**: Belief.apply_message
- **Uses**: SourceTrust (for logit λ_s)
- **Triggers**: CredalSet creation (when v=⊤)

### Validation Rules

- claim and source must be valid identifiers (alphanumeric + underscore)
- A_c must be vectorizable (accept np.ndarray, return boolean array)
- value must be valid BelnapValue member

---

## 4. SourceTrust

**Description**: Reliability parameter r_s ∈ [0,1] for information source, updated via Beta-Bernoulli with forgetting.

### Attributes

| Attribute   | Type    | Description                                    | Constraints              |
|-------------|---------|------------------------------------------------|--------------------------|
| `source_id` | `str`   | Identifier for source                          | Non-empty, unique        |
| `r_s`       | `float` | Reliability parameter (Beta posterior mean)    | ∈ [0, 1]                 |
| `alpha`     | `float` | Beta prior/posterior alpha parameter           | > 0                      |
| `beta_val`  | `float` | Beta prior/posterior beta parameter            | > 0                      |
| `eta`       | `float` | Exponential forgetting factor                  | ∈ (0, 1], default 0.95   |

### Operations

- `logit() → float`: Compute λ_s = log(r_s / (1 - r_s))
- `update(success: bool, weight: float = 1.0) → None`: Beta-Bernoulli update with forgetting
- `reset(r_init: float = 0.7) → None`: Reset to initial reliability

### Relationships

- **Used by**: Message.multiplier (for soft-likelihood scaling)
- **Updated by**: Agent (after claim outcome is observed)

### Validation Rules

- r_s ∈ (0, 1) exclusive (avoid log(0) in logit)
- alpha, beta_val > 0 for valid Beta distribution
- Update weight must be non-negative

---

## 5. BarrierFunction

**Description**: Scalar function B: 𝒳 → ℝ defining safe set {x: B(x) ≤ 0} ⊆ 𝒮, maintained as supermartingale.

### Attributes

| Attribute      | Type                          | Description                                | Constraints                  |
|----------------|-------------------------------|--------------------------------------------|------------------------------|
| `B`            | `Callable[[np.ndarray], float]` | Barrier function B(x)                      | Differentiable               |
| `grad_B`       | `Callable[[np.ndarray], np.ndarray]` | Gradient ∇B(x)                            | Returns (state_dim,) array   |
| `alpha`        | `float`                       | Class-𝒦 function parameter                 | > 0, typical 0.5-2.0         |
| `safe_set`     | `Callable[[np.ndarray], bool]` | Indicator for safe region 𝒮                | Vectorizable                 |

### Operations

- `evaluate(state: np.ndarray) → float`: Compute B(x)
- `lie_derivative(state: np.ndarray, dynamics: Callable) → float`: Compute L_f B(x)
- `control_barrier_constraint(state, u_des) → tuple`: Return QP constraint coefficients

### Relationships

- **Used by**: SafetyFilter (for QP formulation)
- **Validates**: Policy actions (via CBF-QP)
- **Defined by**: Environment (forbidden_circle/safety.py)

### Validation Rules

- B(x) ≤ 0 must imply x ∈ 𝒮 (containment property)
- grad_B must be Lipschitz continuous
- alpha > 0 for forward invariance

---

## 6. Policy

**Description**: Mapping π: 𝒫(𝒳) ∪ CredalSet → 𝒫(𝒰) from belief (or credal set) to action distribution, optimizing risk-Bellman operator.

### Attributes

| Attribute     | Type                      | Description                                   | Constraints                  |
|---------------|---------------------------|-----------------------------------------------|------------------------------|
| `policy_type` | `str`                     | Algorithm type (VI, Perseus, actor-critic)    | Enum-like string             |
| `action_dim`  | `int`                     | Dimensionality of action space 𝒰              | ≥ 1                          |
| `risk_alpha`  | `float`                   | CVaR tail parameter α                         | ∈ (0, 1), typical 0.05-0.20  |
| `discount`    | `float`                   | Discount factor γ                             | ∈ (0, 1]                     |

### Operations

- `select_action(belief: Belief | CredalSet) → np.ndarray`: Choose action maximizing risk-Bellman value
- `update(episode: Episode) → None`: Learn from trajectory (if trainable policy)
- `evaluate_value(belief: Belief | CredalSet) → float`: Compute V(β) under current policy

### Relationships

- **Consumes**: Belief or CredalSet
- **Produces**: Action (np.ndarray)
- **Used by**: Agent.act()
- **Uses**: RiskBellman (for value computation)

### Validation Rules

- action_dim must match environment's action space
- risk_alpha ∈ (0, 1) for valid CVaR
- Policy must handle both Belief and CredalSet inputs

---

## 7. Agent

**Description**: Main control loop integrating belief tracking, risk assessment, safety filtering, and semantic status assignment.

### Attributes

| Attribute        | Type              | Description                                     | Constraints                  |
|------------------|-------------------|-------------------------------------------------|------------------------------|
| `belief`         | `Belief \| CredalSet` | Current belief state                             | Updated each timestep        |
| `policy`         | `Policy`          | Action selection strategy                       | Stateful                     |
| `safety_filter`  | `SafetyFilter`    | CBF-QP filter for action correction             | Optional (can be None)       |
| `source_trusts`  | `dict[str, SourceTrust]` | Mapping source_id → trust params                 | Updated dynamically          |
| `config`         | `Configuration`   | Hyperparameters                                 | Immutable during episode     |

### Operations

- `act(observation, messages) → dict`: Process obs+messages → belief update → select action → apply safety filter
- `learn(episode: Episode) → None`: Update policy and source trusts from trajectory
- `calibrate_thresholds(episodes: list[Episode]) → dict`: Auto-tune τ, τ' for ECE ≤ 0.05

### Relationships

- **Owns**: Belief, Policy, SafetyFilter, SourceTrust instances
- **Produces**: Episode (logged trajectory)
- **Uses**: All core entities in act/learn loop

### Validation Rules

- belief must always be valid (never None after first observation)
- safety_filter activations must be logged (SC-002: ≥1% in 100 episodes)
- source_trusts keys must match message.source values

---

## 8. Episode

**Description**: Logged sequence of (belief_t, action_t, reward_t, observation_t+1, messages_t, safety_diag_t, evi_t, statuses_t) for one trial.

### Attributes

| Attribute       | Type                        | Description                                  | Constraints                  |
|-----------------|-----------------------------|----------------------------------------------|------------------------------|
| `episode_id`    | `int`                       | Unique identifier                            | Auto-increment               |
| `timesteps`     | `int`                       | Length of episode                            | > 0                          |
| `states`        | `list[np.ndarray]`          | True states (if known, for evaluation)       | Optional                     |
| `beliefs`       | `list[Belief \| CredalSet]` | Belief states at each timestep               | len = timesteps + 1          |
| `actions`       | `list[np.ndarray]`          | Actions executed                             | len = timesteps              |
| `rewards`       | `list[float]`               | Immediate rewards                            | len = timesteps              |
| `observations`  | `list[Any]`                 | Observations received                        | len = timesteps              |
| `messages`      | `list[list[Message]]`       | Messages received per timestep               | len = timesteps              |
| `safety_diags`  | `list[dict]`                | Safety filter diagnostics                    | len = timesteps              |
| `evis`          | `list[float]`               | Expected value of information                | len = timesteps              |
| `statuses`      | `list[dict[str, BelnapValue]]` | Claim statuses per timestep                  | len = timesteps              |
| `metadata`      | `dict`                      | Config, seed, timestamps                     | JSON-serializable            |

### Operations

- `to_jsonl(filepath: str) → None`: Serialize episode to JSONL format
- `from_jsonl(filepath: str) → Episode`: Deserialize from JSONL
- `compute_return(discount: float) → float`: Calculate discounted return
- `compute_cvar(alpha: float) → float`: Calculate CVaR of returns

### Relationships

- **Produced by**: Agent.act() loop
- **Consumed by**: Agent.learn(), Reports (calibration, risk, safety)
- **Logged to**: runs/*.jsonl

### Validation Rules

- All list lengths consistent with timesteps
- safety_diags must include "filter_applied" boolean key
- metadata must contain "seed", "config_hash", "timestamp"

---

## 9. Configuration

**Description**: YAML-serialized hyperparameters controlling agent behavior.

### Attributes

| Attribute            | Type    | Description                              | Default       | Constraints        |
|----------------------|---------|------------------------------------------|---------------|--------------------|
| `seed`               | `int`   | Random seed for reproducibility          | 42            | ≥ 0                |
| `discount`           | `float` | Discount factor γ                        | 0.98          | ∈ (0, 1]           |
| `risk.alpha`         | `float` | CVaR tail parameter                      | 0.1           | ∈ (0, 1)           |
| `safety.cbf_enabled` | `bool`  | Enable CBF-QP safety filter              | true          | -                  |
| `safety.qp.max_iter` | `int`   | OSQP max iterations                      | 50            | > 0                |
| `safety.qp.slack`    | `float` | Infeasibility slack penalty              | 1e-3          | ≥ 0                |
| `query.cost`         | `float` | Cost of query action                     | 0.2           | ≥ 0                |
| `query.delta_star`   | `float` | EVI threshold for triggering query       | 0.15          | ≥ 0                |
| `belief.particles`   | `int`   | Number of particles N                    | 5000          | ≥ 1000             |
| `belief.resample_threshold` | `float` | ESS fraction for resampling              | 0.5           | ∈ (0, 1]           |
| `credal.K`           | `int`   | Ensemble size for credal sets            | 5             | 2 ≤ K ≤ 20         |
| `credal.trust_init`  | `float` | Initial source reliability r_s           | 0.7           | ∈ (0, 1)           |
| `thresholds.auto`    | `bool`  | Auto-calibrate τ, τ'                     | true          | -                  |
| `thresholds.ece_target` | `float` | Target ECE for calibration               | 0.05          | ∈ (0, 1)           |

### Operations

- `from_yaml(filepath: str) → Configuration`: Load from YAML file
- `to_yaml(filepath: str) → None`: Save to YAML file
- `validate() → bool`: Check all constraints satisfied
- `hash() → str`: Compute SHA256 hash for metadata

### Relationships

- **Consumed by**: Agent (at initialization)
- **Logged in**: Episode.metadata (as hash)
- **Stored as**: configs/*.yaml

### Validation Rules

- All numeric parameters within specified ranges
- File paths must be valid and writable (for logging/reports)

---

## Entity Relationship Diagram

```
Configuration
    |
    v
  Agent ──────┬──> Policy ──> Action
    |         |
    |         └──> SafetyFilter (CBF-QP)
    |                  |
    v                  v
  Belief ────> CredalSet (if v=⊤)
    ^                  |
    |                  v
Message + SourceTrust  |
                       |
                       v
                    Episode
                       |
                       v
              Reports (calibration, risk, safety, credal)
```

**Key Data Flows:**
1. Observation → Belief update (via G kernel)
2. Message + SourceTrust → Belief or CredalSet (via M multiplier)
3. Belief/CredalSet → Policy → Action
4. Action → SafetyFilter → Safe Action
5. Episode → Reports (calibration, risk metrics, safety traces)

---

## Schema Validation Checklist

Before implementation:
- [ ] All entities have clear attribute types (no ambiguous "Any")
- [ ] Constraints are testable (e.g., "∈ [0,1]" → `assert 0 <= value <= 1`)
- [ ] Relationships are bidirectional where needed (Belief ↔ CredalSet)
- [ ] Operations have clear input/output signatures
- [ ] Validation rules map to unit tests (e.g., FR-002 → test_belief.py::test_commutativity)

**Data Model Status**: ✅ **COMPLETE** - Ready for implementation in `robust_semantic_agent/core/`
