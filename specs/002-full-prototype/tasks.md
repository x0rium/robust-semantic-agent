# Tasks: Robust Semantic Agent Full Prototype

**Input**: Design documents from `/specs/002-full-prototype/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Tests**: INCLUDED - This project follows TDD (Test-Driven Development) as mandated by Constitution Principle V (NON-NEGOTIABLE). All tests must be written FIRST, verified to FAIL, then implementation follows.

**Organization**: Tasks grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `robust_semantic_agent/`, `tests/` at repository root
- Paths below use absolute module notation

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create project directory structure (robust_semantic_agent/, tests/, configs/, docs/, exploration/)
- [X] T002 Create pyproject.toml with dependencies (NumPy 1.24+, SciPy 1.10+, cvxpy 1.3+, pytest 7.0+, ruff, black, mypy, matplotlib 3.7+, PyYAML 6.0+)
- [X] T003 [P] Create Makefile with targets (install, test, lint, report, clean)
- [X] T004 [P] Create .gitignore for Python project (venv/, __pycache__/, *.pyc, runs/, reports/, .pytest_cache/)
- [X] T005 [P] Create README.md with project overview and quickstart instructions
- [X] T006 [P] Initialize all __init__.py files in package structure (robust_semantic_agent/, core/, risk/, safety/, policy/, envs/, reports/, cli/)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T007 Create exploration/001_particle_filter.py MWE demonstrating log-space belief tracking with systematic resampling
- [X] T008 [P] Create exploration/002_cvar.py MWE computing CVaR@α on Gaussian/uniform distributions
- [X] T009 [P] Create exploration/003_qp_solver.py MWE solving CBF-QP with cvxpy OSQP solver
- [X] T010 [P] Create exploration/004_belnap.py MWE implementing 4-valued logic operations
- [X] T011 Create exploration/003_tools.md documenting MCP/agent verification trace for all explorations
- [X] T012 Update docs/verified-apis.md with NumPy/SciPy/cvxpy dependency verification (versions, examples, status)
- [X] T013 Create docs/gotchas.md for documenting theory.md divergences (initially empty placeholder)
- [X] T014 Create configs/default.yaml with all hyperparameters (seed: 42, discount: 0.98, risk.alpha: 0.1, safety.cbf: true, belief.particles: 5000, credal.K: 5, thresholds.auto: true)
- [X] T015 [P] Create configs/risk.yaml with CVaR-specific parameters
- [X] T016 [P] Create configs/safety.yaml with CBF-QP parameters (qp.max_iter: 50, qp.slack: 1e-3)
- [X] T017 [P] Create configs/thresholds.yaml with semantic layer calibration params (auto: true, ece_target: 0.05)

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Safe Navigation with Risk-Aware Decision Making (Priority: P1) 🎯 MVP

**Goal**: Core demonstration of Risk + Safety + Semantics integration via 2D navigation with forbidden zone

**Independent Test**: Run 100 CBF-enabled episodes, verify zero violations and ≥80% goal success rate

### Tests for User Story 1 ⚠️

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T018 [P] [US1] Create test_belief.py with test_update_obs verifying observation likelihood weighting in tests/unit/test_belief.py
- [X] T019 [P] [US1] Create test_belief.py with test_commutativity verifying TV distance ≤ 1e-6 (FR-002) in tests/unit/test_belief.py
- [X] T020 [P] [US1] Create test_belief.py with test_eff_sample_size and test_resample in tests/unit/test_belief.py
- [X] T021 [P] [US1] Create test_risk.py with test_cvar_gaussian_analytical (SC-005) in tests/unit/test_risk.py
- [X] T022 [P] [US1] Create test_risk.py with test_cvar_uniform_analytical in tests/unit/test_risk.py
- [X] T023 [P] [US1] Create test_risk.py with test_cvar_monotonicity (α1 < α2 → CVaR1 < CVaR2) in tests/unit/test_risk.py
- [X] T024 [P] [US1] Create test_safety.py with test_cbf_supermartingale (𝔼[B(x+)|x,u] ≤ B(x)) in tests/unit/test_safety.py
- [X] T025 [P] [US1] Create test_safety.py with test_qp_solve_basic and test_qp_infeasibility_slack in tests/unit/test_safety.py
- [X] T026 [P] [US1] Create test_navigation.py integration test for 100-episode scenario (SC-001, SC-002) in tests/integration/test_navigation.py

### Implementation for User Story 1

- [X] T027 [P] [US1] Implement BelnapValue enum (4 values: NEITHER=00, TRUE=01, FALSE=10, BOTH=11) in robust_semantic_agent/core/semantics.py
- [X] T028 [P] [US1] Implement Belief class with particles, log_weights, update_obs(), resample() in robust_semantic_agent/core/belief.py
- [X] T029 [US1] Implement Belief.apply_message() handling ⊥,t,f values (not ⊤ yet) in robust_semantic_agent/core/belief.py
- [X] T030 [P] [US1] Implement Message dataclass (claim, source, value, A_c) in robust_semantic_agent/core/messages.py
- [X] T031 [P] [US1] Implement SourceTrust class (r_s, logit(), update()) in robust_semantic_agent/core/messages.py
- [X] T032 [P] [US1] Implement cvar(values, alpha) function with sort-and-average in robust_semantic_agent/risk/cvar.py
- [X] T033 [P] [US1] Implement RiskBellman class with backup() method in robust_semantic_agent/risk/cvar.py
- [X] T034 [P] [US1] Implement BarrierFunction class (B, grad_B, evaluate) in robust_semantic_agent/envs/forbidden_circle/safety.py
- [X] T035 [US1] Implement SafetyFilter class with QP solver (project(), solve_qp()) in robust_semantic_agent/safety/cbf.py
- [X] T036 [US1] Implement ForbiddenCircleEnv (dynamics K, observations G, render) in robust_semantic_agent/envs/forbidden_circle/env.py
- [X] T037 [US1] Implement simple Policy with select_action() (greedy or random baseline) in robust_semantic_agent/policy/planner.py
- [X] T038 [US1] Implement Agent class (act(), belief tracking, safety integration) in robust_semantic_agent/policy/agent.py
- [X] T039 [US1] Implement Configuration class (from_yaml(), validate(), hash()) in robust_semantic_agent/core/config.py (create new file)
- [X] T040 [US1] Implement Episode class (to_jsonl(), compute_return()) in robust_semantic_agent/core/episode.py (create new file)
- [X] T041 [US1] Implement cli/rollout.py script with argument parsing and episode logging
- [X] T042 [US1] Run test_navigation.py integration test and verify SC-001 (zero violations) and SC-002 (≥1% filter activations)

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Contradiction Handling via Credal Sets (Priority: P2)

**Goal**: Validate semantic layer's contradiction handling (v=⊤) with credal sets

**Independent Test**: Configure gossip source emitting v=⊤, verify credal set expansion to K posteriors and coherent decisions

### Tests for User Story 2 ⚠️

- [X] T043 [P] [US2] Create test_semantics.py with test_belnap_bilattice_properties (12 properties: commutativity, associativity, absorption, involution, distributivity) in tests/unit/test_semantics.py
- [X] T044 [P] [US2] Create test_semantics.py with test_status_assignment (τ=0.7, τ'=0.3 thresholds) in tests/unit/test_semantics.py
- [X] T045 [P] [US2] Create test_credal.py with test_lower_expectation_monotonicity (SC-004) in tests/unit/test_credal.py
- [X] T046 [P] [US2] Create test_credal.py with test_credal_set_creation and test_posterior_diversity in tests/unit/test_credal.py
- [X] T047 [P] [US2] Create test_contradictions.py integration test for gossip source scenario in tests/integration/test_contradictions.py

### Implementation for User Story 2

- [X] T048 [P] [US2] Implement Belnap operations (and_t, or_t, not_t, consensus, gullibility) in robust_semantic_agent/core/semantics.py
- [X] T049 [P] [US2] Implement status(s_c, s_bar_c, tau, tau_prime) function in robust_semantic_agent/core/semantics.py
- [X] T050 [P] [US2] Implement CredalSet class (posteriors, lower_expectation(), add_posterior()) in robust_semantic_agent/core/credal.py
- [X] T051 [US2] Extend Belief.apply_message() to handle v=⊤ by creating CredalSet with K extreme posteriors in robust_semantic_agent/core/belief.py
- [X] T052 [US2] Extend Message.multiplier() to return logit interval Λ_s for v=⊤ in robust_semantic_agent/core/messages.py
- [X] T053 [US2] Extend Policy.select_action() to accept CredalSet and compute lower expectation in robust_semantic_agent/policy/planner.py
- [X] T054 [US2] Add gossip source to ForbiddenCircleEnv emitting v=⊤ messages in robust_semantic_agent/envs/forbidden_circle/env.py
- [X] T055 [US2] Update Agent.act() to handle Belief | CredalSet types in robust_semantic_agent/policy/agent.py
- [X] T056 [US2] Run test_contradictions.py and verify credal set creation, coherence, and safety maintenance

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - Active Information Acquisition via Query Action (Priority: P3)

**Goal**: Demonstrate EVI-based query action with entropy reduction and regret improvement

**Independent Test**: Run episodes with query enabled/disabled, verify EVI ≥ Δ*, entropy reduction ≥20%, regret reduction ≥10%

### Tests for User Story 3 ⚠️

- [X] T057 [P] [US3] Create test_query.py with test_evi_computation and test_should_query_threshold in tests/unit/test_query.py
- [X] T058 [P] [US3] Create test_query.py with test_entropy_reduction and test_value_improvement in tests/unit/test_query.py
- [X] T059 [P] [US3] Create test_query_action.py integration test for query ROI scenario (SC-006, SC-007) in tests/integration/test_query_action.py

### Implementation for User Story 3

- [X] T060 [P] [US3] Implement evi(belief, action_set, value_fn) function in robust_semantic_agent/core/query.py
- [X] T061 [P] [US3] Implement should_query(evi, delta_star) function in robust_semantic_agent/core/query.py
- [X] T062 [P] [US3] Add Belief.entropy() method computing Shannon entropy in robust_semantic_agent/core/belief.py
- [X] T063 [US3] Extend Agent.act() to compute EVI and trigger query action when EVI ≥ Δ* in robust_semantic_agent/policy/agent.py
- [X] T064 [US3] Extend ForbiddenCircleEnv to support query action (return additional observation at cost c) in robust_semantic_agent/envs/forbidden_circle/env.py
- [X] T065 [US3] Update cli/rollout.py to support --enable-query flag and log EVI/entropy metrics
- [X] T066 [US3] Run test_query_action.py with query enabled vs disabled and verify SC-006 and SC-007

**Checkpoint**: All user stories should now be independently functional

---

## Phase 6: User Story 4 - Semantic Status Calibration and Reporting (Priority: P4)

**Goal**: Auto-calibrate τ, τ' thresholds to achieve ECE ≤ 0.05 with cost-matrix penalties

**Independent Test**: Run calibration on test episodes, verify ECE ≤ 0.05 and reliability diagrams show alignment

### Tests for User Story 4 ⚠️

- [X] T067 [P] [US4] Create test_calibration.py with test_ece_computation and test_brier_score in tests/unit/test_calibration.py
- [X] T068 [P] [US4] Create test_calibration.py with test_threshold_tuning and test_cost_matrix_penalties in tests/unit/test_calibration.py

### Implementation for User Story 4

- [X] T069 [P] [US4] Implement calibrate_thresholds(episodes, cost_matrix) in robust_semantic_agent/core/semantics.py
- [X] T070 [P] [US4] Implement compute_ece(predictions, outcomes) in robust_semantic_agent/reports/calibration.py
- [X] T071 [P] [US4] Implement compute_brier(predictions, outcomes) in robust_semantic_agent/reports/calibration.py
- [X] T072 [P] [US4] Implement generate_reliability_diagram(data, output_path) in robust_semantic_agent/reports/calibration.py
- [X] T073 [P] [US4] Implement generate_roc_curve(data, output_path) in robust_semantic_agent/reports/calibration.py
- [X] T074 [US4] Implement cli/calibrate.py script with --target-ece and --output flags
- [X] T075 [US4] Run calibration on 500 episodes and verify SC-008 (ECE ≤ 0.05)

---

## Phase 7: User Story 5 - Performance Monitoring and Risk/Safety Reports (Priority: P5)

**Goal**: Generate comprehensive reports validating all acceptance criteria

**Independent Test**: Run evaluation on logged episodes, verify all report artifacts generated with correct metrics

### Implementation for User Story 5

- [X] T076 [P] [US5] Implement generate_cvar_curves(episodes, alphas, output_path) in robust_semantic_agent/reports/risk.py
- [X] T077 [P] [US5] Implement generate_tail_distributions(episodes, output_path) in robust_semantic_agent/reports/risk.py
- [X] T078 [P] [US5] Implement generate_barrier_traces(episodes, output_path) in robust_semantic_agent/reports/safety.py
- [X] T079 [P] [US5] Implement compute_violation_rates(episodes) in robust_semantic_agent/reports/safety.py
- [X] T080 [P] [US5] Implement generate_posterior_ensemble_plot(credal_sets, output_path) in robust_semantic_agent/reports/credal.py
- [X] T081 [US5] Implement cli/evaluate.py script with --runs-dir and --output flags
- [X] T082 [US5] Run evaluation and verify all reports generated (calibration/, risk/, safety/, credal/ subdirectories)
- [X] T083 [US5] Verify SC-010 (risk-averse CVaR ≥ baseline with 95% confidence) from risk report

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] T084 [P] Update README.md with actual installation instructions and CLI examples from quickstart.md
- [ ] T085 [P] Create docs/examples/ directory with Jupyter notebook demonstrating full workflow
- [ ] T086 [P] Add type hints to all public APIs and run mypy validation (SC-013)
- [ ] T087 [P] Run ruff and black on entire codebase (SC-013)
- [ ] T088 [P] Add docstrings with theory.md theorem references (e.g., "implements Theorem 2: Doob supermartingale") to all core modules
- [ ] T089 Run pytest with coverage and verify ≥80% for core modules (SC-012)
- [ ] T090 [P] Create cli/train.py script for policy learning (optional actor-critic or VI/Perseus)
- [ ] T091 Profile performance with cProfile and verify 30+ Hz @ 10k particles (SC-009)
- [ ] T092 Final integration test: Run quickstart.md Step 1-6 end-to-end and verify all outputs

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-7)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3 → P4 → P5)
- **Polish (Phase 8)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Extends US1 components but independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - Extends US1 components but independently testable
- **User Story 4 (P4)**: Depends on US1-US3 (needs episodes with claims/statuses for calibration)
- **User Story 5 (P5)**: Depends on US1-US4 (needs full episodes with all metrics for reporting)

### Within Each User Story

- Tests MUST be written and FAIL before implementation (TDD Principle V)
- [P] tasks within a story can run in parallel
- Non-[P] tasks must run sequentially after their dependencies
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel (T003-T006)
- All Foundational tasks marked [P] can run in parallel within groups (T008-T010, T015-T017)
- All tests for a user story marked [P] can run in parallel (e.g., T018-T026 for US1)
- Models/modules within a story marked [P] can run in parallel (e.g., T027-T033 for US1)
- User Stories 1-3 can be worked on in parallel by different team members after Foundational phase
- User Stories 4-5 must wait for US1-3 completion

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together:
Task: T018 [P] [US1] Create test_belief.py with test_update_obs
Task: T019 [P] [US1] Create test_belief.py with test_commutativity
Task: T020 [P] [US1] Create test_belief.py with test_eff_sample_size
Task: T021 [P] [US1] Create test_risk.py with test_cvar_gaussian
Task: T022 [P] [US1] Create test_risk.py with test_cvar_uniform
Task: T023 [P] [US1] Create test_risk.py with test_cvar_monotonicity
Task: T024 [P] [US1] Create test_safety.py with test_cbf_supermartingale
Task: T025 [P] [US1] Create test_safety.py with test_qp_solve
Task: T026 [P] [US1] Create test_navigation.py integration test

# Then launch core modules together:
Task: T027 [P] [US1] Implement BelnapValue enum
Task: T028 [P] [US1] Implement Belief class
Task: T030 [P] [US1] Implement Message dataclass
Task: T031 [P] [US1] Implement SourceTrust class
Task: T032 [P] [US1] Implement cvar() function
Task: T033 [P] [US1] Implement RiskBellman class
Task: T034 [P] [US1] Implement BarrierFunction class

# Sequential tasks depend on previous completions:
Task: T029 [US1] Implement Belief.apply_message() (needs T028, T030, T031)
Task: T035 [US1] Implement SafetyFilter (needs T034)
Task: T036 [US1] Implement ForbiddenCircleEnv (needs T034, T028)
Task: T037 [US1] Implement Policy (needs T028, T032, T033)
Task: T038 [US1] Implement Agent (needs T028, T035, T037)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T006)
2. Complete Phase 2: Foundational (T007-T017) - CRITICAL - blocks all stories
3. Complete Phase 3: User Story 1 (T018-T042)
4. **STOP and VALIDATE**: Test User Story 1 independently via T042
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently (T042) → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently (T056) → Deploy/Demo
4. Add User Story 3 → Test independently (T066) → Deploy/Demo
5. Add User Story 4 → Test independently (T075) → Deploy/Demo
6. Add User Story 5 → Test independently (T082-T083) → Deploy/Demo
7. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together (T001-T017)
2. Once Foundational is done:
   - Developer A: User Story 1 (T018-T042)
   - Developer B: User Story 2 (T043-T056) - can start after US1 core modules ready
   - Developer C: User Story 3 (T057-T066) - can start after US1 core modules ready
3. User Stories 4-5 wait for 1-3 completion
4. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies, can run in parallel
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- **TDD CRITICAL**: Verify tests fail before implementing (Principle V: NON-NEGOTIABLE)
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence

---

## Task Count Summary

**Total Tasks**: 92

**By Phase**:
- Phase 1 (Setup): 6 tasks
- Phase 2 (Foundational): 11 tasks (BLOCKS all user stories)
- Phase 3 (User Story 1): 25 tasks (9 tests + 16 implementation)
- Phase 4 (User Story 2): 14 tasks (5 tests + 9 implementation)
- Phase 5 (User Story 3): 10 tasks (3 tests + 7 implementation)
- Phase 6 (User Story 4): 9 tasks (2 tests + 7 implementation)
- Phase 7 (User Story 5): 8 tasks (8 implementation, tests covered by US4)
- Phase 8 (Polish): 9 tasks

**By User Story**:
- US1 (P1 - Safe Navigation): 25 tasks
- US2 (P2 - Contradictions): 14 tasks
- US3 (P3 - Query Action): 10 tasks
- US4 (P4 - Calibration): 9 tasks
- US5 (P5 - Reporting): 8 tasks

**Parallel Opportunities**: 47 tasks marked [P] can run in parallel (51% parallelizable)

**MVP Scope**: Phase 1 (6) + Phase 2 (11) + Phase 3 (25) = **42 tasks** for minimum viable prototype

**Independent Test Criteria**:
- US1: T042 - Run 100 episodes, zero violations, ≥80% goal success
- US2: T056 - Credal set creation, coherence, safety maintenance
- US3: T066 - Query enabled vs disabled, EVI/entropy/regret metrics
- US4: T075 - ECE ≤ 0.05 on 500 calibration episodes
- US5: T082-T083 - All report artifacts generated, metrics validated

**Format Validation**: ✅ All tasks follow checklist format (`- [ ] [ID] [P?] [Story?] Description with path`)
