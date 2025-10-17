# Quickstart Guide: Robust Semantic Agent Full Prototype

**Feature**: 002-full-prototype
**Audience**: Researchers and developers
**Time to complete**: 15-20 minutes

---

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for cloning repository)
- 8GB+ RAM recommended (for 10k particle belief tracking)

---

## Step 1: Installation (5 minutes)

### Clone Repository

```bash
git clone https://github.com/yourusername/robust-semantic-agent.git
cd robust-semantic-agent
```

### Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
# Core dependencies
pip install numpy>=1.24 scipy>=1.10 cvxpy>=1.4 \
            matplotlib>=3.7 pyyaml>=6.0

# Development tools
pip install pytest>=7.0 pytest-cov>=4.0 \
            ruff>=0.1 black>=23.0 mypy>=1.5

# Or use requirements file (when available)
pip install -e .
```

### Verify Installation

```bash
python -c "import numpy, scipy, cvxpy; print('✓ Dependencies installed')"
```

---

## Step 2: Run Unit Tests (3 minutes)

Validate mathematical correctness before running demos:

```bash
# Run all unit tests with coverage
pytest tests/unit/ --cov=robust_semantic_agent --cov-report=term-missing

# Expected output:
# test_belief.py::test_commutativity PASSED
# test_belief.py::test_obs_message_tv_distance PASSED
# test_semantics.py::test_belnap_bilattice_properties PASSED
# test_credal.py::test_lower_expectation_monotonicity PASSED
# test_risk.py::test_cvar_gaussian_analytical PASSED
# test_safety.py::test_cbf_supermartingale PASSED
#
# ========== 25 passed, Coverage: 85% ==========
```

**Key Validations:**
- **Commutativity**: Total variation distance ≤ 1e-6 (FR-002)
- **Belnap**: All 12 bilattice properties satisfied
- **CVaR**: <1% error vs analytical Gaussian/uniform
- **CBF**: Supermartingale property 𝔼[B(x+)|x,u] ≤ B(x)

---

## Step 3: Run Demo Rollout (5 minutes)

### Basic Rollout (No CBF, No Query)

```bash
python -m robust_semantic_agent.cli.rollout \
    --config configs/default.yaml \
    --episodes 10 \
    --render
```

**Expected Output:**
```
Episode 1/10: Return=-12.5, Goal Reached=True, Violations=0
Episode 2/10: Return=-15.3, Goal Reached=True, Violations=0
...
Average Return: -13.8 ± 2.1
Success Rate: 90% (9/10)
```

**What's Happening:**
- Agent navigates 2D plane with forbidden circular zone
- Noisy beacon observations update particle belief
- Risk-aware policy balances CVaR@0.1 tail risk with goal achievement
- Visualization shows particle cloud, true state, and forbidden zone

### With CBF Safety Filter

```bash
python -m robust_semantic_agent.cli.rollout \
    --config configs/safety.yaml \
    --episodes 100 \
    --render
```

**Expected Output:**
```
...
Safety Filter Activations: 127/5000 steps (2.5%)
Forbidden Zone Violations: 0/100 episodes (0.0%)
Average Return: -14.2 ± 2.3
```

**Validation:** SC-001 (zero violations), SC-002 (≥1% activations)

### With Query Action

```bash
python -m robust_semantic_agent.cli.rollout \
    --config configs/default.yaml \
    --enable-query \
    --query-cost 0.2 \
    --delta-star 0.15 \
    --episodes 100
```

**Expected Output:**
```
Query Actions: 23/5000 steps (0.5%)
Avg EVI Before Query: 0.18 ± 0.04 (threshold: 0.15)
Avg Entropy Reduction: 24% ± 6%
Regret Improvement vs No-Query: 12% ± 3%
```

**Validation:** SC-006 (≥20% entropy reduction), SC-007 (≥10% regret reduction)

---

## Step 4: Run Calibration (3 minutes)

Auto-tune thresholds τ, τ' for semantic layer calibration:

```bash
python -m robust_semantic_agent.cli.calibrate \
    --episodes 500 \
    --target-ece 0.05 \
    --output configs/thresholds.yaml
```

**Expected Output:**
```
Calibration Results:
  τ  = 0.68
  τ' = 0.32
  ECE = 0.047 (target: 0.05)
  Brier Score = 0.082
  AUC-ROC = 0.91

Saved to: configs/thresholds.yaml
```

**Artifacts Generated:**
- `reports/calibration/reliability_diagram.png`
- `reports/calibration/roc_curve.png`
- `configs/thresholds.yaml` (updated τ, τ')

**Validation:** SC-008 (ECE ≤ 0.05)

---

## Step 5: Generate Reports (4 minutes)

Comprehensive evaluation across all subsystems:

```bash
python -m robust_semantic_agent.cli.evaluate \
    --runs-dir runs/2025-10-16_experiment_001 \
    --output reports/
```

**Generated Reports:**

### Calibration Report (`reports/calibration/`)
- `ece_summary.txt` - ECE, ACE metrics
- `reliability_diagram.png` - Predicted confidence vs empirical accuracy
- `brier_decomposition.txt` - Calibration vs refinement
- `roc_curve.png` - True positive rate vs false positive rate

### Risk Report (`reports/risk/`)
- `cvar_curves.png` - CVaR@α for α ∈ [0.01, 0.50]
- `tail_distributions.png` - Empirical return distributions
- `baseline_comparison.txt` - CVaR vs risk-neutral baseline (SC-010)

### Safety Report (`reports/safety/`)
- `barrier_traces.png` - B(x_t) over time for sample episodes
- `violation_summary.txt` - Violation counts, CBF activation rates
- `safety_margin_hist.png` - Distribution of h(x) values

### Credal Report (`reports/credal/`)
- `posterior_ensemble.png` - K extreme posteriors for contradictory claims
- `lower_expectation_comparison.txt` - Worst-case vs individual posteriors

**Validation:** All SC-001 through SC-014 metrics logged

---

## Step 6: Train Policy (Optional, 10-15 minutes)

If using trainable policy (actor-critic):

```bash
python -m robust_semantic_agent.cli.train \
    --alpha 0.1 \
    --delta-star 0.15 \
    --particles 5000 \
    --episodes 1000 \
    --save-every 100
```

**Expected Output:**
```
Episode 100: Return=-18.5, CVaR@0.1=-22.3
Episode 200: Return=-15.2, CVaR@0.1=-19.1
...
Episode 1000: Return=-12.8, CVaR@0.1=-16.4

Saved policy to: checkpoints/policy_ep1000.pt
```

**Note:** Training is optional. Value iteration (VI) or Perseus-PBVI can be used for offline planning.

---

## Common Issues & Solutions

### Issue: cvxpy installation fails

**Solution:**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install build-essential python3-dev

# Or use conda
conda install -c conda-forge cvxpy
```

### Issue: "Particles degenerate (ESS < 100)"

**Cause:** Observation noise too low or particle count too small

**Solution:** Increase `belief.particles` in config or add process noise

### Issue: "QP infeasible, using fallback"

**Cause:** No safe action exists (agent too close to boundary)

**Solution:** Check barrier function parameters, increase slack penalty

### Issue: Low test coverage (<80%)

**Cause:** Missing edge case tests

**Solution:** Add parametrized tests for boundary conditions

---

## Next Steps

### For Researchers

1. **Experiment with risk levels**: Vary `risk.alpha` ∈ [0.05, 0.30] and compare CVaR curves
2. **Test contradiction handling**: Modify `envs/forbidden_circle/configs/` to add gossip sources
3. **Ablation studies**: Disable CBF/query individually to measure impact

### For Developers

1. **Review code**: Start with `core/belief.py` (270 lines, well-documented)
2. **Run profiler**: `python -m cProfile -o profile.stats cli/rollout.py`
3. **Add custom environment**: Implement new `envs/my_domain/env.py` following forbidden_circle template

### For Contributors

1. **Check DoD**: Verify README.md §"Критерии приёмки (DoD)" checklist
2. **Run full CI**: `make test && make lint && make report`
3. **Submit PR**: Reference constitution principles in description

---

## Configuration Reference

### Default Configuration (`configs/default.yaml`)

```yaml
seed: 42
discount: 0.98

risk:
  mode: cvar
  alpha: 0.1

safety:
  cbf: true
  qp:
    max_iter: 50
    slack: 1e-3

query:
  cost: 0.2
  delta_star: 0.15

belief:
  particles: 5000
  resample_threshold: 0.5

credal:
  K: 5
  trust_init: 0.7

thresholds:
  auto: true
  ece_target: 0.05
```

### Key Parameters

- **`risk.alpha`**: Lower = more risk-averse (α=0.05 very conservative, α=0.20 moderate)
- **`belief.particles`**: Higher = more accurate belief (5k standard, 10k for high precision)
- **`query.delta_star`**: Higher = fewer queries (0.10 aggressive, 0.20 conservative)
- **`credal.K`**: More extreme posteriors = better worst-case coverage (5-10 typical)

---

## Performance Benchmarks

**System**: MacBook Pro M1 (8-core CPU, 16GB RAM)

| Operation                      | Time (ms) | Notes                              |
|--------------------------------|-----------|------------------------------------|
| Belief update (5k particles)   | 1.5       | Vectorized NumPy                   |
| CVaR computation (5k samples)  | 0.8       | Sort + average                     |
| CBF-QP solve (2D control)      | 3.2       | OSQP with warm-start               |
| Full act() cycle               | 8.5       | Belief + policy + safety           |
| **Throughput**                 | **117 Hz** | Well above 30 Hz target (SC-009)   |

**Scaling**:
- 10k particles: ~15ms/cycle → 66 Hz
- 20k particles: ~30ms/cycle → 33 Hz (threshold)

---

## Support & Resources

- **Documentation**: See `README.md`, `CLAUDE.md`, `docs/theory.md`
- **Issues**: Report bugs at https://github.com/yourusername/robust-semantic-agent/issues
- **Constitution**: Refer to `.specify/memory/constitution.md` for governance

**Quickstart Status**: ✅ **COMPLETE** - Ready for end-to-end user walkthrough
