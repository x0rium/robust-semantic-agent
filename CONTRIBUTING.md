# Contributing to Robust Semantic Agent

Thank you for your interest in contributing to Robust Semantic Agent!

**⚠️ Note:** This is a research prototype for educational and academic use. Contributions should focus on research, algorithm improvements, and educational value rather than production deployment features.

This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Safety-Critical Considerations](#safety-critical-considerations)

---

## Code of Conduct

This project follows a professional and respectful code of conduct:

- **Be respectful** - Disagreements are fine, personal attacks are not
- **Be constructive** - Focus on improving the project
- **Be collaborative** - This is a team effort
- **Be safety-conscious** - This system has formal safety guarantees that must be preserved

---

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- Familiarity with POMDP, control theory, or formal methods (helpful but not required)

### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/robust-semantic-agent.git
cd robust-semantic-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install pytest pytest-cov pytest-benchmark black ruff mypy

# Run tests to verify setup
make test
```

### Project Structure

```
robust-semantic-agent/
├── robust_semantic_agent/  # Source code
│   ├── core/              # Core algorithms (belief, semantics, etc.)
│   ├── safety/            # Safety filters (CBF-QP)
│   ├── policy/            # Agent and planner
│   ├── envs/              # Environments
│   └── reports/           # Metrics and visualization
├── tests/                 # Test suite (99/99 passing)
├── docs/                  # Documentation
└── configs/               # Configuration files
```

---

## Development Workflow

### 1. Create a Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name

# Or bugfix branch
git checkout -b bugfix/issue-123
```

### 2. Make Changes

- Keep commits small and focused
- Write descriptive commit messages
- Add tests for new features
- Update documentation as needed

### 3. Commit Guidelines

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `test`: Adding or updating tests
- `refactor`: Code refactoring (no behavior change)
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**
```bash
git commit -m "feat(belief): add entropy-based resampling threshold"
git commit -m "fix(safety): handle edge case in CBF-QP solver"
git commit -m "docs(theory): update section 12 with new results"
```

---

## Testing

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
python -m pytest tests/unit/test_belief.py -v

# Performance tests
python -m pytest tests/performance/ --benchmark-only
```

### Writing Tests

**All code changes must include tests.**

#### Test Structure

```python
"""
Test Module: test_your_feature.py
Feature: your-feature
Task: TXXX (if applicable)

Tests for YourFeature functionality.
"""

import pytest
import numpy as np


class TestYourFeature:
    """Test suite for YourFeature."""

    def test_basic_functionality(self):
        """
        Test basic functionality of YourFeature.

        Scenario: [Describe scenario]
        Expected: [Expected result]
        """
        # Arrange
        config = ...

        # Act
        result = your_feature(...)

        # Assert
        assert result == expected

    def test_edge_case(self):
        """Test edge case handling."""
        with pytest.raises(ValueError):
            your_feature(invalid_input)
```

#### Test Requirements

- **Unit tests**: Test individual functions/methods in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark critical paths (optional)
- **Edge cases**: Test None, NaN, Inf, empty inputs, etc.

#### Success Criteria

All changes must:
- [ ] Pass all existing tests (99/99)
- [ ] Add tests for new functionality (aim for >80% coverage)
- [ ] Not degrade performance by more than 10% (run benchmarks)
- [ ] Maintain safety guarantees (zero violations)

---

## Code Style

### Python Style Guide

We follow **PEP 8** with these tools:

```bash
# Format code
black robust_semantic_agent/ tests/

# Run linter
ruff check robust_semantic_agent/ tests/

# Type checking (optional but recommended)
mypy robust_semantic_agent/
```

### Style Requirements

- **Line length**: 100 characters max
- **Type hints**: Required for all functions
- **Docstrings**: Required for all public functions/classes
- **Imports**: Sorted (stdlib, third-party, local)

**Example:**

```python
import logging
from typing import Optional

import numpy as np

from ..core.belief import Belief


def compute_evi(
    belief: Belief,
    value_fn: callable,
    obs_noise: float,
    n_samples: int = 50
) -> float:
    """
    Compute Expected Value of Information (EVI).

    Args:
        belief: Current belief distribution
        value_fn: Value function V(belief) → R
        obs_noise: Observation noise std deviation
        n_samples: Number of Monte Carlo samples

    Returns:
        EVI value (can be negative)

    Raises:
        ValueError: If n_samples < 1 or obs_noise <= 0

    References:
        - docs/theory.md §5: Query action
        - SC-006: EVI trigger correctness
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be ≥1, got {n_samples}")

    # Implementation...
    return evi_value
```

---

## Documentation

### Documentation Requirements

All changes should update relevant documentation:

- **Code**: Docstrings for all public APIs
- **README**: Update if adding major features
- **docs/theory.md**: Update if changing algorithms/theory
- **CHANGELOG.md**: Add entry for all changes

### Documentation Style

- **Clear and concise**: Avoid jargon where possible
- **Examples**: Provide code examples for complex features
- **Mathematics**: Use LaTeX for equations in docs/theory.md
- **References**: Cite relevant papers/sections

---

## Pull Request Process

### Before Submitting

1. **Run all checks:**
   ```bash
   make test        # All tests pass
   make lint        # No linting errors
   make type-check  # No type errors (optional)
   ```

2. **Update documentation:**
   - Add/update docstrings
   - Update README if needed
   - Update CHANGELOG.md

3. **Verify no regressions:**
   - Performance benchmarks (should be within 10% of baseline)
   - Safety guarantees maintained (zero violations)
   - Test coverage >80%

### Submitting Pull Request

1. **Push branch:**
   ```bash
   git push origin feature/your-feature
   ```

2. **Create PR** on GitHub with template:

   **Title**: `<type>: <description>`

   **Description:**
   ```markdown
   ## Summary
   Brief description of changes

   ## Motivation
   Why is this change needed?

   ## Changes Made
   - Added X
   - Fixed Y
   - Updated Z

   ## Testing
   - [ ] All tests pass (99/99)
   - [ ] Added tests for new functionality
   - [ ] Performance within 10% of baseline
   - [ ] Safety guarantees maintained

   ## Documentation
   - [ ] Updated docstrings
   - [ ] Updated README (if applicable)
   - [ ] Updated CHANGELOG.md
   - [ ] Updated docs/theory.md (if applicable)

   ## Related Issues
   Fixes #123
   ```

3. **Wait for review**
   - Address reviewer comments
   - Keep PR focused (avoid scope creep)
   - Be responsive to feedback

### Review Criteria

PRs will be reviewed for:
- ✅ Correctness (tests pass, logic sound)
- ✅ Safety (no violations, proper error handling)
- ✅ Performance (no significant degradation)
- ✅ Code quality (style, type hints, docstrings)
- ✅ Documentation (clear, complete)
- ✅ Test coverage (>80%)

---

## Safety-Critical Considerations

**This system implements formal safety guarantees. Special care must be taken when modifying safety-critical components.**

### Safety-Critical Components

- `safety/cbf.py` - Control Barrier Functions
- `policy/agent.py` - Main control loop (esp. safety filter integration)
- `core/belief.py` - Belief tracking (affects state estimation)

### Safety Requirements

**All changes to safety-critical code must:**

1. **Preserve zero violations**
   - Run `test_navigation.py::test_100_episode_navigation_zero_violations`
   - Verify 0 violations across 100 episodes

2. **Maintain emergency stop protocol**
   - CBF-QP failure → zero action
   - System continues operation (no crashes)
   - Full error logging

3. **Validate inputs**
   - Check for None, NaN, Inf
   - Reject invalid observations
   - Fail-fast on configuration errors

4. **Add production monitoring**
   - Expose relevant metrics in `info` dict
   - Log critical events (errors, violations)
   - Set appropriate alert thresholds

### Testing Safety-Critical Changes

```bash
# Run safety tests
python -m pytest tests/integration/test_navigation.py -v

# Run with high particle count for robustness
python -m pytest tests/integration/test_navigation.py -v --particles=10000

# Check production verification
python exploration/production_verification.py
```

### Documentation for Safety-Critical Changes

- **Update docs/theory.md** (especially §4, §6, §13)
- **Add to PRODUCTION_READY.md** if deployment-relevant
- **Document in AUDIT_REPORT.md** if fixing safety bug

---

## Questions?

- **Documentation**: See [docs/theory.md](docs/theory.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/robust-semantic-agent/issues)
- **Discussion**: Open a discussion on GitHub

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
