## Summary
<!-- Brief description of changes (1-2 sentences) -->

## Motivation
<!-- Why is this change needed? What problem does it solve? -->

## Changes Made
<!-- List of specific changes -->
-
-
-

## Type of Change
<!-- Check all that apply -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring (no functional changes)

## Testing
<!-- Describe testing done -->
- [ ] All tests pass (`make test` or `python -m pytest tests/`)
- [ ] Added tests for new functionality
- [ ] Performance within 10% of baseline (if applicable)
- [ ] Safety guarantees maintained (zero violations in test_navigation.py)

**Test Results:**
```
# Paste test output here
```

## Safety-Critical Changes
<!-- Only if modifying safety-critical components -->
- [ ] This PR modifies safety-critical code (safety/, policy/agent.py, core/belief.py)
- [ ] Zero violations verified in 100+ episodes
- [ ] Emergency stop protocol tested
- [ ] Production monitoring updated

## Documentation
- [ ] Updated docstrings for modified functions
- [ ] Updated README.md (if applicable)
- [ ] Updated CHANGELOG.md
- [ ] Updated docs/theory.md (if algorithmic changes)
- [ ] Updated PRODUCTION_READY.md (if deployment-relevant)

## Performance
<!-- Only if performance-related changes -->
**Benchmarks (before → after):**
- Belief update: ___ Hz → ___ Hz
- CBF-QP filter: ___ Hz → ___ Hz
- Full agent.act(): ___ Hz → ___ Hz

## Related Issues
<!-- Link to related issues -->
Fixes #
Relates to #

## Checklist
- [ ] Code follows project style guidelines (black + ruff)
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] No new warnings introduced
- [ ] Changes are backward compatible (or breaking change documented)

## Additional Context
<!-- Add any other context, screenshots, or information -->

---

## For Reviewers
<!-- Specific areas you'd like reviewers to focus on -->
-
-

## Preview
<!-- For documentation/UI changes, add screenshots or examples -->
```python
# Example usage of new feature
```
