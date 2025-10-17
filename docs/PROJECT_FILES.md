# Project Files Guide

Quick reference for all important files in the project.

## Root Directory Files

### Essential Documentation (for GitHub)

**README.md** - Main project documentation
- Overview and features
- Quick start guide
- Installation instructions
- Performance benchmarks
- Safety considerations
- Citation guide

**CHANGELOG.md** - Version history
- All changes in v1.0.0
- Feature additions
- Bug fixes
- Performance improvements

**CONTRIBUTING.md** - Contribution guidelines
- Development workflow
- Testing requirements
- Code style guide
- Safety-critical considerations
- Pull request process

**LICENSE** - MIT License
- Copyright notice
- Permission terms

### Technical Documentation

**PRODUCTION_READY.md** - Deployment guide
- Production improvements implemented
- Configuration management
- Input validation
- Error handling
- Performance benchmarks
- Deployment checklist

**AUDIT_REPORT.md** - Verification report
- All 10 bug fixes audited
- Theoretical correctness verified
- No hardcoded values (except defaults)
- Test results documented

**TESTING.md** - Test suite documentation
- Test structure (99/99 tests)
- Running tests
- Test categories (unit, integration, performance)
- Coverage requirements

**CLAUDE.md** - AI assistant guidelines
- For Claude Code and other AI developers
- Project overview
- Architecture details
- Development rules

## docs/ Directory

**docs/theory.md** - Formal mathematical specification (724 lines)
- Mathematical formulation
- Implementation mapping (theory → code)
- Success criteria with results
- Production deployment considerations
- System architecture diagram
- Verification appendix

**docs/THEORY_IMPROVEMENTS.md** - Theory documentation changelog
- Summary of theory.md improvements
- Added sections (11-13, Appendix D)
- Quantitative improvements
- Impact assessment

**docs/verified-apis.md** - API verification
- External dependencies verified
- Version information
- Usage examples

**docs/gotchas.md** - Known issues and workarounds
- Edge cases
- Platform-specific issues

## .github/ Directory

**.github/workflows/tests.yml** - CI/CD pipeline
- Automated tests on push/PR
- Ubuntu + macOS
- Python 3.11 + 3.12

**.github/ISSUE_TEMPLATE/** - Issue templates
- bug_report.md
- feature_request.md

**.github/pull_request_template.md** - PR template

## Configuration Files

**pyproject.toml** - Python project metadata
- Dependencies
- Package information
- Build configuration

**pytest.ini** - Test configuration
- Test discovery
- Coverage settings
- Markers

**Makefile** - Build automation
- Common tasks (test, lint, format)

**.gitignore** - Git ignore rules
- Temporary files
- Python cache
- IDE settings

## configs/ Directory

**configs/default.yaml** - Default configuration
- All system parameters
- Belief tracking settings
- Safety parameters
- Query action settings
- Logging configuration

## Quick Navigation

**For users:**
1. Start with README.md
2. Installation → Quick Start
3. Configuration → configs/default.yaml

**For developers:**
1. Read CONTRIBUTING.md
2. Setup environment (README → Installation)
3. Run tests (TESTING.md)
4. Code style (CONTRIBUTING → Code Style)

**For deployment:**
1. PRODUCTION_READY.md
2. configs/default.yaml
3. Performance benchmarks (README or PRODUCTION_READY)

**For researchers:**
1. docs/theory.md (full specification)
2. AUDIT_REPORT.md (verification)
3. Test results (docs/theory.md §12)

## File Count Summary

| Type | Count | Purpose |
|------|-------|---------|
| Root .md files | 7 | Essential documentation |
| docs/ files | 4 | Technical specifications |
| .github/ files | 4 | GitHub templates |
| Config files | 4 | Build/test configuration |
| **Total key files** | **19** | All essential documentation |

## Deleted Files (cleaned up)

The following files were removed as redundant or temporary:
- ~~FINAL_SUMMARY.md~~ (duplicated CHANGELOG)
- ~~PROJECT_STATUS.md~~ (outdated)
- ~~RUN_INTERPRETATION.md~~ (temporary)
- ~~VERIFICATION_REPORT.md~~ (duplicated AUDIT_REPORT)
- ~~VERIFICATION_SUMMARY.md~~ (duplicated AUDIT_REPORT)
- ~~GITHUB_RELEASE.md~~ (temporary, for release creation)
- ~~GITHUB_PUBLICATION_GUIDE.md~~ (temporary, instructions completed)

All essential information preserved in remaining files.
