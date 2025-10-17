.PHONY: help install install-dev test test-unit test-integration lint format type-check coverage clean run-rollout run-train run-evaluate run-calibrate report

help:
	@echo "Robust Semantic Agent - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install package and dependencies"
	@echo "  make install-dev      Install package with dev dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests with coverage"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make coverage         Generate HTML coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run ruff linter"
	@echo "  make format           Format code with black"
	@echo "  make type-check       Run mypy type checker"
	@echo ""
	@echo "CLI Commands:"
	@echo "  make run-rollout      Run demo rollout (requires config)"
	@echo "  make run-train        Train policy"
	@echo "  make run-evaluate     Generate reports"
	@echo "  make run-calibrate    Auto-tune thresholds"
	@echo ""
	@echo "Reporting:"
	@echo "  make report           Generate all reports (calibration, risk, safety, credal)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove generated files and caches"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ --cov=robust_semantic_agent --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -m unit --cov=robust_semantic_agent --cov-report=term-missing

test-integration:
	pytest tests/integration/ -m integration --cov=robust_semantic_agent --cov-report=term-missing

coverage:
	pytest tests/ --cov=robust_semantic_agent --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	ruff check robust_semantic_agent/ tests/
	@echo "Linting complete"

format:
	black robust_semantic_agent/ tests/
	ruff check --fix robust_semantic_agent/ tests/
	@echo "Formatting complete"

type-check:
	mypy robust_semantic_agent/
	@echo "Type checking complete"

run-rollout:
	python -m robust_semantic_agent.cli.rollout --config configs/default.yaml --episodes 10 --render

run-train:
	python -m robust_semantic_agent.cli.train --config configs/default.yaml --episodes 1000

run-evaluate:
	python -m robust_semantic_agent.cli.evaluate --runs-dir runs/ --output reports/

run-calibrate:
	python -m robust_semantic_agent.cli.calibrate --episodes 500 --target-ece 0.05 --output configs/thresholds.yaml

report: run-evaluate
	@echo "Reports generated in reports/"
	@ls -lh reports/*/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Cleanup complete"
