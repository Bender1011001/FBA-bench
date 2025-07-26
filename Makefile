# Makefile for FBA-Bench development tasks
# Provides convenient commands for common development workflows

.PHONY: help install install-dev lint format test test-verbose test-property clean check pre-commit-install pre-commit-run type-check security-check docs-coverage

# Default target
help:
	@echo "FBA-Bench Development Commands"
	@echo "=============================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install development dependencies and pre-commit hooks"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  make lint             Run all linting tools (ruff)"
	@echo "  make format           Run code formatters (black, isort)"
	@echo "  make type-check       Run static type checking (mypy)"
	@echo "  make security-check   Run security analysis (bandit)"
	@echo "  make check            Run all code quality checks"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test             Run all tests"
	@echo "  make test-verbose     Run tests with verbose output"
	@echo "  make test-property    Run property-based tests only"
	@echo "  make docs-coverage    Check documentation coverage"
	@echo ""
	@echo "Pre-commit Commands:"
	@echo "  make pre-commit-install  Install pre-commit hooks"
	@echo "  make pre-commit-run      Run pre-commit on all files"
	@echo ""
	@echo "Cleanup Commands:"
	@echo "  make clean            Clean up generated files and caches"

# Installation commands
install:
	@echo "Installing production dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	pip install -e ".[dev]"
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Development environment setup complete!"

# Code quality commands
lint:
	@echo "Running Ruff linter..."
	ruff check .
	@echo "Linting complete!"

format:
	@echo "Running Black formatter..."
	black .
	@echo "Running isort import sorter..."
	isort .
	@echo "Code formatting complete!"

type-check:
	@echo "Running MyPy type checker..."
	mypy fba_bench/ dashboard/
	@echo "Type checking complete!"

security-check:
	@echo "Running Bandit security analysis..."
	bandit -r fba_bench/ dashboard/ -f json -o bandit-report.json || true
	@echo "Security analysis complete! Check bandit-report.json for results."

check: lint type-check
	@echo "All code quality checks passed!"

# Testing commands
test:
	@echo "Running test suite..."
	pytest tests/ -v
	@echo "Tests complete!"

test-verbose:
	@echo "Running test suite with verbose output..."
	pytest tests/ -v -s --tb=long
	@echo "Verbose tests complete!"

test-property:
	@echo "Running property-based tests..."
	pytest tests/property/ -v
	@echo "Property-based tests complete!"

docs-coverage:
	@echo "Checking documentation coverage..."
	interrogate fba_bench/ dashboard/ --verbose
	@echo "Documentation coverage check complete!"

# Pre-commit commands
pre-commit-install:
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "Pre-commit hooks installed!"

pre-commit-run:
	@echo "Running pre-commit on all files..."
	pre-commit run --all-files
	@echo "Pre-commit checks complete!"

# Cleanup commands
clean:
	@echo "Cleaning up generated files and caches..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "bandit-report.json" -delete
	@echo "Cleanup complete!"

# Development workflow shortcuts
dev-setup: install-dev
	@echo "Development environment ready!"

dev-check: format lint type-check test
	@echo "Full development check complete!"

# CI/CD simulation
ci: check test docs-coverage security-check
	@echo "CI pipeline simulation complete!"