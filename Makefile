# Developer ergonomics. Mirrors the CI gates so `make ci` is the
# single command that has to pass before you push.
#
# Run `make help` for the list of targets.

.DEFAULT_GOAL := help

PY      ?= python
VENV    ?= .venv
ACTIVATE = . $(VENV)/bin/activate

SOURCE_DIRS = geno_lewm tools tests

.PHONY: help
help:  ## Show this help.
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ---------------------------------------------------------------------------
# Environment

.PHONY: venv
venv:  ## Create the dev virtualenv.
	uv venv $(VENV) --python 3.12

.PHONY: install
install:  ## Install the package in editable mode with dev extras.
	uv pip install -e ".[dev,docs]"

.PHONY: hooks
hooks:  ## Install pre-commit hooks.
	$(ACTIVATE) && pre-commit install --install-hooks

# ---------------------------------------------------------------------------
# Quality gates

.PHONY: format
format:  ## Run ruff format.
	$(ACTIVATE) && ruff format $(SOURCE_DIRS)

.PHONY: format-check
format-check:  ## Verify formatting.
	$(ACTIVATE) && ruff format --check $(SOURCE_DIRS)

.PHONY: lint
lint:  ## Run ruff lint.
	$(ACTIVATE) && ruff check $(SOURCE_DIRS)

.PHONY: lint-fix
lint-fix:  ## Run ruff lint with autofix.
	$(ACTIVATE) && ruff check --fix $(SOURCE_DIRS)

.PHONY: types
types:  ## Run mypy --strict.
	$(ACTIVATE) && mypy geno_lewm tools

.PHONY: gates
gates:  ## Run every custom AST gate + the public-surface snapshot.
	$(ACTIVATE) && $(PY) -m tools.lint.check_error_codes
	$(ACTIVATE) && $(PY) -m tools.lint.check_event_names
	$(ACTIVATE) && $(PY) -m tools.lint.check_no_print
	$(ACTIVATE) && $(PY) -m tools.lint.check_network_confined
	$(ACTIVATE) && $(PY) -m tools.lint.check_license_headers
	$(ACTIVATE) && $(PY) -m tools.api.snapshot check

.PHONY: snapshot
snapshot:  ## Regenerate the public-surface snapshot (only after agreed-upon API change).
	$(ACTIVATE) && $(PY) -m tools.api.snapshot write

# ---------------------------------------------------------------------------
# Tests

.PHONY: test
test:  ## Run the test suite once.
	$(ACTIVATE) && pytest -n auto

.PHONY: test-cov
test-cov:  ## Run the test suite with branch coverage.
	$(ACTIVATE) && pytest -n auto --cov=geno_lewm --cov-branch --cov-report=term-missing --cov-report=html

.PHONY: test-fast
test-fast:  ## Run only the fast (unit) test bucket.
	$(ACTIVATE) && pytest tests/unit -n auto

# ---------------------------------------------------------------------------
# Build / release rehearsal

.PHONY: build
build:  ## Build sdist + wheel.
	$(ACTIVATE) && $(PY) -m build

.PHONY: build-check
build-check: build  ## Build and run twine check on the artifacts.
	$(ACTIVATE) && twine check dist/*

.PHONY: clean
clean:  ## Remove build artifacts and caches.
	rm -rf build dist site htmlcov *.egg-info
	find . -type d \( -name __pycache__ -o -name ".pytest_cache" -o -name ".ruff_cache" -o -name ".mypy_cache" -o -name ".hypothesis" \) -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Docs

.PHONY: docs
docs:  ## Build the docs site in strict mode.
	$(ACTIVATE) && mkdocs build --strict

.PHONY: docs-serve
docs-serve:  ## Serve the docs locally with hot reload.
	$(ACTIVATE) && mkdocs serve

# ---------------------------------------------------------------------------
# Aggregate

.PHONY: ci
ci: format-check lint types gates test docs  ## Full local CI rehearsal.
	@echo "✓ Local CI green."
