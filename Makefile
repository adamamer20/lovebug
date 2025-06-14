.PHONY: help install test lint format docs clean build publish dev-install sync

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

sync: ## Sync dependencies with uv
	uv sync

install: ## Install the package
	uv sync --frozen

dev-install: ## Install the package in development mode with all dependencies
	uv sync --all-extras

test: ## Run tests
	uv run pytest

test-cov: ## Run tests with coverage
	uv run pytest --cov=lovebug --cov-report=html --cov-report=xml

test-type: ## Run tests with runtime type checking
	uv run env DEV_TYPECHECK=1 pytest

lint: ## Run linting
	uv run ruff check .

format: ## Format code
	uv run ruff format .

lint-fix: ## Run linting with auto-fix
	uv run ruff check --fix .

pre-commit: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

docs: ## Serve documentation locally
	uv run mkdocs serve

docs-build: ## Build documentation
	uv run mkdocs build

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf site/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: ## Build the package
	uv pip wheel --no-deps --wheel-dir dist .

publish: ## Publish to PyPI (requires proper credentials)
	uv pip wheel --no-deps --wheel-dir dist .
	twine upload dist/*

dev: ## Start development environment
	@echo "Development environment ready!"
	@echo "- Run 'make test' to run tests"
	@echo "- Run 'make docs' to serve documentation"
	@echo "- Run 'make lint' to check code quality"
