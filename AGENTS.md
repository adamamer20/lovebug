# LoveBug Project Agent Instructions 🐞💘

This document provides comprehensive instructions for AI agents working on the LoveBug project, an agent-based model (ABM) of sexual selection and mating-preference co-evolution.

## 📋 Project Overview

**LoveBug** is a high-performance agent-based simulation built with Mesa-Frames and Polars that models sexual selection and evolutionary dynamics. The project simulates large populations (100k+ individuals) of digital "love-bugs" whose genomes encode display traits, mate preferences, and choosiness thresholds.

### Core Components

- **Vectorized core**: All agents stored in a single Polars DataFrame for performance
- **Genetic encoding**: 32-bit unsigned int genome with bit-packed traits
- **Mutual mate choice**: Fast Hamming-similarity based partner selection
- **Evolutionary mechanics**: Uniform crossover, per-bit mutation, energy decay

## 🛠️ Development Environment & Tooling

### Package Management with UV

This project uses **astral-sh/uv** as the primary package manager. Here's why and how to use it:

#### Why UV?

- **Speed**: 10-100x faster than pip for dependency resolution and installation
- **Reliability**: Better dependency resolution algorithm with clear error messages
- **Lockfile management**: Automatic generation and maintenance of `uv.lock`
- **Environment isolation**: Built-in virtual environment management
- **Future-proof**: Rust-based tool that's actively maintained and improving

#### Common UV Commands

```bash
# Install the project in development mode with all dev dependencies
uv pip install -e .[dev]

# Install specific dependency groups
uv pip install -e .[data,ml]

# Add a new dependency
uv add polars>=0.20.0

# Add a development dependency
uv add --dev pytest>=8.0.0

# Run Python scripts (preferred over direct python calls)
uv run python scripts/example.py

# Run pytest
uv run pytest

# Sync dependencies with lockfile
uv pip sync

# Update dependencies
uv pip compile pyproject.toml --upgrade
```

#### Project Dependencies Structure

The project uses modern Python packaging with `pyproject.toml`:

- **Core dependencies**: Essential runtime requirements (Mesa-Frames, Polars, NumPy)
- **Development dependencies**: In `[dependency-groups]` dev section (pytest, ruff, pre-commit)
- **Optional dependencies**: Feature-specific groups like `data` and `ml`

### Target Environment

- **OS**: Pop OS 24.04 (Debian-based)
- **Python**: >=3.11 (using modern features like type hints, dataclasses)
- **GPU**: RTX 3090, CUDA 12
- **Performance**: GPU-first approach, no silent CPU fallbacks

## 🎯 Coding Standards & Philosophy

### Core Principles

1. **Clarity over cleverness**: Code should read like English
2. **DRY (Don't Repeat Yourself)**: Atomic, modular, loosely coupled functions
3. **Type safety**: Full PEP 484/695 type hints everywhere
4. **Performance**: Vectorized operations with Polars/NumPy
5. **Testability**: TDD approach with pytest

### Code Style & Quality

```bash
# Code formatting and linting
uv run ruff check .
uv run ruff format .

# Pre-commit hooks (auto-installed)
uv run pre-commit run --all-files

# Testing
uv run pytest -v
uv run pytest --cov=src/lovebug
```

### Documentation Standards

- **Docstring format**: NumPy-style with Parameters, Returns, Raises, Examples
- **Type hints**: Every function parameter and return value
- **Inline comments**: ≤80 chars, explain "why" not "what"
- **Public API**: Explicit `__all__` exports define what's public

Example function signature:

```python
def mate_selection(
    agents: pl.DataFrame,
    compatibility_threshold: float = 0.8,
    *,
    random_seed: int | None = None,
) -> pl.DataFrame:
    """Select potential mates based on genetic compatibility.

    Parameters
    ----------
    agents : pl.DataFrame
        DataFrame containing agent genomes and traits
    compatibility_threshold : float, default=0.8
        Minimum genetic similarity required for mating
    random_seed : int | None, default=None
        Seed for random number generation

    Returns
    -------
    pl.DataFrame
        DataFrame with mating pairs and compatibility scores

    Raises
    ------
    ValueError
        If compatibility_threshold not in [0, 1]

    Examples
    --------
    >>> agents = create_population(1000)
    >>> pairs = mate_selection(agents, compatibility_threshold=0.7)
    """
```

## 🧪 Testing & Continuous Integration

### Test-Driven Development

1. Start with failing pytest cases
2. Implement minimal code to pass tests
3. Refactor while keeping tests green
4. Add integration tests for complex workflows

### CI Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs on every push and PR:

1. Install dependencies with `uv pip install -e .[dev]`
2. Run pre-commit hooks: `pre-commit run --all-files`
3. Execute test suite: `pytest -q`
4. Generate coverage reports

### Pre-commit Configuration

Automated code quality checks include:

- `ruff check` and `ruff format` for code style
- `pytest` for test validation
- Type checking with runtime validation

## 📊 Data Handling & Performance

### Polars First

- **Prefer Polars over Pandas**: Faster, memory-efficient, better API
- **Lazy evaluation**: Use `.lazy()` for complex operations, `.collect()` only when needed
- **Expression API**: Leverage Polars' powerful expression system
- **Memory efficiency**: Profile memory usage for large populations

### Performance Considerations

- **Vectorization**: No loops where vectorized operations exist
- **GPU acceleration**: Assume CUDA availability, error clearly if unavailable
- **Profiling**: Benchmark before optimizing, measure after changes
- **Scalability**: Design for 100k+ agents from the start

Example vectorized operation:

```python
# Good: Vectorized with Polars
compatible_pairs = (
    population
    .lazy()
    .with_columns(
        pl.col("genome").map_elements(
            lambda x: hamming_distance(x, target_genome)
        ).alias("compatibility")
    )
    .filter(pl.col("compatibility") > threshold)
    .collect()
)

# Bad: Loop-based approach
compatible_pairs = []
for agent in population:
    if hamming_distance(agent.genome, target_genome) > threshold:
        compatible_pairs.append(agent)
```

## 🔧 Project Structure & Modules

### Core Module Organization

```
src/lovebug/
├── __init__.py          # Public API exports
├── model.py            # Main LoveBugs and LoveModel classes
├── genetics.py         # Genome encoding/decoding utilities
├── selection.py        # Mate selection algorithms
├── reproduction.py     # Crossover and mutation operations
└── visualization.py    # Plotting and analysis tools
```

### Configuration Management

- **Environment variables**: Use `.env` files with python-dotenv
- **Settings**: Long-term config in `pyproject.toml` keyed sections
- **No hardcoding**: Paths, URLs, GPU counts should be configurable
- **Secrets**: Never commit tokens, use environment variables

## 🚀 Common Development Tasks

### Running Simulations

```bash
# Basic simulation
uv run python -m lovebug.examples.basic_simulation

# With custom parameters
uv run python -c "
from lovebug import LoveModel
model = LoveModel(population_size=50000, mutation_rate=0.01)
for _ in range(100):
    model.step()
print(f'Final population: {len(model.agents)}')
"
```

### Development Workflow

```bash
# Setup development environment
uv pip install -e .[dev]
uv run pre-commit install

# Create feature branch
git checkout -b feature/new-selection-algorithm

# Run tests continuously during development
uv run pytest --watch

# Before committing
uv run pre-commit run --all-files
uv run pytest -v
```

### Adding New Features

1. **Design phase**: Write docstring and type signature first
2. **Test phase**: Create failing tests that define expected behavior
3. **Implementation**: Write minimal code to pass tests
4. **Integration**: Ensure new feature works with existing codebase
5. **Documentation**: Update docstrings, add examples, update README if needed

## 🛡️ Safety & Best Practices

### Security

- **No eval/exec**: Build dynamic code safely with AST or templates
- **SQL safety**: Use SQLAlchemy Core or parameterized queries
- **Input validation**: Validate all external inputs with beartype or pydantic

### Error Handling

- **Never fail silently**: Always log exceptions and provide meaningful errors
- **GPU requirements**: Raise clear errors if CUDA unavailable
- **Graceful degradation**: Only where explicitly requested, never silent fallbacks

### Logging

```python
import logging

logger = logging.getLogger(__name__)

def risky_operation():
    try:
        # ... operation that might fail
        pass
    except Exception as e:
        logger.exception("Operation failed with parameters: %s", params)
        raise  # Re-raise after logging
```

## 📚 References & Further Reading

- **General coding standards**: See `general.instructions.md` for complete guidelines
- **Mesa-Frames documentation**: [GitHub Repository](https://github.com/projectmesa/mesa-frames)
- **Polars documentation**: [User Guide](https://pola-rs.github.io/polars/)
- **UV documentation**: [Installation and Usage](https://github.com/astral-sh/uv)
- **Type hints**: [PEP 484](https://peps.python.org/pep-0484/) and [PEP 695](https://peps.python.org/pep-0695/)

## 🎉 Quick Start Checklist

For new contributors or agents working on this project:

- [ ] Dependencies managed with `uv` - lockfile up to date
- [ ] Code passes `ruff check` and `ruff format`
- [ ] Pre-commit hooks installed and passing
- [ ] Tests written before implementation (TDD)
- [ ] NumPy-style docstrings with full typing
- [ ] Polars preferred for data operations
- [ ] GPU-first approach (RTX 3090), clear errors if CUDA unavailable
- [ ] GitHub Actions workflow ready
- [ ] Code is DRY, atomic, modular, easily swappable
- [ ] Environment configuration via `.env` or `pyproject.toml`
- [ ] All exceptions logged, never fail silently
- [ ] Use `uv run` for executing Python scripts

---

*This document should be your primary reference when working on the LoveBug project. Always prioritize the principles outlined in `general.instructions.md` and maintain the high-performance, scientifically rigorous approach that makes this simulation valuable for evolutionary biology research.*
