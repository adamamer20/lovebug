# Contributing to LoveBug

Thank you for considering contributing to LoveBug! This guide provides comprehensive information for developers working on this agent-based model of sexual selection and mating-preference co-evolution.

## Project Overview

LoveBug is a high-performance agent-based simulation built with Mesa-Frames and Polars that models sexual selection and evolutionary dynamics. The project simulates large populations (100k+ individuals) of digital "love-bugs" whose genomes encode display traits, mate preferences, and choosiness thresholds.

### Core Components

- **Vectorized core**: All agents stored in a single Polars DataFrame for performance
- **Genetic encoding**: 32-bit unsigned int genome with bit-packed traits
- **Mutual mate choice**: Fast Hamming-similarity based partner selection
- **Evolutionary mechanics**: Uniform crossover, per-bit mutation, energy decay
- **Multi-layer architecture**: Genetic evolution + optional cultural learning

## Development Environment Setup

### Prerequisites

- **Python**: ≥3.11 (using modern features like type hints, dataclasses)
- **Operating System**: Linux, macOS, or Windows (developed on Pop OS 24.04)
- **Optional**: CUDA 12 for GPU acceleration (RTX 3090 target)

### Initial Setup

1. **Fork and clone the repository**:

   ```bash
   git clone https://github.com/YOUR_USERNAME/lovebug.git
   cd lovebug
   ```

2. **Set up development environment**:

   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install development dependencies
   uv pip install -e .[dev]

   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify installation**:

   ```bash
   # Run tests
   pytest -v

   # Check code style
   ruff check .
   ruff format .

   # Run a basic simulation
   uv run python -m lovebug.unified_mesa_model
   ```

### Package Management with UV

This project uses [astral-sh/uv](https://github.com/astral-sh/uv) as the primary package manager for several advantages:

- **Speed**: 10-100x faster than pip for dependency resolution
- **Reliability**: Better dependency resolution with clear error messages
- **Lockfile management**: Automatic generation and maintenance of `uv.lock`
- **Environment isolation**: Built-in virtual environment management

#### Common UV Commands

```bash
# Install project in development mode
uv pip install -e .[dev]

# Add a new dependency
uv add polars>=0.20.0

# Add a development dependency
uv add --dev pytest>=8.0.0

# Run Python scripts (preferred over direct python calls)
uv run python scripts/example.py

# Run tests
uv run pytest

# Sync dependencies with lockfile
uv pip sync

# Update dependencies
uv pip compile pyproject.toml --upgrade
```

## Development Workflow

### Creating a New Feature

1. **Create feature branch**:

   ```bash
   git checkout -b feature/new-selection-algorithm
   ```

2. **Follow Test-Driven Development (TDD)**:
   - Write failing tests that define expected behavior
   - Implement minimal code to pass tests
   - Refactor while keeping tests green
   - Add integration tests for complex workflows

3. **Development cycle**:

   ```bash
   # Run tests continuously during development
   uv run pytest --watch

   # Check code quality before committing
   uv run pre-commit run --all-files
   uv run pytest -v
   ```

4. **Submit changes**:

   ```bash
   git add .
   git commit -m "feat: add new selection algorithm"
   git push origin feature/new-selection-algorithm
   ```

### Code Quality Standards

#### Core Principles

1. **Clarity over cleverness**: Code should read like English
2. **DRY (Don't Repeat Yourself)**: Atomic, modular, loosely coupled functions
3. **Type safety**: Full PEP 484/695 type hints everywhere
4. **Performance**: Vectorized operations with Polars/NumPy
5. **Testability**: TDD approach with pytest

#### Code Style & Quality Tools

```bash
# Code formatting and linting
uv run ruff check .
uv run ruff format .

# Pre-commit hooks (runs automatically)
uv run pre-commit run --all-files

# Testing with coverage
uv run pytest --cov=src/lovebug

# Type checking (optional, enable with DEV_TYPECHECK=1)
DEV_TYPECHECK=1 uv run pytest
```

#### Documentation Standards

- **Docstring format**: NumPy-style with Parameters, Returns, Raises, Examples
- **Type hints**: Every function parameter and return value
- **Inline comments**: ≤80 chars, explain "why" not "what"
- **Public API**: Explicit `__all__` exports define what's public

**Example function signature**:

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

## Testing

### Test Structure

```bash
# Run all tests
pytest

# Run with type checking
DEV_TYPECHECK=1 pytest

# Run with coverage
pytest --cov=src/lovebug

# Run specific test file
pytest tests/test_unified_mesa_model.py

# Run tests with output
pytest -v -s
```

### CI/CD Pipeline

The project includes a GitHub Actions workflow that runs on every push and PR:

1. Install dependencies with `uv pip install -e .[dev]`
2. Run pre-commit hooks: `pre-commit run --all-files`
3. Execute test suite: `pytest -v`
4. Generate coverage reports
5. Build documentation

### Pre-commit Configuration

Automated code quality checks include:

- `ruff check` and `ruff format` for code style
- `pytest` for test validation
- Type checking with runtime validation (when enabled)

## Performance Guidelines

### Polars-First Approach

- **Prefer Polars over Pandas**: Faster, memory-efficient, better API
- **Lazy evaluation**: Use `.lazy()` for complex operations, `.collect()` only when needed
- **Expression API**: Leverage Polars' powerful expression system
- **Memory efficiency**: Profile memory usage for large populations

### Vectorization Requirements

- **No loops**: Use vectorized operations wherever possible
- **GPU acceleration**: Assume CUDA availability, error clearly if unavailable
- **Profiling**: Benchmark before optimizing, measure after changes
- **Scalability**: Design for 100k+ agents from the start

**Good vectorized example**:

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

## Project Structure

### Core Module Organization

```
src/lovebug/
├── __init__.py              # Public API exports
├── unified_mesa_model.py    # Main LoveModel and LoveAgents
├── lande_kirkpatrick.py     # Classical sexual selection model
├── layer_activation.py      # Layer configuration management
├── layer2/                  # Cultural learning layer
│   ├── __init__.py
│   ├── config.py           # Layer 2 configuration
│   ├── cultural_layer.py   # Cultural evolution implementation
│   ├── learning_algorithms.py  # Social learning algorithms
│   ├── network.py          # Social network structures
│   └── monitoring/         # Monitoring and diagnostics
└── visualization/          # Post-hoc visualization system
    ├── core.py             # Visualization engine
    ├── data.py             # Data collection and storage
    ├── charts/             # Chart type implementations
    └── backends/           # Rendering backends
```

### Configuration Management

- **Environment variables**: Use `.env` files with python-dotenv
- **Settings**: Long-term config in `pyproject.toml` keyed sections
- **No hardcoding**: Paths, URLs, GPU counts should be configurable
- **Secrets**: Never commit tokens, use environment variables

## Common Development Tasks

### Running Simulations

```bash
# Basic simulation
uv run python -m lovebug.unified_mesa_model

# Run examples
uv run python examples/layer2_demo.py
uv run python examples/lande_kirkpatrick_demo.py

# Custom simulation
uv run python -c "
from lovebug import LoveModel
model = LoveModel(population_size=50000, mutation_rate=0.01)
model.run_model()
print(f'Final population: {len(model.agents)}')
"
```

### Documentation

```bash
# Serve docs locally
mkdocs serve

# Build docs
mkdocs build

# Check documentation links
mkdocs build --strict
```

### Debugging and Profiling

```bash
# Run with detailed logging
PYTHONPATH=src python -m lovebug.unified_mesa_model --verbose

# Profile performance
python -m cProfile -o profile_output.prof examples/layer2_demo.py

# Memory profiling
python -m memory_profiler examples/layer2_demo.py
```

## Safety & Best Practices

### Security

- **No eval/exec**: Build dynamic code safely with AST or templates
- **Input validation**: Validate all external inputs with beartype or pydantic
- **SQL safety**: Use parameterized queries if database access is added

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

## Extending LoveBug

### Adding New Features

1. **Design phase**: Write docstring and type signature first
2. **Test phase**: Create failing tests that define expected behavior
3. **Implementation**: Write minimal code to pass tests
4. **Integration**: Ensure new feature works with existing codebase
5. **Documentation**: Update docstrings, add examples, update README if needed

### Extension Points

- **New selection algorithms**: Implement in `layer2/learning_algorithms.py`
- **Custom mate choice**: Extend compatibility scoring in `unified_mesa_model.py`
- **Visualization backends**: Add new renderers in `visualization/backends/`
- **Data collectors**: Extend monitoring in `layer2/monitoring/`

## Environment Variables

- `DEV_TYPECHECK=1`: Enable runtime type checking with typeguard
- `CUDA_VISIBLE_DEVICES`: Control GPU access for CUDA operations
- `PYTHONPATH`: Set to `src` for development testing

## Getting Help

### Resources

- **Documentation**: [https://adamamer20.github.io/lovebug/](https://adamamer20.github.io/lovebug/)
- **Mesa-Frames**: [GitHub Repository](https://github.com/projectmesa/mesa-frames)
- **Polars**: [User Guide](https://pola-rs.github.io/polars/)
- **UV**: [Documentation](https://github.com/astral-sh/uv)

### Community

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Pull Requests**: Code contributions and reviews

## Development Checklist

Before submitting contributions:

- [ ] Dependencies managed with `uv` - lockfile up to date
- [ ] Code passes `ruff check` and `ruff format`
- [ ] Pre-commit hooks installed and passing
- [ ] Tests written before implementation (TDD)
- [ ] NumPy-style docstrings with full typing
- [ ] Polars preferred for data operations
- [ ] GPU-first approach with clear error handling
- [ ] All exceptions logged, never fail silently
- [ ] Use `uv run` for executing Python scripts
- [ ] Documentation updated for new features

---

*This guide provides everything needed to contribute effectively to LoveBug. The project maintains high standards for code quality, performance, and scientific rigor to ensure its value for evolutionary biology research.*
