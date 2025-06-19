# LoveBug AI Agent Instructions 🐞💘

This document provides specific instructions for AI agents working on the LoveBug project. For comprehensive development information, see the [Contributing Guide](https://adamamer20.github.io/lovebug/development/contributing/).

## Project Context

**LoveBug** is a high-performance agent-based simulation of sexual selection and mating-preference co-evolution, built with Mesa-Frames and Polars. The project models large populations (100k+ individuals) for evolutionary biology research.

## AI Agent Quick Reference

### Core Architecture
- **Main model**: [`UnifiedLoveModel`](src/lovebug/unified_mesa_model.py) and [`UnifiedLoveBugs`](src/lovebug/unified_mesa_model.py)
- **Genetic layer**: Classical Lande-Kirkpatrick sexual selection
- **Cultural layer**: Optional social learning mechanisms in [`layer2/`](src/lovebug/layer2/)
- **Vectorized operations**: All agents in single Polars DataFrame

### Key Commands
```bash
# Development setup
uv pip install -e .[dev]
uv run pre-commit install

# Code quality
uv run ruff check . && uv run ruff format .
uv run pytest -v

# Run simulations
uv run python -m lovebug.unified_mesa_model
uv run python examples/layer2_demo.py
```

### Performance Requirements
- **Polars-first**: Prefer Polars over Pandas for data operations
- **Vectorization**: No loops where vectorized operations exist
- **Type safety**: Full type hints with NumPy-style docstrings
- **GPU-aware**: Assume CUDA availability, error clearly if unavailable

### File Structure Updates
Important: The project structure has evolved. Use these current modules:

- **Core model**: `src/lovebug/unified_mesa_model.py` (not `model.py`)
- **Configuration**: `src/lovebug/layer_activation.py` and `src/lovebug/layer2/config.py`
- **Cultural learning**: `src/lovebug/layer2/` directory
- **Visualization**: `src/lovebug/visualization/` directory

### Common Patterns

#### Correct Model Usage
```python
from lovebug import UnifiedLoveModel, LayerActivationConfig

# Basic genetic-only simulation
model = UnifiedLoveModel(population_size=5000)
model.run_model()

# With cultural learning
config = LayerActivationConfig(cultural_layer=True)
model = UnifiedLoveModel(population_size=5000, layer_config=config)
model.run_model()
```

#### Vectorized Operations
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
# for agent in population: ...  # Avoid this pattern
```

## AI Agent Specific Guidelines

### When Making Changes
1. **Read first**: Use `read_file` to understand current implementation
2. **Search scope**: Use `search_files` to find related code patterns
3. **Verify consistency**: Ensure changes align with existing patterns
4. **Test thoroughly**: Run full test suite after changes

### Documentation Updates
- **Consistency**: Maintain academic tone suitable for research context
- **References**: Update file paths and class names to match current structure
- **Integration**: Ensure changes work with MkDocs site structure

### Common Pitfalls to Avoid
- **Outdated imports**: Use `from lovebug import UnifiedLoveModel`, not `lovebug.model`
- **Silent failures**: Always log errors and provide clear error messages
- **Performance regressions**: Profile before and after significant changes
- **Type safety**: Never skip type hints or proper validation

### Research Context Awareness
This is a scientific project for evolutionary biology research. Maintain:
- **Scientific rigor**: Accurate implementation of evolutionary mechanisms
- **Reproducibility**: Proper random seed handling and version control
- **Academic standards**: Publication-quality code and documentation
- **Performance**: Support for large-scale simulations (100k+ agents)

## Resources

### Complete Documentation
- **[Full Documentation](https://adamamer20.github.io/lovebug/)** - Complete project documentation
- **[Installation Guide](https://adamamer20.github.io/lovebug/installation/)** - Setup instructions
- **[Contributing Guide](https://adamamer20.github.io/lovebug/development/contributing/)** - Comprehensive development guide
- **[API Reference](https://adamamer20.github.io/lovebug/api/)** - Detailed API documentation

### Technical References
- **[Mesa-Frames](https://github.com/projectmesa/mesa-frames)** - ABM framework
- **[Polars](https://pola-rs.github.io/polars/)** - Data manipulation library
- **[UV Package Manager](https://github.com/astral-sh/uv)** - Fast Python package management

### Project Structure
- **Examples**: [`examples/`](examples/) - Demonstration scripts
- **Notebooks**: [`notebooks/`](notebooks/) - Interactive tutorials
- **Tests**: [`tests/`](tests/) - Test suite
- **Documentation**: [`docs/`](docs/) - Documentation source

## Quick Start Checklist

Before making changes:
- [ ] Read the [Contributing Guide](https://adamamer20.github.io/lovebug/development/contributing/)
- [ ] Understand current architecture via [`src/lovebug/__init__.py`](src/lovebug/__init__.py)
- [ ] Check existing tests for patterns
- [ ] Verify `uv run pytest -v` passes
- [ ] Run `uv run ruff check .` for code quality

---

*For detailed development information, coding standards, and project architecture, see the comprehensive [Contributing Guide](https://adamamer20.github.io/lovebug/development/contributing/). This document provides a quick reference for AI agents working on the LoveBug project.*
