# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# LoveBug: Agent-Based Evolutionary Simulation

LoveBug is a research-oriented agent-based model for studying sexual selection and cultural-genetic coevolution. Built with Mesa-Frames and Polars for high-performance vectorized operations on large populations (100k+ individuals).

## Project Architecture

### Core Components

**Primary Model**: `src/lovebug/unified_mesa_model.py` - Enhanced Mesa-Frames model supporting:
- Genetic-only evolution (Layer 1)
- Cultural learning mechanisms (Layer 2)
- Combined genetic-cultural evolution
- Vectorized operations using Polars DataFrames

**Configuration System**:
- `LayerActivationConfig`: Controls which evolutionary layers are active
- `Layer2Config`: Cultural learning parameters (transmission rates, network topology, memory)
- `LandeKirkpatrickParams`: Classic sexual selection model parameters

**Key Architecture Patterns**:
- All agents stored in single Polars DataFrame for vectorized operations
- 32-bit genome encoding: `[display_traits][mate_preferences][behavior_threshold]`
- Bit manipulation for efficient genetic operations using numpy uint32
- Mesa-Frames AgentSetPolars for agent management

### Project Structure

```
src/lovebug/
├── unified_mesa_model.py      # Main model implementation
├── layer2/                    # Cultural evolution components
│   ├── config.py             # Layer2Config class
│   ├── cultural_layer.py     # Cultural learning algorithms
│   └── network.py           # Social network topologies
├── layer_activation.py       # Layer activation configuration
└── visualization/            # Plotting and data visualization

experiments/                  # Research experiment scripts
├── paper_experiments.py     # Systematic parameter sweeps
├── runner.py               # Experiment execution framework
└── models.py              # Data models for results

notebooks/                   # Interactive research notebooks
├── Lande-Kirkpatrick.py    # Classic sexual selection model
└── Layer2-Social-Learning.py # Cultural transmission demo
```

## Development Commands

### Basic Development
```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Code quality
uv run ruff check .
uv run ruff format .

# Run with type checking
uv run env DEV_TYPECHECK=1 pytest
```

### Using Make Commands
```bash
make dev-install    # Install with all dependencies
make test          # Run tests
make test-cov      # Run tests with coverage
make lint          # Check code quality
make format        # Format code
make docs          # Serve documentation locally
```

### Running Simulations
```bash
# Basic model execution
uv run python -m lovebug.unified_mesa_model

# Interactive notebooks
uv run python notebooks/Lande-Kirkpatrick.py
uv run marimo run notebooks/Layer2-Social-Learning.py

# Research experiments - Multi-phase protocol
uv run python experiments/paper_experiments.py --output experiments/results/paper_data  # Phase 1 validation only
uv run python experiments/paper_experiments.py --quick-test                              # Quick validation test
uv run python experiments/paper_experiments.py --run-lhs --lhs-samples 100             # Phase 2 LHS exploration
uv run python experiments/paper_experiments.py --no-validation --run-lhs               # LHS only (skip validation)
```

## Key Development Patterns

### Model Configuration
```python
from lovebug import LoveModel, LayerActivationConfig, Layer2Config

# Genetic-only evolution
model = LoveModel(
    population_size=5000,
    layer_config=LayerActivationConfig(genetic_enabled=True, cultural_enabled=False)
)

# Combined evolution with cultural learning
cultural_config = Layer2Config(
    horizontal_transmission_rate=0.2,
    innovation_rate=0.05,
    network_type="small_world"
)
model = LoveModel(
    population_size=5000,
    layer_config=LayerActivationConfig(genetic_enabled=True, cultural_enabled=True),
    cultural_params=cultural_config
)
```

### Genetic Encoding
- Genomes are 32-bit unsigned integers with bit-packed traits
- Display traits: bits 0-15 (what others see)
- Mate preferences: bits 16-23 (what they like)
- Behavioral threshold: bits 24-31 (choosiness)
- Use bit masks and shifts for trait extraction

### Performance Considerations
- All operations vectorized using Polars expressions
- Avoid Python loops over agents - use DataFrame operations
- Memory-efficient bit manipulation for genetic operations
- Population sizes 100k+ supported in pure Python

## Testing

- Tests in `tests/` directory use pytest framework
- Test files: `test_model.py`, `test_enhanced_unified_model.py`, `test_vectorized_layer2.py`
- Markers: `@pytest.mark.slow` for computationally intensive tests
- Run with `make test` or `uv run pytest`

## Research Focus

This codebase is designed for studying:
- Sexual selection dynamics (Fisher-Lande-Kirkpatrick mechanisms)
- Cultural-genetic coevolution
- Social learning and cultural transmission
- Population genetics in finite populations
- Network effects on mate choice evolution

The model implements established theoretical frameworks with computational efficiency for large-scale simulations and parameter sweeps.
