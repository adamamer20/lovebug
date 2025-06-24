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

### Research Experiments: Multi-Phase Protocol

The experimental framework (`experiments/paper_experiments.py`) implements a comprehensive research protocol with three distinct phases:

#### Phase 1: Literature Replications (--run-empirical)
Validates our model against established experiments from the evolutionary biology literature:

- **Dugatkin Mate-Choice Copying** (`experiments/replications/dugatkin_replication.py`): Tests whether agents can copy mate preferences from high-prestige individuals, replicating Dugatkin's (1992) guppy experiments
- **Witte Cultural Transmission** (`experiments/replications/witte_replication.py`): Examines how cultural learning affects mate choice evolution in social networks
- **Rodd Sensory Bias** (`experiments/replications/rodd_replication.py`): Tests whether pre-existing sensory biases can drive evolution of ornaments without genetic correlation

#### Phase 2: Parameter Space Exploration (--run-lhs)
Systematic exploration using Latin Hypercube Sampling across key parameter dimensions:

- **Genetic-only LHS**: Explores mutation rates, crossover rates, population sizes, energy dynamics
- **Cultural-only LHS**: Explores learning rates, innovation rates, social network topologies
- **Combined genetic-cultural LHS**: Explores blending weights between genetic and cultural evolution

#### Lande-Kirkpatrick Validation (--run-lk)
Tests three classic sexual selection scenarios with our unlinked gene model:
- **Stasis**: Moderate heritability, balanced energy → no trait-preference correlation
- **Runaway**: High heritability, abundant energy → trait elaboration
- **Costly Choice**: High heritability, scarce energy → constrained evolution

#### Execution Commands
```bash
# Full experimental pipeline (recommended)
uv run python experiments/paper_experiments.py --run-empirical --run-lhs

# Individual phases
uv run python experiments/paper_experiments.py --run-empirical                       # Literature replications only
uv run python experiments/paper_experiments.py --run-lhs --lhs-samples 100          # Parameter exploration only
uv run python experiments/paper_experiments.py --run-lk                             # Lande-Kirkpatrick validation

# Testing and debugging
uv run python experiments/paper_experiments.py --quick-test                          # Quick validation (reduced parameters)
uv run python experiments/paper_experiments.py --quick-test --run-empirical         # Quick replication tests

# Performance optimization
export POLARS_MAX_THREADS=20 RAYON_NUM_THREADS=20  # Single job optimal
export POLARS_MAX_THREADS=10 RAYON_NUM_THREADS=10  # Dual concurrent jobs
```

#### Experiment Validation & Results
Each experiment generates:
- **Statistical validation reports** with replication success rates and confidence intervals
- **Timestamped session directories** with detailed results and logs
- **Consolidated validation reports** combining all phases
- **JSON summaries** for downstream analysis and visualization

## Key Development Patterns

### Model Configuration
```python
from lovebug import LoveModel, LoveBugConfig, GeneticParams, CulturalParams, LayerConfig, SimulationParams

# Genetic-only evolution
config = LoveBugConfig(
    name="genetic_only_simulation",
    genetic=GeneticParams(
        mutation_rate=0.01,
        crossover_rate=0.7,
    ),
    cultural=CulturalParams(),
    simulation=SimulationParams(population_size=5000),
    layer=LayerConfig(genetic_enabled=True, cultural_enabled=False)
)
model = LoveModel(config=config)

# Combined evolution with cultural learning
config = LoveBugConfig(
    name="combined_simulation",
    genetic=GeneticParams(
        mutation_rate=0.01,
        crossover_rate=0.7,
    ),
    cultural=CulturalParams(
        horizontal_transmission_rate=0.2,
        innovation_rate=0.05,
        network_type="small_world"
    ),
    simulation=SimulationParams(population_size=5000),
    layer=LayerConfig(genetic_enabled=True, cultural_enabled=True)
)
model = LoveModel(config=config)
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
