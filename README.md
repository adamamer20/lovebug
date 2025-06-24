# LoveBug: Agent-Based Evolutionary Simulation 🐞💘

[![CI](https://github.com/adamamer20/lovebug/workflows/CI/badge.svg)](https://github.com/adamamer20/lovebug/actions/workflows/ci.yml)
[![Documentation](https://github.com/adamamer20/lovebug/workflows/Documentation/badge.svg)](https://adamamer20.github.io/lovebug/)
[![PyPI version](https://badge.fury.io/py/lovebug.svg)](https://badge.fury.io/py/lovebug)
[![Python versions](https://img.shields.io/pypi/pyversions/lovebug.svg)](https://pypi.org/project/lovebug/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A high-performance agent-based model for studying sexual selection and cultural-genetic coevolution. Built with Mesa-Frames and Polars for vectorized operations on large populations (100k+ individuals).**

---

## 📜 Project Overview

LoveBug is a research-oriented computational laboratory for investigating the evolution of mate choice through multiple inheritance mechanisms. The model integrates genetic inheritance, social learning, perceptual constraints, and cultural transmission within a unified framework designed to test theoretical predictions and replicate empirical findings from evolutionary biology literature.

### Key Research Questions

The model addresses fundamental questions in sexual selection and cultural evolution:

- **When do genetic vs. cultural mechanisms dominate mate choice evolution?**
- **How do social learning and cultural transmission interact with genetic inheritance?**
- **What role do perceptual constraints play in trait-preference coevolution?**
- **Can cultural transmission functionally substitute for genetic linkage in sexual selection?**

---

## ✨ Key Features

### Performance & Architecture
* **Vectorized core**: All agents stored in Polars DataFrames; handles 100k+ individuals efficiently
* **Unlinked gene architecture**: 32-bit genome encoding display traits, mate preferences, and behavioral thresholds
* **Mesa-Frames compatibility**: Full integration with Mesa's agent-based modeling framework

### Evolutionary Mechanisms
* **Two-layer evolution**: Genetic inheritance (Layer 1) + cultural learning (Layer 2)
* **Multiple learning strategies**: Mate-choice copying, conformist bias, prestige bias
* **Perceptual realism**: Noisy sensory channels and detection thresholds
* **Network effects**: Configurable social network topologies for cultural transmission

### Research Validation
* **Literature replications**: Quantitative reproduction of landmark empirical studies
* **Theory validation**: Recovers Fisher-Lande-Kirkpatrick dynamics without genetic linkage
* **Parameter exploration**: Latin Hypercube Sampling across genetic-cultural parameter space

---

## 🚀 Quick Start

### Installation

```bash
# For general use
pip install lovebug

# For development and research
git clone https://github.com/adamamer20/lovebug.git
cd lovebug
uv sync --all-extras
```

### Basic Usage

```python
from lovebug import LoveModel, LoveBugConfig, GeneticParams, CulturalParams, LayerConfig, SimulationParams

# Genetic-only evolution
config = LoveBugConfig(
    name="genetic_only_simulation",
    genetic=GeneticParams(
        mutation_rate=0.01,
        crossover_rate=0.7,
        h2_trait=0.8,
        h2_preference=0.8,
    ),
    cultural=CulturalParams(),
    simulation=SimulationParams(population_size=5000),
    layer=LayerConfig(genetic_enabled=True, cultural_enabled=False)
)

model = LoveModel(config=config)
model.run_model()
print(f"Final population: {len(model.agents)}")
```

### Run Research Experiments

```bash
# Complete experimental pipeline
uv run python experiments/paper_experiments.py --run-empirical --run-lhs

# Quick validation test
uv run python experiments/paper_experiments.py --quick-test

# Literature replications only
uv run python experiments/paper_experiments.py --run-empirical

# Parameter space exploration
uv run python experiments/paper_experiments.py --run-lhs --lhs-samples 100
```

---

## 🧬 Model Architecture

### Agent Representation

Agents are defined by heritable genetic traits and dynamic cultural states:

| Component | Description | Encoding |
|-----------|-------------|----------|
| **Display Trait** | Ornamental features visible to others | Bits 0-15 of 32-bit genome |
| **Mate Preference** | Attraction to specific display patterns | Bits 16-23 of 32-bit genome |
| **Choosiness Threshold** | Behavioral selectivity in mate choice | Bits 24-31 of 32-bit genome |
| **Foraging Efficiency** | Survival-related trait | Independent genetic locus |
| **Cultural Memory** | Learned preferences from social observation | Non-heritable state variable |

### Evolutionary Layers

#### Layer 1: Genetic Evolution
- **Inheritance**: Independent assortment of unlinked loci
- **Selection**: Natural selection on foraging + sexual selection on display/preference
- **Mutation**: Per-bit mutation with configurable rates and effect sizes
- **Population regulation**: Density-dependent resource competition

#### Layer 2: Cultural Evolution
- **Social learning**: Agents observe and copy successful individuals
- **Cultural transmission**: Horizontal, oblique, and vertical learning modes
- **Network structure**: Small-world, scale-free, or random social networks
- **Memory dynamics**: Finite cultural memory with decay and updating

### Mate Choice Process

1. **Preference formation**: Blend genetic and cultural preferences based on layer weights
2. **Partner assessment**: Evaluate potential mates through noisy perceptual channel
3. **Mutual choice**: Mating occurs only if both partners exceed acceptance thresholds
4. **Reproduction**: Genetic crossover with mutation produces offspring

---

## 🔬 Research Experiments

### Three-Phase Experimental Protocol

#### Phase 1: Literature Replications
Validates model against established empirical findings:

- **Dugatkin (1992)**: Mate-choice copying in guppies
- **Witte et al. (2002)**: Cultural transmission of mate preferences
- **Rodd et al. (2002)**: Sensory bias driven trait evolution

#### Phase 2: Parameter Space Exploration
Systematic exploration using Latin Hypercube Sampling:

- **Genetic-only regime**: Pure Fisher-Lande-Kirkpatrick dynamics
- **Cultural-only regime**: Social learning without genetic evolution
- **Combined regime**: Synergistic genetic-cultural coevolution

#### Lande-Kirkpatrick Validation
Tests three classic sexual selection scenarios:

- **Stasis**: Moderate heritability, balanced energy → no trait-preference correlation
- **Runaway**: High heritability, abundant energy → trait elaboration
- **Costly Choice**: High heritability, scarce energy → constrained evolution

### Key Findings

1. **Synergistic acceleration**: Combined genetic-cultural evolution is **2.3x faster** than purely genetic systems
2. **Functional substitution**: Cultural transmission can replace genetic linkage in driving trait-preference coevolution
3. **Regime identification**: Clear parameter boundaries between genetic-dominant, cultural-dominant, and synergistic regimes
4. **Population stability**: Mixed inheritance systems show greater demographic robustness than pure mechanisms

---

## 📊 Performance & Scalability

### Optimized Parameters
- **Population size**: 1,500 agents (optimal for ~11h full experimental suite)
- **Generations**: 3,000 steps for equilibrium dynamics
- **Replications**: 10 per condition for statistical robustness
- **LHS samples**: 100 per parameter sweep

### Parallelization
```bash
# Single job optimization
export POLARS_MAX_THREADS=20 RAYON_NUM_THREADS=20

# Dual concurrent jobs
export POLARS_MAX_THREADS=10 RAYON_NUM_THREADS=10
```

### Memory Efficiency
- **Vectorized operations**: All agent updates use Polars expressions
- **Bit-packed genomes**: 32-bit integers store multiple traits efficiently
- **Lazy evaluation**: Memory-efficient data processing pipelines

---

## 📚 Documentation & Resources

### Primary Resources
* **[Research Paper](paper/paper.qmd)** - Complete scientific methodology and findings
* **[API Documentation](https://adamamer20.github.io/lovebug/)** - Full technical reference
* **[CLAUDE.md](CLAUDE.md)** - Development guidelines and architecture notes

### Examples & Tutorials
* **[Lande-Kirkpatrick Notebook](notebooks/Lande-Kirkpatrick.py)** - Classic sexual selection model
* **[Cultural Learning Demo](notebooks/Layer2-Social-Learning.py)** - Social transmission mechanisms
* **[Replication Scripts](experiments/replications/)** - Literature reproduction code

### Development Commands
```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Code quality
uv run ruff check .
uv run ruff format .

# Documentation
make docs
```

---

## 🎯 Research Applications

LoveBug is designed for studying:

### Sexual Selection Theory
- **Fisher-Lande-Kirkpatrick mechanisms**: Runaway evolution and genetic correlation
- **Sensory bias hypothesis**: Pre-existing biases driving trait evolution
- **Direct vs. indirect benefits**: Mate choice evolution under different selection pressures

### Cultural Evolution
- **Gene-culture coevolution**: Interactions between genetic and cultural inheritance
- **Social learning strategies**: When and why different learning rules evolve
- **Cultural transmission dynamics**: Network effects on preference spread

### Population Genetics
- **Finite population effects**: Drift-selection balance in realistic population sizes
- **Linkage disequilibrium**: Emergence of trait-preference correlation without genetic linkage
- **Metapopulation dynamics**: Spatial structure effects on evolution

---

## 🤝 Contributing

We welcome contributions from researchers and developers!

### Development Workflow
1. **Fork the repository** and create a feature branch
2. **Follow code standards**: Use `ruff` for formatting and linting
3. **Add tests**: Ensure new features have appropriate test coverage
4. **Update documentation**: Include docstrings and update relevant guides
5. **Submit pull request**: With clear description of changes and rationale

### Code Standards
- **Type hints**: Use beartype decorators for runtime type checking
- **Documentation**: Comprehensive docstrings following NumPy style
- **Testing**: pytest with parametrized tests for robustness
- **Performance**: Profile code changes affecting simulation speed

---

## 📄 Citation & License

### License
MIT © 2025 Adam Amer. See [LICENSE](LICENSE) for full terms.

### Citation
If you use LoveBug in academic work, please cite:

```bibtex
@misc{lovebug2025,
  title   = {LoveBug: An agent-based model for studying sexual selection and cultural-genetic coevolution},
  author  = {Adam Amer},
  year    = {2025},
  journal = {In preparation},
  howpublished = {GitHub repository},
  url     = {https://github.com/adamamer20/lovebug}
}
```

---

## 🔗 Links & Support

* **[Documentation](https://adamamer20.github.io/lovebug/)** - Complete documentation site
* **[PyPI Package](https://pypi.org/project/lovebug/)** - Official package releases
* **[GitHub Repository](https://github.com/adamamer20/lovebug)** - Source code and development
* **[Issue Tracker](https://github.com/adamamer20/lovebug/issues)** - Bug reports and feature requests

For research collaboration or technical support, please open an issue on GitHub or contact the development team.

Happy evolutionary modeling! 🐞🧬✨
