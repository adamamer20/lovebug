# LoveBug Documentation

Welcome to the LoveBug documentation! LoveBug is an agent-based model (ABM) of sexual selection and mating-preference co-evolution, built with Mesa-Frames and Polars for high-performance evolutionary simulations.

## What is LoveBug?

LoveBug simulates large populations (100k+ individuals) of digital "love-bugs" whose genomes encode display traits, mate preferences, and choosiness thresholds. Through genetic crossover, mutation, and optional cultural learning mechanisms, the model explores emergent evolutionary dynamics including:

- **Sexual selection**: Fisher-Lande-Kirkpatrick runaway processes
- **Assortative mating**: Preference-trait coevolution patterns
- **Cultural-genetic interactions**: Social learning effects on mate choice
- **Population genetics**: Drift vs. selection dynamics
- **Speciation processes**: Reproductive isolation emergence

## Getting Started

### Quick Navigation

| Topic | Description |
|-------|-------------|
| **[Installation](installation.md)** | Complete setup guide for users and developers |
| **[API Reference](api.md)** | Detailed API documentation and examples |
| **[Contributing](development/contributing.md)** | Development workflow and contribution guidelines |

### Basic Usage

```python
from lovebug import LoveModel, LayerActivationConfig

# Basic genetic-only simulation
model = LoveModel(population_size=5000)
model.run_model()

# With cultural learning layer
config = LayerActivationConfig(cultural_layer=True)
model = LoveModel(population_size=5000, layer_config=config)
model.run_model()

print(f"Final population: {len(model.agents)}")
```

### Key Examples

- **Layer 2 Demo**: [`examples/layer2_demo.py`](https://github.com/adamamer20/lovebug/blob/main/examples/layer2_demo.py) - Cultural-genetic coevolution
- **Lande-Kirkpatrick Demo**: [`examples/lande_kirkpatrick_demo.py`](https://github.com/adamamer20/lovebug/blob/main/examples/lande_kirkpatrick_demo.py) - Classic sexual selection
- **Interactive Notebooks**: [`notebooks/`](https://github.com/adamamer20/lovebug/tree/main/notebooks) - Interactive tutorials and analysis

## Core Features

### High-Performance Architecture

- **Vectorized operations**: All agents in single Polars DataFrame
- **Genetic encoding**: 32-bit packed genomes for efficiency
- **Mesa-Frames integration**: Compatible with existing ABM tools
- **Scalability**: Designed for populations of 100k+ individuals

### Scientific Focus

- **Research-oriented**: Built for sexual selection and evolutionary dynamics research
- **Configurable mechanisms**: Toggle genetic vs. cultural vs. combined evolution
- **Publication-ready**: Integrated with academic paper workflow
- **Reproducible**: Comprehensive testing and version control

### Extensible Design

- **Modular architecture**: Layered approach for different evolutionary mechanisms
- **Custom backends**: Visualization system with multiple output formats
- **Parameter exploration**: Built-in support for systematic parameter studies
- **Integration-friendly**: Works with Jupyter, Marimo, and other scientific computing tools

## Model Architecture

### Genetic Layer (Core)

The foundational layer implements classical population genetics:

- **Genome structure**: 32-bit integers encoding display traits, preferences, and thresholds
- **Mate choice**: Hamming distance-based compatibility scoring
- **Reproduction**: Uniform crossover with per-bit mutation
- **Population dynamics**: Energy decay, aging, and density-dependent effects

### Cultural Layer (Optional)

Layer 2 adds social learning mechanisms:

- **Cultural transmission**: Non-genetic preference inheritance
- **Social networks**: Structured learning interactions
- **Cultural-genetic interactions**: Feedback between learning and evolution
- **Configurable algorithms**: Multiple social learning models

### Visualization System

Post-hoc analysis and visualization tools:

- **Publication quality**: Static figures for academic papers
- **Interactive exploration**: Dynamic parameter exploration
- **Multiple backends**: Matplotlib, Plotly, with extensions for animation
- **Academic integration**: Designed for research workflow compatibility

## Research Applications

LoveBug is particularly suited for investigating:

1. **Sexual selection theory**: Testing Fisher-Lande-Kirkpatrick predictions
2. **Cultural evolution**: Gene-culture coevolution in mate choice
3. **Population genetics**: Finite population effects on selection
4. **Comparative studies**: Cross-species evolutionary pattern analysis
5. **Educational applications**: Teaching evolutionary principles through simulation

## Community and Support

### Getting Help

- **[GitHub Issues](https://github.com/adamamer20/lovebug/issues)**: Bug reports and feature requests
- **[GitHub Discussions](https://github.com/adamamer20/lovebug/discussions)**: Questions and community support
- **[Documentation](https://adamamer20.github.io/lovebug/)**: This site with comprehensive guides

### Contributing

We welcome contributions from the scientific computing and evolutionary biology communities:

- **Code contributions**: See our [Contributing Guide](development/contributing.md)
- **Documentation**: Help improve and expand documentation
- **Research applications**: Share your research uses and findings
- **Feature requests**: Suggest improvements for scientific workflows

### Citation

If you use LoveBug in your research, please cite:

```bibtex
@misc{lovebug2025,
  title   = {LoveBug: An agent‑based model of sexual selection and cultural-genetic coevolution},
  author  = {Adam Amer},
  year    = {2025},
  howpublished = {GitHub repository},
  url     = {https://github.com/adamamer20/lovebug}
}
```

## License

LoveBug is released under the MIT License, making it freely available for research, education, and commercial use. See the [LICENSE](https://github.com/adamamer20/lovebug/blob/main/LICENSE) file for details.

---

*Ready to start? Head to the [Installation Guide](installation.md) to set up LoveBug for your research!*
