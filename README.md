# Love‑Bugs 🐞💘

[![CI](https://github.com/adamamer20/lovebug/workflows/CI/badge.svg)](https://github.com/adamamer20/lovebug/actions/workflows/ci.yml)
[![Documentation](https://github.com/adamamer20/lovebug/workflows/Documentation/badge.svg)](https://adamamer20.github.io/lovebug/)
[![PyPI version](https://badge.fury.io/py/lovebug.svg)](https://badge.fury.io/py/lovebug)
[![Python versions](https://img.shields.io/pypi/pyversions/lovebug.svg)](https://pypi.org/project/lovebug/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An agent‑based model (ABM) of sexual selection and mating‑preference co‑evolution, built with [Mesa‑Frames](https://github.com/projectmesa/mesa-frames) + [Polars](https://pola.rs).**

---

## 📜 Project Overview

LoveBug simulates large populations of digital "love‑bugs" whose *genomes* encode:

1. **Display traits** (what others see)
2. **Mate preferences** (what they like)
3. **Choosiness threshold** (how picky they are)

At every step, bugs move, court potential partners, and—if mutual acceptance criteria are met—produce offspring via genetic crossover and mutation. The result is an emergent arms‑race of display fashions and evolving preferences, enabling exploration of classic questions in *sexual selection*, *assortative mating*, and *speciation* from a computational‑evolutionary perspective.

---

## ✨ Key Features

* **Vectorized core**: All agents stored in a single Polars `DataFrame`; handles 100k+ individuals in pure Python
* **Genetic encoding**: 32‑bit unsigned int genome → `[15‑0 display] [23‑16 preference] [31‑24 threshold]`
* **Mutual mate choice**: Fast Hamming‑similarity based partner selection
* **Evolutionary mechanics**: Uniform crossover + per‑bit mutation + energy decay & aging
* **Mesa‑Frames compatibility**: Drop‑in support for `BatchRunner`, collectors, grid extensions
* **Multi-layer architecture**: Genetic evolution + cultural learning mechanisms
* **Research-focused**: Designed for sexual selection and evolutionary dynamics research

---

## 🚀 Quick Start

### Installation

```bash
# For general use
pip install lovebug

# For development (see full installation guide)
git clone https://github.com/adamamer20/lovebug.git
cd lovebug && uv pip install -e .[dev]
```

**📖 [Complete Installation Guide](https://adamamer20.github.io/lovebug/installation/)**

### Basic Usage

```python
from lovebug import LoveModel

# Create and run a basic simulation
model = LoveModel(population_size=5000, max_steps=200)
model.run_model()

print(f"Final population: {len(model.agents)}")
```

### Command Line

```bash
# Run default simulation
uv run python -m lovebug.unified_mesa_model

# Explore examples
python examples/layer2_demo.py
```

---

## 🧬 Model Architecture

| Component | Description |
|-----------|-------------|
| **Genome** | 32‑bit `uint32` with bit-packed display/preference/threshold traits |
| **Mate Choice** | Hamming similarity: `similarity = 16 – bitcount(display ⊕ preference_partner)` |
| **Reproduction** | Uniform crossover with independent per‑bit mutation rate `μ` |
| **Population Dynamics** | Energy decay (`energy -= 0.2` per tick) and maximum age limits |
| **Cultural Layer** | Optional social learning overlaid on genetic evolution |
| **Vectorized Updates** | Fully vectorized operations using Polars for performance |

### Core Parameters

Key tunable parameters (see [`LoveModel`](src/lovebug/unified_mesa_model.py)):

* `population_size`: Initial population size
* `mutation_rate`: Per-bit genetic mutation probability
* `energy_decay`: Energy cost per time step
* `max_age`: Maximum lifespan
* Cultural learning parameters via [`Layer2Config`](src/lovebug/layer2/config.py)

---

## 📚 Documentation

* **[Full Documentation](https://adamamer20.github.io/lovebug/)** - Complete guide and API reference
* **[Installation Guide](https://adamamer20.github.io/lovebug/installation/)** - Detailed setup instructions
* **[Contributing Guide](https://adamamer20.github.io/lovebug/development/contributing/)** - Development workflow
* **[Examples](examples/)** - Demonstration scripts and notebooks
* **[Research Paper](paper/paper.qmd)** - Scientific background and methodology

---

## 🔬 Research Applications

LoveBug is designed for studying:

* **Sexual selection dynamics**: Fisher-Lande-Kirkpatrick mechanisms
* **Assortative mating**: Preference-trait coevolution
* **Cultural-genetic interactions**: Social learning effects on mate choice
* **Population genetics**: Drift vs. selection in finite populations
* **Speciation processes**: Reproductive isolation emergence

---

## 🤝 Contributing

We welcome contributions! Key development commands:

```bash
# Code quality
ruff check . && ruff format .

# Testing
pytest -v

# Documentation
mkdocs serve
```

**📖 [Full Contributing Guide](https://adamamer20.github.io/lovebug/development/contributing/)**

---

## 📄 License & Citation

**License**: MIT © 2025 Adam Amer. See [`LICENSE`](LICENSE).

**Citation**: If you use LoveBug in academic work:

```bibtex
@misc{lovebug2025,
  title   = {LoveBug: An agent‑based model of sexual selection and cultural-genetic coevolution},
  author  = {Adam Amer},
  year    = {2025},
  howpublished = {GitHub repository},
  url     = {https://github.com/adamamer20/lovebug}
}
```

---

## 🔗 Links

* **[Documentation](https://adamamer20.github.io/lovebug/)** - Complete documentation site
* **[PyPI Package](https://pypi.org/project/lovebug/)** - Official package releases
* **[GitHub Repository](https://github.com/adamamer20/lovebug)** - Source code
* **[Issue Tracker](https://github.com/adamamer20/lovebug/issues)** - Bug reports and feature requests

Happy bug‑breeding! 🐞🎉
