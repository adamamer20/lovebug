# Love‑Bugs 🐞💘

[![CI](https://github.com/adamamer20/lovebug/workflows/CI/badge.svg)](https://github.com/adamamer20/lovebug/actions/workflows/ci.yml)
[![Documentation](https://github.com/adamamer20/lovebug/workflows/Documentation/badge.svg)](https://adamamer20.github.io/lovebug/)
[![PyPI version](https://badge.fury.io/py/lovebug.svg)](https://badge.fury.io/py/lovebug)
[![Python versions](https://img.shields.io/pypi/pyversions/lovebug.svg)](https://pypi.org/project/lovebug/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An agent‑based model (ABM) of sexual selection and mating‑preference co‑evolution, built with [Mesa‑Frames](https://github.com/projectmesa/mesa-frames) + [Polars](https://pola.rs).**

---

## 📜 Project Aim

Simulate a large population of digital "love‑bugs" whose *genomes* encode:

1. **Display traits** (what others see),
2. **Mate preferences** (what they like),
3. **Choosiness threshold** (how picky they are).

At every step, bugs move, court potential partners, and—if mutual acceptance criteria are met—produce offspring via genetic crossover ★ mutation. The result is an emergent arms‑race of display fashions and evolving preferences, letting you explore classic questions in *sexual selection*, *assortative mating*, and *speciation* from a computational‑evolutionary viewpoint.

---

## ✨ Key Features

* **Vectorised core**: All agents live in a single Polars `DataFrame`; 100 k+ individuals in pure Python.
* **Genome layout**: 32‑bit unsigned int → `[15‑0 display] [23‑16 preference] [31‑24 threshold]`.
* **Mutual mate choice** via fast Hamming‑similarity check.
* **Uniform crossover** + per‑bit mutation.
* **Energy decay & ageing** stop unbounded growth.
* **Drop‑in Mesa‑Frames compatibility**: use `BatchRunner`, collectors, grid extensions, etc.
* **Test scaffold**: simple `pytest` examples to ensure basic invariants (optional).

---

## 📦 Installation

```bash
# 1. Clone the repo
$ git clone https://github.com/adamamer20/lovebug.git && cd lovebug

# 2. Create env (conda, venv, hatch… your call)
$ python -m venv .venv && source .venv/bin/activate

# 3. Install deps
$ uv pip install -e .[dev]  # mesa‑frames, polars, numpy, beartype, pytest
```

> **Note**: Polars toggles SIMD and multi‑threading automatically; on Apple Silicon & modern x86 it screams. 🏎️

---

## 🚀 Quick‑Start

```bash
# Run 5 000 bugs for 200 steps
$ uv run python -m lovebug.model
Final population: 8124
```

Graphs & CSVs drop into `outputs/` (hook up your own collector or use the sample Jupyter notebooks).

---

## 🧬 Model In‑Depth

| Component             | Description                                                                                                                |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Genome**            | 32‑bit `uint32` segmented into display / preference / threshold.                                                           |
| **Mate choice**       | Both partners compute `similarity = 16 – bitcount(display ⊕ preference_partner)`; they accept if `similarity ≥ threshold`. |
| **Reproduction**      | Uniform crossover (bit‑mask blend), then independent per‑bit mutation `μ`.                                                 |
| **Energy**            | Each tick: `energy -= 0.2`; death at `energy ≤ 0` or `age ≥ 100`.                                                          |
| **Population update** | Fully vectorised; offspring appended to the shared frame.                                                                  |

### Tunable Parameters *(see top of `src/lovebug/model.py`)*

* `MUTATION_RATE` — per‑bit flip probability.
* `ENERGY_DECAY`, `MAX_AGE`.
* Later: `GRID_SIZE`, `VISION_RADIUS`, `COURTSHIP_COST`, etc.

---

## 🛠️ Extending

1. **Spatiality** – switch random pairing to neighbour queries on a `MultiGridDF` for local mating.
2. **Resource patches** – join a `resources` DF each tick and replenish energy when on food.
3. **Sexual dimorphism** – split genome layout by sex;
4. **Cultural learning** – add non‑genetic preference drift each generation.

PRs welcome—open an issue to discuss!

---

## 🤝 Contributing

| Step   | Command         |
| ------ | --------------- |
| Format | `ruff format .` |
| Lint   | `ruff check .`  |
| Tests  | `pytest -q`     |

We welcome contributions! Please see our [Contributing Guide](https://adamamer20.github.io/lovebug/development/contributing/) for details.

---

## 📄 License

MIT © 2025 Your Name / Lab / Org. See `LICENSE`.

---

## 📚 Citation

If you use Love‑Bugs in academic work:

```text
@misc{lovebugs2025,
  title   = {Love‑Bugs: an agent‑based model of sexual selection},
  author  = {Your Name},
  year    = {2025},
  howpublished = {GitHub repository},
  url     = {https://github.com/your‑org/lovebugs}
}
```

Happy bug‑breeding! 🐞🎉

## Links

* [Documentation](https://adamamer20.github.io/lovebug/)
* [PyPI Package](https://pypi.org/project/lovebug/)
* [Source Code](https://github.com/adamamer20/lovebug)
* [Issue Tracker](https://github.com/adamamer20/lovebug/issues)
