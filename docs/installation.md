# Installation Guide

This guide provides comprehensive installation instructions for the LoveBug project, an agent-based model (ABM) of sexual selection and mating-preference co-evolution.

## System Requirements

- **Python**: ≥3.11 (using modern features like type hints, dataclasses)
- **Operating System**: Linux, macOS, or Windows
- **Recommended**: GPU support with CUDA 12 for enhanced performance

## Installation Methods

### Method 1: PyPI Installation (Recommended for Users)

For general usage, install the latest stable version from PyPI:

```bash
pip install lovebug
```

### Method 2: Development Installation (Recommended for Contributors)

For development work, including running experiments and contributing to the project:

```bash
# 1. Clone the repository
git clone https://github.com/adamamer20/lovebug.git
cd lovebug

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install with development dependencies
uv pip install -e .[dev]
```

#### Why UV Package Manager?

This project uses [astral-sh/uv](https://github.com/astral-sh/uv) as the primary package manager for several advantages:

- **Speed**: 10-100x faster than pip for dependency resolution and installation
- **Reliability**: Better dependency resolution with clear error messages
- **Lockfile management**: Automatic generation and maintenance of `uv.lock`
- **Environment isolation**: Built-in virtual environment management

#### Alternative Development Installation

If you prefer using pip instead of uv:

```bash
# After cloning and setting up virtual environment
pip install -e .[dev]
```

## Verification

Test your installation:

```bash
# Quick test
python -c "import lovebug; print(lovebug.__version__)"

# Run a basic simulation
uv run python -m lovebug.unified_mesa_model
# Expected output: "Final population: [some number]"
```

## Dependencies

### Core Dependencies
- **Mesa-Frames**: Agent-based modeling framework
- **Polars**: High-performance data manipulation
- **NumPy**: Numerical computing
- **Beartype**: Runtime type checking

### Development Dependencies
- **pytest**: Testing framework
- **ruff**: Code formatting and linting
- **pre-commit**: Git hooks for code quality
- **mkdocs**: Documentation generation
- **typeguard**: Runtime type checking

### Optional Dependencies

Install additional features as needed:

```bash
# Data analysis tools
uv pip install -e .[data]

# Machine learning components
uv pip install -e .[ml]
```

## Performance Notes

- **Polars** automatically toggles SIMD and multi-threading; performance is optimized on Apple Silicon and modern x86 processors
- **GPU acceleration** is available but not required; the system will error clearly if CUDA is requested but unavailable
- The vectorized core can handle **100k+ individuals** in pure Python

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're using Python ≥3.11
2. **Permission errors**: Consider using `--user` flag with pip
3. **UV not found**: Install uv first: `pip install uv`

### Getting Help

- **Documentation**: [https://adamamer20.github.io/lovebug/](https://adamamer20.github.io/lovebug/)
- **Issues**: [GitHub Issue Tracker](https://github.com/adamamer20/lovebug/issues)
- **Discussions**: [GitHub Discussions](https://github.com/adamamer20/lovebug/discussions)

## Next Steps

After installation:

1. **Quick Start**: See the [main documentation](index.md) for usage examples
2. **Development**: Read the [Contributing Guide](development/contributing.md)
3. **Examples**: Explore the `examples/` directory for demonstrations
4. **Notebooks**: Check `notebooks/` for interactive tutorials
