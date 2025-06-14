# LoveBug

[![CI](https://github.com/adamamer20/lovebug/workflows/CI/badge.svg)](https://github.com/adamamer20/lovebug/actions/workflows/ci.yml)
[![Documentation](https://github.com/adamamer20/lovebug/workflows/Documentation/badge.svg)](https://adamamer20.github.io/lovebug/)
[![PyPI version](https://badge.fury.io/py/lovebug.svg)](https://badge.fury.io/py/lovebug)
[![Python versions](https://img.shields.io/pypi/pyversions/lovebug.svg)](https://pypi.org/project/lovebug/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An agent‑based model (ABM) of sexual selection and mating‑preference co‑evolution, built with Mesa‑Frames + Polars.

## Features

- ✨ **Modern Python**: Built with Python 3.9+ support
- 🚀 **Fast Development**: Powered by `uv` package manager
- 🛡️ **Type Safe**: Full type hints with runtime validation via `typeguard`
- 🧪 **Well Tested**: Comprehensive test suite with pytest
- 📚 **Documentation**: Beautiful docs with Material for MkDocs
- 🔧 **Developer Experience**: Pre-commit hooks, automated formatting, and linting
- 🏗️ **CI/CD Ready**: GitHub Actions workflows for testing, building, and publishing

## Installation

Install from PyPI:

```bash
pip install lovebug
```

Or with `uv`:

```bash
uv add lovebug
```

## Quick Start

```python
import lovebug

# Your code here
print(f"LoveBug version: {lovebug.__version__}")
```

## Development

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/adamamer20/lovebug.git
   cd lovebug
   ```

2. Install dependencies:
   ```bash
   uv pip install -e .[dev]
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Development Commands

```bash
# Run tests
pytest

# Run tests with type checking
DEV_TYPECHECK=1 pytest

# Run linting and formatting
ruff check .
ruff format .

# Run pre-commit on all files
pre-commit run --all-files

# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build
```

### Environment Variables

Create a `.env` file based on `.env.example`:

- `DEV_TYPECHECK=1`: Enable runtime type checking with typeguard
- `LOG_LEVEL=INFO`: Set logging level

## Contributing

We welcome contributions! Please see our [Contributing Guide](https://adamamer20.github.io/lovebug/development/contributing/) for details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Links

- [Documentation](https://adamamer20.github.io/lovebug/)
- [PyPI Package](https://pypi.org/project/lovebug/)
- [Source Code](https://github.com/adamamer20/lovebug)
- [Issue Tracker](https://github.com/adamamer20/lovebug/issues)
