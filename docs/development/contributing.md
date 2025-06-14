# Contributing

Thank you for considering contributing to LoveBug!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lovebug.git
   cd lovebug
   ```

3. Install development dependencies:
   ```bash
   uv pip install -e .[dev]
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with type checking
DEV_TYPECHECK=1 pytest

# Run with coverage
pytest --cov=lovebug
```

### Code Quality

```bash
# Lint and format
ruff check .
ruff format .

# Run pre-commit on all files
pre-commit run --all-files
```

### Documentation

```bash
# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

## Environment Variables

- `DEV_TYPECHECK=1`: Enable runtime type checking with typeguard
