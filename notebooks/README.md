# Interactive Notebooks 📚

Interactive research notebooks for exploring LoveBug's evolutionary models with real-time parameter exploration and visualization.

> **Note**: For installation and general LoveBug usage, see the [main documentation](../README.md) and [installation guide](../docs/installation.md).

## Available Notebooks

### [`Lande-Kirkpatrick.py`](Lande-Kirkpatrick.py)
**Classic sexual selection model** - Interactive exploration of trait-preference coevolution with real-time visualization, phase portraits, and parameter sensitivity analysis.

### [`Layer2-Social-Learning.py`](Layer2-Social-Learning.py)
**Cultural transmission mechanisms** - Advanced social learning with oblique/horizontal transmission, cultural innovation, memory, and social network effects.

## Usage

```bash
# Install dependencies (see installation guide for details)
uv pip install -e .[dev]

# Direct execution
uv run python notebooks/Lande-Kirkpatrick.py
uv run python notebooks/Layer2-Social-Learning.py

# Using marimo server
uv run marimo run notebooks/Lande-Kirkpatrick.py
```

## Features

Both notebooks include:
- **Interactive parameters**: Real-time sliders for model configuration
- **Live visualization**: Publication-quality plots with dynamic updates
- **Data export**: Results saved as Parquet files with configuration metadata
- **Performance optimization**: Efficient simulation engines with progress monitoring

## Data Output

Results are automatically saved to `output/` with organized directory structure:
- **Population data**: Time series metrics (`.parquet`)
- **Configuration**: Parameter specifications (`.json`)
- **Visualizations**: Generated plots (`.png`)

## Performance Guidelines

| Population | Generations | Memory | Runtime |
|------------|-------------|--------|---------|
| 200-500    | 50-100      | ~100MB | ~1min   |
| 1000       | 200         | ~300MB | ~5min   |
| 2000+      | 500+        | ~1GB   | ~30min  |

## Troubleshooting

- **Import errors**: Ensure installation with `uv pip install -e .[dev]`
- **Memory issues**: Reduce population size or generations
- **Slow performance**: Check network settings and visualization frequency

For detailed documentation and additional examples, see the [main documentation](../docs/index.md).
