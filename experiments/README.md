# Experiments Runner

High-performance experiment runner for systematic LoveBug simulations with parallel processing and comprehensive result management.

> **Note**: For general LoveBug installation and usage, see the [main documentation](../README.md) and [installation guide](../docs/installation.md).

## Quick Start

```bash
# Run experiments with default configuration
uv run python experiments/runner.py --config experiments/config.toml

# Run specific experiment types
uv run python experiments/runner.py --config experiments/config.toml --type layer1     # Lande-Kirkpatrick only
uv run python experiments/runner.py --config experiments/config.toml --type layer2     # Cultural transmission only
uv run python experiments/runner.py --config experiments/config.toml --type combined   # Multi-layer (default)

# Customize resources
uv run python experiments/runner.py --config experiments/config.toml --workers 16 --hours 6 --replications 5
```

## Directory Structure

```
experiments/
├── runner.py              # Unified experiment runner
├── config.toml            # Configuration file
├── results/               # Experiment results (auto-organized by type)
├── logs/                  # Experiment logs with timestamps
└── README.md             # This file
```

## Configuration

The [`config.toml`](config.toml) file contains all experiment parameters:

### Key Parameters
- **Runner**: `experiment_type`, `n_workers`, `max_duration_hours`, `stochastic_replications`
- **Layer 1**: Population size, heritability, genetic correlation, selection parameters
- **Layer 2**: Innovation rates, transmission rates, network topology, cultural memory

See the [config.toml](config.toml) file for complete parameter descriptions and default values.

## Results & Logging

- **Results**: Auto-organized in `results/` by experiment type (`.parquet` + `.json` files)
- **Logs**: Timestamped progress, resource usage, and error reporting in `logs/`

## Key Features

- **Parallel Processing**: Automatic multi-core utilization with configurable worker count
- **Resource Monitoring**: Memory and time limits with graceful shutdown handling
- **Progress Tracking**: Real-time display with Rich console output
- **Checkpointing**: Automatic result saving and recovery

## Troubleshooting

### Common Issues
- **Memory errors**: Reduce `n_workers` or `memory_limit_gb` in config
- **Slow performance**: Increase `n_workers` or reduce `stochastic_replications`
- **Configuration errors**: Validate TOML syntax and check parameter ranges

### Log Analysis
```bash
tail -f experiments/logs/current/experiments_*.log
```

For detailed troubleshooting and usage examples, see the [main documentation](../docs/index.md).
