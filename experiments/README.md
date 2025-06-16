# Experiments Directory

This directory contains the unified experiment runner and configuration for evolutionary simulation experiments.

## Structure

```
experiments/
├── runner.py              # Unified experiment runner
├── config.toml            # Configuration file
├── results/               # Experiment results
│   ├── layer1/           # Layer 1 (Lande-Kirkpatrick) results
│   ├── layer2/           # Layer 2 (Cultural transmission) results
│   ├── combined/         # Multi-layer experiment results
│   └── archived/         # Historical results
├── logs/                 # Experiment logs
│   ├── current/          # Active experiment logs
│   └── archived/         # Historical logs
└── README.md             # This file
```

## Quick Start

### Basic Usage

Run experiments with default configuration:
```bash
uv run python experiments/runner.py --config experiments/config.toml
```

### Experiment Types

**Layer 1 (Lande-Kirkpatrick) only:**
```bash
uv run python experiments/runner.py --config experiments/config.toml --type layer1
```

**Layer 2 (Cultural transmission) only:**
```bash
uv run python experiments/runner.py --config experiments/config.toml --type layer2
```

**Combined experiments (default):**
```bash
uv run python experiments/runner.py --config experiments/config.toml --type combined
```

### Resource Management

Customize computational resources:
```bash
# Use 16 workers, run for max 6 hours
uv run python experiments/runner.py --config experiments/config.toml --workers 16 --hours 6

# Run with different replication count
uv run python experiments/runner.py --config experiments/config.toml --replications 5
```

## Configuration

The `config.toml` file contains all experiment parameters organized into sections:

### Runner Configuration
- `experiment_type`: "layer1", "layer2", or "combined"
- `n_workers`: Number of parallel workers (default: 90% of CPU cores)
- `max_duration_hours`: Maximum experiment duration
- `memory_limit_gb`: Memory usage limit
- `stochastic_replications`: Number of replications per experiment

### Output Configuration
- `results_dir`: Directory for results (organized by experiment type)
- `logs_dir`: Directory for log files
- `batch_size`: Batch size for result processing

### Layer 1 Parameters (Lande-Kirkpatrick Model)
- `layer1_n_generations`: Number of generations to simulate
- `layer1_pop_size`: Population size
- `layer1_h2_trait`: Heritability of display trait
- `layer1_h2_preference`: Heritability of preference
- `layer1_genetic_correlation`: Genetic correlation between trait and preference
- `layer1_selection_strength`: Strength of natural selection
- `layer1_preference_cost`: Cost of preference expression
- `layer1_mutation_variance`: Mutation variance

### Layer 2 Parameters (Cultural Transmission)
- `layer2_innovation_rate`: Rate of cultural innovation
- `layer2_horizontal_transmission_rate`: Rate of horizontal cultural transmission
- `layer2_oblique_transmission_rate`: Rate of oblique cultural transmission
- `layer2_network_type`: Social network topology ("random", "grid", "scale_free", "small_world")
- `layer2_network_connectivity`: Network connectivity parameter
- `layer2_cultural_memory_size`: Size of cultural memory
- `layer2_n_agents`: Number of agents in cultural simulation

## Results

Results are automatically organized by experiment type:

- **Layer 1 results**: `results/layer1/` - Contains Lande-Kirkpatrick simulation outcomes
- **Layer 2 results**: `results/layer2/` - Contains cultural transmission simulation outcomes
- **Combined results**: `results/combined/` - Contains multi-layer experiment outcomes

Each result set includes:
- `.parquet` file with detailed experiment data
- `.json` file with experiment summary and metadata

## Logging

Comprehensive logging with timestamps to `logs/current/`:
- Experiment progress and status
- Resource usage monitoring
- Error reporting and debugging information
- Performance metrics

Historical logs are archived in `logs/archived/`.

## Features

### Built-in Capabilities
- **Parallel Processing**: Automatic multi-core utilization
- **Progress Tracking**: Real-time progress display with Rich
- **Resource Monitoring**: Memory and time limit enforcement
- **Checkpointing**: Automatic result saving at intervals
- **Error Handling**: Graceful failure recovery and reporting
- **Result Organization**: Automatic categorization by experiment type

### Signal Handling
The runner gracefully handles interruption signals (Ctrl+C, SIGTERM) and ensures:
- Current results are saved
- Cleanup of resources
- Proper shutdown of worker processes

### Memory Management
- Automatic memory usage monitoring
- Configurable memory limits
- Process-level isolation to prevent memory leaks

## Examples

### Custom Configuration
Create a custom config file for specific research questions:

```toml
[runner]
experiment_type = "layer1"
n_workers = 8
max_duration_hours = 2.0
stochastic_replications = 50

[layer1]
n_generations = 1000
pop_size = 2000
h2_trait = 0.8
h2_preference = 0.7
genetic_correlation = 0.3
```

### Batch Processing
Run multiple experiment configurations:

```bash
# Run different experiment types sequentially
uv run python experiments/runner.py --config experiments/config.toml --type layer1
uv run python experiments/runner.py --config experiments/config.toml --type layer2

# Or run with different parameters
uv run python experiments/runner.py --config experiments/config.toml --replications 100 --hours 12
```

## Troubleshooting

### Common Issues

**Memory errors:**
- Reduce `n_workers` or `memory_limit_gb` in config
- Check system memory availability with `free -h`

**Slow performance:**
- Increase `n_workers` if CPU cores are available
- Reduce `stochastic_replications` for faster results
- Use `--type layer1` for simpler experiments

**Configuration errors:**
- Validate TOML syntax with `python -c "import tomllib; tomllib.load(open('experiments/config.toml', 'rb'))"`
- Check parameter ranges in config file
- Ensure all required sections are present

### Log Analysis
Check recent logs for detailed error information:
```bash
tail -f experiments/logs/current/experiments_*.log
```

## Migration from Old Structure

If migrating from the previous nested directory structure:
1. Results from `experiments/experiments/mega_results/` have been moved to `experiments/results/combined/`
2. Logs from `experiments/experiments/logs/` have been moved to `experiments/logs/archived/`
3. Old runner scripts (`mega_parallel_runner.py`, `run_mega_logged.py`) have been consolidated into `runner.py`
4. Configuration files have been unified into `config.toml`

The new structure maintains compatibility with existing analysis scripts that read from the results directories.
