# LoveBug Notebooks 📚

Interactive research notebooks for exploring sexual selection models and enhanced social learning mechanisms using the LoveBug framework.

## Available Notebooks

### 1. Lande-Kirkpatrick Model (`Lande-Kirkpatrick.py`)

**Classic trait-preference coevolution model**

The foundational Lande-Kirkpatrick model demonstrating how male display traits and female mating preferences can coevolve through genetic correlations.

**Features:**
- 🧬 Interactive parameter exploration
- 📊 Real-time visualization of evolutionary dynamics
- 🎯 Phase portraits and trajectory analysis
- 📈 Selection pressure analysis
- 🔄 Parameter sensitivity comparison
- 💾 Data export for further analysis

**Key Concepts:**
- Genetic correlation between traits and preferences
- Runaway sexual selection dynamics
- Natural vs. sexual selection trade-offs
- Heritability effects on evolution

### 2. Layer 2 Social Learning (`Layer2-Social-Learning.py`)

**Enhanced cultural transmission mechanisms**

The Layer 2 research extension implementing advanced social learning mechanisms that extend beyond basic genetic inheritance.

**Features:**
- 🔄 **Oblique Transmission**: Parent-offspring cultural learning
- 🤝 **Horizontal Transmission**: Peer-to-peer learning with social networks
- 💡 **Cultural Innovation**: Random cultural mutations and creativity
- 🧠 **Cultural Memory**: Agents remember and weight past experiences
- 🕸️ **Social Networks**: Multiple topologies (small-world, scale-free, random, grid)
- 📊 **Rich Monitoring**: Real-time progress with beautiful console output
- 🔬 **Mechanism Comparison**: Side-by-side analysis of different learning systems

**Key Concepts:**
- Cultural vs. genetic evolution
- Social network effects on transmission
- Innovation and cultural diversity
- Gene-culture coevolution
- Memory and learning persistence

## Running the Notebooks

### Prerequisites

Ensure you have all dependencies installed:

```bash
# Install core dependencies
uv pip install -e .

# Install development dependencies (includes marimo)
uv pip install -e .[dev]
```

### Launch Instructions

#### Option 1: Direct execution
```bash
# Lande-Kirkpatrick model
uv run python notebooks/Lande-Kirkpatrick.py

# Layer 2 social learning
uv run python notebooks/Layer2-Social-Learning.py
```

#### Option 2: Using marimo
```bash
# Start marimo server
uv run marimo run notebooks/Lande-Kirkpatrick.py
uv run marimo run notebooks/Layer2-Social-Learning.py
```

## Notebook Structure

Both notebooks follow a consistent structure for optimal research workflow:

### 1. **Parameter Setup**
- Interactive sliders for model parameters
- Real-time parameter validation
- Configuration persistence

### 2. **Simulation Execution**
- Efficient simulation engines
- Progress monitoring with rich console output
- Error handling and validation

### 3. **Data Analysis**
- Comprehensive metrics calculation
- Statistical analysis of outcomes
- Pattern recognition and interpretation

### 4. **Visualization**
- Publication-quality plots
- Interactive parameter exploration
- Comparative analysis across conditions

### 5. **Export & Documentation**
- Data export in Parquet format
- Configuration preservation
- Theoretical framework explanation

## Research Applications

### Sexual Selection Studies
- **Mate choice copying**: Social learning of preferences
- **Cultural beauty standards**: Transmitted aesthetic preferences
- **Display innovation**: Cultural evolution of courtship behaviors

### Evolutionary Biology
- **Gene-culture coevolution**: Interaction between genetic and cultural inheritance
- **Social learning mechanisms**: Comparison of transmission pathways
- **Innovation dynamics**: Cultural mutation and creativity

### Computational Biology
- **Agent-based modeling**: Individual-based simulation approaches
- **Network effects**: Social structure impacts on evolution
- **Performance optimization**: Efficient large-scale simulations

## Data Output

### File Formats
- **Population data**: `*.parquet` - Time series of population metrics
- **Cultural events**: `*.parquet` - Individual learning event records
- **Network data**: `*.parquet` - Social network statistics over time
- **Configuration**: `*.json` - Complete parameter specifications

### Directory Structure
```
output/
├── lande_kirkpatrick/
│   ├── data/                    # Simulation results
│   ├── *.png                   # Generated plots
│   └── configs/                # Parameter configurations
└── layer2_output/
    ├── *_population.parquet    # Population time series
    ├── *_events.parquet        # Cultural learning events
    ├── *_network.parquet       # Network statistics
    └── *_config.json           # Simulation parameters
```

## Advanced Usage

### Custom Parameter Sweeps

```python
# Example: Layer 2 parameter sweep
from lovebug.layer2 import Layer2Config
from notebooks.Layer2SocialLearning import simulate_layer2_social_learning

# Define parameter ranges
innovation_rates = [0.01, 0.05, 0.1, 0.2]
network_types = ["small_world", "scale_free", "random"]

results = {}
for rate in innovation_rates:
    for network in network_types:
        config = Layer2Config(innovation_rate=rate, network_type=network)
        pop_data, events, net_data = simulate_layer2_social_learning(
            config, n_generations=100, n_agents=500, network_type=network
        )
        results[f"{network}_{rate}"] = {
            'population': pop_data,
            'events': events,
            'network': net_data
        }
```

### Integration with LoveBug Framework

Both notebooks are designed to integrate seamlessly with the broader LoveBug ecosystem:

- **Visualization system**: Uses LoveBug's publication-quality plotting
- **Data formats**: Compatible with LoveBug analysis pipelines
- **Performance**: Leverages Mesa-Frames + Polars for efficiency
- **Extensibility**: Modular design for easy customization

## Performance Considerations

### Optimization Tips

1. **Population size**: Start with 200-500 agents for interactive exploration
2. **Generations**: Use 50-100 generations for parameter exploration, 200+ for final analysis
3. **Network size**: Keep networks under 1000 nodes for visualization
4. **Memory management**: Enable data export for long simulations

### Scaling Guidelines

| Population | Generations | Memory Usage | Runtime |
|------------|-------------|--------------|---------|
| 200        | 100         | ~50 MB       | ~30s    |
| 500        | 100         | ~100 MB      | ~60s    |
| 1000       | 200         | ~300 MB      | ~5min   |
| 2000       | 500         | ~1 GB        | ~30min  |

## Troubleshooting

### Common Issues

**Import errors**: Ensure LoveBug is installed with `uv pip install -e .`

**Memory issues**: Reduce population size or enable data streaming

**Slow performance**: Check network connectivity settings and reduce visualization frequency

**Visualization problems**: Verify matplotlib backend and display settings

### Getting Help

- **Documentation**: See `docs/` directory for detailed API documentation
- **Examples**: Check `examples/` for additional usage patterns
- **Issues**: Report bugs and request features on GitHub
- **Community**: Join discussions in the project repository

## Contributing

We welcome contributions to improve the notebooks:

1. **Bug fixes**: Report and fix issues in simulation or visualization
2. **New features**: Add parameters, plots, or analysis methods
3. **Performance**: Optimize algorithms and data structures
4. **Documentation**: Improve explanations and examples

See `CONTRIBUTING.md` for detailed guidelines.

---

## License

The LoveBug notebooks are released under the MIT License. See `LICENSE` for details.

**Citation**: If you use these notebooks in your research, please cite:
```
Amer, A. (2025). LoveBug: Agent-based models of sexual selection and cultural transmission.
GitHub repository: https://github.com/adamamer20/lovebug
