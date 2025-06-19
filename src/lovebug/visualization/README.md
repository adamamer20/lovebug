# LoveBug Visualization System

A comprehensive post-hoc visualization architecture for sexual selection research, designed for creating publication-quality static visualizations and interactive explorations of LoveBug agent-based model results.

## Overview

The visualization system follows a modular architecture with clear separation between data processing, chart logic, and rendering backends. This design enables:

- **Post-hoc analysis**: Run simulations first, then generate multiple visualization variants
- **Publication quality**: High-DPI static outputs suitable for academic papers
- **Extensibility**: Easy to add new chart types and rendering backends
- **Performance**: Pre-computed data storage with efficient loading

## Architecture

```
src/lovebug/visualization/
├── data.py              # DataCollector and DataLoader
├── core.py              # VisualizationEngine and ChartFactory
├── charts/              # Chart type implementations
│   ├── base.py         # BaseChart abstract class
│   └── trajectory.py   # TrajectoryChart for evolutionary dynamics
├── backends/            # Rendering backend implementations
│   ├── base.py         # BaseBackend abstract class
│   └── static.py       # StaticBackend (Matplotlib)
└── README.md           # This file
```

## Quick Start

### 1. Basic Usage

```python
from lovebug.visualization import VisualizationEngine, DataCollector
from lovebug import UnifiedLoveModel

# Collect data during simulation
model = UnifiedLoveModel(population_size=1000)
collector = DataCollector()

for step in range(100):
    model.step()
    collector.collect_step_data(model, step)

# Save simulation data
collector.save_run_data('simulation_results.parquet')

# Create visualizations
engine = VisualizationEngine('simulation_results.parquet')

# Generate publication-quality trajectory plot
fig = engine.create_chart(
    chart_type='trajectory',
    backend='static',
    config={
        'title': 'Trait-Preference Coevolution',
        'trajectory_type': 'trait_preference',
        'show_confidence_bands': True
    }
)
```

### 2. Available Chart Types

#### TrajectoryChart
Visualizes evolutionary dynamics over time with multiple trajectory types:

- **`trait_preference`**: Display trait and mate preference evolution
- **`covariance`**: Genetic covariance dynamics with runaway thresholds
- **`cultural_genetic`**: Cultural vs genetic preference comparison
- **`population_dynamics`**: Population size and demographic metrics

```python
# Trait-preference coevolution
fig = engine.create_chart('trajectory', 'static', {
    'trajectory_type': 'trait_preference',
    'title': 'Sexual Selection Dynamics',
    'show_confidence_bands': True
})

# Genetic covariance evolution
fig = engine.create_chart('trajectory', 'static', {
    'trajectory_type': 'covariance',
    'title': 'Runaway Sexual Selection'
})
```

### 3. Backend Options

#### StaticBackend (Matplotlib)
- **Purpose**: Publication-quality static outputs
- **Formats**: PNG, PDF, SVG, EPS
- **Features**: High-DPI, academic styling, LaTeX support
- **Requirements**: `matplotlib`, `seaborn`

```python
# Create and save static chart
fig = engine.create_chart('trajectory', 'static', config)

# Save in multiple formats
from lovebug.visualization.backends.static import StaticBackend
backend = StaticBackend()
backend.save_output(fig, 'trajectory.png', dpi=300)
backend.save_output(fig, 'trajectory.pdf')
```

## Data Collection System

### DataCollector

Integrates with Mesa-Frames to collect essential metrics during simulation:

```python
collector = DataCollector()

# Set experiment metadata
collector.set_metadata(
    population_size=1000,
    mutation_rate=1e-4,
    experiment_type="mechanism_comparison"
)

# Collect data each step
for step in range(simulation_steps):
    model.step()
    collector.collect_step_data(model, step)

# Save to efficient Parquet format
collector.save_run_data('results.parquet')
```

**Collected Metrics:**
- Population statistics (size, mean age, mean energy)
- Genetic metrics (trait/preference means and variances)
- Sexual selection metrics (genetic covariance, mating success)
- Cultural evolution metrics (cultural-genetic distance)

### DataLoader

Efficient loading and preprocessing of saved simulation data:

```python
loader = DataLoader('results.parquet')

# Access data and metadata
data = loader.data  # Polars DataFrame
metadata = loader.metadata  # Dict with experiment info

# Get data subsets
recent_data = loader.get_time_range(start_step=50)
final_state = loader.get_final_state()

# Summary statistics
summary = loader.get_summary_stats()
```

## Configuration Options

### Chart Configuration

```python
config = {
    # Appearance
    'title': 'Chart Title',
    'figsize': (12, 8),
    'dpi': 300,
    'color_palette': 'academic',  # 'academic', 'viridis', 'sexual_selection'

    # Data filtering
    'start_step': 0,
    'end_step': 100,

    # Display options
    'show_confidence_bands': True,
    'show_legend': True,
    'show_grid': True,

    # Chart-specific options
    'trajectory_type': 'trait_preference',  # For trajectory charts
    'x_label': 'Generation',
    'y_label': 'Value'
}
```

### Academic Styling

The static backend applies professional academic styling:

- High-DPI output (300+ DPI)
- Serif fonts for publication
- Clean grid and axis styling
- Colorblind-friendly palettes
- Consistent formatting across chart types

## Advanced Usage

### Custom Chart Types

Extend the system by creating new chart classes:

```python
from lovebug.visualization.charts.base import BaseChart

class MyCustomChart(BaseChart):
    def _validate_config(self):
        # Validate chart-specific configuration
        pass

    def _preprocess_data(self):
        # Prepare data for rendering
        pass

    def get_data_for_rendering(self):
        # Return data dictionary for backends
        return {
            'data': self._processed_data,
            'title': self.get_title(),
            # ... other rendering data
        }

# Register with chart factory
engine.chart_factory.register_chart('custom', MyCustomChart)
```

### Multiple Simulation Comparison

Compare results across different parameter settings:

```python
# Load multiple simulation results
results = []
for param_value in [0.1, 0.5, 1.0]:
    engine = VisualizationEngine(f'results_{param_value}.parquet')
    results.append(engine.get_data_summary())

# Create comparison visualization
# (Implementation depends on specific comparison needs)
```

## Examples

### Complete Workflow Example

```python
# examples/complete_workflow.py
from lovebug.visualization import VisualizationEngine, DataCollector
from lovebug import UnifiedLoveModel

# 1. Run simulation with data collection
model = UnifiedLoveModel(population_size=5000)
collector = DataCollector()
collector.set_metadata(experiment="baseline_run")

for step in range(200):
    model.step()
    collector.collect_step_data(model, step)

collector.save_run_data('baseline_simulation.parquet')

# 2. Create visualization suite
engine = VisualizationEngine('baseline_simulation.parquet')

# Publication figures
charts = [
    ('trajectory', {'trajectory_type': 'trait_preference'}),
    ('trajectory', {'trajectory_type': 'covariance'}),
    ('trajectory', {'trajectory_type': 'population_dynamics'})
]

for chart_type, config in charts:
    fig = engine.create_chart(chart_type, 'static', config)
    # Save each chart...
```

## Installation and Dependencies

### Core Dependencies
```bash
pip install polars numpy mesa-frames
```

### Visualization Dependencies
```bash
pip install matplotlib seaborn  # For static backend
pip install plotly             # For interactive backend (future)
```

### Optional Dependencies
```bash
pip install manim             # For animation backend (future)
```

## Testing

Run the test suite to verify installation:

```bash
# Basic functionality test (no matplotlib required)
python test_visualization.py

# Full demo with matplotlib
python examples/visualization_demo.py
```

## Performance Considerations

### Data Storage
- **Parquet format**: ~10x smaller than CSV, faster loading
- **Columnar storage**: Efficient for time series analysis
- **Metadata separation**: JSON sidecar files for run information

### Memory Efficiency
- **Lazy loading**: Data loaded only when needed
- **Polars backend**: Efficient DataFrame operations
- **Streaming support**: Process large datasets in chunks

### Visualization Performance
- **Pre-computation**: Statistical aggregations cached
- **Vectorized operations**: Fast data transformations
- **Progressive rendering**: Interactive charts load incrementally

## Extension Points

The modular architecture supports easy extension:

1. **New Chart Types**: Inherit from `BaseChart`
2. **New Backends**: Inherit from `BaseBackend`
3. **Custom Data Processing**: Extend `DataCollector`
4. **Platform Integration**: Add platform-specific wrappers

## Future Roadmap

### Phase 2: Interactive Backend
- Plotly-based interactive visualizations
- Parameter sliders and controls
- Marimo notebook integration

### Phase 3: Advanced Features
- Manim-based mathematical animations
- Observable Plot web components
- Paper-to-website transformation

## Support

For questions and issues:
1. Check the examples in `examples/`
2. Run the test suite: `python test_visualization.py`
3. Review the architecture documentation in `docs/visualization-architecture.md`
