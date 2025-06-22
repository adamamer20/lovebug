# API Reference

This page provides comprehensive API documentation for the LoveBug package. The API is organized into functional modules that support different aspects of evolutionary simulation.

## Core Model Classes

### LoveModel

::: lovebug.unified_mesa_model.LoveModel
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### LoveAgents

::: lovebug.unified_mesa_model.LoveAgents
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

## Configuration Classes

### LayerActivationConfig

::: lovebug.layer_activation.LayerActivationConfig
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### LandeKirkpatrickParams

::: lovebug.lande_kirkpatrick.LandeKirkpatrickParams
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

## Cultural Learning Layer (Layer 2)

### CulturalLayer

::: lovebug.layer2.cultural_layer.CulturalLayer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### Learning Algorithms

::: lovebug.layer2.learning_algorithms
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### Social Networks

::: lovebug.layer2.network
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

## Visualization System

### Core Visualization

::: lovebug.visualization.core
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### Data Collection

::: lovebug.visualization.data
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### Chart Types

#### Trajectory Charts

::: lovebug.visualization.charts.trajectory
    options:
      show_root_heading: true
      show_source: false
      heading_level: 5

#### Base Chart Classes

::: lovebug.visualization.charts.base
    options:
      show_root_heading: true
      show_source: false
      heading_level: 5

### Visualization Backends

#### Static Backend (Matplotlib)

::: lovebug.visualization.backends.static
    options:
      show_root_heading: true
      show_source: false
      heading_level: 5

#### Base Backend Classes

::: lovebug.visualization.backends.base
    options:
      show_root_heading: true
      show_source: false
      heading_level: 5

## Monitoring and Diagnostics

### Simulation Monitor

::: lovebug.layer2.monitoring.simulation_monitor
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

## Usage Examples

### Basic Simulation

```python
from lovebug import LoveModel

# Create a basic genetic-only simulation
model = LoveModel(
    population_size=5000,
    max_steps=200,
    mutation_rate=1e-4
)

# Run the simulation
model.run_model()

print(f"Final population: {len(model.agents)}")
print(f"Steps completed: {model.schedule.steps}")
```

### Cultural-Genetic Coevolution

```python
from lovebug import LoveModel, LayerActivationConfig

# Example: configure and run a model
layer_config = LayerActivationConfig(
    cultural_layer=True,
    # layer2_config=...  # Provide appropriate config if needed
)

model = LoveModel(
    population_size=10000,
    max_steps=500,
    layer_config=layer_config
)

model.run_model()
```

### Parameter Exploration

```python
from lovebug import LoveModel, LayerActivationConfig
import numpy as np

# Parameter sweep example
population_sizes = [1000, 5000, 10000]
mutation_rates = np.logspace(-5, -3, 5)

results = []
for pop_size in population_sizes:
    for mut_rate in mutation_rates:
        model = LoveModel(
            population_size=pop_size,
            mutation_rate=mut_rate,
            max_steps=200
        )
        model.run_model()

        results.append({
            'population_size': pop_size,
            'mutation_rate': mut_rate,
            'final_population': len(model.agents),
            'mean_display': model.agents['display'].mean(),
            'mean_preference': model.agents['preference'].mean()
        })
```

### Visualization

```python
from lovebug.visualization import VisualizationEngine
from lovebug import LoveModel

# Run simulation with data collection
model = LoveModel(population_size=5000)
model.run_model()

# Create visualization engine (when implemented)
# engine = VisualizationEngine(model.datacollector.get_model_vars_dataframe())
#
# # Generate trajectory plot
# fig = engine.create_chart(
#     chart_type='trajectory',
#     backend='static',
#     config={'title': 'Trait-Preference Evolution'}
# )
```

## Type Definitions

### Common Types

```python
from typing import Dict, List, Optional, Union, Tuple
import polars as pl
import numpy as np

# Agent data structure
AgentData = pl.DataFrame  # Contains columns: agent_id, genome, display, preference, threshold, age, energy

# Genome representation
Genome = np.uint32  # 32-bit packed genome: [display|preference|threshold]

# Configuration dictionaries
ConfigDict = Dict[str, Union[str, int, float, bool]]

# Simulation results
SimulationResults = Dict[str, Union[int, float, List, pl.DataFrame]]
```

## Error Handling

### Common Exceptions

The LoveBug package defines several custom exceptions for better error handling:

```python
# Import custom exceptions (when implemented)
# from lovebug.exceptions import (
#     LoveBugError,
#     ConfigurationError,
#     SimulationError,
#     VisualizationError
# )

# Example error handling
try:
    model = LoveModel(population_size=-100)  # Invalid parameter
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    model.run_model()
except RuntimeError as e:
    print(f"Simulation error: {e}")
```

## Performance Notes

### Memory Usage

- **Large populations**: Memory usage scales linearly with population size
- **Long simulations**: Historical data collection can consume significant memory
- **Cultural layer**: Additional memory overhead for social networks and cultural traits

### Computational Complexity

- **Mate selection**: O(n²) worst case, optimized with vectorized operations
- **Genetic operations**: O(n) for reproduction and mutation
- **Cultural learning**: O(n × k) where k is average network degree
- **Data collection**: O(n) per time step

### Optimization Tips

1. **Use appropriate population sizes**: Start with 1k-10k agents for development
2. **Profile memory usage**: Monitor with large populations (>50k agents)
3. **Vectorized operations**: All core operations use Polars for performance
4. **GPU acceleration**: Available for compatible operations (when implemented)

## Version Compatibility

### Python Version Support

- **Minimum**: Python 3.11
- **Recommended**: Python 3.12+
- **Type hints**: Uses modern type annotation syntax

### Dependency Versions

- **Mesa-Frames**: Compatible with latest stable releases
- **Polars**: Requires ≥0.20.0 for DataFrame operations
- **NumPy**: Compatible with 1.24+
- **Matplotlib**: Required for static visualization backend

## Development and Extensions

### Custom Agent Behaviors

```python
# Example: Extending the agent class (conceptual)
class CustomLoveBug(LoveAgents):
    def custom_mate_choice(self, potential_mates):
        # Custom mate selection logic
        pass
```

### Custom Learning Algorithms

```python
# Example: Custom cultural learning algorithm
def custom_social_learning(agents, network, learning_rate):
    # Implement custom social learning mechanism
    pass
```

### Plugin Architecture

The visualization system supports custom backends and chart types through a plugin architecture (see [Visualization Architecture](visualization-architecture.md) for details).

---

*This API reference is auto-generated from docstrings in the source code. For the most up-to-date information, refer to the source code or use Python's built-in `help()` function.*
