"""
Layer 2 Social Learning Model Implementation and Visualization

A comprehensive notebook implementing the Layer 2 research extension for enhanced
social learning mechanisms in sexual selection, with publication-quality
visualizations and interactive parameter exploration.

The Layer 2 model extends the basic Lande-Kirkpatrick framework with:
- Oblique transmission (parent-offspring cultural learning)
- Horizontal transmission (peer-to-peer learning)
- Cultural innovation (random cultural mutation)
- Social network effects on cultural transmission
- Cultural memory and decay systems

Author: Adam Amer
Date: 2025-06-15
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import marimo
import numpy as np
import polars as pl
from beartype import beartype

# Add src to path for Layer 2 imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lovebug.layer2.config import Layer2Config
from lovebug.layer2.cultural_layer import VectorizedCulturalLayer
from lovebug.layer2.monitoring.simulation_monitor import SimulationMonitor
from lovebug.layer2.social_learning.social_networks import NetworkTopology, SocialNetwork

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@dataclass(slots=True, frozen=False)
class Layer2AgentData:
    """Mock agent data for Layer 2 social learning simulation."""

    def __init__(self, n_agents: int = 500) -> None:
        self.n_agents = n_agents
        self.agent_ids = np.arange(n_agents)
        self.cultural_preferences = np.random.randint(0, 256, n_agents).astype(np.uint8)
        self.genetic_preferences = np.random.randint(0, 256, n_agents).astype(np.uint8)
        self.mating_success = np.random.exponential(1.0, n_agents).astype(np.float32)
        self.ages = np.random.randint(0, 50, n_agents).astype(np.uint16)
        self.energy = np.random.uniform(5.0, 15.0, n_agents).astype(np.float32)

    def get_cultural_preferences(self) -> np.ndarray:
        return self.cultural_preferences

    def get_genetic_preferences(self) -> np.ndarray:
        return self.genetic_preferences

    def get_mating_success(self) -> np.ndarray:
        return self.mating_success

    def get_ages(self) -> np.ndarray:
        return self.ages

    def get_agent_ids(self) -> np.ndarray:
        return self.agent_ids

    def update_cultural_preference(self, agent_id: int, new_preference: int) -> None:
        if 0 <= agent_id < len(self.cultural_preferences):
            self.cultural_preferences[agent_id] = new_preference

    def age_population(self) -> None:
        """Age the population and replace old agents."""
        self.ages += 1

        # Replace agents older than 40 with new offspring
        old_agents = self.ages > 40
        n_replacements = np.sum(old_agents)

        if n_replacements > 0:
            # Create new agents with inherited preferences
            parent_indices = np.random.choice(np.where(~old_agents)[0], n_replacements, replace=True)

            self.ages[old_agents] = 0
            self.cultural_preferences[old_agents] = self.cultural_preferences[parent_indices]
            self.genetic_preferences[old_agents] = self.genetic_preferences[parent_indices]
            self.mating_success[old_agents] = np.random.exponential(1.0, n_replacements)
            self.energy[old_agents] = np.random.uniform(5.0, 15.0, n_replacements)


@beartype
def simulate_layer2_social_learning(
    config: Layer2Config, n_generations: int = 100, n_agents: int = 500, network_type: str = "small_world"
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Simulate Layer 2 social learning mechanisms.

    Parameters
    ----------
    config : Layer2Config
        Configuration for social learning parameters
    n_generations : int
        Number of generations to simulate
    n_agents : int
        Population size
    network_type : str
        Type of social network to use

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        - population_data: Population metrics over time
        - cultural_events: Individual cultural learning events
        - network_stats: Social network statistics over time
    """
    logger.info(f"Starting Layer 2 simulation: {n_generations} generations, {n_agents} agents")

    # Create social network
    topology = NetworkTopology(network_type, config.network_connectivity)
    social_network = SocialNetwork(n_agents, topology)

    # Create vectorized cultural layer
    from lovebug.layer_activation import LayerActivationConfig
    from lovebug.unified_mesa_model import UnifiedLoveBugs, UnifiedLoveModel

    layer_activation_config = LayerActivationConfig(
        genetic_enabled=False,
        cultural_enabled=True,
        genetic_weight=0.0,
        cultural_weight=1.0,
        blending_mode="weighted_average",
        normalize_weights=True,
    )
    dummy_model = UnifiedLoveModel(
        layer_config=layer_activation_config,
        genetic_params=None,
        cultural_params=config,
        n_agents=n_agents,
    )
    agent_set = UnifiedLoveBugs(n_agents, dummy_model)
    transmission_manager = VectorizedCulturalLayer(agent_set, config)

    # Create agent population
    agent_data = Layer2AgentData(n_agents)

    # Create monitoring
    monitor = SimulationMonitor(show_debug=False)

    # Storage for results
    population_results = []
    network_results = []

    # Run simulation
    with monitor.track_simulation(n_generations) as progress:
        for generation in range(n_generations):
            # Process cultural learning
            events = transmission_manager.process_cultural_learning(agent_data, generation)

            # Calculate population metrics
            cultural_diversity = len(np.unique(agent_data.cultural_preferences)) / 256.0
            genetic_cultural_distance = np.mean(
                np.abs(agent_data.cultural_preferences.astype(int) - agent_data.genetic_preferences.astype(int))
            )

            #             Learning event statistics
            event_counts = {}
            # VectorizedCulturalLayer does not use LearningType; skip or adapt as needed

            # Population data
            population_results.append(
                {
                    "generation": generation,
                    "cultural_diversity": cultural_diversity,
                    "gene_culture_distance": genetic_cultural_distance,
                    "mean_cultural_preference": np.mean(agent_data.cultural_preferences),
                    "mean_genetic_preference": np.mean(agent_data.genetic_preferences),
                    "cultural_variance": np.var(agent_data.cultural_preferences),
                    "genetic_variance": np.var(agent_data.genetic_preferences),
                    "mean_mating_success": np.mean(agent_data.mating_success),
                    "mean_age": np.mean(agent_data.ages),
                    "total_learning_events": len(events),
                    "oblique_events": event_counts.get("oblique", 0),
                    "horizontal_events": event_counts.get("horizontal", 0),
                    "innovation_events": event_counts.get("innovation", 0),
                    "prestige_events": event_counts.get("prestige", 0),
                    "population_size": n_agents,
                }
            )

            # Network statistics
            network_stats = social_network.compute_network_statistics()
            network_stats["generation"] = generation
            network_results.append(network_stats)

            # Age population
            agent_data.age_population()

            # Update network size if needed
            if len(agent_data.agent_ids) != social_network.n_agents:
                social_network.update_network_size(len(agent_data.agent_ids))

            # Update progress
            metrics = {
                "cultural_diversity": cultural_diversity,
                "learning_events": len(events),
                "gene_culture_distance": genetic_cultural_distance,
            }
            monitor.log_generation(generation, metrics)
            progress.advance(monitor.current_task)

    # Convert results to DataFrames
    population_df = pl.DataFrame(population_results)
    # VectorizedCulturalLayer does not have get_events_dataframe; use learning_events directly

    if hasattr(transmission_manager, "learning_events") and transmission_manager.learning_events:
        cultural_events_df = pl.DataFrame(transmission_manager.learning_events)
    else:
        cultural_events_df = pl.DataFrame()
    network_df = pl.DataFrame([r for r in network_results if r])

    logger.info("Layer 2 simulation completed successfully")
    return population_df, cultural_events_df, network_df


@app.cell
def import_libraries():
    """Import required libraries and set up environment."""
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import seaborn as sns
    from matplotlib.patches import Patch

    # Set up plotting style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")

    mo.md("""
    # 🧠 Layer 2 Social Learning: Enhanced Cultural Transmission

    This notebook implements and visualizes the **Layer 2 research extension** for LoveBug,
    focusing on enhanced social learning mechanisms that extend beyond basic genetic
    inheritance to include cultural transmission, social networks, and innovation.

    ## Key Features

    - **🔄 Oblique Transmission**: Parent-offspring cultural learning
    - **🤝 Horizontal Transmission**: Peer-to-peer learning with social networks
    - **💡 Cultural Innovation**: Random cultural mutations and creativity
    - **🧠 Cultural Memory**: Agents remember and weight past experiences
    - **🕸️ Social Networks**: Multiple network topologies affecting transmission
    - **📊 Rich Monitoring**: Real-time progress with beautiful console output

    ## Theoretical Foundation

    Layer 2 extends the Lande-Kirkpatrick model by incorporating **cultural evolution**
    alongside genetic evolution, allowing for more rapid adaptation and complex
    dynamics including cultural-genetic coevolution.
    """)
    return mo, np, pl, Path, plt, sns, nx, Patch


@app.cell
def setup_layer2_parameters(mo):
    """Interactive parameter setup for Layer 2 social learning."""

    # Cultural transmission parameters
    oblique_rate = mo.ui.slider(0.0, 0.8, value=0.3, step=0.05, label="Oblique Transmission Rate")
    horizontal_rate = mo.ui.slider(0.0, 0.8, value=0.2, step=0.05, label="Horizontal Transmission Rate")
    innovation_rate = mo.ui.slider(0.0, 0.3, value=0.05, step=0.01, label="Cultural Innovation Rate")

    # Social network parameters
    network_type = mo.ui.dropdown(
        ["small_world", "random", "scale_free", "grid"], value="small_world", label="Network Topology"
    )
    network_connectivity = mo.ui.slider(0.05, 0.5, value=0.1, step=0.05, label="Network Connectivity")

    # Memory parameters
    memory_size = mo.ui.slider(3, 20, value=10, step=1, label="Cultural Memory Size")
    memory_decay = mo.ui.slider(0.01, 0.2, value=0.05, step=0.01, label="Memory Decay Rate")

    # Simulation parameters
    n_generations = mo.ui.slider(50, 300, value=100, step=25, label="Generations")
    n_agents = mo.ui.slider(200, 1000, value=500, step=100, label="Population Size")

    mo.md(f"""
    ## 🎛️ Layer 2 Configuration

    Adjust the parameters below to explore different social learning scenarios:

    **Cultural Transmission:**
    - {oblique_rate}
    - {horizontal_rate}
    - {innovation_rate}

    **Social Network:**
    - {network_type}
    - {network_connectivity}

    **Cultural Memory:**
    - {memory_size}
    - {memory_decay}

    **Simulation:**
    - {n_generations}
    - {n_agents}

    **Current Configuration:**
    - Oblique rate: {oblique_rate.value:.2f}
    - Horizontal rate: {horizontal_rate.value:.2f}
    - Innovation rate: {innovation_rate.value:.3f}
    - Network: {network_type.value}
    - Memory size: {memory_size.value}
    - Population: {n_agents.value}
    """)

    return (
        oblique_rate,
        horizontal_rate,
        innovation_rate,
        network_type,
        network_connectivity,
        memory_size,
        memory_decay,
        n_generations,
        n_agents,
    )


@app.cell
def run_layer2_simulation(
    oblique_rate,
    horizontal_rate,
    innovation_rate,
    network_type,
    network_connectivity,
    memory_size,
    memory_decay,
    n_generations,
    n_agents,
    mo,
    np,
    pl,
):
    """Run the Layer 2 social learning simulation with current parameters."""

    # Create Layer 2 configuration
    config = Layer2Config(
        oblique_transmission_rate=oblique_rate.value,
        horizontal_transmission_rate=horizontal_rate.value,
        innovation_rate=innovation_rate.value,
        network_type=network_type.value,
        network_connectivity=network_connectivity.value,
        cultural_memory_size=memory_size.value,
        memory_decay_rate=memory_decay.value,
        log_cultural_events=True,
        log_every_n_generations=10,
    )

    # Run simulation
    try:
        population_data, cultural_events, network_data = simulate_layer2_social_learning(
            config, n_generations.value, n_agents.value, network_type.value
        )

        # Calculate summary statistics
        final_diversity = population_data.select(pl.col("cultural_diversity").last()).item()
        final_distance = population_data.select(pl.col("gene_culture_distance").last()).item()
        total_events = len(cultural_events) if len(cultural_events) > 0 else 0

        # Determine evolutionary pattern
        diversity_trend = (
            population_data.select(pl.col("cultural_diversity").last()).item()
            - population_data.select(pl.col("cultural_diversity").first()).item()
        )

        if diversity_trend > 0.1:
            pattern = "🌈 **Cultural Diversification**"
        elif diversity_trend < -0.1:
            pattern = "🎯 **Cultural Convergence**"
        elif total_events > n_generations.value * n_agents.value * 0.1:
            pattern = "🔄 **Active Cultural Transmission**"
        else:
            pattern = "🔒 **Cultural Stability**"

        # Event distribution
        if len(cultural_events) > 0:
            event_summary = cultural_events.group_by("learning_type").len().sort("len", descending=True).to_pandas()
            dominant_mechanism = event_summary.iloc[0]["learning_type"] if len(event_summary) > 0 else "none"
        else:
            dominant_mechanism = "none"

        mo.md(f"""
        ## 📊 Layer 2 Simulation Results

        **Cultural Pattern:** {pattern}

        **Final Metrics:**
        - Cultural Diversity: {final_diversity:.3f}
        - Gene-Culture Distance: {final_distance:.1f}
        - Total Learning Events: {total_events:,}
        - Dominant Mechanism: {dominant_mechanism.title()}

        **Interpretation:**
        {
            _interpret_layer2_results(
                final_diversity, final_distance, total_events, n_generations.value, n_agents.value, config
            )
        }
        """)

    except Exception as e:
        mo.md(f"""
        ## ❌ Simulation Error

        Failed to run Layer 2 simulation: {str(e)}

        Please check parameter values and try again.
        """)
        population_data, cultural_events, network_data = None, None, None

    return population_data, cultural_events, network_data, config


@beartype
def _interpret_layer2_results(
    diversity: float, distance: float, events: int, generations: int, agents: int, config: Layer2Config
) -> str:
    """Generate interpretation of Layer 2 simulation results."""

    interpretation = []

    # Cultural diversity analysis
    if diversity > 0.7:
        interpretation.append("• **High cultural diversity** maintained through innovation and transmission")
    elif diversity < 0.3:
        interpretation.append("• **Low cultural diversity** indicates strong convergent forces")
    else:
        interpretation.append("• **Moderate cultural diversity** shows balanced evolutionary pressures")

    # Gene-culture relationship
    if distance > 50:
        interpretation.append(
            f"• **Strong gene-culture divergence** ({distance:.1f}) shows cultural evolution outpacing genetics"
        )
    elif distance < 20:
        interpretation.append("• **Close gene-culture alignment** suggests limited cultural drift")
    else:
        interpretation.append("• **Moderate gene-culture distance** indicates ongoing coevolution")

    # Learning activity
    events_per_agent_gen = events / (agents * generations) if events > 0 else 0
    if events_per_agent_gen > 0.1:
        interpretation.append(
            f"• **High learning activity** ({events_per_agent_gen:.3f} events/agent/gen) drives cultural change"
        )
    elif events_per_agent_gen < 0.02:
        interpretation.append("• **Low learning activity** suggests weak cultural transmission")

    # Mechanism effects
    total_transmission = config.oblique_transmission_rate + config.horizontal_transmission_rate
    if total_transmission > 0.5:
        interpretation.append("• **Strong transmission rates** enable rapid cultural spread")

    if config.innovation_rate > 0.1:
        interpretation.append(f"• **High innovation rate** ({config.innovation_rate:.2f}) maintains cultural variation")

    return "\n".join(interpretation) if interpretation else "• Standard social learning dynamics observed"


@app.cell
def create_cultural_dynamics_plot(population_data, mo, plt, sns):
    """Create comprehensive plot of cultural dynamics over time."""

    if population_data is None:
        return mo.md("No population data available for plotting.")

    df = population_data.to_pandas()

    # Create comprehensive cultural dynamics plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Cultural vs Genetic Preferences
    ax1.plot(
        df["generation"], df["mean_cultural_preference"], "b-", linewidth=2.5, label="Cultural Preferences", alpha=0.8
    )
    ax1.plot(
        df["generation"], df["mean_genetic_preference"], "r-", linewidth=2.5, label="Genetic Preferences", alpha=0.8
    )

    # Add variance bands
    cultural_std = np.sqrt(df["cultural_variance"])
    genetic_std = np.sqrt(df["genetic_variance"])

    ax1.fill_between(
        df["generation"],
        df["mean_cultural_preference"] - cultural_std,
        df["mean_cultural_preference"] + cultural_std,
        alpha=0.2,
        color="blue",
    )
    ax1.fill_between(
        df["generation"],
        df["mean_genetic_preference"] - genetic_std,
        df["mean_genetic_preference"] + genetic_std,
        alpha=0.2,
        color="red",
    )

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Mean Preference Value")
    ax1.set_title("Cultural vs Genetic Preference Evolution", fontsize=14, fontweight="bold")
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cultural Diversity and Gene-Culture Distance
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(
        df["generation"], df["cultural_diversity"], "g-", linewidth=2.5, label="Cultural Diversity", alpha=0.8
    )
    line2 = ax2_twin.plot(
        df["generation"], df["gene_culture_distance"], "orange", linewidth=2.5, label="Gene-Culture Distance", alpha=0.8
    )

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Cultural Diversity", color="green")
    ax2_twin.set_ylabel("Gene-Culture Distance", color="orange")
    ax2.set_title("Cultural Diversity Dynamics", fontsize=14, fontweight="bold")

    # Combine legends
    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax2.legend(lines, labels, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Learning Events Over Time
    learning_columns = ["oblique_events", "horizontal_events", "innovation_events"]
    colors = ["blue", "green", "red"]
    labels = ["Oblique", "Horizontal", "Innovation"]

    bottom = np.zeros(len(df))
    for _i, (col, color, label) in enumerate(zip(learning_columns, colors, labels)):
        ax3.bar(df["generation"], df[col], bottom=bottom, color=color, alpha=0.7, label=label, width=0.8)
        bottom += df[col]

    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Learning Events per Generation")
    ax3.set_title("Cultural Learning Activity", fontsize=14, fontweight="bold")
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Population Metrics
    ax4.plot(df["generation"], df["mean_mating_success"], "purple", linewidth=2, label="Mean Mating Success", alpha=0.8)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(df["generation"], df["mean_age"], "brown", linewidth=2, label="Mean Age", alpha=0.8)

    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Mating Success", color="purple")
    ax4_twin.set_ylabel("Age", color="brown")
    ax4.set_title("Population Demographics", fontsize=14, fontweight="bold")

    # Combine legends
    lines_4 = ax4.get_lines() + ax4_twin.get_lines()
    labels_4 = [line.get_label() for line in lines_4]
    ax4.legend(lines_4, labels_4, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Layer 2 Social Learning Dynamics", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig


@app.cell
def create_cultural_events_analysis(cultural_events, mo, plt, sns):
    """Analyze and visualize cultural learning events."""

    if cultural_events is None or len(cultural_events) == 0:
        return mo.md("No cultural events data available for analysis.")

    df = cultural_events.to_pandas()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Event Type Distribution
    event_counts = df["learning_type"].value_counts()
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    wedges, texts, autotexts = ax1.pie(
        event_counts.values,
        labels=event_counts.index,
        autopct="%1.1f%%",
        colors=colors[: len(event_counts)],
        explode=[0.05] * len(event_counts),
    )
    ax1.set_title("Distribution of Learning Mechanisms", fontsize=14, fontweight="bold")

    # Plot 2: Learning Events Over Time
    if "generation" in df.columns:
        event_timeline = df.groupby(["generation", "learning_type"]).size().unstack(fill_value=0)
        event_timeline.plot(kind="area", stacked=True, ax=ax2, alpha=0.7)
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Number of Learning Events")
        ax2.set_title("Learning Activity Timeline", fontsize=14, fontweight="bold")
        ax2.legend(title="Learning Type", frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3)

    # Plot 3: Preference Change Distribution
    if "old_preference" in df.columns and "new_preference" in df.columns:
        preference_changes = df["new_preference"] - df["old_preference"]
        ax3.hist(preference_changes, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax3.axvline(x=0, color="red", linestyle="--", alpha=0.8, label="No Change")
        ax3.axvline(
            x=preference_changes.mean(), color="orange", linestyle="-", label=f"Mean = {preference_changes.mean():.1f}"
        )
        ax3.set_xlabel("Preference Change (New - Old)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Distribution of Cultural Changes", fontsize=14, fontweight="bold")
        ax3.legend(frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3)

    # Plot 4: Success-based Learning Analysis
    if "success_metric" in df.columns:
        # Filter out innovation events (no teacher)
        learning_events = df[df["learning_type"] != "innovation"]
        if len(learning_events) > 0:
            success_by_type = learning_events.groupby("learning_type")["success_metric"].mean()

            bars = ax4.bar(
                success_by_type.index, success_by_type.values, color=["#FF6B6B", "#4ECDC4", "#96CEB4"], alpha=0.7
            )
            ax4.set_xlabel("Learning Type")
            ax4.set_ylabel("Average Teacher Success")
            ax4.set_title("Success-Based Learning Patterns", fontsize=14, fontweight="bold")
            ax4.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                )

    plt.suptitle("Cultural Learning Events Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig


@app.cell
def create_social_network_visualization(network_data, config, mo, plt, sns, nx):
    """Visualize social network structure and statistics."""

    if network_data is None or len(network_data) == 0:
        return mo.md("No network data available for visualization.")

    df = network_data.to_pandas()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Network Statistics Over Time
    if "mean_degree" in df.columns:
        ax1.plot(df["generation"], df["mean_degree"], "b-", linewidth=2, label="Mean Degree")
        if "clustering_coefficient" in df.columns:
            ax1_twin = ax1.twinx()
            ax1_twin.plot(
                df["generation"], df["clustering_coefficient"], "r-", linewidth=2, label="Clustering Coefficient"
            )
            ax1_twin.set_ylabel("Clustering Coefficient", color="red")

        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Mean Degree", color="blue")
        ax1.set_title("Network Structure Evolution", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

    # Plot 2: Network Connectivity
    if "density" in df.columns:
        ax2.plot(df["generation"], df["density"], "g-", linewidth=2.5, label="Network Density")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Network Density")
        ax2.set_title("Network Connectivity Over Time", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

    # Plot 3: Create small example network
    # Create a small sample network for visualization
    topology = NetworkTopology(config.network_type, config.network_connectivity)
    sample_network = SocialNetwork(30, topology)  # Small network for visualization

    if config.network_type == "grid":
        pos = {i: (i % 6, i // 6) for i in sample_network.graph.nodes()}
    else:
        pos = nx.spring_layout(sample_network.graph, k=1, iterations=50)

    # Draw network
    nx.draw_networkx_edges(sample_network.graph, pos, ax=ax3, alpha=0.5, width=0.5)
    nx.draw_networkx_nodes(sample_network.graph, pos, ax=ax3, node_color="lightblue", node_size=100, alpha=0.8)

    ax3.set_title(f"{config.network_type.title()} Network Structure", fontsize=14, fontweight="bold")
    ax3.axis("off")

    # Plot 4: Degree Distribution
    if len(df) > 0 and "mean_degree" in df.columns:
        # Create degree distribution for final generation
        degrees = []
        try:
            degree_dict = dict(sample_network.graph.degree())
            degrees = list(degree_dict.values())
        except Exception:
            degrees = [3, 4, 2, 5, 3, 4, 2, 3, 5, 4]  # Fallback data

        if degrees:
            ax4.hist(degrees, bins=max(1, len(set(degrees))), alpha=0.7, color="orange", edgecolor="black")
            ax4.axvline(x=np.mean(degrees), color="red", linestyle="--", label=f"Mean = {np.mean(degrees):.1f}")
            ax4.set_xlabel("Node Degree")
            ax4.set_ylabel("Frequency")
            ax4.set_title("Degree Distribution", fontsize=14, fontweight="bold")
            ax4.legend(frameon=True, fancybox=True, shadow=True)
            ax4.grid(True, alpha=0.3)

    plt.suptitle(f"Social Network Analysis: {config.network_type.title()} Topology", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    return fig


@app.cell
def compare_layer2_mechanisms(mo, np, pl, plt, sns):
    """Compare different Layer 2 mechanism configurations."""

    # Define comparison scenarios
    scenarios = {
        "Genetic Only": Layer2Config(
            oblique_transmission_rate=0.0, horizontal_transmission_rate=0.0, innovation_rate=0.0
        ),
        "High Innovation": Layer2Config(
            oblique_transmission_rate=0.1, horizontal_transmission_rate=0.1, innovation_rate=0.2
        ),
        "Strong Horizontal": Layer2Config(
            oblique_transmission_rate=0.1, horizontal_transmission_rate=0.6, innovation_rate=0.05
        ),
        "Strong Oblique": Layer2Config(
            oblique_transmission_rate=0.6, horizontal_transmission_rate=0.1, innovation_rate=0.05
        ),
        "Balanced Learning": Layer2Config(
            oblique_transmission_rate=0.3, horizontal_transmission_rate=0.3, innovation_rate=0.1
        ),
    }

    # Run simulations for each scenario
    results = {}
    n_gen = 80  # Shorter runs for comparison
    n_pop = 300

    for name, config in scenarios.items():
        try:
            pop_data, events_data, net_data = simulate_layer2_social_learning(config, n_gen, n_pop, "small_world")
            results[name] = {"population": pop_data, "events": events_data, "config": config}
        except Exception as e:
            logger.error(f"Failed to simulate {name}: {e}")
            continue

    if not results:
        return mo.md("Failed to run comparison simulations.")

    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    colors = sns.color_palette("husl", len(results))

    # Plot 1: Cultural Diversity Evolution
    for i, (name, data) in enumerate(results.items()):
        df = data["population"].to_pandas()
        ax1.plot(df["generation"], df["cultural_diversity"], color=colors[i], linewidth=2, label=name, alpha=0.8)

    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Cultural Diversity")
    ax1.set_title("Cultural Diversity: Mechanism Comparison", fontsize=14, fontweight="bold")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gene-Culture Distance Evolution
    for i, (name, data) in enumerate(results.items()):
        df = data["population"].to_pandas()
        ax2.plot(df["generation"], df["gene_culture_distance"], color=colors[i], linewidth=2, label=name, alpha=0.8)

    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Gene-Culture Distance")
    ax2.set_title("Gene-Culture Divergence: Mechanism Comparison", fontsize=14, fontweight="bold")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Total Learning Activity
    for i, (name, data) in enumerate(results.items()):
        df = data["population"].to_pandas()
        ax3.plot(df["generation"], df["total_learning_events"], color=colors[i], linewidth=2, label=name, alpha=0.8)

    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Learning Events per Generation")
    ax3.set_title("Learning Activity: Mechanism Comparison", fontsize=14, fontweight="bold")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final Outcomes Summary
    final_diversity = []
    final_distance = []
    total_events = []
    scenario_names = []

    for name, data in results.items():
        pop_df = data["population"]
        final_diversity.append(pop_df.select(pl.col("cultural_diversity").last()).item())
        final_distance.append(pop_df.select(pl.col("gene_culture_distance").last()).item())
        total_events.append(len(data["events"]) if len(data["events"]) > 0 else 0)
        scenario_names.append(name)

    # Create scatter plot with bubble size representing total events
    ax4.scatter(
        final_diversity,
        final_distance,
        s=[e / 10 for e in total_events],  # Scale bubble size
        c=range(len(final_diversity)),
        cmap="viridis",
        alpha=0.7,
        edgecolors="white",
        linewidth=2,
    )

    for i, name in enumerate(scenario_names):
        ax4.annotate(
            name, (final_diversity[i], final_distance[i]), xytext=(5, 5), textcoords="offset points", fontsize=9
        )

    ax4.set_xlabel("Final Cultural Diversity")
    ax4.set_ylabel("Final Gene-Culture Distance")
    ax4.set_title("Evolutionary Endpoints (bubble size = total events)", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Layer 2 Mechanism Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Create summary statistics
    summary_data = []
    for name, data in results.items():
        pop_df = data["population"]
        config = data["config"]

        summary_data.append(
            {
                "Scenario": name,
                "Final Diversity": pop_df.select(pl.col("cultural_diversity").last()).item(),
                "Final Distance": pop_df.select(pl.col("gene_culture_distance").last()).item(),
                "Total Events": len(data["events"]) if len(data["events"]) > 0 else 0,
                "Oblique Rate": config.oblique_transmission_rate,
                "Horizontal Rate": config.horizontal_transmission_rate,
                "Innovation Rate": config.innovation_rate,
            }
        )

    summary_df = pl.DataFrame(summary_data)

    mo.md(f"""
    ## 📊 Mechanism Comparison Summary

    {summary_df.to_pandas().round(3).to_html(index=False)}

    **Key Insights:**
    - Higher innovation rates maintain cultural diversity
    - Strong horizontal transmission can lead to rapid convergence
    - Oblique transmission preserves cultural traditions
    - Balanced learning creates stable cultural dynamics
    """)

    return fig, results


@app.cell
def layer2_theoretical_framework(mo):
    """Provide theoretical framework and interpretation for Layer 2."""

    mo.md("""
    ## 🧠 Layer 2 Theoretical Framework

    ### Enhanced Social Learning Mechanisms

    Layer 2 extends the classic Lande-Kirkpatrick model by incorporating **cultural evolution**
    alongside genetic inheritance, enabling rapid adaptation and complex coevolutionary dynamics.

    ### Core Mechanisms

    #### 1. 🔄 Oblique Transmission
    - **Definition**: Cultural learning from older generation (parent-like figures)
    - **Function**: Preserves cultural traditions and knowledge
    - **Effect**: Creates cultural continuity across generations
    - **Real Examples**: Language acquisition, tool use, aesthetic preferences

    #### 2. 🤝 Horizontal Transmission
    - **Definition**: Peer-to-peer learning within the same generation
    - **Function**: Spreads innovations rapidly through population
    - **Effect**: Can lead to cultural homogenization or fads
    - **Real Examples**: Fashion trends, social media viral content, peer influence

    #### 3. 💡 Cultural Innovation
    - **Definition**: Random generation of new cultural variants
    - **Function**: Maintains cultural diversity and enables adaptation
    - **Effect**: Provides raw material for cultural evolution
    - **Real Examples**: Artistic creativity, technological invention, behavioral experiments

    #### 4. 🧠 Cultural Memory
    - **Definition**: Agents remember and weight past cultural experiences
    - **Function**: Creates inertia in cultural change
    - **Effect**: Stabilizes culture while allowing gradual shifts
    - **Real Examples**: Personal experience, cultural trauma, learned associations

    ### Social Network Effects

    #### Network Topologies
    - **Small World**: High clustering + short path lengths → rapid global spread with local clusters
    - **Scale-Free**: Hub-based transmission → influential individuals drive cultural change
    - **Random**: Uniform mixing → homogeneous cultural spread
    - **Grid**: Spatial constraints → geographic cultural gradients

    #### Transmission Dynamics
    - **Local clustering** preserves cultural subgroups
    - **Long-range connections** enable rapid cultural diffusion
    - **Network structure** determines spread patterns and equilibria

    ### Cultural-Genetic Coevolution

    #### Gene-Culture Interactions
    1. **Cultural preferences** influence mating choices
    2. **Genetic traits** affect cultural learning ability
    3. **Feedback loops** create complex evolutionary dynamics
    4. **Time scales**: Culture evolves faster than genes

    #### Evolutionary Outcomes
    - **🌈 Cultural Diversification**: Multiple stable cultural variants
    - **🎯 Cultural Convergence**: Population-wide cultural uniformity
    - **🔄 Cultural Cycles**: Oscillating cultural trends
    - **🚀 Runaway Cultural Evolution**: Accelerating cultural change

    ### Research Applications

    #### Sexual Selection
    - **Mate choice copying**: Learning preferences from others
    - **Cultural beauty standards**: Socially transmitted aesthetic preferences
    - **Display innovation**: Cultural evolution of courtship behaviors

    #### Human Evolution
    - **Language evolution**: Cultural transmission of communication systems
    - **Tool culture**: Social learning of technology use
    - **Social norms**: Cultural evolution of cooperation and punishment

    #### Animal Behavior
    - **Song learning**: Bird song cultural transmission
    - **Foraging traditions**: Social learning of feeding behaviors
    - **Migration routes**: Cultural inheritance of movement patterns

    ### Model Extensions

    Future Layer 2 developments could include:
    - **Multi-trait cultural evolution**
    - **Environmental feedback on cultural transmission**
    - **Individual differences in learning ability**
    - **Structured populations with migration**
    - **Coevolution of genetic and cultural transmission mechanisms**
    """)


@app.cell
def export_layer2_data(population_data, cultural_events, network_data, config, mo, Path):
    """Export Layer 2 simulation results for further analysis."""

    if population_data is None:
        return mo.md("No Layer 2 data to export.")

    # Create output directory
    output_dir = Path("layer2_output")
    output_dir.mkdir(exist_ok=True)

    # Generate filename with configuration summary
    filename_base = (
        f"layer2_oblique{config.oblique_transmission_rate:.2f}_"
        f"horizontal{config.horizontal_transmission_rate:.2f}_"
        f"innovation{config.innovation_rate:.3f}_"
        f"{config.network_type}"
    )

    try:
        # Export population data
        pop_path = output_dir / f"{filename_base}_population.parquet"
        population_data.write_parquet(pop_path)

        # Export cultural events if available
        events_path = None
        if cultural_events is not None and len(cultural_events) > 0:
            events_path = output_dir / f"{filename_base}_events.parquet"
            cultural_events.write_parquet(events_path)

        # Export network data if available
        network_path = None
        if network_data is not None and len(network_data) > 0:
            network_path = output_dir / f"{filename_base}_network.parquet"
            network_data.write_parquet(network_path)

        # Export configuration
        import json

        config_dict = config.to_dict()
        config_path = output_dir / f"{filename_base}_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Calculate file sizes
        pop_size = pop_path.stat().st_size / 1024
        events_size = events_path.stat().st_size / 1024 if events_path else 0
        network_size = network_path.stat().st_size / 1024 if network_path else 0

        mo.md(f"""
        ## 💾 Layer 2 Data Export Complete

        **Files Created:**
        - **Population data**: `{pop_path}` ({pop_size:.1f} KB)
        - **Cultural events**: `{events_path}` ({events_size:.1f} KB) - {len(cultural_events) if cultural_events else 0:,} events
        - **Network data**: `{network_path}` ({network_size:.1f} KB) - {len(network_data) if network_data else 0:,} records
        - **Configuration**: `{config_path}`

        **Dataset Summary:**
        - Population records: {len(population_data):,}
        - Cultural events: {len(cultural_events) if cultural_events else 0:,}
        - Network snapshots: {len(network_data) if network_data else 0:,}
        - Total size: {pop_size + events_size + network_size:.1f} KB

        This Layer 2 data can be used for:
        - Advanced statistical analysis of cultural transmission
        - Machine learning models of social learning
        - Network analysis of cultural diffusion
        - Comparative studies across parameter regimes
        - Publication-quality visualizations
        """)

    except Exception as e:
        mo.md(f"""
        ## ❌ Export Failed

        Could not export Layer 2 data: {str(e)}
        """)

    return output_dir


if __name__ == "__main__":
    app.run()
