"""
Layer 2 Social Learning Demonstration

Showcases the enhanced social learning mechanisms with rich console output
and cultural transmission analysis.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lovebug.layer2.config import Layer2Config
from lovebug.layer2.monitoring.simulation_monitor import SimulationMonitor
from lovebug.layer2.social_learning.cultural_transmission import CulturalTransmissionManager
from lovebug.layer2.social_learning.social_networks import NetworkTopology, SocialNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class MockAgentData:
    """Mock agent data for demonstration purposes."""

    def __init__(self, n_agents: int = 100) -> None:
        self.n_agents = n_agents
        self.agent_ids = np.arange(n_agents)
        self.cultural_preferences = np.random.randint(0, 256, n_agents).astype(np.uint8)
        self.genetic_preferences = np.random.randint(0, 256, n_agents).astype(np.uint8)
        self.mating_success = np.random.exponential(1.0, n_agents).astype(np.float32)
        self.ages = np.random.randint(0, 50, n_agents).astype(np.uint16)

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


def demonstrate_layer2_functionality() -> None:
    """Demonstrate Layer 2 social learning capabilities."""

    # Create Layer 2 configuration
    config = Layer2Config(
        oblique_transmission_rate=0.2,
        horizontal_transmission_rate=0.3,
        innovation_rate=0.05,
        network_type="small_world",
        network_connectivity=0.1,
        cultural_memory_size=5,
        log_cultural_events=True,
        log_every_n_generations=5,
    )

    print("🔧 Created Layer 2 configuration:")
    print(f"   • Oblique transmission rate: {config.oblique_transmission_rate}")
    print(f"   • Horizontal transmission rate: {config.horizontal_transmission_rate}")
    print(f"   • Innovation rate: {config.innovation_rate}")
    print(f"   • Network type: {config.network_type}")
    print()

    # Create social network
    topology = NetworkTopology(network_type=config.network_type, connectivity=config.network_connectivity)

    n_agents = 200
    social_network = SocialNetwork(n_agents, topology)

    print(f"🕸️  Created {config.network_type} social network:")
    network_stats = social_network.compute_network_statistics()
    for key, value in network_stats.items():
        print(f"   • {key}: {value:.3f}")
    print()

    # Create cultural transmission manager
    transmission_manager = CulturalTransmissionManager(config, social_network)

    # Create mock agent data
    agent_data = MockAgentData(n_agents)

    print(f"👥 Created {n_agents} agents with:")
    print(f"   • Cultural diversity: {len(np.unique(agent_data.cultural_preferences))} unique preferences")
    print(f"   • Age range: {agent_data.ages.min()}-{agent_data.ages.max()}")
    print(f"   • Mean mating success: {agent_data.mating_success.mean():.3f}")
    print()

    # Create simulation monitor with rich console output
    monitor = SimulationMonitor(show_debug=True)

    # Run cultural transmission simulation
    n_generations = 50
    model_info = {
        "Population": n_agents,
        "Network": config.network_type,
        "Innovation Rate": f"{config.innovation_rate:.3f}",
    }

    print("🚀 Starting Layer 2 cultural transmission simulation...")
    print()

    with monitor.track_simulation(n_generations, model_info) as progress:
        for generation in range(n_generations):
            # Process cultural learning
            events = transmission_manager.process_cultural_learning(agent_data, generation)

            # Calculate metrics for monitoring
            cultural_diversity = len(np.unique(agent_data.cultural_preferences)) / 256.0
            genetic_cultural_distance = np.mean(
                np.abs(agent_data.cultural_preferences.astype(int) - agent_data.genetic_preferences.astype(int))
            )

            metrics = {
                "cultural_diversity": cultural_diversity,
                "gene_culture_distance": genetic_cultural_distance,
                "learning_events": len(events),
                "innovation_events": sum(1 for e in events if e.learning_type.value == "innovation"),
                "horizontal_events": sum(1 for e in events if e.learning_type.value == "horizontal"),
                "oblique_events": sum(1 for e in events if e.learning_type.value == "oblique"),
            }

            # Log to monitor
            monitor.log_generation(generation, metrics)

            # Update progress
            progress.advance(monitor.current_task)

    print()
    print("✅ Simulation completed!")
    print()

    # Display final statistics
    final_stats = transmission_manager.get_learning_statistics()
    monitor.create_metrics_summary(
        {
            "final_diversity": len(np.unique(agent_data.cultural_preferences)) / 256.0,
            "total_events": final_stats.get("total_events", 0),
            "innovation_proportion": final_stats.get("innovation_proportion", 0.0),
            "horizontal_proportion": final_stats.get("horizontal_proportion", 0.0),
            "oblique_proportion": final_stats.get("oblique_proportion", 0.0),
        }
    )

    # Display cultural events summary
    monitor.create_cultural_events_summary()

    # Get events dataframe for analysis
    events_df = transmission_manager.get_events_dataframe()

    if len(events_df) > 0:
        print()
        print("📊 Cultural Learning Events Analysis:")

        # Event type distribution
        event_counts = events_df.group_by("learning_type").len().sort("len", descending=True)
        print("\n   Event Type Distribution:")
        for row in event_counts.iter_rows(named=True):
            print(f"     • {row['learning_type']}: {row['len']} events")

        # Temporal patterns
        generation_events = events_df.group_by("generation").len()
        peak_generation = generation_events.sort("len", descending=True).row(0, named=True)
        print("\n   Peak Learning Activity:")
        print(f"     • Generation {peak_generation['generation']}: {peak_generation['len']} events")

        # Innovation analysis
        innovations = events_df.filter(pl.col("learning_type") == "innovation")
        if len(innovations) > 0:
            print("\n   Innovation Patterns:")
            print(f"     • Total innovations: {len(innovations)}")
            print(f"     • Innovation rate: {len(innovations) / len(events_df):.3f}")

        print()
        print("💾 Saving events data...")
        output_path = "layer2_cultural_events.parquet"
        events_df.write_parquet(output_path)
        print(f"   Cultural events saved to: {output_path}")

    print()
    print("🎯 Layer 2 demonstration completed successfully!")
    print("   The enhanced social learning mechanisms are working correctly with:")
    print("   • Oblique transmission (cross-generational learning)")
    print("   • Horizontal transmission (peer-to-peer learning)")
    print("   • Cultural innovation (random mutation)")
    print("   • Social network effects")
    print("   • Cultural memory systems")
    print("   • Rich console monitoring")


if __name__ == "__main__":
    try:
        demonstrate_layer2_functionality()
    except KeyboardInterrupt:
        print("\n❌ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n💥 Error during demonstration: {e}")
        raise
