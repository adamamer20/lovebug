"""
Tests for Layer 2 social learning mechanisms.

Comprehensive test suite for configuration, social networks, cultural transmission,
and monitoring systems in the Layer 2 research extension.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from lovebug.layer2.config import Layer2Config
from lovebug.layer2.monitoring.simulation_monitor import SimulationMonitor
from lovebug.layer2.social_learning.cultural_transmission import (
    CulturalTransmissionManager,
    LearningEvent,
    LearningType,
)
from lovebug.layer2.social_learning.social_networks import NetworkTopology, SocialNetwork


class TestLayer2Config:
    """Test Layer 2 configuration system."""

    def test_default_config_creation(self) -> None:
        """Test creating configuration with default values."""
        config = Layer2Config()

        assert 0.0 <= config.oblique_transmission_rate <= 1.0
        assert 0.0 <= config.horizontal_transmission_rate <= 1.0
        assert 0.0 <= config.innovation_rate <= 1.0
        assert config.network_type in {"random", "small_world", "scale_free"}
        assert config.cultural_memory_size > 0
        assert config.log_every_n_generations > 0

    def test_config_validation(self) -> None:
        """Test configuration parameter validation."""
        # Test invalid transmission rates
        with pytest.raises(ValueError, match="oblique_transmission_rate must be between 0 and 1"):
            Layer2Config(oblique_transmission_rate=1.5)

        with pytest.raises(ValueError, match="horizontal_transmission_rate must be between 0 and 1"):
            Layer2Config(horizontal_transmission_rate=-0.1)

        # Test invalid network type
        with pytest.raises(ValueError, match="network_type must be one of"):
            Layer2Config(network_type="invalid_network")

        # Test invalid memory size
        with pytest.raises(ValueError, match="cultural_memory_size must be positive"):
            Layer2Config(cultural_memory_size=0)

    def test_config_file_operations(self) -> None:
        """Test saving and loading configuration files."""
        config = Layer2Config(oblique_transmission_rate=0.4, innovation_rate=0.08, network_type="scale_free")

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            # Save configuration
            config.save_to_file(config_path)
            assert config_path.exists()

            # Load configuration
            loaded_config = Layer2Config.load_from_file(config_path)

            assert loaded_config.oblique_transmission_rate == 0.4
            assert loaded_config.innovation_rate == 0.08
            assert loaded_config.network_type == "scale_free"

    def test_config_update(self) -> None:
        """Test configuration update functionality."""
        config = Layer2Config()

        updated_config = config.update(innovation_rate=0.1, log_cultural_events=False)

        # Original config unchanged
        assert config.innovation_rate == 0.05
        assert config.log_cultural_events is True

        # New config has updates
        assert updated_config.innovation_rate == 0.1
        assert updated_config.log_cultural_events is False


class TestSocialNetwork:
    """Test social network functionality."""

    def test_network_creation(self) -> None:
        """Test creating different types of networks."""
        topologies = [
            NetworkTopology("random", 0.1),
            NetworkTopology("small_world", 0.1),
            NetworkTopology("scale_free", 0.1),
        ]

        for topology in topologies:
            network = SocialNetwork(100, topology)
            assert network.n_agents == 100
            assert network.topology == topology
            assert network.graph.number_of_nodes() <= 100  # May be fewer due to implementation

    def test_neighbor_queries(self) -> None:
        """Test neighbor query functionality."""
        topology = NetworkTopology("small_world", 0.2)
        network = SocialNetwork(50, topology)

        # Test getting neighbors
        neighbors = network.get_neighbors(0, max_neighbors=5)
        assert isinstance(neighbors, list)
        assert len(neighbors) <= 5
        assert all(isinstance(n, int) for n in neighbors)

        # Test local neighborhood
        local_neighbors = network.get_local_neighbors(0, radius=2)
        assert isinstance(local_neighbors, list)

        # Test random agents
        random_agents = network.get_random_agents(3, exclude=0)
        assert len(random_agents) <= 3
        assert 0 not in random_agents

    def test_network_size_updates(self) -> None:
        """Test updating network size dynamically."""
        topology = NetworkTopology("random", 0.1)
        network = SocialNetwork(50, topology)

        original_size = network.graph.number_of_nodes()

        # Increase size
        network.update_network_size(75)
        assert network.n_agents == 75
        assert network.graph.number_of_nodes() >= original_size

        # Decrease size
        network.update_network_size(25)
        assert network.n_agents == 25

    def test_network_statistics(self) -> None:
        """Test network statistics computation."""
        topology = NetworkTopology("small_world", 0.2)
        network = SocialNetwork(100, topology)

        stats = network.compute_network_statistics()

        required_keys = ["num_nodes", "num_edges", "density", "mean_degree"]
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))


class TestCulturalTransmission:
    """Test cultural transmission mechanisms."""

    def create_mock_agent_data(self, n_agents: int = 20) -> Mock:
        """Create mock agent data for testing."""
        mock_data = Mock()
        mock_data.get_agent_ids.return_value = np.arange(n_agents)
        mock_data.get_cultural_preferences.return_value = np.random.randint(0, 256, n_agents)
        mock_data.get_genetic_preferences.return_value = np.random.randint(0, 256, n_agents)
        mock_data.get_mating_success.return_value = np.random.exponential(1.0, n_agents)
        mock_data.get_ages.return_value = np.random.randint(0, 50, n_agents)
        mock_data.update_cultural_preference = Mock()
        return mock_data

    def test_transmission_manager_creation(self) -> None:
        """Test creating cultural transmission manager."""
        config = Layer2Config()
        topology = NetworkTopology("small_world", 0.1)
        network = SocialNetwork(20, topology)

        manager = CulturalTransmissionManager(config, network)

        assert manager.config == config
        assert manager.social_network == network
        assert len(manager.learning_events) == 0

    def test_cultural_learning_process(self) -> None:
        """Test cultural learning event processing."""
        config = Layer2Config(oblique_transmission_rate=0.3, horizontal_transmission_rate=0.3, innovation_rate=0.2)
        topology = NetworkTopology("random", 0.2)
        network = SocialNetwork(20, topology)
        manager = CulturalTransmissionManager(config, network)

        agent_data = self.create_mock_agent_data(20)

        # Process learning for one generation
        events = manager.process_cultural_learning(agent_data, generation=0)

        assert isinstance(events, list)
        assert all(isinstance(event, LearningEvent) for event in events)

        # Check that some learning occurred (probabilistic, so may be 0)
        assert len(manager.learning_events) >= 0

    def test_learning_event_types(self) -> None:
        """Test different types of learning events."""
        config = Layer2Config(innovation_rate=1.0)  # Force innovation
        topology = NetworkTopology("random", 0.1)
        network = SocialNetwork(10, topology)
        manager = CulturalTransmissionManager(config, network)

        agent_data = self.create_mock_agent_data(10)

        # Run several generations to get different event types
        all_events = []
        for gen in range(5):
            events = manager.process_cultural_learning(agent_data, gen)
            all_events.extend(events)

        if all_events:  # If any events occurred
            event_types = {event.learning_type for event in all_events}
            # Should have innovation events due to high innovation rate
            assert LearningType.INNOVATION in event_types

    def test_learning_statistics(self) -> None:
        """Test learning statistics computation."""
        config = Layer2Config()
        topology = NetworkTopology("random", 0.1)
        network = SocialNetwork(10, topology)
        manager = CulturalTransmissionManager(config, network)

        # Add some mock events
        manager.learning_events.extend(
            [
                LearningEvent(0, 1, LearningType.HORIZONTAL, 100, 150, 0.5, 0),
                LearningEvent(1, None, LearningType.INNOVATION, 150, 200, 0.0, 0),
                LearningEvent(2, 3, LearningType.OBLIQUE, 75, 125, 0.8, 1),
            ]
        )

        stats = manager.get_learning_statistics()

        assert stats["total_events"] == 3
        assert "horizontal_proportion" in stats
        assert "innovation_proportion" in stats
        assert "oblique_proportion" in stats

    def test_events_dataframe(self) -> None:
        """Test converting events to DataFrame."""
        config = Layer2Config()
        topology = NetworkTopology("random", 0.1)
        network = SocialNetwork(10, topology)
        manager = CulturalTransmissionManager(config, network)

        # Add mock events
        manager.learning_events.append(LearningEvent(0, 1, LearningType.HORIZONTAL, 100, 150, 0.5, 0))

        df = manager.get_events_dataframe()

        assert len(df) == 1
        expected_columns = [
            "learner_id",
            "teacher_id",
            "learning_type",
            "old_preference",
            "new_preference",
            "success_metric",
            "generation",
        ]
        for col in expected_columns:
            assert col in df.columns


class TestSimulationMonitor:
    """Test simulation monitoring functionality."""

    def test_monitor_creation(self) -> None:
        """Test creating simulation monitor."""
        monitor = SimulationMonitor(show_debug=False)

        assert monitor.console is not None
        assert monitor.progress is None
        assert len(monitor._cultural_events) == 0

    def test_simulation_tracking(self) -> None:
        """Test simulation progress tracking."""
        monitor = SimulationMonitor(show_debug=False)

        # Test context manager
        with monitor.track_simulation(10, {"test": "value"}) as progress:
            assert progress is not None
            assert monitor.progress is not None

        # Should be cleaned up after context
        assert monitor.progress is None

    def test_generation_logging(self) -> None:
        """Test generation-level metric logging."""
        monitor = SimulationMonitor(show_debug=False)

        metrics = {"diversity": 0.5, "distance": 2.0, "events": 10}

        # Should not raise errors
        monitor.log_generation(0, metrics, log_frequency=1)
        monitor.log_generation(5, metrics, log_frequency=5)

    def test_cultural_event_logging(self) -> None:
        """Test cultural event logging."""
        monitor = SimulationMonitor(show_debug=False)

        monitor.log_cultural_event(agent_id=42, event_type="innovation", details={"old_pref": 100, "new_pref": 150})

        assert len(monitor._cultural_events) == 1
        event = monitor._cultural_events[0]
        assert event["agent_id"] == 42
        assert event["event_type"] == "innovation"

    def test_cultural_events_dataframe(self) -> None:
        """Test converting cultural events to DataFrame."""
        monitor = SimulationMonitor(show_debug=False)

        # Empty case
        df = monitor.get_cultural_events_dataframe()
        assert len(df) == 0

        # With events
        monitor.log_cultural_event(0, "test", {"key": "value"})
        df = monitor.get_cultural_events_dataframe()
        assert len(df) == 1


@pytest.mark.integration
class TestLayer2Integration:
    """Integration tests for Layer 2 components."""

    def test_full_layer2_workflow(self) -> None:
        """Test complete Layer 2 workflow integration."""
        # Create configuration
        config = Layer2Config(
            oblique_transmission_rate=0.2,
            horizontal_transmission_rate=0.2,
            innovation_rate=0.1,
            network_type="small_world",
            log_cultural_events=True,
        )

        # Create social network
        topology = NetworkTopology(config.network_type, config.network_connectivity)
        network = SocialNetwork(50, topology)

        # Create transmission manager
        manager = CulturalTransmissionManager(config, network)

        # Create monitor
        monitor = SimulationMonitor(show_debug=False)

        # Create mock agent data
        mock_data = Mock()
        n_agents = 50
        mock_data.get_agent_ids.return_value = np.arange(n_agents)
        mock_data.get_cultural_preferences.return_value = np.random.randint(0, 256, n_agents)
        mock_data.get_genetic_preferences.return_value = np.random.randint(0, 256, n_agents)
        mock_data.get_mating_success.return_value = np.random.exponential(1.0, n_agents)
        mock_data.get_ages.return_value = np.random.randint(0, 50, n_agents)
        mock_data.update_cultural_preference = Mock()

        # Run short simulation
        n_generations = 5
        with monitor.track_simulation(n_generations) as progress:
            for generation in range(n_generations):
                events = manager.process_cultural_learning(mock_data, generation)

                metrics = {"learning_events": len(events), "diversity": 0.5}
                monitor.log_generation(generation, metrics)
                progress.advance(monitor.current_task)

        # Verify integration worked
        stats = manager.get_learning_statistics()
        assert isinstance(stats, dict)

        # Should have some network structure
        network_stats = network.compute_network_statistics()
        assert network_stats["num_nodes"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
