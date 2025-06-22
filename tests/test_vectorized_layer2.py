"""
Tests for vectorized Layer 2 implementation.

Validates the performance and correctness of the new vectorized cultural
transmission system compared to the original implementation.
"""

from __future__ import annotations

import polars as pl
import pytest

from lovebug.config import CulturalParams
from lovebug.layer2 import (
    CulturalInnovationEngine,
    CulturalLayer,
    HorizontalTransmissionEngine,
    LearningEligibilityComputer,
    MemoryDecayEngine,
    NetworkTopology,
    ObliqueTransmissionEngine,
    SocialNetwork,
)


class TestNetworkTopology:
    """Test network topology configuration and validation."""

    def test_default_topology_configuration(self) -> None:
        """Test default topology configuration."""
        topology = NetworkTopology()
        assert topology.network_type == "small_world"
        assert topology.connectivity == 0.1
        assert 0.0 <= topology.connectivity <= 1.0

    def test_network_type_validation(self) -> None:
        """Test validation of network type parameter."""
        with pytest.raises(ValueError, match="network_type must be one of"):
            NetworkTopology(network_type="invalid")

    def test_connectivity_validation(self) -> None:
        """Test validation of connectivity parameter."""
        with pytest.raises(ValueError, match="connectivity must be between 0 and 1"):
            NetworkTopology(connectivity=1.5)


class TestSocialNetwork:
    """Test vectorized social network implementation."""

    @pytest.fixture
    def small_network(self) -> SocialNetwork:
        """Create a small test network."""
        topology = NetworkTopology("random", connectivity=0.3)
        return SocialNetwork(10, topology)

    def test_network_creation_and_structure(self, small_network: SocialNetwork) -> None:
        """Test that network is created with correct structure."""
        assert small_network.n_agents == 10
        assert len(small_network.adjacency_df) == 10

        expected_columns = {"agent_id", "neighbors", "degree"}
        assert set(small_network.adjacency_df.columns) == expected_columns

    def test_vectorized_neighbor_lookup(self, small_network: SocialNetwork) -> None:
        """Test vectorized neighbor lookup functionality."""
        agent_ids = pl.Series([0, 1, 2])
        neighbors_df = small_network.get_neighbors_vectorized(agent_ids)

        assert isinstance(neighbors_df, pl.DataFrame)
        expected_columns = {"agent_id", "neighbor_id"}
        assert set(neighbors_df.columns) == expected_columns

        # Check that agents appear in results
        agent_ids_in_result = set(neighbors_df.get_column("agent_id").to_list())
        assert agent_ids_in_result.issubset({0, 1, 2})

    def test_k_hop_neighbor_computation(self, small_network: SocialNetwork) -> None:
        """Test k-hop neighbor computation."""
        agent_ids = pl.Series([0])
        k_hop_df = small_network.get_k_hop_neighbors(agent_ids, k=2)

        assert isinstance(k_hop_df, pl.DataFrame)
        if len(k_hop_df) > 0:
            expected_columns = {"original_agent", "k_hop_neighbor", "hop_distance"}
            assert set(k_hop_df.columns) == expected_columns

    def test_network_statistics(self, small_network: SocialNetwork) -> None:
        """Test network statistics computation."""
        stats = small_network.get_network_statistics()

        assert isinstance(stats, dict)
        required_stats = {"num_nodes", "num_edges", "mean_degree"}
        assert all(stat in stats for stat in required_stats)
        assert stats["num_nodes"] == 10

    def test_network_size_updates(self, small_network: SocialNetwork) -> None:
        """Test dynamic network size updates."""
        # Test growing network
        small_network.update_network_size(15)
        assert small_network.n_agents == 15
        assert len(small_network.adjacency_df) == 15

        # Test shrinking network
        small_network.update_network_size(8)
        assert small_network.n_agents == 8
        assert len(small_network.adjacency_df) == 8


class TestLearningEligibilityComputer:
    """Test learning eligibility computation logic."""

    @pytest.fixture
    def eligibility_computer(self, cultural_params: CulturalParams) -> LearningEligibilityComputer:
        """Create eligibility computer with standard config."""
        return LearningEligibilityComputer(cultural_params)

    def test_eligibility_computation_logic(
        self, eligibility_computer: LearningEligibilityComputer, sample_agents_df: pl.DataFrame
    ) -> None:
        """Test eligibility computation with realistic agent data."""
        eligibility = eligibility_computer.compute_eligibility(sample_agents_df, generation=10)

        assert isinstance(eligibility, pl.Series)
        assert len(eligibility) == len(sample_agents_df)
        assert eligibility.dtype == pl.Boolean

        # Agent 0 should not be eligible (age=0, energy=0.5, no neighbors)
        assert not eligibility[0]

        # Agent 4 should be eligible (age=10, energy=5.0, degree=4)
        assert eligibility[4]


class TestTransmissionEngines:
    """Test cultural transmission engines."""

    @pytest.fixture
    def oblique_engine(self) -> ObliqueTransmissionEngine:
        """Create oblique transmission engine."""
        config = CulturalParams()
        return ObliqueTransmissionEngine(config)

    @pytest.fixture
    def horizontal_engine(self) -> HorizontalTransmissionEngine:
        """Create horizontal transmission engine."""
        config = CulturalParams()
        return HorizontalTransmissionEngine(config)

    @pytest.fixture
    def agents_with_network(self) -> pl.DataFrame:
        """Create sample agents with network connections."""
        return pl.DataFrame(
            {
                "agent_id": range(5),
                "pref_culture": [100, 110, 120, 130, 140],
                "prestige_score": [0.2, 0.4, 0.6, 0.8, 1.0],
                "neighbors": [
                    [1, 2],  # Agent 0 connected to 1, 2
                    [0, 2, 3],  # Agent 1 connected to 0, 2, 3
                    [0, 1, 4],  # Agent 2 connected to 0, 1, 4
                    [1, 4],  # Agent 3 connected to 1, 4
                    [2, 3],  # Agent 4 connected to 2, 3
                ],
            }
        )

    def test_oblique_transmission_execution(
        self, oblique_engine: ObliqueTransmissionEngine, sample_agents_df: pl.DataFrame
    ) -> None:
        """Test oblique transmission execution."""
        learner_mask = pl.Series([True, True, False, False, False, False, False, False, False, False])

        result = oblique_engine.execute_transmission(sample_agents_df, learner_mask)

        assert isinstance(result, pl.DataFrame)
        expected_columns = {"learner_id", "new_preference", "teacher_id", "learning_type"}
        assert set(result.columns) == expected_columns

        if len(result) > 0:
            assert all(result.get_column("learning_type") == "oblique")
            assert all(result.get_column("learner_id").is_in([0, 1]))

    def test_horizontal_transmission_execution(
        self, horizontal_engine: HorizontalTransmissionEngine, agents_with_network: pl.DataFrame
    ) -> None:
        """Test horizontal transmission execution."""
        learner_mask = pl.Series([True, True, False, False, False])

        result = horizontal_engine.execute_transmission(agents_with_network, learner_mask)

        assert isinstance(result, pl.DataFrame)
        expected_columns = {"learner_id", "new_preference", "teacher_id", "learning_type"}
        assert set(result.columns) == expected_columns

        if len(result) > 0:
            assert all(result.get_column("learning_type") == "horizontal")
            assert all(result.get_column("learner_id").is_in([0, 1]))


class TestCulturalInnovationEngine:
    """Test cultural innovation mechanisms."""

    @pytest.fixture
    def innovation_engine(self) -> CulturalInnovationEngine:
        """Create innovation engine."""
        config = CulturalParams()
        return CulturalInnovationEngine(config)

    @pytest.fixture
    def uniform_agents(self) -> pl.DataFrame:
        """Create agents with uniform preferences for innovation testing."""
        return pl.DataFrame(
            {
                "agent_id": range(10),
                "pref_culture": [100] * 10,  # All same preference initially
            }
        )

    def test_cultural_innovation_execution(
        self, innovation_engine: CulturalInnovationEngine, uniform_agents: pl.DataFrame
    ) -> None:
        """Test cultural innovation execution."""
        result = innovation_engine.execute_innovation(uniform_agents, innovation_rate=0.5)

        assert isinstance(result, pl.DataFrame)
        expected_columns = {"learner_id", "new_preference", "teacher_id", "learning_type"}
        assert set(result.columns) == expected_columns

        if len(result) > 0:
            # Verify innovation events are properly marked
            assert all(result.get_column("learning_type") == "innovation")
            assert all(result.get_column("teacher_id").is_null())

            # Check that new preferences are different from original
            new_prefs = result.get_column("new_preference").to_list()
            assert any(pref != 100 for pref in new_prefs)


class TestMemoryDecayEngine:
    """Test cultural memory decay mechanisms."""

    @pytest.fixture
    def memory_engine(self) -> MemoryDecayEngine:
        """Create memory decay engine."""
        config = CulturalParams()
        return MemoryDecayEngine(config)

    def test_memory_decay_application(self, memory_engine: MemoryDecayEngine) -> None:
        """Test memory decay application."""
        agents_with_memory = pl.DataFrame(
            {
                "agent_id": range(3),
                "pref_culture": [100, 110, 120],
                "cultural_memory_0": [90.0, 95.0, 105.0],
                "cultural_memory_1": [80.0, 85.0, 95.0],
                "cultural_memory_2": [70.0, 75.0, 85.0],
                "memory_weights_0": [0.8, 0.9, 0.7],
                "memory_weights_1": [0.6, 0.7, 0.5],
                "memory_weights_2": [0.4, 0.5, 0.3],
            }
        )

        result = memory_engine.apply_memory_decay(agents_with_memory)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(agents_with_memory)

        # Check that memory columns are updated
        assert "cultural_memory_0" in result.columns
        assert "memory_weights_0" in result.columns

        # Check that new memory in position 0 comes from current preference
        new_memory_0 = result.get_column("cultural_memory_0").to_list()
        current_prefs = agents_with_memory.get_column("pref_culture").to_list()
        assert new_memory_0 == [float(p) for p in current_prefs]

    def test_effective_preference_computation(self, memory_engine: MemoryDecayEngine) -> None:
        """Test effective preference computation."""
        agents_with_memory = pl.DataFrame(
            {
                "agent_id": range(3),
                "pref_culture": [100, 110, 120],  # Add missing pref_culture column
                "cultural_memory_0": [90.0, 95.0, 105.0],
                "cultural_memory_1": [80.0, 85.0, 95.0],
                "cultural_memory_2": [70.0, 75.0, 85.0],
                "memory_weights_0": [0.8, 0.9, 0.7],
                "memory_weights_1": [0.6, 0.7, 0.5],
                "memory_weights_2": [0.4, 0.5, 0.3],
            }
        )

        effective_prefs = memory_engine.compute_effective_preference(agents_with_memory)

        assert isinstance(effective_prefs, pl.Series)
        assert len(effective_prefs) == len(agents_with_memory)
        assert effective_prefs.dtype == pl.UInt8


class TestCulturalLayer:
    """Test the integrated vectorized cultural layer."""

    @pytest.fixture
    def cultural_layer(self, mock_agent_set, cultural_params: CulturalParams) -> CulturalLayer:
        """Create vectorized cultural layer with mock agents."""
        return CulturalLayer(mock_agent_set, cultural_params)

    def test_cultural_layer_initialization(self, cultural_layer: CulturalLayer, mock_agent_set) -> None:
        """Test cultural layer initialization."""
        assert cultural_layer.agents == mock_agent_set
        assert cultural_layer.generation == 0
        assert hasattr(cultural_layer, "config")
        assert isinstance(cultural_layer.network, SocialNetwork)

    def test_cultural_layer_step_execution(self, cultural_layer: CulturalLayer) -> None:
        """Test cultural layer step execution."""
        initial_generation = cultural_layer.generation

        # Execute one step
        cultural_layer.step()

        # Check that generation incremented
        assert cultural_layer.generation == initial_generation + 1

        # Check that statistics were generated
        stats = cultural_layer.get_generation_statistics()
        assert isinstance(stats, dict)
        assert "generation" in stats
        assert "total_agents" in stats

    def test_cultural_diversity_computation(self, cultural_layer: CulturalLayer) -> None:
        """Test cultural diversity computation."""
        diversity = cultural_layer.compute_cultural_diversity()

        assert isinstance(diversity, float)
        assert diversity >= 0.0

    def test_reset_functionality(self, cultural_layer: CulturalLayer) -> None:
        """Test cultural layer reset functionality."""
        # Execute some steps
        cultural_layer.step()
        cultural_layer.step()

        assert cultural_layer.generation > 0

        # Reset and verify clean state
        cultural_layer.reset()
        assert cultural_layer.generation == 0
        assert len(cultural_layer.learning_events) == 0


class TestPerformanceCharacteristics:
    """Test performance characteristics of vectorized implementation."""

    def test_vectorized_operations_efficiency(self, cultural_params: CulturalParams) -> None:
        """Test that vectorized operations are efficient with large datasets."""
        import time

        computer = LearningEligibilityComputer(cultural_params)

        # Create large agent population
        import numpy as np

        large_df = pl.DataFrame(
            {
                "agent_id": range(1000),
                "age": np.random.randint(1, 21, 1000),
                "energy": np.random.uniform(1, 10, 1000),
                "network_degree": np.random.randint(0, 11, 1000),
                "last_learning_event": np.random.randint(0, 6, 1000),
            }
        )

        # Measure execution time
        start_time = time.time()
        eligibility = computer.compute_eligibility(large_df, generation=10)
        execution_time = time.time() - start_time

        # Should be very fast for 1000 agents
        assert execution_time < 0.1  # Less than 100ms
        assert isinstance(eligibility, pl.Series)
        assert len(eligibility) == 1000

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency of vectorized operations."""
        config = CulturalParams()
        engine = MemoryDecayEngine(config)

        # Create DataFrame with memory columns
        df = pl.DataFrame(
            {
                "agent_id": range(100),
                "pref_culture": [128] * 100,
            }
        )

        # Add memory columns
        for i in range(10):
            df = df.with_columns([pl.lit(0.0).alias(f"cultural_memory_{i}"), pl.lit(0.0).alias(f"memory_weights_{i}")])

        # Apply memory decay multiple times
        for _ in range(10):
            df = engine.apply_memory_decay(df)

        # Check that DataFrame maintains reasonable structure
        assert len(df) == 100
        assert len(df.columns) >= 22  # Base + memory columns
