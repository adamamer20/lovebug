"""
Tests for vectorized Layer 2 implementation.

Validates the performance and correctness of the new vectorized cultural
transmission system compared to the original implementation.
"""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

from lovebug.layer2 import (
    CulturalInnovationEngine,
    HorizontalTransmissionEngine,
    LearningEligibilityComputer,
    MemoryDecayEngine,
    NetworkTopology,
    ObliqueTransmissionEngine,
    VectorizedCulturalLayer,
    VectorizedSocialNetwork,
)
from lovebug.layer2.config import Layer2Config


class TestNetworkTopology:
    """Test network topology configuration."""

    def test_default_topology(self) -> None:
        """Test default topology configuration."""
        topology = NetworkTopology()
        assert topology.network_type == "small_world"
        assert topology.connectivity == 0.1
        assert 0.0 <= topology.connectivity <= 1.0

    def test_invalid_network_type(self) -> None:
        """Test validation of network type."""
        with pytest.raises(ValueError, match="network_type must be one of"):
            NetworkTopology(network_type="invalid")

    def test_invalid_connectivity(self) -> None:
        """Test validation of connectivity parameter."""
        with pytest.raises(ValueError, match="connectivity must be between 0 and 1"):
            NetworkTopology(connectivity=1.5)


class TestVectorizedSocialNetwork:
    """Test vectorized social network implementation."""

    @pytest.fixture
    def small_network(self) -> VectorizedSocialNetwork:
        """Create a small test network."""
        topology = NetworkTopology("random", connectivity=0.3)
        return VectorizedSocialNetwork(10, topology)

    def test_network_creation(self, small_network: VectorizedSocialNetwork) -> None:
        """Test network is created successfully."""
        assert small_network.n_agents == 10
        assert len(small_network.adjacency_df) == 10
        assert "agent_id" in small_network.adjacency_df.columns
        assert "neighbors" in small_network.adjacency_df.columns
        assert "degree" in small_network.adjacency_df.columns

    def test_neighbor_lookup(self, small_network: VectorizedSocialNetwork) -> None:
        """Test vectorized neighbor lookup."""
        agent_ids = pl.Series([0, 1, 2])
        neighbors_df = small_network.get_neighbors_vectorized(agent_ids)

        assert isinstance(neighbors_df, pl.DataFrame)
        assert "agent_id" in neighbors_df.columns
        assert "neighbor_id" in neighbors_df.columns

        # Check that agents appear in results
        agent_ids_in_result = set(neighbors_df.get_column("agent_id").to_list())
        assert agent_ids_in_result.issubset({0, 1, 2})

    def test_k_hop_neighbors(self, small_network: VectorizedSocialNetwork) -> None:
        """Test k-hop neighbor computation."""
        agent_ids = pl.Series([0])
        k_hop_df = small_network.get_k_hop_neighbors(agent_ids, k=2)

        assert isinstance(k_hop_df, pl.DataFrame)
        if len(k_hop_df) > 0:
            assert "original_agent" in k_hop_df.columns
            assert "k_hop_neighbor" in k_hop_df.columns
            assert "hop_distance" in k_hop_df.columns

    def test_network_statistics(self, small_network: VectorizedSocialNetwork) -> None:
        """Test network statistics computation."""
        stats = small_network.get_network_statistics()

        assert isinstance(stats, dict)
        assert "num_nodes" in stats
        assert "num_edges" in stats
        assert "mean_degree" in stats
        assert stats["num_nodes"] == 10

    def test_network_size_update(self, small_network: VectorizedSocialNetwork) -> None:
        """Test updating network size."""
        small_network.update_network_size(15)

        assert small_network.n_agents == 15
        assert len(small_network.adjacency_df) == 15

        # Test shrinking
        small_network.update_network_size(8)
        assert small_network.n_agents == 8
        assert len(small_network.adjacency_df) == 8


class TestLearningEligibilityComputer:
    """Test learning eligibility computation."""

    @pytest.fixture
    def eligibility_computer(self) -> LearningEligibilityComputer:
        """Create eligibility computer."""
        config = Layer2Config()
        return LearningEligibilityComputer(config)

    @pytest.fixture
    def sample_agents(self) -> pl.DataFrame:
        """Create sample agent DataFrame."""
        return pl.DataFrame(
            {
                "agent_id": range(5),
                "age": [0, 1, 2, 5, 10],
                "energy": [0.5, 1.5, 2.0, 3.0, 5.0],
                "network_degree": [0, 1, 2, 3, 4],
                "last_learning_event": [0, 0, 0, 2, 5],
            }
        )

    def test_eligibility_computation(
        self, eligibility_computer: LearningEligibilityComputer, sample_agents: pl.DataFrame
    ) -> None:
        """Test eligibility computation."""
        eligibility = eligibility_computer.compute_eligibility(sample_agents, generation=10)

        assert isinstance(eligibility, pl.Series)
        assert len(eligibility) == len(sample_agents)
        assert eligibility.dtype == pl.Boolean

        # Agent 0 should not be eligible (age=0, energy=0.5, no neighbors)
        assert not eligibility[0]

        # Agent 4 should be eligible (age=10, energy=5.0, degree=4)
        assert eligibility[4]


class TestObliqueTransmissionEngine:
    """Test oblique transmission engine."""

    @pytest.fixture
    def oblique_engine(self) -> ObliqueTransmissionEngine:
        """Create oblique transmission engine."""
        config = Layer2Config(oblique_transmission_rate=0.5)
        return ObliqueTransmissionEngine(config)

    @pytest.fixture
    def sample_agents(self) -> pl.DataFrame:
        """Create sample agent DataFrame with age distribution."""
        return pl.DataFrame(
            {
                "agent_id": range(10),
                "age": [1, 1, 2, 2, 5, 5, 10, 10, 15, 20],
                "pref_culture": [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
                "prestige_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "mating_success": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            }
        )

    def test_oblique_transmission(self, oblique_engine: ObliqueTransmissionEngine, sample_agents: pl.DataFrame) -> None:
        """Test oblique transmission execution."""
        learner_mask = pl.Series([True, True, False, False, False, False, False, False, False, False])

        result = oblique_engine.execute_transmission(sample_agents, learner_mask)

        assert isinstance(result, pl.DataFrame)
        expected_columns = {"learner_id", "new_preference", "teacher_id", "learning_type"}
        assert set(result.columns) == expected_columns

        if len(result) > 0:
            # Check that learning events are valid
            assert all(result.get_column("learning_type") == "oblique")
            assert all(result.get_column("learner_id").is_in([0, 1]))


class TestHorizontalTransmissionEngine:
    """Test horizontal transmission engine."""

    @pytest.fixture
    def horizontal_engine(self) -> HorizontalTransmissionEngine:
        """Create horizontal transmission engine."""
        config = Layer2Config(horizontal_transmission_rate=0.5)
        return HorizontalTransmissionEngine(config)

    @pytest.fixture
    def sample_agents_with_network(self) -> pl.DataFrame:
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

    def test_horizontal_transmission(
        self, horizontal_engine: HorizontalTransmissionEngine, sample_agents_with_network: pl.DataFrame
    ) -> None:
        """Test horizontal transmission execution."""
        learner_mask = pl.Series([True, True, False, False, False])

        result = horizontal_engine.execute_transmission(sample_agents_with_network, learner_mask)

        assert isinstance(result, pl.DataFrame)
        expected_columns = {"learner_id", "new_preference", "teacher_id", "learning_type"}
        assert set(result.columns) == expected_columns

        if len(result) > 0:
            # Check that learning events are valid
            assert all(result.get_column("learning_type") == "horizontal")
            assert all(result.get_column("learner_id").is_in([0, 1]))


class TestCulturalInnovationEngine:
    """Test cultural innovation engine."""

    @pytest.fixture
    def innovation_engine(self) -> CulturalInnovationEngine:
        """Create innovation engine."""
        config = Layer2Config(innovation_rate=0.3)
        return CulturalInnovationEngine(config)

    @pytest.fixture
    def sample_agents(self) -> pl.DataFrame:
        """Create sample agents."""
        return pl.DataFrame(
            {
                "agent_id": range(10),
                "pref_culture": [100] * 10,  # All same preference initially
            }
        )

    def test_cultural_innovation(
        self, innovation_engine: CulturalInnovationEngine, sample_agents: pl.DataFrame
    ) -> None:
        """Test cultural innovation execution."""
        result = innovation_engine.execute_innovation(sample_agents, innovation_rate=0.5)

        assert isinstance(result, pl.DataFrame)
        expected_columns = {"learner_id", "new_preference", "teacher_id", "learning_type"}
        assert set(result.columns) == expected_columns

        if len(result) > 0:
            # Check that innovation events are valid
            assert all(result.get_column("learning_type") == "innovation")
            assert all(result.get_column("teacher_id").is_null())

            # Check that new preferences are different from original
            new_prefs = result.get_column("new_preference").to_list()
            assert any(pref != 100 for pref in new_prefs)


class TestMemoryDecayEngine:
    """Test memory decay engine."""

    @pytest.fixture
    def memory_engine(self) -> MemoryDecayEngine:
        """Create memory decay engine."""
        config = Layer2Config(cultural_memory_size=3, memory_decay_rate=0.1, memory_update_strength=1.0)
        return MemoryDecayEngine(config)

    @pytest.fixture
    def sample_agents_with_memory(self) -> pl.DataFrame:
        """Create sample agents with memory."""
        return pl.DataFrame(
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

    def test_memory_decay(self, memory_engine: MemoryDecayEngine, sample_agents_with_memory: pl.DataFrame) -> None:
        """Test memory decay application."""
        result = memory_engine.apply_memory_decay(sample_agents_with_memory)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == len(sample_agents_with_memory)

        # Check that memory columns are updated
        assert "cultural_memory_0" in result.columns
        assert "memory_weights_0" in result.columns

        # Check that new memory in position 0 comes from current preference
        new_memory_0 = result.get_column("cultural_memory_0").to_list()
        current_prefs = sample_agents_with_memory.get_column("pref_culture").to_list()
        assert new_memory_0 == [float(p) for p in current_prefs]

    def test_effective_preference_computation(
        self, memory_engine: MemoryDecayEngine, sample_agents_with_memory: pl.DataFrame
    ) -> None:
        """Test effective preference computation."""
        effective_prefs = memory_engine.compute_effective_preference(sample_agents_with_memory)

        assert isinstance(effective_prefs, pl.Series)
        assert len(effective_prefs) == len(sample_agents_with_memory)
        assert effective_prefs.dtype == pl.UInt8


class TestVectorizedCulturalLayer:
    """Test the main vectorized cultural layer."""

    @pytest.fixture
    def mock_agent_set(self) -> Mock:
        """Create mock agent set."""
        mock_agents = Mock()
        mock_agents.agents = pl.DataFrame(
            {
                "agent_id": range(10),
                "genome": [123456] * 10,
                "energy": [5.0] * 10,
                "age": [2] * 10,
                "mating_success": [1] * 10,
                "pref_culture": [128] * 10,
                "cultural_innovation_count": [0] * 10,
                "prestige_score": [0.5] * 10,
                "last_learning_event": [0] * 10,
                "learning_eligibility": [True] * 10,
                "effective_preference": [128] * 10,
                "neighbors": [[] for _ in range(10)],
                "network_degree": [0] * 10,
            }
        )

        # Mock len() to return number of agents - fix lambda to accept self parameter
        mock_agents.__len__ = lambda self: 10

        return mock_agents

    @pytest.fixture
    def cultural_layer(self, mock_agent_set: Mock) -> VectorizedCulturalLayer:
        """Create vectorized cultural layer."""
        config = Layer2Config(
            oblique_transmission_rate=0.2, horizontal_transmission_rate=0.3, innovation_rate=0.1, cultural_memory_size=5
        )
        return VectorizedCulturalLayer(mock_agent_set, config)

    def test_cultural_layer_initialization(self, cultural_layer: VectorizedCulturalLayer, mock_agent_set: Mock) -> None:
        """Test cultural layer initialization."""
        assert cultural_layer.agents == mock_agent_set
        assert cultural_layer.generation == 0
        assert isinstance(cultural_layer.config, Layer2Config)
        assert isinstance(cultural_layer.network, VectorizedSocialNetwork)

    def test_cultural_layer_step(self, cultural_layer: VectorizedCulturalLayer) -> None:
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

    def test_cultural_diversity_computation(self, cultural_layer: VectorizedCulturalLayer) -> None:
        """Test cultural diversity computation."""
        diversity = cultural_layer.compute_cultural_diversity()

        assert isinstance(diversity, float)
        assert diversity >= 0.0

    def test_reset_functionality(self, cultural_layer: VectorizedCulturalLayer) -> None:
        """Test reset functionality."""
        # Execute some steps
        cultural_layer.step()
        cultural_layer.step()

        assert cultural_layer.generation > 0

        # Reset
        cultural_layer.reset()

        assert cultural_layer.generation == 0
        assert len(cultural_layer.learning_events) == 0


class TestPerformanceComparison:
    """Test performance characteristics of vectorized implementation."""

    def test_vectorized_vs_sequential_performance(self) -> None:
        """Test that vectorized operations are efficient."""
        # This is a placeholder for performance testing
        # In practice, we would benchmark against the original implementation

        config = Layer2Config()
        computer = LearningEligibilityComputer(config)

        # Create large agent population
        large_df = pl.DataFrame(
            {
                "agent_id": range(1000),
                "age": np.random.randint(1, 20, 1000),
                "energy": np.random.uniform(1, 10, 1000),
                "network_degree": np.random.randint(0, 10, 1000),
                "last_learning_event": np.random.randint(0, 5, 1000),
            }
        )

        # Measure time for vectorized computation
        import time

        start_time = time.time()
        eligibility = computer.compute_eligibility(large_df, generation=10)
        end_time = time.time()

        vectorized_time = end_time - start_time

        # Should be very fast for 1000 agents
        assert vectorized_time < 0.1  # Less than 100ms
        assert isinstance(eligibility, pl.Series)
        assert len(eligibility) == 1000

    def test_memory_efficiency(self) -> None:
        """Test memory efficiency of vectorized operations."""
        # Test that operations don't create excessive memory overhead
        config = Layer2Config(cultural_memory_size=10)
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

        # Check that DataFrame is still reasonable size
        assert len(df) == 100
        assert len(df.columns) >= 22  # Base + memory columns
