"""
Vectorized social network implementation for cultural transmission.

Provides fully vectorized network operations using Polars DataFrames and NumPy
arrays, with hybrid storage using adjacency lists and edge tables for optimal
performance across different operation types.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from beartype import beartype

__all__ = ["NetworkTopology", "VectorizedSocialNetwork"]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class NetworkTopology:
    """
    Configuration for social network topology generation.

    Parameters
    ----------
    network_type : str
        Type of network ('random', 'small_world', 'scale_free', 'grid')
    connectivity : float
        Network connectivity parameter (0-1)
    rewiring_prob : float
        Rewiring probability for small-world networks
    degree_preference : float
        Preferential attachment strength for scale-free networks

    Examples
    --------
    >>> topology = NetworkTopology("scale_free", connectivity=0.1)
    >>> topology.network_type
    'scale_free'
    """

    network_type: str = "small_world"
    connectivity: float = 0.1
    rewiring_prob: float = 0.1
    degree_preference: float = 1.0

    def __post_init__(self) -> None:
        """Validate topology parameters."""
        valid_types = {"random", "small_world", "scale_free", "grid"}
        if self.network_type not in valid_types:
            raise ValueError(f"network_type must be one of {valid_types}")

        if not (0.0 <= self.connectivity <= 1.0):
            raise ValueError("connectivity must be between 0 and 1")


class VectorizedSocialNetwork:
    """
    Fully vectorized social network operations for cultural transmission.

    Uses hybrid storage with adjacency lists for fast neighbor lookups and
    edge tables for complex graph operations. All operations are vectorized
    using Polars DataFrames and NumPy arrays.

    Parameters
    ----------
    n_agents : int
        Number of agents in the network
    topology : NetworkTopology
        Network topology configuration

    Examples
    --------
    >>> topology = NetworkTopology("scale_free", connectivity=0.1)
    >>> network = VectorizedSocialNetwork(1000, topology)
    >>> neighbors = network.get_neighbors_vectorized(pl.Series([0, 1, 2]))
    """

    def __init__(self, n_agents: int, topology: NetworkTopology) -> None:
        self.n_agents = n_agents
        self.topology = topology

        # Primary storage: adjacency lists in DataFrame
        self.adjacency_df = self._generate_adjacency_lists()

        # Secondary storage: edge table for complex operations
        self.edge_table = self._adjacency_to_edge_table()

        # Cached network metrics (computed on demand)
        self._metrics_cache: pl.DataFrame | None = None
        self._cache_generation = 0

        logger.info(f"Initialized {topology.network_type} network with {n_agents} agents")

    @beartype
    def _generate_adjacency_lists(self) -> pl.DataFrame:
        """Generate adjacency lists using vectorized operations."""
        if self.topology.network_type == "scale_free":
            return self._generate_scale_free_adjacency()
        elif self.topology.network_type == "small_world":
            return self._generate_small_world_adjacency()
        elif self.topology.network_type == "random":
            return self._generate_random_adjacency()
        elif self.topology.network_type == "grid":
            return self._generate_grid_adjacency()
        else:
            raise ValueError(f"Unknown network type: {self.topology.network_type}")

    def _generate_scale_free_adjacency(self) -> pl.DataFrame:
        """Vectorized Barabási-Albert network generation."""
        # Use numpy for efficient preferential attachment
        adjacency_matrix = np.zeros((self.n_agents, self.n_agents), dtype=bool)
        degrees = np.zeros(self.n_agents, dtype=int)

        m = max(1, int(self.topology.connectivity * self.n_agents / 2))

        # Initialize with complete graph for first m nodes
        for i in range(min(m, self.n_agents)):
            for j in range(i + 1, min(m, self.n_agents)):
                adjacency_matrix[i, j] = True
                adjacency_matrix[j, i] = True
                degrees[i] += 1
                degrees[j] += 1

        # Vectorized preferential attachment for remaining nodes
        for i in range(m, self.n_agents):
            if np.sum(degrees[:i]) == 0:
                # Fallback to random connection if no degrees yet
                target = np.random.randint(0, i)
                adjacency_matrix[i, target] = True
                adjacency_matrix[target, i] = True
                degrees[i] += 1
                degrees[target] += 1
            else:
                # Preferential attachment based on degrees
                probabilities = degrees[:i].astype(float)
                probabilities /= probabilities.sum()

                targets = np.random.choice(i, size=min(m, i), replace=False, p=probabilities)

                for target in targets:
                    adjacency_matrix[i, target] = True
                    adjacency_matrix[target, i] = True
                    degrees[i] += 1
                    degrees[target] += 1

        # Convert to Polars DataFrame
        return self._adjacency_matrix_to_dataframe(adjacency_matrix, degrees)

    def _generate_small_world_adjacency(self) -> pl.DataFrame:
        """Vectorized Watts-Strogatz small-world network generation."""
        # Calculate initial ring lattice degree
        k = max(2, int(self.topology.connectivity * self.n_agents))
        if k % 2 == 1:  # k must be even
            k += 1
        k = min(k, self.n_agents - 1)

        adjacency_matrix = np.zeros((self.n_agents, self.n_agents), dtype=bool)

        # Create ring lattice
        for i in range(self.n_agents):
            for j in range(1, k // 2 + 1):
                neighbor = (i + j) % self.n_agents
                adjacency_matrix[i, neighbor] = True
                adjacency_matrix[neighbor, i] = True

                neighbor = (i - j) % self.n_agents
                adjacency_matrix[i, neighbor] = True
                adjacency_matrix[neighbor, i] = True

        # Vectorized rewiring
        rewire_mask = np.random.random((self.n_agents, self.n_agents)) < self.topology.rewiring_prob
        rewire_mask = rewire_mask & adjacency_matrix  # Only rewire existing edges

        # Apply rewiring
        rewire_positions = np.where(rewire_mask)
        for i, j in zip(rewire_positions[0], rewire_positions[1]):
            if i < j:  # Avoid double-rewiring
                # Remove old edge
                adjacency_matrix[i, j] = False
                adjacency_matrix[j, i] = False

                # Add new random edge
                new_target = np.random.randint(0, self.n_agents)
                while new_target == i or adjacency_matrix[i, new_target]:
                    new_target = np.random.randint(0, self.n_agents)

                adjacency_matrix[i, new_target] = True
                adjacency_matrix[new_target, i] = True

        degrees = adjacency_matrix.sum(axis=1)
        return self._adjacency_matrix_to_dataframe(adjacency_matrix, degrees)

    def _generate_random_adjacency(self) -> pl.DataFrame:
        """Vectorized Erdős-Rényi random network generation."""
        adjacency_matrix = np.random.random((self.n_agents, self.n_agents)) < self.topology.connectivity

        # Make symmetric and remove self-loops
        adjacency_matrix = adjacency_matrix | adjacency_matrix.T
        np.fill_diagonal(adjacency_matrix, False)

        degrees = adjacency_matrix.sum(axis=1)
        return self._adjacency_matrix_to_dataframe(adjacency_matrix, degrees)

    def _generate_grid_adjacency(self) -> pl.DataFrame:
        """Vectorized 2D grid network with periodic boundaries."""
        side_length = int(np.ceil(np.sqrt(self.n_agents)))
        adjacency_matrix = np.zeros((self.n_agents, self.n_agents), dtype=bool)

        for i in range(self.n_agents):
            row = i // side_length
            col = i % side_length

            # Connect to 4 neighbors with periodic boundaries
            neighbors = [
                ((row + 1) % side_length) * side_length + col,  # down
                ((row - 1) % side_length) * side_length + col,  # up
                row * side_length + ((col + 1) % side_length),  # right
                row * side_length + ((col - 1) % side_length),  # left
            ]

            for neighbor in neighbors:
                if neighbor < self.n_agents and neighbor != i:
                    adjacency_matrix[i, neighbor] = True
                    adjacency_matrix[neighbor, i] = True

        degrees = adjacency_matrix.sum(axis=1)
        return self._adjacency_matrix_to_dataframe(adjacency_matrix, degrees)

    def _adjacency_matrix_to_dataframe(self, adjacency_matrix: np.ndarray, degrees: np.ndarray) -> pl.DataFrame:
        """Convert adjacency matrix to Polars DataFrame with neighbor lists."""
        neighbor_lists = []
        for i in range(self.n_agents):
            neighbors = np.where(adjacency_matrix[i])[0].tolist()
            neighbor_lists.append(neighbors)

        return pl.DataFrame(
            {
                "agent_id": pl.Series(range(self.n_agents), dtype=pl.UInt32),
                "neighbors": neighbor_lists,
                "degree": pl.Series(degrees.astype(int), dtype=pl.UInt32),
            }
        )

    def _adjacency_to_edge_table(self) -> pl.DataFrame:
        """Convert adjacency lists to edge table for complex operations."""
        # Explode neighbor lists to create edge pairs
        exploded = self.adjacency_df.select([pl.col("agent_id"), pl.col("neighbors")]).explode("neighbors")

        # Filter to unique edges (i < j to avoid duplicates)
        edges_df = exploded.filter(pl.col("agent_id") < pl.col("neighbors")).select(
            [
                pl.col("agent_id").alias("source"),
                pl.col("neighbors").alias("target"),
                pl.lit(1.0).alias("weight"),  # Default weight
                pl.lit(0).alias("created_gen"),  # Creation generation
            ]
        )

        return edges_df

    @beartype
    def get_neighbors_vectorized(self, agent_ids: pl.Series, max_neighbors: int = 10) -> pl.DataFrame:
        """
        Vectorized neighbor lookup for multiple agents.

        Parameters
        ----------
        agent_ids : pl.Series
            Series of agent IDs to get neighbors for
        max_neighbors : int, default=10
            Maximum number of neighbors to return per agent

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: agent_id, neighbor_id

        Examples
        --------
        >>> network = VectorizedSocialNetwork(100, NetworkTopology("random"))
        >>> agent_ids = pl.Series([0, 1, 2])
        >>> neighbors = network.get_neighbors_vectorized(agent_ids)
        """
        # Join agent IDs with adjacency list
        neighbor_df = pl.DataFrame({"agent_id": agent_ids}).join(self.adjacency_df, on="agent_id", how="inner")

        # Explode neighbor lists and limit per agent
        result = (
            neighbor_df.select([pl.col("agent_id"), pl.col("neighbors")])
            .explode("neighbors")
            .rename({"neighbors": "neighbor_id"})
        )

        # Apply max_neighbors limit using window function
        if max_neighbors > 0:
            result = (
                result.with_columns(pl.col("neighbor_id").rank("random").over("agent_id").alias("neighbor_rank"))
                .filter(pl.col("neighbor_rank") <= max_neighbors)
                .drop("neighbor_rank")
            )

        return result

    @beartype
    def get_k_hop_neighbors(self, agent_ids: pl.Series, k: int = 2) -> pl.DataFrame:
        """
        Vectorized k-hop neighbor computation.

        Parameters
        ----------
        agent_ids : pl.Series
            Series of agent IDs to start from
        k : int, default=2
            Number of hops to traverse

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: original_agent, k_hop_neighbor, hop_distance
        """
        current_neighbors = pl.DataFrame({"agent_id": agent_ids})
        all_neighbors = []

        for hop in range(k):
            # Get neighbors for current level
            next_neighbors = self.get_neighbors_vectorized(current_neighbors.get_column("agent_id")).rename(
                {"agent_id": "source", "neighbor_id": "agent_id"}
            )

            # Track hop distance
            next_neighbors = next_neighbors.with_columns(pl.lit(hop + 1).alias("hop_distance"))

            all_neighbors.append(next_neighbors)
            current_neighbors = next_neighbors.select(["agent_id"]).unique()

        # Combine all hop levels
        if all_neighbors:
            return pl.concat(all_neighbors).rename({"source": "original_agent", "agent_id": "k_hop_neighbor"})
        else:
            return pl.DataFrame({"original_agent": [], "k_hop_neighbor": [], "hop_distance": []})

    @beartype
    def compute_local_cultural_diversity(self, agent_df: pl.DataFrame) -> pl.Series:
        """
        Vectorized computation of local cultural diversity.

        Parameters
        ----------
        agent_df : pl.DataFrame
            Agent DataFrame with agent_id and pref_culture columns

        Returns
        -------
        pl.Series
            Shannon diversity index for each agent's neighborhood
        """
        # Join agents with their neighbors' cultural preferences
        neighbor_culture = (
            agent_df.select([pl.col("agent_id"), pl.col("pref_culture")])
            .join(self.adjacency_df, on="agent_id", how="inner")
            .explode("neighbors")
            .rename({"neighbors": "neighbor_id"})
        )

        # Join with neighbor cultural preferences
        neighbor_culture = neighbor_culture.join(
            agent_df.select(
                [pl.col("agent_id").alias("neighbor_id"), pl.col("pref_culture").alias("neighbor_culture")]
            ),
            on="neighbor_id",
            how="inner",
        )

        # Compute Shannon diversity for each agent's neighborhood
        diversity = neighbor_culture.group_by("agent_id").agg(
            [
                # Shannon entropy: -sum(p * log(p))
                pl.col("neighbor_culture")
                .value_counts()
                .map_elements(lambda counts: -np.sum((counts / counts.sum()) * np.log(counts / counts.sum() + 1e-10)))
                .alias("shannon_diversity")
            ]
        )

        return diversity.get_column("shannon_diversity")

    def update_network_size(self, new_size: int) -> None:
        """
        Update network when population size changes.

        Parameters
        ----------
        new_size : int
            New number of agents in the network
        """
        if new_size == self.n_agents:
            return

        old_size = self.n_agents
        self.n_agents = new_size

        if new_size > old_size:
            # Add new agents with random connections
            self._add_new_agents(old_size, new_size)
        elif new_size < old_size:
            # Remove agents and their connections
            self._remove_agents(new_size, old_size)

        # Invalidate cache
        self._metrics_cache = None

        logger.info(f"Updated network size from {old_size} to {new_size}")

    def _add_new_agents(self, old_size: int, new_size: int) -> None:
        """Add new agents to existing network."""
        new_agents = list(range(old_size, new_size))

        # Generate connections for new agents based on topology
        if self.topology.network_type == "scale_free":
            # Preferential attachment for new agents
            existing_degrees = self.adjacency_df.get_column("degree").to_numpy()

            for new_agent in new_agents:
                if np.sum(existing_degrees) > 0:
                    probs = existing_degrees / np.sum(existing_degrees)
                    n_connections = max(1, int(self.topology.connectivity * old_size))
                    targets = np.random.choice(old_size, size=min(n_connections, old_size), replace=False, p=probs)

                    new_neighbors = targets.tolist()
                else:
                    new_neighbors = []

                # Add new agent to adjacency DataFrame with consistent types
                new_row = pl.DataFrame(
                    {
                        "agent_id": pl.Series([new_agent], dtype=pl.UInt32),
                        "neighbors": [new_neighbors],
                        "degree": pl.Series([len(new_neighbors)], dtype=pl.UInt32),
                    }
                )
                self.adjacency_df = pl.concat([self.adjacency_df, new_row])

                # Update existing agents' neighbor lists
                for target in new_neighbors:
                    self._add_edge_to_adjacency(target, new_agent)
        else:
            # Random connections for other network types
            for new_agent in new_agents:
                n_connections = max(1, int(self.topology.connectivity * old_size))
                if old_size > 0:
                    targets = np.random.choice(old_size, size=min(n_connections, old_size), replace=False)
                    new_neighbors = targets.tolist()
                else:
                    new_neighbors = []

                new_row = pl.DataFrame(
                    {
                        "agent_id": pl.Series([new_agent], dtype=pl.UInt32),
                        "neighbors": [new_neighbors],
                        "degree": pl.Series([len(new_neighbors)], dtype=pl.UInt32),
                    }
                )
                self.adjacency_df = pl.concat([self.adjacency_df, new_row])

                for target in new_neighbors:
                    self._add_edge_to_adjacency(target, new_agent)

    def _remove_agents(self, new_size: int, old_size: int) -> None:
        """Remove agents from network."""
        # Filter out removed agents
        self.adjacency_df = self.adjacency_df.filter(pl.col("agent_id") < new_size)

        # Remove references to removed agents from neighbor lists
        self.adjacency_df = self.adjacency_df.with_columns(
            [
                pl.col("neighbors")
                .map_elements(lambda neighbors: [n for n in neighbors if n < new_size], return_dtype=pl.List(pl.UInt32))
                .alias("neighbors")
            ]
        ).with_columns([pl.col("neighbors").list.len().alias("degree")])

    def _add_edge_to_adjacency(self, agent_id: int, new_neighbor: int) -> None:
        """Add edge to existing agent's neighbor list."""
        # This is less efficient but necessary for dynamic updates
        current_neighbors = (
            self.adjacency_df.filter(pl.col("agent_id") == agent_id).get_column("neighbors").to_list()[0]
        )

        if new_neighbor not in current_neighbors:
            current_neighbors.append(new_neighbor)

            # Update the DataFrame
            self.adjacency_df = self.adjacency_df.with_columns(
                [
                    pl.when(pl.col("agent_id") == agent_id)
                    .then(pl.lit(current_neighbors))
                    .otherwise(pl.col("neighbors"))
                    .alias("neighbors")
                ]
            ).with_columns([pl.col("neighbors").list.len().alias("degree")])

    def get_network_statistics(self) -> dict[str, Any]:
        """
        Compute comprehensive network statistics.

        Returns
        -------
        dict[str, Any]
            Dictionary containing network metrics
        """
        stats = {}

        # Basic statistics
        degrees = self.adjacency_df.get_column("degree").to_numpy()
        stats["num_nodes"] = self.n_agents
        stats["num_edges"] = np.sum(degrees) // 2  # Undirected graph
        stats["density"] = stats["num_edges"] / (self.n_agents * (self.n_agents - 1) / 2)

        # Degree statistics
        stats["mean_degree"] = float(np.mean(degrees))
        stats["std_degree"] = float(np.std(degrees))
        stats["max_degree"] = int(np.max(degrees)) if len(degrees) > 0 else 0
        stats["min_degree"] = int(np.min(degrees)) if len(degrees) > 0 else 0

        # Clustering coefficient (approximate for performance)
        stats["mean_clustering"] = self._compute_mean_clustering()

        return stats

    def _compute_mean_clustering(self) -> float:
        """Compute approximate mean clustering coefficient."""
        clustering_sum = 0.0
        n_computed = 0

        # Sample subset for performance on large networks
        sample_size = min(100, self.n_agents)
        sampled_agents = np.random.choice(self.n_agents, sample_size, replace=False)

        for agent_id in sampled_agents:
            neighbors = self.adjacency_df.filter(pl.col("agent_id") == agent_id).get_column("neighbors").to_list()[0]

            if len(neighbors) < 2:
                continue

            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2

            for i, neighbor1 in enumerate(neighbors):
                for neighbor2 in neighbors[i + 1 :]:
                    # Check if neighbor1 and neighbor2 are connected
                    neighbor1_neighbors = (
                        self.adjacency_df.filter(pl.col("agent_id") == neighbor1).get_column("neighbors").to_list()[0]
                    )

                    if neighbor2 in neighbor1_neighbors:
                        triangles += 1

            if possible_triangles > 0:
                clustering_sum += triangles / possible_triangles
                n_computed += 1

        return clustering_sum / n_computed if n_computed > 0 else 0.0

    def to_edge_dataframe(self) -> pl.DataFrame:
        """
        Convert network to edge DataFrame for analysis.

        Returns
        -------
        pl.DataFrame
            DataFrame with source, target, weight columns
        """
        return self.edge_table.clone()
