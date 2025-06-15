"""
Social network structures for cultural transmission in Layer 2.

Implements different network topologies and manages social connections
for cultural learning interactions between agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import polars as pl
from beartype import beartype

__all__ = ["SocialNetwork", "NetworkTopology"]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class NetworkTopology:
    """
    Configuration for social network topology.

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
    """

    network_type: str = "small_world"
    connectivity: float = 0.1
    rewiring_prob: float = 0.1
    degree_preference: float = 1.0


class SocialNetwork:
    """
    Social network for managing cultural learning interactions.

    Provides efficient neighbor queries, network statistics, and supports
    different network topologies for studying cultural transmission patterns.

    Parameters
    ----------
    n_agents : int
        Number of agents in the network
    topology : NetworkTopology
        Network topology configuration

    Examples
    --------
    >>> network = SocialNetwork(1000, NetworkTopology("small_world", 0.1))
    >>> neighbors = network.get_neighbors(agent_id=42, max_neighbors=5)
    >>> network.update_network_size(1200)
    """

    def __init__(self, n_agents: int, topology: NetworkTopology) -> None:
        self.n_agents = n_agents
        self.topology = topology
        self.graph: nx.Graph = self._create_network()
        self._neighbor_cache: dict[int, list[int]] = {}
        self._cache_valid = False

    def _create_network(self) -> nx.Graph:
        """Create network graph based on topology configuration."""
        try:
            if self.topology.network_type == "random":
                return self._create_random_network()
            elif self.topology.network_type == "small_world":
                return self._create_small_world_network()
            elif self.topology.network_type == "scale_free":
                return self._create_scale_free_network()
            elif self.topology.network_type == "grid":
                return self._create_grid_network()
            else:
                raise ValueError(f"Unknown network type: {self.topology.network_type}")

        except Exception as e:
            logger.error(f"Failed to create {self.topology.network_type} network: {e}")
            # Fallback to simple random network
            logger.warning("Falling back to random network")
            return self._create_random_network()

    def _create_random_network(self) -> nx.Graph:
        """Create Erdős-Rényi random network."""
        return nx.erdos_renyi_graph(self.n_agents, self.topology.connectivity)

    def _create_small_world_network(self) -> nx.Graph:
        """Create Watts-Strogatz small-world network."""
        # Calculate degree from connectivity
        k = max(2, int(self.topology.connectivity * self.n_agents))
        if k % 2 == 1:  # k must be even
            k += 1
        k = min(k, self.n_agents - 1)

        return nx.watts_strogatz_graph(self.n_agents, k, self.topology.rewiring_prob)

    def _create_scale_free_network(self) -> nx.Graph:
        """Create Barabási-Albert scale-free network."""
        m = max(1, int(self.topology.connectivity * self.n_agents / 2))
        m = min(m, self.n_agents - 1)

        return nx.barabasi_albert_graph(self.n_agents, m)

    def _create_grid_network(self) -> nx.Graph:
        """Create 2D grid network with periodic boundary conditions."""
        # Create square-ish grid
        side_length = int(np.sqrt(self.n_agents))
        if side_length * side_length < self.n_agents:
            side_length += 1

        # Create grid and add extra nodes if needed
        graph = nx.grid_2d_graph(side_length, side_length, periodic=True)

        # Convert to integer node labels
        mapping = {node: i for i, node in enumerate(graph.nodes())}
        graph = nx.relabel_nodes(graph, mapping)

        # Add remaining nodes if needed
        for i in range(len(graph.nodes()), self.n_agents):
            graph.add_node(i)
            # Connect to random existing nodes
            existing_nodes = list(graph.nodes())
            if existing_nodes:
                target = np.random.choice(existing_nodes[:-1])
                graph.add_edge(i, target)

        return graph

    @beartype
    def get_neighbors(self, agent_id: int, max_neighbors: int = 5) -> list[int]:
        """
        Get neighbors of an agent for cultural learning.

        Parameters
        ----------
        agent_id : int
            ID of the agent
        max_neighbors : int, default=5
            Maximum number of neighbors to return

        Returns
        -------
        list[int]
            List of neighbor agent IDs
        """
        if agent_id not in self.graph:
            logger.warning(f"Agent {agent_id} not in network")
            return []

        # Use cached neighbors if available
        if self._cache_valid and agent_id in self._neighbor_cache:
            neighbors = self._neighbor_cache[agent_id]
        else:
            neighbors = list(self.graph.neighbors(agent_id))
            self._neighbor_cache[agent_id] = neighbors

        # Return random subset if too many neighbors
        if len(neighbors) <= max_neighbors:
            return neighbors
        else:
            return list(np.random.choice(neighbors, max_neighbors, replace=False))

    @beartype
    def get_local_neighbors(self, agent_id: int, radius: int = 1) -> list[int]:
        """
        Get neighbors within a specific radius (local neighborhood).

        Parameters
        ----------
        agent_id : int
            ID of the agent
        radius : int, default=1
            Network radius for neighborhood

        Returns
        -------
        list[int]
            List of agents within radius
        """
        if agent_id not in self.graph:
            return []

        if radius == 1:
            return self.get_neighbors(agent_id)

        # Use BFS to find neighbors within radius
        neighbors = set()
        current_level = {agent_id}

        for _ in range(radius):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.neighbors(node))
            neighbors.update(next_level)
            current_level = next_level

        # Remove the agent itself
        neighbors.discard(agent_id)
        return list(neighbors)

    @beartype
    def get_random_agents(self, n_agents: int, exclude: int | None = None) -> list[int]:
        """
        Get random agents from the network (for global learning).

        Parameters
        ----------
        n_agents : int
            Number of random agents to select
        exclude : int, optional
            Agent ID to exclude from selection

        Returns
        -------
        list[int]
            List of randomly selected agent IDs
        """
        available_agents = list(self.graph.nodes())
        if exclude is not None and exclude in available_agents:
            available_agents.remove(exclude)

        n_agents = min(n_agents, len(available_agents))
        if n_agents == 0:
            return []

        return list(np.random.choice(available_agents, n_agents, replace=False))

    @beartype
    def update_network_size(self, new_size: int) -> None:
        """
        Update network size when population changes.

        Parameters
        ----------
        new_size : int
            New number of agents
        """
        if new_size == self.n_agents:
            return

        old_size = self.n_agents
        self.n_agents = new_size

        if new_size > old_size:
            # Add new nodes
            for i in range(old_size, new_size):
                self.graph.add_node(i)
                # Connect to existing nodes based on topology
                self._connect_new_node(i)
        elif new_size < old_size:
            # Remove nodes
            nodes_to_remove = list(range(new_size, old_size))
            self.graph.remove_nodes_from(nodes_to_remove)

        self._invalidate_cache()

    def _connect_new_node(self, node_id: int) -> None:
        """Connect a new node to the network based on topology."""
        if self.topology.network_type == "random":
            # Connect to random existing nodes
            existing_nodes = [n for n in self.graph.nodes() if n != node_id]
            n_connections = int(self.topology.connectivity * len(existing_nodes))
            if n_connections > 0 and existing_nodes:
                targets = np.random.choice(existing_nodes, min(n_connections, len(existing_nodes)), replace=False)
                for target in targets:
                    self.graph.add_edge(node_id, target)

        elif self.topology.network_type == "scale_free":
            # Preferential attachment
            existing_nodes = [n for n in self.graph.nodes() if n != node_id]
            if existing_nodes:
                degrees = [dict(self.graph.degree())[n] + 1 for n in existing_nodes]  # +1 to avoid zero
                probabilities = np.array(degrees, dtype=float)
                probabilities /= probabilities.sum()

                n_connections = max(1, int(self.topology.connectivity * len(existing_nodes)))
                targets = np.random.choice(
                    existing_nodes, min(n_connections, len(existing_nodes)), p=probabilities, replace=False
                )
                for target in targets:
                    self.graph.add_edge(node_id, target)

        else:
            # For other topologies, connect to random neighbors
            existing_nodes = [n for n in self.graph.nodes() if n != node_id]
            if existing_nodes:
                target = np.random.choice(existing_nodes)
                self.graph.add_edge(node_id, target)

    def _invalidate_cache(self) -> None:
        """Invalidate neighbor cache."""
        self._neighbor_cache.clear()
        self._cache_valid = False

    def compute_network_statistics(self) -> dict[str, float]:
        """
        Compute network statistics for analysis.

        Returns
        -------
        dict[str, float]
            Dictionary of network statistics
        """
        if self.graph.number_of_nodes() == 0:
            return {}

        stats = {}

        try:
            # Basic statistics
            stats["num_nodes"] = self.graph.number_of_nodes()
            stats["num_edges"] = self.graph.number_of_edges()
            stats["density"] = nx.density(self.graph)

            # Degree statistics
            degree_dict = dict(self.graph.degree())
            degrees = list(degree_dict.values())
            stats["mean_degree"] = np.mean(degrees)
            stats["std_degree"] = np.std(degrees)
            stats["max_degree"] = np.max(degrees) if degrees else 0

            # Connectivity
            if nx.is_connected(self.graph):
                stats["is_connected"] = 1.0
                stats["diameter"] = nx.diameter(self.graph)
                stats["average_path_length"] = nx.average_shortest_path_length(self.graph)
            else:
                stats["is_connected"] = 0.0
                stats["diameter"] = float("inf")
                stats["average_path_length"] = float("inf")
                stats["num_components"] = nx.number_connected_components(self.graph)

            # Clustering
            stats["clustering_coefficient"] = nx.average_clustering(self.graph)

        except Exception as e:
            logger.warning(f"Error computing network statistics: {e}")

        return stats

    def visualize_network(self, filename: str | None = None, highlight_nodes: list[int] | None = None) -> None:
        """
        Create a visualization of the network.

        Parameters
        ----------
        filename : str, optional
            File to save the visualization
        highlight_nodes : list[int], optional
            Nodes to highlight in the visualization
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # Choose layout based on network type
            if self.topology.network_type == "grid":
                pos = {
                    i: (i % int(np.sqrt(self.n_agents)), i // int(np.sqrt(self.n_agents))) for i in self.graph.nodes()
                }
            elif self.n_agents < 100:
                pos = nx.spring_layout(self.graph, k=1, iterations=50)
            else:
                pos = nx.spring_layout(self.graph, k=0.5, iterations=20)

            # Draw network
            nx.draw_networkx_edges(self.graph, pos, alpha=0.5, width=0.5)

            # Highlight specific nodes if provided
            if highlight_nodes:
                regular_nodes = [n for n in self.graph.nodes() if n not in highlight_nodes]
                nx.draw_networkx_nodes(self.graph, pos, nodelist=regular_nodes, node_color="lightblue", node_size=20)
                nx.draw_networkx_nodes(self.graph, pos, nodelist=highlight_nodes, node_color="red", node_size=50)
            else:
                nx.draw_networkx_nodes(self.graph, pos, node_color="lightblue", node_size=20)

            plt.title(f"{self.topology.network_type.title()} Network (N={self.n_agents})")
            plt.axis("off")

            if filename:
                plt.savefig(filename, dpi=150, bbox_inches="tight")
                logger.info(f"Network visualization saved to {filename}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("Matplotlib not available for network visualization")

    def to_polars_dataframe(self) -> pl.DataFrame:
        """
        Convert network to Polars DataFrame for analysis.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns: source, target, weight (if applicable)
        """
        edges = list(self.graph.edges())
        if not edges:
            return pl.DataFrame({"source": [], "target": []})

        edge_data = {"source": [edge[0] for edge in edges], "target": [edge[1] for edge in edges]}

        # Add weights if they exist
        if nx.is_weighted(self.graph):
            edge_data["weight"] = [self.graph[u][v].get("weight", 1.0) for u, v in edges]

        return pl.DataFrame(edge_data)
