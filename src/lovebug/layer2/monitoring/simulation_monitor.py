"""
Rich console monitoring system for Layer 2 social learning simulations.

Provides beautiful progress tracking, metrics visualization, and debugging
support using rich console and icecream for development.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import polars as pl
from beartype import beartype
from icecream import ic
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

__all__ = ["SimulationMonitor"]

# Configure icecream for debugging
ic.configureOutput(prefix="🐛 Layer2 | ", includeContext=True)

logger = logging.getLogger(__name__)


class SimulationMonitor:
    """
    Rich console monitoring for Layer 2 social learning simulations.

    Provides beautiful progress visualization, metrics tracking, and event logging
    using rich console for enhanced debugging and monitoring experience.

    Parameters
    ----------
    console : Console, optional
        Rich console instance for output
    show_debug : bool, default=True
        Whether to show debug output via icecream

    Examples
    --------
    >>> monitor = SimulationMonitor()
    >>> with monitor.track_simulation(1000) as tracker:
    ...     for generation in range(1000):
    ...         monitor.log_generation(generation, {"diversity": 0.5})
    ...         tracker.advance()
    """

    def __init__(self, console: Console | None = None, show_debug: bool = True) -> None:
        self.console = console or Console()
        self.show_debug = show_debug
        self.progress: Progress | None = None
        self.current_task: TaskID | None = None
        self._start_time: float = 0.0
        self._generation_times: list[float] = []
        self._cultural_events: list[dict[str, Any]] = []

        if not show_debug:
            ic.disable()

    @beartype
    def start_simulation(self, n_generations: int, model_info: dict[str, Any] | None = None) -> None:
        """
        Start monitoring a new simulation run.

        Parameters
        ----------
        n_generations : int
            Total number of generations to simulate
        model_info : dict[str, Any], optional
            Additional model information to display
        """
        self._start_time = time.time()
        self._generation_times.clear()
        self._cultural_events.clear()

        # Create beautiful header
        title = "🐞💘 LoveBug Layer 2 Simulation"
        if model_info:
            subtitle = " | ".join(f"{k}: {v}" for k, v in model_info.items())
            header_text = f"{title}\n{subtitle}"
        else:
            header_text = title

        self.console.print(Panel.fit(header_text, style="bold blue"))

        # Setup progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        self.progress.start()
        self.current_task = self.progress.add_task("Running simulation...", total=n_generations)

        ic(f"Simulation started: {n_generations} generations")
        logger.info(f"Started simulation with {n_generations} generations")

    @contextmanager
    def track_simulation(
        self, n_generations: int, model_info: dict[str, Any] | None = None
    ) -> Generator[Progress, None, None]:
        """
        Context manager for tracking simulation progress.

        Parameters
        ----------
        n_generations : int
            Total number of generations
        model_info : dict[str, Any], optional
            Model information to display

        Yields
        ------
        Progress
            Rich progress instance for manual updates
        """
        self.start_simulation(n_generations, model_info)
        try:
            if self.progress is None:
                raise RuntimeError("Progress not initialized")
            yield self.progress
        finally:
            self.end_simulation()

    @beartype
    def log_generation(self, generation: int, metrics: dict[str, float], log_frequency: int = 10) -> None:
        """
        Log metrics for a specific generation.

        Parameters
        ----------
        generation : int
            Current generation number
        metrics : dict[str, float]
            Metrics to log and display
        log_frequency : int, default=10
            How often to display detailed metrics
        """
        generation_time = time.time()
        self._generation_times.append(generation_time)

        # Update progress
        if self.progress and self.current_task is not None:
            self.progress.advance(self.current_task)

            # Update description with key metrics
            if metrics:
                key_metric = next(iter(metrics.items()))
                description = f"Gen {generation} | {key_metric[0]}: {key_metric[1]:.3f}"
                self.progress.update(self.current_task, description=description)

        # Detailed logging at specified frequency
        if generation % log_frequency == 0 and metrics:
            self._display_generation_metrics(generation, metrics)

        # Debug logging
        if self.show_debug and generation % (log_frequency * 2) == 0:
            ic(f"Generation {generation} metrics", metrics)

    def _display_generation_metrics(self, generation: int, metrics: dict[str, float]) -> None:
        """Display detailed metrics table for a generation."""
        table = Table(title=f"Generation {generation} Metrics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")
        table.add_column("Status", style="yellow")

        for key, value in metrics.items():
            # Add status indicators based on metric type and value
            status = self._get_metric_status(key, value)
            table.add_row(key.replace("_", " ").title(), f"{value:.4f}", status)

        self.console.print(table)

    def _get_metric_status(self, metric_name: str, value: float) -> str:
        """Get status indicator for a metric value."""
        if "diversity" in metric_name.lower():
            if value > 0.7:
                return "🟢 High"
            elif value > 0.3:
                return "🟡 Medium"
            else:
                return "🔴 Low"
        elif "covariance" in metric_name.lower() or "correlation" in metric_name.lower():
            if abs(value) > 0.5:
                return "🟢 Strong"
            elif abs(value) > 0.2:
                return "🟡 Moderate"
            else:
                return "🔴 Weak"
        elif "distance" in metric_name.lower():
            if value > 2.0:
                return "🟢 Diverged"
            elif value > 1.0:
                return "🟡 Diverging"
            else:
                return "🔴 Similar"
        else:
            return "📊 Normal"

    @beartype
    def log_cultural_event(self, agent_id: int, event_type: str, details: dict[str, Any]) -> None:
        """
        Log an individual cultural learning event.

        Parameters
        ----------
        agent_id : int
            ID of the agent involved in the event
        event_type : str
            Type of cultural learning event
        details : dict[str, Any]
            Additional details about the event
        """
        event = {"timestamp": time.time(), "agent_id": agent_id, "event_type": event_type, "details": details}
        self._cultural_events.append(event)

        # Debug output for cultural events
        if self.show_debug:
            ic(f"Cultural event: Agent {agent_id} - {event_type}", details)

    @beartype
    def log_memory_usage(self, stage: str) -> None:
        """
        Log current memory usage at a specific stage.

        Parameters
        ----------
        stage : str
            Description of the current stage
        """
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            # Use rich to display memory info
            memory_text = Text(f"Memory: {memory_mb:.1f} MB", style="blue")
            self.console.print(f"[{stage}] {memory_text}")

            if self.show_debug:
                ic(f"Memory usage at {stage}", f"{memory_mb:.1f} MB")

        except ImportError:
            logger.warning("psutil not available for memory monitoring")

    @beartype
    def create_metrics_summary(self, final_metrics: dict[str, float]) -> None:
        """
        Create a beautiful summary of final simulation metrics.

        Parameters
        ----------
        final_metrics : dict[str, float]
            Final simulation metrics to display
        """
        # Calculate simulation statistics
        total_time = time.time() - self._start_time
        avg_generation_time = total_time / len(self._generation_times) if self._generation_times else 0

        # Create summary panel
        summary_text = []
        summary_text.append(f"🕒 Total Time: {total_time:.2f}s")
        summary_text.append(f"⚡ Avg Generation: {avg_generation_time:.3f}s")
        summary_text.append(f"📊 Cultural Events: {len(self._cultural_events)}")

        if final_metrics:
            summary_text.append("")
            summary_text.append("📈 Final Metrics:")
            for key, value in final_metrics.items():
                summary_text.append(f"  • {key.replace('_', ' ').title()}: {value:.4f}")

        summary_panel = Panel("\n".join(summary_text), title="🎯 Simulation Summary", style="green")
        self.console.print(summary_panel)

    @beartype
    def create_cultural_events_summary(self) -> None:
        """Create a summary of cultural learning events."""
        if not self._cultural_events:
            return

        # Analyze cultural events
        event_types = {}
        for event in self._cultural_events:
            event_type = event["event_type"]
            event_types[event_type] = event_types.get(event_type, 0) + 1

        # Create events table
        events_table = Table(title="📚 Cultural Learning Events", show_header=True)
        events_table.add_column("Event Type", style="cyan")
        events_table.add_column("Count", style="green", justify="right")
        events_table.add_column("Percentage", style="yellow", justify="right")

        total_events = len(self._cultural_events)
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_events) * 100
            events_table.add_row(event_type.replace("_", " ").title(), str(count), f"{percentage:.1f}%")

        self.console.print(events_table)

    def end_simulation(self) -> None:
        """End simulation monitoring and display final summary."""
        if self.progress:
            self.progress.stop()
            self.progress = None

        total_time = time.time() - self._start_time
        self.console.print(f"\n✅ Simulation completed in {total_time:.2f} seconds")

        ic("Simulation completed", f"Total time: {total_time:.2f}s")
        logger.info(f"Simulation completed in {total_time:.2f} seconds")

    @beartype
    def plot_generation_performance(self) -> None:
        """Plot generation timing performance using matplotlib."""
        if len(self._generation_times) < 2:
            self.console.print("❌ Not enough timing data to plot performance")
            return

        try:
            import matplotlib.pyplot as plt

            # Calculate generation intervals
            intervals = [
                self._generation_times[i] - self._generation_times[i - 1] for i in range(1, len(self._generation_times))
            ]

            # Create simple performance plot
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(intervals) + 1), intervals, "b-", alpha=0.7)
            plt.axhline(
                y=sum(intervals) / len(intervals),
                color="r",
                linestyle="--",
                label=f"Average: {sum(intervals) / len(intervals):.3f}s",
            )
            plt.xlabel("Generation")
            plt.ylabel("Time per Generation (s)")
            plt.title("Generation Performance Over Time")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            output_path = "layer2_performance.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            self.console.print(f"📊 Performance plot saved to {output_path}")

        except ImportError:
            self.console.print("❌ Matplotlib not available for performance plotting")

    def get_cultural_events_dataframe(self) -> pl.DataFrame:
        """
        Get cultural events as a Polars DataFrame for analysis.

        Returns
        -------
        pl.DataFrame
            DataFrame containing all logged cultural events
        """
        if not self._cultural_events:
            return pl.DataFrame()

        return pl.DataFrame(self._cultural_events)

    def __enter__(self) -> SimulationMonitor:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        if self.progress:
            self.progress.stop()
