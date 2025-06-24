#!/usr/bin/env python3
"""
Replication Validation and Statistical Analysis

This module provides statistical validation for replicated experiments,
ensuring results are within plausible ranges and providing confidence
intervals and significance testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from beartype import beartype
from scipy import stats

__all__ = ["ReplicationValidator", "ReplicationResults", "ValidationConfig"]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ValidationConfig:
    """Configuration for replication validation."""

    min_replications: int = 3
    confidence_level: float = 0.95
    outlier_threshold: float = 3.0  # Standard deviations

    # Plausible ranges for key metrics
    trait_range: tuple[float, float] = (-5.0, 5.0)
    preference_range: tuple[float, float] = (-5.0, 5.0)
    correlation_range: tuple[float, float] = (-1.0, 1.0)
    population_size_range: tuple[int, int] = (10, 10000)

    # Success rate thresholds
    min_success_rate: float = 0.7
    warning_success_rate: float = 0.8


@dataclass(slots=True, frozen=False)
class ReplicationResults:
    """Statistical results from replication analysis."""

    experiment_name: str
    total_replications: int
    successful_replications: int
    failed_replications: int
    success_rate: float

    # Statistical measures
    means: dict[str, float]
    std_devs: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]

    # Validation flags
    is_valid: bool
    validation_warnings: list[str]
    validation_errors: list[str]

    # Outlier detection
    outliers: dict[str, list[int]]  # metric -> list of replication indices

    # Range validation
    range_violations: dict[str, list[tuple[int, float]]]  # metric -> [(replication_idx, value)]


class ReplicationValidator:
    """Validates replicated experiment results with statistical analysis."""

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    @beartype
    def validate_replications(self, results: list[Any], experiment_name: str) -> ReplicationResults:
        """
        Validate a set of replicated experiment results.

        Parameters
        ----------
        results : List[Any]
            List of experiment results from replications
        experiment_name : str
            Name of the experiment being validated

        Returns
        -------
        ReplicationResults
            Comprehensive validation results with statistics
        """
        logger.info(f"Validating {len(results)} replications for {experiment_name}")

        # Separate successful and failed results
        successful_results = []
        failed_results = []

        for i, result in enumerate(results):
            if self._is_successful_result(result):
                successful_results.append((i, result))
            else:
                failed_results.append((i, result))

        total_reps = len(results)
        success_count = len(successful_results)
        failure_count = len(failed_results)
        success_rate = success_count / total_reps if total_reps > 0 else 0.0

        logger.info(f"Success rate: {success_rate:.2%} ({success_count}/{total_reps})")

        # Extract metrics from successful results
        metrics = self._extract_metrics(successful_results)

        # Calculate statistics
        means = {}
        std_devs = {}
        confidence_intervals = {}
        outliers = {}
        range_violations = {}

        for metric_name, values in metrics.items():
            if len(values) > 0:
                means[metric_name] = np.mean(values)
                std_devs[metric_name] = np.std(values, ddof=1) if len(values) > 1 else 0.0

                # Confidence intervals
                if len(values) >= 2:
                    confidence_intervals[metric_name] = self._calculate_confidence_interval(
                        values, self.config.confidence_level
                    )
                else:
                    confidence_intervals[metric_name] = (means[metric_name], means[metric_name])

                # Outlier detection
                outliers[metric_name] = self._detect_outliers(values, self.config.outlier_threshold)

                # Range validation
                range_violations[metric_name] = self._check_range_violations(
                    metric_name, values, [idx for idx, _ in successful_results]
                )

        # Overall validation
        validation_warnings = []
        validation_errors = []

        # Check minimum replications
        if success_count < self.config.min_replications:
            validation_errors.append(
                f"Insufficient successful replications: {success_count} < {self.config.min_replications}"
            )

        # Check success rate
        if success_rate < self.config.min_success_rate:
            validation_errors.append(f"Success rate too low: {success_rate:.2%} < {self.config.min_success_rate:.2%}")
        elif success_rate < self.config.warning_success_rate:
            validation_warnings.append(
                f"Success rate concerning: {success_rate:.2%} < {self.config.warning_success_rate:.2%}"
            )

        # Check for excessive outliers
        for metric_name, outlier_indices in outliers.items():
            if len(outlier_indices) > len(successful_results) * 0.2:  # >20% outliers
                validation_warnings.append(
                    f"High outlier rate for {metric_name}: {len(outlier_indices)}/{len(successful_results)}"
                )

        # Check range violations
        for metric_name, violations in range_violations.items():
            if violations:
                validation_errors.append(f"Range violations for {metric_name}: {len(violations)} instances")

        is_valid = len(validation_errors) == 0

        return ReplicationResults(
            experiment_name=experiment_name,
            total_replications=total_reps,
            successful_replications=success_count,
            failed_replications=failure_count,
            success_rate=success_rate,
            means=means,
            std_devs=std_devs,
            confidence_intervals=confidence_intervals,
            is_valid=is_valid,
            validation_warnings=validation_warnings,
            validation_errors=validation_errors,
            outliers=outliers,
            range_violations=range_violations,
        )

    @beartype
    def _is_successful_result(self, result: Any) -> bool:
        """Check if a single result indicates successful experiment completion."""
        try:
            # Check if result has expected structure
            if hasattr(result, "metadata"):
                return getattr(result.metadata, "success", False)
            elif isinstance(result, dict):
                if "error" in result:
                    return False
                metadata = result.get("metadata", {})
                return metadata.get("success", True)  # Assume success if no metadata
            else:
                # If we can't determine, assume success
                return True
        except Exception as e:
            logger.warning(f"Could not determine result success status: {e}")
            return False

    @beartype
    def _extract_metrics(self, successful_results: list[tuple[int, Any]]) -> dict[str, list[float]]:
        """Extract numerical metrics from successful results."""
        metrics = {}

        for idx, result in successful_results:
            try:
                # Extract common metrics based on result structure
                if hasattr(result, "final_trait"):
                    self._add_metric(metrics, "final_trait", result.final_trait)
                if hasattr(result, "final_preference"):
                    self._add_metric(metrics, "final_preference", result.final_preference)
                if hasattr(result, "final_covariance"):
                    self._add_metric(metrics, "final_covariance", result.final_covariance)
                if hasattr(result, "genetic_correlation"):
                    self._add_metric(metrics, "genetic_correlation", result.genetic_correlation)

                # Extract from dictionary structure
                if isinstance(result, dict):
                    for key in [
                        "final_trait",
                        "final_preference",
                        "final_covariance",
                        "genetic_correlation",
                        "population_size",
                    ]:
                        if key in result:
                            self._add_metric(metrics, key, result[key])

                    # Check nested structures
                    if "common_params" in result:
                        params = result["common_params"]
                        if isinstance(params, dict) and "population_size" in params:
                            self._add_metric(metrics, "population_size", params["population_size"])

            except Exception as e:
                logger.warning(f"Could not extract metrics from result {idx}: {e}")
                continue

        return metrics

    @beartype
    def _add_metric(self, metrics: dict[str, list[float]], name: str, value: Any) -> None:
        """Add a metric value to the metrics dictionary."""
        try:
            float_value = float(value)
            if not np.isfinite(float_value):
                logger.warning(f"Non-finite value for {name}: {value}")
                return

            if name not in metrics:
                metrics[name] = []
            metrics[name].append(float_value)

        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert {name} value to float: {value} ({e})")

    @beartype
    def _calculate_confidence_interval(self, values: list[float], confidence_level: float) -> tuple[float, float]:
        """Calculate confidence interval for a set of values."""
        if len(values) < 2:
            mean_val = values[0] if values else 0.0
            return (mean_val, mean_val)

        mean_val = np.mean(values)
        sem = stats.sem(values)  # Standard error of the mean

        # Use t-distribution for small samples
        df = len(values) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df)

        margin_of_error = t_critical * sem

        return (mean_val - margin_of_error, mean_val + margin_of_error)

    @beartype
    def _detect_outliers(self, values: list[float], threshold: float) -> list[int]:
        """Detect outliers using z-score method."""
        if len(values) < 3:
            return []

        z_scores = np.abs(stats.zscore(values))
        outlier_indices = [i for i, z in enumerate(z_scores) if z > threshold]

        return outlier_indices

    @beartype
    def _check_range_violations(
        self, metric_name: str, values: list[float], replication_indices: list[int]
    ) -> list[tuple[int, float]]:
        """Check for values outside plausible ranges."""
        violations = []

        # Get appropriate range for this metric
        if "trait" in metric_name.lower():
            min_val, max_val = self.config.trait_range
        elif "preference" in metric_name.lower():
            min_val, max_val = self.config.preference_range
        elif "correlation" in metric_name.lower() or "covariance" in metric_name.lower():
            min_val, max_val = self.config.correlation_range
        elif "population" in metric_name.lower():
            min_val, max_val = self.config.population_size_range
        else:
            # No range checking for unknown metrics
            return violations

        for i, value in enumerate(values):
            if value < min_val or value > max_val:
                rep_idx = replication_indices[i] if i < len(replication_indices) else i
                violations.append((rep_idx, value))

        return violations

    @beartype
    def generate_report(self, validation_result: ReplicationResults) -> str:
        """Generate a human-readable validation report."""
        report = []
        report.append(f"=== Replication Validation Report: {validation_result.experiment_name} ===")
        report.append("")

        # Overall status
        status = "✅ VALID" if validation_result.is_valid else "❌ INVALID"
        report.append(f"Status: {status}")
        report.append(
            f"Success Rate: {validation_result.success_rate:.2%} ({validation_result.successful_replications}/{validation_result.total_replications})"
        )
        report.append("")

        # Errors and warnings
        if validation_result.validation_errors:
            report.append("🚫 ERRORS:")
            for error in validation_result.validation_errors:
                report.append(f"  - {error}")
            report.append("")

        if validation_result.validation_warnings:
            report.append("⚠️  WARNINGS:")
            for warning in validation_result.validation_warnings:
                report.append(f"  - {warning}")
            report.append("")

        # Statistical summary
        report.append("📊 STATISTICAL SUMMARY:")
        for metric_name in sorted(validation_result.means.keys()):
            mean_val = validation_result.means[metric_name]
            std_val = validation_result.std_devs[metric_name]
            ci_low, ci_high = validation_result.confidence_intervals[metric_name]

            report.append(f"  {metric_name}:")
            report.append(f"    Mean: {mean_val:.4f} ± {std_val:.4f}")
            report.append(f"    95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

            # Outliers
            outliers = validation_result.outliers.get(metric_name, [])
            if outliers:
                report.append(f"    Outliers: {len(outliers)} ({outliers})")

            # Range violations
            violations = validation_result.range_violations.get(metric_name, [])
            if violations:
                report.append(f"    Range violations: {len(violations)}")
                for rep_idx, value in violations[:3]:  # Show first 3
                    report.append(f"      Rep {rep_idx}: {value:.4f}")
                if len(violations) > 3:
                    report.append(f"      ... and {len(violations) - 3} more")

            report.append("")

        return "\n".join(report)
