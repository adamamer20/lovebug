#!/usr/bin/env python3
"""
Analysis script for Phase 3 Latin Hypercube Sampling results.
Extracts parameter importance and creates visualizations for research paper.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


def parse_string_dict(s):
    """Parse string representation of dictionary with datetime objects."""
    try:
        # Clean up the string representation
        s_clean = s.strip()
        if s_clean.startswith('"') and s_clean.endswith('"'):
            s_clean = s_clean[1:-1]  # Remove outer quotes

        # Replace datetime objects with None for simplicity
        import re

        s_clean = re.sub(r"datetime\.datetime\([^)]+\)", "None", s_clean)

        # Use eval to parse (needed for complex dict structures)
        result = eval(s_clean)
        return result
    except Exception as e:
        print(f"Parse error: {e}")
        return None


def extract_key_metrics(result_dict):
    """Extract key parameters and outcome metrics from a single experiment result."""
    if not result_dict or "metadata" not in result_dict or "model_results" not in result_dict:
        return None

    metadata = result_dict["metadata"]
    model_results = result_dict["model_results"]

    # Basic experiment info
    data = {
        "experiment_id": metadata.get("experiment_id", ""),
        "experiment_type": metadata.get("experiment_type", ""),
        "success": metadata.get("success", False),
        "duration_seconds": metadata.get("duration_seconds", 0),
        "population_size": metadata.get("population_size", 0),
        "n_generations": metadata.get("n_generations", 0),
        "random_seed": metadata.get("random_seed", 0),
    }

    # Final outcomes
    data["final_population"] = model_results.get("final_population", 0)
    data["extinct"] = data["final_population"] == 0
    data["n_steps"] = model_results.get("n_steps", 0)

    # Layer configuration
    layer_config = model_results.get("layer_config", {})
    data["genetic_enabled"] = layer_config.get("genetic_enabled", False)
    data["cultural_enabled"] = layer_config.get("cultural_enabled", False)
    data["genetic_weight"] = layer_config.get("genetic_weight", 0.0)
    data["cultural_weight"] = layer_config.get("cultural_weight", 0.0)
    data["sigma_perception"] = layer_config.get("sigma_perception", 0.0)
    data["theta_detect"] = layer_config.get("theta_detect", 0.0)
    data["sigmoid_steepness"] = layer_config.get("sigmoid_steepness", 1.0)

    # Extract trajectory data for analysis
    trajectory = model_results.get("trajectory", [])
    if trajectory:
        # Initial state
        initial = trajectory[0]
        data["initial_population"] = initial.get("population_size", 0)
        data["initial_mean_display"] = initial.get("mean_gene_display", 0)
        data["initial_mean_preference"] = initial.get("mean_gene_preference", 0)
        data["initial_var_display"] = initial.get("var_gene_display", 0)
        data["initial_var_preference"] = initial.get("var_gene_preference", 0)

        # Final state (if population survived)
        if len(trajectory) > 1:
            final = trajectory[-1]
            data["final_mean_display"] = final.get("mean_gene_display", 0)
            data["final_mean_preference"] = final.get("mean_gene_preference", 0)
            data["final_var_display"] = final.get("var_gene_display", 0)
            data["final_var_preference"] = final.get("var_gene_preference", 0)

            # Calculate evolutionary change
            data["display_change"] = data["final_mean_display"] - data["initial_mean_display"]
            data["preference_change"] = data["final_mean_preference"] - data["initial_mean_preference"]
            data["display_var_change"] = data["final_var_display"] - data["initial_var_display"]
            data["preference_var_change"] = data["final_var_preference"] - data["initial_var_preference"]

            # Calculate trait-preference correlation if possible
            displays = [step.get("mean_gene_display", 0) for step in trajectory]
            preferences = [step.get("mean_gene_preference", 0) for step in trajectory]
            if len(displays) > 1 and len(preferences) > 1:
                corr, p_val = pearsonr(displays, preferences)
                data["trait_preference_correlation"] = corr
                data["correlation_p_value"] = p_val
            else:
                data["trait_preference_correlation"] = 0
                data["correlation_p_value"] = 1.0

            # Population persistence metrics
            pop_sizes = [step.get("population_size", 0) for step in trajectory]
            data["min_population"] = min(pop_sizes)
            data["max_population"] = max(pop_sizes)
            data["mean_population"] = np.mean(pop_sizes)
            data["population_variance"] = np.var(pop_sizes)

            # Evolutionary rate (change per generation)
            if data["n_steps"] > 0:
                data["display_rate"] = abs(data["display_change"]) / data["n_steps"]
                data["preference_rate"] = abs(data["preference_change"]) / data["n_steps"]
            else:
                data["display_rate"] = 0
                data["preference_rate"] = 0

            # Selection intensity metrics
            selection_diffs = [
                step.get("selection_differential", 0)
                for step in trajectory
                if step.get("selection_differential") is not None
            ]
            if selection_diffs:
                data["mean_selection_differential"] = np.mean(selection_diffs)
                data["selection_variance"] = np.var(selection_diffs)
            else:
                data["mean_selection_differential"] = 0
                data["selection_variance"] = 0
        else:
            # Population went extinct immediately
            for key in [
                "final_mean_display",
                "final_mean_preference",
                "final_var_display",
                "final_var_preference",
                "display_change",
                "preference_change",
                "display_var_change",
                "preference_var_change",
                "trait_preference_correlation",
                "correlation_p_value",
                "min_population",
                "max_population",
                "mean_population",
                "population_variance",
                "display_rate",
                "preference_rate",
                "mean_selection_differential",
                "selection_variance",
            ]:
                data[key] = 0

    return data


def load_and_parse_results(file_path):
    """Load and parse the experimental results file."""
    print(f"Loading results from {file_path}")

    with open(file_path) as f:
        lines = f.readlines()

    raw_data = []
    print("Parsing line by line...")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line in ["[", "]", ","]:
            continue

        # Remove trailing comma
        if line.endswith(","):
            line = line[:-1]

        # Parse the string representation
        parsed = parse_string_dict(line)
        if parsed:
            raw_data.append(parsed)
        else:
            print(f"Failed to parse line {i + 1}")

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} lines, {len(raw_data)} successful")

    print(f"Loaded {len(raw_data)} experiments from {len(lines)} lines")

    # Extract metrics from each experiment
    processed_data = []
    for i, result in enumerate(raw_data):
        metrics = extract_key_metrics(result)
        if metrics:
            processed_data.append(metrics)
        else:
            print(f"Failed to extract metrics from experiment {i}")

    print(f"Successfully processed {len(processed_data)} experiments")
    return pd.DataFrame(processed_data)


def analyze_parameter_importance(df, target_variables):
    """Analyze parameter importance using multiple methods."""
    print("\n=== PARAMETER IMPORTANCE ANALYSIS ===")

    # Select predictor variables (parameters)
    parameter_cols = [
        "population_size",
        "genetic_weight",
        "cultural_weight",
        "sigma_perception",
        "theta_detect",
        "sigmoid_steepness",
    ]

    # Filter to experiments that have all required parameters
    complete_data = df.dropna(subset=parameter_cols + target_variables)
    print(f"Analyzing {len(complete_data)} complete experiments")

    results = {}

    for target in target_variables:
        print(f"\nAnalyzing parameter importance for: {target}")

        if target not in complete_data.columns:
            print(f"Target variable {target} not found in data")
            continue

        # Prepare the data
        X = complete_data[parameter_cols]
        y = complete_data[target]

        # Remove any remaining NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < 10:
            print(f"Insufficient data for {target} ({len(X_clean)} samples)")
            continue

        print(f"Using {len(X_clean)} samples for {target}")

        # Correlation analysis
        correlations = {}
        for param in parameter_cols:
            if param in X_clean.columns:
                corr, p_val = pearsonr(X_clean[param], y_clean)
                correlations[param] = {"correlation": corr, "p_value": p_val}

        # Random Forest importance
        rf_importance = {}
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_clean, y_clean)

            for _i, param in enumerate(parameter_cols):
                if param in X_clean.columns:
                    rf_importance[param] = rf.feature_importances_[X_clean.columns.get_loc(param)]
        except Exception as e:
            print(f"Random Forest failed for {target}: {e}")

        results[target] = {"correlations": correlations, "rf_importance": rf_importance, "n_samples": len(X_clean)}

    return results


def create_visualizations(df, importance_results, output_dir):
    """Create comprehensive visualizations for the analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Set up plotting style
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    print("\n=== CREATING VISUALIZATIONS ===")
    print(f"Output directory: {output_dir}")

    # 1. Experiment success rate by type
    plt.figure(figsize=(10, 6))
    success_by_type = df.groupby("experiment_type")["success"].agg(["count", "sum", "mean"]).reset_index()
    success_by_type["failure_rate"] = 1 - success_by_type["mean"]

    plt.subplot(1, 2, 1)
    bars = plt.bar(success_by_type["experiment_type"], success_by_type["mean"], color=["#2E86AB", "#A23B72", "#F18F01"])
    plt.title("Experiment Success Rate by Type")
    plt.ylabel("Success Rate")
    plt.ylim(0, 1)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.2f}\n(n={success_by_type.iloc[i]['count']})",
            ha="center",
            va="bottom",
        )

    plt.subplot(1, 2, 2)
    extinction_by_type = df.groupby("experiment_type")["extinct"].mean()
    bars = plt.bar(extinction_by_type.index, extinction_by_type.values, color=["#2E86AB", "#A23B72", "#F18F01"])
    plt.title("Extinction Rate by Experiment Type")
    plt.ylabel("Extinction Rate")
    plt.ylim(0, 1)
    for _i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{height:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_dir / "experiment_success_rates.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Parameter importance heatmap
    if importance_results:
        # Create correlation heatmap
        target_vars = list(importance_results.keys())
        param_vars = [
            "population_size",
            "genetic_weight",
            "cultural_weight",
            "sigma_perception",
            "theta_detect",
            "sigmoid_steepness",
        ]

        corr_matrix = np.zeros((len(target_vars), len(param_vars)))
        for i, target in enumerate(target_vars):
            for j, param in enumerate(param_vars):
                if param in importance_results[target]["correlations"]:
                    corr_matrix[i, j] = importance_results[target]["correlations"][param]["correlation"]

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            corr_matrix,
            xticklabels=param_vars,
            yticklabels=target_vars,
            annot=True,
            cmap="RdBu_r",
            center=0,
            fmt=".3f",
            square=True,
        )
        plt.title("Parameter-Outcome Correlations")
        plt.xlabel("Parameters")
        plt.ylabel("Outcome Variables")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / "parameter_correlation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        # Random Forest importance plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for i, target in enumerate(target_vars[:4]):  # Plot top 4 targets
            if i >= len(axes):
                break

            rf_imp = importance_results[target]["rf_importance"]
            if rf_imp:
                params = list(rf_imp.keys())
                importances = list(rf_imp.values())

                # Sort by importance
                sorted_idx = np.argsort(importances)[::-1]
                params_sorted = [params[idx] for idx in sorted_idx]
                importances_sorted = [importances[idx] for idx in sorted_idx]

                axes[i].barh(params_sorted, importances_sorted)
                axes[i].set_title(f"Random Forest Importance: {target}")
                axes[i].set_xlabel("Feature Importance")

        # Remove empty subplots
        for i in range(len(target_vars), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.savefig(output_dir / "random_forest_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 3. Population dynamics
    non_extinct = df[~df["extinct"]]
    if len(non_extinct) > 0:
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.scatter(
            non_extinct["initial_population"],
            non_extinct["final_population"],
            alpha=0.6,
            c=non_extinct["experiment_type"].astype("category").cat.codes,
        )
        plt.plot(
            [0, non_extinct["initial_population"].max()], [0, non_extinct["initial_population"].max()], "r--", alpha=0.5
        )
        plt.xlabel("Initial Population")
        plt.ylabel("Final Population")
        plt.title("Population Change")

        plt.subplot(2, 3, 2)
        plt.hist(non_extinct["trait_preference_correlation"], bins=30, alpha=0.7, edgecolor="black")
        plt.xlabel("Trait-Preference Correlation")
        plt.ylabel("Frequency")
        plt.title("Distribution of Trait-Preference Correlations")

        plt.subplot(2, 3, 3)
        plt.scatter(
            non_extinct["display_change"],
            non_extinct["preference_change"],
            alpha=0.6,
            c=non_extinct["experiment_type"].astype("category").cat.codes,
        )
        plt.xlabel("Display Trait Change")
        plt.ylabel("Preference Change")
        plt.title("Evolutionary Change Patterns")
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)

        plt.subplot(2, 3, 4)
        plt.boxplot(
            [
                non_extinct[non_extinct["experiment_type"] == exp_type]["display_rate"].dropna()
                for exp_type in non_extinct["experiment_type"].unique()
            ],
            labels=non_extinct["experiment_type"].unique(),
        )
        plt.ylabel("Display Evolution Rate")
        plt.title("Evolution Rates by Experiment Type")

        plt.subplot(2, 3, 5)
        plt.boxplot(
            [
                non_extinct[non_extinct["experiment_type"] == exp_type]["mean_selection_differential"].dropna()
                for exp_type in non_extinct["experiment_type"].unique()
            ],
            labels=non_extinct["experiment_type"].unique(),
        )
        plt.ylabel("Mean Selection Differential")
        plt.title("Selection Intensity by Experiment Type")

        plt.subplot(2, 3, 6)
        plt.scatter(
            non_extinct["population_size"],
            non_extinct["trait_preference_correlation"],
            alpha=0.6,
            c=non_extinct["experiment_type"].astype("category").cat.codes,
        )
        plt.xlabel("Population Size")
        plt.ylabel("Trait-Preference Correlation")
        plt.title("Population Size vs Correlation")

        plt.tight_layout()
        plt.savefig(output_dir / "population_dynamics.png", dpi=300, bbox_inches="tight")
        plt.close()

    # 4. Parameter sensitivity analysis
    if len(df) > 0:
        param_cols = [
            "population_size",
            "genetic_weight",
            "cultural_weight",
            "sigma_perception",
            "theta_detect",
            "sigmoid_steepness",
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, param in enumerate(param_cols):
            if param in df.columns and i < len(axes):
                # Extinction rate vs parameter value
                param_values = df[param].dropna()
                if len(param_values.unique()) > 1:
                    # Bin the parameter values
                    n_bins = min(10, len(param_values.unique()))
                    df_param = df.dropna(subset=[param])
                    df_param["param_bin"] = pd.cut(df_param[param], bins=n_bins)

                    extinction_by_bin = df_param.groupby("param_bin")["extinct"].mean()
                    bin_centers = [interval.mid for interval in extinction_by_bin.index]

                    axes[i].plot(bin_centers, extinction_by_bin.values, "o-", linewidth=2, markersize=6)
                    axes[i].set_xlabel(param.replace("_", " ").title())
                    axes[i].set_ylabel("Extinction Rate")
                    axes[i].set_title(f"Extinction Rate vs {param.replace('_', ' ').title()}")
                    axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_sensitivity.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Visualizations saved to {output_dir}")


def generate_summary_report(df, importance_results, output_dir):
    """Generate a comprehensive summary report."""
    output_dir = Path(output_dir)

    report = []
    report.append("# LoveBug Phase 3 Parameter Space Analysis Report")
    report.append("=" * 60)
    report.append("")

    # Basic statistics
    report.append("## Experiment Overview")
    report.append(f"- Total experiments: {len(df)}")
    report.append(f"- Successful experiments: {df['success'].sum()} ({df['success'].mean():.1%})")
    report.append(f"- Extinct populations: {df['extinct'].sum()} ({df['extinct'].mean():.1%})")
    report.append("")

    # By experiment type
    report.append("## Results by Experiment Type")
    for exp_type in df["experiment_type"].unique():
        subset = df[df["experiment_type"] == exp_type]
        report.append(f"### {exp_type.title()} Experiments")
        report.append(f"- Count: {len(subset)}")
        report.append(f"- Success rate: {subset['success'].mean():.1%}")
        report.append(f"- Extinction rate: {subset['extinct'].mean():.1%}")

        if len(subset[~subset["extinct"]]) > 0:
            survivors = subset[~subset["extinct"]]
            report.append(
                f"- Mean trait-preference correlation: {survivors['trait_preference_correlation'].mean():.3f}"
            )
            report.append(f"- Mean population persistence: {survivors['mean_population'].mean():.0f}")
        report.append("")

    # Parameter importance findings
    if importance_results:
        report.append("## Key Parameter Importance Findings")
        report.append("")

        # Collect top correlations across all outcomes
        all_correlations = []
        for target, results in importance_results.items():
            for param, corr_data in results["correlations"].items():
                all_correlations.append(
                    {
                        "target": target,
                        "parameter": param,
                        "correlation": abs(corr_data["correlation"]),
                        "p_value": corr_data["p_value"],
                        "significant": corr_data["p_value"] < 0.05,
                    }
                )

        corr_df = pd.DataFrame(all_correlations)

        # Top significant correlations
        significant_corrs = corr_df[corr_df["significant"]].sort_values("correlation", ascending=False)

        report.append("### Strongest Parameter Effects (|correlation| > 0.3, p < 0.05)")
        for _, row in significant_corrs[significant_corrs["correlation"] > 0.3].head(10).iterrows():
            report.append(
                f"- **{row['parameter']}** → {row['target']}: r = {row['correlation']:.3f} (p = {row['p_value']:.4f})"
            )
        report.append("")

        # Parameter ranking by average importance
        param_importance = (
            corr_df.groupby("parameter")["correlation"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        )
        report.append("### Parameters Ranked by Average Importance")
        for param, stats in param_importance.iterrows():
            report.append(
                f"- **{param}**: Average |correlation| = {stats['mean']:.3f} (across {stats['count']} outcomes)"
            )
        report.append("")

    # Evolutionary dynamics insights
    non_extinct = df[~df["extinct"]]
    if len(non_extinct) > 0:
        report.append("## Evolutionary Dynamics Insights")
        report.append("")

        report.append("### Trait-Preference Evolution")
        pos_corr = (non_extinct["trait_preference_correlation"] > 0.1).sum()
        neg_corr = (non_extinct["trait_preference_correlation"] < -0.1).sum()
        report.append(
            f"- Positive trait-preference correlation: {pos_corr} experiments ({pos_corr / len(non_extinct):.1%})"
        )
        report.append(
            f"- Negative trait-preference correlation: {neg_corr} experiments ({neg_corr / len(non_extinct):.1%})"
        )
        report.append(
            f"- Mean correlation across all survivors: {non_extinct['trait_preference_correlation'].mean():.3f}"
        )
        report.append("")

        report.append("### Population Persistence")
        report.append(f"- Mean final population size: {non_extinct['final_population'].mean():.0f}")
        report.append(
            f"- Range: {non_extinct['final_population'].min():.0f} - {non_extinct['final_population'].max():.0f}"
        )
        report.append(
            f"- Populations with growth (final > initial): {(non_extinct['final_population'] > non_extinct['initial_population']).sum()}"
        )
        report.append("")

        report.append("### Evolution Rates")
        report.append(f"- Mean display trait evolution rate: {non_extinct['display_rate'].mean():.3f} units/generation")
        report.append(f"- Mean preference evolution rate: {non_extinct['preference_rate'].mean():.3f} units/generation")
        report.append("")

    # Critical insights for paper
    report.append("## Critical Insights for Research Paper")
    report.append("")

    # Population size effects
    if "population_size" in df.columns:
        large_pops = df[df["population_size"] > df["population_size"].median()]
        small_pops = df[df["population_size"] <= df["population_size"].median()]

        large_extinct_rate = large_pops["extinct"].mean()
        small_extinct_rate = small_pops["extinct"].mean()

        report.append("### Population Size Effects")
        report.append(
            f"- Large populations (N > {df['population_size'].median():.0f}): {large_extinct_rate:.1%} extinction rate"
        )
        report.append(
            f"- Small populations (N ≤ {df['population_size'].median():.0f}): {small_extinct_rate:.1%} extinction rate"
        )

        if abs(large_extinct_rate - small_extinct_rate) > 0.1:
            better = "larger" if large_extinct_rate < small_extinct_rate else "smaller"
            report.append(f"- **Finding**: {better.title()} populations show significantly better survival")
        report.append("")

    # Cultural vs genetic effects
    genetic_only = df[df["genetic_enabled"] & ~df["cultural_enabled"]]
    cultural_only = df[~df["genetic_enabled"] & df["cultural_enabled"]]
    combined = df[df["genetic_enabled"] & df["cultural_enabled"]]

    if len(genetic_only) > 0 and len(cultural_only) > 0:
        report.append("### Genetic vs Cultural Evolution")
        report.append(
            f"- Genetic-only experiments: {len(genetic_only)} ({genetic_only['extinct'].mean():.1%} extinction)"
        )
        report.append(
            f"- Cultural-only experiments: {len(cultural_only)} ({cultural_only['extinct'].mean():.1%} extinction)"
        )
        if len(combined) > 0:
            report.append(f"- Combined experiments: {len(combined)} ({combined['extinct'].mean():.1%} extinction)")
        report.append("")

        # Compare trait-preference correlations
        g_surv = genetic_only[~genetic_only["extinct"]]
        c_surv = cultural_only[~cultural_only["extinct"]]

        if len(g_surv) > 0 and len(c_surv) > 0:
            report.append("### Trait-Preference Coevolution Comparison")
            report.append(f"- Genetic-only mean correlation: {g_surv['trait_preference_correlation'].mean():.3f}")
            report.append(f"- Cultural-only mean correlation: {c_surv['trait_preference_correlation'].mean():.3f}")

            if len(combined[~combined["extinct"]]) > 0:
                comb_surv = combined[~combined["extinct"]]
                report.append(f"- Combined mean correlation: {comb_surv['trait_preference_correlation'].mean():.3f}")
        report.append("")

    # Save report
    with open(output_dir / "analysis_summary.md", "w") as f:
        f.write("\n".join(report))

    print(f"Summary report saved to {output_dir / 'analysis_summary.md'}")


def main():
    # File paths
    results_file = (
        "/home/adam/projects/lovebug/experiments/results/paper_data/session_20250624_213015/detailed_results.json"
    )
    output_dir = "/home/adam/projects/lovebug/experiments/results/lhs_analysis"

    # Load and parse results
    df = load_and_parse_results(results_file)

    if df.empty:
        print("No data loaded successfully!")
        return

    print(f"\nLoaded {len(df)} experiments")
    print(f"Columns: {list(df.columns)}")
    print(f"Experiment types: {df['experiment_type'].value_counts()}")

    # Define key outcome variables for analysis
    target_variables = [
        "extinct",
        "final_population",
        "trait_preference_correlation",
        "display_rate",
        "preference_rate",
        "mean_selection_differential",
    ]

    # Filter target variables to those that exist in the data
    available_targets = [var for var in target_variables if var in df.columns]
    print(f"Analyzing these outcome variables: {available_targets}")

    # Analyze parameter importance
    importance_results = analyze_parameter_importance(df, available_targets)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create visualizations
    create_visualizations(df, importance_results, output_dir)

    # Generate summary report
    generate_summary_report(df, importance_results, output_dir)

    # Save processed data
    df.to_csv(Path(output_dir) / "processed_results.csv", index=False)
    print(f"Processed data saved to {output_dir}/processed_results.csv")

    print("\nAnalysis complete!")
    print(f"Results available in: {output_dir}")


if __name__ == "__main__":
    main()
