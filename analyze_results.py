#!/usr/bin/env python3
"""
Analysis script for Delamain RL training logs.

This script provides tools to analyze training results, compare different runs,
and generate visualizations from the CSV logs and YAML configuration files.

Usage:
    python analyze_results.py [command] [options]

Commands:
    single      Analyze a single training run
    compare     Compare multiple training runs
    info        Show information about a training run

Examples:
    python analyze_results.py single --log Delamain_2_6_log_test.csv
    python analyze_results.py compare --logs run1.csv run2.csv run3.csv
    python analyze_results.py info --config Delamain_2_6_config.yaml
"""

import argparse
import os
import yaml
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

# Set matplotlib style for better looking plots
plt.style.use("seaborn-v0_8")
matplotlib.rcParams["figure.figsize"] = (12, 6)
matplotlib.rcParams["font.size"] = 12


class TrainingRun:
    """Class to represent a single training run with its logs and configuration."""

    def __init__(
        self,
        log_file: str,
        config_file: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize a TrainingRun with log and optional config files.

        Args:
            log_file: Path to CSV log file
            config_file: Path to YAML config file (optional)
            name: Custom run name (optional, auto-extracted from log filename if not provided)
        """
        self.log_file = log_file
        self.config_file = config_file
        self.log_data = {}
        self.config_data = {}

        # Extract run name from log file (override with custom name if provided)
        if name:
            self.name = name
        else:
            self.name = os.path.splitext(os.path.basename(log_file))[0].replace(
                "_log_test", ""
            )

        # Load data
        self._load_log_data()
        if config_file:
            self._load_config_data()

    def _load_log_data(self):
        """Load CSV log data into memory."""
        with open(self.log_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) > 1:  # Skip header rows with single entries
                    key = row[0]
                    values = [self._convert_value(v) for v in row[1:]]
                    self.log_data[key] = values

    def _load_config_data(self):
        """Load YAML configuration data."""
        with open(self.config_file, "r") as f:
            self.config_data = yaml.safe_load(f)

    def _convert_value(self, value: str):
        """Convert string values to appropriate Python types."""
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                # Check for boolean values
                if value.lower() in ("true", "false"):
                    return value.lower() == "true"
                # Check for None values
                if value.lower() in ("none", "null", ""):
                    return None
                return value

    def get_metric(self, metric_name: str) -> List:
        """Get a specific metric from the log data."""
        return self.log_data.get(metric_name, [])

    def get_config(self, section: str = None, key: str = None):
        """Get configuration data with optional section and key filtering."""
        if not self.config_data:
            return None

        data = self.config_data
        if section:
            data = data.get(section, {})
            if key:
                return data.get(key)
        return data

    def get_summary(self) -> Dict:
        """Get a summary of the training run."""
        summary = {
            "name": self.name,
            "episodes": len(self.get_metric("reward")),
            "max_reward": (
                max(self.get_metric("reward")) if self.get_metric("reward") else 0
            ),
            "avg_reward": (
                np.mean(self.get_metric("reward")) if self.get_metric("reward") else 0
            ),
            "total_timesteps": (
                sum(self.get_metric("length")) if self.get_metric("length") else 0
            ),
        }

        if self.config_data:
            summary.update(
                {
                    "model": self.config_data.get("model_info", {}).get(
                        "model_class", "Unknown"
                    ),
                    "algorithm": str(
                        self.config_data.get("training_metadata", {}).get(
                            "algorithm", "Unknown"
                        )
                    ),
                    "parameters": self.config_data.get("model_info", {}).get(
                        "total_parameters", "Unknown"
                    ),
                }
            )

        return summary

    def _save_summary_txt(self, output_dir: str = "analysis"):
        """Save a text summary file alongside plots, including median episode times."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.name}_summary.txt")

        summary = self.get_summary()
        length_data = self.get_metric("length")
        date_data = self.get_metric("date")
        time_data = self.get_metric("time")

        with open(path, "w") as f:
            f.write(f"Run: {summary['name']}\n")
            f.write(f"Episodes: {summary['episodes']}\n")
            f.write(f"Max Reward: {summary['max_reward']:.2f}\n")
            f.write(f"Average Reward: {summary['avg_reward']:.2f}\n")
            f.write(f"Total Timesteps: {summary['total_timesteps']}\n")

            if length_data:
                f.write(
                    f"Median Episode Length (timesteps): {np.median(length_data):.2f}\n"
                )

            if (
                date_data
                and time_data
                and len(date_data) == len(time_data)
                and len(time_data) > 1
            ):
                durations = []
                for i in range(1, len(time_data)):
                    prev_i = i - 1
                    while prev_i >= 0 and time_data[prev_i] == time_data[i]:
                        prev_i -= 1
                    if prev_i < 0:
                        continue
                    try:
                        dt_prev = datetime.strptime(
                            f"{date_data[prev_i]} {time_data[prev_i]}",
                            "%Y-%m-%d %H:%M:%S",
                        )
                        dt_curr = datetime.strptime(
                            f"{date_data[i]} {time_data[i]}", "%Y-%m-%d %H:%M:%S"
                        )
                        durations.append((dt_curr - dt_prev).total_seconds())
                    except (ValueError, AttributeError):
                        pass
                if durations:
                    f.write(
                        f"Median Wall Clock Time (seconds): {np.median(durations):.2f}\n"
                    )

            reasons = self.get_metric("termination_reasons")
            if reasons:
                from collections import Counter

                reason_counts = Counter(reasons)
                f.write("\nTermination Reasons:\n")
                for reason, count in reason_counts.items():
                    pct = (count / len(reasons)) * 100
                    f.write(f"  {reason}: {count} ({pct:.1f}%)\n")

        print(f"📝 Saved summary to {path}")


def plot_learning_curve(runs: List[TrainingRun], output_dir: str = "analysis"):
    """Plot learning curves (reward vs episodes) for one or more runs."""
    plt.figure(figsize=(12, 6))

    for run in runs:
        rewards = run.get_metric("reward")
        episodes = list(range(1, len(rewards) + 1))

        # Smooth the curve with moving average
        window_size = min(20, len(rewards) // 10)
        if window_size > 1:
            smoothed_rewards = np.convolve(
                rewards, np.ones(window_size) / window_size, mode="valid"
            )
            smoothed_episodes = episodes[window_size - 1 :]
        else:
            smoothed_rewards = rewards
            smoothed_episodes = episodes

        plt.plot(smoothed_episodes, smoothed_rewards, label=run.name, alpha=0.8)

        # Add scatter points for key milestones (skip episode 0)
        if len(rewards) >= 10:
            milestone_indices = [
                len(rewards) // 4,
                len(rewards) // 2,
                3 * len(rewards) // 4,
                -1,
            ]
            for idx in milestone_indices:
                plt.scatter(
                    episodes[idx], rewards[idx], s=60, edgecolors="black", linewidth=0.5
                )

        # Add scatter point for first track completion
        reasons = run.get_metric("termination_reasons")
        if reasons:
            for i, reason in enumerate(reasons):
                if reason == "success":
                    plt.scatter(
                        episodes[i],
                        rewards[i],
                        marker="*",
                        s=200,
                        color="green",
                        edgecolors="black",
                        linewidth=0.5,
                        zorder=5,
                        label=(
                            f"{run.name} pierwsze ukończenie"
                            if len(runs) == 1
                            else None
                        ),
                    )
                    break

    plt.title("Krzywa uczenia: nagroda w czasie", fontsize=16)
    plt.xlabel("Epizod", fontsize=14)
    plt.ylabel("Nagroda", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "learning_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"📊 Saved learning curve plot to {plot_path}")
    plt.close()


def plot_action_distribution(run: TrainingRun, output_dir: str = "analysis"):
    """Plot action distribution over time."""
    actions_in_row = run.get_metric("actions_in_row")

    if not actions_in_row:
        print("⚠️  No action distribution data available")
        return

    plt.figure(figsize=(12, 6))

    # Plot raw data
    episodes = list(range(1, len(actions_in_row) + 1))
    plt.plot(episodes, actions_in_row, alpha=0.6, label="Akcje pod rząd")

    # Add moving average
    window_size = min(50, len(actions_in_row) // 5)
    if window_size > 1:
        smoothed = np.convolve(
            actions_in_row, np.ones(window_size) / window_size, mode="valid"
        )
        smoothed_episodes = episodes[window_size - 1 :]
        plt.plot(
            smoothed_episodes,
            smoothed,
            "r-",
            linewidth=2,
            label=f"Średnia ruchoma ({window_size})",
        )

    plt.title(f"Akcje pod rząd w czasie: {run.name}", fontsize=16)
    plt.xlabel("Epizod", fontsize=14)
    plt.ylabel("Średnie akcje pod rząd", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{run.name}_action_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"🎯 Saved action distribution plot to {plot_path}")
    plt.close()


def _plot_single_metric(run: TrainingRun, metric: str, output_dir: str = "analysis"):
    """Plot a single training metric and save as its own PNG."""
    values = run.get_metric(metric)
    if not values:
        print(f"⚠️  No data available for metric: {metric}")
        return

    episodes = list(range(1, len(values) + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(episodes, values, alpha=0.7)
    match metric:
        case "length":
            plt.title(f"Długość epizodu: {run.name}", fontsize=16)
            plt.ylabel("Długość epizodu", fontsize=14)
        case "loss":
            plt.title(f"Strata: {run.name}", fontsize=16)
            plt.ylabel("Strata", fontsize=14)
        case "reward":
            plt.title(f"Nagroda: {run.name}", fontsize=16)
            plt.ylabel("Nagroda", fontsize=14)
        case "lr":
            plt.title(f"learning_rate: {run.name}", fontsize=16)
            plt.ylabel("learning_rate", fontsize=14)
        case _:
            plt.title(f"{metric.capitalize()}: {run.name}", fontsize=16)
            plt.ylabel(metric.capitalize(), fontsize=14)
    plt.xlabel("Epizod", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{run.name}_{metric}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"📈 Saved {metric} plot to {plot_path}")
    plt.close()


def plot_reward(run: TrainingRun, output_dir: str = "analysis"):
    """Plot reward metric."""
    _plot_single_metric(run, "reward", output_dir)


def plot_loss(run: TrainingRun, output_dir: str = "analysis"):
    """Plot loss metric."""
    _plot_single_metric(run, "loss", output_dir)


def plot_epsilon(run: TrainingRun, output_dir: str = "analysis"):
    """Plot epsilon metric."""
    _plot_single_metric(run, "epsilon", output_dir)


def plot_lr(run: TrainingRun, output_dir: str = "analysis"):
    """Plot learning rate metric."""
    _plot_single_metric(run, "lr", output_dir)


def plot_length(run: TrainingRun, output_dir: str = "analysis"):
    """Plot episode length metric."""
    _plot_single_metric(run, "length", output_dir)


def plot_training_metrics(run: TrainingRun, output_dir: str = "analysis"):
    """Plot multiple training metrics, one per PNG."""
    metrics_to_plot = ["reward", "loss", "epsilon", "lr", "length"]
    for metric in metrics_to_plot:
        if metric in run.log_data:
            _plot_single_metric(run, metric, output_dir)


def plot_termination_reasons(run: TrainingRun, output_dir: str = "analysis"):
    """Plot pie chart of termination reasons"""
    from collections import Counter

    reasons = run.get_metric("termination_reasons")
    if not reasons:
        print("⚠️  No termination reason data available")
        return

    # Count each reason type
    reason_counts = Counter(reasons)
    labels = list(reason_counts.keys())
    sizes = list(reason_counts.values())

    # Use consistent colors for each reason type
    colors = {
        "success": "#2ecc71",
        "off_track": "#e74c3c",
        "timeout": "#f39c12",
        "early": "#9b59b6",
        "unknown": "#95a5a6",
    }

    # Map labels to colors
    plot_colors = [colors.get(label, "#3498db") for label in labels]

    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=plot_colors)
    plt.title(f"Termination Reasons: {run.name}", fontsize=16)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{run.name}_termination_reasons.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"🏁 Saved termination reasons plot to {plot_path}")
    plt.close()


def plot_completion_rate(
    run: TrainingRun, output_dir: str = "analysis", window: int = 50
):
    """Plot success rate over time (rolling % of last n episodes)."""
    reasons = run.get_metric("termination_reasons")
    if not reasons:
        print("⚠️  No termination reason data available")
        return

    # Calculate rolling success rate over last `window` episodes
    success_rates = []
    for i in range(len(reasons)):
        start = max(0, i - window + 1)
        count = i - start + 1
        successes = sum(1 for j in range(start, i + 1) if reasons[j] == "success")
        success_rates.append((successes / count) * 100)

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(success_rates) + 1), success_rates, "g-", linewidth=2)
    plt.title(
        f"Ukończone trasy w czasie (ostatnie {window} epizodów): {run.name}",
        fontsize=16,
    )
    plt.xlabel("Epizod", fontsize=14)
    plt.ylabel(f"Wskaźnik ukończenia (%, ostatnie {window} epizodów)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"{run.name}_completion_rate.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"🎯 Saved completion rate plot to {plot_path}")
    plt.close()


def compare_runs(runs: List[TrainingRun], output_dir: str = "analysis"):
    """Generate comparison plots and tables for multiple runs."""
    if len(runs) < 2:
        print("⚠️  Need at least 2 runs to compare")
        return

    # Create comparison table
    summaries = [run.get_summary() for run in runs]

    print("\n📋 Comparison Summary:")
    print("-" * 80)
    print(
        f"{'Run':<15} {'Episodes':<10} {'Max Reward':<12} {'Avg Reward':<12} {'Total Steps':<12} {'Model':<15} {'Algorithm':<12}"
    )
    print("-" * 80)

    for i, (run, summary) in enumerate(zip(runs, summaries)):
        print(
            f"{run.name:<15} {summary['episodes']:<10} {summary['max_reward']:<12.2f} {summary['avg_reward']:<12.2f} {summary['total_timesteps']:<12} {summary.get('model', 'N/A'):<15} {summary.get('algorithm', 'N/A'):<12}"
        )

    print("-" * 80)

    # Plot comparison of learning curves
    plot_learning_curve(runs, output_dir)

    # Plot parameter efficiency if available
    params_available = all("parameters" in run.get_summary() for run in runs)
    if params_available:
        plt.figure(figsize=(10, 6))

        for run in runs:
            summary = run.get_summary()
            plt.scatter(summary["parameters"], summary["max_reward"], s=100)
            plt.text(
                summary["parameters"],
                summary["max_reward"],
                run.name,
                ha="right",
                va="bottom",
            )

        plt.title("Parameter Efficiency: Max Reward vs Model Size", fontsize=16)
        plt.xlabel("Total Parameters", fontsize=14)
        plt.ylabel("Max Reward Achieved", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(output_dir, "parameter_efficiency.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"🎯 Saved parameter efficiency plot to {plot_path}")
        plt.close()


def show_run_info(run: TrainingRun):
    """Display detailed information about a training run."""
    print(f"\n📊 Training Run: {run.name}")
    print("=" * 60)

    # Show summary statistics
    summary = run.get_summary()
    print(f"Episodes: {summary['episodes']}")
    print(f"Max Reward: {summary['max_reward']:.2f}")
    print(f"Average Reward: {summary['avg_reward']:.2f}")
    print(f"Total Timesteps: {summary['total_timesteps']}")

    if run.config_data:
        print("\n🔧 Configuration:")
        print(f"  Model: {run.config_data['model_info']['model_class']}")
        print(f"  Parameters: {run.config_data['model_info']['total_parameters']:,}")
        print(f"  Algorithm: {run.config_data['training_metadata']['algorithm']}")
        print(f"  Device: {run.config_data['training_metadata']['device']}")
        print(f"  Environment: {run.config_data['environment']['name']}")
        print(f"  Observation: {run.config_data['environment']['observation_type']}")

        print("\n🌍 Environment Wrappers:")
        for wrapper in run.config_data["environment"]["wrapper_chain"]:
            print(f"  - {wrapper}")

    # Add termination statistics if available
    reasons = run.get_metric("termination_reasons")
    if reasons:
        from collections import Counter

        reason_counts = Counter(reasons)
        print("\n🏁 Termination Statistics:")
        for reason, count in reason_counts.items():
            percentage = (count / len(reasons)) * 100
            print(f"  {reason}: {count} episodes ({percentage:.1f}%)")

    print("\n📈 Metrics Available:")
    for metric, values in run.log_data.items():
        print(f"  - {metric}: {len(values)} data points")


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(description="Analyze Delamain RL training results")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single run analysis
    single_parser = subparsers.add_parser(
        "single", help="Analyze a single training run"
    )
    single_parser.add_argument("--log", required=True, help="Path to CSV log file")
    single_parser.add_argument("--config", help="Path to YAML config file")
    single_parser.add_argument(
        "--output", default="analysis", help="Output directory for plots"
    )
    single_parser.add_argument(
        "--name", help="Custom run name (overrides auto-extraction)"
    )

    # Compare multiple runs
    compare_parser = subparsers.add_parser(
        "compare", help="Compare multiple training runs"
    )
    compare_parser.add_argument(
        "--logs", nargs="+", required=True, help="Paths to CSV log files"
    )
    compare_parser.add_argument(
        "--configs", nargs="+", help="Paths to YAML config files"
    )
    compare_parser.add_argument(
        "--output", default="analysis", help="Output directory for plots"
    )
    compare_parser.add_argument(
        "--names",
        nargs="*",
        help="Custom run names (positional match with --logs, extras ignored)",
    )

    # Show run information
    info_parser = subparsers.add_parser(
        "info", help="Show information about a training run"
    )
    info_parser.add_argument("--log", required=True, help="Path to CSV log file")
    info_parser.add_argument("--config", help="Path to YAML config file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Handle different commands
    if args.command == "single":
        print(f"🔍 Analyzing single run: {args.log}")

        run = TrainingRun(args.log, args.config, name=args.name)
        show_run_info(run)
        run._save_summary_txt(args.output)

        # Generate plots
        plot_learning_curve([run], args.output)
        plot_action_distribution(run, args.output)
        plot_training_metrics(run, args.output)

        # Add termination analysis if data available
        if run.get_metric("termination_reasons"):
            plot_termination_reasons(run, args.output)
            plot_completion_rate(run, args.output)

        print(f"\n✅ Analysis complete! Plots saved to {args.output}/")

    elif args.command == "compare":
        print(f"🔍 Comparing {len(args.logs)} runs")

        # Create TrainingRun objects with optional custom names
        configs = args.configs if args.configs else [None] * len(args.logs)
        names = args.names if args.names else [None] * len(args.logs)
        runs = []
        for i, (log, config) in enumerate(zip(args.logs, configs)):
            custom_name = names[i] if i < len(names) else None
            runs.append(TrainingRun(log, config, name=custom_name))

        # Save per-run summaries
        for run in runs:
            run._save_summary_txt(args.output)

        # Generate comparison
        compare_runs(runs, args.output)

        print(f"\n✅ Comparison complete! Plots saved to {args.output}/")

    elif args.command == "info":
        print(f"📋 Showing information for: {args.log}")

        run = TrainingRun(args.log, args.config)
        show_run_info(run)


if __name__ == "__main__":
    main()
