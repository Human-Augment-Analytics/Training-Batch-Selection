"""
Metrics logging and efficiency computation.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import time


@dataclass
class ExperimentMetrics:
    """Comprehensive metrics for an experiment run."""
    # Training performance
    train_losses: List[float] = field(default_factory=list)
    eval_losses: List[float] = field(default_factory=list)
    eval_metrics: Dict[str, List[float]] = field(default_factory=dict)

    # Efficiency metrics
    total_wall_time: float = 0.0
    total_samples_seen: int = 0
    total_gradient_steps: int = 0
    selection_overhead_time: float = 0.0

    # Per-step details
    samples_per_step: List[int] = field(default_factory=list)
    wall_time_per_step: List[float] = field(default_factory=list)

    # Batch selection specific
    selection_ratio_actual: List[float] = field(default_factory=list)
    selected_sample_losses: List[float] = field(default_factory=list)

    # Metadata
    selector_name: str = ""
    model_name: str = ""
    dataset_name: str = ""
    config: Dict[str, Any] = field(default_factory=dict)


class MetricsLogger:
    """
    Logger for experiment metrics.

    Provides:
    - Real-time metric tracking
    - Comparison across selectors
    - Export to JSON for analysis
    """

    def __init__(self, experiment_name: str, output_dir: str = "./results"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.runs: Dict[str, ExperimentMetrics] = {}
        self.current_run: Optional[str] = None
        self._step_start_time: float = 0.0

    def start_run(
        self,
        run_name: str,
        selector_name: str,
        model_name: str,
        dataset_name: str,
        config: Dict[str, Any],
    ) -> None:
        """Start tracking a new run."""
        self.current_run = run_name
        self.runs[run_name] = ExperimentMetrics(
            selector_name=selector_name,
            model_name=model_name,
            dataset_name=dataset_name,
            config=config,
        )

    def log_step(
        self,
        loss: float,
        samples: int,
        wall_time: float,
        selection_time: float = 0.0,
        **kwargs,
    ) -> None:
        """Log metrics for a single training step."""
        if self.current_run is None:
            return

        metrics = self.runs[self.current_run]
        metrics.train_losses.append(loss)
        metrics.samples_per_step.append(samples)
        metrics.wall_time_per_step.append(wall_time)
        metrics.total_samples_seen += samples
        metrics.total_gradient_steps += 1
        metrics.selection_overhead_time += selection_time

        # Log any additional metrics
        for key, value in kwargs.items():
            if key not in metrics.eval_metrics:
                metrics.eval_metrics[key] = []
            metrics.eval_metrics[key].append(value)

    def log_eval(self, loss: float, **metrics) -> None:
        """Log evaluation metrics."""
        if self.current_run is None:
            return

        self.runs[self.current_run].eval_losses.append(loss)
        for key, value in metrics.items():
            if key not in self.runs[self.current_run].eval_metrics:
                self.runs[self.current_run].eval_metrics[key] = []
            self.runs[self.current_run].eval_metrics[key].append(value)

    def end_run(self, total_wall_time: float) -> None:
        """Finalize a run."""
        if self.current_run is None:
            return

        self.runs[self.current_run].total_wall_time = total_wall_time

    def get_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison of all runs."""
        comparison = {}
        for run_name, metrics in self.runs.items():
            comparison[run_name] = compute_efficiency_metrics(metrics)
        return comparison

    def save(self) -> str:
        """Save all metrics to JSON."""
        output_file = self.output_dir / f"{self.experiment_name}_{int(time.time())}.json"

        data = {
            "experiment_name": self.experiment_name,
            "runs": {
                name: asdict(metrics) for name, metrics in self.runs.items()
            },
            "comparison": self.get_comparison(),
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        return str(output_file)


def compute_efficiency_metrics(metrics: ExperimentMetrics) -> Dict[str, Any]:
    """
    Compute efficiency metrics for fair comparison.

    Key metrics:
    1. Samples-to-performance: Final loss per sample seen
    2. Time-to-performance: Final loss per wall-clock second
    3. Selection overhead ratio: Selection time / Total time
    4. Effective speedup: Samples saved - overhead cost
    """
    if not metrics.train_losses:
        return {}

    final_loss = metrics.train_losses[-1]
    avg_loss = sum(metrics.train_losses) / len(metrics.train_losses)

    # Samples efficiency
    samples_to_loss = metrics.total_samples_seen / max(1, len(metrics.train_losses))

    # Time efficiency
    time_to_loss = metrics.total_wall_time / max(1, len(metrics.train_losses))

    # Selection overhead
    selection_overhead_ratio = (
        metrics.selection_overhead_time / metrics.total_wall_time
        if metrics.total_wall_time > 0 else 0
    )

    # Average samples per step (vs full batch)
    avg_samples = (
        sum(metrics.samples_per_step) / len(metrics.samples_per_step)
        if metrics.samples_per_step else 0
    )

    return {
        "final_loss": final_loss,
        "avg_loss": avg_loss,
        "total_samples": metrics.total_samples_seen,
        "total_wall_time": metrics.total_wall_time,
        "samples_per_step_avg": avg_samples,
        "selection_overhead_ratio": selection_overhead_ratio,
        "selection_overhead_time": metrics.selection_overhead_time,
        "samples_to_loss_ratio": final_loss / max(1, metrics.total_samples_seen),
        "time_to_loss_ratio": final_loss / max(0.001, metrics.total_wall_time),
        "gradient_steps": metrics.total_gradient_steps,
    }


def compare_selectors(
    baseline_metrics: ExperimentMetrics,
    selector_metrics: ExperimentMetrics,
) -> Dict[str, float]:
    """
    Compare a batch selector against baseline (random/full batch).

    Returns relative improvements and overhead analysis.
    """
    baseline = compute_efficiency_metrics(baseline_metrics)
    selector = compute_efficiency_metrics(selector_metrics)

    # Sample efficiency improvement
    sample_reduction = 1 - (selector["total_samples"] / max(1, baseline["total_samples"]))

    # Time efficiency (accounting for overhead)
    time_speedup = baseline["total_wall_time"] / max(0.001, selector["total_wall_time"])

    # Loss comparison (lower is better)
    loss_improvement = (baseline["final_loss"] - selector["final_loss"]) / max(0.001, baseline["final_loss"])

    # Net benefit: speedup - overhead
    net_speedup = time_speedup * (1 - selector["selection_overhead_ratio"])

    return {
        "sample_reduction": sample_reduction,
        "time_speedup": time_speedup,
        "loss_improvement": loss_improvement,
        "net_speedup": net_speedup,
        "overhead_ratio": selector["selection_overhead_ratio"],
        "worth_it": net_speedup > 1.0 and loss_improvement >= -0.05,  # Speedup with <5% loss degradation
    }
