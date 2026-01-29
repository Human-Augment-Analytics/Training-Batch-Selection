"""
Timing utilities for measuring overhead.

Critical for fair comparison of batch selection methods.
Following "No Train No Gain" methodology for proper overhead accounting.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class TimingStats:
    """Statistics for a timed operation."""
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0

    @property
    def avg_time(self) -> float:
        return self.total_time / max(1, self.call_count)

    def update(self, duration: float) -> None:
        self.total_time += duration
        self.call_count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)


class Timer:
    """
    Context manager for timing operations.

    Usage:
        timer = Timer()
        with timer("forward_pass"):
            outputs = model(inputs)
        with timer("selection"):
            selected = selector.select(batch)

        print(timer.summary())
    """

    def __init__(self, use_cuda_sync: bool = True):
        self.use_cuda_sync = use_cuda_sync
        self.stats: Dict[str, TimingStats] = {}
        self._current_name: Optional[str] = None
        self._current_start: float = 0.0

    @contextmanager
    def __call__(self, name: str):
        """Time a named operation."""
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            yield
        finally:
            if self.use_cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()

            duration = time.perf_counter() - start

            if name not in self.stats:
                self.stats[name] = TimingStats()
            self.stats[name].update(duration)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get timing summary."""
        return {
            name: {
                "total": stats.total_time,
                "count": stats.call_count,
                "avg": stats.avg_time,
                "min": stats.min_time if stats.min_time != float('inf') else 0,
                "max": stats.max_time,
            }
            for name, stats in self.stats.items()
        }

    def reset(self) -> None:
        """Reset all timings."""
        self.stats.clear()


class OverheadTracker:
    """
    Track overhead of batch selection relative to training.

    Key metric: selection_overhead / total_training_time

    A good batch selection method should have:
    - Low overhead (<5% of training time)
    - Speedup from reduced samples > overhead cost
    """

    def __init__(self):
        self.timer = Timer()
        self.epoch_stats: List[Dict[str, float]] = []

    @contextmanager
    def track_selection(self):
        """Track batch selection time."""
        with self.timer("selection"):
            yield

    @contextmanager
    def track_forward(self):
        """Track forward pass time."""
        with self.timer("forward"):
            yield

    @contextmanager
    def track_backward(self):
        """Track backward pass time."""
        with self.timer("backward"):
            yield

    @contextmanager
    def track_step(self):
        """Track full training step."""
        with self.timer("step"):
            yield

    def end_epoch(self) -> None:
        """Record epoch statistics and reset."""
        stats = self.timer.summary()

        selection_time = stats.get("selection", {}).get("total", 0)
        step_time = stats.get("step", {}).get("total", 0)
        forward_time = stats.get("forward", {}).get("total", 0)
        backward_time = stats.get("backward", {}).get("total", 0)

        epoch_stat = {
            "selection_time": selection_time,
            "step_time": step_time,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "selection_overhead_ratio": selection_time / step_time if step_time > 0 else 0,
            "compute_time": forward_time + backward_time,
        }
        self.epoch_stats.append(epoch_stat)
        self.timer.reset()

    def get_summary(self) -> Dict[str, float]:
        """Get overall summary across all epochs."""
        if not self.epoch_stats:
            return {}

        total_selection = sum(e["selection_time"] for e in self.epoch_stats)
        total_step = sum(e["step_time"] for e in self.epoch_stats)
        total_compute = sum(e["compute_time"] for e in self.epoch_stats)

        return {
            "total_selection_time": total_selection,
            "total_step_time": total_step,
            "total_compute_time": total_compute,
            "overall_selection_overhead": total_selection / total_step if total_step > 0 else 0,
            "selection_vs_compute": total_selection / total_compute if total_compute > 0 else 0,
            "num_epochs": len(self.epoch_stats),
        }


def benchmark_selection_overhead(
    selector,
    model,
    dataloader,
    n_batches: int = 100,
) -> Dict[str, float]:
    """
    Benchmark the overhead of a batch selector.

    Returns timing statistics for selection vs baseline.
    """
    timer = Timer()

    selection_times = []
    forward_times = []

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        # Time selection
        with timer("selection"):
            _ = selector.select(model, batch)

        # Time forward pass for comparison
        with timer("forward"):
            if "input_ids" in batch:
                with torch.no_grad():
                    _ = model(input_ids=batch["input_ids"])
            else:
                with torch.no_grad():
                    _ = model(batch["inputs"])

    summary = timer.summary()

    return {
        "avg_selection_time_ms": summary["selection"]["avg"] * 1000,
        "avg_forward_time_ms": summary["forward"]["avg"] * 1000,
        "selection_to_forward_ratio": summary["selection"]["avg"] / summary["forward"]["avg"] if summary["forward"]["avg"] > 0 else 0,
        "total_selection_time": summary["selection"]["total"],
        "total_forward_time": summary["forward"]["total"],
    }
