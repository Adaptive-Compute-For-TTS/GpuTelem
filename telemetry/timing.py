"""Correct wall-time measurement for model runs, on CPU or CUDA."""

import statistics
import time
from functools import partial

import torch

from .toy_model import run_toy_model


def _time_once_cpu(fn):
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def _time_once_cuda(fn):
    # Drain work queued before us so it isn't billed to this run.
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    # Events are filled in by the GPU; wait until it has reached `end`.
    end.synchronize()
    return start.elapsed_time(end)  # already in milliseconds


def time_callable(fn, steps, warmups=3, repeats=10):
    """Time `fn()` and return a metrics dict ready for a run-record.

    `steps` is not passed to `fn`; it is only used to compute per-step time.
    """
    if steps < 1 or warmups < 0 or repeats < 1:
        raise ValueError("need steps >= 1, warmups >= 0, repeats >= 1")

    use_cuda = torch.cuda.is_available()
    time_once = _time_once_cuda if use_cuda else _time_once_cpu

    for _ in range(warmups):
        fn()

    samples_ms = [time_once(fn) for _ in range(repeats)]
    median_ms = statistics.median(samples_ms)

    return {
        "steps": steps,
        "median_ms": median_ms,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "per_step_ms": median_ms / steps,
        "device": "cuda" if use_cuda else "cpu",
        "timer": "cuda_event" if use_cuda else "perf_counter",
        "warmups": warmups,
        "repeats": repeats,
    }


if __name__ == "__main__":
    for steps in (8, 32):
        r = time_callable(partial(run_toy_model, steps), steps)
        print(
            f"steps={r['steps']:>3}  median={r['median_ms']:8.2f} ms  "
            f"[min {r['min_ms']:.2f}, max {r['max_ms']:.2f}]  "
            f"per_step={r['per_step_ms']:.3f} ms  ({r['device']}, {r['timer']})"
        )
