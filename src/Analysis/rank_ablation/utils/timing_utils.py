"""Timing and reproducibility helpers for LoRA rank ablation pipeline."""

from __future__ import annotations

import random
import statistics
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # noqa: BLE001
        pass


def resolve_device():
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize_device(device) -> None:
    try:
        import torch

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
    except Exception:  # noqa: BLE001
        pass


def time_callable(fn: Callable[[], Any], device=None) -> Tuple[Any, float]:
    if device is not None:
        synchronize_device(device)
    start = time.perf_counter()
    result = fn()
    if device is not None:
        synchronize_device(device)
    elapsed = time.perf_counter() - start
    return result, float(elapsed)


def summarize_latencies(latencies: List[float]) -> Dict[str, float]:
    if not latencies:
        return {
            "avg_latency_sec": 0.0,
            "median_latency_sec": 0.0,
            "min_latency_sec": 0.0,
            "max_latency_sec": 0.0,
            "p95_latency_sec": 0.0,
        }

    sorted_latencies = sorted(latencies)
    idx = int(round(0.95 * (len(sorted_latencies) - 1)))
    return {
        "avg_latency_sec": float(statistics.mean(latencies)),
        "median_latency_sec": float(statistics.median(latencies)),
        "min_latency_sec": float(sorted_latencies[0]),
        "max_latency_sec": float(sorted_latencies[-1]),
        "p95_latency_sec": float(sorted_latencies[idx]),
    }
