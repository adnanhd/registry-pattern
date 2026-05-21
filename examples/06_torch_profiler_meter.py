#!/usr/bin/env python
"""Example 06: torch.profiler-backed meter as a downstream extension.

The library ships only stdlib-based meters (LifetimeMeter, CPUMeter, ...).
Domain-specific meters live in YOUR project. This example shows how to
extend FactoryMeter with torch.profiler to capture CUDA / op-level kernel
breakdowns alongside every build.

Each registered training-step build gets:

  - meta["torch_profile_self_cpu_time_total_us"]   sum of self-CPU time
  - meta["torch_profile_self_cuda_time_total_us"]  sum of self-CUDA time
  - meta["torch_profile_top_ops"]                  top-N op summary

Requires `torch>=2.0`. Optionally `[torch]` extra in pyproject.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from registry import FactoryMeter, FunctionalRegistry, attach_meter, build


# =============================================================================
# Downstream extension: torch.profiler-backed FactoryMeter
# =============================================================================


class TorchProfilerMeter(FactoryMeter):
    """Wraps each build() invocation in a `torch.profiler.profile` context.

    Stack-based so nested builds get per-level profiles. Writes summary
    statistics into the envelope's meta dict.
    """

    name: str = "torch_profiler"

    def __init__(self, top_n: int = 5, record_shapes: bool = False) -> None:
        self._top_n: int = top_n
        self._record_shapes: bool = record_shapes
        self._stack: list[Any] = []  # active profile() context managers

    def on_build_start(self, *, cfg, ctx, meta) -> None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        prof = profile(activities=activities, record_shapes=self._record_shapes)
        prof.__enter__()
        self._stack.append(prof)

    def on_built(self, *, target, result, meta, ctx) -> None:
        if not self._stack:
            return
        prof = self._stack.pop()
        prof.__exit__(None, None, None)

        events = prof.key_averages()
        meta["torch_profile_self_cpu_time_total_us"] = sum(e.self_cpu_time_total for e in events)
        meta["torch_profile_self_cuda_time_total_us"] = sum(
            getattr(e, "self_cuda_time_total", 0) for e in events
        )
        top = sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[: self._top_n]
        meta["torch_profile_top_ops"] = [
            {"op": e.key, "self_cpu_us": e.self_cpu_time_total} for e in top
        ]

    def on_error(self, *, cfg, exc, ctx, meta) -> None:
        if self._stack:
            prof = self._stack.pop()
            try:
                prof.__exit__(type(exc), exc, exc.__traceback__)
            except Exception:
                pass


# =============================================================================
# Demo: attach the meter, build a model, look at the profile in meta
# =============================================================================


class StepRegistry(FunctionalRegistry, repo="myapp.steps"):
    pass


@StepRegistry.register_artifact
def one_forward_pass(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """A trivial forward; the profiler captures all kernels it triggers."""
    return model(batch)


def main() -> None:
    attach_meter(TorchProfilerMeter(top_n=3, record_shapes=False))

    model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    batch = torch.randn(32, 128)

    cfg: dict[str, Any] = {
        "type": "one_forward_pass",
        "data": {"model": model, "batch": batch},
        "meta": {},
    }
    build(cfg, repo="myapp.steps")

    print("Captured profile entries in meta:")
    for k, v in cfg["meta"].items():
        if k.startswith("torch_profile_"):
            preview = v if isinstance(v, (int, float)) else str(v)[:80]
            print(f"  {k}: {preview}")


if __name__ == "__main__":
    main()
