#!/usr/bin/env python
"""Example 06: torch.profiler instrumentation via TorchProfilerMeter.

``TorchProfilerMeter`` ships in ``registry.experimental.torch_compat``. Attach
it once; every build() captures CPU + CUDA op breakdowns into the envelope's
meta dict. Nested builds get per-level profiles automatically.

Requires the ``[torch]`` extra::

    pip install 'registry-pattern[torch]'
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from registry import FunctionalRegistry, attach_meter, build
from registry.experimental.torch_compat import TorchProfilerMeter


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
