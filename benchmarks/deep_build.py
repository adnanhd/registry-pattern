#!/usr/bin/env python
"""Benchmark: recursive ``build()`` at increasing nesting depth.

Constructs N distinct sub-registries and builds an N-deep envelope tree
where each level instantiates a class from its own registry. Reports
wall-clock build time for depths 1, 5, 10, 25, 50, 100.

Run::

    PYTHONPATH=. python benchmarks/deep_build.py
"""

from __future__ import annotations

import gc
import time
from typing import Any

from registry import TypeRegistry, build


# ---------------------------------------------------------------------------
# Setup -- one TypeRegistry per depth level
# ---------------------------------------------------------------------------


MAX_DEPTH = 100


class _Node:
    """Each level's class -- holds whatever its child built."""

    def __init__(self, child: Any = None, label: str = "") -> None:
        self.child = child
        self.label = label


def _make_registry(level: int) -> type:
    """Dynamically build TypeRegistry[Any] subclass for level N."""
    name = f"_DepthReg{level}"
    return type(
        name,
        (TypeRegistry,),                                # __class_getitem__ later
        {},
        repo=f"bench.depth.l{level}",
    )


# Pre-create MAX_DEPTH registries; each holds an artifact named "Node{i}".
_registries: list[type] = []
for i in range(MAX_DEPTH):
    # Subscript Generic to bind Any
    reg = TypeRegistry[Any].__class_getitem__(Any) if False else None  # placeholder
    # Simple approach: define via real subclass syntax (kwargs supported)
    class _R(TypeRegistry[Any], repo=f"bench.depth.l{i}"):
        pass
    _R.__name__ = f"_DepthReg{i}"

    # Per-level class, registered into this registry
    class _NodeI(_Node):
        pass
    _NodeI.__name__ = f"Node{i}"
    _NodeI.__qualname__ = f"Node{i}"

    _R.register_artifact(_NodeI)
    _registries.append(_R)


# ---------------------------------------------------------------------------
# Envelope builder + driver
# ---------------------------------------------------------------------------


def make_envelope(depth: int) -> dict[str, Any]:
    """Bottom-up: leaf is a Node{depth-1}, wrap upward to Node0."""
    if depth <= 0:
        raise ValueError("depth must be positive")
    cfg: dict[str, Any] = {
        "type": f"Node{depth - 1}",
        "repo": f"bench.depth.l{depth - 1}",
        "data": {"label": f"L{depth - 1}"},
        "meta": {},
    }
    for i in range(depth - 2, -1, -1):
        cfg = {
            "type": f"Node{i}",
            "repo": f"bench.depth.l{i}",
            "data": {"label": f"L{i}", "child": cfg},
            "meta": {},
        }
    return cfg


def time_build(depth: int, runs: int = 10) -> tuple[float, float]:
    """Returns (best_ms, mean_ms) over `runs` repetitions."""
    samples: list[float] = []
    for _ in range(runs):
        cfg = make_envelope(depth)
        gc.collect()
        t0 = time.perf_counter()
        build(cfg)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return min(samples), sum(samples) / len(samples)


def main() -> None:
    print(f"{'depth':>6} {'best_ms':>10} {'mean_ms':>10}")
    print("-" * 32)
    for depth in (1, 5, 10, 25, 50, 100):
        best, mean = time_build(depth)
        print(f"{depth:>6} {best:>10.3f} {mean:>10.3f}")


if __name__ == "__main__":
    main()
