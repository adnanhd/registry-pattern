#!/usr/bin/env python
"""Comprehensive profiling of the registry-pattern build pipeline.

Covers six angles:

  1. Depth sweep -- wall-clock and pstats at depth 1, 5, 10, 25, 50
  2. Pipeline stage breakdown -- isolate is_build_cfg, derive_config_schema,
     validators.pydantic, factory.build at fixed depth 10
  3. Cold-vs-warm -- first call after import vs. Nth call (cache effects)
  4. Serialize round-trip -- factory.serialize cost vs. build cost
  5. Buildable[T] Pydantic validation -- the public type-guard surface
  6. tracemalloc allocators -- where memory is going

Run::

    PYTHONPATH=. python benchmarks/profile_pipeline.py
"""

from __future__ import annotations

import cProfile
import gc
import io
import pstats
import time
import tracemalloc
from typing import Any

from pydantic import BaseModel
from registry import (
    Buildable,
    BuildCfg,
    TypeRegistry,
    build,
    is_build_cfg,
    normalize_cfg,
    serialize,
)

# ---------------------------------------------------------------------------
# Common setup -- registries with up to 50 levels
# ---------------------------------------------------------------------------


MAX_DEPTH = 50


class _Node:
    def __init__(self, child: Any = None, label: str = "") -> None:
        self.child = child
        self.label = label


_REGISTRIES: list[type] = []

for i in range(MAX_DEPTH):

    class _R(TypeRegistry[Any], repo=f"prof.depth.l{i}"):
        pass

    _R.__name__ = f"_ProfDepthReg{i}"

    class _NodeI(_Node):
        pass

    _NodeI.__name__ = f"ProfNode{i}"
    _NodeI.__qualname__ = f"ProfNode{i}"
    _R.register_artifact(_NodeI)
    _REGISTRIES.append(_R)


def make_envelope(depth: int) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "type": f"ProfNode{depth - 1}",
        "repo": f"prof.depth.l{depth - 1}",
        "data": {"label": f"L{depth - 1}"},
        "meta": {},
    }
    for i in range(depth - 2, -1, -1):
        cfg = {
            "type": f"ProfNode{i}",
            "repo": f"prof.depth.l{i}",
            "data": {"label": f"L{i}", "child": cfg},
            "meta": {},
        }
    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def banner(text: str) -> None:
    print()
    print("#" * 72)
    print(f"# {text}")
    print("#" * 72)


def print_pstats(pr: cProfile.Profile, top: int = 12) -> None:
    for sort_key, header in (
        ("cumulative", "by cumulative time"),
        ("tottime", "by internal time"),
    ):
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
        ps.print_stats(top)
        print(f"-- {header} --")
        print(s.getvalue())


def time_ms(fn, iterations: int) -> tuple[float, float]:
    """Returns (best_ms_per_call, mean_ms_per_call)."""
    samples = []
    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return min(samples), sum(samples) / len(samples)


# ---------------------------------------------------------------------------
# Section 1: depth sweep
# ---------------------------------------------------------------------------


def section_depth_sweep() -> None:
    banner("SECTION 1 -- depth sweep (wall clock per build, 100 iterations)")
    print(f"{'depth':>6} {'best_ms':>10} {'mean_ms':>10}")
    print("-" * 30)
    for depth in (1, 5, 10, 25, 50):
        cfg = make_envelope(depth)
        # Warm up
        for _ in range(3):
            build(cfg)
        best, mean = time_ms(lambda c=cfg: build(c), iterations=100)
        print(f"{depth:>6} {best:>10.3f} {mean:>10.3f}")


# ---------------------------------------------------------------------------
# Section 2: pipeline stage breakdown at fixed depth
# ---------------------------------------------------------------------------


def section_stage_breakdown() -> None:
    banner("SECTION 2 -- pipeline stage breakdown at depth 10, 500 iterations each")
    depth = 10
    cfg = make_envelope(depth)
    # Warm up so first-call schema derivation does not pollute timings.
    for _ in range(10):
        build(cfg)

    stages = [
        ("is_build_cfg(envelope)", lambda: is_build_cfg(cfg)),
        ("is_build_cfg(plain dict)", lambda: is_build_cfg({"foo": "bar"})),
        ("normalize_cfg(envelope)", lambda: normalize_cfg(cfg)),
        ("build(envelope)", lambda: build(cfg)),
        ("serialize(built)", None),  # filled in below
    ]
    built = build(cfg)
    stages[-1] = ("serialize(built)", lambda b=built: serialize(b))

    print(f"{'stage':<30} {'best_us':>12} {'mean_us':>12}")
    print("-" * 58)
    for label, fn in stages:
        best, mean = time_ms(fn, iterations=500)
        # Convert ms to us for finer-grained reading on cheap operations.
        print(f"{label:<30} {best * 1000:>12.2f} {mean * 1000:>12.2f}")


# ---------------------------------------------------------------------------
# Section 3: cold vs. warm
# ---------------------------------------------------------------------------


def section_cold_vs_warm() -> None:
    banner("SECTION 3 -- cold-vs-warm: first call after import vs. amortized")
    # Use a fresh registry pair so the schema cache (if any) is cold.
    class _ColdReg(TypeRegistry[Any], repo="prof.cold.warm"):
        pass

    class _ColdNode(_Node):
        pass

    _ColdNode.__name__ = "ColdNode"
    _ColdReg.register_artifact(_ColdNode)

    cfg = {
        "type": "ColdNode",
        "repo": "prof.cold.warm",
        "data": {"label": "cold"},
        "meta": {},
    }

    # Force cold path -- this is the first-ever build for ColdNode.
    gc.collect()
    t0 = time.perf_counter()
    build(cfg)
    cold_ms = (time.perf_counter() - t0) * 1000.0

    # Now warm: same class, same envelope.
    warm_samples = []
    for _ in range(50):
        gc.collect()
        t0 = time.perf_counter()
        build(cfg)
        warm_samples.append((time.perf_counter() - t0) * 1000.0)
    warm_best = min(warm_samples)
    warm_mean = sum(warm_samples) / len(warm_samples)

    print(f"cold first call    : {cold_ms:>10.3f} ms")
    print(f"warm best of 50    : {warm_best:>10.3f} ms")
    print(f"warm mean of 50    : {warm_mean:>10.3f} ms")
    print(f"cold-to-warm ratio : {cold_ms / warm_mean:>10.2f}x")


# ---------------------------------------------------------------------------
# Section 4: cProfile of the build pipeline at depth 25
# ---------------------------------------------------------------------------


def section_build_profile() -> None:
    banner("SECTION 4 -- cProfile of build() at depth 25, 200 iterations")
    cfg = make_envelope(25)
    for _ in range(10):
        build(cfg)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(200):
        build(cfg)
    pr.disable()
    print_pstats(pr, top=15)


# ---------------------------------------------------------------------------
# Section 5: cProfile of serialize() round-trip
# ---------------------------------------------------------------------------


def section_serialize_profile() -> None:
    banner("SECTION 5 -- cProfile of serialize() of a depth-25 tree, 200 iterations")
    cfg = make_envelope(25)
    built = build(cfg)
    for _ in range(10):
        serialize(built)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(200):
        serialize(built)
    pr.disable()
    print_pstats(pr, top=15)


# ---------------------------------------------------------------------------
# Section 6: Buildable[T] Pydantic validation
# ---------------------------------------------------------------------------


def section_buildable_validation() -> None:
    banner("SECTION 6 -- Buildable[T] Pydantic validation, 500 iterations")

    class _BVReg(TypeRegistry[_Node], repo="prof.buildable"):
        pass

    class _BVNode(_Node):
        pass

    _BVNode.__name__ = "BVNode"
    _BVReg.register_artifact(_BVNode)

    class Container(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

        node: Buildable[_Node]

    cfg = {
        "type": "BVNode",
        "repo": "prof.buildable",
        "data": {"label": "x"},
        "meta": {},
    }

    # Warm up
    for _ in range(10):
        Container(node=cfg)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(500):
        Container(node=cfg)
    pr.disable()
    print_pstats(pr, top=12)


# ---------------------------------------------------------------------------
# Section 7: tracemalloc allocator hot spots
# ---------------------------------------------------------------------------


def section_tracemalloc() -> None:
    banner("SECTION 7 -- tracemalloc top allocators during 100 builds at depth 25")
    cfg = make_envelope(25)
    # Warm up
    for _ in range(5):
        build(cfg)
    gc.collect()

    tracemalloc.start(25)
    snap_before = tracemalloc.take_snapshot()
    for _ in range(100):
        build(cfg)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    diff = snap_after.compare_to(snap_before, "lineno")
    print(f"{'rank':>4}  {'size_kb':>10}  {'count':>8}  location")
    print("-" * 72)
    for i, stat in enumerate(diff[:15], 1):
        loc = str(stat.traceback)
        print(f"{i:>4}  {stat.size_diff / 1024:>10.1f}  {stat.count_diff:>8}  {loc}")


# ---------------------------------------------------------------------------


def main() -> None:
    section_depth_sweep()
    section_stage_breakdown()
    section_cold_vs_warm()
    section_build_profile()
    section_serialize_profile()
    section_buildable_validation()
    section_tracemalloc()


if __name__ == "__main__":
    main()
