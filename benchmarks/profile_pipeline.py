#!/usr/bin/env python
"""Comprehensive profiling of the registry-pattern build pipeline (v2).

Seven sections; wall-clock sections go through ``bench.BenchSuite`` so they
collect stats (best / median / p95 / stddev), can dump JSON, and can diff
against a saved baseline. cProfile sections stay as standalone helpers.

Run::

    PYTHONPATH=. python benchmarks/profile_pipeline.py            # full run
    PYTHONPATH=. python benchmarks/profile_pipeline.py --quiet
    PYTHONPATH=. python benchmarks/profile_pipeline.py --output baseline.json
    PYTHONPATH=. python benchmarks/profile_pipeline.py --baseline baseline.json --strict
    PYTHONPATH=. python benchmarks/profile_pipeline.py --iterations-scale 0.1   # smoke
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
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

# Local framework -- vendored copy lives in this directory.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from bench import BenchSuite, banner, finalize, parse_args  # noqa: E402

# ---------------------------------------------------------------------------
# Setup -- one TypeRegistry per depth level
# ---------------------------------------------------------------------------


MAX_DEPTH = 50


class _Node:
    def __init__(self, child: Any = None, label: str = "") -> None:
        self.child = child
        self.label = label


_REGISTRIES: list = []
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


def _print_pstats(pr: cProfile.Profile, top: int = 12) -> None:
    for sort_key, header in (
        ("cumulative", "by cumulative time"),
        ("tottime", "by internal time"),
    ):
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
        ps.print_stats(top)
        print(f"-- {header} --")
        print(s.getvalue())


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_depth_sweep(suite: BenchSuite) -> None:
    banner("SECTION 1 -- depth sweep (best/median/p95/stddev, 100 iter)")
    for depth in (1, 5, 10, 25, 50):
        cfg = make_envelope(depth)
        suite.measure(
            f"build depth={depth:<3}", lambda c=cfg: build(c), iterations=100, unit="ms"
        )


def section_stage_breakdown(suite: BenchSuite) -> None:
    banner("SECTION 2 -- pipeline stage breakdown at depth 10, 500 iter")
    depth = 10
    cfg = make_envelope(depth)
    suite.measure("is_build_cfg(envelope)", lambda: is_build_cfg(cfg), iterations=500)
    suite.measure(
        "is_build_cfg(plain dict)", lambda: is_build_cfg({"foo": "bar"}), iterations=500
    )
    suite.measure("normalize_cfg(envelope)", lambda: normalize_cfg(cfg), iterations=500)
    suite.measure("build(envelope)", lambda: build(cfg), iterations=500, unit="ms")
    built = build(cfg)
    suite.measure(
        "serialize(built)", lambda: serialize(built), iterations=500, unit="ms"
    )


def section_cold_vs_warm(suite: BenchSuite) -> None:
    banner("SECTION 3 -- cold-vs-warm")

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
    # Single sample for cold (warmup=0 measures the very first call).
    suite.measure(
        "cold first call", lambda: build(cfg), iterations=1, warmup=0, unit="ms"
    )
    suite.measure("warm 50 calls", lambda: build(cfg), iterations=50, unit="ms")


def section_build_profile() -> None:
    banner("SECTION 4 -- cProfile of build() at depth 25, 200 iter")
    cfg = make_envelope(25)
    for _ in range(10):
        build(cfg)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(200):
        build(cfg)
    pr.disable()
    _print_pstats(pr, top=15)


def section_serialize_profile() -> None:
    banner("SECTION 5 -- cProfile of serialize() depth 25, 200 iter")
    cfg = make_envelope(25)
    built = build(cfg)
    for _ in range(10):
        serialize(built)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(200):
        serialize(built)
    pr.disable()
    _print_pstats(pr, top=15)


def section_buildable_validation(suite: BenchSuite) -> None:
    banner("SECTION 6 -- Buildable[T] Pydantic validation, 500 iter")

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
    suite.measure(
        "Buildable[T] validation", lambda: Container(node=cfg), iterations=500
    )


def section_tracemalloc() -> None:
    banner("SECTION 7 -- tracemalloc top allocators during 100 builds at depth 25")
    cfg = make_envelope(25)
    for _ in range(5):
        build(cfg)
    import gc

    gc.collect()
    tracemalloc.start(25)
    snap_before = tracemalloc.take_snapshot()
    for _ in range(100):
        build(cfg)
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    diff = snap_after.compare_to(snap_before, "lineno")
    print(f"{'rank':>4}  {'size_kb':>10}  {'count':>8}  location")
    print("-" * 78)
    for i, stat in enumerate(diff[:15], 1):
        print(
            f"{i:>4}  {stat.size_diff / 1024:>10.1f}  {stat.count_diff:>8}  "
            f"{stat.traceback}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    suite = BenchSuite(name="registry-pattern.profile_pipeline", cli=args)

    section_depth_sweep(suite)
    section_stage_breakdown(suite)
    section_cold_vs_warm(suite)
    section_build_profile()
    section_serialize_profile()
    section_buildable_validation(suite)
    section_tracemalloc()

    # CI gates -- values picked from the post-cache numbers with a comfortable
    # margin. Bump if the floor moves.
    banner("ASSERT_WITHIN GATES")
    suite.assert_within("build depth=25 ", 5.0)  # ms -- floor ~0.5 ms
    suite.assert_within("build depth=50 ", 10.0)  # ms -- floor ~1.0 ms
    suite.assert_within("Buildable[T] validation", 500.0)  # us -- floor ~120 us

    return finalize(suite)


if __name__ == "__main__":
    sys.exit(main())
