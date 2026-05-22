#!/usr/bin/env python
"""cProfile the ``build()`` pipeline at depth 25.

Complements ``deep_build.py`` (which reports wall-clock per depth) by
attributing the time inside ``build()`` to specific functions. Useful for
spotting hot spots that don't show up in coarse-grained timings.

Run::

    PYTHONPATH=. python benchmarks/profile_build.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
from typing import Any

from registry import TypeRegistry, build

DEPTH = 25
ITERATIONS = 200


class _Node:
    def __init__(self, child: Any = None, label: str = "") -> None:
        self.child = child
        self.label = label


for i in range(DEPTH):

    class _R(TypeRegistry[Any], repo=f"prof.depth.l{i}"):
        pass

    _R.__name__ = f"_ProfDepthReg{i}"

    class _NodeI(_Node):
        pass

    _NodeI.__name__ = f"ProfNode{i}"
    _NodeI.__qualname__ = f"ProfNode{i}"
    _R.register_artifact(_NodeI)


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


def main() -> None:
    cfg = make_envelope(DEPTH)

    # Warm up so first-call schema derivation does not dominate.
    for _ in range(10):
        build(cfg)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(ITERATIONS):
        build(cfg)
    pr.disable()

    for sort_key, header in (
        ("cumulative", "TOP 25 BY CUMULATIVE TIME"),
        ("tottime", "TOP 25 BY INTERNAL TIME"),
    ):
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(sort_key)
        ps.print_stats(25)
        print("=" * 70)
        print(header)
        print("=" * 70)
        print(s.getvalue())


if __name__ == "__main__":
    main()
