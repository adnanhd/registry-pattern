"""Factory meters -- measure something at each pipeline stage, write to ``meta``.

A ``FactoryMeter`` is a *probe*: it reads pipeline state (cfg / target /
kwargs / result) at the moments it cares about and writes the measurement
into the envelope's ``meta`` dict. It does **not** ship anything externally.

For external delivery (syslog / HTTP / OpenTelemetry / etc.) see
:mod:`registry.reporters`.

Built-in meters all use a stack-based baseline so nested builds get
per-level measurements:

- :class:`LifetimeMeter`  -- wall-clock seconds
- :class:`CPUMeter`       -- user + system CPU seconds
- :class:`MemoryMeter`    -- max RSS + delta (KB on Linux)
- :class:`IOMeter`        -- disk read/write bytes via /proc/self/io
- :class:`NetworkMeter`   -- rx/tx bytes via /proc/self/net/dev
- :class:`HeapMeter`      -- Python heap via tracemalloc
- :class:`RecursionMeter` -- factory recursion depth
"""

from __future__ import annotations

import resource
import threading
import time
import tracemalloc
from typing import Any

__all__ = [
    "FactoryMeter",
    "LifetimeMeter",
    "CPUMeter",
    "MemoryMeter",
    "IOMeter",
    "NetworkMeter",
    "HeapMeter",
    "RecursionMeter",
    "attach_meter",
    "detach_meter",
    "meters",
    "emit_meter",
]


class FactoryMeter:
    """Base meter. Override the stage methods you care about; all default no-op.

    Stage methods receive the pipeline's ``meta`` dict; write your measurements
    there. The factory pipeline calls meters BEFORE reporters at every stage,
    so reporters always see the latest meta.
    """

    name: str = "factory_meter"

    def on_build_start(self, *, cfg: Any, ctx: dict[str, Any], meta: dict[str, Any]) -> None:
        """Pre-recursion / pre-validation. Good place to record baselines."""

    def on_validated(
        self,
        *,
        target: Any,
        kwargs: dict[str, Any],
        ctx: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        """After kwargs assembled + validated, before ``target(**kwargs)``."""

    def on_built(
        self,
        *,
        target: Any,
        result: Any,
        meta: dict[str, Any],
        ctx: dict[str, Any],
    ) -> None:
        """After invocation + post hooks. Compute deltas and write to ``meta``."""

    def on_error(
        self,
        *,
        cfg: Any,
        exc: BaseException,
        ctx: dict[str, Any],
        meta: dict[str, Any],
    ) -> None:
        """Build failed. Clean up any baseline state."""


# ---------------------------------------------------------------------------
# Module-level meter registry
# ---------------------------------------------------------------------------

_METERS: dict[str, FactoryMeter] = {}
_LOCK = threading.Lock()


def attach_meter(meter: FactoryMeter) -> FactoryMeter:
    """Register a meter. Replaces any prior meter with the same ``name``."""
    with _LOCK:
        _METERS[meter.name] = meter
    return meter


def detach_meter(name: str) -> FactoryMeter | None:
    """Remove and return the meter registered under ``name``."""
    with _LOCK:
        return _METERS.pop(name, None)


def meters() -> dict[str, FactoryMeter]:
    """Snapshot of currently attached meters."""
    with _LOCK:
        return dict(_METERS)


def emit_meter(method: str, /, **payload: Any) -> None:
    """Call ``method`` on every attached meter. Per-meter errors are swallowed."""
    for m in meters().values():
        fn = getattr(m, method, None)
        if fn is None:
            continue
        try:
            fn(**payload)
        except Exception:  # noqa: BLE001 - meters must never break builds
            pass


# ---------------------------------------------------------------------------
# Stack-based baseline helper (shared by all per-build delta meters)
# ---------------------------------------------------------------------------


class _StackedMeter(FactoryMeter):
    """Base for meters that take a baseline at on_build_start and emit a delta at on_built."""

    def __init__(self) -> None:
        self._stack: list[tuple[Any, ...]] = []

    def _sample(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def _write(self, meta: dict[str, Any], before: tuple[Any, ...], after: tuple[Any, ...]) -> None:
        raise NotImplementedError

    def on_build_start(self, *, cfg, ctx, meta) -> None:
        self._stack.append(self._sample())

    def on_built(self, *, target, result, meta, ctx) -> None:
        if not self._stack:
            return
        self._write(meta, self._stack.pop(), self._sample())

    def on_error(self, *, cfg, exc, ctx, meta) -> None:
        if self._stack:
            self._stack.pop()


# ---------------------------------------------------------------------------
# Concrete meters
# ---------------------------------------------------------------------------


class LifetimeMeter(_StackedMeter):
    """Wall-clock seconds the build took."""

    name: str = "lifetime"

    def _sample(self) -> tuple[float]:
        return (time.perf_counter(),)

    def _write(self, meta, before, after) -> None:
        meta["lifetime_seconds"] = after[0] - before[0]


class CPUMeter(_StackedMeter):
    """User + system CPU seconds (process-wide via ``resource.getrusage``)."""

    name: str = "cpu"

    def _sample(self) -> tuple[float, float]:
        r = resource.getrusage(resource.RUSAGE_SELF)
        return (r.ru_utime, r.ru_stime)

    def _write(self, meta, before, after) -> None:
        meta["cpu_user_seconds"] = after[0] - before[0]
        meta["cpu_system_seconds"] = after[1] - before[1]


class MemoryMeter(_StackedMeter):
    """Max RSS (KB on Linux) and delta vs. build start."""

    name: str = "memory"

    def _sample(self) -> tuple[int]:
        return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,)

    def _write(self, meta, before, after) -> None:
        meta["rss_max_kb"] = after[0]
        meta["rss_delta_kb"] = after[0] - before[0]


class IOMeter(_StackedMeter):
    """Disk IO bytes via ``/proc/self/io`` (Linux)."""

    name: str = "io"

    @staticmethod
    def _read_proc_io() -> dict[str, int]:
        try:
            with open("/proc/self/io") as f:
                out: dict[str, int] = {}
                for line in f:
                    k, _, v = line.partition(": ")
                    out[k.strip()] = int(v.strip())
                return out
        except OSError:
            return {}

    def _sample(self) -> tuple[int, int, int, int]:
        s = self._read_proc_io()
        return (s.get("read_bytes", 0), s.get("write_bytes", 0),
                s.get("rchar", 0), s.get("wchar", 0))

    def _write(self, meta, before, after) -> None:
        meta["io_read_bytes"] = after[0] - before[0]
        meta["io_write_bytes"] = after[1] - before[1]
        meta["io_rchar_bytes"] = after[2] - before[2]
        meta["io_wchar_bytes"] = after[3] - before[3]


class NetworkMeter(_StackedMeter):
    """Bytes rx/tx summed across all interfaces via ``/proc/self/net/dev``."""

    name: str = "network"

    @staticmethod
    def _read_proc_net_dev() -> tuple[int, int]:
        rx_total = 0
        tx_total = 0
        try:
            with open("/proc/self/net/dev") as f:
                lines = f.readlines()[2:]
            for line in lines:
                parts = line.split()
                if len(parts) < 10:
                    continue
                rx_total += int(parts[1])
                tx_total += int(parts[9])
        except (OSError, ValueError):
            pass
        return (rx_total, tx_total)

    def _sample(self) -> tuple[int, int]:
        return self._read_proc_net_dev()

    def _write(self, meta, before, after) -> None:
        meta["net_rx_bytes"] = after[0] - before[0]
        meta["net_tx_bytes"] = after[1] - before[1]


class HeapMeter(_StackedMeter):
    """Python heap via ``tracemalloc``. Starts tracemalloc if not already on."""

    name: str = "heap"

    def __init__(self) -> None:
        super().__init__()
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def _sample(self) -> tuple[int, int]:
        return tracemalloc.get_traced_memory()

    def _write(self, meta, before, after) -> None:
        meta["heap_current_bytes"] = after[0]
        meta["heap_delta_bytes"] = after[0] - before[0]
        meta["heap_peak_bytes"] = after[1]


class RecursionMeter(FactoryMeter):
    """Tracks how deep the factory recursion went per top-level build.

    Writes ``build_depth`` (depth of THIS envelope) and ``build_max_depth``
    (deepest level seen so far this top-level build) into meta.
    """

    name: str = "recursion"

    def __init__(self) -> None:
        self._depth: int = 0
        self._max_depth: int = 0

    def on_build_start(self, *, cfg, ctx, meta) -> None:
        self._depth += 1
        if self._depth > self._max_depth:
            self._max_depth = self._depth

    def on_built(self, *, target, result, meta, ctx) -> None:
        meta["build_depth"] = self._depth
        meta["build_max_depth"] = self._max_depth
        self._depth -= 1
        if self._depth == 0:
            self._max_depth = 0

    def on_error(self, *, cfg, exc, ctx, meta) -> None:
        if self._depth > 0:
            self._depth -= 1
        if self._depth == 0:
            self._max_depth = 0
