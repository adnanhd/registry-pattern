"""Factory observers — pluggable diagnostic sinks for the build pipeline.

Pattern adapted from ``effect-by-residual.observers``: subclass
:class:`FactoryObserver`, override the pipeline-stage methods you care
about, and ``attach(observer)`` it. Observers are keyed by ``name`` so
re-attaching the same name replaces it.

Two concrete sinks ship out of the box:

- :class:`JournalObserver`: writes one ``syslog`` line per stage.
  journald picks these up automatically; read with
  ``journalctl -t <ident>``.
- :class:`HTTPDashboardObserver`: serves the last N stage events as
  JSON on ``localhost:<port>``. ``curl localhost:8765 | jq`` to peek.

Build your own by subclassing and registering it; no other API.
"""

from __future__ import annotations

import http.server
import json
import resource
import syslog
import threading
import time
import tracemalloc
from typing import Any, Callable

__all__ = [
    "FactoryObserver",
    "JournalObserver",
    "HTTPDashboardObserver",
    "LifetimeObserver",
    "CPUObserver",
    "MemoryObserver",
    "IOObserver",
    "NetworkObserver",
    "HeapObserver",
    "RecursionObserver",
    "attach",
    "detach",
    "observers",
    "emit",
]


class FactoryObserver:
    """Base observer. Override the stage methods you care about; defaults no-op."""

    name: str = "factory_observer"

    def on_build_start(self, *, cfg: Any, ctx: dict[str, Any]) -> None:
        """Fires before any recursion/validation, with the raw envelope."""

    def on_validated(
        self,
        *,
        target: type | Callable[..., Any],
        kwargs: dict[str, Any],
        ctx: dict[str, Any],
    ) -> None:
        """Fires after kwargs assembled and validated, before ``target(**kwargs)``."""

    def on_built(
        self,
        *,
        target: type | Callable[..., Any],
        result: Any,
        meta: dict[str, Any],
        ctx: dict[str, Any],
    ) -> None:
        """Fires after ``target`` invoked and all post hooks ran."""

    def on_error(
        self,
        *,
        cfg: Any,
        exc: BaseException,
        ctx: dict[str, Any],
    ) -> None:
        """Fires when any step of the build raises."""


# ---------------------------------------------------------------------------
# Module-level registry of attached observers
# ---------------------------------------------------------------------------

_OBSERVERS: dict[str, FactoryObserver] = {}
_LOCK = threading.Lock()


def attach(observer: FactoryObserver) -> FactoryObserver:
    """Register an observer under its ``name``. Replaces any prior with same name."""
    with _LOCK:
        _OBSERVERS[observer.name] = observer
    return observer


def detach(name: str) -> FactoryObserver | None:
    """Remove and return the observer registered under ``name`` (or ``None``)."""
    with _LOCK:
        return _OBSERVERS.pop(name, None)


def observers() -> dict[str, FactoryObserver]:
    """Snapshot of currently attached observers."""
    with _LOCK:
        return dict(_OBSERVERS)


def emit(method: str, /, **payload: Any) -> None:
    """Call ``method`` on every attached observer. Swallow per-observer errors."""
    for obs in observers().values():
        fn = getattr(obs, method, None)
        if fn is None:
            continue
        try:
            fn(**payload)
        except Exception:  # noqa: BLE001 - observers must never break builds
            pass


# ---------------------------------------------------------------------------
# Sink: systemd journal (via syslog — journald captures by default)
# ---------------------------------------------------------------------------


class JournalObserver(FactoryObserver):
    """Logs each pipeline stage to syslog. journald picks up via ``-t <ident>``.

    Usage::

        attach(JournalObserver(ident="my-trainer"))
        # later: journalctl -t my-trainer
    """

    name: str = "journal"

    def __init__(self, ident: str = "registry") -> None:
        syslog.openlog(ident, syslog.LOG_PID, syslog.LOG_DAEMON)

    @staticmethod
    def _target_name(target: type | Callable[..., Any]) -> str:
        return getattr(target, "__name__", str(target))

    def on_build_start(self, *, cfg: Any, ctx: dict[str, Any]) -> None:
        t = cfg.type if hasattr(cfg, "type") else (cfg.get("type") if isinstance(cfg, dict) else "?")
        syslog.syslog(syslog.LOG_DEBUG, f"build.start type={t}")

    def on_validated(self, *, target, kwargs, ctx) -> None:
        syslog.syslog(
            syslog.LOG_INFO,
            f"build.validated target={self._target_name(target)} "
            f"kwargs={list(kwargs)}",
        )

    def on_built(self, *, target, result, meta, ctx) -> None:
        syslog.syslog(
            syslog.LOG_INFO,
            f"build.done target={self._target_name(target)} meta={meta}",
        )

    def on_error(self, *, cfg, exc, ctx) -> None:
        t = cfg.type if hasattr(cfg, "type") else (cfg.get("type") if isinstance(cfg, dict) else "?")
        syslog.syslog(
            syslog.LOG_ERR,
            f"build.error type={t} error_type={type(exc).__name__} error={exc}",
        )


# ---------------------------------------------------------------------------
# Sink: HTTP dashboard on localhost:PORT
# ---------------------------------------------------------------------------


class HTTPDashboardObserver(FactoryObserver):
    """Serves the last N pipeline events as JSON on ``localhost:<port>``.

    A simple read-only ring buffer; daemon thread, no shutdown semantics.
    Use::

        attach(HTTPDashboardObserver(port=8765, max_events=500))
        # then: curl localhost:8765
    """

    name: str = "http_dashboard"

    def __init__(self, port: int = 8765, max_events: int = 200, host: str = "127.0.0.1") -> None:
        self._events: list[dict[str, Any]] = []
        self._max_events: int = max_events
        self._lock: threading.Lock = threading.Lock()
        self._server: http.server.HTTPServer = self._start_server(host, port)
        self.url: str = f"http://{host}:{port}"

    def _start_server(self, host: str, port: int) -> http.server.HTTPServer:
        outer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                with outer._lock:
                    payload = json.dumps(outer._events, default=str, indent=2)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload.encode())

            def log_message(self, *a: Any, **kw: Any) -> None:  # silence
                pass

        server = http.server.HTTPServer((host, port), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def _push(self, **event: Any) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events.pop(0)

    @staticmethod
    def _target_name(target: type | Callable[..., Any]) -> str:
        return getattr(target, "__name__", str(target))

    def on_build_start(self, *, cfg, ctx) -> None:
        t = cfg.type if hasattr(cfg, "type") else (cfg.get("type") if isinstance(cfg, dict) else None)
        self._push(phase="start", type=t)

    def on_validated(self, *, target, kwargs, ctx) -> None:
        self._push(
            phase="validated",
            target=self._target_name(target),
            kwargs={k: type(v).__name__ for k, v in kwargs.items()},
        )

    def on_built(self, *, target, result, meta, ctx) -> None:
        self._push(
            phase="built",
            target=self._target_name(target),
            meta=dict(meta),
            result_type=type(result).__name__,
        )

    def on_error(self, *, cfg, exc, ctx) -> None:
        t = cfg.type if hasattr(cfg, "type") else (cfg.get("type") if isinstance(cfg, dict) else None)
        self._push(
            phase="error",
            type=t,
            error_type=type(exc).__name__,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Resource-consumption observers (per envelope, stack-based for nested builds)
# ---------------------------------------------------------------------------


class _StackedObserver(FactoryObserver):
    """Base for observers that need a baseline at on_build_start and a delta at on_built.

    Subclasses implement ``_sample()`` (returns a tuple of numeric counters)
    and ``_write(meta, before, after)`` (writes deltas / absolutes into meta).
    Handles nested builds via an internal stack.
    """

    def __init__(self) -> None:
        self._stack: list[tuple[Any, ...]] = []

    def _sample(self) -> tuple[Any, ...]:
        raise NotImplementedError

    def _write(self, meta: dict[str, Any], before: tuple[Any, ...], after: tuple[Any, ...]) -> None:
        raise NotImplementedError

    def on_build_start(self, *, cfg, ctx) -> None:
        self._stack.append(self._sample())

    def on_built(self, *, target, result, meta, ctx) -> None:
        if not self._stack:
            return
        before = self._stack.pop()
        self._write(meta, before, self._sample())

    def on_error(self, *, cfg, exc, ctx) -> None:
        if self._stack:
            self._stack.pop()


class LifetimeObserver(_StackedObserver):
    """Wall-clock seconds the build (and its recursion) took."""

    name: str = "lifetime"

    def _sample(self) -> tuple[float]:
        return (time.perf_counter(),)

    def _write(self, meta, before, after) -> None:
        meta["lifetime_seconds"] = after[0] - before[0]


class CPUObserver(_StackedObserver):
    """User + system CPU seconds consumed during the build (process-wide)."""

    name: str = "cpu"

    def _sample(self) -> tuple[float, float]:
        r = resource.getrusage(resource.RUSAGE_SELF)
        return (r.ru_utime, r.ru_stime)

    def _write(self, meta, before, after) -> None:
        meta["cpu_user_seconds"] = after[0] - before[0]
        meta["cpu_system_seconds"] = after[1] - before[1]


class MemoryObserver(_StackedObserver):
    """Max resident set size (KB on Linux) and delta vs. build start."""

    name: str = "memory"

    def _sample(self) -> tuple[int]:
        # ru_maxrss is KB on Linux, bytes on macOS. We assume Linux here.
        return (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,)

    def _write(self, meta, before, after) -> None:
        meta["rss_max_kb"] = after[0]
        meta["rss_delta_kb"] = after[0] - before[0]


class IOObserver(_StackedObserver):
    """Disk IO bytes via ``/proc/self/io`` (Linux). Falls back to zeros elsewhere.

    Tracks both ``rchar``/``wchar`` (bytes read/written through the syscall
    interface, including from the page cache) and ``read_bytes``/``write_bytes``
    (bytes that actually reached the storage layer).
    """

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


class NetworkObserver(_StackedObserver):
    """Network bytes received/sent across all interfaces via ``/proc/self/net/dev``.

    Counters are per-namespace, summed across all interfaces (including ``lo``).
    Subtract ``lo`` separately if you only care about external traffic.
    """

    name: str = "network"

    @staticmethod
    def _read_proc_net_dev() -> tuple[int, int]:
        rx_total = 0
        tx_total = 0
        try:
            with open("/proc/self/net/dev") as f:
                lines = f.readlines()[2:]  # skip 2 header lines
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


class HeapObserver(_StackedObserver):
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


class RecursionObserver(FactoryObserver):
    """Tracks how deep the factory recursion went for each top-level build.

    Writes ``build_depth`` (the depth of THIS envelope, 1 = top-level) and
    ``build_max_depth`` (deepest level seen so far in the current top-level
    build) into meta on every envelope.
    """

    name: str = "recursion"

    def __init__(self) -> None:
        self._depth: int = 0
        self._max_depth: int = 0

    def on_build_start(self, *, cfg, ctx) -> None:
        self._depth += 1
        if self._depth > self._max_depth:
            self._max_depth = self._depth

    def on_built(self, *, target, result, meta, ctx) -> None:
        meta["build_depth"] = self._depth
        meta["build_max_depth"] = self._max_depth
        self._depth -= 1
        if self._depth == 0:
            self._max_depth = 0

    def on_error(self, *, cfg, exc, ctx) -> None:
        if self._depth > 0:
            self._depth -= 1
        if self._depth == 0:
            self._max_depth = 0
