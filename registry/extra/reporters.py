"""Concrete factory reporters -- opt-in batteries.

Each reporter ships pipeline events (with ``meta`` already populated by meters)
to an external destination. Attach them with the bus in
:mod:`registry.reporters`::

    from registry import attach_reporter
    from registry.extra.reporters import JournalReporter, HTTPDashboardReporter

    attach_reporter(JournalReporter(ident="my-app"))
    attach_reporter(HTTPDashboardReporter(port=8765))

- :class:`JournalReporter`       -- syslog -> systemd journald
- :class:`HTTPDashboardReporter` -- JSON over ``localhost:port``
- :class:`OpenTelemetryReporter` -- spans + Histogram metrics (needs ``[otel]``)
"""

from __future__ import annotations

import http.server
import json
import syslog
import threading
import time
from typing import Any, Dict, List, Tuple

from ..reporters import FactoryReporter

__all__ = [
    "JournalReporter",
    "HTTPDashboardReporter",
    "OpenTelemetryReporter",
]


# ---------------------------------------------------------------------------
# Helpers used by multiple reporters
# ---------------------------------------------------------------------------


def _type_of(cfg: Any) -> Any:
    if hasattr(cfg, "type"):
        return cfg.type
    if isinstance(cfg, dict):
        return cfg.get("type")
    return None


def _target_name(target: Any) -> str:
    return getattr(target, "__name__", str(target))


# ---------------------------------------------------------------------------
# JournalReporter -- syslog
# ---------------------------------------------------------------------------


class JournalReporter(FactoryReporter):
    """One syslog line per stage. journald captures by default.

    Usage::

        attach_reporter(JournalReporter(ident="my-trainer"))
        # later: journalctl -t my-trainer
    """

    name: str = "journal"

    def __init__(self, ident: str = "registry") -> None:
        syslog.openlog(ident, syslog.LOG_PID, syslog.LOG_DAEMON)

    def on_build_start(self, *, cfg, ctx, meta) -> None:
        syslog.syslog(syslog.LOG_DEBUG, f"build.start type={_type_of(cfg)}")

    def on_validated(self, *, target, kwargs, ctx, meta) -> None:
        syslog.syslog(
            syslog.LOG_INFO,
            f"build.validated target={_target_name(target)} kwargs={list(kwargs)}",
        )

    def on_built(self, *, target, result, meta, ctx) -> None:
        syslog.syslog(
            syslog.LOG_INFO,
            f"build.done target={_target_name(target)} meta={meta}",
        )

    def on_error(self, *, cfg, exc, ctx, meta) -> None:
        syslog.syslog(
            syslog.LOG_ERR,
            f"build.error type={_type_of(cfg)} error_type={type(exc).__name__} error={exc}",
        )


# ---------------------------------------------------------------------------
# HTTPDashboardReporter -- JSON on localhost:PORT
# ---------------------------------------------------------------------------


class HTTPDashboardReporter(FactoryReporter):
    """Serves the last N pipeline events as JSON on ``localhost:<port>``.

    Daemon thread, no shutdown semantics. Use::

        attach_reporter(HTTPDashboardReporter(port=8765))
        # curl localhost:8765 | jq
    """

    name: str = "http_dashboard"

    def __init__(
        self, port: int = 8765, max_events: int = 200, host: str = "127.0.0.1"
    ) -> None:
        self._events: List[Dict[str, Any]] = []
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

            def log_message(self, *a: Any, **kw: Any) -> None:
                pass

        server = http.server.HTTPServer((host, port), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def _push(self, **event: Any) -> None:
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events.pop(0)

    def on_build_start(self, *, cfg, ctx, meta) -> None:
        self._push(phase="start", type=_type_of(cfg))

    def on_validated(self, *, target, kwargs, ctx, meta) -> None:
        self._push(
            phase="validated",
            target=_target_name(target),
            kwargs={k: type(v).__name__ for k, v in kwargs.items()},
        )

    def on_built(self, *, target, result, meta, ctx) -> None:
        self._push(
            phase="built",
            target=_target_name(target),
            meta=dict(meta),
            result_type=type(result).__name__,
        )

    def on_error(self, *, cfg, exc, ctx, meta) -> None:
        self._push(
            phase="error",
            type=_type_of(cfg),
            error_type=type(exc).__name__,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# OpenTelemetryReporter -- spans + metric Histogram
# ---------------------------------------------------------------------------


class OpenTelemetryReporter(FactoryReporter):
    """Emits a span per build (start at on_build_start, end at on_built / on_error)
    and records a ``registry.build.duration`` Histogram. Requires the ``otel``
    extra (``pip install 'registry-pattern[otel]'``).

    Meta keys written by Meters become span attributes prefixed with
    ``registry.meta.`` so they show up alongside the span in your OTel backend
    (Jaeger / Tempo / Honeycomb / etc.).

    Configure OTel globally (TracerProvider + MeterProvider + exporters) before
    attaching this reporter; this class only consumes the OTel API.
    """

    name: str = "opentelemetry"

    def __init__(
        self,
        tracer_name: str = "registry.factory",
        meter_name: str = "registry.factory",
        span_name_prefix: str = "registry.build",
    ) -> None:
        from opentelemetry import (  # pyright: ignore[reportMissingImports]; raises ImportError if [otel] not installed
            metrics,
            trace,
        )

        self._tracer = trace.get_tracer(tracer_name)
        self._meter = metrics.get_meter(meter_name)
        self._duration = self._meter.create_histogram(
            name="registry.build.duration",
            unit="s",
            description="Wall time per registry.build invocation",
        )
        self._span_prefix = span_name_prefix
        self._stack: List[Tuple[Any, float, str]] = []  # (span, start_perf, type)

    def on_build_start(self, *, cfg, ctx, meta) -> None:
        t = _type_of(cfg) or "unknown"
        span = self._tracer.start_span(f"{self._span_prefix}.{t}")
        span.set_attribute("registry.type", str(t))
        repo = getattr(cfg, "repo", None) if hasattr(cfg, "repo") else None
        if repo:
            span.set_attribute("registry.repo", str(repo))
        self._stack.append((span, time.perf_counter(), str(t)))

    def on_built(self, *, target, result, meta, ctx) -> None:
        if not self._stack:
            return
        span, start, type_name = self._stack.pop()
        elapsed = time.perf_counter() - start
        span.set_attribute("registry.target", _target_name(target))
        for k, v in meta.items():
            if isinstance(v, (int, float, str, bool)):
                span.set_attribute(f"registry.meta.{k}", v)
        span.end()
        self._duration.record(elapsed, attributes={"registry.type": type_name})

    def on_error(self, *, cfg, exc, ctx, meta) -> None:
        if not self._stack:
            return
        from opentelemetry.trace import (  # pyright: ignore[reportMissingImports]
            Status,
            StatusCode,
        )

        span, _, _ = self._stack.pop()
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.end()
