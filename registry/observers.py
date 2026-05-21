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
import syslog
import threading
from typing import Any, Callable

__all__ = [
    "FactoryObserver",
    "JournalObserver",
    "HTTPDashboardObserver",
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
