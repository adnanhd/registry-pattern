"""Factory reporter bus -- attach sinks, dispatch stage events externally.

A ``FactoryReporter`` is a *sink*: it receives pipeline events (with the
``meta`` dict already populated by meters) and ships them somewhere
external -- syslog, an HTTP endpoint, an OpenTelemetry collector, a file,
whatever. Reporters do **not** write to ``meta``.

This module is the BUS only -- the base class plus attach / detach / emit. The
concrete sinks (journald, an HTTP dashboard, OpenTelemetry) are opt-in and live
in :mod:`registry.extra.reporters`. For measurements (CPU, memory, time, etc.)
see :mod:`registry.meters`.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

__all__ = [
    "FactoryReporter",
    "attach_reporter",
    "detach_reporter",
    "reporters",
    "emit_reporter",
]


class FactoryReporter:
    """Base reporter. Override the stage methods you care about."""

    name: str = "factory_reporter"

    def on_build_start(
        self, *, cfg: Any, ctx: Dict[str, Any], meta: Dict[str, Any]
    ) -> None: ...

    def on_validated(
        self,
        *,
        target: Any,
        kwargs: Dict[str, Any],
        ctx: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None: ...

    def on_built(
        self,
        *,
        target: Any,
        result: Any,
        meta: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> None: ...

    def on_error(
        self,
        *,
        cfg: Any,
        exc: BaseException,
        ctx: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None: ...


# ---------------------------------------------------------------------------
# Reporter registry (the bus)
# ---------------------------------------------------------------------------

_REPORTERS: Dict[str, FactoryReporter] = {}
_LOCK = threading.Lock()


def attach_reporter(reporter: FactoryReporter) -> FactoryReporter:
    """Register a reporter. Replaces any prior with the same ``name``."""
    with _LOCK:
        _REPORTERS[reporter.name] = reporter
    return reporter


def detach_reporter(name: str) -> Optional[FactoryReporter]:
    """Remove and return the reporter registered under ``name``."""
    with _LOCK:
        return _REPORTERS.pop(name, None)


def reporters() -> Dict[str, FactoryReporter]:
    """Snapshot of currently attached reporters."""
    with _LOCK:
        return dict(_REPORTERS)


def emit_reporter(method: str, /, **payload: Any) -> None:
    """Call ``method`` on every attached reporter. Errors swallowed."""
    for r in reporters().values():
        fn = getattr(r, method, None)
        if fn is None:
            continue
        try:
            fn(**payload)
        except Exception:  # noqa: BLE001
            pass
