"""Factory meter bus -- attach probes, dispatch stage events into ``meta``.

A ``FactoryMeter`` is a *probe*: it reads pipeline state (cfg / target / kwargs /
result) at the stages it cares about and writes its measurement into the
envelope's ``meta`` dict. It does not ship anything externally (for that see
:mod:`registry.reporters`).

This module is the BUS only -- the base class plus attach / detach / emit. The
concrete batteries (CPU, memory, IO, network, heap, recursion, lifetime) are
opt-in and live in :mod:`registry.extra.meters`.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

__all__ = [
    "FactoryMeter",
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

    def on_build_start(
        self, *, cfg: Any, ctx: Dict[str, Any], meta: Dict[str, Any]
    ) -> None:
        """Pre-recursion / pre-validation. Good place to record baselines."""

    def on_validated(
        self,
        *,
        target: Any,
        kwargs: Dict[str, Any],
        ctx: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None:
        """After kwargs assembled + validated, before ``target(**kwargs)``."""

    def on_built(
        self,
        *,
        target: Any,
        result: Any,
        meta: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> None:
        """After invocation + post hooks. Compute deltas and write to ``meta``."""

    def on_error(
        self,
        *,
        cfg: Any,
        exc: BaseException,
        ctx: Dict[str, Any],
        meta: Dict[str, Any],
    ) -> None:
        """Build failed. Clean up any baseline state."""


# ---------------------------------------------------------------------------
# Module-level meter registry (the bus)
# ---------------------------------------------------------------------------

_METERS: Dict[str, FactoryMeter] = {}
_LOCK = threading.Lock()


def attach_meter(meter: FactoryMeter) -> FactoryMeter:
    """Register a meter. Replaces any prior meter with the same ``name``."""
    with _LOCK:
        _METERS[meter.name] = meter
    return meter


def detach_meter(name: str) -> Optional[FactoryMeter]:
    """Remove and return the meter registered under ``name``."""
    with _LOCK:
        return _METERS.pop(name, None)


def meters() -> Dict[str, FactoryMeter]:
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
