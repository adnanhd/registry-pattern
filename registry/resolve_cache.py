"""Process-wide memoization for ``factory.resolve(type_name, repo)``.

Lives in its own leaf module so both ``factory`` (the reader) and
``typ_registry`` / ``fnc_registry`` (the invalidators) can import it
without re-introducing the factory <-> typ_registry import cycle.

Any mutation of the registry universe -- a new artifact registered, an
existing one unregistered, a new ``TypeRegistry`` / ``FunctionalRegistry``
subclass declared -- must call :func:`invalidate_resolve_cache` so stale
entries are dropped.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

__all__ = [
    "RESOLVE_CACHE",
    "invalidate_resolve_cache",
    "current_generation",
    "write_if_current",
]


# (type_name, repo) -> (registry_class, artifact). ``repo`` is None when
# the resolve call did not narrow by repo.
RESOLVE_CACHE: Dict[Tuple[str, Optional[str]], Tuple[type, Any]] = {}

# Epoch counter bumped on every invalidation. ``resolve()`` snapshots it
# before scanning the registries and passes the snapshot to
# ``write_if_current`` so a scan that started before a concurrent
# invalidation can't write its (now-stale) result back in after the fact.
_LOCK = threading.Lock()
_GENERATION = 0


def invalidate_resolve_cache() -> None:
    """Drop every memoized resolve() result and advance the epoch."""
    global _GENERATION
    with _LOCK:
        _GENERATION += 1
        RESOLVE_CACHE.clear()


def current_generation() -> int:
    """Return the current cache epoch.

    Callers doing a scan-then-cache sequence should snapshot this *before*
    scanning and pass it to :func:`write_if_current` when the scan
    completes, so a concurrent invalidation mid-scan is not silently
    overwritten by a stale write.
    """
    with _LOCK:
        return _GENERATION


def write_if_current(
    key: Tuple[str, Optional[str]], value: Tuple[type, Any], generation: int
) -> bool:
    """Write ``value`` into the cache iff no invalidation happened since ``generation``.

    Returns whether the write happened. Guards against the race where a
    ``resolve()`` cache miss starts scanning the registries, a concurrent
    unregister invalidates the cache mid-scan, and the miss then finishes
    and would otherwise write a stale (already-unregistered) result back in.
    """
    with _LOCK:
        if generation != _GENERATION:
            return False
        RESOLVE_CACHE[key] = value
        return True
