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

from typing import Any, Dict, Optional, Tuple

__all__ = ["RESOLVE_CACHE", "invalidate_resolve_cache"]


# (type_name, repo) -> (registry_class, artifact). ``repo`` is None when
# the resolve call did not narrow by repo.
RESOLVE_CACHE: Dict[Tuple[str, Optional[str]], Tuple[type, Any]] = {}


def invalidate_resolve_cache() -> None:
    """Drop every memoized resolve() result."""
    RESOLVE_CACHE.clear()
