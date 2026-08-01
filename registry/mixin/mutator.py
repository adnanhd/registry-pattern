r"""Mutable registry mixin with rich error context.

This module adds write operations on top of the read-only accessor mixin.

Behavior:
  - `_set_mapping` replaces the entire mapping after asserting the current one is clear.
  - `_update_mapping` inserts multiple items, asserting absence for each key.
  - Single-item ops (`_set_artifact`, `_update_artifact`, `_del_artifact`) delegate
    to presence/absence guards that raise `RegistryError` with context.

Simple inheritance diagram (Doxygen dot):

.. code-block:: text

   digraph RegistryPattern {
       rankdir=LR;
       node [shape=rectangle];
       "RegistryAccessorMixin" -> "RegistryMutatorMixin";
   }
"""

from __future__ import annotations

import contextlib
from collections.abc import Hashable, Mapping, MutableMapping
from typing import Dict, Iterator, TypeVar, Union

from ..utils import RegistryError, get_type_name
from .accessor import RegistryAccessorMixin

__all__ = [
    "RegistryMutatorMixin",
]


# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------

KeyType = TypeVar("KeyType", bound=Hashable)
ValType = TypeVar("ValType")


# -----------------------------------------------------------------------------
# Base Mixin for Mutating Registry Items
# -----------------------------------------------------------------------------


class RegistryMutatorMixin(RegistryAccessorMixin[KeyType, ValType]):
    """Write-side extensions for a registry.

    Error semantics:
        Presence/absence checks raise `RegistryError` with rich context.
    """

    # -----------------------------------------------------------------------------
    # Locking helper -- makes check-then-act sequences atomic
    # -----------------------------------------------------------------------------

    @classmethod
    @contextlib.contextmanager
    def _mapping_lock(cls) -> Iterator[None]:
        """Hold the backing mapping's lock across a compound check-then-act.

        Uses the mapping's own ``lock()`` (``ThreadSafeLocalStorage`` exposes
        one) if it has one, so a presence/absence assertion and the write or
        delete that follows it happen atomically under concurrent access --
        no other thread can observe or mutate the mapping in between. Falls
        back to a no-op context for custom backing mappings that don't
        expose a lock (see ``TypeRegistry``/``FunctionalRegistry`` docstrings
        on assigning a custom ``_repository``).
        """
        mapping = cls._get_mapping()
        lock = getattr(mapping, "lock", None)
        if callable(lock):
            with lock():
                yield
        else:
            yield

    # -----------------------------------------------------------------------------
    # Setter Functions for Registry Object
    # -----------------------------------------------------------------------------
    @classmethod
    def _set_mapping(cls, mapping: Mapping[KeyType, ValType]) -> None:
        """Replace the underlying mapping with `mapping` after clearing."""
        with cls._mapping_lock():
            cls._clear_mapping()
            cls._update_mapping(mapping)

    @classmethod
    def _update_mapping(cls, mapping: Mapping[KeyType, ValType]) -> None:
        """Insert all items from `mapping`, asserting current absence for each key.

        Raises:
            RegistryError: if any key already exists.
        """
        with cls._mapping_lock():
            if cls._len_mapping() > 0:
                for key in mapping.keys():
                    cls._assert_absence(key)
            cls._get_mapping().update(mapping)

    # -----------------------------------------------------------------------------
    # Deleter Functions for Registry Object
    # -----------------------------------------------------------------------------

    @classmethod
    def _clear_mapping(cls) -> None:
        """Clear all entries from the underlying mapping."""
        cls._get_mapping().clear()

    # -----------------------------------------------------------------------------
    # Setter Functions for Registry Items
    # -----------------------------------------------------------------------------

    @classmethod
    def _set_artifact(cls, key: KeyType, item: ValType) -> None:
        """Insert `item` under `key`.

        The absence check and the write are atomic under concurrent access
        (see `_mapping_lock`): if two threads race to register the same
        key, exactly one write wins and the other raises `RegistryError`.

        Raises:
            RegistryError: if `key` is already present.
        """
        with cls._mapping_lock():
            cls._assert_absence(key)[key] = item

    @classmethod
    def _update_artifact(cls, key: KeyType, item: ValType) -> None:
        """Replace `item` under `key`.

        The presence check and the write are atomic under concurrent access
        (see `_mapping_lock`).

        Raises:
            RegistryError: if `key` is not present.
        """
        with cls._mapping_lock():
            cls._assert_presence(key)[key] = item

    # -----------------------------------------------------------------------------
    # Deleter Functions for Registry Items
    # -----------------------------------------------------------------------------

    @classmethod
    def _del_artifact(cls, key: KeyType) -> None:
        """Delete the entry under `key`.

        The presence check and the delete are atomic under concurrent access
        (see `_mapping_lock`): a concurrent unregister of the same key
        cannot slip in between the check and the delete, so callers see a
        proper `RegistryError` rather than a raw `KeyError`.

        Raises:
            RegistryError: if `key` is not present.
        """
        with cls._mapping_lock():
            del cls._assert_presence(key)[key]

    # -----------------------------------------------------------------------------
    # Helper Functions for Error Handling with Rich Context
    # -----------------------------------------------------------------------------

    @classmethod
    def _assert_absence(
        cls, key: KeyType
    ) -> Union[Dict[KeyType, ValType], MutableMapping[KeyType, ValType]]:
        """Return mapping if `key` is absent; otherwise raise `RegistryError`."""
        mapping = cls._get_mapping()
        if key in mapping:
            suggestions = [
                f"Key '{key}' already exists in {getattr(cls, '__name__', 'registry')}",
                "Use a different key name",
                "Use _update_artifact() to modify existing entries",
                "Remove the existing entry first with _del_artifact()",
            ]
            context = {
                "operation": "assert_absence",
                "registry_name": getattr(cls, "__name__", "Unknown"),
                "registry_type": get_type_name(cls),
                "key": str(key),
                "key_type": get_type_name(type(key)),
                "registry_size": len(mapping),
                "conflicting_key": str(key),
            }
            raise RegistryError(
                f"Key '{key}' is already found in the mapping", suggestions, context
            )
        return mapping
