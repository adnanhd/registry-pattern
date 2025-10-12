r"""Mutable registry mixin with rich error context.

This module adds write operations on top of the read-only accessor mixin.

Behavior:
  - `_set_mapping` replaces the entire mapping after asserting the current one is clear.
  - `_update_mapping` inserts multiple items, asserting absence for each key.
  - Single-item ops (`_set_artifact`, `_update_artifact`, `_del_artifact`) delegate
    to presence/absence guards that raise `RegistryError` with context.

Simple inheritance diagram (Doxygen dot):
\dot
digraph RegistryPattern {
    rankdir=LR;
    node [shape=rectangle];
    "RegistryAccessorMixin" -> "RegistryMutatorMixin";
}
\enddot
"""

from __future__ import annotations

from typing import Dict, Hashable, Mapping, MutableMapping, TypeVar, Union

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
    # Setter Functions for Registry Object
    # -----------------------------------------------------------------------------
    @classmethod
    def _set_mapping(cls, mapping: Mapping[KeyType, ValType]) -> None:
        """Replace the underlying mapping with `mapping` after clearing."""
        cls._clear_mapping()
        cls._update_mapping(mapping)

    @classmethod
    def _update_mapping(cls, mapping: Mapping[KeyType, ValType]) -> None:
        """Insert all items from `mapping`, asserting current absence for each key.

        Raises:
            RegistryError: if any key already exists.
        """
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

        Raises:
            RegistryError: if `key` is already present.
        """
        cls._assert_absence(key)[key] = item

    @classmethod
    def _update_artifact(cls, key: KeyType, item: ValType) -> None:
        """Replace `item` under `key`.

        Raises:
            RegistryError: if `key` is not present.
        """
        cls._assert_presence(key)[key] = item

    # -----------------------------------------------------------------------------
    # Deleter Functions for Registry Items
    # -----------------------------------------------------------------------------

    @classmethod
    def _del_artifact(cls, key: KeyType) -> None:
        """Delete the entry under `key`.

        Raises:
            RegistryError: if `key` is not present.
        """
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
