r"""Read-only registry mixin with rich error context.

This module implements `RegistryAccessorMixin`, an abstract, read-focused
interface over a mapping of identifiers to artifacts. It provides consistent
error reporting and context when lookups fail.

Key points:
  - Subclasses must implement `_get_mapping()` to return the backing mapping.
  - Presence checks and retrieval raise `RegistryError` with suggestions.
  - No mutation APIs are exposed here; see `RegistryMutatorMixin` for writes.

Simple inheritance diagram (Doxygen dot):
\dot
digraph RegistryPattern {
    rankdir=LR;
    node [shape=rectangle];
    "RegistryAccessorMixin" -> "Enhanced Operations";
}
\enddot
"""

from __future__ import annotations

from typing import Dict, Generic, Hashable, Iterator, MutableMapping, TypeVar, Union

from ..utils import RegistryError, get_type_name

__all__ = [
    "RegistryAccessorMixin",
]


# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------

KeyType = TypeVar("KeyType", bound=Hashable)
ValType = TypeVar("ValType")


# -----------------------------------------------------------------------------
# Base Mixin for Accessing Registry Items
# -----------------------------------------------------------------------------


class RegistryAccessorMixin(Generic[KeyType, ValType]):
    """Abstract accessor over a registry mapping.

    Subclasses must provide a concrete storage via `_get_mapping()`.

    Error semantics:
        Missing keys are reported via `RegistryError` with
        a suggestions list and context payload suitable for logs.
    """

    # -----------------------------------------------------------------------------
    # Getter Functions for Registry
    # -----------------------------------------------------------------------------
    @classmethod
    def _get_mapping(
        cls,
    ) -> MutableMapping[KeyType, ValType]:
        """Return the underlying mapping for this registry."""
        raise NotImplementedError(
            f"Subclasses must implement `{cls.__name__}._get_mapping` method."
        )

    @classmethod
    def _len_mapping(cls) -> int:
        """Return the number of artifacts in the registry."""
        return len(cls._get_mapping())

    @classmethod
    def _iter_mapping(cls) -> Iterator[KeyType]:
        """Iterate over all identifiers in the registry."""
        return iter(cls._get_mapping())

    # -----------------------------------------------------------------------------
    # Getter Functions for Registry Contents
    # -----------------------------------------------------------------------------

    @classmethod
    def _get_artifact(cls, key: KeyType) -> ValType:
        """Return the artifact registered under `key`, or raise.

        Raises:
            RegistryError: if `key` is not present.
        """
        return cls._assert_presence(key)[key]

    @classmethod
    def _has_identifier(cls, key: KeyType) -> bool:
        """Return True if `key` exists in the registry."""
        return key in cls._get_mapping().keys()

    @classmethod
    def _has_artifact(cls, item: ValType) -> bool:
        """Return True if `item` exists among registry values."""
        return item in cls._get_mapping().values()

    # -----------------------------------------------------------------------------
    # Helper Functions
    # -----------------------------------------------------------------------------

    @classmethod
    def _assert_presence(
        cls, key: KeyType
    ) -> Union[Dict[KeyType, ValType], MutableMapping[KeyType, ValType]]:
        """Return mapping if `key` is present; otherwise raise `RegistryError`."""
        mapping = cls._get_mapping()
        if key not in mapping:
            suggestions = [
                f"Key '{key}' not found in {getattr(cls, '__name__', 'registry')}",
                "Check that the key was registered correctly",
                "Use _has_identifier() to verify key existence",
                f"Registry contains {len(mapping)} items",
            ]
            context = {
                "operation": "assert_presence",
                "registry_name": getattr(cls, "__name__", "Unknown"),
                "registry_type": get_type_name(cls),
                "key": str(key),
                "key_type": type(key).__name__,
                "registry_size": len(mapping),
                "available_keys": (
                    list(mapping.keys())
                    if len(mapping) <= 10
                    else f"{list(mapping.keys())[:10]}. ({len(mapping)} total)"
                ),
            }
            raise RegistryError(
                f"Key '{key}' is not found in the mapping", suggestions, context
            )
        return mapping
