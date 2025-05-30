r"""Enhanced Registry Mutator Module with Rich Error Context
========================================================

This module provides base classes for implementing a registry pattern,
allowing the registration, lookup, and management of classes or functions
with enhanced error reporting and rich context information.

The main classes are:
 - RegistryMutatorMixin: A mutable registry for managing registered items with enhanced errors.

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph RegistryPattern {
    rankdir=LR;
    node [shape=rectangle];
    "RegistryAccessorMixin" -> "RegistryMutatorMixin";
}
\enddot
"""

from __future__ import annotations

from typing_compat import Dict, Hashable, Mapping, MutableMapping, TypeVar, Union

from .accessor import RegistryAccessorMixin, RegistryError

__all__ = [
    "RegistryMutatorMixin",
]


# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------

# K is a type variable for keys that must be hashable.
K = TypeVar("K", bound=Hashable)
# T is a type variable for the registered item.
T = TypeVar("T")


# -----------------------------------------------------------------------------
# Enhanced Base Mixin for Mutating Registry Items
# -----------------------------------------------------------------------------


class RegistryMutatorMixin(RegistryAccessorMixin[K, T]):

    # -----------------------------------------------------------------------------
    # Setter Functions for Registry Object
    # -----------------------------------------------------------------------------
    @classmethod
    def _set_mapping(cls, mapping: Mapping[K, T]) -> None:
        """
        Set the underlying mapping of the registry.

        Parameters:
            mapping (Registry[K, T]): The new mapping.
        """
        cls._clear_mapping()
        cls._update_mapping(mapping)

    @classmethod
    def _update_mapping(cls, mapping: Mapping[K, T]) -> None:
        """
        Update the underlying mapping of the registry.

        Parameters:
            mapping (Registry[K, T]): The new mapping.
        Raises:
            RegistryError: If any key is already found in the mapping.
        """
        if cls._len_mapping() > 0:
            for key, value in mapping.items():
                cls._assert_absence(key)
        cls._get_mapping().update(mapping)

    # -----------------------------------------------------------------------------
    # Deleter Functions for Registry Object
    # -----------------------------------------------------------------------------

    @classmethod
    def _clear_mapping(cls) -> None:
        """
        Clear the underlying mapping of the registry.
        """
        cls._get_mapping().clear()

    # -----------------------------------------------------------------------------
    # Setter Functions for Registry Items with Enhanced Error Handling
    # -----------------------------------------------------------------------------

    @classmethod
    def _set_artifact(cls, key: K, item: T) -> None:
        """
        Set an artifact in the registry with enhanced error handling.

        Parameters:
            key (K): The key of the artifact to set.
            item (T): The item to associate with the key.
        Raises:
            RegistryError: If the key is already found in the mapping (with rich context).
        """
        cls._assert_absence(key)[key] = item

    @classmethod
    def _update_artifact(cls, key: K, item: T) -> None:
        """
        Update an existing artifact in the registry.

        Parameters:
            key (K): The key of the artifact to set.
            item (T): The item to associate with the key.
        Raises:
            RegistryError: If the key is not found in the mapping.
        """
        cls._assert_presence(key)[key] = item

    # -----------------------------------------------------------------------------
    # Deleter Functions for Registry Items with Enhanced Error Handling
    # -----------------------------------------------------------------------------

    @classmethod
    def _del_artifact(cls, key: K) -> None:
        """
        Delete an artifact from the registry with enhanced error handling.

        Parameters:
            key (K): The key of the artifact to delete.
        Raises:
            RegistryError: If the key is not found in the mapping (with rich context).
        """
        del cls._assert_presence(key)[key]

    # -----------------------------------------------------------------------------
    # Helper Functions for Error Handling with Rich Context
    # -----------------------------------------------------------------------------

    @classmethod
    def _assert_absence(cls, key: K) -> Union[Dict[K, T], MutableMapping[K, T]]:
        """
        Raise an enhanced error if a given key is already found in the mapping.

        Parameters:
            key (K): The key that should be absent.
        Returns:
            Union[Dict[K, T], MutableMapping[K, T]]: The mapping.
        Raises:
            RegistryError: If the key is already found in the mapping (with rich context).
        """
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
                "registry_type": cls.__class__.__name__,
                "key": str(key),
                "key_type": type(key).__name__,
                "registry_size": len(mapping),
                "conflicting_key": str(key),
            }
            raise RegistryError(
                f"Key '{key}' is already found in the mapping", suggestions, context
            )
        return mapping
