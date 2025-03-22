r"""Registry Pattern Module
=======================

This module provides base classes for implementing a registry pattern,
allowing the registration, lookup, and management of classes or functions.

The main classes are:
 - BaseRegistry: A read-only registry for managing registered items.
 - BaseMutableRegistry: An extension that allows adding and removing items.
 - Registry: Provides dictionary-like read-only access.
 - MutableRegistry: A mutable registry supporting set and delete operations.

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph RegistryPattern {
    rankdir=LR;
    node [shape=rectangle];
    "BaseRegistry" -> "Registry";
    "BaseRegistry" -> "BaseMutableRegistry";
    "BaseMutableRegistry" -> "MutableRegistry";
    "Registry" -> "MutableRegistry";
}
\enddot
"""

from __future__ import annotations

from typing_compat import (
    Dict,
    Hashable,
    Mapping,
    MutableMapping,
    TypeVar,
    Union,
)

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
T = TypeVar("T")  # You may later bind this to Stringable or any other type if needed.


# -----------------------------------------------------------------------------
# Base Mixin for Accessing Registry Items
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
    # Setter Functions for Registry Items
    # -----------------------------------------------------------------------------

    @classmethod
    def _set_artifact(cls, key: K, item: T) -> None:
        """
        Set an artifact in the registry.

        Parameters:
            artifact (K): The key of the artifact to set.
            item (T): The item to associate with the key.
        Raises:
            RegistryError: If the key is already found in the mapping.
        """
        cls._assert_absence(key)[key] = item

    @classmethod
    def _update_artifact(cls, key: K, item: T) -> None:
        """
        Update an existing artifact in the registry.

        Parameters:
            artifact (K): The key of the artifact to set.
            item (T): The item to associate with the key.
        Raises:
            RegistryError: If the key is already found in the mapping.
        """
        cls._assert_presence(key)[key] = item

    # -----------------------------------------------------------------------------
    # Deleter Functions for Registry Items
    # -----------------------------------------------------------------------------

    @classmethod
    def _del_artifact(cls, key: K) -> None:
        """
        Delete an artifact from the registry.

        Parameters:
            key (K): The key of the key to delete.
        Raises:
            RegistryError: If the key is not found in the mapping.
        """
        del cls._assert_presence(key)[key]

    # -----------------------------------------------------------------------------
    # Helper Functions for Error Handling
    # -----------------------------------------------------------------------------

    @classmethod
    def _assert_absence(cls, key: K) -> Union[Dict[K, T], MutableMapping[K, T]]:
        """
        Raise an error indicating that a given key is already found in the mapping.

        Parameters:
            key (K): The key that was not found.
        Returns:
            Union[Dict[K, T], MutableRegistry[K, T]]: The mapping.
        Raises:
            RegistryError: If the key is already found in the mapping.
        """
        mapping = cls._get_mapping()
        if key in mapping:
            raise RegistryError(f"Key '{key}' is already found in the mapping.")
        return mapping
