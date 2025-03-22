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
    Generic,
    Hashable,
    MutableMapping,
    Iterator,
    TypeVar,
    Union,
)


__all__ = [
    "RegistryError",
    "RegistryAccessorMixin",
]

# -----------------------------------------------------------------------------
# Exception Definitions
# -----------------------------------------------------------------------------


class RegistryError(KeyError):
    """Exception raised for errors during mapping operations."""


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


class RegistryAccessorMixin(Generic[K, T]):

    # -----------------------------------------------------------------------------
    # Getter Functions for Registry
    # -----------------------------------------------------------------------------

    @classmethod
    def _get_mapping(cls) -> Union[Dict[K, T], MutableMapping[K, T]]:
        """
        Get the underlying mapping of the registry.

        Returns:
            Registry[K, T]: The mapping of the registry.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def _len_mapping(cls) -> int:
        """
        Get the number of artifacts in the registry.

        Returns:
            int: The number of artifacts in the registry.
        """
        return len(cls._get_mapping())

    @classmethod
    def _iter_mapping(cls) -> Iterator[K]:
        """
        Iterate over the keys in the registry.

        Returns:
            Iterator[K]: An iterator over the keys in the registry.
        """
        return iter(cls._get_mapping())

    # -----------------------------------------------------------------------------
    # Getter Functions for Registry Contents
    # -----------------------------------------------------------------------------

    @classmethod
    def _get_artifact(cls, key: K) -> T:
        """
        Get an artifact from the registry.

        Parameters:
            artifact (K): The key of the artifact to retrieve.

        Returns:
            T: The artifact associated with the key.

        Raises:
            RegistryError: If the key is not found in the registry.
        """
        return cls._assert_presence(key)[key]

    @classmethod
    def _has_identifier(cls, key: K) -> bool:
        """
        Check if an artifact exists in the registry.

        Parameters:
            key (K): The key of the item to check.

        Returns:
            bool: True if the item exists, False otherwise.
        """
        return key in cls._get_mapping().keys()

    @classmethod
    def _has_artifact(cls, item: T) -> bool:
        """
        Check if an artifact exists in the registry.

        Parameters:
            item (K): The key of the item to check.

        Returns:
            bool: True if the item exists, False otherwise.
        """
        return item in cls._get_mapping().values()

    # -----------------------------------------------------------------------------
    # Helper Functions for Error Handling
    # -----------------------------------------------------------------------------

    @classmethod
    def _assert_presence(cls, key: K) -> Union[Dict[K, T], MutableMapping[K, T]]:
        """
        Raise an error indicating that a given key is not found in the mapping.

        Parameters:
            key (K): The key that was not found.
        Returns:
            Union[Dict[K, T], MutableRegistry[K, T]]: The mapping.
        Raises:
            RegistryError: If the key is not found in the mapping.
        """
        mapping = cls._get_mapping()
        if key not in mapping:
            raise RegistryError(f"Key '{key}' is not found in the mapping")
        return mapping
