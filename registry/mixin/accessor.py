r"""Enhanced Registry Accessor Module with Rich Error Context
========================================================

This module provides base classes for implementing a registry pattern,
allowing the registration, lookup, and management of classes or functions
with enhanced error reporting and rich context information.

The main classes are:
 - RegistryAccessorMixin: A read-only registry for managing registered items with enhanced errors.

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph RegistryPattern {
    rankdir=LR;
    node [shape=rectangle];
    "RegistryAccessorMixin" -> "Enhanced Operations";
}
\enddot
"""

from __future__ import annotations

from typing_compat import (
    Dict,
    Generic,
    Hashable,
    Iterator,
    MutableMapping,
    TypeVar,
    Union,
    Optional,
    List,
    Any,
)

# Global imports - never import in except blocks
from registry.core._validator import ValidationError, get_type_name

__all__ = [
    "RegistryError",
    "RegistryAccessorMixin",
]

# -----------------------------------------------------------------------------
# Enhanced Exception Definition
# -----------------------------------------------------------------------------

if ValidationError:

    class RegistryError(ValidationError, KeyError):
        """Enhanced exception raised for errors during mapping operations with rich context."""

        def __init__(
            self,
            message: str,
            suggestions: Optional[List[str]] = None,
            context: Optional[Dict[str, Any]] = None,
        ):
            # Initialize ValidationError with rich context (this handles enhanced message formatting)
            ValidationError.__init__(self, message, suggestions, context)
            # Initialize KeyError with the basic message for backward compatibility
            KeyError.__init__(self, message)

else:
    # Fallback to simple RegistryError if ValidationError not available
    class RegistryError(KeyError):
        """Simple registry error for backward compatibility."""

        def __init__(
            self,
            message: str,
            suggestions: Optional[List[str]] = None,
            context: Optional[Dict[str, Any]] = None,
        ):
            super().__init__(message)
            self.suggestions = suggestions or []
            self.context = context or {}


# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------

# K is a type variable for keys that must be hashable.
K = TypeVar("K", bound=Hashable)
# T is a type variable for the registered item.
T = TypeVar("T")


# -----------------------------------------------------------------------------
# Enhanced Base Mixin for Accessing Registry Items
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
    # Getter Functions for Registry Contents with Enhanced Error Handling
    # -----------------------------------------------------------------------------

    @classmethod
    def _get_artifact(cls, key: K) -> T:
        """
        Get an artifact from the registry with enhanced error context.

        Parameters:
            key (K): The key of the artifact to retrieve.

        Returns:
            T: The artifact associated with the key.

        Raises:
            RegistryError: If the key is not found in the registry (with rich context).
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
            item (T): The artifact to check.

        Returns:
            bool: True if the item exists, False otherwise.
        """
        return item in cls._get_mapping().values()

    # -----------------------------------------------------------------------------
    # Helper Functions for Error Handling with Rich Context
    # -----------------------------------------------------------------------------

    @classmethod
    def _assert_presence(cls, key: K) -> Union[Dict[K, T], MutableMapping[K, T]]:
        """
        Raise an enhanced error if a given key is not found in the mapping.

        Parameters:
            key (K): The key that was not found.
        Returns:
            Union[Dict[K, T], MutableMapping[K, T]]: The mapping.
        Raises:
            RegistryError: If the key is not found in the mapping (with rich context).
        """
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
                "key": str(key)[:999],
                "key_type": type(key).__name__,
                "registry_size": len(mapping),
                "available_keys": (
                    list(mapping.keys())
                    if len(mapping) <= 10
                    else f"{list(mapping.keys())[:10]}... ({len(mapping)} total)"
                ),
            }
            raise RegistryError(
                f"Key '{key}' is not found in the mapping", suggestions, context
            )
        return mapping
