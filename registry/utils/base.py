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

from functools import lru_cache
from typing import (
    Any,
    ClassVar,
    Container,
    Dict,
    Generic,
    Hashable,
    List,
    MutableMapping,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from ._validator import ValidationError

__all__ = [
    "RegistryError",
    "RegistryLookupError",
    "Registry",
    "MutableRegistry",
]

# -----------------------------------------------------------------------------
# Exception Definitions
# -----------------------------------------------------------------------------


class RegistryError(KeyError):
    """
    Exception raised for errors during registration.

    This exception is used when a registration operation fails, e.g.,
    attempting to register a duplicate key.
    """

    pass


class RegistryLookupError(RegistryError):
    """
    Exception raised when a lookup fails.

    This exception is raised when attempting to retrieve a value for a key
    that is not present in the registry.
    """

    pass


# -----------------------------------------------------------------------------
# Protocol Definitions
# -----------------------------------------------------------------------------


class Stringable(Protocol):
    """
    Protocol for objects that can be converted to a string.

    This can be used as a bound for generic types that are expected to be string-like.
    """

    def __str__(self) -> str: ...


# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------

# K is a type variable for keys that must be hashable.
K = TypeVar("K", bound=Hashable)
# T is a type variable for the registered item.
T = TypeVar("T")  # You may later bind this to Stringable or any other type if needed.

# -----------------------------------------------------------------------------
# Helper Functions for Error Handling
# -----------------------------------------------------------------------------


def _not_registered_error(cls: Type, key: Hashable) -> None:
    """
    Raise an error indicating that a given key is not registered.

    Parameters:
        cls (Type): The class performing the lookup.
        key (Hashable): The key that was not found.
    """
    raise RegistryLookupError(f"{cls.__name__}: {key!r} not registered")


def _dup_registered_error(cls: Type, key: Hashable) -> None:
    """
    Raise an error indicating that a given key is already registered.

    Parameters:
        cls (Type): The class attempting the registration.
        key (Hashable): The duplicate key.
    """
    raise RegistryError(f"{cls.__name__}: {key!r} already registered")


# -----------------------------------------------------------------------------
# Base Classes for the Registry Pattern
# -----------------------------------------------------------------------------


class BaseRegistry(Container[T], Generic[K, T]):
    r"""
    Base class for managing a registry of items.

    This class provides the core functionality to validate keys and items,
    check membership, and retrieve registered items. Subclasses may override
    the validation methods to add custom behavior.

    \dot
    digraph BaseRegistry {
        node [shape=rectangle];
        "BaseRegistry" -> "Registry";
        "BaseRegistry" -> "BaseMutableRegistry";
    }
    \enddot
    """

    # The original base classes (for advanced type checking; can be omitted)
    __orig_bases__: ClassVar[Tuple[Type, ...]]

    # Repository to store registered items.
    # This should be set in subclasses or during class initialization.
    _repository: Union[Dict[K, T], MutableMapping[K, T]]

    def __contains__(self, key: Any) -> bool:
        """
        Check if a given key exists in the registry.

        Parameters:
            key (Any): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        lookup_key: K = self.validate_key(key)
        return lookup_key in self.__class__._repository

    @classmethod
    # The lru_cache decorator can be uncommented to cache key validations.
    # @lru_cache(maxsize=16, typed=False)
    def validate_key(cls, key: Any) -> K:
        """
        Validate and convert the input key to a registry key.

        This method can be overridden to add custom key validation or transformation.

        Parameters:
            key (Any): The key to validate.

        Returns:
            K: The validated key.
        """
        return key  # type: ignore

    @classmethod
    def has_registry_key(cls, value: K) -> bool:
        """
        Check if a validated key exists in the registry.

        Parameters:
            value (K): The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return cls.validate_key(value) in cls._repository

    @classmethod
    def validate_item(cls, value: T) -> T:
        """
        Validate the item before registration.

        This method can be overridden to enforce specific constraints on registered items.

        Parameters:
            value (T): The item to validate.

        Returns:
            T: The validated item.
        """
        return value

    @classmethod
    def has_registry_item(cls, value: T) -> bool:
        """
        Check if a validated item exists in the registry.

        Parameters:
            value (T): The item to check.

        Returns:
            bool: True if the item exists, False otherwise.
        """
        try:
            return cls.validate_item(value) in cls._repository.values()
        except ValidationError:
            return False

    @classmethod
    @lru_cache(maxsize=16, typed=False)
    def get_registry_item(cls, key: K) -> T:
        """
        Retrieve an item from the registry by key.

        This method uses caching to improve performance on repeated lookups.

        Parameters:
            key (K): The key for the item.

        Returns:
            T: The registered item.

        Raises:
            RegistryLookupError: If the key is not found in the registry.
        """
        lookup_key: K = cls.validate_key(key)
        if lookup_key not in cls._repository:
            _not_registered_error(cls, key)  # Raise a lookup error.
        return cls._repository[lookup_key]


class BaseMutableRegistry(BaseRegistry[K, T], Generic[K, T]):
    r"""
    Base class for a mutable registry that supports adding and deleting items.

    This class extends BaseRegistry with methods to register new items,
    delete existing ones, and clear the registry.

    \dot
    digraph BaseMutableRegistry {
        node [shape=rectangle];
        "BaseMutableRegistry" -> "MutableRegistry";
    }
    \enddot
    """

    @classmethod
    def add_registry_item(cls, key: K, value: T) -> None:
        """
        Register a new item with the given key.

        Parameters:
            key (K): The key to associate with the item.
            value (T): The item to register.

        Raises:
            RegistryError: If the key is already registered.
        """
        lookup_key: K = cls.validate_key(key)
        if lookup_key not in cls._repository:
            cls._repository[lookup_key] = cls.validate_item(value)
        else:
            _dup_registered_error(cls, key)

    @classmethod
    def del_registry_item(cls, key: K) -> None:
        """
        Delete a registered item by key.

        Parameters:
            key (K): The key of the item to delete.

        Raises:
            RegistryLookupError: If the key is not found.
        """
        lookup_key: K = cls.validate_key(key)
        if lookup_key in cls._repository:
            del cls._repository[lookup_key]
            # Clear the cache of get_registry_item to maintain consistency.
            cls.get_registry_item.cache_clear()
        else:
            _not_registered_error(cls, key)

    @classmethod
    def clear_registry(cls) -> None:
        """
        Clear all registered items from the registry.
        """
        cls._repository.clear()


class Registry(BaseRegistry[K, T], Generic[K, T]):
    r"""
    Registry with dictionary-like read-only access.

    This class provides a familiar mapping interface with methods such as
    __getitem__, len, keys, and values.

    \dot
    digraph Registry {
        node [shape=rectangle];
        "Registry" -> "BaseRegistry";
    }
    \enddot
    """

    def __getitem__(self, key: Any) -> T:
        """
        Allow bracket-notation lookup for registered items.

        Parameters:
            key (Any): The key of the item to retrieve.

        Returns:
            T: The registered item.
        """
        return self.get_registry_item(key)

    @classmethod
    def len(cls) -> int:
        """
        Get the number of registered items.

        Returns:
            int: The total count of items in the registry.
        """
        return len(cls._repository)

    @classmethod
    def keys(cls) -> List[K]:
        """
        Get a list of all registered keys.

        Returns:
            List[K]: A list of keys in the registry.
        """
        return list(cls._repository.keys())

    @classmethod
    def values(cls) -> List[T]:
        """
        Get a list of all registered items.

        Returns:
            List[T]: A list of items in the registry.
        """
        return list(cls._repository.values())


class MutableRegistry(BaseMutableRegistry[K, T], Registry[K, T], Generic[K, T]):
    r"""
    A mutable registry supporting both item assignment and deletion.

    This class combines the functionality of a read-only Registry and a
    mutable BaseMutableRegistry, providing a dictionary-like interface that
    supports setting and deleting items.

    \dot
    digraph MutableRegistry {
        node [shape=rectangle];
        "MutableRegistry" -> "Registry";
        "MutableRegistry" -> "BaseMutableRegistry";
    }
    \enddot
    """

    def __setitem__(self, key: K, value: T) -> None:
        """
        Allow setting items using bracket-notation.

        Parameters:
            key (K): The key to register.
            value (T): The item to associate with the key.
        """
        self.add_registry_item(key, value)

    def __delitem__(self, key: K) -> None:
        """
        Allow deletion of items using the del statement.

        Parameters:
            key (K): The key of the item to remove.
        """
        self.del_registry_item(key)
