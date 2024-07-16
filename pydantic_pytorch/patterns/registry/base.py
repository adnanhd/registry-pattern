"""
Base metaclass for registering classes and functions.
"""

import abc
from functools import lru_cache
from typing import Any, Hashable, Container, TypeVar, Iterator, ClassVar, Generic, Protocol, MutableMapping

__all__ = [
    'RegistryError',
    'BaseRegistry',
    'BaseMutableRegistry',
]


class RegistryError(KeyError):
    """Exception raised for registration errors."""


class RegistryLookupError(RegistryError):
    """Exception raised for lookup errors."""


class Stringable(Protocol):
    """A protocol for stringable objects."""

    def __str__(self) -> str:
        ...


K = TypeVar('K', bound=Hashable)
T = TypeVar('T', bound=Stringable)


class BaseRegistry(Container[T], Generic[K, T]):
    """Metaclass for managing registrations."""
    __orig_bases__: ClassVar[tuple[type, ...]]
    _repository: ClassVar[dict | MutableMapping]

    @classmethod
    def __init_subclass__(cls) -> None:
        """Initialize the subclass."""
        cls._repository = dict()

    @staticmethod
    def __absence__(key: K):
        """Raise an error for absent classes."""
        raise RegistryLookupError(f'{key} not registered')

    def __contains__(self, key: Any) -> bool:
        return self.keys().__contains__(self.get_lookup_key(key))

    def __getitem__(self, key: Any) -> T:
        # check if key is not of type K
        return self.get_registry_item(key)

    def __len__(self) -> int:
        """Return the number of registered classes."""
        return len(self._repository)

    def __iter__(self) -> Iterator[K]:
        """Iterate over the registered class names."""
        return iter(self.keys())

    @staticmethod
    @lru_cache(maxsize=16, typed=False)
    def get_lookup_key(key: K) -> K:
        """Get the lookup key."""
        return key

    @lru_cache(maxsize=16, typed=False)
    def get_registry_item(self, key: K) -> T:
        """Return a registered class."""
        lookup_key = self.get_lookup_key(key)
        if lookup_key not in self.keys():
            self.__absence__(key)  # raise an error
        return self._repository[lookup_key]

    @classmethod
    def keys(cls) -> list[K]:
        """Return a list of registered class names."""
        return list(cls._repository.keys())

    @classmethod
    def values(cls) -> list[T]:
        """Return a list of registered classes."""
        return list(cls._repository.values())


class BaseMutableRegistry(BaseRegistry[K, T], Generic[K, T]):
    """Metaclass for managing mutable registrations."""

    @staticmethod
    def __presence__(key: K) -> Exception:
        """Raise an error for duplicate registrations."""
        raise RegistryError(f'{key} already registered')

    def __setitem__(self, key: K, value: Any) -> None:
        return self.add_registry_item(key, value)

    def __delitem__(self, key: K) -> None:
        return self.del_registry_item(key)

    @classmethod
    def add_registry_item(cls, key: K, value: T) -> None:
        """Register a class."""
        lookup_key = cls.get_lookup_key(key)
        if lookup_key not in cls.keys():
            cls._repository[lookup_key] = cls._validate_item(value)
        else:
            cls.__presence__(key)

    def del_registry_item(self, key: K) -> None:
        """Delete a registered class."""
        lookup_key = self.get_lookup_key(key)
        if lookup_key not in self.keys():
            self.__absence__(key)
        else:
            del self._repository[lookup_key]

    @classmethod
    @abc.abstractmethod
    def _validate_item(cls, value: Any) -> T:
        """Validate a class."""
        raise NotImplementedError
