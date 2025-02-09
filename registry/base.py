"""
Base metaclass for registering classes and functions.
"""

from functools import lru_cache
from typing import Any
from typing import ClassVar
from typing import Container
from typing import Dict
from typing import Generic
from typing import Hashable
from typing import List
from typing import MutableMapping
from typing import Protocol
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from registry._validator import ValidationError

__all__ = [
    "RegistryError",
    "RegistryLookupError",
    "Registry",
    "MutableRegistry",
]


class RegistryError(KeyError):
    """Exception raised for registration errors."""


class RegistryLookupError(RegistryError):
    """Exception raised for lookup errors."""


class Stringable(Protocol):
    """A protocol for stringable objects."""

    def __str__(self) -> str: ...


K = TypeVar("K", bound=Hashable)
T = TypeVar("T")  # , bound=Stringable)


def _not_registered_error(cls: type, key: Hashable) -> None:
    """Raise an error for absent classes."""
    raise RegistryLookupError(f"{cls.__name__}: {key!r} not registered")


def _dup_registered_error(cls: type, key: Hashable) -> None:
    """Raise an error for duplicate registrations."""
    raise RegistryError(f"{cls.__name__}: {key!r} already registered")


class BaseRegistry(Container[T], Generic[K, T]):
    """Metaclass for managing registrations."""

    __orig_bases__: ClassVar[Tuple[Type, ...]]
    # Drop ClassVar to suppress mypy and pyright errors
    # That is: ClassVar cannot include type variable
    _repository: Union[Dict[K, T], MutableMapping[K, T]]

    def __contains__(self, key: Any) -> bool:
        return self.__class__._repository.keys().__contains__(self.validate_key(key))

    @classmethod
    # @lru_cache(maxsize=16, typed=False)
    def validate_key(cls, key: Any) -> K:
        """Get the lookup key."""
        return key

    @classmethod
    def has_registry_key(cls, value: K) -> bool:
        return cls.validate_key(value) in cls._repository.keys()

    @classmethod
    def validate_item(cls, value: T) -> T:
        """Validate a class."""
        return value

    @classmethod
    def has_registry_item(cls, value: T) -> bool:
        try:
            return cls.validate_item(value) in cls._repository.values()
        except ValidationError:
            return False

    @classmethod
    @lru_cache(maxsize=16, typed=False)
    def get_registry_item(cls, key: K) -> T:
        """Return a registered class."""
        lookup_key = cls.validate_key(key)
        if lookup_key not in cls._repository.keys():
            _not_registered_error(cls, key)  # raise an error
        return cls._repository[lookup_key]


class BaseMutableRegistry(BaseRegistry[K, T], Generic[K, T]):
    """Metaclass for managing mutable registrations."""

    @classmethod
    def add_registry_item(cls, key: K, value: T) -> None:
        """Register a class."""
        lookup_key = cls.validate_key(key)
        if lookup_key not in cls._repository.keys():
            cls._repository[lookup_key] = cls.validate_item(value)
        else:
            _dup_registered_error(cls, key)

    @classmethod
    def del_registry_item(cls, key: K) -> None:
        """Delete a registered class."""
        lookup_key = cls.validate_key(key)
        if lookup_key in cls._repository.keys():
            del cls._repository[lookup_key]
            cls.get_registry_item.cache_clear()
        else:
            _not_registered_error(cls, key)

    @classmethod
    def clear_registry(cls) -> None:
        cls._repository.clear()


class Registry(BaseRegistry[K, T], Generic[K, T]):
    def __getitem__(self, key: Any) -> T:
        # check if key is not of type K
        return self.get_registry_item(key)

    @classmethod
    def len(cls) -> int:
        """Return the number of registered classes."""
        return len(cls._repository)

    @classmethod
    def keys(cls) -> List[K]:
        """Return a list of registered class names."""
        return list(cls._repository.keys())

    @classmethod
    def values(cls) -> List[T]:
        """Return a list of registered classes."""
        return list(cls._repository.values())


class MutableRegistry(BaseMutableRegistry[K, T], Registry[K, T], Generic[K, T]):
    def __setitem__(self, key: K, value: Any) -> None:
        return self.add_registry_item(key, value)

    def __delitem__(self, key: K) -> None:
        return self.del_registry_item(key)
