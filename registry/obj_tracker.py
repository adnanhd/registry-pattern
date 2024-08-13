"""Object registry pattern."""

from typing import Protocol, Any, TypeVar, Generic, get_args, Type, Tuple
from abc import ABCMeta
import weakref


class Stringable(Protocol):
    """A protocol for stringable objects."""

    def __str__(self) -> str:
        ...


T = TypeVar("T")


class ClassTracker(Generic[T], ABCMeta):
    """Metaclass for managing trackable registrations."""

    _artifacts: weakref.WeakSet[T]
    __slots__ = ()

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._artifacts = weakref.WeakSet[T]()

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        cls._artifacts.add(instance)
        return instance

    def __instancecheck__(cls, instance: T) -> bool:
        return instance in cls._artifacts

    def is_instance(cls, instance: T) -> bool:
        """Check if an instance."""
        return instance in cls._artifacts

    def get_instances(cls) -> weakref.WeakSet[T]:
        """Return the set of instances."""
        return weakref.WeakSet(cls._artifacts)
