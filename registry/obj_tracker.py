"""Object registry pattern."""

import sys
import weakref
from abc import ABCMeta
from typing import TypeVar, TYPE_CHECKING, Generic, Protocol


class Stringable(Protocol):
    """A protocol for stringable objects."""

    def __str__(self) -> str: ...


T = TypeVar("T")

if TYPE_CHECKING or sys.version_info >= (3, 9):
    from weakref import WeakSet

    WeakSetT = weakref.WeakSet[T]  # Use subscriptable WeakSet for Python 3.9+
else:
    WeakSetT = weakref.WeakSet  # Unsubscriptable WeakSet for Python < 3.9


class ClassTracker(Generic[T], ABCMeta):
    """Metaclass for managing trackable registrations."""

    _artifacts: WeakSetT
    __slots__ = ()

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        cls._artifacts = WeakSetT()

    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        cls._artifacts.add(instance)
        return instance

    def __instancecheck__(cls, instance: T) -> bool:
        return instance in cls._artifacts

    def is_instance(cls, instance: T) -> bool:
        """Check if an instance."""
        return instance in cls._artifacts

    def get_instances(cls) -> WeakSetT:
        """Return the set of instances."""
        return weakref.WeakSet(cls._artifacts)
