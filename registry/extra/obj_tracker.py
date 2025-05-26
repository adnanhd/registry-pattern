r"""Object registry pattern.

This module provides a metaclass, `ClassTracker`, which automatically tracks all
instances of classes that use it. Instances are stored using a weak reference set so
that they do not prevent garbage collection.

Doxygen Dot Graph for ClassTracker:
-------------------------------------
\dot
digraph ClassTracker {
    "ABCMeta" -> "ClassTracker";
    "Generic" -> "ClassTracker";
}
\enddot
"""

import sys
import weakref
from abc import ABCMeta

from typing_compat import TYPE_CHECKING, Generic, Protocol, TypeVar

# -----------------------------------------------------------------------------
# Protocol Definitions
# -----------------------------------------------------------------------------


class Stringable(Protocol):
    """Protocol for objects that can be converted to a string."""

    def __str__(self) -> str: ...


# -----------------------------------------------------------------------------
# Type Variables and Conditional Imports
# -----------------------------------------------------------------------------

T = TypeVar("T")

# For Python 3.9+, WeakSet is subscriptable. Otherwise, we fall back to the unsubscriptable version.
if TYPE_CHECKING or sys.version_info >= (3, 9):
    pass

    WeakSetT = weakref.WeakSet[T]  # Type alias for a subscriptable WeakSet.
else:
    WeakSetT = weakref.WeakSet  # Fallback for earlier Python versions.


# -----------------------------------------------------------------------------
# Metaclass Definition
# -----------------------------------------------------------------------------


class ClassTracker(Generic[T], ABCMeta):
    r"""
    Metaclass for managing trackable registrations.

    This metaclass automatically tracks every instance created for classes that use it.
    Each instance is stored in a weak reference set (_artifacts) so that the registry
    does not prevent garbage collection.

    Doxygen Dot Graph:
    \dot
    digraph ClassTracker {
        "ABCMeta" -> "ClassTracker";
        "Generic" -> "ClassTracker";
    }
    \enddot
    """

    # WeakSet to store instances. The use of __slots__ ensures no instance dictionary is created.
    _artifacts: WeakSetT
    __slots__ = ()

    def __init__(cls, name: str, bases: tuple, attrs: dict) -> None:
        """
        Initialize the class and set up the instance tracking container.

        Parameters:
            name (str): The name of the class.
            bases (tuple): Base classes for the class.
            attrs (dict): Attribute dictionary of the class.
        """
        super().__init__(name, bases, attrs)
        # Initialize the _artifacts attribute as a new weak reference set.
        cls._artifacts = WeakSetT()

    def __call__(cls, *args, **kwargs) -> T:
        """
        Create a new instance of the class and register it.

        After creating the instance, it is added to the _artifacts weak set.

        Returns:
            T: The newly created instance.
        """
        instance = super().__call__(*args, **kwargs)
        cls._artifacts.add(instance)
        return instance

    def __instancecheck__(cls, instance: T) -> bool:
        """
        Custom instance check that verifies if an object is in the tracked artifacts.

        Parameters:
            instance (T): The instance to check.

        Returns:
            bool: True if the instance is tracked; False otherwise.
        """
        return instance in cls._artifacts

    def is_instance(cls, instance: T) -> bool:
        """
        Check if a given object is an instance that has been tracked.

        Parameters:
            instance (T): The object to verify.

        Returns:
            bool: True if the object is in the registry; otherwise, False.
        """
        return instance in cls._artifacts

    def get_instances(cls) -> WeakSetT:
        """
        Get a copy of the set of currently tracked instances.

        Returns:
            WeakSetT: A new weak reference set containing all tracked instances.
        """
        # Return a new WeakSet containing the current artifacts.
        return weakref.WeakSet(cls._artifacts)
