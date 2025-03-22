r"""Object registry pattern.

This module implements a generic wrapper pattern that allows objects to be
wrapped with additional behavior. The core components include:
  - A `Wrappable` protocol that defines the required attributes for a wrapped object.
  - A `WrapperMeta` metaclass that automatically delegates special methods and
    creates dynamic properties for non-callable attributes.
  - A generic `Wrapped` class that uses the `WrapperMeta` to wrap an instance along
    with configuration parameters.

Doxygen Dot Graph for Wrapper Pattern:
----------------------------------------
\dot
digraph WrapperPattern {
    "type" -> "WrapperMeta";
    "WrapperMeta" -> "Wrapped";
    "Wrapped" -> "Wrappable";
}
\enddot
"""

from typing_compat import Any, Generic, Protocol, Tuple, Type, TypeVar, get_args

# -----------------------------------------------------------------------------
# Protocol Definitions
# -----------------------------------------------------------------------------


class Stringable(Protocol):
    """A protocol for objects that can be converted to a string."""

    def __str__(self) -> str: ...


class Wrappable(Protocol):
    """
    A protocol for wrapping objects.

    Classes conforming to this protocol must define the following attributes:
      - __wrapobj__: The wrapped object.
      - __wrapcfg__: A dictionary containing configuration options.
    """

    __wrapobj__: object
    __wrapcfg__: dict


# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------

T = TypeVar("T")


# -----------------------------------------------------------------------------
# Metaclass Definition
# -----------------------------------------------------------------------------


class WrapperMeta(type):
    """
    Metaclass for creating wrapper classes.

    This metaclass automatically delegates common special methods (e.g.,
    arithmetic, comparison, etc.) to the wrapped object (stored in __wrapobj__).
    In addition, during instance creation it dynamically creates properties for
    non-callable attributes of the wrapped object.
    """

    def __new__(
        mcs, name: str, bases: Tuple[Type, ...], attrs: dict
    ) -> "WrapperMeta":  # Type[Wrapped]
        # Create the new class using the standard type.__new__.
        cls: WrapperMeta = super().__new__(mcs, name, bases, attrs)

        # List of special method names to be delegated to the wrapped object.
        special_methods = [
            "__add__",
            "__sub__",
            "__mul__",
            "__truediv__",
            "__floordiv__",
            "__mod__",
            "__pow__",
            "__and__",
            "__or__",
            "__xor__",
            "__lshift__",
            "__rshift__",
            "__lt__",
            "__le__",
            "__eq__",
            "__ne__",
            "__gt__",
            "__ge__",
            "__round__",
            "__floor__",
            "__ceil__",
            "__trunc__",
            "__int__",
            "__float__",
            "__complex__",
            "__neg__",
            "__pos__",
            "__abs__",
            "__invert__",
            "__call__",
            "__getitem__",
            "__setitem__",
            "__delitem__",
            "__len__",
            "__iter__",
            "__contains__",
        ]

        def __delegate_method(name: str):
            """
            Create a method that delegates the special method call to __wrapobj__.

            Parameters:
                name (str): The name of the method to delegate.
            Returns:
                A callable that delegates the method call.
            """

            def method(self: Wrappable, *args, **kwargs):
                # Debug print can be removed or replaced by logging if desired.
                print("method", name)
                # Delegate the call to the wrapped object.
                return getattr(self.__wrapobj__, name)(*args, **kwargs)

            return method

        # Dynamically create and attach the delegate methods to the class.
        for method_name in special_methods:
            setattr(cls, method_name, __delegate_method(method_name))

        return cls

    def __call__(cls, *args, **kwargs) -> Wrappable:
        """
        Create a new instance of the wrapper class and dynamically generate properties.

        After the instance is created, iterate over its attributes (from __wrapobj__)
        and create properties for non-callable attributes to ensure they are properly
        delegated.

        Returns:
            The newly created wrapped instance.
        """
        instance: Wrappable = super().__call__(*args, **kwargs)

        def __delegate_property(name: str) -> property:
            """
            Create a property that delegates get, set, and delete to the wrapped object.

            Parameters:
                name (str): The attribute name to delegate.
            Returns:
                property: A property object that wraps the attribute access.
            """

            def property_getter(self: Wrappable):
                return getattr(self.__wrapobj__, name)

            def property_setter(self: Wrappable, value: Any):
                setattr(self.__wrapobj__, name, value)

            def property_deleter(self: Wrappable):
                delattr(self.__wrapobj__, name)

            return property(property_getter, property_setter, property_deleter)

        # Iterate over the attributes of the instance's wrapped object.
        for attr_name in dir(instance):
            # Skip magic methods and callable attributes.
            if attr_name.startswith("__") or callable(getattr(instance, attr_name)):
                continue
            # Dynamically attach the property to the class.
            setattr(cls, attr_name, __delegate_property(attr_name))
        return instance


# -----------------------------------------------------------------------------
# Wrapped Class Definition
# -----------------------------------------------------------------------------


class Wrapped(Generic[T], metaclass=WrapperMeta):
    """
    A generic class for wrapping objects.

    This class wraps an instance of another class along with its configuration.
    It provides a transparent delegation of attribute access and special method
    calls to the underlying wrapped object (__wrapobj__).

    Attributes:
        __wrapobj__ (T): The wrapped object instance.
        __wrapcfg__ (dict): A dictionary containing configuration settings.
    """

    __orig_bases__: Tuple[Type, ...]
    __wrapobj__: T
    __wrapcfg__: dict
    __slots__ = ("__wrapobj__", "__wrapcfg__")

    @classmethod
    def from_config(cls, config: dict) -> "Wrapped[T]":
        """
        Create an instance from a configuration dictionary.

        The configuration must be a dictionary that will be used to instantiate
        the underlying wrapped object. The wrapped object's type is inferred from
        the type argument of the generic base.

        Parameters:
            config (dict): Configuration options to instantiate the wrapped object.

        Returns:
            Wrapped[T]: A new instance of the Wrapped class.
        """
        # Ensure that the configuration is a dictionary.
        assert isinstance(config, dict), f"{config} is not a dictionary"
        # Retrieve the expected type from the original base generic parameters.
        (instance_class,) = get_args(cls.__orig_bases__[0])
        # Ensure that the extracted type is a proper class.
        assert isinstance(instance_class, type), f"{instance_class} is not a type"
        # Create and return a new wrapped instance.
        return cls(instance_class(**config), config)

    def __init__(self, instance: T, config: dict) -> None:
        """
        Initialize the wrapped instance.

        Parameters:
            instance (T): The instance to wrap.
            config (dict): Configuration options for the wrapped object.
        """
        self.__wrapobj__ = instance
        self.__wrapcfg__ = config

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to the wrapped object.

        If the attribute is one of the internal attributes (__wrapobj__ or __wrapcfg__),
        use the normal attribute access. Otherwise, delegate to __wrapobj__.

        Parameters:
            name (str): The attribute name.
        Returns:
            Any: The attribute value from the wrapped object.
        """
        if name in {"__wrapobj__", "__wrapcfg__"}:
            return self.__getattribute__(name)
        return getattr(self.__wrapobj__, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Delegate attribute setting to the wrapped object.

        For internal attributes (__wrapobj__ and __wrapcfg__), use the standard
        setting behavior. Otherwise, set the attribute on __wrapobj__.

        Parameters:
            name (str): The attribute name.
            value (Any): The value to set.
        """
        if name in {"__wrapobj__", "__wrapcfg__"}:
            super().__setattr__(name, value)
        else:
            setattr(self.__wrapobj__, name, value)

    def __delattr__(self, name: str) -> None:
        """
        Delegate attribute deletion to the wrapped object.

        Parameters:
            name (str): The attribute name.
        """
        if name in {"__wrapobj__", "__wrapcfg__"}:
            super().__delattr__(name)
        else:
            delattr(self.__wrapobj__, name)

    def __dir__(self) -> list:
        """
        Return the list of attributes of the wrapped object.

        Returns:
            list: The attribute names from __wrapobj__.
        """
        return dir(self.__wrapobj__)

    def __repr__(self) -> str:
        """
        Return the string representation of the wrapped object.

        This includes the wrapper's class name, the wrapped object's class name,
        and the configuration parameters.

        Returns:
            str: The string representation.
        """
        base_class = self.__class__.__name__
        wrapped_class = self.__wrapobj__.__class__.__name__
        params = ", ".join(f"{key}={val!r}" for key, val in self.__wrapcfg__.items())
        return f"{base_class}[{wrapped_class}]({params})"
