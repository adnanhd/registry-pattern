"""Object registry pattern."""

from typing import Any
from typing import Generic
from typing import Protocol
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import get_args


class Stringable(Protocol):
    """A protocol for stringable objects."""

    def __str__(self) -> str: ...


T = TypeVar("T")


class Wrappable(Protocol):
    """A protocol for wrapping objects."""

    __wrapobj__: object
    __wrapcfg__: dict


class WrapperMeta(type):
    """Metaclass for creating wrapper classes."""

    def __new__(mcs, name, bases, attrs):
        cls: Type[Wrappable] = super().__new__(mcs, name, bases, attrs)

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
            def method(self: Wrappable, *args, **kwargs):
                print("method", name)
                return getattr(self.__wrapobj__, name)(*args, **kwargs)

            return method

        # create methods for all special methods
        for name in special_methods:
            setattr(cls, name, __delegate_method(name))

        return cls

    def __call__(cls, *args, **kwargs):
        instance: Wrappable = super().__call__(*args, **kwargs)

        def __delegate_property(name: str) -> property:
            def property_getter(self: Wrappable):
                return getattr(self.__wrapobj__, name)

            def property_setter(self: Wrappable, value: Any):
                setattr(self.__wrapobj__, name, value)

            def property_deleter(self: Wrappable):
                delattr(self.__wrapobj__, name)

            return property(property_getter, property_setter, property_deleter)

        # create properties for all non-callable attributes
        for name in dir(instance):
            if name.startswith("__") or callable(getattr(instance, name)):
                continue
            setattr(cls, name, __delegate_property(name))
        return instance


class Wrapped(Generic[T], metaclass=WrapperMeta):
    """A class for wrapping objects."""

    __orig_bases__: Tuple[Type, ...]
    __wrapobj__: T
    __wrapcfg__: dict
    __slots__ = ("__wrapobj__", "__wrapcfg__")

    @classmethod
    def from_config(cls, config):
        """Create an instance from a configuration."""
        assert isinstance(config, dict), f"{config} is not a dictionary"
        (instance_class,) = get_args(cls.__orig_bases__[0])
        assert isinstance(instance_class, type), f"{instance_class} is not a type"
        return cls(instance_class(**config), config)

    def __init__(self, instance, config):
        self.__wrapobj__ = instance
        self.__wrapcfg__ = config

    def __getattr__(self, name):
        if name in {"__wrapobj__", "__wrapcfg__"}:
            return self.__getattribute__(name)
        else:
            return getattr(self.__wrapobj__, name)

    def __setattr__(self, name, value):
        if name in {"__wrapobj__", "__wrapcfg__"}:
            super().__setattr__(name, value)
        else:
            setattr(self.__wrapobj__, name, value)

    def __delattr__(self, name):
        if name in {"__wrapobj__", "__wrapcfg__"}:
            super().__delattr__(name)
        else:
            delattr(self.__wrapobj__, name)

    def __dir__(self):
        return dir(self.__wrapobj__)

    def __repr__(self):
        base_class = self.__class__.__name__
        wrapped_class = self.__wrapobj__.__class__.__name__
        params = ", ".join(f"{key}={val!r}" for key, val in self.__wrapcfg__.items())
        return f"{base_class}[{wrapped_class}]({params})"
