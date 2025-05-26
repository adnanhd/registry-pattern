"""Utilities for development."""

from functools import partial, reduce, wraps
from inspect import getmembers, isclass, ismodule
from types import ModuleType

from typing_compat import (
    Any,
    Callable,
    Generic,
    List,
    ParamSpec,
    TypeVar,
    get_args,
    runtime_checkable,
)

from ._validator import ValidationError, validate_class, validate_function

T = TypeVar("T")


def _def_checking(v: Any) -> Any:
    return v


def get_protocol(cls: type):
    """Get the protocol from the class."""
    assert isclass(cls), f"{cls} is not a class"
    assert Generic in cls.mro(), f"{cls} is not a generic class"

    type_arg = get_args(cls.__orig_bases__[0])[0]

    # If it's already a runtime_checkable Protocol, just return it
    if hasattr(type_arg, "_is_runtime_protocol") and type_arg._is_runtime_protocol:
        return type_arg

    # If it's a Protocol but not runtime_checkable, make it so
    if hasattr(type_arg, "_is_protocol") and type_arg._is_protocol:
        return runtime_checkable(type_arg)

    # Otherwise, it's just a regular type, so return it as is
    return type_arg


P = ParamSpec("P")
"""Type variable for the parameters."""
M = TypeVar("M")
"""Type variable for the middle value."""
R = TypeVar("R")
"""Type variable for the result value."""


def get_subclasses(cls: type) -> List[type]:
    """Get subclasses of a class."""
    return validate_class(cls).__subclasses__()


def get_module_members(
    module: ModuleType, ignore_all_keyword: bool = False
) -> List[Any]:
    """Get members of a module."""
    assert ismodule(module), f"{module} is not a module"
    if ignore_all_keyword or not hasattr(module, "__all__"):
        _names, members = zip(*getmembers(module))
    else:
        _members = filter(lambda m: isinstance(m, str), module.__all__)
        _members = filter(lambda m: hasattr(module, m), _members)
        members = tuple(map(lambda m: getattr(module, m), _members))

    result = []
    for member in filter(
        lambda m: hasattr(m, "__name__") and not m.__name__.startswith("_"), members
    ):
        try:
            result.append(validate_class(member))
        except ValidationError:
            pass
        try:
            result.append(validate_function(member))
        except ValidationError:
            pass
    return result


def compose_two_funcs(
    f: Callable[P, M], g: Callable[[M], R], wrap: bool = True
) -> Callable[P, R]:
    """Compose two functions"""
    assert callable(f), "First function must be callable"
    assert callable(g), "Second function must be callable"

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return g(f(*args, **kwargs))

    return wraps(f)(wrapper) if wrap else wrapper


def compose(*functions: Callable[..., Any], wrap: bool = True) -> Callable[..., Any]:
    """Compose functions"""
    composed_functions = reduce(
        partial(compose_two_funcs, wrap=False), reversed(functions)
    )
    return wraps(functions[0])(composed_functions) if wrap else composed_functions
