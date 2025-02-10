"""Utilities for development."""

import sys
from functools import partial
from functools import reduce
from functools import wraps
from inspect import getmembers
from inspect import ismodule
from types import ModuleType
from typing import Any
from typing import Callable
from typing import Generic
from typing import List
from typing import TypeVar
from typing import get_args
from typing import runtime_checkable

from ._validator import validate_class

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

T = TypeVar("T")


def _def_checking(v: T) -> T:
    return v


def get_protocol(cls: type):
    """Get the protocol from the class."""
    assert issubclass(cls, Generic), f"{cls} is not a generic class"
    return runtime_checkable(get_args(cls.__orig_bases__[0])[0])  # type: ignore


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
        members = (getattr(module, name) for name in module.__all__)

    return list(filter(lambda m: hasattr(m, "__name__"), members))


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
