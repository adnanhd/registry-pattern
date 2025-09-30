"""Utility exceptions and helpers for registry mixins.

This module defines a small hierarchy of rich exceptions used throughout the
registry access/validation/mutation mixins, plus simple helpers.

Exceptions:
    ValidationError: Base class carrying `suggestions` and `context` metadata.
    CoercionError: Raised when coercion of values fails.
    ConformanceError: Raised when callables/classes violate required signatures.
    InheritanceError: Raised when classes fail required inheritance.
    RegistryKeyError: Key-related mapping errors with rich context.
    RegistryValueError: Value-related mapping errors with rich context.

Helpers:
    get_type_name(cls, qualname=False): Return a human-readable type name.
"""

import logging
from functools import partial, reduce, wraps
from inspect import getmembers, isbuiltin, isclass, isfunction, ismethod, ismodule
from types import ModuleType
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from typing_extensions import ParamSpec, get_args, runtime_checkable

T = TypeVar("T")


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation with structured context.

    Attributes:
        message: Human-readable error text.
        suggestions: List of short, imperative hints for remediation.
        context: Free-form key/value details safe to log and render.
    """

    def __init__(
        self,
        message: str,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.suggestions = suggestions or []
        self.context = context or {}
        super().__init__(self._build_enhanced_message())

    def _build_enhanced_message(self) -> str:
        """Embed key context and suggestions into the exception string."""
        lines = [self.message]

        if self.context:
            if "expected_type" in self.context and "actual_type" in self.context:
                lines.append(f"  Expected: {self.context['expected_type']}")
                lines.append(f"  Actual: {self.context['actual_type']}")
            if "artifact_name" in self.context:
                lines.append(f"  Artifact: {self.context['artifact_name']}")

        if self.suggestions:
            lines.append("  Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"    • {suggestion}")

        return "\n".join(lines)


class CoercionError(ValidationError):
    """Raised when a value cannot be coerced into the required representation."""


class ConformanceError(ValidationError):
    """Raised when a callable or class does not conform to required signatures."""


class InheritanceError(ValidationError):
    """Raised when a class does not inherit from a required base."""


class RegistryKeyError(ValidationError, KeyError):
    """Raised for key-related mapping errors with rich context attached."""


class RegistryValueError(ValidationError, ValueError):
    """Raised for value-related mapping errors with rich context attached."""


def get_type_name(cls: type, qualname: bool = False) -> str:
    """Return a readable name for a type.

    Args:
        cls: The class or type object.
        qualname: If True, return the qualified name when available.

    Returns:
        The type's `__qualname__`, `__name__`, or a string fallback.
    """
    if not isclass(cls):
        raise ValidationError(f"{cls} is not a class")
    if qualname and hasattr(cls, "__qualname__"):
        return getattr(cls, "__qualname__")
    elif hasattr(cls, "__name__"):
        return getattr(cls, "__name__")
    else:
        return str(cls)


def get_func_name(func: Callable[..., Any], qualname: bool = False) -> str:
    """Return a readable name for a function.

    Args:
        func: The function object.
        qualname: If True, return the qualified name when available.

    Returns:
        The function's `__qualname__`, `__name__`, or a string fallback.
    """
    if not callable(func):
        raise ValidationError(f"{func} is not callable")
    while hasattr(func, "__wrapped__"):
        func = getattr(func, "__wrapped__")
    return getattr(func, "__qualname__" if qualname else "__name__", str(func))


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
    if isclass(cls):
        return cls.__subclasses__()
    raise ValidationError(f"{cls} is not a class")


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
        if isclass(member):
            result.append(member)
        elif (isfunction(member) or isbuiltin(member)) and callable(member):
            result.append(member)
        elif ismethod(member) and callable(member):
            logging.info(f"Method {member.__name__} found")
            # result.append(member)
        else:
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
