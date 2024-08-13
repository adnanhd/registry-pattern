"""Functional registry for registering functions."""

from typing import (
    TypeVar,
    ClassVar,
    Callable,
    Any,
    Generic,
    Hashable,
    get_args,
    Type,
    Tuple,
    List,
)
from typing_extensions import ParamSpec, Union
from functools import lru_cache, partial
from warnings import warn

from ._validator import (
    validate_function,
    validate_function_parameters,
    ValidationError,
    ConformanceError,
)
from ._dev_utils import get_module_members, _def_checking, compose
from .base import MutableRegistry

R = TypeVar("R")
P = ParamSpec("P")


# noka: W0223 # pylint: disable=abstract-method
class FunctionalRegistry(MutableRegistry[Hashable, Callable[P, R]]):
    """Metaclass for registering functions."""

    runtime_conformance_checking: Callable[
        [Callable[P, R]], Callable[P, R]
    ] = _def_checking
    __orig_bases__: ClassVar[Tuple[Type, ...]]
    __slots__ = ()

    @classmethod
    def __init_subclass__(cls, strict: bool = False, coercion: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if coercion:
            warn("Coersion not yet supported! Thus, it has no effect :(")
        cls._repository = dict()
        param, ret = get_args(cls.__orig_bases__[0])

        if not isinstance(param, list):
            param = [param]

        callable_type: Type[Callable[P, R]] = Callable[param, ret]  # type: ignore
        # validators: List[Callable[..., Any]] = [validate_function]

        if strict:
            cls.runtime_conformance_checking = partial(
                validate_function_parameters, expected_type=callable_type
            )

    @classmethod
    def validate_item(cls, value: Callable[P, R]) -> Callable[P, R]:
        validate_function(value)
        return cls.runtime_conformance_checking(value)

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        try:
            validate_function(value)
        except ValidationError:
            print("no validation")
            return False

        try:
            cls.runtime_conformance_checking(value)
        except ConformanceError:
            print("no conformation")
            return False

        return True
        cls._validate_item = compose(*validators)

    @classmethod
    def register_function(cls, func: Callable[P, R]) -> Callable[P, R]:
        """Register a function."""
        cls.add_registry_item(func.__name__, func)
        return func

    @classmethod
    def unregister_function(cls, func: Callable[P, R]) -> Callable[P, R]:
        """Unregister a function."""
        cls.del_registry_item(func.__name__)
        return func

    @classmethod
    def register_module_functions(cls, module, raise_error: bool = True):
        """Register all functions in a given module."""
        members = filter(callable, get_module_members(module))
        for obj in members:
            try:
                cls.register_function(obj)  # type: ignore
            except ValidationError as e:
                if raise_error:
                    raise ValidationError(e)
                else:
                    print(e)
        return module
