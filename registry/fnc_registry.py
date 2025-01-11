"""Functional registry for registering functions."""

import sys

from functools import partial
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Hashable
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Iterable
from typing import get_args
from warnings import warn

from typing_extensions import ParamSpec

from ._dev_utils import _def_checking
from ._dev_utils import compose
from ._dev_utils import get_module_members
from ._validator import ConformanceError
from ._validator import ValidationError
from ._validator import validate_function
from ._validator import validate_function_parameters
from .base import MutableRegistry

R = TypeVar("R")
P = ParamSpec("P")


# noka: W0223 # pylint: disable=abstract-method
class FunctionalRegistry(MutableRegistry[Hashable, Callable[P, R]]):
    """Metaclass for registering functions."""

    runtime_conformance_checking: Callable[[Callable[P, R]], Callable[P, R]] = (
        _def_checking
    )
    __orig_bases__: ClassVar[Tuple[Type, ...]]
    __slots__ = ()

    @classmethod
    def __class_getitem__(cls, params):
        if sys.version_info < (3, 10):
            args, kwargs = params
            params = (Tuple[tuple(args)], kwargs)
        return super().__class_getitem__(params)  # type: ignore

    @classmethod
    def __init_subclass__(cls, strict: bool = False, coercion: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        if coercion:
            warn("Coersion not yet supported! Thus, it has no effect :(")
        cls._repository = dict()
        param, ret = get_args(cls.__orig_bases__[0])

        if sys.version_info < (3, 10):
            param = list(get_args(param))

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
        members: Iterable[Callable] = filter(callable, get_module_members(module))
        for obj in members:
            cls.register_function(obj)
        return module
