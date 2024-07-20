"""Functional registry for registering functions."""

from typing import TypeVar, ClassVar, Callable, Any, Generic, Hashable, get_args
from typing_extensions import ParamSpec
from functools import lru_cache
from pydantic import validate_call
from ._validator import validate_function, validate_function_parameters, ValidationError
from ._dev_utils import get_module_members
from .base import BaseMutableRegistry
from ._dev_utils import compose

R = TypeVar("R")
P = ParamSpec("P")


# noka: W0223 # pylint: disable=abstract-method
class _BaseFunctionalRegistry(BaseMutableRegistry[Hashable, Callable[P, R]]):
    """Metaclass for registering functions."""

    __orig_bases__: ClassVar[tuple[type, ...]]
    # __check_type__: ClassVar[Callable[P, R]]
    # _registrar: ClassVar[dict[str, Callable[P, R]]]
    ignore_structural_subtyping: ClassVar[bool] = False
    __slots__ = ()

    def register_function(self, func: Callable[P, R]) -> Callable[P, R]:
        """Register a function."""
        self.add_registry_item(func.__name__, func)
        return func

    def unregister_function(self, func: Callable[P, R]) -> Callable[P, R]:
        """Unregister a function."""
        self.del_registry_item(func.__name__)
        return func

    def register_module_functions(self, module):
        """Register all functions in a given module."""
        members = filter(callable, get_module_members(module))
        for obj in members:
            try:
                self.register_function(obj)  # type: ignore
            except ValidationError:
                pass
        return module


class FunctionalRegistry(_BaseFunctionalRegistry[P, R], Generic[P, R]):
    """Functional registry for registering functions."""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        param, ret = get_args(cls.__orig_bases__[0])

        if not isinstance(param, list):
            param = [param]

        callable_type = Callable[param, ret]  # type: ignore
        validators: list[Callable[..., Any]] = [validate_function]

        if not cls.ignore_structural_subtyping:

            @validators.append
            def _validate_function_parameters(value: Any) -> Callable[P, R]:
                return validate_function_parameters(
                    validate_function(value), expected_type=callable_type
                )

        cls._validate_item = compose(*validators)


class PyFunctionalRegistry(_BaseFunctionalRegistry[P, R], Generic[P, R]):
    """Functional registry for registering functions."""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        param, ret = get_args(cls.__orig_bases__[0])

        if not isinstance(param, list):
            param = [param]

        callable_type = Callable[param, ret]  # type: ignore

        validators: list[Callable[..., Any]] = [validate_function]

        if not cls.ignore_structural_subtyping:

            def _validate_function_parameters(value: Any) -> Callable[P, R]:
                return validate_function_parameters(
                    validate_function(value), expected_type=callable_type
                )

            validators.append(_validate_function_parameters)

        cls._validate_item = compose(*validators)

    @lru_cache(maxsize=16, typed=False)
    def get_registry_item(self, key: str) -> Callable[P, R]:
        """Return a registered function."""
        return validate_call(super().get_registry_item(key))
