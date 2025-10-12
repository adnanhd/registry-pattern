r"""Function registry with optional signature checks."""

from __future__ import annotations

import inspect
import logging
import sys
from abc import ABC
from typing import (
    Any,
    Callable,
    ClassVar,
    Hashable,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
)

from typing_extensions import ParamSpec, get_args

from .mixin import RegistryFactorizorMixin
from .storage import RemoteStorageProxy, ThreadSafeLocalStorage
from .utils import (
    ConformanceError,
    ValidationError,
    _validate_function_signature,
    get_func_name,
    get_module_members,
    get_type_name,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _validate_function(func: Any) -> Callable[..., Any]:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Validating function object: %r", func)
    if (inspect.isfunction(func) or inspect.isbuiltin(func)) and callable(func):
        return func
    raise ValidationError(
        f"{func} is not a function",
        [
            "Pass a function object, not a value",
            "Define it with 'def' or ensure it is callable",
        ],
        {
            "expected_type": "function",
            "actual_type": type(func).__name__,
            "artifact_name": get_func_name(func),
        },
    )


class FunctionalRegistry(RegistryFactorizorMixin[Hashable, Callable[P, R]], ABC):
    """Registry for functions, with optional strict signature validation.

    Strict mode: when enabled on subclass, enforce `Callable[Params, Return]`
    declared by the generic parameterization of the subclass.
    """

    _repository: MutableMapping[Hashable, Callable[P, R]]
    _strict: ClassVar[bool] = False
    __orig_bases__: ClassVar[Tuple[type, ...]]  # used to extract Callable[P, R]
    __slots__: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def _get_mapping(cls):
        return cls._repository

    @classmethod
    def __class_getitem__(cls, params: Any) -> Any:
        # Keep typing.ParameterSpec generics working across Python versions
        if sys.version_info < (3, 10):
            args, kwargs = params
            params = (Tuple[tuple(args)], kwargs)
        return super().__class_getitem__(params)  # type: ignore

    @classmethod
    def __init_subclass__(
        cls,
        strict: bool = False,
        scheme_namespace: Optional[str] = None,
        logic_namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(scheme_namespace=scheme_namespace, **kwargs)
        if logic_namespace is None:
            cls._repository = ThreadSafeLocalStorage[Hashable, Callable[P, R]]()
        else:
            cls._repository = RemoteStorageProxy[Hashable, Callable[P, R]](
                namespace=logic_namespace
            )
        cls._strict = strict
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Initialized FunctionalRegistry subclass %s (strict=%s, logic_namespace=%s, schema_namespace=%s)",
                cls.__name__,
                strict,
                logic_namespace,
                scheme_namespace,
            )

    @classmethod
    def _internalize_artifact(cls, value: Any) -> Callable[P, R]:
        fn = _validate_function(value)
        if cls._strict:
            param, ret = get_args(cls.__orig_bases__[0])
            if sys.version_info < (3, 10):
                param = list(get_args(param))
            expected = Callable[param, ret]  # type: ignore
            fn = _validate_function_signature(fn, expected_callable_alias=expected)
        return super()._internalize_artifact(fn)

    @classmethod
    def _identifier_of(cls, item: Callable[P, R]) -> Hashable:
        return get_func_name(_validate_function(item))

    @classmethod
    def register_module_functions(cls, module: Any, raise_error: bool = True) -> Any:
        members = get_module_members(module)
        ok, fail = 0, 0
        for obj in members:
            name = getattr(obj, "__name__", str(obj))
            try:
                _validate_function(obj)
                cls.register_artifact(obj)
                ok += 1
            except (ConformanceError, ValidationError) as e:
                fail += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Skipping %s: %s", name, e)
                if raise_error:
                    if hasattr(e, "context"):
                        e.context.update(
                            {
                                "module_name": getattr(module, "__name__", str(module)),
                                "operation": "register_module_functions",
                            }
                        )
                    raise
        logger.info(
            "%s: registered %d function(s), %d failed", get_type_name(cls), ok, fail
        )
        return module
