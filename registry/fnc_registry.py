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

from .mixin import MutableRegistryValidatorMixin
from .storage import RemoteStorageProxy, ThreadSafeLocalStorage
from .utils import (
    ConformanceError,
    ValidationError,
    get_func_name,
    get_module_members,
    get_type_name,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def _validate_function(func: Callable[..., Any]) -> Callable[..., Any]:
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
            "artifact_name": (
                get_func_name(func) if hasattr(func, "__name__") else str(func)
            ),
        },
    )


def _validate_function_parameters(
    func: Callable[P, R], expected_callable: Any
) -> Callable[P, R]:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Validating signature of %s against %s",
            get_func_name(func),
            expected_callable,
        )

    try:
        sig = inspect.signature(func)
        exp_args, exp_ret = get_args(expected_callable)
    except Exception as e:
        raise ValidationError(
            f"Cannot analyze function type: {e}",
            ["Ensure expected type is a valid typing.Callable"],
            {"artifact_name": get_func_name(func)},
        )

    errors: list[str] = []
    hints: list[str] = []

    if len(sig.parameters) != len(exp_args):
        errors.append(
            f"Parameter count mismatch: expected {len(exp_args)}, got {len(sig.parameters)}"
        )
        hints.append(f"Use exactly {len(exp_args)} parameters")

    for param, exp in zip(sig.parameters.values(), exp_args):
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            errors.append(f"Parameter {param.name} missing type annotation")
            hints.append(f"Annotate: {param.name}: {get_type_name(exp)}")
        elif ann != exp:
            errors.append(
                f"Parameter {param.name} type mismatch: expected {get_type_name(exp)}, got {get_type_name(ann)}"
            )
            hints.append(f"Change to: {param.name}: {get_type_name(exp)}")

    ret_ann = sig.return_annotation
    if ret_ann is inspect.Signature.empty:
        errors.append("Missing return type annotation")
        hints.append(f"Annotate return: -> {get_type_name(exp_ret)}")
    elif ret_ann != exp_ret:
        errors.append(
            f"Return type mismatch: expected {get_type_name(exp_ret)}, got {get_type_name(ret_ann)}"
        )
        hints.append(f"Change return to: -> {get_type_name(exp_ret)}")

    if errors:
        raise ConformanceError(
            "Function signature validation failed:\n"
            + "\n".join(f"  • {e}" for e in errors),
            hints,
            {
                "artifact_name": get_func_name(func),
                "expected_type": str(expected_callable),
                "actual_type": str(sig),
            },
        )
    return func


class FunctionalRegistry(MutableRegistryValidatorMixin[Hashable, Callable[P, R]], ABC):
    """Registry for functions, with optional strict signature validation.

    Strict mode: when enabled on subclass, enforce `Callable[Params, Return]`
    declared by the generic parameterization of the subclass.
    """

    _repository: MutableMapping[Hashable, Callable[P, R]]
    _strict: bool = False
    __orig_bases__: ClassVar[Tuple[type, ...]]  # used to extract Callable[P, R]
    __slots__ = ()

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
        proxy_namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if proxy_namespace is None:
            cls._repository = ThreadSafeLocalStorage[Hashable, Callable[P, R]]()
        else:
            cls._repository = RemoteStorageProxy[Hashable, Callable[P, R]](
                namespace=proxy_namespace
            )
        cls._strict = strict
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Initialized FunctionalRegistry subclass %s (strict=%s)",
                cls.__name__,
                strict,
            )

    @classmethod
    def _internalize_artifact(cls, value: Any) -> Callable[P, R]:
        fn = _validate_function(value)
        if cls._strict:
            param, ret = get_args(cls.__orig_bases__[0])
            if sys.version_info < (3, 10):
                param = list(get_args(param))
            expected = Callable[param, ret]  # type: ignore
            fn = _validate_function_parameters(fn, expected_callable=expected)
        return fn

    @classmethod
    def _externalize_artifact(cls, value: Callable[P, R]) -> Callable[P, R]:
        return value

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
        logger.info("%s: registered %d function(s), %d failed", cls.__name__, ok, fail)
        return module
