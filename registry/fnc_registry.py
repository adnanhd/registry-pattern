r"""Registry for functions, with optional signature checks."""

from __future__ import annotations

import inspect
import logging
import sys
from abc import ABC
from types import ModuleType
from typing import (
    Any,
    Callable,
    ClassVar,
    Hashable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

from pydantic import BaseModel
from typing_extensions import ParamSpec, get_args

from .mixin import ContainerMixin
from .storage import RemoteStorageProxy, ThreadSafeLocalStorage
from .utils import (
    ConformanceError,
    ValidationError,
    _validate_function_signature,
    get_func_name,
    get_module_members,
    get_module_name,
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


class FunctionalRegistry(ContainerMixin[Hashable, Callable[P, R]], ABC):
    """Registry for functions, with optional strict signature validation.

    Supports:
      - Registration with optional explicit params_model
      - Auto-extraction of params_model from function signature
      - Strict mode for signature type checking
      - Multi-repo DI container via ContainerMixin
    """

    _repository: MutableMapping[Hashable, Callable[P, R]]
    _strict: ClassVar[bool] = False
    __orig_bases__: ClassVar[Tuple[type, ...]]
    __slots__: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def _get_mapping(cls) -> MutableMapping[Hashable, Callable[P, R]]:
        return cls._repository

    @classmethod
    def __class_getitem__(cls, params: Any) -> Any:
        # Keep typing.ParameterSpec generics working across Python versions
        if sys.version_info < (3, 10):
            args, kwargs = params
            params = (Tuple[tuple(args)], kwargs)
        return super().__class_getitem__(params)  # type: ignore[misc]

    @classmethod
    def __init_subclass__(
        cls,
        strict: bool = False,
        logic_namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize a FunctionalRegistry subclass.

        Args:
            strict: Enforce signature type checking.
            logic_namespace: Optional remote storage namespace.
        """
        super().__init_subclass__(**kwargs)
        if logic_namespace is None:
            cls._repository = ThreadSafeLocalStorage[Hashable, Callable[P, R]]()
        else:
            cls._repository = RemoteStorageProxy[Hashable, Callable[P, R]](
                namespace=logic_namespace
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
        """Validate and internalize a function artifact."""
        fn = _validate_function(value)
        if cls._strict:
            param, ret = get_args(cls.__orig_bases__[0])
            if sys.version_info < (3, 10):
                param = list(get_args(param))
            expected = Callable[param, ret]  # type: ignore[valid-type]
            _validate_function_signature(fn, expected_callable_alias=expected)
        return super()._internalize_artifact(fn)

    @classmethod
    def _identifier_of(cls, item: Callable[P, R]) -> Hashable:
        """Return the identifier (function name) for an artifact."""
        return get_func_name(_validate_function(item))

    # -------------------------------------------------------------------------
    # Registration with params_model support
    # -------------------------------------------------------------------------

    @overload
    @classmethod
    def register_artifact(
        cls,
        artifact: Callable[P, R],
        *,
        params_model: Optional[Type[BaseModel]] = None,
    ) -> Callable[P, R]:
        """Register a function artifact with optional params_model."""
        ...

    @overload
    @classmethod
    def register_artifact(
        cls,
        *,
        params_model: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """Decorator form: register a function with optional params_model."""
        ...

    @classmethod
    def register_artifact(
        cls,
        artifact: Optional[Callable[P, R]] = None,
        *,
        params_model: Optional[Type[BaseModel]] = None,
    ) -> Union[Callable[P, R], Any]:
        """Register a function artifact with optional explicit params_model.

        Can be used as a decorator or called directly.

        Args:
            artifact: The function to register.
            params_model: Optional Pydantic model for parameter validation.
                If not provided, auto-extracted from function signature.

        Returns:
            The registered function (or decorator if artifact is None).

        Example::

            @MyRegistry.register_artifact(params_model=MyParams)
            def my_func(x: int) -> str: ...

            # Or auto-extract:
            @MyRegistry.register_artifact
            def my_func(x: int) -> str: ...
        """

        def _do_register(art: Callable[P, R]) -> Callable[P, R]:
            # Save explicit params_model before registration
            if params_model is not None:
                identifier = get_func_name(_validate_function(art))
                cls._save_scheme(identifier, params_model)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Saved explicit params_model for %s: %s",
                        identifier,
                        params_model.__name__,
                    )

            # Use parent's register_artifact for the actual registration
            return cast(
                Callable[P, R], super(FunctionalRegistry, cls).register_artifact(art)
            )

        if artifact is None:
            # Decorator form: @register_artifact(params_model=...)
            return _do_register
        else:
            # Direct call: register_artifact(my_func, params_model=...)
            return _do_register(artifact)

    # -------------------------------------------------------------------------
    # Module Registration
    # -------------------------------------------------------------------------

    @classmethod
    def register_module_functions(
        cls,
        module: ModuleType,
        raise_error: bool = True,
    ) -> ModuleType:
        """Register all functions from a module.

        Args:
            module: The module to scan.
            raise_error: Whether to raise on registration failure.

        Returns:
            The module (for chaining).
        """
        assert isinstance(module, ModuleType), (
            f"Expected ModuleType, got {type(module)}"
        )
        members: List[Any] = get_module_members(module)
        ok, fail = 0, 0
        for obj in members:
            if not (inspect.isfunction(obj) or inspect.isbuiltin(obj)):
                continue
            name = getattr(obj, "__name__", str(obj))
            try:
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
                                "module_name": get_module_name(module),
                                "operation": "register_module_functions",
                            }
                        )
                    raise
        logger.info(
            "%s: registered %d function(s), %d failed", get_type_name(cls), ok, fail
        )
        return module
