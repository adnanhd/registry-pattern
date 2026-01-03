r"""Type (class) registry with optional inheritance/protocol checks."""

from __future__ import annotations

import inspect
import logging
from abc import ABC
from types import ModuleType
from typing import (
    Any,
    ClassVar,
    Generic,
    Hashable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from pydantic import BaseModel

from .mixin import ContainerMixin
from .storage import RemoteStorageProxy, ThreadSafeLocalStorage
from .utils import (
    ConformanceError,
    InheritanceError,
    ValidationError,
    _validate_function_signature,
    get_module_members,
    get_module_name,
    get_object_name,
    get_protocol,
    get_type_name,
)

logger = logging.getLogger(__name__)

Cls = TypeVar("Cls")


def _validate_class(obj: Any) -> type:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Validating class object: %r", obj)
    if inspect.isclass(obj):
        return obj
    raise ValidationError(
        f"{obj} is not a class",
        ["Pass a class, not an instance"],
        {
            "expected_type": "class",
            "actual_type": get_type_name(type(obj)),
            "artifact_name": str(obj),
        },
    )


def _validate_class_hierarchy(subcls: type, /, abc_class: type) -> type:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Checking %s subclass of %s",
            get_type_name(subcls),
            get_type_name(abc_class),
        )
    if not issubclass(subcls, abc_class):
        raise InheritanceError(
            f"{get_type_name(subcls)} is not a subclass of {get_type_name(abc_class)}",
            [f"Inherit from {get_type_name(abc_class)}"],
            {
                "expected_type": get_type_name(abc_class),
                "actual_type": get_type_name(subcls),
                "artifact_name": get_type_name(subcls),
            },
        )
    return subcls


def _validate_class_structure(subcls: type, /, exp_type: type) -> type:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Validating structure of %s against protocol %s",
            get_type_name(subcls),
            get_type_name(exp_type),
        )
    missing: List[str] = []
    sig_errs: List[str] = []
    for name in dir(exp_type):
        if name.startswith("_"):
            continue
        if not hasattr(subcls, name):
            missing.append(name)
            continue
        exp_attr = getattr(exp_type, name)
        sub_attr = getattr(subcls, name)
        if callable(exp_attr):
            if not callable(sub_attr):
                sig_errs.append(f"Attribute '{name}' should be callable")
                continue
            try:
                _validate_function_signature(sub_attr, exp_attr)
            except ConformanceError as e:
                sig_errs.append(f"{name}: {e.message}")
    if missing or sig_errs:
        parts: List[str] = []
        hints: List[str] = []
        if missing:
            parts.append(f"Missing methods: {', '.join(missing)}")
            hints.extend([f"Add method: {m}(...) -> ..." for m in missing])
        if sig_errs:
            parts.append("Signature mismatches:")
            parts.extend(f"  {e}" for e in sig_errs)
            hints.append("Match parameter and return annotations")
        raise ConformanceError(
            f"Class {get_type_name(subcls)} does not conform to protocol {get_type_name(exp_type)}:\n"
            + "\n".join(parts),
            hints,
            {
                "exp_type": get_type_name(exp_type),
                "actual_type": get_type_name(subcls),
                "artifact_name": get_type_name(subcls),
            },
        )
    return subcls


class TypeRegistry(
    ContainerMixin[Hashable, Type[Cls]],
    ABC,
    Generic[Cls],
):
    """Registry for classes, with optional inheritance/protocol enforcement.

    Supports:
      - Registration with optional explicit params_model
      - Auto-extraction of params_model from __init__ signature
      - Strict mode for protocol/inheritance checking
      - Multi-repo DI container via ContainerMixin
    """

    _repository: MutableMapping[Hashable, Type[Cls]]
    _strict: ClassVar[bool] = False
    _abstract: ClassVar[bool] = False
    __slots__: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def _get_mapping(cls) -> MutableMapping[Hashable, Type[Cls]]:
        return cls._repository

    @classmethod
    def __init_subclass__(
        cls,
        strict: bool = False,
        abstract: bool = False,
        logic_namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the TypeRegistry subclass.

        Args:
            strict: Enforce protocol conformance checks.
            abstract: Require registered classes to inherit from this registry.
            logic_namespace: Optional remote storage namespace.
        """
        super().__init_subclass__(**kwargs)
        if logic_namespace is None:
            cls._repository = ThreadSafeLocalStorage[Hashable, Type[Cls]]()
        else:
            cls._repository = RemoteStorageProxy[Hashable, Type[Cls]](
                namespace=logic_namespace
            )
        cls._strict = strict
        cls._abstract = abstract
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Initialized TypeRegistry subclass %s (strict=%s, abstract=%s)",
                get_type_name(cls),
                strict,
                abstract,
            )

    @classmethod
    def _internalize_artifact(cls, value: Any) -> Type[Cls]:
        """Validate and internalize a class artifact."""
        v = _validate_class(value)
        if cls._abstract:
            v = _validate_class_hierarchy(v, abc_class=cls)
        if cls._strict:
            protocol = get_protocol(cls)
            v = _validate_class_structure(v, exp_type=protocol)
        return super()._internalize_artifact(v)

    @classmethod
    def _identifier_of(cls, item: Type[Cls]) -> Hashable:
        """Return the identifier (class name) for an artifact."""
        return get_type_name(_validate_class(item))

    # -------------------------------------------------------------------------
    # Registration with params_model support
    # -------------------------------------------------------------------------

    @overload
    @classmethod
    def register_artifact(
        cls,
        artifact: Type[Cls],
        *,
        params_model: Optional[Type[BaseModel]] = None,
    ) -> Type[Cls]:
        """Register a class artifact with optional params_model."""
        ...

    @overload
    @classmethod
    def register_artifact(
        cls,
        *,
        params_model: Optional[Type[BaseModel]] = None,
    ) -> Any:
        """Decorator form: register a class with optional params_model."""
        ...

    @classmethod
    def register_artifact(
        cls,
        artifact: Optional[Type[Cls]] = None,
        *,
        params_model: Optional[Type[BaseModel]] = None,
    ) -> Union[Type[Cls], Any]:
        """Register a class artifact with optional explicit params_model.

        Can be used as a decorator or called directly.

        Args:
            artifact: The class to register.
            params_model: Optional Pydantic model for parameter validation.
                If not provided, auto-extracted from __init__ signature.

        Returns:
            The registered class (or decorator if artifact is None).

        Example::

            @MyRegistry.register_artifact(params_model=MyParams)
            class MyClass:
                def __init__(self, x: int): ...

            # Or auto-extract:
            @MyRegistry.register_artifact
            class MyClass:
                def __init__(self, x: int): ...
        """

        def _do_register(art: Type[Cls]) -> Type[Cls]:
            # Save explicit params_model before registration
            if params_model is not None:
                identifier = get_type_name(_validate_class(art))
                cls._save_scheme(identifier, params_model)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Saved explicit params_model for %s: %s",
                        identifier,
                        params_model.__name__,
                    )

            # Use parent's register_artifact for the actual registration
            return cast(Type[Cls], super(TypeRegistry, cls).register_artifact(art))

        if artifact is None:
            # Decorator form: @register_artifact(params_model=...)
            return _do_register
        else:
            # Direct call: register_artifact(MyClass, params_model=...)
            return _do_register(artifact)

    # -------------------------------------------------------------------------
    # Module Registration
    # -------------------------------------------------------------------------

    @classmethod
    def register_module_subclasses(
        cls,
        module: ModuleType,
        raise_error: bool = True,
    ) -> ModuleType:
        """Register all classes from a module.

        Args:
            module: The module to scan.
            raise_error: Whether to raise on registration failure.

        Returns:
            The module (for chaining).
        """
        assert isinstance(module, ModuleType), (
            f"Expected ModuleType, got {type(module)}"
        )
        members = get_module_members(module)
        ok, fail = 0, 0
        for obj in members:
            if not inspect.isclass(obj):
                continue
            name = get_object_name(obj)
            try:
                cls.register_artifact(obj)
                ok += 1
            except (ValidationError, ConformanceError, InheritanceError) as e:
                fail += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Skipping %s: %s", name, e)
                if raise_error:
                    if hasattr(e, "context"):
                        e.context.update(
                            {
                                "module_name": get_module_name(module),
                                "operation": "register_module_subclasses",
                            }
                        )
                    raise
        logger.info(
            "%s: registered %d class(es), %d failed", get_type_name(cls), ok, fail
        )
        return module
