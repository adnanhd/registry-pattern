r"""Type (class) registry with optional inheritance/protocol checks."""

from __future__ import annotations

import inspect
import logging
from abc import ABC
from typing import Any, Generic, Hashable, MutableMapping, Optional, Type, TypeVar

from .mixin import MutableRegistryValidatorMixin
from .storage import RemoteStorageProxy, ThreadSafeLocalStorage
from .utils import (
    ConformanceError,
    InheritanceError,
    ValidationError,
    get_func_name,
    get_module_members,
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
            "actual_type": type(obj).__name__,
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


def _validate_function_signature(func: Any, expected_func: Any) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Validating method signature %s against %s",
            get_func_name(func),
            get_func_name(expected_func),
        )
    try:
        fs = inspect.signature(func)
        es = inspect.signature(expected_func)
    except (ValueError, TypeError) as e:
        raise ConformanceError(
            f"Cannot inspect function signature: {e}",
            ["Ensure both are valid callables"],
            {"artifact_name": get_func_name(func), "operation": "signature_inspection"},
        )
    errs, hints = [], []
    fparams, eparams = list(fs.parameters.values()), list(es.parameters.values())
    if len(fparams) != len(eparams):
        errs.append(
            f"Parameter count mismatch: expected {len(eparams)}, got {len(fparams)}"
        )
        hints.append(f"Use exactly {len(eparams)} parameters")
    for p, ep in zip(fparams, eparams):
        if ep.annotation != inspect.Parameter.empty and p.annotation != ep.annotation:
            if p.annotation == inspect.Parameter.empty:
                errs.append(f"Parameter '{p.name}' missing type annotation")
                hints.append(
                    f"Annotate: {p.name}: {getattr(ep.annotation, '__name__', str(ep.annotation))}"
                )
            else:
                errs.append(
                    f"Parameter '{p.name}' type mismatch: expected {getattr(ep.annotation, '__name__', str(ep.annotation))}, "
                    f"got {getattr(p.annotation, '__name__', str(p.annotation))}"
                )
                hints.append("Align parameter type annotation")
    if (
        es.return_annotation != inspect.Signature.empty
        and fs.return_annotation != es.return_annotation
    ):
        if fs.return_annotation == inspect.Signature.empty:
            errs.append("Missing return type annotation")
            hints.append(
                f"Annotate return: -> {getattr(es.return_annotation, '__name__', str(es.return_annotation))}"
            )
        else:
            errs.append(
                f"Return type mismatch: expected {getattr(es.return_annotation, '__name__', str(es.return_annotation))}, "
                f"got {getattr(fs.return_annotation, '__name__', str(fs.return_annotation))}"
            )
            hints.append("Align return annotation")
    if errs:
        raise ConformanceError(
            "Method signature validation failed:\n"
            + "\n".join(f"  • {e}" for e in errs),
            hints,
            {
                "artifact_name": get_func_name(func),
                "expected_type": str(es),
                "actual_type": str(fs),
            },
        )


def _validate_class_structure(subcls: type, /, expected_type: type) -> type:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Validating structure of %s against protocol %s",
            get_type_name(subcls),
            get_type_name(expected_type),
        )
    missing, sig_errs = [], []
    for name in dir(expected_type):
        if name.startswith("_"):
            continue
        if not hasattr(subcls, name):
            missing.append(name)
            continue
        exp_attr, sub_attr = getattr(expected_type, name), getattr(subcls, name)
        if callable(exp_attr):
            if not callable(sub_attr):
                sig_errs.append(f"Attribute '{name}' should be callable")
                continue
            try:
                _validate_function_signature(sub_attr, exp_attr)
            except ConformanceError as e:
                sig_errs.append(f"{name}: {e.message}")
    if missing or sig_errs:
        parts, hints = [], []
        if missing:
            parts.append(f"Missing methods: {', '.join(missing)}")
            hints.extend([f"Add method: {m}(...) -> ..." for m in missing])
        if sig_errs:
            parts.append("Signature mismatches:")
            parts.extend(f"  {e}" for e in sig_errs)
            hints.append("Match parameter and return annotations")
        raise ConformanceError(
            f"Class {get_type_name(subcls)} does not conform to protocol {get_type_name(expected_type)}:\n"
            + "\n".join(parts),
            hints,
            {
                "expected_type": get_type_name(expected_type),
                "actual_type": get_type_name(subcls),
                "artifact_name": get_type_name(subcls),
            },
        )
    return subcls


class TypeRegistry(
    MutableRegistryValidatorMixin[Hashable, Type[Cls]],
    ABC,
    Generic[Cls],
):
    """Registry for classes, with optional inheritance/protocol enforcement."""

    _repository: MutableMapping[Hashable, Type[Cls]]
    _strict: bool = False
    _abstract: bool = False
    __slots__ = ()

    @classmethod
    def _get_mapping(cls):
        return cls._repository

    @classmethod
    def __init_subclass__(
        cls,
        strict: bool = False,
        abstract: bool = False,
        proxy_namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)
        if proxy_namespace is None:
            cls._repository = ThreadSafeLocalStorage[Hashable, Type[Cls]]()
        else:
            cls._repository = RemoteStorageProxy[Hashable, Type[Cls]](
                namespace=proxy_namespace
            )
        cls._strict = strict
        cls._abstract = abstract
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Initialized TypeRegistry subclass %s (strict=%s, abstract=%s)",
                cls.__name__,
                strict,
                abstract,
            )

    @classmethod
    def _internalize_artifact(cls, value: Any) -> Type[Cls]:
        v = _validate_class(value)
        if cls._abstract:
            v = _validate_class_hierarchy(v, abc_class=cls)
        if cls._strict:
            protocol = get_protocol(cls)
            v = _validate_class_structure(v, expected_type=protocol)
        return v

    @classmethod
    def _identifier_of(cls, item: Type[Cls]) -> Hashable:
        return get_type_name(_validate_class(item))

    @classmethod
    def _externalize_artifact(cls, value: Type[Cls]) -> Type[Cls]:
        return value

    @classmethod
    def register_module_subclasses(cls, module: Any, raise_error: bool = True) -> Any:
        members = get_module_members(module)
        ok, fail = 0, 0
        for obj in members:
            name = getattr(obj, "__name__", str(obj))
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
                                "module_name": getattr(module, "__name__", str(module)),
                                "operation": "register_module_subclasses",
                            }
                        )
                    raise
        logger.info("%s: registered %d class(es), %d failed", cls.__name__, ok, fail)
        return module
