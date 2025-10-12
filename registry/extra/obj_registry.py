r"""Object registry with optional conformance checks and weakref storage."""

from __future__ import annotations

import inspect
import logging
import weakref
from typing import Any, Dict, Generic, Hashable, TypeVar, List, Tuple, ClassVar

from .mixin import MutableValidatorMixin
from .utils import (
    ConformanceError,
    ValidationError,
    get_protocol,
    get_type_name,
    _validate_function_signature,
)

logger = logging.getLogger(__name__)

ObjT = TypeVar("ObjT")


class RegistryError(ValidationError):
    """Raised when a stored weak reference cannot be resolved."""


def get_mem_addr(obj: Any, with_prefix: bool = True) -> str:
    addr = id(obj)
    return f"{addr:#x}" if with_prefix else f"{addr:x}"


def _validate_instance_hierarchy(instance: ObjT, /, expected_type: type) -> ObjT:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Checking instance %s isa %s",
            get_type_name(type(instance)),
            get_type_name(expected_type),
        )
    if not isinstance(instance, expected_type):
        raise ValidationError(
            f"{instance} is not an instance of {get_type_name(expected_type)}",
            [f"Instantiate or cast to {get_type_name(expected_type)}"],
            {
                "expected_type": get_type_name(expected_type),
                "actual_type": get_type_name(type(instance)),
                "artifact_name": str(instance),
            },
        )
    return instance


def _validate_instance_structure(obj: Any, /, expected_type: type):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Validating structure of instance %s against protocol %s",
            get_type_name(type(obj)),
            get_type_name(expected_type),
        )
    missing: List[str] = []
    sig_errs: List[str] = []
    for name in dir(expected_type):
        if name.startswith("_"):
            continue
        if not hasattr(obj, name):
            missing.append(name)
            continue
        exp_attr, obj_attr = getattr(expected_type, name), getattr(obj, name)
        if callable(exp_attr):
            if not callable(obj_attr):
                sig_errs.append(f"Attribute '{name}' should be callable")
                continue
            f = obj_attr.__func__ if inspect.ismethod(obj_attr) else obj_attr
            ef = exp_attr.__func__ if inspect.ismethod(exp_attr) else exp_attr
            try:
                _validate_function_signature(f, ef)
            except ConformanceError as e:
                sig_errs.append(f"{name}: {e.message}")
    if missing or sig_errs:
        parts: List[str] = []
        hints: List[str] = []
        if missing:
            parts.append(f"Missing attributes: {', '.join(missing)}")
            hints.extend([f"Add attribute/method: {a}" for a in missing])
        if sig_errs:
            parts.append("Signature mismatches:")
            parts.extend(f"  {e}" for e in sig_errs)
            hints.append("Match method signatures to protocol")
        raise ConformanceError(
            f"Instance of {get_type_name(type(obj))} does not conform to protocol {get_type_name(expected_type)}:\n"
            + "\n".join(parts),
            hints,
            {
                "expected_type": get_type_name(expected_type),
                "actual_type": get_type_name(type(obj)),
                "artifact_name": str(obj),
            },
        )
    return obj


class ObjectRegistry(MutableValidatorMixin[Hashable, ObjT], Generic[ObjT]):
    """Registry for instances; stores weakrefs by default."""

    _repository: Dict[Hashable, Any]
    _strict: ClassVar[bool] = False
    _abstract: ClassVar[bool] = False
    _strict_weakref: ClassVar[bool] = False
    __slots__: ClassVar[Tuple[str, ...]] = ()

    @classmethod
    def _get_mapping(cls):
        return cls._repository

    @classmethod
    def __init_subclass__(
        cls, strict: bool = False, abstract: bool = False, strict_weakref: bool = False
    ) -> None:
        super().__init_subclass__()
        cls._repository = {}
        cls._strict = strict
        cls._abstract = abstract
        cls._strict_weakref = strict_weakref
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Initialized ObjectRegistry subclass %s (strict=%s, abstract=%s, strict_weakref=%s)",
                cls.__name__,
                strict,
                abstract,
                strict_weakref,
            )

    @classmethod
    def _internalize_artifact(cls, value: Any) -> ObjT:
        v = value
        if cls._abstract:
            v = _validate_instance_hierarchy(v, expected_type=cls)
        if cls._strict:
            protocol = get_protocol(cls)
            v = _validate_instance_structure(v, expected_type=protocol)
        try:
            return weakref.ref(v)  # type: ignore[return-value]
        except TypeError as e:
            if cls._strict_weakref:
                raise ValidationError(
                    f"Cannot create weak reference: {e}",
                    [
                        "Register objects that support weak references or disable strict_weakref"
                    ],
                    {
                        "registry_name": cls.__name__,
                        "operation": "weak_reference_creation",
                        "artifact_type": type(v).__name__,
                    },
                ) from e
            return v  # strong reference fallback

    @classmethod
    def _externalize_artifact(cls, value: Any) -> ObjT:
        if isinstance(value, weakref.ref):
            resolved = value()
            if resolved is None:
                raise RegistryError(
                    "Weak reference is dead (object was collected)",
                    ["Keep a strong reference; call cleanup() to purge dead entries"],
                    {
                        "operation": "weak_reference_resolution",
                        "registry_name": cls.__name__,
                    },
                )
            return resolved
        return value

    @classmethod
    def _identifier_of(cls, item: ObjT) -> Hashable:
        return get_mem_addr(item)

    @classmethod
    def cleanup(cls) -> int:
        """Purge dead weakrefs; return number removed."""
        dead = []
        for k, v in list(cls._repository.items()):
            try:
                if isinstance(v, weakref.ref) and v() is None:
                    dead.append(k)
            except Exception:
                dead.append(k)
        for k in dead:
            cls._repository.pop(k, None)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Cleaned %d dead references from %s", len(dead), cls.__name__)
        return len(dead)
