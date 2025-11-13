r"""Config registry mapping weakref(object) -> configuration dict."""

from __future__ import annotations

import logging
import weakref
from abc import ABC
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, _ProtocolMeta

from .mixin import MutableValidatorMixin
from .utils import (
    ConformanceError,
    ValidationError,
    get_object_name,
    get_protocol,
    get_type_name,
)

logger = logging.getLogger(__name__)

ObjT = TypeVar("ObjT")
CfgT = TypeVar("CfgT", bound=Dict[str, Any])


class RegistryError(ValidationError):
    """Raised when a weakref key cannot be resolved."""


def _is_hashable(x: Any) -> bool:
    if hasattr(x, "__hash__"):
        return True
    try:
        hash(x)
        return True
    except Exception:
        return False


def _validate_instance_hierarchy(instance: Any, /, expected_type: _ProtocolMeta) -> Any:
    if not isinstance(instance, expected_type):
        raise ValidationError(
            f"{instance} is not an instance of {getattr(expected_type, '__name__', str(expected_type))}",
            ["Ensure the key object matches the expected base type"],
            {
                "expected_type": get_type_name(expected_type),
                "actual_type": get_type_name(type(instance)),
                "artifact_name": get_object_name(instance),
            },
        )
    return instance


def _validate_instance_structure(obj: Any, /, expected_type: _ProtocolMeta) -> Any:
    missing: List[str] = []
    for name in dir(expected_type):
        if name.startswith("_"):
            continue
        if not hasattr(obj, name):
            missing.append(name)
    if missing:
        raise ConformanceError(
            f"Key object missing attributes: {', '.join(missing)}",
            [f"Add attribute/method: {a}" for a in missing],
            {
                "expected_type": get_type_name(expected_type),
                "actual_type": get_type_name(type(obj)),
                "artifact_name": get_object_name(obj),
            },
        )
    return obj


class ConfigRegistry(MutableValidatorMixin[Any, CfgT], ABC, Generic[ObjT, CfgT]):
    """Registry of configurations keyed by a weak reference to objects."""

    _repository: Dict[Any, CfgT]
    _strict: bool = False
    _abstract: bool = False
    __slots__: Tuple[str, ...] = ()

    @classmethod
    def _get_mapping(cls):
        return cls._repository

    @classmethod
    def __init_subclass__(cls, strict: bool = False, abstract: bool = False) -> None:
        super().__init_subclass__()
        cls._repository = {}
        cls._strict = strict
        cls._abstract = abstract
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Initialized ConfigRegistry subclass %s (strict=%s, abstract=%s)",
                cls.__name__,
                strict,
                abstract,
            )

    @classmethod
    def _internalize_identifier(cls, value: ObjT) -> Any:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Validating config key object: %r", value)
        if not _is_hashable(value):
            raise ValidationError(
                f"Key must be hashable, got {get_type_name(type(value))}",
                [
                    "Avoid mutable types as keys",
                    "Wrap primitives in a small class if necessary",
                ],
                {
                    "expected_type": "Hashable",
                    "actual_type": get_type_name(type(value)),
                    "artifact_name": str(value),
                },
            )
        v = value
        if cls._abstract:
            v = _validate_instance_hierarchy(v, expected_type=cls)
        if cls._strict:
            protocol = get_protocol(cls)
            v = _validate_instance_structure(v, expected_type=protocol)
        try:
            return weakref.ref(v)
        except TypeError as e:
            raise ValidationError(
                f"Cannot create weak reference for key: {e}",
                [
                    "Use objects that support weak references",
                    "Do not use primitives as keys",
                ],
                {
                    "operation": "weak_reference_creation",
                    "artifact_type": type(v).__name__,
                },
            ) from e

    @classmethod
    def _externalize_identifier(cls, value: Any) -> ObjT:
        if isinstance(value, weakref.ref):
            actual = value()
            if actual is None:
                raise RegistryError(
                    "Weakref key is dead (object collected)",
                    [
                        "Keep a strong reference to the key object",
                        "Call cleanup() to purge dead entries",
                    ],
                    {
                        "operation": "weak_reference_resolution",
                        "registry_name": cls.__name__,
                    },
                )
            return actual  # type: ignore
        if not _is_hashable(value):
            raise ValidationError(
                f"Key must be hashable, got {type(value).__name__}",
                ["Use hashable objects as keys"],
                {
                    "expected_type": "Hashable",
                    "actual_type": type(value).__name__,
                    "key": str(value),
                },
            )
        return value  # type: ignore

    @classmethod
    def _internalize_artifact(cls, value: Any) -> CfgT:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Validating configuration payload: %s", type(value).__name__)
        if not isinstance(value, dict):
            raise ValidationError(
                f"Configuration must be a dict, got {type(value).__name__}",
                ["Use dict() or {} to create configuration"],
                {"expected_type": "dict", "actual_type": type(value).__name__},
            )
        return value

    @classmethod
    def _externalize_artifact(cls, value: CfgT) -> CfgT:
        return value

    @classmethod
    def _find_weakref_key(cls, key: Any) -> Optional[weakref.ref]:
        if isinstance(key, weakref.ref):
            return key if key in cls._repository else None
        dead = []
        for wref in list(cls._repository.keys()):
            if isinstance(wref, weakref.ref):
                try:
                    obj = wref()
                    if obj is key:
                        return wref
                    if obj is None:
                        dead.append(wref)
                except Exception:
                    dead.append(wref)
        for w in dead:
            cls._repository.pop(w, None)
        if logger.isEnabledFor(logging.DEBUG) and dead:
            logger.debug("Purged %d dead weakref keys during lookup", len(dead))
        return None

    @classmethod
    def has_identifier(cls, key: Any) -> bool:
        try:
            return cls._find_weakref_key(key) is not None
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Error checking key existence in %s: %s", cls.__name__, e)
            return False

    @classmethod
    def get_artifact(cls, key: Any) -> CfgT:
        try:
            wkey = cls._find_weakref_key(key)
            if wkey is None:
                raise RegistryError(
                    f"Key '{key}' is not present",
                    [
                        "Use has_identifier() before get",
                        "Ensure key object has not been garbage collected",
                    ],
                    {
                        "operation": "get_artifact",
                        "registry_name": cls.__name__,
                        "registry_size": len(cls._repository),
                    },
                )
            return cls._repository[wkey]
        except (ValidationError, RegistryError):
            raise
        except Exception as e:
            raise RegistryError(
                f"Failed to retrieve configuration for key '{key}': {e}",
                ["Check that the key exists and is accessible"],
                {"operation": "get_artifact", "registry_name": cls.__name__},
            ) from e

    @classmethod
    def cleanup(cls) -> int:
        """Remove entries whose weakref keys are dead."""
        dead = [
            w
            for w in list(cls._repository.keys())
            if isinstance(w, weakref.ref) and w() is None
        ]
        for w in dead:
            cls._repository.pop(w, None)
        if logger.isEnabledFor(logging.DEBUG) and dead:
            logger.debug(
                "Cleaned %d dead weakref keys from %s", len(dead), cls.__name__
            )
        return len(dead)
