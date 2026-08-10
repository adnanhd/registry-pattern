"""Pydantic integration: ``ArtifactOf[Registrar]`` -- the pydantic-native ``build()``.

``ArtifactOf[SomeRegistry]`` is an annotated type (in the spirit of pydantic's
``InstanceOf``) parameterized by a *registrar*. A pydantic field or a
``@validate_call`` argument typed ``ArtifactOf[SomeRegistry]``:

- **validates** either a live artifact of the registry's element type OR a config
  (``{type, data, ...}``) that is built via :func:`registry.build` (the built type
  must match -- so a union discriminates cleanly);
- **serializes** mode-aware: ``model_dump()`` (python) returns the live artifact,
  ``model_dump(mode="json")`` returns the ``{type, data, meta}`` config envelope.

Several registrars are accepted as a union, either spelling:

    ArtifactOf[FooRegistry, BarRegistry]              # sugar for ...
    Union[ArtifactOf[FooRegistry], ArtifactOf[BarRegistry]]

It drops into ``Annotated`` chains alongside after-validators, so wiring stays
pydantic-native: value checks (``no_info_after_validator_function``) and
cross-argument dependencies (``with_info_after_validator_function``, reading the
already-validated sibling from ``info.data``).
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import GetCoreSchemaHandler, ValidationInfo
from pydantic_core import CoreSchema, core_schema
from pydantic_core.core_schema import SerializationInfo

from .. import factory
from ..container import BuildCfg, is_build_cfg, normalize_cfg
from ..factory import _envelope_for
from ..fnc_registry import FunctionalRegistry
from ..typ_registry import TypeRegistry, _base_type_of
from ..type_guard import _runtime_type, _type_name

__all__ = ["ArtifactOf"]

_REGISTRY_BASES = (TypeRegistry, FunctionalRegistry)


def _artifact_type(registrar: type) -> type:
    """Build the pydantic type for a single registrar (instance-or-config + serialize)."""
    element_type = _base_type_of(registrar)
    if element_type is Any:
        # Unbound registrar: accept any built artifact (the top-type fallback).
        element_type = object
    runtime_type = _runtime_type(element_type)
    type_label = _type_name(element_type)
    repo: Optional[str] = getattr(registrar, "repo", None)
    reg_name = getattr(registrar, "__name__", str(registrar))

    def validate(
        value: Any, info: ValidationInfo
    ) -> Any:  # pydantic callback: an instance or a build config
        if element_type is not object and isinstance(value, runtime_type):
            return value  # already a live artifact
        if isinstance(value, BuildCfg) or is_build_cfg(value):
            cfg = normalize_cfg(value) if isinstance(value, dict) else value
            # Thread already-validated siblings (``info.data``) into the build
            # scope, so a nested ``$sibling.attr()`` ref resolves against them --
            # e.g. an optimizer built over ``$model.parameters()`` reads the model
            # field validated just before it.
            result = factory.build(cfg, ctx=dict(info.data))
            # Require the built type to match, so ``Union[ArtifactOf[A], ArtifactOf[B]]``
            # discriminates (a mismatch raises -> pydantic tries the next member).
            if element_type is not object and not isinstance(result, runtime_type):
                raise ValueError(
                    f"built {type(result).__name__}, expected {type_label}"
                )
            return result
        if element_type is object:
            return value
        raise ValueError(
            f"expected {type_label} instance or a build config, got {type(value).__name__}"
        )

    def serialize(
        value: Any, info: SerializationInfo
    ) -> Any:  # pydantic callback: value is the built artifact
        # json mode -> the {type,data,meta} config envelope; python mode -> the live artifact.
        if info.mode == "json":
            return _envelope_for(value, repo=repo)
        return value

    class _ArtifactOfType:
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> CoreSchema:
            return core_schema.with_info_plain_validator_function(
                validate,
                serialization=core_schema.plain_serializer_function_ser_schema(
                    serialize,
                    info_arg=True,
                    return_schema=core_schema.any_schema(),
                ),
            )

    _ArtifactOfType.__name__ = f"ArtifactOf[{reg_name}]"
    _ArtifactOfType.__qualname__ = f"ArtifactOf[{reg_name}]"
    _ArtifactOfType.__doc__ = (
        f"Live {type_label} artifact of {reg_name}, or a config to build one."
    )
    return _ArtifactOfType


def _check_registrar(registrar: object) -> None:
    if not (isinstance(registrar, type) and issubclass(registrar, _REGISTRY_BASES)):
        raise TypeError(
            "ArtifactOf[...] expects a registry class (a TypeRegistry or "
            f"FunctionalRegistry subclass), got {registrar!r}"
        )


class ArtifactOf:
    """``ArtifactOf[SomeRegistry]`` (or ``ArtifactOf[A, B]``) -- a live artifact, or a config to build one."""

    __slots__ = ()

    def __class_getitem__(
        cls, registrar: Any
    ) -> Any:  # a single pydantic type, or a Union special form
        # Multiple registrars -> a pydantic Union of the single-registrar types.
        if isinstance(registrar, tuple):
            if not registrar:
                raise TypeError("ArtifactOf[...] needs at least one registry")
            for r in registrar:
                _check_registrar(r)
            return Union[tuple(_artifact_type(r) for r in registrar)]
        _check_registrar(registrar)
        return _artifact_type(registrar)
