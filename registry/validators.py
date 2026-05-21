"""Validator engine registry.

A validator's contract::

    validator(target, data: dict) -> dict     # raises on invalid

Three engines ship by default -- ``pydantic`` (uses ``derive_config_schema``),
``jsonargparse`` (uses jsonargparse's parser), and ``noop`` (passthrough).
Pick by string name at ``build(cfg, validator=...)`` time.
"""

from __future__ import annotations

from typing import Any, Callable

from .fnc_registry import FunctionalRegistry
from .schema import resolve_data_schema

__all__ = ["ValidatorRegistry", "Validator"]


Validator = Callable[[type | Callable[..., Any], dict[str, Any]], dict[str, Any]]


class ValidatorRegistry(FunctionalRegistry):
    """String-keyed registry of validator engines used by ``registry.factory.build``."""


@ValidatorRegistry.register_artifact
def pydantic(target: type | Callable[..., Any], data: dict[str, Any]) -> dict[str, Any]:
    """Validate ``data`` against the target's derived config schema.

    Resolves the schema via ``resolve_data_schema`` so explicit
    ``registry.data_schema`` overrides win.
    """
    from .schema import derive_config_schema

    schema = derive_config_schema(target)
    return schema.model_validate(data).model_dump()


@ValidatorRegistry.register_artifact
def jsonargparse(target: type | Callable[..., Any], data: dict[str, Any]) -> dict[str, Any]:
    """Validate via jsonargparse's parser. Requires the ``jsonargparse`` extra."""
    import jsonargparse as ja

    parser = ja.ArgumentParser()
    if isinstance(target, type):
        parser.add_class_arguments(target)
    else:
        parser.add_function_arguments(target)
    return vars(parser.parse_object(data))


@ValidatorRegistry.register_artifact
def noop(target: type | Callable[..., Any], data: dict[str, Any]) -> dict[str, Any]:
    """Passthrough -- no validation, no coercion."""
    return dict(data)


def resolve_validator(name: str) -> Validator:
    """String lookup with helpful error message."""
    return ValidatorRegistry.get_artifact(name)
