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


@ValidatorRegistry.register_artifact
def python(target: type | Callable[..., Any], data: Any) -> dict[str, Any]:
    """Python-native dict input; validate against target's signature via Pydantic."""
    return pydantic(target, data if isinstance(data, dict) else dict(data))


@ValidatorRegistry.register_artifact
def yaml(target: type | Callable[..., Any], data: Any) -> dict[str, Any]:
    """YAML string input; decode then python-validate."""
    import yaml as _yaml

    decoded = _yaml.safe_load(data) if isinstance(data, str) else data
    return python(target, decoded)


@ValidatorRegistry.register_artifact
def json(target: type | Callable[..., Any], data: Any) -> dict[str, Any]:
    """JSON string input; decode then python-validate."""
    import json as _json

    decoded = _json.loads(data) if isinstance(data, str) else data
    return python(target, decoded)


@ValidatorRegistry.register_artifact
def argparse(target: type | Callable[..., Any], data: Any) -> dict[str, Any]:
    """argparse.Namespace input; ``vars(ns)`` then python-validate."""
    decoded = vars(data) if hasattr(data, "__dict__") and not isinstance(data, dict) else data
    return python(target, decoded)


def resolve_validator(name: str) -> Validator:
    """String lookup with helpful error message."""
    return ValidatorRegistry.get_artifact(name)
