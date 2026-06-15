"""Validator engine registry.

A validator's contract::

    validator(target, data: dict) -> dict     # raises on invalid

Three engines ship by default -- ``pydantic`` (uses ``derive_config_schema``),
``jsonargparse`` (uses jsonargparse's parser), and ``noop`` (passthrough).
Pick by string name at ``build(cfg, validator=...)`` time.
"""

from __future__ import annotations
from typing import Dict, Union

from collections.abc import Callable
from typing import Any

from .fnc_registry import FunctionalRegistry
from .schema import ensure_schema

__all__ = ["ValidatorRegistry", "Validator"]


Validator = Callable[[Union[type, Callable[..., Any]], Dict[str, Any]], Dict[str, Any]]


class ValidatorRegistry(FunctionalRegistry):
    """String-keyed registry of validator engines used by ``registry.factory.build``."""


@ValidatorRegistry.register_artifact
def pydantic(target: Union[type, Callable[..., Any]], data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ``data`` against the target's cached config schema.

    Reads from the per-target schema cache populated at registration time.
    Falls back to deriving on the fly if the target was never registered
    (e.g. an anonymous callable passed directly to ``build``).
    """
    schema = ensure_schema(target).config
    return schema.model_validate(data).model_dump()


@ValidatorRegistry.register_artifact
def jsonargparse(
    target: Union[type, Callable[..., Any]], data: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate via jsonargparse's parser. Requires the ``jsonargparse`` extra."""
    import jsonargparse as ja  # pyright: ignore[reportMissingImports]

    parser = ja.ArgumentParser()
    if isinstance(target, type):
        parser.add_class_arguments(target)
    else:
        parser.add_function_arguments(target)
    return vars(parser.parse_object(data))


@ValidatorRegistry.register_artifact
def noop(target: Union[type, Callable[..., Any]], data: Dict[str, Any]) -> Dict[str, Any]:
    """Passthrough -- no validation, no coercion."""
    return dict(data)


@ValidatorRegistry.register_artifact
def python(target: Union[type, Callable[..., Any]], data: Any) -> Dict[str, Any]:
    """Python-native dict input; validate against target's signature via Pydantic."""
    return pydantic(target, data if isinstance(data, dict) else dict(data))


@ValidatorRegistry.register_artifact
def yaml(target: Union[type, Callable[..., Any]], data: Any) -> Dict[str, Any]:
    """YAML string input; decode then python-validate."""
    import yaml as _yaml

    decoded = _yaml.safe_load(data) if isinstance(data, str) else data
    return python(target, decoded)


@ValidatorRegistry.register_artifact
def json(target: Union[type, Callable[..., Any]], data: Any) -> Dict[str, Any]:
    """JSON string input; decode then python-validate."""
    import json as _json

    decoded = _json.loads(data) if isinstance(data, str) else data
    return python(target, decoded)


@ValidatorRegistry.register_artifact
def argparse(target: Union[type, Callable[..., Any]], data: Any) -> Dict[str, Any]:
    """argparse.Namespace input; ``vars(ns)`` then python-validate."""
    decoded = (
        vars(data) if hasattr(data, "__dict__") and not isinstance(data, dict) else data
    )
    return python(target, decoded)


def resolve_validator(name: str) -> Validator:
    """String lookup with helpful error message."""
    return ValidatorRegistry.get_artifact(name)
