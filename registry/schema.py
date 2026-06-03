"""Schema derivation for the recursive factory.

Two layers, two derived schemas:

- **Config layer** (input envelope.data): runtime types like ``nn.Module`` are
  rewritten to ``Buildable[T]`` so Pydantic can validate the envelope as JSON.
  No ``arbitrary_types_allowed`` needed.
- **Meta layer** (output envelope.meta): derived from ``Annotated[T, ...]``
  ``compute`` markers on the target's signature. All fields are primitives.

Explicit ``data_schema``/``meta_schema`` class attributes on a registry
override the derived schemas; both can coexist with ``Annotated`` markers,
which fire on the runtime layer independently.

A process-wide ``WeakKeyDictionary`` cache keyed by target class/function
holds an :class:`ArtifactSchema` (config + meta + resolved hints) per
artifact. Population happens at registration time
(:func:`cache_schema`), lookup at build time (:func:`ensure_schema`),
explicit eviction at unregistration time (:func:`drop_schema`). When the
target is itself garbage-collected the cache entry vanishes automatically.
"""

from __future__ import annotations

import inspect
import threading
import typing
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, create_model

from .type_guard import Buildable

__all__ = [
    "ArtifactSchema",
    "build_schema",
    "cache_schema",
    "ensure_schema",
    "get_schema",
    "drop_schema",
    "derive_config_schema",
    "derive_meta_schema",
    "resolve_data_schema",
    "resolve_meta_schema",
    "process_validate",
    "process_compute",
]


_JSON_NATIVE: tuple[type, ...] = (int, float, str, bool, type(None), list, dict, tuple)


def _unwrap_annotated(tp: Any) -> Any:
    """``Annotated[T, ...]`` -> ``T``; otherwise passthrough."""
    if hasattr(tp, "__metadata__"):
        return typing.get_args(tp)[0]
    return tp


def _resolved_hints(target: type | Callable[..., Any]) -> dict[str, Any]:
    """Resolve annotations on ``target``, evaluating string forms.

    Falls back to ``inspect.signature(...).parameters[name].annotation`` if
    ``get_type_hints`` can't resolve (e.g. forward refs in locals).
    """
    obj = target.__init__ if isinstance(target, type) else target
    try:
        return typing.get_type_hints(obj, include_extras=True)
    except Exception:
        # Best effort: use raw annotations from the signature.
        sig = inspect.signature(obj)
        return {
            name: p.annotation
            for name, p in sig.parameters.items()
            if p.annotation is not inspect.Parameter.empty
        }


def _is_json_native(tp: Any) -> bool:
    """True iff ``tp`` is a JSON-safe primitive, or a parametrized generic
    over JSON-safe types. Conservative: any non-native generic param
    contaminates the result.
    """
    origin = typing.get_origin(tp)
    if origin is None:
        return tp in _JSON_NATIVE
    if origin in _JSON_NATIVE:
        # Parametrized list/dict/tuple -- every arg must also be safe.
        return all(_is_json_native(a) for a in typing.get_args(tp))
    return False


def _config_type_for(tp: Any) -> Any:
    """Map a runtime annotation to a JSON-safe config annotation.

    - JSON-native types pass through (``int``, ``str``, ``list``, ...).
    - Arbitrary classes become ``Buildable[T]`` so Pydantic accepts either a
      ``BuildCfg`` dict or an already-built instance.
    - ``Any`` / un-annotated stays ``Any``.
    """
    base = _unwrap_annotated(tp)
    if base is Any or base is inspect.Parameter.empty:
        return Any
    if _is_json_native(base):
        return tp  # keep generic parameters like list[int]
    if isinstance(base, type):
        return Buildable[base]
    return Any


def _signature_of(target: type | Callable[..., Any]) -> inspect.Signature:
    return inspect.signature(target.__init__ if isinstance(target, type) else target)


def derive_config_schema(target: type | Callable[..., Any]) -> type[BaseModel]:
    """Build a Pydantic model from ``target``'s signature for input validation.

    No ``arbitrary_types_allowed``. Arbitrary classes are rewritten to
    ``Buildable[T]`` via ``_config_type_for``.
    """
    sig = _signature_of(target)
    hints = _resolved_hints(target)
    fields: dict[str, Any] = {}
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        annotation = hints.get(name, p.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = Any
        config_ann = _config_type_for(annotation)
        default = p.default if p.default is not inspect.Parameter.empty else ...
        fields[name] = (config_ann, default)
    schema_name = getattr(target, "__name__", "Anonymous")
    return create_model(f"{schema_name}ConfigSchema", **fields)


def derive_meta_schema(target: type | Callable[..., Any]) -> type[BaseModel] | None:
    """Walk ``Annotated[T, Marker(...)]`` metadata; collect compute markers.

    Returns a Pydantic model with one field per marker (name -> return type),
    or ``None`` if the target has no compute markers. The model is configured
    with ``extra="allow"`` so that hooks / meters / pre_call writes that don't
    correspond to a marker survive ``model_dump()`` round-trip.
    """
    from pydantic import ConfigDict

    hints = _resolved_hints(target)
    fields: dict[str, Any] = {}
    for hint in hints.values():
        if not hasattr(hint, "__metadata__"):
            continue
        for marker in hint.__metadata__:
            if hasattr(marker, "compute"):
                ret = inspect.signature(marker.compute).return_annotation
                if ret is inspect.Signature.empty:
                    ret = Any
                marker_name = getattr(marker, "name", None)
                if marker_name:
                    fields[marker_name] = (ret, ...)
    if not fields:
        return None
    schema_name = getattr(target, "__name__", "Anonymous")
    return create_model(
        f"{schema_name}MetaSchema",
        __config__=ConfigDict(extra="allow"),
        **fields,
    )


def resolve_data_schema(
    registry: type, target: type | Callable[..., Any]
) -> type[BaseModel]:
    """Explicit ``registry.data_schema`` wins; otherwise serve from cache."""
    explicit = getattr(registry, "data_schema", None)
    if explicit is not None:
        return explicit
    return ensure_schema(target).config


def resolve_meta_schema(
    registry: type, target: type | Callable[..., Any]
) -> type[BaseModel] | None:
    """Explicit ``registry.meta_schema`` wins; otherwise serve from cache."""
    explicit = getattr(registry, "meta_schema", None)
    if explicit is not None:
        return explicit
    return ensure_schema(target).meta


def process_validate(
    target: type | Callable[..., Any],
    kwargs: dict[str, Any],
    ctx: dict[str, Any],
) -> None:
    """Run every ``Annotated[..., marker_with_validate].validate(value, kwargs, ctx)``."""
    hints = ensure_schema(target).hints
    for name, hint in hints.items():
        if name not in kwargs or not hasattr(hint, "__metadata__"):
            continue
        for marker in hint.__metadata__:
            if hasattr(marker, "validate"):
                marker.validate(kwargs[name], kwargs, ctx)


def process_compute(
    target: type | Callable[..., Any],
    kwargs: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    """Run every ``Annotated[..., marker_with_compute].compute(value)`` and write to meta."""
    hints = ensure_schema(target).hints
    for name, hint in hints.items():
        if name not in kwargs or not hasattr(hint, "__metadata__"):
            continue
        for marker in hint.__metadata__:
            if hasattr(marker, "compute"):
                marker_name = getattr(marker, "name", None) or name
                meta[marker_name] = marker.compute(kwargs[name])


# ---------------------------------------------------------------------------
# Schema cache: WeakKeyDictionary keyed by target class / function
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactSchema:
    """Per-target schemas + introspection state, built once at registration.

    - ``config`` is the Pydantic model used to validate the envelope's
      ``data`` dict (input layer). May be an explicit ``params_model``
      passed at registration, or the auto-derived schema.
    - ``meta`` is the model derived from ``Annotated`` ``compute`` markers
      on the target's signature; ``None`` if the target has no markers.
    - ``hints`` is the resolved ``get_type_hints`` mapping; cached because
      both ``process_validate`` and ``process_compute`` need it on every
      build and ``get_type_hints`` is itself expensive.
    """

    config: type[BaseModel]
    meta: type[BaseModel] | None
    hints: dict[str, Any]


_SCHEMA_CACHE: weakref.WeakKeyDictionary[Any, ArtifactSchema] = (
    weakref.WeakKeyDictionary()
)
_SCHEMA_LOCK = threading.Lock()


def build_schema(
    target: type | Callable[..., Any],
    *,
    config_override: type[BaseModel] | None = None,
) -> ArtifactSchema:
    """Derive an :class:`ArtifactSchema` for ``target`` without caching.

    ``config_override``, if given, takes the place of the auto-derived
    config schema (the explicit ``params_model`` path from
    ``register_artifact``). Meta + hints are still derived.
    """
    hints = _resolved_hints(target)
    config = (
        config_override if config_override is not None else derive_config_schema(target)
    )
    meta = derive_meta_schema(target)
    return ArtifactSchema(config=config, meta=meta, hints=hints)


def cache_schema(
    target: type | Callable[..., Any],
    *,
    config_override: type[BaseModel] | None = None,
) -> ArtifactSchema:
    """Build the schema for ``target`` and store it in the weak cache.

    Called eagerly from ``register_artifact`` so the build pipeline sees a
    pre-populated cache entry. Falls back to returning the freshly built
    schema if ``target`` is not weakly referenceable (rare; some builtins).
    """
    schema = build_schema(target, config_override=config_override)
    try:
        _SCHEMA_CACHE[target] = schema
    except TypeError:
        pass
    return schema


def get_schema(target: Any) -> ArtifactSchema | None:
    """Return the cached :class:`ArtifactSchema` for ``target`` or ``None``."""
    return _SCHEMA_CACHE.get(target)


def ensure_schema(target: type | Callable[..., Any]) -> ArtifactSchema:
    """Return the cached schema, deriving + caching on the first miss.

    Double-checked under a lock so concurrent first-builds for the same
    target do not duplicate the Pydantic model creation.
    """
    cached = _SCHEMA_CACHE.get(target)
    if cached is not None:
        return cached
    with _SCHEMA_LOCK:
        cached = _SCHEMA_CACHE.get(target)
        if cached is not None:
            return cached
        return cache_schema(target)


def drop_schema(target: Any) -> None:
    """Evict the cache entry for ``target``. No-op if absent."""
    _SCHEMA_CACHE.pop(target, None)
