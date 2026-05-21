"""Recursive factory for ``BuildCfg``-shaped envelopes.

One source-agnostic entry point: ``build(cfg, ctx=...)``. The envelope can
come from YAML, JSON, callpyback RPC, or a hand-rolled Python dict — the
factory doesn't care.

Pipeline per envelope::

    [1] recurse on nested envelopes; resolve ``$ref`` strings against sibling scope
    [2] validate config layer (validator engine, e.g. pydantic)
    [3] optional ``registry.pre_call`` hook + ``Annotated.validate`` markers
    [4] ``target(**kwargs)`` — instantiate or invoke
    [5] optional ``registry.post_init`` hook + ``Annotated.compute`` markers
    [6] optional ``registry.post_call`` hook + meta_schema validation

The factory writes computed values back into ``cfg.meta`` and attaches
``__meta__`` to the result instance.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .container import BuildCfg, is_build_cfg, normalize_cfg
from .fnc_registry import _ALL_FN_REGISTRIES, FunctionalRegistry
from .schema import process_compute, process_validate, resolve_meta_schema
from .typ_registry import _ALL_TYPE_REGISTRIES, TypeRegistry
from .validators import resolve_validator

__all__ = ["build", "resolve"]


_REF_RE = re.compile(r"^\$([A-Za-z_][\w.]*)(\(\))?$")


def _resolve_ref(s: str, scope: dict[str, Any]) -> Any:
    """``$name`` / ``$name.attr`` / ``$name.method()`` against ``scope``."""
    m = _REF_RE.match(s)
    if not m:
        return s
    path, call = m.group(1), m.group(2)
    parts = path.split(".")
    if parts[0] not in scope:
        raise KeyError(f"$ref {s!r}: '{parts[0]}' not in scope (have: {sorted(scope)})")
    obj: Any = scope[parts[0]]
    for p in parts[1:]:
        obj = getattr(obj, p)
    return obj() if call else obj


def resolve(type_name: str, repo: str | None = None) -> tuple[type, Any]:
    """Find the registry that holds ``type_name``.

    Returns (registry_class, artifact). When ``repo`` is provided, it disambiguates
    among multiple matches.
    """
    matches: list[tuple[type, Any]] = []
    for registries in (_ALL_TYPE_REGISTRIES, _ALL_FN_REGISTRIES):
        for reg in registries.values():
            try:
                if reg.has_identifier(type_name):
                    matches.append((reg, reg.get_artifact(type_name)))
            except Exception:
                continue
    if not matches:
        raise KeyError(
            f"'{type_name}' not registered in any TypeRegistry/FunctionalRegistry"
        )
    if len(matches) == 1:
        return matches[0]
    if repo:
        filtered = [(r, a) for r, a in matches if r.__name__ == repo]
        if len(filtered) == 1:
            return filtered[0]
    raise KeyError(
        f"'{type_name}' is ambiguous (found in "
        f"{[r.__name__ for r, _ in matches]}); set `repo` on the envelope"
    )


def build(
    cfg: BuildCfg | dict[str, Any],
    *,
    validator: str = "pydantic",
    ctx: dict[str, Any] | None = None,
) -> Any:
    """Recursively construct from a normalized envelope.

    Args:
        cfg: ``BuildCfg`` instance or dict shaped like one.
        validator: Name of a registered validator engine (default: ``"pydantic"``).
        ctx: Sibling/parent context for ``$ref`` resolution. Mutated as siblings build.

    Returns:
        The constructed instance (for class targets) or the function's return value.
    """
    cfg = normalize_cfg(cfg)
    ctx = dict(ctx) if ctx else {}
    registry, target = resolve(cfg.type, cfg.repo if cfg.repo != "default" else None)
    validator_fn = resolve_validator(validator)

    # [1] recurse + $ref; later siblings can reference earlier ones
    data: dict[str, Any] = {}
    for k, v in cfg.data.items():
        scope = {**ctx, **data}
        if is_build_cfg(v):
            data[k] = build(v, validator=validator, ctx=scope)
        elif isinstance(v, str) and v.startswith("$"):
            data[k] = _resolve_ref(v, scope)
        else:
            data[k] = v

    # [2] config-layer validation
    kwargs: dict[str, Any] = validator_fn(target, data)

    # [3] runtime-layer pre-validation
    meta: dict[str, Any] = dict(cfg.meta)
    pre = getattr(registry, "pre_call", None)
    if callable(pre):
        pre(target, kwargs, ctx, meta)
    process_validate(target, kwargs, ctx)

    # [4] invoke
    result: Any = target(**kwargs)

    # [5] meta-layer computation
    post = getattr(registry, "post_init", None)
    if callable(post):
        post(result, meta)
    process_compute(target, kwargs, meta)
    tree = getattr(registry, "post_call", None)
    if callable(tree):
        tree(result, meta, ctx)

    # [6] meta_schema validation (if any)
    mschema = resolve_meta_schema(registry, target)
    if mschema is not None and meta:
        meta = mschema.model_validate(meta).model_dump()

    # Write meta back to the envelope and attach to the result.
    cfg.meta.clear()
    cfg.meta.update(meta)
    try:
        setattr(result, "__meta__", meta)
    except Exception:
        pass

    return result
