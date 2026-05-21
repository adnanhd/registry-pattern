"""Recursive factory for ``BuildCfg``-shaped envelopes.

One source-agnostic entry point: ``build(cfg, ctx=...)``. The envelope can
come from YAML, JSON, callpyback RPC, or a hand-rolled Python dict -- the
factory doesn't care.

Pipeline per envelope::

    [1] recurse on nested envelopes; resolve ``$ref`` strings against sibling scope
    [2] validate config layer (validator engine, e.g. pydantic)
    [3] optional ``registry.pre_call`` hook + ``Annotated.validate`` markers
    [4] ``target(**kwargs)`` -- instantiate or invoke
    [5] optional ``registry.post_init`` hook + ``Annotated.compute`` markers
    [6] optional ``registry.post_call`` hook + meta_schema validation

The factory writes computed values back into ``cfg.meta`` and attaches
``__meta__`` to the result instance.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from .container import BuildCfg, is_build_cfg, normalize_cfg
from .fnc_registry import _ALL_FN_REGISTRIES, FunctionalRegistry
from .meters import emit_meter
from .reporters import emit_reporter
from .schema import process_compute, process_validate, resolve_meta_schema
from .typ_registry import _ALL_TYPE_REGISTRIES, TypeRegistry
from .validators import resolve_validator

logger = logging.getLogger(__name__)

__all__ = ["build", "resolve", "validate", "serialize"]


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


def _repo_matches(reg_repo: str, query: str) -> bool:
    """Hierarchical match: exact, or `query` is an ancestor of `reg_repo`."""
    return reg_repo == query or reg_repo.startswith(query + ".")


def resolve(type_name: str, repo: str | None = None) -> tuple[type, Any]:
    """Find the registry that holds ``type_name``.

    Returns ``(registry_class, artifact)``. When ``repo`` is provided, the
    search is restricted to registries whose ``repo`` path equals it OR has
    it as a dotted prefix -- so ``repo="cofinn"`` matches both ``"cofinn"``
    and ``"cofinn.networks"`` and ``"cofinn.losses"``.

    If multiple registries match, an exact-repo match wins; otherwise the
    call raises with a hint showing the candidate paths.
    """
    matches: list[tuple[type, Any]] = []
    for repo_path, reg in {**_ALL_TYPE_REGISTRIES, **_ALL_FN_REGISTRIES}.items():
        if repo is not None and not _repo_matches(repo_path, repo):
            continue
        try:
            if reg.has_identifier(type_name):
                matches.append((reg, reg.get_artifact(type_name)))
        except Exception:
            continue
    if not matches:
        suffix = f" under repo='{repo}'" if repo else ""
        raise KeyError(
            f"'{type_name}' not registered in any TypeRegistry/FunctionalRegistry{suffix}"
        )
    if len(matches) == 1:
        return matches[0]
    if repo is not None:
        exact = [(r, a) for r, a in matches if r.repo == repo]
        if len(exact) == 1:
            return exact[0]
    raise KeyError(
        f"'{type_name}' is ambiguous (found in repos "
        f"{[r.repo for r, _ in matches]}); narrow with repo=..."
    )


def validate(
    target: type | Callable[..., Any] | str,
    data: Any,
    *,
    validator: str = "python",
) -> dict[str, Any]:
    """Run the medium decoder + Pydantic schema check, return validated kwargs.

    Sits between ``resolve`` (no work) and ``build`` (full pipeline). Useful
    for dry-run / config-lint flows where you want to verify a payload before
    paying the construction cost.
    """
    if isinstance(target, str):
        _, target = resolve(target)
    medium_fn = resolve_validator(validator)
    return medium_fn(target, data if data is not None else {})


def build(
    cfg_or_target: BuildCfg | dict[str, Any] | type | Callable[..., Any],
    data: Any = None,
    *,
    validator: str = "pydantic",
    ctx: dict[str, Any] | None = None,
    repo: str | None = None,
) -> Any:
    """Recursively construct from a normalized envelope OR explicit class+data.

    Two call modes:

    1. **Envelope mode** -- ``build(cfg)`` where ``cfg`` is a ``BuildCfg`` or
       a ``{"type": ..., "data": ...}`` dict. ``data`` is ignored.

    2. **Explicit class mode** -- ``build(MLP, raw_data, validator="yaml")``
       where the first arg is the class/callable itself and ``raw_data`` is
       in the format the chosen ``validator`` medium expects. Available
       mediums: ``"python"`` (dict), ``"yaml"`` (str), ``"json"`` (str),
       ``"argparse"`` (Namespace), plus the registry's ``"pydantic"``,
       ``"jsonargparse"``, ``"noop"``.

    Both modes go through the same pipeline (validation, meters, reporters,
    post hooks, meta). Returns the constructed instance or function result.
    """
    # Explicit class mode: decode raw data via the medium, then run the
    # standard envelope pipeline with validator="noop" (already validated).
    if not isinstance(cfg_or_target, (BuildCfg, dict)) and callable(cfg_or_target):
        target = cfg_or_target
        medium_fn = resolve_validator(validator)
        kwargs = medium_fn(target, data if data is not None else {})
        return build(
            BuildCfg(type=target.__name__, repo=repo or "default", data=kwargs),
            ctx=ctx,
            validator="noop",
        )

    cfg = cfg_or_target
    # Caller can override the envelope's repo via the kwarg.
    if repo is not None and isinstance(cfg, BuildCfg):
        cfg = cfg.model_copy(update={"repo": repo})
    elif repo is not None and isinstance(cfg, dict):
        cfg = {**cfg, "repo": repo}
    # Preserve a reference to the original dict so meta can be written back.
    raw_dict: dict[str, Any] | None = cfg if isinstance(cfg, dict) else None
    cfg = normalize_cfg(cfg)
    ctx = dict(ctx) if ctx else {}
    meta: dict[str, Any] = dict(cfg.meta)  # available to meters from the very first stage

    logger.info("build.start type=%s repo=%s", cfg.type, cfg.repo)
    # Pipeline contract at every stage: METERS first (they write to meta),
    # then REPORTERS (they read the final meta and ship externally).
    emit_meter("on_build_start", cfg=cfg, ctx=ctx, meta=meta)
    emit_reporter("on_build_start", cfg=cfg, ctx=ctx, meta=meta)

    try:
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
        pre = getattr(registry, "pre_call", None)
        if callable(pre):
            pre(target, kwargs, ctx, meta)
        process_validate(target, kwargs, ctx)

        emit_meter("on_validated", target=target, kwargs=kwargs, ctx=ctx, meta=meta)
        emit_reporter("on_validated", target=target, kwargs=kwargs, ctx=ctx, meta=meta)

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

        # Meters write THEIR measurements (lifetime delta, CPU, IO, ...) here.
        emit_meter("on_built", target=target, result=result, meta=meta, ctx=ctx)
        # Reporters see the now-fully-populated meta and ship it.
        emit_reporter("on_built", target=target, result=result, meta=meta, ctx=ctx)

        logger.info(
            "build.done type=%s target=%s meta_keys=%s",
            cfg.type,
            getattr(target, "__name__", "?"),
            list(meta) or "-",
        )

        # [6] meta_schema validation (if any)
        mschema = resolve_meta_schema(registry, target)
        if mschema is not None and meta:
            meta = mschema.model_validate(meta).model_dump()

        # Write meta back to the envelope (BuildCfg) AND to the original dict if any.
        cfg.meta.clear()
        cfg.meta.update(meta)
        if raw_dict is not None:
            raw_dict.setdefault("meta", {})
            if isinstance(raw_dict["meta"], dict):
                raw_dict["meta"].clear()
                raw_dict["meta"].update(meta)
        try:
            setattr(result, "__meta__", meta)
        except Exception:
            pass

        return result

    except Exception as exc:
        logger.warning("build.error type=%s exc=%s: %s", cfg.type, type(exc).__name__, exc)
        emit_meter("on_error", cfg=cfg, exc=exc, ctx=ctx, meta=meta)
        emit_reporter("on_error", cfg=cfg, exc=exc, ctx=ctx, meta=meta)
        raise


# ---------------------------------------------------------------------------
# Serialization side (instance -> medium-encoded output)
# ---------------------------------------------------------------------------


class SerializerRegistry(FunctionalRegistry):
    """String-keyed registry of serializer engines for ``serialize()``."""


@SerializerRegistry.register_artifact
def python(instance: Any) -> dict[str, Any]:
    """Read constructor-arg attributes off the instance into a dict."""
    import inspect

    sig = inspect.signature(type(instance).__init__)
    return {
        n: getattr(instance, n)
        for n in sig.parameters
        if n != "self" and hasattr(instance, n)
    }


@SerializerRegistry.register_artifact
def yaml(instance: Any) -> str:
    """YAML string from ``python()``."""
    import yaml as _yaml

    return _yaml.safe_dump(python(instance))


@SerializerRegistry.register_artifact
def json(instance: Any) -> str:
    """JSON string from ``python()``."""
    import json as _json

    return _json.dumps(python(instance))


def serialize(
    instance: Any,
    *,
    ctx: dict[str, Any] | None = None,  # noqa: ARG001 -- reserved for future nested serdes
    serializator: str = "python",
) -> Any:
    """Serialize ``instance`` via the named serializer medium.

    Symmetric counterpart to ``build()``:

    - ``serialize(obj, serializator="python")`` -> dict
    - ``serialize(obj, serializator="yaml")``   -> yaml string
    - ``serialize(obj, serializator="json")``   -> json string

    Register custom mediums via ``SerializerRegistry.register_artifact``.
    """
    encoder = SerializerRegistry.get_artifact(serializator)
    return encoder(instance)
