r"""Callable/Class → Pydantic config scheme registry.

This registry accepts either a function or a class as the artifact. On
registration it synthesizes a Pydantic `BaseModel` from the callable
signature (for classes, from `__init__` excluding `self`). The resulting
"scheme" lets you:

  - build a validated config object (verifiable, serializable),
  - execute the function with validated kwargs,
  - or instantiate the class with validated kwargs.

Design highlights
-----------------
- Integrates with the existing validator/mutator mixins for consistent
  error semantics and batch ops.
- Logs helpful DEBUG traces without adding overhead when DEBUG is off.
- Tolerates both Pydantic v1 and v2 by aliasing `model_create` to
  `pydantic.create_model`.
- Skips `*args/**kwargs` parameters (they are not representable as named
  model fields). If present, you'll get a DEBUG note.

Public surface
--------------
- `SchemeRegistry.register_artifact(func_or_cls)` -> `InvocationScheme`
- `SchemeRegistry.get_model(key)` -> Type[BaseModel]
- `SchemeRegistry.build_config(key, **kwargs)` -> BaseModel
- `SchemeRegistry.execute(key, data)` -> Any
- `SchemeRegistry.register_module_callables(module, ...)` convenience

Usage
-----
    from registry.scheme_registry import SchemeRegistry

    def f(a: int, b: float = 1.0): return a + b
    class Foo:
        def __init__(self, x: int, y: str = "ok"): self.x, self.y = x, y

    sf = SchemeRegistry.register_artifact(f)
    sc = SchemeRegistry.register_artifact(Foo)

    FModel = SchemeRegistry.get_model(sf.name)   # or by string key
    cfg = FModel(a=3, b=2.5)
    out = SchemeRegistry.execute(sf.name, cfg)   # calls f(a=3, b=2.5)

    CModel = SchemeRegistry.get_model("Foo")
    obj = SchemeRegistry.execute("Foo", CModel(x=10))  # Foo(x=10, y="ok")
"""

from __future__ import annotations

import logging
from abc import ABC
from inspect import Parameter, Signature, _empty, isclass, signature
from typing import Any, Callable, Dict, Hashable, Literal, Mapping, Tuple, Type, Union

# Pydantic import strategy: support both v1 and v2
from pydantic import BaseModel, create_model

from .mixin import MutableValidatorMixin
from .utils import ValidationError, get_func_name  # vends readable names

logger = logging.getLogger(__name__)

__all__ = ["InvocationScheme", "SchemeRegistry"]


def _callable_signature(
    artifact: Union[Callable[..., Any], Type[Any]],
) -> Tuple[str, Signature, Literal["function", "class"]]:
    """Return a display name, the signature to modelize, and kind."""
    if isclass(artifact):
        name = getattr(artifact, "__name__", str(artifact))
        try:
            sig = signature(artifact.__init__)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Cannot inspect __init__ of class {name}: {e}",
                ["Provide an explicit __init__ with type annotations"],
                {"artifact_name": name, "operation": "inspect_signature"},
            ) from e
        # strip `self`
        params = list(sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        sig = sig.replace(parameters=params)
        return name, sig, "class"
    elif callable(artifact):
        name = get_func_name(artifact, qualname=False)
        try:
            sig = signature(artifact)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Cannot inspect callable {name}: {e}",
                ["Ensure it's a Python callable with inspectable signature"],
                {"artifact_name": name, "operation": "inspect_signature"},
            ) from e
        return name, sig, "function"
    else:
        raise ValidationError(
            f"{artifact!r} is neither a class nor a callable",
            ["Pass a function or class object"],
            {"actual_type": type(artifact).__name__},
        )


def _to_pydantic_fields(sig: Signature) -> Dict[str, Tuple[type, Any]]:
    """Translate a Python signature into Pydantic `create_model` field specs.

    Notes:
        - Parameters of kind VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
          are skipped, with a DEBUG log.
        - Missing annotations default to `Any`.
        - Missing defaults become required fields (Ellipsis).
    """
    fields: Dict[str, Tuple[Any, Any]] = {}
    for p in sig.parameters.values():
        if p.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Skipping variadic parameter '%s' in %s", p.name, sig)
            continue

        ann = p.annotation if p.annotation is not _empty else Any
        default = p.default if p.default is not _empty else ...
        fields[p.name] = (ann, default)
    return fields


class InvocationScheme:
    """Compiled scheme around a callable/class and its Pydantic config model."""

    __slots__ = ("name", "artifact", "kind", "signature", "model")

    def __init__(
        self,
        *,
        name: str,
        artifact: Union[Callable[..., Any], Type[Any]],
        kind: Literal["function", "class"],
        signature: Signature,
        model: Type[BaseModel],
    ) -> None:
        self.name = name
        self.artifact = artifact
        self.kind = kind
        self.signature = signature
        self.model = model

    def new_config(self, **kwargs: Any) -> BaseModel:
        """Build and validate a config instance."""
        return self.model(**kwargs)

    def _to_kwargs(self, data: Union[BaseModel, Mapping[str, Any]]) -> Dict[str, Any]:
        if isinstance(data, BaseModel):
            # pydantic v2 uses model_dump; v1 uses dict()
            if hasattr(data, "model_dump"):
                return dict(data.model_dump())
            return dict(data.dict())
        if isinstance(data, Mapping):
            return dict(data)
        raise ValidationError(
            "Configuration must be a Pydantic model instance or a mapping",
            ["Pass a config built via scheme.model(...)"],
            {"artifact_name": self.name, "operation": "config_to_kwargs"},
        )

    def execute(self, data: Union[BaseModel, Mapping[str, Any]]) -> Any:
        """Call function or instantiate class with validated kwargs."""
        kwargs = self._to_kwargs(data)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Executing %s '%s' with kwargs=%r", self.kind, self.name, kwargs
            )
        if self.kind == "class":
            cls = self.artifact  # Type[Any]
            return cls(**kwargs)  # type: ignore[misc]
        func = self.artifact  # Callable[..., Any]
        return func(**kwargs)

    def __repr__(self) -> str:
        return f"InvocationScheme(name={self.name!r}, kind={self.kind}, model={self.model.__name__})"


class SchemeRegistry(MutableValidatorMixin[Hashable, InvocationScheme], ABC):
    """Registry of invocation schemes compiled from callables/classes.

    Key:   human-readable name (function __name__ / class __name__)
    Value: InvocationScheme (artifact + generated Pydantic model)
    """

    _repository: Dict[Hashable, InvocationScheme]
    __slots__ = ()

    @classmethod
    def _get_mapping(cls):
        return cls._repository

    @classmethod
    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        cls._repository = {}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Initialized SchemeRegistry subclass %s", cls.__name__)

    # ------------------- validation/coercion -------------------

    @classmethod
    def _internalize_artifact(cls, value: Any) -> InvocationScheme:
        name, sig, kind = _callable_signature(value)
        fields = _to_pydantic_fields(sig)
        model_name = f"{name}Config"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Creating pydantic model '%s' for %s '%s' with fields=%r",
                model_name,
                kind,
                name,
                tuple(fields.keys()),
            )
        try:
            model = create_model(model_name, __base__=BaseModel, **fields)  # type: ignore[arg-type]
        except Exception as e:
            raise ValidationError(
                f"Failed to synthesize config model for {name}: {e}",
                [
                    "Check parameter annotations and defaults",
                    "Avoid variadics if possible",
                ],
                {"artifact_name": name, "operation": "pydantic_create_model"},
            ) from e
        return InvocationScheme(
            name=name, artifact=value, kind=kind, signature=sig, model=model
        )

    @classmethod
    def _externalize_artifact(cls, value: InvocationScheme) -> InvocationScheme:
        return value

    @classmethod
    def _identifier_of(cls, item: Any) -> Hashable:
        return getattr(item, "__name__", str(item))

    # ------------------- convenience API -------------------

    @classmethod
    def get_model(cls, key: Union[str, InvocationScheme]) -> Type[BaseModel]:
        """Return the generated Pydantic model class."""
        if isinstance(key, InvocationScheme):
            return key.model
        scheme = cls.get_artifact(key)  # validation + presence handled upstream
        return scheme.model

    @classmethod
    def build_config(
        cls, key: Union[str, InvocationScheme], **kwargs: Any
    ) -> BaseModel:
        """Construct a validated config instance for `key`."""
        model = cls.get_model(key)
        return model(**kwargs)

    @classmethod
    def execute(
        cls,
        key: Union[str, InvocationScheme],
        data: Union[BaseModel, Mapping[str, Any]],
    ) -> Any:
        """Execute function or instantiate class with validated config."""
        scheme = key if isinstance(key, InvocationScheme) else cls.get_artifact(key)
        return scheme.execute(data)

    @classmethod
    def register_module_callables(
        cls,
        module: Any,
        *,
        include_functions: bool = True,
        include_classes: bool = True,
        raise_error: bool = True,
    ) -> Any:
        """Scan a module and register public callables/classes (no underscore names)."""
        if hasattr(module, "__all__"):
            module_items = [
                (nm, getattr(module, nm))
                for nm in getattr(module, "__all__")
                if isinstance(nm, str)
            ]
        else:
            module_items = [(nm, getattr(module, nm)) for nm in dir(module)]

        module_items = [
            (nm, obj)
            for nm, obj in module_items
            if obj is not None and not nm.startswith("_")
        ]

        ok, fail = 0, 0
        for name, obj in module_items:
            if include_functions and callable(obj) or include_classes and isclass(obj):
                try:
                    cls.register_artifact(name, obj)
                    ok += 1
                except ValidationError as e:
                    fail += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Skipping %s: %s", name, e)
                    if raise_error:
                        if hasattr(e, "context"):
                            e.context.update(
                                {
                                    "operation": "register_module_callables",
                                    "module_name": getattr(
                                        module, "__name__", str(module)
                                    ),
                                }
                            )
                        raise
        logger.info("%s: registered %d scheme(s), %d failed", cls.__name__, ok, fail)
        return module
