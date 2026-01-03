r"""Container mixin for DI / IoC object graph construction.

This module provides `ContainerMixin`, a mixin that adds recursive object
instantiation from nested configs using the BuildCfg envelope schema.

Features:
  - Multi-repo support via _repos class variable
  - Context injection (ctx parameter) for cross-references
  - Recursive nested config resolution
  - Extras separation (unknown kwargs -> meta._unused_data)
  - Pydantic schema validation per (repo, type)

Usage::

    class ModelRegistry(TypeRegistry[nn.Module], ContainerMixin):
        pass

    class TransformRegistry(TypeRegistry[Any], ContainerMixin):
        pass

    # Configure repos
    ContainerMixin.configure_repos({
        "models": ModelRegistry,
        "transforms": TransformRegistry,
        "default": ModelRegistry,
    })

    # Build from config
    cfg = BuildCfg(type="ResNet18", repo="models", data={"num_classes": 10})
    model = ContainerMixin.build_cfg(cfg)
"""

from __future__ import annotations

import inspect
import logging
from inspect import Parameter
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Hashable,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ConfigDict, create_model

from ..container import BuildCfg, is_build_cfg, normalize_cfg
from ..utils import ValidationError, get_callable_signature, pydantic_to_dict
from .validator import MutableValidatorMixin

logger = logging.getLogger(__name__)

KeyType = TypeVar("KeyType", bound=Hashable)
ValType = TypeVar("ValType")

__all__ = ["ContainerMixin", "RegistryFactorizorMixin"]


class ParamsBase(BaseModel):
    """Base model for auto-extracted parameter schemas."""

    model_config = ConfigDict(extra="allow")


class ContainerMixin(
    MutableValidatorMixin[KeyType, ValType], Generic[KeyType, ValType]
):
    """Mixin for DI container / IoC object graph construction.

    Class Variables:
        _repos: Mapping of repo names to registry classes.
        _ctx: Shared context for cross-references between built objects.
        _schemetry: Mapping of identifiers to Pydantic parameter models.

    Methods:
        configure_repos: Set up multi-repo mapping.
        build_cfg: Build an object from a BuildCfg.
        build_value: Recursively resolve nested configs.
        build_named: Build and store in context.
        clear_context: Reset the shared context.
    """

    _repos: ClassVar[Dict[str, type]] = {}
    _ctx: ClassVar[Dict[str, Any]] = {}
    _schemetry: ClassVar[Dict[str, Type[BaseModel]]] = {}

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own scheme storage
        cls._schemetry = {}
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Initialized ContainerMixin for %s", cls.__name__)

    # -------------------------------------------------------------------------
    # Repo Configuration
    # -------------------------------------------------------------------------

    @classmethod
    def configure_repos(cls, repos: Dict[str, type]) -> None:
        """Configure the multi-repo mapping.

        Args:
            repos: Mapping of repo names to registry classes.
        """
        cls._repos = dict(repos)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Configured repos: %s", list(repos.keys()))

    @classmethod
    def get_repo(cls, name: str) -> type:
        """Get a registry class by repo name.

        Args:
            name: The repo name.

        Returns:
            The registry class.

        Raises:
            ValidationError: If repo not found.
        """
        if name not in cls._repos:
            available = list(cls._repos.keys()) or ["<none>"]
            raise ValidationError(
                f"Unknown repo '{name}'",
                [f"Available repos: {', '.join(available)}"],
                {"repo": name, "available_repos": available},
            )
        return cls._repos[name]

    # -------------------------------------------------------------------------
    # Context Management
    # -------------------------------------------------------------------------

    @classmethod
    def clear_context(cls) -> None:
        """Clear the shared context."""
        cls._ctx.clear()

    @classmethod
    def get_context(cls) -> Dict[str, Any]:
        """Get the shared context."""
        return cls._ctx

    # -------------------------------------------------------------------------
    # Schema Extraction
    # -------------------------------------------------------------------------

    @classmethod
    def _extract_params_model(cls, artifact: Any) -> Type[BaseModel]:
        """Extract a Pydantic params model from artifact signature.

        Args:
            artifact: Class or callable to extract signature from.

        Returns:
            A dynamically created Pydantic model.
        """
        name, sig, params = get_callable_signature(artifact)

        fields: Dict[str, Any] = {}
        for param in params:
            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                continue
            # Skip 'ctx' parameter - it's injected, not from config
            if param.name == "ctx":
                continue

            annotation = (
                param.annotation if param.annotation is not Parameter.empty else Any
            )
            default = param.default if param.default is not Parameter.empty else ...
            fields[param.name] = (annotation, default)

        model_name = f"{name}Params"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Extracting params model '%s' with fields: %s",
                model_name,
                list(fields.keys()),
            )

        return create_model(model_name, __base__=ParamsBase, **fields)  # type: ignore[call-overload]

    @classmethod
    def _save_scheme(cls, identifier: str, scheme: Type[BaseModel]) -> None:
        """Save a params model for an identifier."""
        cls._schemetry[identifier] = scheme

    @classmethod
    def _get_scheme(cls, identifier: str) -> Optional[Type[BaseModel]]:
        """Get a params model for an identifier."""
        return cls._schemetry.get(identifier)

    # -------------------------------------------------------------------------
    # Registration Override
    # -------------------------------------------------------------------------

    @classmethod
    def _internalize_artifact(cls, value: Any) -> ValType:
        """Override to extract and save params model on registration."""
        artifact = super()._internalize_artifact(value)

        # Extract and save scheme
        try:
            identifier = str(cls._identifier_of(artifact))
            # Check if a custom params_model was provided (via register_artifact)
            if not cls._get_scheme(identifier):
                scheme = cls._extract_params_model(artifact)
                cls._save_scheme(identifier, scheme)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Saved auto-extracted scheme for %s", identifier)
        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Could not extract scheme for %r: %s", value, e)

        return artifact

    # -------------------------------------------------------------------------
    # Build Methods
    # -------------------------------------------------------------------------

    @classmethod
    def build_value(cls, value: Any) -> Any:
        """Recursively resolve nested configs.

        Args:
            value: Any value that may contain nested BuildCfg.

        Returns:
            The resolved value with all nested configs built.
        """
        if isinstance(value, BuildCfg):
            return cls.build_cfg(value)
        if is_build_cfg(value):
            return cls.build_cfg(normalize_cfg(value))
        if isinstance(value, dict):
            return {k: cls.build_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls.build_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cls.build_value(v) for v in value)
        return value

    @classmethod
    def build_named(cls, key: str, cfg: Union[BuildCfg, Dict[str, Any]]) -> Any:
        """Build an object and store it in the context.

        Args:
            key: Context key to store the built object under.
            cfg: BuildCfg or raw dict config.

        Returns:
            The built object.
        """
        if not isinstance(cfg, BuildCfg):
            cfg = normalize_cfg(cfg)
        obj = cls.build_cfg(cfg)
        cls._ctx[key] = obj
        return obj

    @classmethod
    def build_cfg(cls, cfg: Union[BuildCfg, Dict[str, Any]]) -> Any:
        """Build an object from a BuildCfg.

        This is the main entry point for object graph construction.

        Args:
            cfg: BuildCfg or raw dict config.

        Returns:
            The built object with meta attached.

        Raises:
            ValidationError: If building fails.
        """
        if not isinstance(cfg, BuildCfg):
            cfg = normalize_cfg(cfg)

        # 1. Resolve repo -> registry
        repo_name = cfg.repo
        if repo_name in cls._repos:
            registry = cls._repos[repo_name]
        elif repo_name == "default" and not cls._repos:
            # Use self as registry if no repos configured
            registry = cls
        else:
            registry = cls.get_repo(repo_name)

        # 2. Get artifact from registry
        artifact = registry.get_artifact(cfg.type)

        # 3. Get params model and validate data
        params_model = (
            registry._get_scheme(cfg.type) if hasattr(registry, "_get_scheme") else None
        )
        validated_data = dict(cfg.data)
        unused_data: Dict[str, Any] = {}

        if params_model is not None:
            try:
                # Get the known field names from the model
                known_fields = set(params_model.model_fields.keys())

                # Separate known fields from extras BEFORE validation
                data_for_validation = {}
                for k, v in cfg.data.items():
                    if k in known_fields:
                        data_for_validation[k] = v
                    else:
                        unused_data[k] = v

                # Validate only the known fields
                validated = params_model.model_validate(data_for_validation)

                # Also check for any pydantic extras (if model allows extra="allow")
                extras = getattr(validated, "__pydantic_extra__", {}) or {}
                if extras:
                    unused_data.update(dict(extras))

                # Get validated/coerced fields
                validated_data = {
                    k: getattr(validated, k)
                    for k in known_fields
                    if hasattr(validated, k)
                }
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Schema validation failed for %s: %s", cfg.type, e)
                # Fall back to raw data

        # 4. Recursively build nested configs
        data = {k: cls.build_value(v) for k, v in validated_data.items()}

        # 5. Inspect builder signature for ctx and filter kwargs
        if inspect.isclass(artifact):
            try:
                sig = inspect.signature(artifact.__init__)
            except (ValueError, TypeError):
                sig = None
        elif callable(artifact):
            try:
                sig = inspect.signature(artifact)
            except (ValueError, TypeError):
                sig = None
        else:
            sig = None

        pass_kwargs: Dict[str, Any] = {}
        if sig is not None:
            params = sig.parameters
            accepts_var_kw = any(
                p.kind == Parameter.VAR_KEYWORD for p in params.values()
            )
            allowed = {
                name
                for name, p in params.items()
                if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
                and name != "self"
            }

            for k, v in data.items():
                if accepts_var_kw or k in allowed:
                    pass_kwargs[k] = v
                else:
                    unused_data[k] = v

            # Inject ctx if builder accepts it
            if "ctx" in params:
                pass_kwargs["ctx"] = cls._ctx
        else:
            pass_kwargs = data

        # 6. Update meta with unused data
        meta = dict(cfg.meta)
        if unused_data:
            meta.setdefault("_unused_data", {}).update(unused_data)

        # 7. Instantiate
        try:
            if inspect.isclass(artifact):
                obj = artifact(**pass_kwargs)
            else:
                obj = artifact(**pass_kwargs)
        except TypeError as e:
            raise ValidationError(
                f"Build failed for repo='{repo_name}' type='{cfg.type}'",
                [
                    f"Error: {e}",
                    f"Passed kwargs: {list(pass_kwargs.keys())}",
                ],
                {
                    "repo": repo_name,
                    "type": cfg.type,
                    "kwargs": list(pass_kwargs.keys()),
                },
            ) from e

        # 8. Attach meta
        if meta:
            try:
                setattr(obj, "__meta__", meta)
            except Exception:
                pass

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Built %s from repo=%s type=%s",
                type(obj).__name__,
                repo_name,
                cfg.type,
            )

        return obj


# Alias for backward compatibility
RegistryFactorizorMixin = ContainerMixin
