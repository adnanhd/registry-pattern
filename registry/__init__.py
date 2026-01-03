r"""Registry Pattern - DI Container / IoC Framework.

This package provides a registry pattern implementation with dependency injection
capabilities for building object graphs from nested configurations.

Core Registries:
    TypeRegistry: Class registry with inheritance/protocol checks.
    FunctionalRegistry: Function registry with signature validation.
    SchemeRegistry: Pydantic config scheme registry.

DI Container:
    BuildCfg: Configuration envelope (type/repo/data/meta).
    ContainerMixin: Mixin for recursive object graph construction.

Engines:
    ConfigFileEngine: Registry for config file loaders (json, yaml, toml).
    SocketEngine: Registry for network handlers (rpc, http).

Exceptions:
    ValidationError: Base validation error with suggestions and context.
    RegistryError: Key-related mapping errors.
    CoercionError: Value coercion failures.
    ConformanceError: Signature/protocol conformance failures.
    InheritanceError: Class inheritance failures.

Usage::

    from registry import TypeRegistry, BuildCfg, ContainerMixin

    class ModelRegistry(TypeRegistry[nn.Module], ContainerMixin):
        pass

    @ModelRegistry.register_artifact
    class ResNet18:
        def __init__(self, num_classes: int = 10): ...

    # Configure repos and build
    ContainerMixin.configure_repos({"models": ModelRegistry})

    cfg = BuildCfg(
        type="ResNet18",
        repo="models",
        data={"num_classes": 10},
    )
    model = ContainerMixin.build_cfg(cfg)
"""

from ._version import __version__
from .container import BuildCfg, is_build_cfg, normalize_cfg
from .engines import ConfigFileEngine, SocketEngine
from .fnc_registry import FunctionalRegistry
from .mixin import ContainerMixin, RegistryFactorizorMixin
from .sch_registry import SchemeRegistry
from .typ_registry import TypeRegistry
from .type_guard import Buildable, BuildableValidator
from .utils import (
    CoercionError,
    ConformanceError,
    InheritanceError,
    RegistryError,
    ValidationError,
)

__all__ = [
    # Version
    "__version__",
    # Core registries
    "TypeRegistry",
    "FunctionalRegistry",
    "SchemeRegistry",
    # DI Container
    "BuildCfg",
    "ContainerMixin",
    "RegistryFactorizorMixin",
    "is_build_cfg",
    "normalize_cfg",
    # Engines
    "ConfigFileEngine",
    "SocketEngine",
    # Type Guard
    "Buildable",
    "BuildableValidator",
    # Exceptions
    "ValidationError",
    "RegistryError",
    "CoercionError",
    "ConformanceError",
    "InheritanceError",
]
