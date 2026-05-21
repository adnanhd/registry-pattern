r"""Registry Pattern - DI Container / IoC Framework.

Core API:
    TypeRegistry, FunctionalRegistry, SchemeRegistry
    BuildCfg, ContainerMixin, Buildable
    ValidationError, RegistryError, CoercionError, ConformanceError, InheritanceError

Optional features live in submodules and require extras:
    from registry.engines import ConfigFileEngine, SocketEngine
        # SocketEngine.rpc requires pip install 'registry-pattern[rpc]'
        # SocketEngine.http and ConfigFileEngine.yaml require [http] and [yaml]
    from registry.remote_storage import RemoteStorageProxy   # needs [http]
    python -m registry server ...                            # needs [server]

Usage::

    from registry import TypeRegistry, BuildCfg, ContainerMixin

    class ModelRegistry(TypeRegistry[nn.Module], ContainerMixin):
        pass

    @ModelRegistry.register_artifact
    class ResNet18:
        def __init__(self, num_classes: int = 10): ...

    ContainerMixin.configure_repos({"models": ModelRegistry})
    cfg = BuildCfg(type="ResNet18", repo="models", data={"num_classes": 10})
    model = ContainerMixin.build_cfg(cfg)
"""

from ._version import __version__, get_debug_info, get_version_info, print_version_info
from .container import BuildCfg, is_build_cfg, normalize_cfg
from .factory import build, resolve
from .fnc_registry import FunctionalRegistry
from .meters import (
    CPUMeter,
    FactoryMeter,
    HeapMeter,
    IOMeter,
    LifetimeMeter,
    MemoryMeter,
    NetworkMeter,
    RecursionMeter,
    attach_meter,
    detach_meter,
    meters,
)
from .reporters import (
    FactoryReporter,
    HTTPDashboardReporter,
    JournalReporter,
    OpenTelemetryReporter,
    attach_reporter,
    detach_reporter,
    reporters,
)
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
from .validators import ValidatorRegistry

__all__ = [
    # Version utilities
    "__version__",
    "get_version_info",
    "get_debug_info",
    "print_version_info",
    # Core registries
    "TypeRegistry",
    "FunctionalRegistry",
    "SchemeRegistry",
    "ValidatorRegistry",
    # DI Container
    "BuildCfg",
    "ContainerMixin",
    "RegistryFactorizorMixin",
    "is_build_cfg",
    "normalize_cfg",
    # Factory
    "build",
    "resolve",
    # Meters (measure -> meta)
    "FactoryMeter",
    "LifetimeMeter",
    "CPUMeter",
    "MemoryMeter",
    "IOMeter",
    "NetworkMeter",
    "HeapMeter",
    "RecursionMeter",
    "attach_meter",
    "detach_meter",
    "meters",
    # Reporters (ship event externally)
    "FactoryReporter",
    "JournalReporter",
    "HTTPDashboardReporter",
    "OpenTelemetryReporter",
    "attach_reporter",
    "detach_reporter",
    "reporters",
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
