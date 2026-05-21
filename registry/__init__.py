r"""Registry Pattern -- recursive factory + name-based registries.

Core API:
    TypeRegistry, FunctionalRegistry, BuildCfg, Buildable
    build, resolve, validate, serialize
    ValidationError, RegistryError, CoercionError, ConformanceError, InheritanceError

Optional submodules:
    from registry.engines import ConfigFileEngine                # [yaml] for YAML
    from registry.reporters import OpenTelemetryReporter         # [otel]
    from registry.experimental.torch_compat import ...           # [torch]

Usage::

    from registry import TypeRegistry, build

    class ModelRegistry(TypeRegistry[nn.Module], repo="models"):
        pass

    @ModelRegistry.register_artifact
    class ResNet18(nn.Module):
        def __init__(self, num_classes: int = 10): ...

    model = build({"type": "ResNet18", "data": {"num_classes": 10}})
"""

from ._version import __version__, get_debug_info, get_version_info, print_version_info
from .container import BuildCfg, is_build_cfg, normalize_cfg
from .factory import build, resolve, serialize, validate
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
    # Version utilities
    "__version__",
    "get_version_info",
    "get_debug_info",
    "print_version_info",
    # Core registries (pipeline-internal registries -- ValidatorRegistry /
    # SerializerRegistry -- live in their submodules; import from there if needed.)
    "TypeRegistry",
    "FunctionalRegistry",
    # Envelope
    "BuildCfg",
    "is_build_cfg",
    "normalize_cfg",
    # Factory
    "build",
    "resolve",
    "validate",
    "serialize",
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
