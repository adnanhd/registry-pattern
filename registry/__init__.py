r"""Registry Pattern -- recursive factory + name-based registries.

Core API:
    TypeRegistry, FunctionalRegistry, BuildCfg, Buildable
    build, resolve, validate, serialize
    ValidationError, RegistryError, CoercionError, ConformanceError, InheritanceError

Observability ships as buses (base class + attach / detach / emit); the concrete
batteries are opt-in:
    from registry.extra.meters import CPUMeter, MemoryMeter
    from registry.extra.reporters import JournalReporter, OpenTelemetryReporter  # [otel]

Optional submodules:
    from registry.engines import ConfigFileEngine                # [yaml] for YAML
    from examples.torch_compat import ...                        # [torch] example

Usage::

    from registry import TypeRegistry, build

    class Shape:
        def area(self) -> float: ...

    class ShapeRegistry(TypeRegistry[Shape], repo="shapes"):
        pass

    @ShapeRegistry.register_artifact
    class Circle(Shape):
        def __init__(self, radius: float = 1.0): ...

    circle = build({"type": "Circle", "data": {"radius": 2.0}})
"""

from __future__ import annotations

from ._version import __version__, get_debug_info, get_version_info, print_version_info
from .container import BuildCfg, is_build_cfg, normalize_cfg
from .factory import build, resolve, serialize, validate
from .fnc_registry import FunctionalRegistry
from .meters import (
    FactoryMeter,
    attach_meter,
    detach_meter,
    meters,
)
from .reporters import (
    FactoryReporter,
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
    # Meter bus (concrete meters live in registry.extra.meters)
    "FactoryMeter",
    "attach_meter",
    "detach_meter",
    "meters",
    # Reporter bus (concrete reporters live in registry.extra.reporters)
    "FactoryReporter",
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
