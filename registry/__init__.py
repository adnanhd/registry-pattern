# from . import utils, core, extra

from .utils import (
    RegistryError,
    RegistryLookupError,
    Registry,
    MutableRegistry,
    CoercionError,
    ConformanceError,
    InheritanceError,
    ValidationError,
)

# Alias TypeRegistry to ClassRegistry and ObjectRegistry to InstanceRegistry
from .core import TypeRegistry, FunctionalRegistry, ObjectRegistry
from .extra import ObjectConfigMap, ClassTracker, Wrapped

# Alias the factory functions to the names expected in __all__
from .utils.api import type_registry_factory, type_registry_decorator
from ._version import __version__

__all__ = [
    "RegistryError",
    "TypeRegistry",
    "ObjectRegistry",
    "FunctionalRegistry",
    "type_registry_factory",
    "type_registry_decorator",
    "CoercionError",
    "ConformanceError",
    "InheritanceError",
    "ValidationError",
]
