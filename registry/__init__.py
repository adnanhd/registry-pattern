from ._validator import CoercionError
from ._validator import ConformanceError
from ._validator import InheritanceError
from ._validator import ValidationError
from .base import RegistryError
from .fnc_registry import FunctionalRegistry
from .obj_registry import ObjectRegistry
from .typ_registry import TypeRegistry
from .api import type_registry_factory, type_registry_decorator

__all__ = [
    "RegistryError",
    "ClassRegistry",
    "InstanceRegistry",
    # "InstanceKeyRegistry",
    "FunctionalRegistry",
    "make_class_registry",
    "make_functional_registry",
    "CoercionError",
    "ConformanceError",
    "InheritanceError",
    "ValidationError",
]

# TODO: coersion, cache, proxy (instance)
