from .base import RegistryError
from .api import type_registry_decorator, type_registry_factory
from .api import functional_registry_decorator, functional_registry_factory
from .typ_registry import TypeRegistry
from .obj_registry import ObjectRegistry
from .obj_conf_map import ObjectConfigMap
from .fnc_registry import FunctionalRegistry
from ._validator import CoercionError, ValidationError
from ._validator import InheritanceError, ConformanceError

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
