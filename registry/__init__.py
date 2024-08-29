from ._validator import CoercionError
from ._validator import ConformanceError
from ._validator import InheritanceError
from ._validator import ValidationError
from .base import RegistryError
from .fnc_registry import FunctionalRegistry

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
