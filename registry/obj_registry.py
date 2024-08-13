"""Object registry pattern."""

from abc import ABC
from functools import lru_cache, partial
import weakref
from typing import (
    Type,
    TypeVar,
    ClassVar,
    MutableMapping,
    Generic,
    Any,
    cast,
    Hashable,
    Callable,
    Dict,
    List,
)
from .base import MutableRegistry
from ._validator import validate_instance_hierarchy, validate_instance_structure
from ._validator import ValidationError, ConformanceError, InheritanceError
from ._dev_utils import compose, get_protocol, _def_checking


CfgT = Dict[str, Any]  # TypeVar("CfgT", bound=dict[str, Any])
ObjT = TypeVar("ObjT")


class ObjectRegistry(MutableRegistry[Hashable, ObjT], ABC, Generic[ObjT]):
    """Base class for registering instances."""

    runtime_conformance_checking: Callable[..., Any] = _def_checking
    runtime_inheritance_checking: Callable[..., Any] = _def_checking
    __slots__ = ()

    @classmethod
    def __init_subclass__(cls, strict: bool = False, abstract: bool = False):
        super().__init_subclass__()
        cls._repository = weakref.WeakValueDictionary[Hashable, ObjT]()

        if abstract:
            cls.runtime_inheritance_checking = partial(
                validate_instance_hierarchy, expected_type=cls
            )

        if strict:
            cls.runtime_conformance_checking = partial(
                validate_instance_structure, expected_type=get_protocol(cls)
            )

    @classmethod
    def validate_item(cls, value: ObjT) -> ObjT:
        cls.runtime_inheritance_checking(value)
        cls.runtime_conformance_checking(value)
        return value

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        try:
            cls.runtime_inheritance_checking(value)
        except InheritanceError:
            print("no inheritaion")
            return False

        try:
            cls.runtime_conformance_checking(value)
        except ConformanceError:
            print("no conformation")
            return False

        return True

    @classmethod
    def register_instance(cls, instance: ObjT) -> ObjT:
        """Register a subclass."""
        cls.add_registry_item(instance, instance)
        return instance

    @classmethod
    def unregister_instance(cls, instance: ObjT) -> ObjT:
        """Unregister a subclass."""
        cls.del_registry_item(instance)
        return instance

    @classmethod
    def validate_key(cls, key: ObjT) -> str:
        return hex(id(key))

    @classmethod
    def register_class_instances(cls, supercls: Type[ObjT]) -> Type[ObjT]:
        """Track a class."""
        mcs = type(supercls)

        class newmcs(mcs):
            def __call__(self, *args: Any, **kwds: Any) -> ObjT:
                obj = super().__call__(*args, **kwds)
                return cls.register_instance(obj)

        newmcs.__name__ = mcs.__name__
        newmcs.__qualname__ = mcs.__qualname__
        newmcs.__module__ = mcs.__module__
        newmcs.__doc__ = mcs.__doc__
        return newmcs(supercls.__name__, (supercls,), {})
