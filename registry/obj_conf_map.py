from typing import TypeVar, Generic, Hashable, Any, Dict, Callable, List, Type
from functools import partial, wraps
from weakref import WeakKeyDictionary

from .base import BaseMutableRegistry
from ._validator import validate_instance_hierarchy, validate_instance_structure
from ._dev_utils import _def_checking, get_protocol

K = TypeVar("K", bound=Hashable)
CfgT = Dict[str, Any]  # TypeVar("CfgT", bound=dict[str, Any])


class ObjectConfigMap(BaseMutableRegistry[K, CfgT], Generic[K]):
    """Functional registry for registering instances."""

    runtime_conformance_checking: Callable[..., Any] = _def_checking
    runtime_inheritance_checking: Callable[..., Any] = _def_checking
    __slots__ = ()

    @classmethod
    def __init_subclass__(cls, strict: bool = False, abstract: bool = False):
        print("foo")
        super().__init_subclass__()
        cls._repository = WeakKeyDictionary[K, CfgT]()

        if abstract:
            cls.runtime_inheritance_checking = partial(
                validate_instance_hierarchy, expected_type=cls
            )

        if strict:
            cls.runtime_conformance_checking = partial(
                validate_instance_structure, expected_type=get_protocol(cls)
            )

    @classmethod
    def validate_item(cls, value: dict) -> CfgT:
        if not isinstance(value, dict):
            raise TypeError(f"{value} is not a dict")
        return value

    @classmethod
    def validate_key(cls, key: Any) -> K:
        cls.runtime_inheritance_checking(key)
        cls.runtime_conformance_checking(key)
        return key

    @classmethod
    def register_instance(cls, obj: K, cfg: Dict[str, Any]) -> K:
        """Track a class."""
        cls.add_registry_item(obj, cfg)
        return obj

    @classmethod
    def unregister_instance(cls, obj: K) -> K:
        """Track a class."""
        cls.del_registry_item(obj)
        return obj

    @classmethod
    def register_class_instances(cls, supercls: Type[K]) -> Type[K]:
        """Track a class."""
        mcs = type(supercls)

        class newmcs(mcs):
            def __call__(self, **kwds: Any) -> K:
                obj = super().__call__(**kwds)
                return cls.register_instance(obj, kwds)

        newmcs.__name__ = mcs.__name__
        newmcs.__qualname__ = mcs.__qualname__
        newmcs.__module__ = mcs.__module__
        newmcs.__doc__ = mcs.__doc__
        return newmcs(supercls.__name__, (supercls,), {})

    @classmethod
    def register_builder(cls, func: Callable[..., K]) -> Callable[..., K]:
        @wraps(func)
        def wrapper(**kwds):
            obj = func(**kwds)
            cls.register_instance(obj, kwds)
            return obj

        return wrapper
