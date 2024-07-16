"""Metaclass for registering classes."""

from abc import ABC
from typing import TypeVar, ClassVar, Any, Callable, Hashable, Generic
from functools import lru_cache, partial
from .base import BaseMutableRegistry
from ._dev_utils import get_protocol, compose, get_subclasses, get_module_members
from ._validator import validate_class, validate_init
from ._validator import validate_class_hierarchy, validate_class_structure


Prtcl = TypeVar("Prtcl")


class _BaseClassRegistry(BaseMutableRegistry[Hashable, type[Prtcl]], ABC, Generic[Prtcl]):
    """Base Class for registering classes."""
    ignore_structural_subtyping: ClassVar[bool] = False
    ignore_abcnominal_subtyping: ClassVar[bool] = False
    __slots__ = ()

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        return issubclass(value, cls)

    def register_class(self, subcls: type[Prtcl]) -> type[Prtcl]:
        """Register a subclass."""
        self.add_registry_item(subcls.__name__, subcls)
        return subcls

    def unregister_class(self, subcls: type[Prtcl]) -> type[Prtcl]:
        """Unregister a subclass."""
        self.del_registry_item(subcls.__name__)
        return subcls

    def register_module_subclasses(self, module):
        """Register all subclasses of a given module."""
        module_members = get_module_members(module)
        # module_members = filter(self.__subclasscheck__, module_members)
        for obj in module_members:
            try:
                self.register_class(obj)
            except AssertionError:
                print(f"Could not register {obj}")
        return module

    def register_subclasses(self, supercls: type[Prtcl], recursive: bool = False) -> type[Prtcl]:
        """Register all subclasses of a given superclass."""
        # self.register(supercls)
        for subcls in get_subclasses(supercls):
            if recursive and get_subclasses(subcls):
                self.register_subclasses(subcls, recursive)
            self.register_class(subcls)
        return supercls


class ClassRegistry(_BaseClassRegistry[Prtcl], Generic[Prtcl]):
    """Metaclass for registering classes."""
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validators = [validate_class]

        if not cls.ignore_abcnominal_subtyping:
            @validators.append
            def _validate_class_hierarchy(subcls: type) -> type:
                return validate_class_hierarchy(subcls, abc_class=cls)

        if not cls.ignore_structural_subtyping:
            @validators.append
            def _validate_class_structure(subcls: type) -> type:
                return validate_class_structure(subcls, expected_type=get_protocol(cls))

        cls._validate_item = compose(*validators)

    @classmethod
    def _validate_item(cls, value: Any) -> type[Prtcl]:
        ...


class PyClassRegistry(_BaseClassRegistry[Prtcl], Generic[Prtcl]):
    """Metaclass for registering classes."""
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validators = [validate_class]

        if not cls.ignore_abcnominal_subtyping:
            @validators.append
            def _validate_class_hierarchy(subcls: type) -> type:
                return validate_class_hierarchy(subcls, abc_class=cls)

        if not cls.ignore_structural_subtyping:
            @validators.append
            def _validate_class_structure(subcls: type) -> type:
                return validate_class_structure(subcls, expected_type=get_protocol(cls))

        cls._validate_item = compose(*validators)

        cacher = lru_cache(maxsize=16, typed=False)
        get_registry_item = compose(validate_init,
                                    cls.get_registry_item.__wrapped__)
        cls.get_registry_item = cacher(get_registry_item)

    '''
    @lru_cache(maxsize=16, typed=False)
    def get_registry_item(self, key: Hashable) -> Prtcl:
        """Return a registered function."""
        return validate_init(super().get_registry_item.__wrapped__(key))
    '''
