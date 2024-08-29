"""Metaclass for registering classes."""

from abc import ABC
from functools import partial
from typing import Any
from typing import Callable
from typing import Generic
from typing import Hashable
from typing import Type
from typing import TypeVar

from typing_extensions import Literal

from ._dev_utils import _def_checking
from ._dev_utils import get_module_members
from ._dev_utils import get_protocol
from ._dev_utils import get_subclasses
from ._validator import ConformanceError
from ._validator import InheritanceError
from ._validator import ValidationError
from ._validator import validate_class
from ._validator import validate_class_hierarchy
from ._validator import validate_class_structure
from .base import MutableRegistry

Cls = TypeVar("Cls")


class TypeRegistry(MutableRegistry[Hashable, Type[Cls]], ABC, Generic[Cls]):
    """Base Class for registering classes."""

    runtime_conformance_checking: Callable[[Type], Type] = _def_checking
    runtime_inheritance_checking: Callable[[Type], Type] = _def_checking
    __slots__ = ()

    @classmethod
    def __init_subclass__(cls, strict: bool = False, abstract: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._repository = dict()
        # validators: List[Callable[[Type], Type]] = [cls.validate_item]

        if abstract:
            cls.runtime_inheritance_checking = partial(
                validate_class_hierarchy, abc_class=cls
            )

        if strict:
            cls.runtime_conformance_checking = partial(
                validate_class_structure, expected_type=get_protocol(cls)
            )

    @classmethod
    def validate_item(cls, value: Type[Cls]) -> Type[Cls]:
        value = validate_class(value)
        value = cls.runtime_inheritance_checking(value)
        value = cls.runtime_conformance_checking(value)
        return value

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        try:
            validate_class(value)
        except ValidationError:
            print("no validation")
            return False

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
    def register_class(cls, subcls: Type[Cls]) -> Type[Cls]:
        """Register a subclass."""
        if hasattr(subcls, "__name__"):
            cls.add_registry_item(subcls.__name__, subcls)
        else:
            ValidationError(f"{subcls} is not a class or type")
        return subcls

    @classmethod
    def unregister_class(cls, subcls: Type[Cls]) -> Type[Cls]:
        """Unregister a subclass."""
        if hasattr(subcls, "__name__"):
            cls.del_registry_item(subcls.__name__)
        else:
            ValidationError(f"{subcls} is not a class or type")
        return subcls

    @classmethod
    def register_module_subclasses(cls, module, raise_error: bool = True):
        """Register all subclasses of a given module."""
        module_members = get_module_members(module)
        # module_members = filter(cls.__subclasscheck__, module_members)
        for obj in module_members:
            try:
                cls.register_class(obj)
            except AssertionError:
                print(f"Could not register {obj}")
            except ValidationError as e:
                if raise_error:
                    raise ValidationError(e)
                else:
                    print(e)
        return module

    @classmethod
    def register_subclasses(
        cls,
        supercls: Type[Cls],
        recursive: bool = False,
        raise_error: bool = True,
        mode: Literal["immediate", "deferred", "both"] = "both",
    ) -> Type[Cls]:
        """Register all subclasses of a given superclass."""
        # cls.register(supercls)
        if mode in {"immediate", "both"}:
            for subcls in get_subclasses(supercls):
                if recursive and get_subclasses(subcls):
                    cls.register_subclasses(subcls, recursive)
                try:
                    cls.register_class(subcls)
                except ValidationError as e:
                    if raise_error:
                        raise ValidationError(e)
                    else:
                        print(e)

        if mode in {"deferred", "both"}:
            mcs = type(supercls)
            register_class = cls.register_class

            class newmcs(mcs):
                def __new__(mcs, name, bases, attrs) -> Type[Cls]:
                    cls = super().__new__(mcs, name, bases, attrs)
                    try:
                        cls = register_class(cls)
                    except Exception as e:
                        print(e)
                    return cls

            newmcs.__name__ = mcs.__name__
            newmcs.__qualname__ = mcs.__qualname__
            newmcs.__module__ = mcs.__module__
            newmcs.__doc__ = mcs.__doc__

            supercls = newmcs(supercls.__name__, (supercls,), {})
        return supercls
