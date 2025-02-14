r"""Baseclass for registering objects.

This module defines the ObjectRegistry class, a base class for registering
object instances. It uses a weak value dictionary as the internal repository
to avoid strong references to the registered objects. The registry supports
runtime validations of instance inheritance and structure via configurable
validators.

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph ObjectRegistry {
    "MutableRegistry" -> "ObjectRegistry";
    "ABC" -> "ObjectRegistry";
    "Generic" -> "ObjectRegistry";
}
\enddot
"""

import sys
import weakref
from abc import ABC
from functools import partial
from typing_compat import Any, Callable, Dict, Generic, Hashable, Type, TypeVar

from ..utils import (
    # base
    MutableRegistry,
    # _dev_utils
    _def_checking,
    get_protocol,
    # _validator
    ConformanceError,
    InheritanceError,
    validate_instance_hierarchy,
    validate_instance_structure,
)

# Configuration type for object registries (not used directly in this module).
CfgT = Dict[str, Any]  # Alternatively: TypeVar("CfgT", bound=dict[str, Any])
# Type variable representing the object type to be registered.
ObjT = TypeVar("ObjT")

# Define a generic alias for a WeakValueDictionary that holds objects of type ObjT.
if sys.version_info >= (3, 9):
    WeakValueDictionaryT = weakref.WeakValueDictionary[Hashable, ObjT]
else:
    WeakValueDictionaryT = weakref.WeakValueDictionary


class ObjectRegistry(MutableRegistry[Hashable, ObjT], ABC, Generic[ObjT]):
    """
    Base class for registering instances.

    This registry stores instances in a weak value dictionary so that the
    registry does not keep objects alive. It also supports runtime validations
    to ensure that registered instances conform to the expected inheritance and
    structural requirements.

    Attributes:
        runtime_conformance_checking (Callable[..., Any]):
            A callable used to check that an instance conforms to an expected structure.
        runtime_inheritance_checking (Callable[..., Any]):
            A callable used to verify that an instance has the proper inheritance.
    """

    # Default runtime checking functions; can be overridden in subclasses.
    runtime_conformance_checking: Callable[..., Any] = _def_checking
    runtime_inheritance_checking: Callable[..., Any] = _def_checking
    __slots__ = ()

    @classmethod
    def __init_subclass__(cls, strict: bool = False, abstract: bool = False) -> None:
        """
        Initialize a subclass of ObjectRegistry.

        This method initializes the registry repository and configures the runtime
        validation functions based on the provided flags. If 'abstract' is True, the
        registry will enforce inheritance checks; if 'strict' is True, it will enforce
        structure (conformance) checks.

        Parameters:
            strict (bool): If True, enforce strict structure validation using
                           validate_instance_structure.
            abstract (bool): If True, enforce inheritance validation using
                             validate_instance_hierarchy.
        """
        super().__init_subclass__()
        # Set up the internal repository as a weak value dictionary.
        cls._repository = WeakValueDictionaryT()

        # Configure runtime inheritance checking if the registry is for abstract classes.
        if abstract:
            cls.runtime_inheritance_checking = partial(
                validate_instance_hierarchy, expected_type=cls
            )

        # Configure runtime conformance checking in strict mode.
        if strict:
            cls.runtime_conformance_checking = partial(
                validate_instance_structure, expected_type=get_protocol(cls)
            )

    @classmethod
    def validate_artifact(cls, value: ObjT) -> ObjT:
        """
        Validate an instance before registration.

        This method applies both inheritance and structure (conformance) checks to
        the given instance.

        Parameters:
            value (ObjT): The instance to validate.

        Returns:
            ObjT: The validated instance.

        Raises:
            InheritanceError: If the instance does not meet the inheritance requirements.
            ConformanceError: If the instance does not conform to the expected structure.
        """
        cls.runtime_inheritance_checking(value)
        cls.runtime_conformance_checking(value)
        return value

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        """
        Custom subclass check for runtime validation of an instance.

        This method uses the configured runtime validation functions to determine
        whether the given value meets the criteria for being considered a subclass.

        Parameters:
            value (Any): The instance to check.

        Returns:
            bool: True if the value passes all validations; False otherwise.
        """
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
        """
        Register an instance in the registry.

        The instance is stored in the registry using a key generated by
        validate_artifact_id (typically a unique identifier).

        Parameters:
            instance (ObjT): The instance to register.

        Returns:
            ObjT: The registered instance.
        """
        cls.register_artifact(instance, instance)
        return instance

    @classmethod
    def unregister_instance(cls, instance: ObjT) -> ObjT:
        """
        Unregister an instance from the registry.

        The instance is removed from the registry using the key generated by
        validate_artifact_id.

        Parameters:
            instance (ObjT): The instance to unregister.

        Returns:
            ObjT: The unregistered instance.
        """
        cls.unregister_artifacts(instance)
        return instance

    @classmethod
    def validate_artifact_id(cls, key: ObjT) -> str:
        """
        Generate a unique key for an instance.

        This default implementation uses the object's id (in hexadecimal form).

        Parameters:
            key (ObjT): The instance to generate a key for.

        Returns:
            str: The generated key.
        """
        return hex(id(key))

    @classmethod
    def register_class_instances(cls, supercls: Type[ObjT]) -> Type[ObjT]:
        """
        Create a tracked version of a class so that all its instances are registered.

        This function dynamically creates a new metaclass that wraps the __call__
        method of the provided class. Each time an instance is created, it is
        automatically registered in the registry.

        Parameters:
            supercls (Type[ObjT]): The class to track.

        Returns:
            Type[ObjT]: A new class that behaves like the original but automatically
                        registers its instances.
        """
        meta: Type[Any] = type(supercls)

        class newmcs(meta):
            def __call__(self, *args: Any, **kwds: Any) -> ObjT:
                # Create a new instance using the original __call__.
                obj = super().__call__(*args, **kwds)
                try:
                    # Register the new instance.
                    obj = cls.register_instance(obj)
                except Exception as e:
                    print(e)
                return obj

        # Copy meta attributes from the original metaclass.
        newmcs.__name__ = meta.__name__
        newmcs.__qualname__ = meta.__qualname__
        newmcs.__module__ = meta.__module__
        newmcs.__doc__ = meta.__doc__

        # Create and return the new tracked class.
        return newmcs(supercls.__name__, (supercls,), {})
