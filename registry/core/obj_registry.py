r"""Baseclass for registering objects.

This module defines the ObjectRegistry class, a base class for registering
object instances. It uses a standard dictionary as the internal repository
and handles weak references in the seal and probe methods. The registry supports
runtime validations of instance inheritance and structure via configurable
validators.

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph ObjectRegistry {
    "MutableRegistryValidatorMixin" -> "ObjectRegistry";
    "ABC" -> "ObjectRegistry";
    "Generic" -> "ObjectRegistry";
}
\enddot
"""

import weakref
from abc import ABC
from typing_compat import Any, Dict, Generic, Hashable, Type, TypeVar

from registry.mixin.accessor import RegistryError

from ..mixin import MutableRegistryValidatorMixin
from ._dev_utils import (
    # _dev_utils
    get_protocol,
)
from ._validator import (
    # _validator
    ConformanceError,
    InheritanceError,
    ValidationError,
    validate_instance_hierarchy,
    validate_instance_structure,
)

# Type variable representing the object type to be registered.
ObjT = TypeVar("ObjT")


class ObjectRegistry(MutableRegistryValidatorMixin[Hashable, ObjT], ABC, Generic[ObjT]):
    """
    Base class for registering instances.

    This registry stores instances in a standard dictionary with weak references
    to prevent memory leaks. It supports runtime validations to ensure that
    registered instances conform to the expected inheritance and structural requirements.

    Attributes:
        _repository (Dict[Hashable, Any]): Dictionary for storing registered instances with weak references.
        _strict (bool): Whether to enforce instance structure validation.
        _abstract (bool): Whether to enforce instance hierarchy validation.
    """

    _repository: Dict[Hashable, Any]
    _strict: bool = False
    _abstract: bool = False
    _strict_weakref: bool = False
    __slots__ = ()

    @classmethod
    def _get_mapping(cls):
        """Return the repository dictionary."""
        return cls._repository

    @classmethod
    def __init_subclass__(
        cls, strict: bool = False, abstract: bool = False, strict_weakref: bool = False
    ) -> None:
        """
        Initialize a subclass of ObjectRegistry.

        This method initializes the registry repository and sets the validation
        flags based on the provided parameters.

        Parameters:
            strict (bool): If True, enforce structure validation.
            abstract (bool): If True, enforce inheritance validation.
            strict_weakref (bool): If True, enforce weak reference validation.
        """
        super().__init_subclass__()
        # Set up the internal repository as a standard dictionary.
        cls._repository = {}
        # Store validation flags
        cls._strict = strict
        cls._abstract = abstract
        cls._strict_weakref = strict_weakref

    @classmethod
    def _probe_artifact(cls, value: Any) -> ObjT:
        """
        Validate an instance before registration.

        This method applies both inheritance and structure (conformance) checks to
        the given instance based on the class's validation flags. It then wraps
        the instance in a weak reference for storage.

        Parameters:
            value (Any): The instance to validate.

        Returns:
            ObjT: The validated instance.

        Raises:
            InheritanceError: If the instance does not meet the inheritance requirements.
            ConformanceError: If the instance does not conform to the expected structure.
        """
        # Apply inheritance checking if abstract mode is enabled
        if cls._abstract:
            value = validate_instance_hierarchy(value, expected_type=cls)

        # Apply conformance checking if strict mode is enabled
        if cls._strict:
            value = validate_instance_structure(value, expected_type=get_protocol(cls))

        # Return the actual instance (not wrapped in weakref)
        # The wrapping happens explicitly when storing in the repository
        try:
            value = weakref.ref(value)
        except TypeError:
            if cls._strict_weakref:
                raise RegistryError("ObjectRegistry only supports weak references")
        return value

    @classmethod
    def _seal_artifact(cls, value: Any) -> ObjT:
        """
        Validate an instance when retrieving from the registry.

        This method resolves the weak reference to get the actual object.
        If the reference is dead (object has been garbage collected),
        it raises a KeyError.

        Parameters:
            value (Any): The weak reference to resolve.

        Returns:
            ObjT: The actual object.

        Raises:
            KeyError: If the weak reference is dead (object has been collected).
        """
        # Check if the value is a weak reference
        if isinstance(value, weakref.ref):
            # Dereference to get the actual object
            actual_value = value()
            if actual_value is None:
                # The referenced object has been garbage collected
                raise RegistryError(
                    "Weak reference is dead (object has been collected)"
                )
            return actual_value
        return value

    @classmethod
    def _probe_identifier(cls, value: Any) -> Hashable:
        """
        Validate an identifier before storing in the registry.

        This default implementation simply ensures the value is hashable.
        Override this method to add custom validation if needed.

        Parameters:
            value (Any): The value to validate.

        Returns:
            Hashable: The validated value.
        """
        return super()._probe_identifier(value)

    @classmethod
    def _seal_identifier(cls, value: Any) -> Hashable:
        """
        Validate an identifier when retrieving from the registry.

        This default implementation ensures the value is hashable.
        Override this method to add custom validation if needed.

        Parameters:
            value (Any): The value to validate.

        Returns:
            Hashable: The validated value.
        """
        return super()._seal_identifier(value)

    @classmethod
    def register_instance(cls, key: Hashable, item: ObjT) -> ObjT:
        """
        Register an instance in the registry.

        The instance is stored in the registry using the provided key.
        It is stored as a weak reference to avoid memory leaks.

        Parameters:
            key (Hashable): The key to use for the instance.
            item (ObjT): The instance to register.

        Returns:
            ObjT: The registered instance.
        """
        # Use the existing register_artifact but wrap the item in a weak reference
        # after it's been validated by _probe_artifact
        cls.register_artifact(key, item)
        return item

    @classmethod
    def unregister_instance(cls, key: Hashable) -> None:
        """
        Unregister an instance from the registry.

        The instance is removed from the registry using the provided key.

        Parameters:
            key (Hashable): The key of the instance to unregister.
        """
        cls.unregister_artifact(key)

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
                    # Register the new instance with itself as the key
                    obj = cls.register_instance(obj, obj)
                except Exception as e:
                    print(f"Registration error: {e}")
                return obj

        # Copy meta attributes from the original metaclass.
        newmcs.__name__ = meta.__name__
        newmcs.__qualname__ = meta.__qualname__
        newmcs.__module__ = meta.__module__
        newmcs.__doc__ = meta.__doc__

        # Create and return the new tracked class.
        return newmcs(supercls.__name__, (supercls,), {})

    @classmethod
    def cleanup(cls) -> int:
        """
        Clean up dead references in the repository.

        This method removes all entries that contain dead weak references
        (references to objects that have been garbage collected).

        Returns:
            int: The number of dead references removed.
        """
        dead_refs = []
        for key, value in cls._repository.items():
            if isinstance(value, weakref.ref) and value() is None:
                dead_refs.append(key)

        for key in dead_refs:
            del cls._repository[key]

        return len(dead_refs)
