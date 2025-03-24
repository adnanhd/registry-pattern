r"""Baseclass for registering objects with configuration.

This module defines the ConfigRegistry class, which provides a registry
for objects along with their configuration dictionaries. It uses weak references
to keys to avoid memory leaks, but stores them in a standard dictionary for
more flexible access patterns.

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph ConfigRegistry {
    "MutableRegistryValidatorMixin" -> "ConfigRegistry";
    "ABC" -> "ConfigRegistry";
    "Generic" -> "ConfigRegistry";
}
\enddot
"""

import weakref
from abc import ABC
from typing_compat import Any, Dict, Generic, Hashable, Type, TypeVar, Optional

from ..mixin import MutableRegistryValidatorMixin, RegistryError
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

# Type variables for object and configuration
ObjT = TypeVar("ObjT", bound=Hashable)
CfgT = TypeVar("CfgT", bound=Dict[str, Any])


class ConfigRegistry(
    MutableRegistryValidatorMixin[Any, CfgT], ABC, Generic[ObjT, CfgT]
):
    """
    Base class for registering objects with configuration.

    This registry uses weak references for keys to prevent memory leaks,
    but stores them in a standard dictionary for more flexible access.
    It supports runtime validations to ensure registered objects conform
    to expected inheritance and structural requirements.

    Attributes:
        _repository (dict): Dictionary mapping weak references to configurations.
        _strict (bool): Whether to enforce structure validation.
        _abstract (bool): Whether to enforce inheritance validation.
    """

    _repository: Dict[Any, CfgT]
    _strict: bool = False
    _abstract: bool = False
    __slots__ = ()

    @classmethod
    def _get_mapping(cls):
        """Return the repository dictionary."""
        return cls._repository

    @classmethod
    def __init_subclass__(cls, strict: bool = False, abstract: bool = False) -> None:
        """
        Initialize a subclass of ConfigRegistry.

        This method initializes the registry repository and sets validation flags.

        Parameters:
            strict (bool): If True, enforce structure validation.
            abstract (bool): If True, enforce inheritance validation.
        """
        super().__init_subclass__()
        # Initialize with a standard dictionary rather than a WeakKeyDictionary
        cls._repository = dict()
        # Store validation flags
        cls._strict = strict
        cls._abstract = abstract

    @classmethod
    def _probe_identifier(cls, value: ObjT) -> Any:
        """
        Validate and transform a value before storing in the registry.

        This method validates the value and wraps it in a weakref.ref
        to prevent memory leaks.

        Parameters:
            value (ObjT): The value to validate and transform.

        Returns:
            Any: The validated and transformed value (a weakref).

        Raises:
            TypeError: If the value is not hashable.
        """
        # Basic type validation for the value
        if not isinstance(value, Hashable):
            raise TypeError(f"Key must be hashable, got {type(value)}")

        # Apply inheritance checking if abstract mode is enabled
        if cls._abstract:
            value = validate_instance_hierarchy(value, expected_type=cls)

        # Apply conformance checking if strict mode is enabled
        if cls._strict:
            value = validate_instance_structure(value, expected_type=get_protocol(cls))

        # Wrap the value in a weakref.ref
        return weakref.ref(value)

    @classmethod
    def _seal_identifier(cls, value: Any) -> ObjT:
        """
        Validate and transform a value when retrieving from the registry.

        If the value is a weakref.ref, call it to get the actual object.
        If the object has been garbage collected, raise a KeyError.

        Parameters:
            value (Any): The value to validate and transform.

        Returns:
            ObjT: The validated and transformed value.

        Raises:
            KeyError: If the value is a dead weakref.
            TypeError: If the value is not hashable.
        """
        # If the value is already a weakref, get the referenced object
        if isinstance(value, weakref.ref):
            actual_value = value()
            if actual_value is None:
                # The referenced object has been garbage collected
                raise KeyError(f"Weakref {value} is dead (object has been collected)")
            return actual_value

        # If it's not a weakref, make sure it's hashable
        if not isinstance(value, Hashable):
            raise TypeError(f"Key must be hashable, got {type(value)}")

        return value

    @classmethod
    def _probe_artifact(cls, value: CfgT) -> CfgT:
        """
        Validate a configuration before storing in the registry.

        This method ensures the configuration is a dictionary.

        Parameters:
            value (CfgT): The configuration to validate.

        Returns:
            CfgT: The validated configuration.

        Raises:
            TypeError: If the value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError(f"Configuration must be a dictionary, got {type(value)}")
        return value

    @classmethod
    def _seal_artifact(cls, value: CfgT) -> CfgT:
        """
        Validate a configuration when retrieving from the registry.

        This method is a no-op by default, as validation is primarily
        done during storage.

        Parameters:
            value (CfgT): The configuration to validate.

        Returns:
            CfgT: The validated configuration.
        """
        return value

    @classmethod
    def _find_weakref_key(cls, key: Any) -> Optional[weakref.ref]:
        """
        Find the weakref key that points to the given object.

        Parameters:
            key: The object to search for.

        Returns:
            Optional[weakref.ref]: The weakref key if found, None otherwise.
        """
        # If key is already a weakref.ref, check if it's in the repository
        if isinstance(key, weakref.ref):
            if key in cls._repository:
                return key
            return None

        # Otherwise, search for a weakref that points to our object
        for weakref_key in cls._repository:
            if isinstance(weakref_key, weakref.ref):
                obj = weakref_key()
                if obj is not None and obj is key:
                    return weakref_key

        return None

    @classmethod
    def has_identifier(cls, key: Any) -> bool:
        """
        Check if a key exists in the registry.

        Parameters:
            key: The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        return cls._find_weakref_key(key) is not None

    @classmethod
    def get_artifact(cls, key: Any) -> CfgT:
        """
        Get an artifact from the registry.

        Parameters:
            key: The key to look up.

        Returns:
            CfgT: The artifact associated with the key.

        Raises:
            RegistryError: If the key is not found.
        """
        weakref_key = cls._find_weakref_key(key)
        if weakref_key is None:
            raise RegistryError(f"Key '{key}' is not found in the repository")
        return cls._repository[weakref_key]

    @classmethod
    def validate_artifact(cls, value: Any) -> CfgT:
        """
        Validate a configuration dictionary.

        Parameters:
            value: The configuration to validate.

        Returns:
            CfgT: The validated configuration.

        Raises:
            TypeError: If the value is not a dictionary.
        """
        return cls._probe_artifact(value)

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        """
        Custom subclass check for runtime validation.

        Parameters:
            value (Any): The value to check.

        Returns:
            bool: True if the value passes all validations; False otherwise.
        """
        try:
            # Check if the value can be used as a key
            cls._probe_identifier(value)
            return True
        except (ValidationError, InheritanceError, ConformanceError, TypeError):
            return False

    @classmethod
    def register_config(cls, obj: ObjT, config: CfgT) -> CfgT:
        """
        Register a configuration for an object.

        Parameters:
            obj (ObjT): The object to associate with the configuration.
            config (CfgT): The configuration to register.

        Returns:
            CfgT: The registered configuration.
        """
        cls.register_artifact(obj, config)
        return config

    @classmethod
    def unregister_config(cls, obj: Any) -> None:
        """
        Unregister a configuration for an object.

        Parameters:
            obj: The object whose configuration should be unregistered.
        """
        weakref_key = cls._find_weakref_key(obj)
        if weakref_key is None:
            raise RegistryError(f"Key '{obj}' is not found in the repository")
        del cls._repository[weakref_key]

    @classmethod
    def register_class_configs(cls, supercls: Type[ObjT]) -> Type[ObjT]:
        """
        Create a tracked version of a class that registers configurations for instances.

        Parameters:
            supercls (Type[ObjT]): The class to track.

        Returns:
            Type[ObjT]: A wrapped version of the class that auto-registers configurations.
        """
        meta: Type[Any] = type(supercls)

        class newmcs(meta):
            def __call__(self, *args: Any, **kwds: Any) -> ObjT:
                # Create the instance
                obj = super().__call__(*args, **kwds)
                try:
                    # Register the instance with its initialization kwargs
                    cls.register_config(obj, kwds)  # type: ignore
                except Exception as e:
                    print(f"Registration error: {e}")
                return obj

        # Copy meta attributes
        newmcs.__name__ = meta.__name__
        newmcs.__qualname__ = meta.__qualname__
        newmcs.__module__ = meta.__module__
        newmcs.__doc__ = meta.__doc__

        # Return the new tracked class
        return newmcs(supercls.__name__, (supercls,), {})

    @classmethod
    def cleanup(cls) -> int:
        """
        Clean up dead references in the repository.

        This method removes all keys that are dead weakrefs
        (references to objects that have been garbage collected).

        Returns:
            int: The number of dead references removed.
        """
        dead_refs = []
        for key in cls._repository:
            if isinstance(key, weakref.ref) and key() is None:
                dead_refs.append(key)

        for key in dead_refs:
            del cls._repository[key]

        return len(dead_refs)
