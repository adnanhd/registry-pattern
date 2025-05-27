r"""Baseclass for registering classes.

This module defines the TypeRegistry class, which extends a mutable registry to
manage the registration of classes. It provides runtime validations to ensure that
registered classes conform to expected inheritance and protocol requirements.

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph TypeRegistry {
    "MutableRegistryValidatorMixin" -> "TypeRegistry";
    "ABC" -> "TypeRegistry";
    "Generic" -> "TypeRegistry";
}
\enddot
"""

import logging
from abc import ABC

from typing_compat import Any, Generic, Hashable, Literal, Type, TypeVar

from ..mixin import MutableRegistryValidatorMixin
from ._dev_utils import get_module_members, get_protocol, get_subclasses  # _dev_utils
from ._validator import InheritanceError  # _validator
from ._validator import (
    get_type_name,
    ConformanceError,
    ValidationError,
    validate_class,
    validate_class_hierarchy,
    validate_class_structure,
)

# Type variable for classes to be registered.
Cls = TypeVar("Cls")
logger = logging.getLogger(__name__)


class TypeRegistry(
    MutableRegistryValidatorMixin[Hashable, Type[Cls]], ABC, Generic[Cls]
):
    """
    Base class for registering classes.

    This class extends MutableRegistryValidatorMixin to enable runtime registration and validation
    of classes. It performs validations for:
      - Ensuring the provided value is a class.
      - Verifying that the class inherits from the expected base (if marked as abstract).
      - Enforcing protocol conformance (if strict mode is enabled).

    Attributes:
        _repository (dict): Dictionary for storing registered classes.
        _strict (bool): Whether to enforce protocol conformance.
        _abstract (bool): Whether to enforce inheritance validation.
    """

    _repository: dict
    _strict: bool = False
    _abstract: bool = False
    __slots__ = ()

    @classmethod
    def _get_mapping(cls):
        """Return the repository dictionary."""
        return cls._repository

    @classmethod
    def __init_subclass__(
        cls, strict: bool = False, abstract: bool = False, **kwargs
    ) -> None:
        """
        Initialize a subclass of TypeRegistry.

        This method initializes the registry repository and sets the validation
        flags based on the provided parameters.

        Parameters:
            strict (bool): If True, enforce protocol conformance checking.
            abstract (bool): If True, enforce inheritance checking.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init_subclass__(**kwargs)
        # Initialize the repository for registered classes.
        cls._repository = dict()
        # Store validation flags
        cls._strict = strict
        cls._abstract = abstract

    @classmethod
    def _intern_artifact(cls, value: Any) -> Type[Cls]:
        """
        Validate a class before registration.

        The validation process includes:
          1. Verifying that the value is indeed a class.
          2. Checking that it adheres to the required inheritance structure (if _abstract is True).
          3. Ensuring that it conforms to the expected protocol (if _strict is True).

        Parameters:
            value (Any): The class to validate.

        Returns:
            Type[Cls]: The validated class.

        Raises:
            ValidationError: If value is not a class.
            InheritanceError: If value does not inherit from the required base class.
            ConformanceError: If value does not conform to the required protocol.
        """
        # Basic validation to ensure it's a class
        value = validate_class(value)

        # Apply inheritance checking if abstract mode is enabled
        if cls._abstract:
            value = validate_class_hierarchy(value, abc_class=cls)

        # Apply conformance checking if strict mode is enabled
        if cls._strict:
            value = validate_class_structure(value, expected_type=get_protocol(cls))

        return value


    @classmethod
    def _identifier_of(cls, item: Type[Cls]) -> Hashable:
        """
        Generate a unique identifier for a class.

        This method is used to generate a unique identifier for a class
        when registering it in the registry. The identifier is typically
        used as a key in the registry's internal data structure.

        Parameters:
            item (Type[Cls]): The class to generate an identifier for.

        Returns:
            Hashable: The generated identifier.
        """
        return get_type_name(validate_class(item))

    @classmethod
    def _extern_artifact(cls, value: Type[Cls]) -> Type[Cls]:
        """
        Validate a class when retrieving from the registry.

        This method is a no-op by default, as validation is primarily
        done during registration. Override this method to add validation
        during retrieval if needed.

        Parameters:
            value (Type[Cls]): The class to validate.

        Returns:
            Type[Cls]: The validated class.
        """
        return value

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        """
        Custom subclass check to perform runtime validation.

        This method uses the configured validation methods to determine
        whether the given value meets the criteria for being considered a subclass.

        Parameters:
            value (Any): The class to check.

        Returns:
            bool: True if the value passes all validations; False otherwise.
        """
        try:
            cls._intern_artifact(value)
        except (ValidationError, InheritanceError, ConformanceError):
            return False
        else:
            return True

    @classmethod
    def register_module_subclasses(cls, module: Any, raise_error: bool = True) -> Any:
        """
        Register all subclasses found within a given module.

        This function retrieves all members from the module and attempts to register
        each one as a subclass. Validation errors are either raised or printed based
        on the 'raise_error' flag.

        Parameters:
            module (Any): The module from which to retrieve and register subclasses.
            raise_error (bool): If True, re-raise validation errors; otherwise, print them.

        Returns:
            Any: The module after processing its members.
        """
        module_members = get_module_members(module)
        for obj in module_members:
            try:
                cls.register_artifact(obj)
            except AssertionError:
                logger.debug(f"Could not register {obj}")
            except ValidationError as e:
                if raise_error:
                    raise ValidationError(e)
                else:
                    logger.debug(e)
        return module

    @classmethod
    def register_subclasses(
        cls,
        supercls: Type[Cls],
        recursive: bool = False,
        raise_error: bool = True,
        mode: Literal["immediate", "deferred", "both"] = "both",
    ) -> Type[Cls]:
        """
        Register all subclasses of a given superclass.

        Registration can occur in one of three modes:
          - "immediate": Register all current subclasses immediately.
          - "deferred": Set up a metaclass to register future subclasses.
          - "both": Apply both immediate and deferred registration.

        Parameters:
            supercls (Type[Cls]): The superclass whose subclasses are to be registered.
            recursive (bool): If True, recursively register subclasses of subclasses.
            raise_error (bool): If True, re-raise validation errors; otherwise, print them.
            mode (Literal["immediate", "deferred", "both"]): The mode of registration.

        Returns:
            Type[Cls]: The superclass, potentially re-created with a new metaclass for deferred registration.
        """
        if mode in {"immediate", "both"}:
            # Perform immediate registration of current subclasses.
            for subcls in get_subclasses(supercls):
                if recursive and get_subclasses(subcls):
                    cls.register_subclasses(subcls, recursive, mode="immediate")
                try:
                    cls.register_artifact(subcls)
                except ValidationError as e:
                    if raise_error:
                        raise ValidationError(e)
                    else:
                        logger.debug(e)

        if mode in {"deferred", "both"}:
            # Define a dynamic metaclass for deferred registration.
            meta: Type[Any] = type(supercls)
            register_artifact_func = cls.register_artifact

            class newmcs(meta):
                def __new__(cls, name, bases, attrs) -> Type[Cls]:
                    new_artifact = super().__new__(cls, name, bases, attrs)
                    try:
                        new_artifact = register_artifact_func(new_artifact)
                    except Exception as e:
                        logger.debug(e)
                    return new_artifact

            # Copy meta attributes from the original metaclass.
            newmcs.__name__ = meta.__name__
            newmcs.__qualname__ = meta.__qualname__
            newmcs.__module__ = meta.__module__
            newmcs.__doc__ = meta.__doc__

            # Re-create the superclass using the new metaclass for deferred registration.
            supercls = newmcs(supercls.__name__, (supercls,), {})
        return supercls
