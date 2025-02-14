r"""Baseclass for registering classes.

This module defines the TypeRegistry class, which extends a mutable registry to
manage the registration of classes. It provides runtime validations to ensure that
registered classes conform to expected inheritance and protocol requirements.

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph TypeRegistry {
    "MutableRegistry" -> "TypeRegistry";
    "ABC" -> "TypeRegistry";
    "Generic" -> "TypeRegistry";
}
\enddot
"""

from abc import ABC
from functools import partial
from typing_compat import Any, Callable, Generic, Hashable, Type, TypeVar, Literal

from ..utils import (
    # base
    MutableRegistry,
    # _dev_utils
    _def_checking,
    get_module_members,
    get_protocol,
    get_subclasses,
    # _validator
    ConformanceError,
    InheritanceError,
    ValidationError,
    validate_class,
    validate_class_hierarchy,
    validate_class_structure,
)

# Type variable for classes to be registered.
Cls = TypeVar("Cls")


class TypeRegistry(MutableRegistry[Hashable, Type[Cls]], ABC, Generic[Cls]):
    """
    Base class for registering classes.

    This metaclass extends MutableRegistry to enable runtime registration and validation
    of classes. It performs validations for:
      - Ensuring the provided value is a class.
      - Verifying that the class inherits from the expected base (if marked as abstract).
      - Enforcing protocol conformance (if strict mode is enabled).

    Attributes:
        runtime_conformance_checking (Callable[[Type], Type]):
            A function used to check that a class conforms to a specific protocol.
        runtime_inheritance_checking (Callable[[Type], Type]):
            A function used to verify that a class inherits from an expected abstract base.
    """

    # Default functions for runtime validations; these can be replaced in subclasses.
    runtime_conformance_checking: Callable[[Type[Any]], Type[Any]] = _def_checking
    runtime_inheritance_checking: Callable[[Type[Any]], Type[Any]] = _def_checking

    __slots__ = ()

    @classmethod
    def __init_subclass__(
        cls, strict: bool = False, abstract: bool = False, **kwargs
    ) -> None:
        """
        Initialize a subclass of TypeRegistry.

        This method initializes the registry repository and configures runtime validation
        functions based on the provided flags.

        Parameters:
            strict (bool): If True, enforce protocol conformance checking.
            abstract (bool): If True, enforce inheritance checking.
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init_subclass__(**kwargs)
        # Initialize the repository for registered classes.
        cls._repository = dict()

        # Configure runtime inheritance checking if the class is marked as abstract.
        if abstract:
            cls.runtime_inheritance_checking = partial(
                validate_class_hierarchy, abc_class=cls
            )

        # Configure runtime conformance checking if strict mode is enabled.
        if strict:
            cls.runtime_conformance_checking = partial(
                validate_class_structure, expected_type=get_protocol(cls)
            )

    @classmethod
    def validate_artifact(cls, value: Type[Cls]) -> Type[Cls]:
        """
        Validate a class before registration.

        The validation process includes:
          1. Verifying that the value is indeed a class.
          2. Checking that it adheres to the required inheritance structure.
          3. Ensuring that it conforms to the expected protocol.

        Parameters:
            value (Type[Cls]): The class to validate.

        Returns:
            Type[Cls]: The validated class.

        Raises:
            ValidationError: If any validation step fails.
        """
        value = validate_class(value)
        value = cls.runtime_inheritance_checking(value)
        value = cls.runtime_conformance_checking(value)
        return value

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        """
        Custom subclass check to perform runtime validation.

        This method overrides the standard subclass check to ensure that the value
        passes class, inheritance, and protocol validations.

        Parameters:
            value (Any): The class to check.

        Returns:
            bool: True if the class passes all validations, False otherwise.
        """
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
        """
        Register a subclass in the registry.

        The class is registered using its __name__ attribute as the key.

        Parameters:
            subcls (Type[Cls]): The subclass to register.

        Returns:
            Type[Cls]: The registered subclass.

        Raises:
            ValidationError: If the provided value is not a class (i.e., lacks a __name__).
        """
        if hasattr(subcls, "__name__"):
            cls.register_artifact(subcls.__name__, subcls)
        else:
            # Raising error to indicate that the value cannot be registered.
            raise ValidationError(f"{subcls} is not a class or type")
        return subcls

    @classmethod
    def unregister_class(cls, subcls: Type[Cls]) -> Type[Cls]:
        """
        Unregister a subclass from the registry.

        The class is removed from the registry using its __name__ attribute as the key.

        Parameters:
            subcls (Type[Cls]): The subclass to unregister.

        Returns:
            Type[Cls]: The unregistered subclass.

        Raises:
            ValidationError: If the provided value is not a class (i.e., lacks a __name__).
        """
        if hasattr(subcls, "__name__"):
            cls.unregister_artifacts(subcls.__name__)
        else:
            raise ValidationError(f"{subcls} is not a class or type")
        return subcls

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
                    cls.register_subclasses(subcls, recursive)
                try:
                    cls.register_class(subcls)
                except ValidationError as e:
                    if raise_error:
                        raise ValidationError(e)
                    else:
                        print(e)

        if mode in {"deferred", "both"}:
            # Define a dynamic metaclass for deferred registration.
            meta: Type[Any] = type(supercls)
            register_class_func = cls.register_class

            class newmcs(meta):
                def __new__(cls, name, bases, attrs) -> Type[Cls]:
                    new_class = super().__new__(cls, name, bases, attrs)
                    try:
                        new_class = register_class_func(new_class)
                    except Exception as e:
                        print(e)
                    return new_class

            # Copy meta attributes from the original metaclass.
            newmcs.__name__ = meta.__name__
            newmcs.__qualname__ = meta.__qualname__
            newmcs.__module__ = meta.__module__
            newmcs.__doc__ = meta.__doc__

            # Re-create the superclass using the new metaclass for deferred registration.
            supercls = newmcs(supercls.__name__, (supercls,), {})
        return supercls
