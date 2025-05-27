r"""Enhanced TypeRegistry with improved validation system.

This module defines the TypeRegistry class, which extends a mutable registry to
manage the registration of classes. It provides runtime validations with rich
error context and helpful suggestions to ensure that registered classes conform
to expected inheritance and protocol requirements.

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

import inspect
import logging
from abc import ABC

from typing_compat import Any, Generic, Hashable, Literal, Type, TypeVar, Dict

from ..mixin import MutableRegistryValidatorMixin
from ._dev_utils import get_module_members, get_protocol, get_subclasses
from ._validator import (
    InheritanceError,
    ConformanceError,
    ValidationError,
    get_type_name,
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
    Enhanced registry for registering classes with comprehensive validation.

    This class extends MutableRegistryValidatorMixin to enable runtime registration
    and validation of classes with rich error messages and helpful suggestions.
    It performs validations for:
      - Ensuring the provided value is a class.
      - Verifying that the class inherits from the expected base (if abstract mode enabled).
      - Enforcing protocol conformance (if strict mode enabled).

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
        Initialize a subclass of TypeRegistry with enhanced validation configuration.

        This method initializes the registry repository and sets the validation
        flags based on the provided parameters.

        Parameters:
            strict (bool): If True, enforce protocol conformance checking with detailed feedback.
            abstract (bool): If True, enforce inheritance checking with helpful suggestions.
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
        Validate a class before registration with enhanced error reporting.

        The validation process includes:
          1. Verifying that the value is indeed a class.
          2. Checking that it adheres to the required inheritance structure (if _abstract is True).
          3. Ensuring that it conforms to the expected protocol (if _strict is True).

        Parameters:
            value (Any): The class to validate.

        Returns:
            Type[Cls]: The validated class.

        Raises:
            ValidationError: If value is not a class, with suggestions for fixing.
            InheritanceError: If value does not inherit from the required base class, with inheritance suggestions.
            ConformanceError: If value does not conform to the required protocol, with protocol implementation suggestions.
        """
        # Basic validation to ensure it's a class
        try:
            value = validate_class(value)
        except ValidationError as e:
            # Add registry-specific suggestions
            if hasattr(e, "suggestions"):
                e.suggestions.extend(
                    [
                        f"Registry {cls.__name__} only accepts class definitions",
                        "Make sure you're registering the class itself, not an instance",
                    ]
                )
            raise

        # Apply inheritance checking if abstract mode is enabled
        if cls._abstract:
            try:
                value = validate_class_hierarchy(value, abc_class=cls)
            except InheritanceError as e:
                # Add registry-specific context
                if hasattr(e, "context"):
                    e.context.update(
                        {
                            "registry_name": cls.__name__,
                            "registry_mode": "abstract",
                            "base_class": get_type_name(cls),
                        }
                    )
                if hasattr(e, "suggestions"):
                    e.suggestions.extend(
                        [
                            f"Register classes that inherit from {cls.__name__}",
                            f"Example: class YourClass({cls.__name__}): ...",
                        ]
                    )
                raise

        # Apply conformance checking if strict mode is enabled
        if cls._strict:
            protocol = get_protocol(cls)
            try:
                value = validate_class_structure(value, expected_type=protocol)
            except ConformanceError as e:
                # Add registry-specific context
                if hasattr(e, "context"):
                    e.context.update(
                        {
                            "registry_name": cls.__name__,
                            "registry_mode": "strict",
                            "required_protocol": get_type_name(protocol),
                        }
                    )
                if hasattr(e, "suggestions"):
                    e.suggestions.extend(
                        [
                            f"Use cls.validate_artifact(YourClass) to check conformance before registration",
                            f"Registry {cls.__name__} requires strict protocol conformance",
                        ]
                    )
                raise

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
        Custom subclass check to perform runtime validation with caching.

        This method uses the configured validation methods to determine
        whether the given value meets the criteria for being considered a subclass.
        Results are cached for performance.

        Parameters:
            value (Any): The class to check.

        Returns:
            bool: True if the value passes all validations; False otherwise.
        """
        try:
            cls._intern_artifact(value)
            return True
        except (ValidationError, InheritanceError, ConformanceError):
            return False

    @classmethod
    def register_module_subclasses(cls, module: Any, raise_error: bool = True) -> Any:
        """
        Register all subclasses found within a given module with enhanced error reporting.

        This function retrieves all members from the module and attempts to register
        each one as a subclass. Validation errors include detailed context and suggestions.

        Parameters:
            module (Any): The module from which to retrieve and register subclasses.
            raise_error (bool): If True, re-raise validation errors; otherwise, log them with context.

        Returns:
            Any: The module after processing its members.
        """
        module_members = get_module_members(module)
        registration_results = {"successful": [], "failed": [], "errors": []}

        for obj in module_members:
            try:
                cls.register_artifact(obj)
                registration_results["successful"].append(get_type_name(obj))
            except (ValidationError, ConformanceError, InheritanceError) as e:
                obj_name = get_type_name(obj)
                registration_results["failed"].append(obj_name)
                registration_results["errors"].append(
                    {
                        "name": obj_name,
                        "error": str(e),
                        "suggestions": getattr(e, "suggestions", []),
                    }
                )

                if raise_error:
                    # Add module context to the error
                    if hasattr(e, "context"):
                        e.context.update(
                            {
                                "module_name": getattr(module, "__name__", str(module)),
                                "operation": "register_module_subclasses",
                            }
                        )
                    raise
                else:
                    logger.debug(f"Could not register {obj_name} from module: {e}")
            except Exception as e:
                obj_name = getattr(obj, "__name__", str(obj))
                enhanced_error = ValidationError(
                    f"Unexpected error registering {obj_name}: {e}",
                    suggestions=[
                        "Check that the object is a valid class",
                        "Ensure the module is properly imported",
                    ],
                    context={
                        "module_name": getattr(module, "__name__", str(module)),
                        "operation": "register_module_subclasses",
                        "artifact_name": obj_name,
                    },
                )

                registration_results["failed"].append(obj_name)
                registration_results["errors"].append(
                    {
                        "name": obj_name,
                        "error": str(enhanced_error),
                        "suggestions": enhanced_error.suggestions,
                    }
                )

                if raise_error:
                    raise enhanced_error from e
                else:
                    logger.debug(f"Unexpected error with {obj_name}: {enhanced_error}")

        # Log summary
        logger.info(
            f"Module registration complete: {len(registration_results['successful'])} successful, {len(registration_results['failed'])} failed"
        )

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
        Register all subclasses of a given superclass with enhanced error handling.

        Registration can occur in one of three modes:
          - "immediate": Register all current subclasses immediately.
          - "deferred": Set up a metaclass to register future subclasses.
          - "both": Apply both immediate and deferred registration.

        Parameters:
            supercls (Type[Cls]): The superclass whose subclasses are to be registered.
            recursive (bool): If True, recursively register subclasses of subclasses.
            raise_error (bool): If True, re-raise validation errors; otherwise, log them.
            mode (Literal["immediate", "deferred", "both"]): The mode of registration.

        Returns:
            Type[Cls]: The superclass, potentially re-created with a new metaclass for deferred registration.
        """
        registration_results = {"successful": [], "failed": [], "errors": []}

        if mode in {"immediate", "both"}:
            # Perform immediate registration of current subclasses
            for subcls in get_subclasses(supercls):
                if recursive and get_subclasses(subcls):
                    try:
                        cls.register_subclasses(subcls, recursive, mode="immediate")
                    except Exception as e:
                        logger.debug(
                            f"Recursive registration failed for {get_type_name(subcls)}: {e}"
                        )

                try:
                    cls.register_artifact(subcls)
                    registration_results["successful"].append(get_type_name(subcls))
                except (ValidationError, ConformanceError, InheritanceError) as e:
                    subcls_name = get_type_name(subcls)
                    registration_results["failed"].append(subcls_name)
                    registration_results["errors"].append(
                        {
                            "name": subcls_name,
                            "error": str(e),
                            "suggestions": getattr(e, "suggestions", []),
                        }
                    )

                    if raise_error:
                        # Add subclass registration context
                        if hasattr(e, "context"):
                            e.context.update(
                                {
                                    "operation": "register_subclasses",
                                    "parent_class": get_type_name(supercls),
                                    "registration_mode": mode,
                                }
                            )
                        raise
                    else:
                        logger.debug(f"Could not register subclass {subcls_name}: {e}")

        if mode in {"deferred", "both"}:
            # Define a dynamic metaclass for deferred registration with enhanced error handling
            meta: Type[Any] = type(supercls)
            register_artifact_func = cls.register_artifact

            class EnhancedRegistrationMeta(meta):
                def __new__(cls, name, bases, attrs) -> Type[Cls]:
                    new_artifact = super().__new__(cls, name, bases, attrs)
                    try:
                        new_artifact = register_artifact_func(new_artifact)
                    except (ValidationError, ConformanceError, InheritanceError) as e:
                        # Add deferred registration context
                        if hasattr(e, "context"):
                            e.context.update(
                                {
                                    "operation": "deferred_registration",
                                    "parent_class": get_type_name(supercls),
                                    "new_class": name,
                                }
                            )
                        logger.debug(f"Deferred registration failed for {name}: {e}")
                    except Exception as e:
                        enhanced_error = ValidationError(
                            f"Unexpected error in deferred registration of {name}: {e}",
                            suggestions=[
                                "Check that the new class meets registry requirements",
                                "Verify parent class registration was successful",
                            ],
                            context={
                                "operation": "deferred_registration",
                                "parent_class": get_type_name(supercls),
                                "new_class": name,
                            },
                        )
                        logger.debug(str(enhanced_error))
                    return new_artifact

            # Copy meta attributes from the original metaclass
            EnhancedRegistrationMeta.__name__ = meta.__name__
            EnhancedRegistrationMeta.__qualname__ = meta.__qualname__
            EnhancedRegistrationMeta.__module__ = meta.__module__
            EnhancedRegistrationMeta.__doc__ = meta.__doc__

            # Re-create the superclass using the new metaclass for deferred registration
            supercls = EnhancedRegistrationMeta(supercls.__name__, (supercls,), {})

        # Log registration summary
        if registration_results["successful"] or registration_results["failed"]:
            logger.info(
                f"Subclass registration complete: {len(registration_results['successful'])} successful, {len(registration_results['failed'])} failed"
            )

        return supercls

    @classmethod
    def get_registration_report(cls) -> Dict[str, Any]:
        """
        Get a comprehensive report on the current registry state.

        Returns:
            Dictionary containing registry statistics and validation status.
        """
        return cls.validate_registry_state()

    @classmethod
    def diagnose_registration_failure(cls, failed_class: Type) -> Dict[str, Any]:
        """
        Diagnose why a class failed to register and provide detailed suggestions.

        Parameters:
            failed_class: The class that failed to register.

        Returns:
            Dictionary containing diagnostic information and suggestions.
        """
        diagnosis = {
            "class_name": get_type_name(failed_class),
            "is_class": inspect.isclass(failed_class),
            "validation_errors": [],
            "suggestions": [],
            "registry_config": {
                "strict_mode": cls._strict,
                "abstract_mode": cls._abstract,
                "registry_name": cls.__name__,
            },
        }

        # Test basic class validation
        try:
            validate_class(failed_class)
        except ValidationError as e:
            diagnosis["validation_errors"].append(
                {
                    "type": "class_validation",
                    "error": str(e),
                    "suggestions": getattr(e, "suggestions", []),
                }
            )
            diagnosis["suggestions"].extend(getattr(e, "suggestions", []))

        # Test inheritance if in abstract mode
        if cls._abstract:
            try:
                validate_class_hierarchy(failed_class, abc_class=cls)
            except InheritanceError as e:
                diagnosis["validation_errors"].append(
                    {
                        "type": "inheritance_validation",
                        "error": str(e),
                        "suggestions": getattr(e, "suggestions", []),
                    }
                )
                diagnosis["suggestions"].extend(getattr(e, "suggestions", []))

        # Test protocol conformance if in strict mode
        if cls._strict:
            protocol = get_protocol(cls)
            try:
                validate_class_structure(failed_class, expected_type=protocol)
            except ConformanceError as e:
                diagnosis["validation_errors"].append(
                    {
                        "type": "protocol_validation",
                        "error": str(e),
                        "suggestions": getattr(e, "suggestions", []),
                    }
                )
                diagnosis["suggestions"].extend(getattr(e, "suggestions", []))

        # Add general suggestions based on registry configuration
        if cls._strict and cls._abstract:
            diagnosis["suggestions"].append(
                "This registry requires both inheritance and protocol conformance"
            )
        elif cls._strict:
            diagnosis["suggestions"].append(
                "This registry requires protocol conformance - check method implementations"
            )
        elif cls._abstract:
            diagnosis["suggestions"].append(
                "This registry requires inheritance - ensure your class inherits from the base"
            )

        diagnosis["suggestions"].append(
            f"Use {cls.__name__}.validate_artifact(YourClass) to test before registration"
        )

        return diagnosis
