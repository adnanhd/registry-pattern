r"""Enhanced ObjectRegistry with improved validation system.

This module defines the ObjectRegistry class, a base class for registering
object instances with comprehensive validation, rich error context, and
performance optimizations. It uses weak references and handles validation
failures with detailed suggestions.

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

import logging
import weakref
from abc import ABC

from typing_compat import Any, Dict, Generic, Hashable, Type
from typing import TypeVar

from registry.mixin.accessor import RegistryError

from ..mixin import MutableRegistryValidatorMixin
from ._dev_utils import get_protocol
from ._validator import (
    InheritanceError,
    ConformanceError,
    ValidationError,
    get_mem_addr,
    validate_instance_hierarchy,
    validate_instance_structure,
)

# Type variable representing the object type to be registered.
ObjT = TypeVar("ObjT")
logger = logging.getLogger(__name__)


class ObjectRegistry(MutableRegistryValidatorMixin[Hashable, ObjT], ABC, Generic[ObjT]):
    """
    Enhanced registry for registering instances with comprehensive validation.

    This registry stores instances with weak references to prevent memory leaks
    and provides rich validation feedback. It supports runtime validations to
    ensure that registered instances conform to the expected inheritance and
    structural requirements.

    Attributes:
        _repository (Dict[Hashable, Any]): Dictionary for storing registered instances with weak references.
        _strict (bool): Whether to enforce instance structure validation.
        _abstract (bool): Whether to enforce instance hierarchy validation.
        _strict_weakref (bool): Whether to enforce weak reference validation.
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
        Initialize a subclass of ObjectRegistry with enhanced validation configuration.

        This method initializes the registry repository and sets the validation
        flags based on the provided parameters.

        Parameters:
            strict (bool): If True, enforce structure validation with detailed feedback.
            abstract (bool): If True, enforce inheritance validation with helpful suggestions.
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
    def _intern_artifact(cls, value: Any) -> ObjT:
        """
        Validate an instance before registration with enhanced error reporting.

        This method applies both inheritance and structure (conformance) checks to
        the given instance based on the class's validation flags. It then wraps
        the instance in a weak reference for storage.

        Parameters:
            value (Any): The instance to validate.

        Returns:
            ObjT: The validated instance (wrapped in weakref for storage).

        Raises:
            InheritanceError: If the instance does not meet the inheritance requirements.
            ConformanceError: If the instance does not conform to the expected structure.
            ValidationError: If weak reference creation fails in strict mode.
        """
        # Apply inheritance checking if abstract mode is enabled
        if cls._abstract:
            try:
                value = validate_instance_hierarchy(value, expected_type=cls)
            except InheritanceError as e:
                # Add registry-specific context
                if hasattr(e, "context"):
                    e.context.update(
                        {
                            "registry_name": cls.__name__,
                            "registry_mode": "abstract",
                            "registry_type": "ObjectRegistry",
                            "base_class": cls.__name__,
                        }
                    )
                if hasattr(e, "suggestions"):
                    e.suggestions.extend(
                        [
                            f"Register instances that inherit from or are compatible with {cls.__name__}",
                            "Check that the instance's class has proper inheritance",
                            "Verify that you're registering the correct type of object",
                        ]
                    )
                raise

        # Apply conformance checking if strict mode is enabled
        if cls._strict:
            protocol = get_protocol(cls)
            try:
                value = validate_instance_structure(value, expected_type=protocol)
            except ConformanceError as e:
                # Add registry-specific context
                if hasattr(e, "context"):
                    e.context.update(
                        {
                            "registry_name": cls.__name__,
                            "registry_mode": "strict",
                            "registry_type": "ObjectRegistry",
                            "required_protocol": (
                                str(protocol) if protocol else "unknown"
                            ),
                        }
                    )
                if hasattr(e, "suggestions"):
                    e.suggestions.extend(
                        [
                            "Use cls.validate_artifact(your_instance) to check conformance before registration",
                            f"Registry {cls.__name__} requires strict protocol conformance",
                            "Ensure the instance implements all required methods",
                        ]
                    )
                raise
            except Exception as e:
                # Convert unexpected errors to ValidationError
                suggestions = [
                    "Check that the instance implements the required protocol",
                    "Verify that all required methods are present and have correct signatures",
                    f"Registry {cls.__name__} in strict mode requires protocol conformance",
                ]
                context = {
                    "registry_name": cls.__name__,
                    "registry_mode": "strict",
                    "operation": "structure_validation",
                    "artifact_name": str(value),
                }
                enhanced_error = ValidationError(
                    f"Instance structure validation failed: {e}", suggestions, context
                )
                raise enhanced_error from e

        # Try to create weak reference
        try:
            weak_value = weakref.ref(value)
        except TypeError as e:
            if cls._strict_weakref:
                suggestions = [
                    "Registry only supports objects that can be weakly referenced",
                    "Avoid registering primitive types (int, str, etc.) in strict weak reference mode",
                    "Consider using strong reference mode by setting strict_weakref=False",
                ]
                context = {
                    "registry_name": cls.__name__,
                    "operation": "weak_reference_creation",
                    "artifact_name": str(value),
                    "artifact_type": type(value).__name__,
                }
                enhanced_error = ValidationError(
                    f"Cannot create weak reference: {e}", suggestions, context
                )
                raise enhanced_error from e
            else:
                # In non-strict mode, return the value directly
                return value

        return weak_value

    @classmethod
    def _extern_artifact(cls, value: Any) -> ObjT:
        """
        Validate an instance when retrieving from the registry with enhanced error handling.

        This method resolves the weak reference to get the actual object.
        If the reference is dead (object has been garbage collected),
        it raises a RegistryError for consistency with access operations.

        Parameters:
            value (Any): The weak reference to resolve.

        Returns:
            ObjT: The actual object.

        Raises:
            RegistryError: If the weak reference is dead or resolution fails (with rich context).
        """
        # Check if the value is a weak reference
        if isinstance(value, weakref.ref):
            # Dereference to get the actual object
            actual_value = value()
            if actual_value is None:
                # The referenced object has been garbage collected
                suggestions = [
                    "The referenced object has been garbage collected",
                    "Keep a strong reference to prevent garbage collection",
                    "Use cleanup() method to remove dead references",
                    "Consider using strong reference mode if objects are short-lived",
                ]
                context = {
                    "operation": "weak_reference_resolution",
                    "registry_name": cls.__name__,
                    "registry_type": "ObjectRegistry",
                    "weakref_address": str(value),
                }
                raise RegistryError(
                    "Weak reference is dead (object has been collected)",
                    suggestions,
                    context,
                )
            return actual_value
        return value

    @classmethod
    def _intern_identifier(cls, value: Any) -> Hashable:
        """
        Validate an identifier before storing in the registry.

        This default implementation simply ensures the value is hashable.
        Override this method to add custom validation if needed.

        Parameters:
            value (Any): The value to validate.

        Returns:
            Hashable: The validated value.
        """
        return super()._intern_identifier(value)

    @classmethod
    def _extern_identifier(cls, value: Any) -> Hashable:
        """
        Validate an identifier when retrieving from the registry.

        This default implementation ensures the value is hashable.
        Override this method to add custom validation if needed.

        Parameters:
            value (Any): The value to validate.

        Returns:
            Hashable: The validated value.
        """
        return super()._extern_identifier(value)

    @classmethod
    def _identifier_of(cls, item: Type[ObjT]) -> Hashable:
        """Generate identifier from memory address."""
        return get_mem_addr(item)

    @classmethod
    def register_class_instances(cls, supercls: Type[ObjT]) -> Type[ObjT]:
        """
        Create a tracked version of a class so that all its instances are registered.

        This function dynamically creates a new metaclass that wraps the __call__
        method of the provided class. Each time an instance is created, it is
        automatically registered in the registry with enhanced error handling.

        Parameters:
            supercls (Type[ObjT]): The class to track.

        Returns:
            Type[ObjT]: A new class that behaves like the original but automatically
                        registers its instances.
        """
        meta: Type[Any] = type(supercls)

        class EnhancedRegistrationMeta(meta):
            def __call__(self, *args: Any, **kwds: Any) -> ObjT:
                # Create a new instance using the original __call__.
                obj = super().__call__(*args, **kwds)
                try:
                    # Register the new instance with itself as the key
                    obj = cls.register_artifact(obj, obj)
                except (ValidationError, ConformanceError, InheritanceError) as e:
                    # Add automatic registration context
                    if hasattr(e, "context"):
                        e.context.update(
                            {
                                "operation": "automatic_instance_registration",
                                "tracked_class": supercls.__name__,
                                "instance_args": str(args)[:100] if args else "none",
                                "instance_kwargs": str(kwds)[:100] if kwds else "none",
                            }
                        )
                    logger.debug(
                        f"Automatic registration failed for {supercls.__name__} instance: {e}"
                    )
                except Exception as e:
                    enhanced_error = ValidationError(
                        f"Automatic registration error for {supercls.__name__} instance: {e}",
                        suggestions=[
                            "Check that the instance meets registry requirements",
                            "Verify registry validation settings",
                            "Consider disabling automatic registration if instances don't conform",
                        ],
                        context={
                            "operation": "automatic_instance_registration",
                            "tracked_class": supercls.__name__,
                            "registry_name": cls.__name__,
                        },
                    )
                    logger.debug(str(enhanced_error))
                return obj

        # Copy meta attributes from the original metaclass
        EnhancedRegistrationMeta.__name__ = meta.__name__
        EnhancedRegistrationMeta.__qualname__ = meta.__qualname__
        EnhancedRegistrationMeta.__module__ = meta.__module__
        EnhancedRegistrationMeta.__doc__ = meta.__doc__

        # Create and return the new tracked class
        return EnhancedRegistrationMeta(supercls.__name__, (supercls,), {})

    @classmethod
    def cleanup(cls) -> int:
        """
        Clean up dead references in the repository with enhanced reporting.

        This method removes all entries that contain dead weak references
        (references to objects that have been garbage collected).

        Returns:
            int: The number of dead references removed.
        """
        dead_refs = []
        cleanup_errors = []

        for key, value in cls._repository.items():
            try:
                if isinstance(value, weakref.ref) and value() is None:
                    dead_refs.append(key)
            except Exception as e:
                cleanup_errors.append({"key": str(key), "error": str(e)})

        # Remove dead references
        for key in dead_refs:
            try:
                del cls._repository[key]
            except Exception as e:
                cleanup_errors.append(
                    {"key": str(key), "error": f"Failed to remove dead reference: {e}"}
                )

        # Log cleanup results
        if dead_refs:
            logger.debug(
                f"Cleaned up {len(dead_refs)} dead references from {cls.__name__}"
            )

        if cleanup_errors:
            logger.warning(
                f"Encountered {len(cleanup_errors)} errors during cleanup in {cls.__name__}"
            )

        return len(dead_refs)

    @classmethod
    def diagnose_instance_failure(cls, failed_instance: Any) -> dict:
        """
        Diagnose why an instance failed to register and provide detailed suggestions.

        Parameters:
            failed_instance: The instance that failed to register.

        Returns:
            Dictionary containing diagnostic information and suggestions.
        """
        diagnosis = {
            "instance_type": type(failed_instance).__name__,
            "instance_str": str(failed_instance)[:100],
            "is_weakref_compatible": True,
            "validation_errors": [],
            "suggestions": [],
            "registry_config": {
                "strict_mode": cls._strict,
                "abstract_mode": cls._abstract,
                "strict_weakref": cls._strict_weakref,
                "registry_name": cls.__name__,
            },
        }

        # Test weak reference compatibility
        try:
            weakref.ref(failed_instance)
        except TypeError as e:
            diagnosis["is_weakref_compatible"] = False
            diagnosis["validation_errors"].append(
                {
                    "type": "weakref_compatibility",
                    "error": str(e),
                    "suggestions": [
                        "Object cannot be weakly referenced",
                        "Consider using strong reference mode (strict_weakref=False)",
                        "Avoid registering primitive types in weak reference mode",
                    ],
                }
            )

        # Test inheritance if in abstract mode
        if cls._abstract:
            try:
                validate_instance_hierarchy(failed_instance, expected_type=cls)
            except (InheritanceError, ValidationError) as e:
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
                validate_instance_structure(failed_instance, expected_type=protocol)
            except (ConformanceError, ValidationError) as e:
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
                "This registry requires inheritance - ensure proper class hierarchy"
            )

        if cls._strict_weakref and not diagnosis["is_weakref_compatible"]:
            diagnosis["suggestions"].append(
                "This registry requires weak reference compatibility"
            )

        diagnosis["suggestions"].append(
            f"Use {cls.__name__}.validate_artifact(your_instance) to test before registration"
        )

        return diagnosis

    @classmethod
    def get_instance_stats(cls) -> dict:
        """
        Get statistics about registered instances.

        Returns:
            Dictionary containing instance statistics and health information.
        """
        stats = {
            "total_instances": len(cls._repository),
            "alive_instances": 0,
            "dead_references": 0,
            "strong_references": 0,
            "weak_references": 0,
            "instance_types": {},
            "health_status": "unknown",
        }

        type_counts = {}

        for key, value in cls._repository.items():
            if isinstance(value, weakref.ref):
                stats["weak_references"] += 1
                actual_value = value()
                if actual_value is None:
                    stats["dead_references"] += 1
                else:
                    stats["alive_instances"] += 1
                    instance_type = type(actual_value).__name__
                    type_counts[instance_type] = type_counts.get(instance_type, 0) + 1
            else:
                stats["strong_references"] += 1
                stats["alive_instances"] += 1
                instance_type = type(value).__name__
                type_counts[instance_type] = type_counts.get(instance_type, 0) + 1

        stats["instance_types"] = type_counts

        # Determine health status
        if stats["total_instances"] == 0:
            stats["health_status"] = "empty"
        elif stats["dead_references"] == 0:
            stats["health_status"] = "healthy"
        elif stats["dead_references"] < stats["alive_instances"]:
            stats["health_status"] = "needs_cleanup"
        else:
            stats["health_status"] = "mostly_dead"

        return stats

    @classmethod
    def validate_instance_batch(cls, instances: dict) -> dict:
        """
        Validate multiple instances without registering them.

        Parameters:
            instances: Dictionary mapping keys to instances.

        Returns:
            Dictionary containing validation results for each instance.
        """
        results = {"valid": [], "invalid": [], "results": {}}

        for key, instance in instances.items():
            try:
                cls._intern_artifact(instance)
                results["valid"].append(key)
                results["results"][key] = {
                    "status": "valid",
                    "instance_type": type(instance).__name__,
                    "weakref_compatible": True,
                }

                # Test weak reference compatibility
                try:
                    weakref.ref(instance)
                except TypeError:
                    results["results"][key]["weakref_compatible"] = False

            except (ValidationError, ConformanceError, InheritanceError) as e:
                results["invalid"].append(key)
                results["results"][key] = {
                    "status": "invalid",
                    "error": str(e),
                    "suggestions": getattr(e, "suggestions", []),
                    "instance_type": type(instance).__name__,
                }
            except Exception as e:
                results["invalid"].append(key)
                results["results"][key] = {
                    "status": "error",
                    "error": f"Unexpected error: {e}",
                    "suggestions": [
                        "Check that the object meets registry requirements"
                    ],
                    "instance_type": type(instance).__name__,
                }

        return results

    @classmethod
    def get_memory_usage_info(cls) -> dict:
        """
        Get information about memory usage and weak reference health.

        Returns:
            Dictionary containing memory usage information.
        """
        import sys

        info = {
            "repository_size": len(cls._repository),
            "estimated_memory_bytes": sys.getsizeof(cls._repository),
            "weak_references": 0,
            "strong_references": 0,
            "dead_references": 0,
            "cleanup_recommended": False,
        }

        for value in cls._repository.values():
            if isinstance(value, weakref.ref):
                info["weak_references"] += 1
                if value() is None:
                    info["dead_references"] += 1
                info["estimated_memory_bytes"] += sys.getsizeof(value)
            else:
                info["strong_references"] += 1
                info["estimated_memory_bytes"] += sys.getsizeof(value)

        # Recommend cleanup if more than 10% of references are dead
        if info["weak_references"] > 0:
            dead_percentage = info["dead_references"] / info["weak_references"]
            info["cleanup_recommended"] = dead_percentage > 0.10

        return info
