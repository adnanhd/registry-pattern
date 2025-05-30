r"""Enhanced ConfigRegistry with improved validation system.

This module defines the ConfigRegistry class, which provides a registry
for objects along with their configuration dictionaries with comprehensive
validation, rich error context, and performance optimizations. It uses weak
references to keys to avoid memory leaks.

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
import logging
from abc import ABC

from typing_compat import Any, Dict, Generic, Hashable, Optional, Type, TypeVar

from ..mixin import MutableRegistryValidatorMixin, RegistryError
from ._dev_utils import get_protocol
from ._validator import (
    InheritanceError,
    ConformanceError,
    ValidationError,
    validate_instance_hierarchy,
    validate_instance_structure,
)

# Type variables for object and configuration
ObjT = TypeVar("ObjT", bound=Hashable)
CfgT = TypeVar("CfgT", bound=Dict[str, Any])
logger = logging.getLogger(__name__)


class ConfigRegistry(
    MutableRegistryValidatorMixin[Any, CfgT], ABC, Generic[ObjT, CfgT]
):
    """
    Enhanced registry for registering objects with configuration and comprehensive validation.

    This registry uses weak references for keys to prevent memory leaks,
    but stores them in a standard dictionary for more flexible access.
    It supports runtime validations with rich error feedback to ensure
    registered objects conform to expected inheritance and structural requirements.

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
        Initialize a subclass of ConfigRegistry with enhanced validation configuration.

        This method initializes the registry repository and sets validation flags.

        Parameters:
            strict (bool): If True, enforce structure validation with detailed feedback.
            abstract (bool): If True, enforce inheritance validation with helpful suggestions.
        """
        super().__init_subclass__()
        # Initialize with a standard dictionary rather than a WeakKeyDictionary
        cls._repository = dict()
        # Store validation flags
        cls._strict = strict
        cls._abstract = abstract

    @classmethod
    def _intern_identifier(cls, value: ObjT) -> Any:
        """
        Validate and transform a value before storing in the registry with enhanced error handling.

        This method validates the value and wraps it in a weakref.ref
        to prevent memory leaks.

        Parameters:
            value (ObjT): The value to validate and transform.

        Returns:
            Any: The validated and transformed value (a weakref).

        Raises:
            ValidationError: If the value validation fails with detailed context.
        """
        # Basic type validation for the value
        if not isinstance(value, Hashable):
            suggestions = [
                "Ensure the key object is hashable",
                f"Got {type(value).__name__}, which is not hashable",
                "Consider using objects with __hash__ method implemented",
                "Avoid mutable objects (lists, dicts) as keys",
            ]
            context = {
                "registry_name": cls.__name__,
                "registry_type": "ConfigRegistry",
                "operation": "key_validation",
                "expected_type": "Hashable",
                "actual_type": type(value).__name__,
                "artifact_name": str(value),
            }
            raise ValidationError(
                f"Key must be hashable, got {type(value).__name__}",
                suggestions,
                context,
            )

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
                            "registry_type": "ConfigRegistry",
                            "base_class": cls.__name__,
                        }
                    )
                if hasattr(e, "suggestions"):
                    e.suggestions.extend(
                        [
                            f"Register objects that inherit from or are compatible with {cls.__name__}",
                            "Check that the object's class has proper inheritance",
                            "Verify that you're registering the correct type of object",
                        ]
                    )
                raise

        # Apply conformance checking if strict mode is enabled
        if cls._strict:
            try:
                protocol = get_protocol(cls)
                value = validate_instance_structure(value, expected_type=protocol)
            except ConformanceError as e:
                protocol = get_protocol(cls)

                # Add registry-specific context
                if hasattr(e, "context"):
                    e.context.update(
                        {
                            "registry_name": cls.__name__,
                            "registry_mode": "strict",
                            "registry_type": "ConfigRegistry",
                            "required_protocol": (
                                str(protocol) if protocol else "unknown"
                            ),
                        }
                    )
                if hasattr(e, "suggestions"):
                    e.suggestions.extend(
                        [
                            f"Use cls.validate_artifact(your_config) to check conformance before registration",
                            f"Registry {cls.__name__} requires strict protocol conformance",
                            "Ensure the object implements all required methods",
                        ]
                    )
                raise
            except Exception as e:
                # Convert unexpected errors to ValidationError
                suggestions = [
                    "Check that the object implements the required protocol",
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
                    f"Object structure validation failed: {e}", suggestions, context
                )
                raise enhanced_error from e

        # Wrap the value in a weakref.ref with enhanced error handling
        try:
            return weakref.ref(value)
        except TypeError as e:
            suggestions = [
                "Object cannot be weakly referenced",
                "Avoid registering primitive types (int, str, etc.) as keys",
                "Use objects that support weak references",
                "Consider wrapping primitive values in a custom class",
            ]
            context = {
                "registry_name": cls.__name__,
                "operation": "weak_reference_creation",
                "artifact_name": str(value),
                "artifact_type": type(value).__name__,
            }
            enhanced_error = ValidationError(
                f"Cannot create weak reference for key: {e}", suggestions, context
            )
            raise enhanced_error from e

    @classmethod
    def _is_weakref_compatible(cls, obj: Any) -> bool:
        """Check if an object is compatible with weak references."""
        try:
            weakref.ref(obj)
            return True
        except TypeError:
            return False

    @classmethod
    def _extern_identifier(cls, value: Any) -> ObjT:
        """
        Validate and transform a value when retrieving from the registry with enhanced error handling.

        If the value is a weakref.ref, call it to get the actual object.
        If the object has been garbage collected, raise RegistryError for access operations.

        Parameters:
            value (Any): The value to validate and transform.

        Returns:
            ObjT: The validated and transformed value.

        Raises:
            RegistryError: If the value is a dead weakref (with rich context).
            ValidationError: If validation fails.
        """
        # If the value is already a weakref, get the referenced object
        if isinstance(value, weakref.ref):
            actual_value = value()
            if actual_value is None:
                # The referenced object has been garbage collected
                suggestions = [
                    "The referenced object has been garbage collected",
                    "Keep a strong reference to the key object to prevent garbage collection",
                    "Use cleanup() method to remove dead references",
                    "Consider the object's lifetime when using as a key",
                ]
                context = {
                    "operation": "weak_reference_resolution",
                    "registry_name": cls.__name__,
                    "registry_type": "ConfigRegistry",
                    "weakref_address": str(value),
                }
                raise RegistryError(
                    "Weakref key is dead (object has been collected)",
                    suggestions,
                    context,
                )
            return actual_value

        # If it's not a weakref, make sure it's hashable
        if not isinstance(value, Hashable):
            suggestions = [
                "Key must be hashable",
                f"Got {type(value).__name__}, which is not hashable",
                "Use hashable objects as keys",
            ]
            context = {
                "operation": "key_validation",
                "registry_name": cls.__name__,
                "registry_type": "ConfigRegistry",
                "expected_type": "Hashable",
                "actual_type": type(value).__name__,
                "key": str(value),
            }
            raise ValidationError(
                f"Key must be hashable, got {type(value).__name__}",
                suggestions,
                context,
            )

        return value

    @classmethod
    def _intern_artifact(cls, value: CfgT) -> CfgT:
        """
        Validate a configuration before storing in the registry with enhanced error handling.

        This method ensures the configuration is a dictionary and validates its structure.

        Parameters:
            value (CfgT): The configuration to validate.

        Returns:
            CfgT: The validated configuration.

        Raises:
            ValidationError: If the configuration validation fails with detailed feedback.
        """
        if not isinstance(value, dict):
            suggestions = [
                "Configuration must be a dictionary",
                f"Got {type(value).__name__}, expected dict",
                "Use dict() or {} to create configuration",
                "Ensure configuration data is in key-value format",
            ]
            context = {
                "registry_name": cls.__name__,
                "registry_type": "ConfigRegistry",
                "operation": "config_validation",
                "expected_type": "dict",
                "actual_type": type(value).__name__,
                "artifact_name": str(value)[:100],
            }
            raise ValidationError(
                f"Configuration must be a dictionary, got {type(value).__name__}",
                suggestions,
                context,
            )

        # Additional configuration validation can be added here
        # For example, checking for required keys, validating value types, etc.
        return value

    @classmethod
    def _extern_artifact(cls, value: CfgT) -> CfgT:
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
        Find the weakref key that points to the given object with enhanced error handling.

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
        dead_refs = []  # Track dead references for cleanup

        for weakref_key in cls._repository:
            if isinstance(weakref_key, weakref.ref):
                try:
                    obj = weakref_key()
                    if obj is not None and obj is key:
                        return weakref_key
                    elif obj is None:
                        dead_refs.append(weakref_key)
                except Exception:
                    # Handle any unexpected errors during weak reference resolution
                    dead_refs.append(weakref_key)

        # Log dead references for potential cleanup
        if dead_refs:
            logger.debug(
                f"Found {len(dead_refs)} dead references during key search in {cls.__name__}"
            )

        return None

    @classmethod
    def has_identifier(cls, key: Any) -> bool:
        """
        Check if a key exists in the registry with enhanced error handling.

        Parameters:
            key: The key to check.

        Returns:
            bool: True if the key exists, False otherwise.
        """
        try:
            return cls._find_weakref_key(key) is not None
        except Exception as e:
            # Log unexpected errors but don't raise them for existence checks
            logger.debug(f"Error checking key existence in {cls.__name__}: {e}")
            return False

    @classmethod
    def get_artifact(cls, key: Any) -> CfgT:
        """
        Get an artifact from the registry with enhanced error handling.

        Parameters:
            key: The key to look up.

        Returns:
            CfgT: The artifact associated with the key.

        Raises:
            RegistryError: If the key is not found or retrieval fails (with rich context).
            ValidationError: If key validation fails.
        """
        try:
            weakref_key = cls._find_weakref_key(key)
            if weakref_key is None:
                suggestions = [
                    f"Key not found in {cls.__name__} registry",
                    "Use has_identifier() to check existence first",
                    "Verify that the key object hasn't been garbage collected",
                    "Check that the key was registered properly",
                    f"Registry contains {len(cls._repository)} configurations",
                ]
                context = {
                    "operation": "get_artifact",
                    "registry_name": cls.__name__,
                    "registry_type": "ConfigRegistry",
                    "key": str(key),
                    "key_type": type(key).__name__,
                    "registry_size": len(cls._repository),
                    "is_weakref_compatible": cls._is_weakref_compatible(key),
                }
                raise RegistryError(
                    f"Key '{key}' is not found in the repository", suggestions, context
                )

            return cls._repository[weakref_key]
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except RegistryError:
            # Re-raise RegistryError as-is
            raise
        except Exception as e:
            # Convert unexpected errors to RegistryError with rich context for access operations
            suggestions = [
                "Check that the key exists and is accessible",
                "Verify that the key object hasn't been garbage collected",
                "Use cleanup() to remove dead references",
                f"Registry {cls.__name__} contains {len(cls._repository)} items",
            ]
            context = {
                "operation": "get_artifact",
                "registry_name": cls.__name__,
                "registry_type": "ConfigRegistry",
                "key": str(key),
                "key_type": type(key).__name__,
                "registry_size": len(cls._repository),
            }
            raise RegistryError(
                f"Failed to retrieve artifact for key '{key}': {e}",
                suggestions,
                context,
            ) from e

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        """
        Custom subclass check for runtime validation with caching.

        Parameters:
            value (Any): The value to check.

        Returns:
            bool: True if the value passes all validations; False otherwise.
        """
        try:
            # Check if the value can be used as a key
            cls._intern_identifier(value)
            return True
        except (ValidationError, InheritanceError, ConformanceError, TypeError):
            return False

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

        class EnhancedConfigMeta(meta):
            def __call__(self, *args: Any, **kwds: Any) -> ObjT:
                # Create the instance
                obj = super().__call__(*args, **kwds)
                try:
                    # Register the instance with its initialization kwargs
                    cls.register_artifact(obj, kwds)  # type: ignore
                except (ValidationError, ConformanceError, InheritanceError) as e:
                    # Add automatic configuration registration context
                    if hasattr(e, "context"):
                        e.context.update(
                            {
                                "operation": "automatic_config_registration",
                                "tracked_class": supercls.__name__,
                                "instance_kwargs": str(kwds)[:100] if kwds else "none",
                            }
                        )
                    logger.debug(
                        f"Automatic config registration failed for {supercls.__name__} instance: {e}"
                    )
                except Exception as e:
                    enhanced_error = ValidationError(
                        f"Automatic config registration error for {supercls.__name__} instance: {e}",
                        suggestions=[
                            "Check that the instance can be used as a registry key",
                            "Verify that the configuration is a valid dictionary",
                            "Consider manual registration if automatic registration fails",
                        ],
                        context={
                            "operation": "automatic_config_registration",
                            "tracked_class": supercls.__name__,
                            "registry_name": cls.__name__,
                        },
                    )
                    logger.debug(str(enhanced_error))
                return obj

        # Copy meta attributes
        EnhancedConfigMeta.__name__ = meta.__name__
        EnhancedConfigMeta.__qualname__ = meta.__qualname__
        EnhancedConfigMeta.__module__ = meta.__module__
        EnhancedConfigMeta.__doc__ = meta.__doc__

        # Create and return the new tracked class
        return EnhancedConfigMeta(supercls.__name__, (supercls,), {})

    @classmethod
    def cleanup(cls) -> int:
        """
        Clean up dead references in the repository with enhanced reporting.

        This method removes all keys that are dead weakrefs
        (references to objects that have been garbage collected).

        Returns:
            int: The number of dead references removed.
        """
        dead_refs = []
        cleanup_errors = []

        for key in cls._repository:
            try:
                if isinstance(key, weakref.ref) and key() is None:
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
    def diagnose_config_failure(cls, failed_object: Any, failed_config: Any) -> dict:
        """
        Diagnose why an object-config pair failed to register.

        Parameters:
            failed_object: The object that failed to register.
            failed_config: The configuration that failed to register.

        Returns:
            Dictionary containing diagnostic information and suggestions.
        """
        diagnosis = {
            "object_type": type(failed_object).__name__,
            "config_type": type(failed_config).__name__,
            "object_str": str(failed_object)[:100],
            "config_str": str(failed_config)[:100],
            "is_object_hashable": isinstance(failed_object, Hashable),
            "is_config_dict": isinstance(failed_config, dict),
            "is_weakref_compatible": True,
            "validation_errors": [],
            "suggestions": [],
            "registry_config": {
                "strict_mode": cls._strict,
                "abstract_mode": cls._abstract,
                "registry_name": cls.__name__,
            },
        }

        # Test object weak reference compatibility
        try:
            weakref.ref(failed_object)
        except TypeError as e:
            diagnosis["is_weakref_compatible"] = False
            diagnosis["validation_errors"].append(
                {
                    "type": "weakref_compatibility",
                    "error": str(e),
                    "suggestions": [
                        "Object cannot be weakly referenced",
                        "Avoid registering primitive types as keys",
                        "Use objects that support weak references",
                    ],
                }
            )

        # Test config validation
        if not isinstance(failed_config, dict):
            diagnosis["validation_errors"].append(
                {
                    "type": "config_type_validation",
                    "error": f"Configuration must be dict, got {type(failed_config).__name__}",
                    "suggestions": [
                        "Configuration must be a dictionary",
                        "Use dict() or {} to create configuration",
                        "Convert configuration data to dictionary format",
                    ],
                }
            )

        # Test object inheritance if in abstract mode
        if cls._abstract:
            try:
                validate_instance_hierarchy(failed_object, expected_type=cls)
            except (InheritanceError, ValidationError) as e:
                diagnosis["validation_errors"].append(
                    {
                        "type": "inheritance_validation",
                        "error": str(e),
                        "suggestions": getattr(e, "suggestions", []),
                    }
                )
                diagnosis["suggestions"].extend(getattr(e, "suggestions", []))

        # Test object protocol conformance if in strict mode
        if cls._strict:
            protocol = get_protocol(cls)
            try:
                validate_instance_structure(failed_object, expected_type=protocol)
            except (ConformanceError, ValidationError) as e:
                diagnosis["validation_errors"].append(
                    {
                        "type": "protocol_validation",
                        "error": str(e),
                        "suggestions": getattr(e, "suggestions", []),
                    }
                )
                diagnosis["suggestions"].extend(getattr(e, "suggestions", []))

        # Add general suggestions
        diagnosis["suggestions"].extend(
            [
                f"Use {cls.__name__}.validate_artifact(your_config) to test configuration",
                "Ensure the object can serve as a hashable key",
                "Check that the configuration is a proper dictionary",
            ]
        )

        return diagnosis

    @classmethod
    def get_config_stats(cls) -> dict:
        """
        Get statistics about registered configurations.

        Returns:
            Dictionary containing configuration statistics and health information.
        """
        stats = {
            "total_configs": len(cls._repository),
            "alive_keys": 0,
            "dead_references": 0,
            "config_sizes": [],
            "key_types": {},
            "health_status": "unknown",
            "average_config_size": 0,
        }

        key_type_counts = {}
        config_sizes = []

        for key, config in cls._repository.items():
            if isinstance(key, weakref.ref):
                actual_key = key()
                if actual_key is None:
                    stats["dead_references"] += 1
                else:
                    stats["alive_keys"] += 1
                    key_type = type(actual_key).__name__
                    key_type_counts[key_type] = key_type_counts.get(key_type, 0) + 1
            else:
                stats["alive_keys"] += 1
                key_type = type(key).__name__
                key_type_counts[key_type] = key_type_counts.get(key_type, 0) + 1

            # Analyze config size if it's a dict
            if isinstance(config, dict):
                config_sizes.append(len(config))

        stats["key_types"] = key_type_counts
        stats["config_sizes"] = config_sizes

        if config_sizes:
            stats["average_config_size"] = sum(config_sizes) / len(config_sizes)

        # Determine health status
        if stats["total_configs"] == 0:
            stats["health_status"] = "empty"
        elif stats["dead_references"] == 0:
            stats["health_status"] = "healthy"
        elif stats["dead_references"] < stats["alive_keys"]:
            stats["health_status"] = "needs_cleanup"
        else:
            stats["health_status"] = "mostly_dead"

        return stats

    @classmethod
    def validate_config_batch(cls, configs: dict) -> dict:
        """
        Validate multiple object-config pairs without registering them.

        Parameters:
            configs: Dictionary mapping objects to their configurations.

        Returns:
            Dictionary containing validation results for each pair.
        """
        results = {"valid": [], "invalid": [], "results": {}}

        for obj, config in configs.items():
            obj_key = str(obj)  # Use string representation as key for results

            try:
                # Validate object as key
                cls._intern_identifier(obj)
                # Validate configuration
                cls._intern_artifact(config)

                results["valid"].append(obj_key)
                results["results"][obj_key] = {
                    "status": "valid",
                    "object_type": type(obj).__name__,
                    "config_type": type(config).__name__,
                    "config_size": (
                        len(config) if isinstance(config, dict) else "unknown"
                    ),
                }

            except (ValidationError, ConformanceError, InheritanceError) as e:
                results["invalid"].append(obj_key)
                results["results"][obj_key] = {
                    "status": "invalid",
                    "error": str(e),
                    "suggestions": getattr(e, "suggestions", []),
                    "object_type": type(obj).__name__,
                    "config_type": type(config).__name__,
                }
            except Exception as e:
                results["invalid"].append(obj_key)
                results["results"][obj_key] = {
                    "status": "error",
                    "error": f"Unexpected error: {e}",
                    "suggestions": [
                        "Check that both object and config meet registry requirements"
                    ],
                    "object_type": type(obj).__name__,
                    "config_type": type(config).__name__,
                }

        return results

    @classmethod
    def export_configs(cls, include_dead_refs: bool = False) -> dict:
        """
        Export all configurations to a dictionary format.

        Parameters:
            include_dead_refs: Whether to include entries with dead weak references.

        Returns:
            Dictionary mapping object representations to their configurations.
        """
        exported = {}
        dead_count = 0

        for key, config in cls._repository.items():
            if isinstance(key, weakref.ref):
                actual_key = key()
                if actual_key is None:
                    dead_count += 1
                    if include_dead_refs:
                        exported[f"<dead_ref_{dead_count}>"] = config
                else:
                    exported[str(actual_key)] = config
            else:
                exported[str(key)] = config

        return {
            "configs": exported,
            "metadata": {
                "total_entries": len(cls._repository),
                "exported_entries": len(exported),
                "dead_references": dead_count,
                "registry_name": cls.__name__,
            },
        }

    @classmethod
    def import_configs(cls, config_data: dict, overwrite: bool = False) -> dict:
        """
        Import configurations from a dictionary format.

        Parameters:
            config_data: Dictionary containing configuration data to import.
            overwrite: Whether to overwrite existing configurations.

        Returns:
            Dictionary containing import results and statistics.
        """
        results = {"imported": 0, "skipped": 0, "failed": 0, "errors": []}

        if "configs" not in config_data:
            results["errors"].append("Invalid format: missing 'configs' key")
            return results

        for obj_str, config in config_data["configs"].items():
            try:
                # This is a simplified import - in practice, you'd need a way to
                # reconstruct the original objects from their string representations
                # For now, we'll use the string as a key (this is just for demonstration)

                if not overwrite and cls.has_identifier(obj_str):
                    results["skipped"] += 1
                    continue

                # Validate and register
                cls.register_artifact(obj_str, config)
                results["imported"] += 1

            except Exception as e:
                results["failed"] += 1
                results["errors"].append({"object": obj_str, "error": str(e)})

        return results
