from typing import (
    Any,
    Generator,
    Generic,
    Hashable,
    Iterator,
    TypeVar,
    overload,
    Union,
    Optional,
    cast,
    Dict,
    MutableMapping,
)

from registry.core._validator import (
    ValidationError,
    ConformanceError,
    InheritanceError,
    clear_validation_cache,
    get_cache_stats,
    configure_validation_cache,
)

from .accessor import RegistryAccessorMixin, RegistryError
from .mutator import RegistryMutatorMixin

# -----------------------------------------------------------------------------
# Enhanced RegistryError with Rich Context - Remove duplicate definition
# -----------------------------------------------------------------------------

# RegistryError is now imported from .accessor

# -----------------------------------------------------------------------------
# Type Variables
# -----------------------------------------------------------------------------

K = TypeVar("K", bound=Hashable)  # Keys must be hashable.
T = TypeVar("T")  # Registered items.

# -----------------------------------------------------------------------------
# Enhanced Registry Validator Mixin
# -----------------------------------------------------------------------------


class ImmutableRegistryValidatorMixin(RegistryAccessorMixin[K, T], Generic[K, T]):
    """
    Mixin providing validation for immutable registry operations.

    This mixin extends the RegistryAccessorMixin to add validation
    to all read operations on the registry. It provides methods for
    validating identifiers and artifacts during retrieval operations.
    """

    @classmethod
    def _intern_identifier(cls, value: Any) -> K:
        """
        Validate an identifier when storing in registry.

        Parameters:
            value: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValidationError: If the identifier is invalid.
        """
        if not isinstance(value, Hashable):

            suggestions = [
                "Ensure the key is hashable (str, int, tuple, etc.)",
                f"Got {type(value).__name__}, which is not hashable",
                "Try converting to string: str(your_key)",
            ]
            context = {
                "expected_type": "Hashable",
                "actual_type": type(value).__name__,
                "artifact_name": str(value),
            }
            raise ValidationError(
                f"Key must be hashable, got {type(value).__name__}",
                suggestions,
                context,
            )
        return value  # type: ignore

    @classmethod
    def _intern_artifact(cls, value: T) -> T:
        """
        Validate an artifact when storing in registry.

        Parameters:
            value: The artifact to validate.

        Returns:
            The validated artifact.
        """
        return value

    @classmethod
    def _extern_identifier(cls, value: Any) -> K:
        """
        Validate an identifier when retrieving from registry.

        Parameters:
            value: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValidationError: If the identifier is invalid.
        """
        if not isinstance(value, Hashable):

            suggestions = [
                "Ensure the key is hashable (str, int, tuple, etc.)",
                f"Got {type(value).__name__}, which is not hashable",
            ]
            context = {
                "expected_type": "Hashable",
                "actual_type": type(value).__name__,
                "artifact_name": str(value),
            }
            raise ValidationError(
                f"Key must be hashable, got {type(value).__name__}",
                suggestions,
                context,
            )
        return value  # type: ignore

    @classmethod
    def _extern_artifact(cls, value: T) -> T:
        """
        Validate an artifact when retrieving from registry.

        Parameters:
            value: The artifact to validate.

        Returns:
            The validated artifact.
        """
        return value

    @classmethod
    def get_artifact(cls, key: K) -> T:
        """
        Retrieve an artifact from the registry with validation.

        Parameters:
            key: The key of the artifact to retrieve.

        Returns:
            The artifact if it exists.

        Raises:
            RegistryError: If the key is not in the registry (with rich context).
            ValidationError: If the key or artifact validation fails.
        """
        try:
            validated_key = cls._intern_identifier(key)
            item = cls._get_artifact(validated_key)
            return cls._extern_artifact(item)
        except RegistryError:
            # Re-raise RegistryError as-is - it already has rich context from accessor
            raise
        except ValidationError:
            # Re-raise validation errors as-is (they already have rich context)
            raise
        except Exception as e:
            # Convert other exceptions to RegistryError with rich context for access operations
            suggestions = [
                f"Check that key '{key}' exists in the registry",
                "Use has_identifier() to check existence first",
                f"Registry {getattr(cls, '__name__', 'Unknown')} contains {cls._len_mapping()} items",
                "Use iter_identifiers() to see all available keys",
            ]
            context = {
                "operation": "get_artifact",
                "registry_name": getattr(cls, "__name__", "Unknown"),
                "registry_type": cls.__class__.__name__,
                "key": str(key),
                "key_type": type(key).__name__,
                "registry_size": cls._len_mapping(),
            }
            raise RegistryError(
                f"Failed to retrieve artifact with key '{key}': {e}",
                suggestions,
                context,
            ) from e

    @classmethod
    def has_identifier(cls, key: K) -> bool:
        """
        Check if an artifact with the given key exists in the registry.

        Parameters:
            key: The key to check.

        Returns:
            True if the artifact exists, False otherwise.
        """
        try:
            validated_key = cls._intern_identifier(key)
            return cls._has_identifier(validated_key)
        except ValidationError:
            return False

    @classmethod
    def has_artifact(cls, item: T) -> bool:
        """
        Check if an artifact with the given value exists in the registry.

        Parameters:
            item: The artifact to check.

        Returns:
            True if the artifact exists, False otherwise.
        """
        try:
            validated_item = cls._intern_artifact(item)
            return cls._has_artifact(validated_item)
        except ValidationError:
            return False

    @classmethod
    def iter_identifiers(cls) -> Iterator[K]:
        """
        Iterate over all artifact keys in the registry.

        Returns:
            Iterator over all keys in the registry.
        """
        return cls._iter_mapping()

    @classmethod
    def validate_artifact(cls, item: Any) -> T:
        """
        Validate an artifact without registering it.

        Parameters:
            item: The artifact to validate.

        Returns:
            The validated artifact.

        Raises:
            ValidationError: If the artifact is invalid.
        """
        try:
            validated_item = cls._intern_artifact(item)
            return cls._extern_artifact(validated_item)
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to ValidationError

            suggestions = [
                "Check that the artifact meets the registry requirements",
                "Use the registry's validation mode settings",
            ]
            context = {
                "operation": "validate_artifact",
                "artifact_name": getattr(item, "__name__", str(item)),
            }
            raise ValidationError(
                f"Artifact validation failed: {e}", suggestions, context
            ) from e


# -----------------------------------------------------------------------------
# Mutable Registry Validator Mixin
# -----------------------------------------------------------------------------


class MutableRegistryValidatorMixin(
    ImmutableRegistryValidatorMixin[K, T], RegistryMutatorMixin[K, T], Generic[K, T]
):
    """
    Mixin providing validation for mutable registry operations.

    This mixin extends both ImmutableRegistryValidatorMixin and RegistryMutatorMixin
    to add validation to all write operations on the registry.
    """

    @classmethod
    def _identifier_of(cls, item: T) -> K:
        """Extract the identifier from an artifact."""
        raise NotImplementedError("Subclasses must implement _identifier_of")

    @overload
    @classmethod
    def register_artifact(cls, key: K, item: T) -> T:
        """Register an artifact with explicit key."""
        ...

    @overload
    @classmethod
    def register_artifact(cls, key: T, item: None = None) -> T:
        """Register an artifact using extracted key."""
        ...

    @classmethod
    def register_artifact(cls, key: Union[K, T], item: Optional[T] = None) -> T:
        """
        Register an artifact in the registry with enhanced validation.

        Supports two calling patterns:
        1. register_artifact(key, item) - explicit key
        2. register_artifact(item) - extracted key from item

        Parameters:
            key: Either the explicit key or the item itself
            item: The item to register (None for single-argument form)

        Returns:
            The registered item

        Raises:
            ValidationError: If validation fails with detailed context.
            RegistryError: If registration fails.
        """
        try:
            if item is None:
                # Single argument case: register_artifact(item)
                artifact = cast(T, key)  # key is actually the item in this case
                identifier = cls._identifier_of(artifact)
                validated_item = cls._intern_artifact(artifact)
                validated_key = cls._intern_identifier(identifier)
                cls._set_artifact(validated_key, validated_item)
                return artifact
            else:
                # Two argument case: register_artifact(key, item)
                actual_key = key  # key is actually the key in this case
                artifact = item
                validated_key = cls._intern_identifier(actual_key)
                validated_item = cls._intern_artifact(artifact)
                cls._set_artifact(validated_key, validated_item)
                return artifact

        except (ValidationError, ConformanceError, InheritanceError):
            # Re-raise validation errors as-is (they already have rich context)
            raise
        except Exception as e:
            # Convert other exceptions to ValidationError with context

            artifact_name = getattr(
                artifact if "artifact" in locals() else key, "__name__", str(key)
            )
            suggestions = [
                "Check that the artifact meets the registry requirements",
                "Ensure the key is unique and hashable",
                "Verify that validation mode settings are correct",
            ]
            context = {"operation": "register_artifact", "artifact_name": artifact_name}
            raise ValidationError(
                f"Registration failed: {e}", suggestions, context
            ) from e


    @classmethod
    def unregister_identifier(cls, key: K) -> None:
        """
        Unregister an artifact from the registry using its identifier/key.

        Parameters:
            key: The key/identifier of the artifact to unregister.

        Raises:
            ValidationError: If key validation fails.
            RegistryError: If the key does not exist in the registry (with rich context).
        """
        try:
            validated_key = cls._intern_identifier(key)
            cls._del_artifact(validated_key)
        except RegistryError:
            # Re-raise RegistryError as-is - it already has rich context from mutator
            raise
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to RegistryError with rich context for access operations
            suggestions = [
                f"Check that key '{key}' exists in the registry",
                "Use has_identifier() to check existence first",
                f"Registry {getattr(cls, '__name__', 'Unknown')} contains {cls._len_mapping()} items",
                "Use iter_identifiers() to see all available keys",
            ]
            context = {
                "operation": "unregister_identifier",
                "registry_name": getattr(cls, "__name__", "Unknown"),
                "registry_type": cls.__class__.__name__,
                "key": str(key),
                "key_type": type(key).__name__,
                "registry_size": cls._len_mapping(),
            }
            raise RegistryError(
                f"Unregistration failed for key '{key}': {e}", suggestions, context
            ) from e

    @classmethod
    def unregister_artifact(cls, item: T) -> None:
        """
        Unregister an artifact from the registry using the artifact itself.

        This method extracts the identifier from the artifact using _identifier_of()
        and then removes the corresponding entry from the registry.

        Parameters:
            item: The artifact to unregister.

        Raises:
            ValidationError: If artifact validation fails or identifier extraction fails.
            RegistryError: If the artifact is not found in the registry (with rich context).
        """
        try:
            # Extract the identifier from the artifact
            identifier = cls._identifier_of(item)
            # Use unregister_identifier to do the actual removal
            cls.unregister_identifier(identifier)
        except RegistryError:
            # Re-raise RegistryError as-is but update context to reflect artifact-based operation
            raise
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # Convert other exceptions to ValidationError with context
            artifact_name = getattr(item, "__name__", str(item))
            suggestions = [
                f"Check that artifact '{artifact_name}' exists in the registry",
                "Ensure the artifact can be identified using _identifier_of()",
                "Use has_artifact() to check existence first",
                f"Registry {getattr(cls, '__name__', 'Unknown')} contains {cls._len_mapping()} items",
            ]
            context = {
                "operation": "unregister_artifact",
                "registry_name": getattr(cls, "__name__", "Unknown"),
                "registry_type": cls.__class__.__name__,
                "artifact_name": artifact_name,
                "artifact_type": type(item).__name__,
                "registry_size": cls._len_mapping(),
            }
            raise ValidationError(
                f"Unregistration failed for artifact '{artifact_name}': {e}", suggestions, context
            ) from e

    @classmethod
    def clear_artifacts(cls) -> None:
        """
        Clear all artifacts from the registry.
        """
        cls._clear_mapping()

    # -----------------------------------------------------------------------------
    # Enhanced Registry Management Methods
    # -----------------------------------------------------------------------------

    @classmethod
    def validate_registry_state(cls) -> Dict[str, Any]:
        """
        Validate the current state of the registry and return a report.

        Returns:
            Dict containing validation results and statistics.
        """
        report = {
            "total_artifacts": cls._len_mapping(),
            "validation_errors": [],
            "warnings": [],
            "cache_stats": get_cache_stats(),
        }

        # Validate each artifact in the registry
        for key in cls._iter_mapping():
            try:
                artifact = cls._get_artifact(key)
                cls.validate_artifact(artifact)
            except ValidationError as e:
                report["validation_errors"].append(
                    {
                        "key": str(key),
                        "error": str(e),
                        "suggestions": getattr(e, "suggestions", []),
                    }
                )
            except Exception as e:
                report["warnings"].append(
                    {
                        "key": str(key),
                        "warning": f"Unexpected error during validation: {e}",
                    }
                )

        return report

    @classmethod
    def clear_validation_cache(cls) -> int:
        """
        Clear the validation cache and return number of entries cleared.

        Returns:
            Number of cache entries that were cleared.
        """
        return clear_validation_cache()

    @classmethod
    def configure_validation(
        cls, cache_size: Optional[int] = None, cache_ttl: Optional[float] = None
    ) -> None:
        """
        Configure validation settings for this registry.

        Parameters:
            cache_size: Maximum number of validation results to cache.
            cache_ttl: Time-to-live for cached validation results in seconds.
        """
        configure_validation_cache(cache_size, cache_ttl)

    @classmethod
    def get_validation_stats(cls) -> Dict[str, Any]:
        """
        Get validation statistics for this registry.

        Returns:
            Dictionary containing validation performance statistics.
        """
        return get_cache_stats()

    @classmethod
    def batch_validate(cls, items: Dict[K, T]) -> Dict[K, Union[bool, ValidationError]]:
        """
        Validate multiple items without registering them.

        Parameters:
            items: Dictionary of key-item pairs to validate.

        Returns:
            Dictionary mapping keys to validation results (True for success, ValidationError for failure).
        """
        results = {}

        for key, item in items.items():
            try:
                cls._intern_identifier(key)
                cls._intern_artifact(item)
                results[key] = True
            except ValidationError as e:
                results[key] = e
            except Exception as e:

                suggestions = ["Check that the item meets registry requirements"]
                context = {
                    "operation": "batch_validate",
                    "artifact_name": getattr(item, "__name__", str(item)),
                }
                results[key] = ValidationError(
                    f"Validation failed: {e}", suggestions, context
                )

        return results

    @classmethod
    def safe_register_batch(
        cls, items: Dict[K, T], skip_invalid: bool = True
    ) -> Dict[str, Any]:
        """
        Safely register multiple items with detailed error reporting.

        Parameters:
            items: Dictionary of key-item pairs to register.
            skip_invalid: If True, skip invalid items; if False, stop on first error.

        Returns:
            Dictionary containing registration results and statistics.
        """
        results = {"successful": [], "failed": [], "total": len(items), "errors": []}

        for key, item in items.items():
            try:
                cls.register_artifact(key, item)
                results["successful"].append(key)
            except (ValidationError, ConformanceError, InheritanceError) as e:
                error_info = {
                    "key": key,
                    "error": str(e),
                    "suggestions": getattr(e, "suggestions", []),
                    "context": getattr(e, "context", {}),
                }
                results["failed"].append(key)
                results["errors"].append(error_info)

                if not skip_invalid:
                    raise
            except Exception as e:

                enhanced_error = ValidationError(
                    f"Registration failed for key '{key}': {e}"
                )
                results["failed"].append(key)
                results["errors"].append(
                    {
                        "key": key,
                        "error": str(enhanced_error),
                        "suggestions": [],
                        "context": {},
                    }
                )

                if not skip_invalid:
                    raise enhanced_error from e

        return results
