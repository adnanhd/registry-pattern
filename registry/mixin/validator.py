r"""Validation mixins and a process-local validation cache.

This module provides:

  - `ValidationCache`: a process-local cache keyed by (object identity, type name, operation).
    Expiry uses a monotonic clock; entries are touched on read for coarse LRU-like eviction.
    Returned suggestion lists are copied to avoid shared mutation.

  - Global cache helpers (`clear_validation_cache`, `get_cache_stats`, `configure_validation_cache`)
    operating on a single module-level cache instance.

  - `ImmutableValidatorMixin` and `MutableValidatorMixin`:
    front-ends that wrap accessor/mutator calls with identifier/artifact validation and
    convert failures to `ValidationError` or key/value errors from `_utils`.

Notes on timing:
  - Expiry checks inside the cache use `time.monotonic()`.
  - `get_cache_stats()` compares timestamps to `time.time()`. If your wall clock jumps,
    the expired count may be inaccurate; it does not affect eviction behavior.
"""

import time
from threading import RLock
from typing import (
    Any,
    Dict,
    Generic,
    Hashable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

from ..utils import (
    ConformanceError,
    InheritanceError,
    RegistryError,
    ValidationError,
    get_type_name,
)
from .accessor import RegistryAccessorMixin
from .mutator import RegistryMutatorMixin


class ValidationCache:
    """Process-local validation result cache with TTL and touch-on-read.

    Threading:
        Access to the internal map is guarded by an `RLock`. Returned suggestion
        sequences are copied to avoid cross-thread mutation.

    Keys:
        Tuple ``(id(obj), get_type_name(type(obj)), operation)``. Identity reuse after GC
        can collide within TTL. If you need stronger semantics, supply a stable,
        content-derived key in your own wrapper.

    Expiry/Eviction:
        Expiry uses `time.monotonic()`. On `get`, entries are refreshed to approximate
        LRU. When capacity is reached, the oldest timestamped entry is evicted.

    Returns:
        ``(result: bool, suggestions: List[str])`` when a live entry is found, else `None`.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size: int = max_size
        self.ttl: float = ttl_seconds
        # (id, type_name, op) -> (result, ts, suggestions_tuple)
        self._cache: Dict[Tuple[Hashable, str, str], Tuple[bool, float, List[str]]] = {}
        self._lock: RLock = RLock()

    def _now(self) -> float:
        return time.monotonic()

    def get(self, obj: Any, operation: str) -> Optional[Tuple[bool, List[str]]]:
        """Return cached `(result, suggestions)` for `obj`/`operation`, or `None`."""
        key = (id(obj), get_type_name(type(obj)), operation)
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            result, ts, sugg_tuple = entry
            if self._now() - ts >= self.ttl:
                del self._cache[key]
                return None
            # touch timestamp so eviction is time-of-last-access
            self._cache[key] = (result, self._now(), sugg_tuple)
            return result, list(sugg_tuple)

    def set(
        self, obj: Any, operation: str, result: bool, suggestions: List[str]
    ) -> None:
        """Insert or replace cache entry for `obj`/`operation` with a TTL."""
        key = (id(obj), get_type_name(type(obj)), operation)
        now = self._now()
        with self._lock:
            if len(self._cache) >= self.max_size:
                # Evict the oldest by timestamp (coarse LRU)
                oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            # store an immutable snapshot of suggestions
            self._cache[key] = (result, now, tuple(suggestions))  # type: ignore[assignment]


# Global validation cache (singletons are convenient until they aren't)
_validation_cache = ValidationCache()


def clear_validation_cache() -> int:
    """Clear the module-level validation cache; return number of entries removed."""
    with _validation_cache._lock:
        count = len(_validation_cache._cache)
        _validation_cache._cache.clear()
        return count


def get_cache_stats() -> Dict[str, Any]:
    """Return simple stats for the module-level cache.

    Expired entries are counted against `time.time()`, not `time.monotonic()`.
    This is a best-effort view; eviction/expiry logic uses monotonic time.
    """
    with _validation_cache._lock:
        total_entries = len(_validation_cache._cache)
        expired_entries = 0
        current_time = time.time()
        for _, (_, timestamp, _) in _validation_cache._cache.items():
            if current_time - timestamp >= _validation_cache.ttl:
                expired_entries += 1
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "max_size": _validation_cache.max_size,
            "ttl_seconds": _validation_cache.ttl,
        }


def configure_validation_cache(
    max_size: Optional[int] = None, ttl_seconds: Optional[float] = None
) -> None:
    """Mutate the module-level cache configuration.

    This function updates the live cache instance without replacing it. Changes
    are not atomic across readers, but assignments are single operations in CPython.
    """
    global _validation_cache
    if max_size is not None:
        _validation_cache.max_size = max_size
    if ttl_seconds is not None:
        _validation_cache.ttl = ttl_seconds


# -----------------------------------------------------------------------------
# Registry Validator Mixins
# -----------------------------------------------------------------------------

KeyType = TypeVar("KeyType", bound=Hashable)
ValType = TypeVar("ValType")


def _is_hashable(value: Any) -> bool:
    return isinstance(value, Hashable)


def _resolve_typevars(cls: type) -> Tuple[Any, Any]:
    if not hasattr(cls, "__orig_bases__"):
        raise TypeError(f"Expected a parametrized Generic subclass, got {cls!r}")

    key = Any
    val = Any
    for base in getattr(cls, "__orig_bases__", ()):
        if get_origin(base) is RegistryAccessorMixin:
            key, val = get_args(base)

    # Substitute unresolved TypeVars with your desired defaults
    if isinstance(key, TypeVar) and key.__name__ == "KeyType":
        key = Hashable
    if isinstance(val, TypeVar) and val.__name__ == "ValType":
        val = Any

    return key, val


class ImmutableValidatorMixin(
    RegistryAccessorMixin[KeyType, ValType], Generic[KeyType, ValType]
):
    """Read-side validation wrapper for registries.

    Responsibilities:
        - Validate identifiers and artifacts on read paths.
        - Convert bad inputs into `ValidationError` with context.
        - Delegate presence errors to accessor (`RegistryError`).

    Override points:
        `_internalize_identifier`, `_externalize_identifier`, `_internalize_artifact`, `_externalize_artifact`.
    """

    @classmethod
    def _internalize_identifier(cls, value: Any) -> KeyType:
        """Validate/coerce an identifier for storage in the registry.

        Raises:
            ValidationError: if `value` is not hashable or otherwise invalid.
        """
        key_t, _ = _resolve_typevars(cls)

        if not _is_hashable(value):
            suggestions = [
                "Ensure the key is hashable (str, int, tuple, etc.)",
                f"Got {get_type_name(type(value))}, which is not hashable",
                "Try converting to string: str(your_key)",
            ]
            context = {
                "expected_type": "Hashable",
                "actual_type": get_type_name(type(value)),
                "artifact_name": str(value),
            }
            raise ValidationError(
                f"Key must be hashable, got {get_type_name(type(value))}",
                suggestions,
                context,
            )
        if key_t != Any and not isinstance(value, key_t):
            suggestions = [
                f"Ensure the key is of type {key_t.__name__}",
                f"Got {get_type_name(type(value))}, which is not of type {key_t.__name__}",
                "Try converting to the expected type: your_key = key_t(your_key)",
            ]
            context = {
                "expected_type": key_t.__name__,
                "actual_type": get_type_name(type(value)),
                "artifact_name": str(value),
            }
            raise ValidationError(
                f"Key must be of type {key_t.__name__}, got {get_type_name(type(value))}",
                suggestions,
                context,
            )
        return value

    @classmethod
    def _internalize_artifact(cls, value: ValType) -> ValType:
        """Validate/coerce an artifact for storage. Default passthrough."""
        _, val_t = _resolve_typevars(cls)
        if val_t != Any and not isinstance(value, val_t):
            suggestions = [
                f"Ensure the artifact is of type {val_t.__name__}",
                f"Got {get_type_name(type(value))}, which is not of type {val_t.__name__}",
            ]
            context = {
                "expected_type": val_t.__name__,
                "actual_type": get_type_name(type(value)),
                "artifact_name": str(value),
            }
            raise ValidationError(
                f"Artifact must be of type {val_t.__name__}, got {get_type_name(type(value))}",
                suggestions,
                context,
            )
        return value

    @classmethod
    def _externalize_identifier(cls, value: KeyType) -> KeyType:
        """Validate an identifier on retrieval. Default: hashable check."""
        return value

    @classmethod
    def _externalize_artifact(cls, value: ValType) -> ValType:
        """Validate an artifact on retrieval. Default passthrough."""
        return value

    @classmethod
    def get_artifact(cls, key: KeyType) -> ValType:
        """Retrieve an artifact with validation and rich failures.

        Raises:
            RegistryError: if key is missing.
            ValidationError: if the key or artifact validation fails.
        """
        try:
            validated_key = cls._internalize_identifier(key)
            item = cls._get_artifact(validated_key)
            return cls._externalize_artifact(item)
        except RegistryError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            suggestions = [
                f"Check that key '{key}' exists in the registry",
                "Use has_identifier() to check existence first",
                f"Registry {get_type_name(cls)} contains {cls._len_mapping()} items",
                "Use iter_identifiers() to see all available keys",
            ]
            context = {
                "operation": "get_artifact",
                "registry_name": get_type_name(cls),
                "registry_type": get_type_name(type(cls)),
                "key": str(key),
                "key_type": get_type_name(type(key)),
                "registry_size": cls._len_mapping(),
            }
            raise ValidationError(
                f"Failed to retrieve artifact with key '{key}': {e}",
                suggestions,
                context,
            ) from e

    @classmethod
    def has_identifier(cls, key: KeyType) -> bool:
        """Return True if an artifact with `key` exists (after key validation)."""
        try:
            validated_key = cls._internalize_identifier(key)
            return cls._has_identifier(validated_key)
        except ValidationError:
            return False

    @classmethod
    def has_artifact(cls, item: ValType) -> bool:
        """Return True if `item` exists in the registry (after artifact validation)."""
        try:
            validated_item = cls._internalize_artifact(item)
            return cls._has_artifact(validated_item)
        except ValidationError:
            return False

    @classmethod
    def iter_identifiers(cls) -> Iterator[KeyType]:
        """Iterate over all artifact identifiers."""
        return map(cls._externalize_identifier, cls._iter_mapping())

    @classmethod
    def validate_artifact(cls, item: Any) -> ValType:
        """Validate an artifact without registering it; return the validated value."""
        try:
            validated_item = cls._internalize_artifact(item)
            return cls._externalize_artifact(validated_item)
        except ValidationError:
            raise
        except Exception as e:
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


class MutableValidatorMixin(
    ImmutableValidatorMixin[KeyType, ValType],
    RegistryMutatorMixin[KeyType, ValType],
    Generic[KeyType, ValType],
):
    """Write-side validation wrapper for registries.

    Responsibilities:
        - Validate identifiers/artifacts on write paths.
        - Convert write failures to `ValidationError` or surface key errors.
        - Provide batch operations with structured error reports.

    Subclasses must implement:
        `_identifier_of(item) -> KeyType`
    """

    @classmethod
    def _identifier_of(cls, item: ValType) -> KeyType:
        """Extract the identifier from an artifact."""
        raise NotImplementedError("Subclasses must implement _identifier_of")

    @overload
    @classmethod
    def register_artifact(cls, key: KeyType, item: ValType) -> ValType:
        """Register an artifact with explicit key."""
        ...

    @overload
    @classmethod
    def register_artifact(cls, key: ValType, item: None = None) -> ValType:
        """Register an artifact using extracted key."""
        ...

    @classmethod
    def register_artifact(
        cls, key: Union[KeyType, ValType], item: Optional[ValType] = None
    ) -> ValType:
        """Register an artifact with validation; supports explicit or inferred keys."""
        if item is None:
            artifact = cast(ValType, key)
        else:
            artifact = item

        try:
            if item is None:
                identifier = cls._identifier_of(artifact)
            else:
                identifier = key
            validated_key = cls._internalize_identifier(identifier)
            validated_item = cls._internalize_artifact(artifact)
            cls._set_artifact(validated_key, validated_item)
            return artifact
        except (ValidationError, ConformanceError, InheritanceError):
            raise
        except Exception as e:
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
    def unregister_identifier(cls, key: KeyType) -> None:
        """Remove an artifact by its identifier.

        Raises:
            ValidationError: if key validation fails.
            RegistryError: if the key does not exist.
        """
        try:
            validated_key = cls._internalize_identifier(key)
            cls._del_artifact(validated_key)
        except RegistryError:
            raise
        except (ConformanceError, InheritanceError, ValidationError):
            raise
        except Exception as e:
            suggestions = [
                f"Check that key '{key}' exists in the registry",
                "Use has_identifier() to check existence first",
                f"Registry {get_type_name(cls)} contains {cls._len_mapping()} items",
                "Use iter_identifiers() to see all available keys",
            ]
            context = {
                "operation": "unregister_identifier",
                "registry_name": get_type_name(cls),
                "registry_type": get_type_name(type(cls)),
                "key": str(key),
                "key_type": get_type_name(type(key)),
                "registry_size": cls._len_mapping(),
            }
            raise ValidationError(
                f"Unregistration failed for key '{key}': {e}", suggestions, context
            ) from e

    @classmethod
    def unregister_artifact(cls, item: ValType) -> None:
        """Remove an artifact by value.

        Extracts the identifier with `_identifier_of` and delegates to
        `unregister_identifier`.
        """
        try:
            identifier = cls._identifier_of(item)
            cls.unregister_identifier(identifier)
        except (ValidationError, RegistryError):
            raise
        except Exception as e:
            artifact_name = getattr(item, "__name__", str(item))
            suggestions = [
                f"Check that artifact '{artifact_name}' exists in the registry",
                "Ensure the artifact can be identified using _identifier_of()",
                "Use has_artifact() to check existence first",
                f"Registry {get_type_name(cls)} contains {cls._len_mapping()} items",
            ]
            context = {
                "operation": "unregister_artifact",
                "registry_name": get_type_name(cls),
                "registry_type": get_type_name(type(cls)),
                "artifact_name": artifact_name,
                "artifact_type": get_type_name(type(item)),
                "registry_size": cls._len_mapping(),
            }
            raise ValidationError(
                f"Unregistration failed for artifact '{artifact_name}': {e}",
                suggestions,
                context,
            ) from e

    @classmethod
    def clear_artifacts(cls) -> None:
        """Delete all artifacts in the registry."""
        cls._clear_mapping()

    # -----------------------------------------------------------------------------
    # Registry Management Methods
    # -----------------------------------------------------------------------------

    @classmethod
    def validate_registry_state(cls) -> Dict[str, Any]:
        """Validate every artifact and return a summary report."""
        total_artifacts = cls._len_mapping()
        validation_errors = []
        warnings = []
        cache_stats = get_cache_stats()

        for key in cls._iter_mapping():
            try:
                artifact = cls._get_artifact(key)
                cls.validate_artifact(artifact)
            except ValidationError as e:
                validation_errors.append(
                    {
                        "key": str(key),
                        "error": str(e),
                        "suggestions": getattr(e, "suggestions", []),
                    }
                )
            except Exception as e:
                warnings.append(
                    {
                        "key": str(key),
                        "warning": f"Unexpected error during validation: {e}",
                    }
                )
        return {
            "total_artifacts": total_artifacts,
            "validation_errors": validation_errors,
            "warnings": warnings,
            "cache_stats": cache_stats,
        }

    @classmethod
    def clear_validation_cache(cls) -> int:
        """Clear the shared validation cache; return number of entries cleared."""
        return clear_validation_cache()

    @classmethod
    def configure_validation(
        cls, cache_size: Optional[int] = None, cache_ttl: Optional[float] = None
    ) -> None:
        """Adjust shared validation cache parameters for this process."""
        configure_validation_cache(cache_size, cache_ttl)

    @classmethod
    def get_validation_stats(cls) -> Dict[str, Any]:
        """Return shared validation cache statistics."""
        return get_cache_stats()

    @classmethod
    def batch_validate(
        cls, items: Dict[KeyType, ValType]
    ) -> Dict[KeyType, Union[bool, ValidationError]]:
        """Validate multiple items without registering them.

        Returns:
            Dict mapping keys to True or a `ValidationError` instance.
        """
        results = {}
        for key, item in items.items():
            try:
                valid_key = cls._internalize_identifier(key)
                recon_key = cls._externalize_identifier(valid_key)

                valid_item = cls._internalize_artifact(item)
                recon_item = cls._externalize_artifact(valid_item)

                results[key] = bool(recon_item == item and recon_key == key)
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
        cls, items: Dict[KeyType, ValType], skip_invalid: bool = True
    ) -> Dict[str, Any]:
        """Register multiple items with per-item error capture.

        Returns:
            Dict with keys: successful, failed, total, errors.
        """
        successful: List[KeyType] = []
        failed: List[KeyType] = []
        errors: List[Dict[str, Any]] = []
        for key, item in items.items():
            try:
                cls.register_artifact(key, item)
                successful.append(key)
            except (ValidationError, ConformanceError, InheritanceError) as e:
                error_info = {
                    "key": key,
                    "error": str(e),
                    "suggestions": getattr(e, "suggestions", []),
                    "context": getattr(e, "context", {}),
                }
                failed.append(key)
                errors.append(error_info)
                if not skip_invalid:
                    raise
            except Exception as e:
                enhanced_error = ValidationError(
                    f"Registration failed for key '{key}': {e}"
                )
                failed.append(key)
                errors.append(
                    {
                        "key": key,
                        "error": str(enhanced_error),
                        "suggestions": [],
                        "context": {},
                    }
                )
                if not skip_invalid:
                    raise enhanced_error from e
        return {
            "successful": successful,
            "failed": failed,
            "total": len(items),
            "errors": errors,
        }
